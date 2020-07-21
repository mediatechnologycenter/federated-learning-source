# Copyright 2019-2020 Media Technology Center (MTC) ETH Zürich
#
# This file is part of Federated Learning Project at MTC.
#
# Federated Learning is a free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Federated Learning is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser Public License for more details.
#
# You should have received a copy of the GNU Lesser Public License
# along with Federated Learning.  If not, see <https://www.gnu.org/licenses/>.

'''The Python implementation of the globalserver'''

import json
import logging
import os
import sys
import time
from concurrent import futures

import grpc

sys.path.append(os.getcwd())
from utils import globalserver_pb2_grpc, globalserver_pb2
import utils.api_utils as utils
from google.protobuf.json_format import MessageToDict
from bson.objectid import ObjectId

config = json.load(open(os.getenv("STATIC_VARIABLES_FILE_PATH", "static_variables.json"), 'r'))

config['PATH_TO_GLOBALSERVER'] = os.getenv("PATH_TO_GLOBALSERVER", config['DEFAULT_GLOBAL_SERVER_PATH'])
SERVER_PORT = os.getenv('SERVER_PORT')


class TaskController(globalserver_pb2_grpc.TaskControllerServicer):
    def __init__(self):
        self.client_is_working = {}  # Dict ofcthe current task every client is working on. (if None they are idle)
        self.client_is_stopped = {}
        self.utils = utils

        self.fl_db, self.db_session = self.utils.get_db_connection()

    @staticmethod
    def test_connection(request, context):
        return globalserver_pb2.DefaultResponse(
            message=f'Hello. Thanks for following the protocol. Fetch your next task.', ok=True)

    def start_experiment(self, request, context):
        """
        tells the node controller for which experiments a worker should be running/spawned
        """
        experiment_documents = list(self.fl_db.experiment.find({"is_running": True}))
        started_experiments = [str(experiment_document['_id']) for experiment_document in experiment_documents
                               if not experiment_document.get('is_finished', False)
                               and not experiment_document.get('has_failed', False) and request.client in
                               experiment_document['clients']]
        if len(started_experiments) > 0:
            logging.debug(f"Telling {request.client} to start {started_experiments}")
        return globalserver_pb2.ExperimentResponse(message='', experiment_id=json.dumps(started_experiments), ok=True)

    def get_task_request(self, request, context):
        """
        Communication with workers. Whenever a worker pings the global_server it checks the DB for the next
        tasks in the experiment. If no task is scheduled it schedules the next task for all clients.
        """
        if not self.utils.client_is_valid(request.client, request.secret):
            return globalserver_pb2.TaskResponse(message='Client not known....', ok=False)

        logging.debug(f"received task request from {request.client}/{request.experiment_id}")
        task_document, protocol = self.utils.get_task_from_experiment(db=self.fl_db,
                                                                      experiment_id=ObjectId(request.experiment_id),
                                                                      client=request.client, db_session=self.db_session)

        if task_document is None:
            return globalserver_pb2.TaskResponse(message=f'Hello {request.client}. No task for you!', protocol=protocol,
                                                 ok=True)

        self._set_client_status(request)

        if self.client_is_working[request.experiment_id].get(request.client, False):
            return globalserver_pb2.TaskResponse(
                message=f'Last task not finished: ',
                task=task_document["task_name"], task_id=str(task_document['_id']),
                protocol=protocol, ok=False)

        self.client_is_working[request.experiment_id][request.client] = True

        logging.info(f"sent {task_document['task_name']} to {request.client}/{request.experiment_id}")
        return globalserver_pb2.TaskResponse(message=f'Hello {request.client}. Thanks for following the protocol!',
                                             task=task_document["task_name"], task_id=str(task_document['_id']),
                                             protocol=protocol, ok=True)

    def fetch_model_request(self, request, context):
        """
        Looks for the current experiment state in the db/global_models folder and returns the model_definition
        and model_parameters it to the worker. The model corresponding to the last experiment state is always stored
        under the experiment_id.  Marks the task as done.
        """
        if not self.utils.client_is_valid(request.client, request.secret):
            logging.debug('Client not known....')
            return globalserver_pb2.DefaultResponse(message='Client not known....', ok=False)

        logging.info(f"received fetch_model request from {request.client}/{request.experiment_id}/{request.model_id}")
        model_parameters, experiment_id, task_id, result = self._get_model_parameters(request)
        if request.client !='INTERFACE': #todo make nicer
            experiment_documents = list(self.fl_db.experiment.find({"_id": experiment_id}).limit(1))[0]
            if request.client not in experiment_documents['clients']:
                logging.debug('Experiment not for client....')
                return globalserver_pb2.DefaultResponse(message='Experiment not for client....', ok=False)
        else:
            experiment_documents = list(self.fl_db.model.find({"_id": experiment_id}).limit(1))[0]
        # todo make proper stream
        logging.info(f'Streaming model to {request.client}...')
        yield globalserver_pb2.Model(message="Streaming model", ok=True,protocol=experiment_documents['protocol'],
                                     model_parameters=json.dumps(model_parameters['parameters']).encode('utf-8'),
                                     model_definition=json.dumps(model_parameters['model']).encode('utf-8'))
        logging.info(f'Streaming model to {request.client} finished')

        if request.experiment_id != '' and request.task_id != '':
            correct_task_completed = self.utils.task_completion(db=self.fl_db, task_id=task_id,
                                                                experiment_id=experiment_id,
                                                                client=request.client, result=result,
                                                                db_session=self.db_session)

            if self.client_is_working[request.experiment_id].get(request.client, False) and correct_task_completed:
                self.client_is_working[request.experiment_id][request.client] = False

    def train_model_response(self, request, context):
        """
        This is simply lets the global_server know that the worker finished training.
        Marks task as done
        """
        if not self.utils.client_is_valid(request.client, request.secret):
            return globalserver_pb2.DefaultResponse(message='Client not known....', ok=False)

        logging.info(f"received train_model_response from {request.client}/{request.experiment_id}")
        task_id = ObjectId(request.task_id) if int(os.getenv('SERVER', 1)) else request.task_id

        result = MessageToDict(request)
        result.pop('secret', None)

        logging.info(f'{request.client} finished training')
        return self._finish_up_task_response(request, task_id, result)

    def send_model_update_response(self, request, context):
        """
        Stores the response from the worker in db/local_model_updates/experiment/task and marks the task as done
        """
        first_request = next(request)
        weights_stream, request = request, first_request
        if not self.utils.client_is_valid(request.client, request.secret):
            return globalserver_pb2.DefaultResponse(message='Client not known....', ok=False)

        logging.info(f"received send_model_update_response from {request.client}/{request.experiment_id}")
        task_id, result = self._save_model_updates(request, weights_stream)
        logging.info(f'Model from {request.client} received')

        return self._finish_up_task_response(request, task_id, result)

    def send_validation_loss_response(self, request, context):
        """
        Stores the performance results in the DB in the task collection as well as on the experiment collection.
        """

        return self._send_loss_response(request, context, 'validation')

    def send_train_loss_response(self, request, context):

        """
        Stores the performance results in the DB in the task collection as well as on the experiment collection.
        """

        return self._send_loss_response(request, context, 'training')

    def send_test_loss_response(self, request, context):
        """
        Stores the performance results in the DB in the task collection as well as on the experiment collection.
        """

        return self._send_loss_response(request, context, 'test')

    # todo split in services
    def stop_experiment(self, request, context):
        """
        Tells the node controller which experiments they should stop
        """
        if not self.utils.client_is_valid(request.client, request.secret):
            return globalserver_pb2.DefaultResponse(message='Client not known....', ok=False)

        experiment_documents = list(self.fl_db.experiment.find({"is_running": True}))
        finished_experiments = [str(experiment_document['_id']) for experiment_document in experiment_documents if
                                (request.client in experiment_document['clients']) and (
                                        experiment_document.get("is_finished", False) or experiment_document.get(
                                    "has_failed", False))]
        if len(finished_experiments) > 0:
            logging.info(f"Telling {request.client} to stop {finished_experiments}")
        return globalserver_pb2.ExperimentResponse(message='', experiment_id=json.dumps(finished_experiments), ok=True)

    def stopped_experiment_response(self, request, context):
        """
        This tells the global_server that the worker has stopped. If all worker for an experiment stopped it changes
        status of the experiment to failed, not finished and not running.
        """
        if not self.utils.client_is_valid(request.client, request.secret):
            return globalserver_pb2.DefaultResponse(message='Client not known....', ok=False)

        logging.info(f'Client {request.experiment_id} {request.client} stopped')

        self._check_if_all_clients_stopped(request)
        return globalserver_pb2.DefaultResponse(
            message=f'Hello. Thanks for following the protocol. Fetch your next task.', ok=True)

    def failed_experiment_response(self, request, context):
        """
        This tells the global_server that the worker has stopped due to failure. If all worker for an experiment stopped
        it changes status of the experiment to failed, not finished and not running.
        """
        if not self.utils.client_is_valid(request.client, request.secret):
            return globalserver_pb2.DefaultResponse(message='Client not known....', ok=False)

        logging.info(f'Client {request.experiment_id}  {request.client} failed')
        logging.info(request.protocol)
        result = self.fl_db.experiment.update({"_id": ObjectId(request.experiment_id)}, {"$set": {"has_failed": True}})

        self._check_if_all_clients_stopped(request)

        return globalserver_pb2.DefaultResponse(
            message=f'Hello. Thanks for following the protocol. Fetch your next task.', ok=True)

    def send_datasets(self, request, context):
        if not self.utils.client_is_valid(request.client, request.secret):
            return globalserver_pb2.DefaultResponse(message='Client not known....', ok=False)
        try:
            json.dump(json.loads(request.protocol), open(f"{request.client}_datasets.json", "w+"))
            logging.info(f"datasets stored")
        except json.decoder.JSONDecodeError as error:
            logging.info(request.protocol)
        return globalserver_pb2.ExperimentResponse(message='', experiment_id='', ok=True)

    def get_latest_global_model(self, request, context):
        """
        This is simply lets the global_server know that the worker finished training.
        Marks task as done
        """
        if not self.utils.client_is_valid(request.client, request.secret):
            return globalserver_pb2.ModelIdResponse(message='Client not known....', ok=False)

        model_documents = self.utils.get_latest_models(db=self.fl_db)
        logging.info(
            f"received check_latest_global_model from {request.client} with <{request.model_id}>")
        return globalserver_pb2.ModelIdResponse(
            message=f'Hello {request.client}. Thanks for following the protocol. Fetch your next task.',
            model_id=json.dumps(model_documents), ok=True)

    def _finish_up_task_response(self, request, task_id, result):
        if not self.utils.task_completion(db=self.fl_db, task_id=task_id, experiment_id=ObjectId(request.experiment_id),
                                          client=request.client, result=result, db_session=self.db_session):
            return globalserver_pb2.DefaultResponse(message='Wrong task finished', ok=False)

        if request.experiment_id in self.client_is_working and request.client in self.client_is_working[
            request.experiment_id]:
            self.client_is_working[request.experiment_id][request.client] = False
        return globalserver_pb2.DefaultResponse(
            message=f'Hello {request.client}. Thanks for following the protocol. Fetch your next task.', ok=True)

    def _check_if_all_clients_stopped(self, request):
        try:
            self.client_is_stopped[request.experiment_id][request.client] = True
            self.client_is_working[request.experiment_id][request.client] = False
        except KeyError:
            self.client_is_stopped[request.experiment_id]=self.client_is_stopped.get(request.experiment_id,{})
            self.client_is_working[request.experiment_id]=self.client_is_working.get(request.experiment_id,{})
            self.client_is_stopped[request.experiment_id][request.client] = True
            self.client_is_working[request.experiment_id][request.client] = False
            logging.info(f"keyerror for {request.experiment_id} {request.client}")

        try:
            if all(self.client_is_stopped[request.experiment_id].values()):
                logging.info(f"Experiment {request.experiment_id} stopped")

                result = self.fl_db.experiment.update({"_id": ObjectId(request.experiment_id)},
                                                      {"$set": {"is_running": False}})

        except KeyError:
            logging.info(f"keyerror for {request.experiment_id} {request.client}")

    def _send_loss_response(self, request, context, data_type):
        if not self.utils.client_is_valid(request.client, request.secret):
            return globalserver_pb2.DefaultResponse(message='Client not known....', ok=False)

        logging.info(f"received send_validation_loss_response from {request.client}/{request.experiment_id}")
        task_id, result = self._save_loss(request, data_type=data_type)
        logging.info(f'Loss from {request.client} recieved')
        return self._finish_up_task_response(request, task_id, result)

    def _set_client_status(self, request):
        self.client_is_working[request.experiment_id] = self.client_is_working.get(request.experiment_id, {})
        self.client_is_stopped[request.experiment_id] = self.client_is_stopped.get(request.experiment_id, {})
        self.client_is_stopped[request.experiment_id][request.client] = False

    def _get_model_parameters(self, request):
        if request.model_id != '':
            model_parameters_path = self.utils.get_model_parameter_path(request.model_id)
            with open(model_parameters_path, 'r') as reader:  # todo load config from db?
                model_parameters = reader.read().encode('utf-8')
            model_parameters = json.loads(model_parameters)
            experiment_id, task_id, result = ObjectId(request.model_id), '', ''
        elif request.experiment_id != '':
            experiment_id = ObjectId(request.experiment_id)

            task_id = '' if request.task_id == '' else ObjectId(request.task_id)
            result = MessageToDict(request)
            result.pop('secret', None)
            model_parameters = self.utils.load_global_model(db=self.fl_db, experiment_id=experiment_id)

        return model_parameters, experiment_id, task_id, result

    def _save_model_updates(self, request, weights_stream):
        task_id = ObjectId(request.task_id) if int(os.getenv('SERVER', 1)) else request.task_id
        local_model_path = self.utils.get_clients_response_path(experiment_id=ObjectId(request.experiment_id),
                                                                task_id=task_id,
                                                                client=request.client)
        result = MessageToDict(request)
        result.pop('secret', None)
        result['model_update'] = local_model_path

        logging.info(f'Parsing Model from {request.client}...')
        first_row = next(weights_stream)
        result['model_update'] = json.loads(first_row.model_update)
        json.dump(result, open(local_model_path, 'w'))
        result.pop('model_update', None)
        return task_id, result

    def _save_loss(self, request, data_type):
        task_id = ObjectId(request.task_id) if int(os.getenv('SERVER', 1)) else request.task_id
        result = MessageToDict(request)
        result.pop('secret', None)
        loss_method = getattr(self.utils, f"add_{data_type}_result_to_experiment")
        loss_method(db=self.fl_db, task_id=task_id,
                    loss=request.loss,
                    experiment_id=ObjectId(request.experiment_id),
                    client=request.client)
        return task_id, result


def serve():
    # todo secuuure
    options = [('grpc.max_send_message_length', 512 * 1024 * 1024),
               ('grpc.max_receive_message_length', 512 * 1024 * 1024)]
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=config["MAX_WORKERS"]),
                         options=options)  # todo does this work with the self.variables?

    globalserver_pb2_grpc.add_TaskControllerServicer_to_server(TaskController(), server)

    server.add_insecure_port(f'[::]:{SERVER_PORT}')
    server.start()

    logging.info("Global Server Started")
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == '__main__':
    logging.basicConfig(
        level=int(os.getenv('LOGGING_LEVEL', 0)),
        format=f"%(asctime)s [%(processName)-12.12s] [%(levelname)-5.5s] [GLOBAL_SERVER] [%(filename)s / %(funcName)s / %(lineno)d] %(message)s")
    serve()
