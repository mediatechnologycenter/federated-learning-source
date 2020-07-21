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

'''The Python implementation of the interface'''

from concurrent import futures
import time
import logging
import grpc
import json
import os
import sys

# todo do better
config = json.load(open(os.getenv("STATIC_VARIABLES_FILE_PATH", "static_variables.json"), 'r'))
sys.path.append(os.getcwd())
sys.path.append(os.getcwd() + '/' + os.getenv("PATH_TO_GLOBALSERVER", config['DEFAULT_GLOBAL_SERVER_PATH'])[:-1])

##### Must be run with client_interface_python client_interface_clone/client_interface_controller.py
from client_interface_clone.interface_utils import interface_pb2_grpc, interface_pb2
from client_interface_clone.interface_utils import client_interface_utils
from api import globalserver_task_controller
from api.utils.globalserver_pb2 import DefaultRequest

CLIENT_SECRET = os.getenv('CLIENT_SECRET', '')
SERVER_PORT = os.getenv('SERVER_PORT')
CLIENT_INTERFACE_PORT = os.getenv('CLIENT_INTERFACE_PORT')
EXPERIMENT_ID = '000000000000000000000000'


class InterfaceController(globalserver_task_controller.TaskController):
    def __init__(self):
        self.client_is_working = {}  # Dict of the current task every client is working on. (if None they are idle)
        self.client_is_stopped = {}
        self.utils = client_interface_utils
        self.fl_db, self.db_session = self.utils.get_db_connection()

    def do_task(self, request, context):
        """
        If task is empty string we just get feeedback if the worker is busy
        """


        self.client_is_working[EXPERIMENT_ID] = self.client_is_working.get(EXPERIMENT_ID, {})
        self.client_is_stopped[EXPERIMENT_ID] = self.client_is_stopped.get(EXPERIMENT_ID, {})

        if self.fl_db['task_name']:  # busy
            logging.debug(f"Worker busy {self.fl_db['task_name']}")
            return interface_pb2.DefaultResponse(message=f'Error: Worker is busy.', ok=False)
        if request.task == '':
            return interface_pb2.DefaultResponse(message=f'Worker Idle', ok=True)
        if request.task not in config['VALID_TASKS']:
            return interface_pb2.DefaultResponse(message=f'Error: Task not valid.', ok=False)

        logging.info(f'Task {request.task} received...')
        self.fl_db['task_name'] = request.task
        self.fl_db['protocol'] = request.protocol
        self.fl_db['model_id'] = request.model_id

        # self.client_is_stopped[EXPERIMENT_ID][request.client] = False
        # self.client_is_working[EXPERIMENT_ID][request.client] = True

        logging.info(f'Task {request.task} submitted with {request.protocol}...')
        return interface_pb2.DefaultResponse(message=f'Task submitted!', ok=True)

    def get_last_response(self, request, context):
        return interface_pb2.InterfaceStringResponse(message=f'Task submitted!', ok=True,
                                                     response=json.dumps(self.fl_db))

    def fetch_model_request(self, request, context):

        request.client = config['WORKER_CLIENT_NAME']
        request.experiment_id = EXPERIMENT_ID
        if not self.utils.client_is_valid(config['WORKER_CLIENT_NAME'], request.secret):
            return interface_pb2.DefaultResponse(message='Client not known....', ok=False)
        # if request.model_id == '' and "model_id" not in self.fl_db:

        model_id = self.fl_db['model_id'] if request.model_id == '' else request.model_id
        _, _, responses = self.utils.get_grpc_connection(grpc_function='fetch_model_request',
                                                         request=DefaultRequest(
                                                             client=config['WORKER_CLIENT_NAME'],
                                                             secret=CLIENT_SECRET,
                                                             model_id=model_id),
                                                         timeout=60)

        logging.info(f'Streaming model to {config["WORKER_CLIENT_NAME"]}...')
        # todo only if flagged model
        for row in responses:
            yield interface_pb2.Model(model_parameters=row.model_parameters,
                                      model_definition=row.model_definition)
        logging.info(f'Streaming model to {config["WORKER_CLIENT_NAME"]} finished')

        correct_task_completed = self.utils.task_completion(db=self.fl_db, task_id=request.task_id, experiment_id='',
                                                            client=request.client, result='',
                                                            db_session=self.db_session)

        if self.client_is_working[request.experiment_id].get(request.client, False) and correct_task_completed:
            self.client_is_working[request.experiment_id][request.client] = False

    def get_models(self, request, context):
        logging.info("Getting models from server")
        _, _, responses = self.utils.get_grpc_connection(grpc_function='get_latest_global_model',
                                                         request=DefaultRequest(
                                                             client=config['WORKER_CLIENT_NAME'],
                                                             secret=CLIENT_SECRET),
                                                         timeout=60)

        return interface_pb2.InterfaceStringResponse(message=f'Models fetched!', ok=True,
                                                     response=responses.model_id)

    def stop_task(self, request, context):
        # todo do that

        logging.info(f'Stopping tasks...')
        self.utils.stop_clients([config['WORKER_CLIENT_NAME']])
        return interface_pb2.DefaultResponse(message=f'Stop submitted!', ok=True)

    def train_model_response(self, request, context):
        request.client = config['WORKER_CLIENT_NAME']
        request.experiment_id = EXPERIMENT_ID

        return super().train_model_response(request, context)

    def send_validation_loss_response(self, request, context):
        request.client = config['WORKER_CLIENT_NAME']
        request.experiment_id = EXPERIMENT_ID

        return super().send_validation_loss_response(request, context)

    def get_task_request(self, request, context):
        request.client = config['WORKER_CLIENT_NAME']
        request.experiment_id = EXPERIMENT_ID

        return super().get_task_request(request, context)

    def stop_response(self, request, context):
        request.client = config['WORKER_CLIENT_NAME']
        request.experiment_id = EXPERIMENT_ID

        return super().stop_response(request, context)


def serve():
    # todo secuuure
    options = [('grpc.max_send_message_length', 512 * 1024 * 1024),
               ('grpc.max_receive_message_length', 512 * 1024 * 1024)]
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=config["MAX_WORKERS"]),
                         options=options)

    interface_pb2_grpc.add_InterfaceControllerServicer_to_server(InterfaceController(), server)

    server.add_insecure_port(f'[::]:{CLIENT_INTERFACE_PORT}')
    server.start()
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == '__main__':
    logging.basicConfig(
        level=int(os.getenv('LOGGING_LEVEL', 0)),
        format=f"%(asctime)s [%(processName)-12.12s] [%(levelname)-5.5s] [CLIENT_INTERFACE_CONTROLLER] [%(filename)s / %(funcName)s / %(lineno)d] %(message)s")
    serve()
