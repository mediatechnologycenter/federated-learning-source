# Copyright 2019-2020, ETH Zurich, Media Technology Center
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
"""
Worker Class. This Class gets the tasks from node_task_controller and fulfills it. It storas a local model in memory.
That is the model that all task are processed on. As soon as a task it finished it gives the task specific feedback
directly to the server. There are currently 4 tasks:
    fetch_model:    fetches the global model from the server and compiles it.
    train_model:    trains the local model with the provided training data and the parameters given in the model config.
                    Pings the global server when finished.
    send_model:     sends the local model to the global server
    send_validation_loss:      sends the evaluation loss to the server

"""
import os
import psutil

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from pebble import concurrent

import utils.grpc_util as grpc_util

import logging
import time
import json
import requests
import traceback
import os
import numpy as np
from utils.models import NN, RF, P2P
from utils.data_wrapper import NewWrapper, OldWrapper

from functools import partial

if int(os.getenv('SERVER', 1)):
    from api.utils import globalserver_pb2 as globalserver_pb2
else:
    from client_interface_clone.interface_utils import interface_pb2 as globalserver_pb2

SERVER_PORT = os.getenv('SERVER_PORT')
config = json.load(open(os.getenv("STATIC_VARIABLES_FILE_PATH", "static_variables.json"), 'r'))

config['CONTROLLER_IDLE_SLEEP'] = config['CONTROLLER_IDLE_SLEEP'] if int(os.getenv('LOGGING_LEVEL', 30)) >= 20 else 1

# todo
import sys
import numpy

numpy.set_printoptions(threshold=sys.maxsize)


def pretty_print_error(error, node, ):
    error_msg = traceback.format_exc()
    logging.error(error_msg)
    logging.error("Variable information for the previous error: ")
    try:
        error_msg = error_msg + str(error)
        logging.error(error)
        logging_vars = json.dumps({"model_config": node.config,
                                   "experiment_id": node.experiment_id})
        error_msg = error_msg + logging_vars
        logging.error(logging_vars)
        return error_msg

    except Exception as error:
        logging.error(traceback.format_exc())
        return error_msg


@concurrent.process
def worker(client, secret, error_queue, experiment_id, node=None):
    """
    Separate process for each experiment. It just creates a worker and starts it. the error_queue is shared with the
    node_task_controller.py to tell the controller if a process failed.
    """
    try:
        # Initialization
        node = Experiment(client, secret, experiment_id)
        node.start_working()

    # Enhanced Error logging:
    except Exception as error:
        logging.info(error)
        error_msg = pretty_print_error(error, node)
        error_queue.put([experiment_id, error_msg], timeout=100)
        while error_queue.empty():
            time.sleep(0.1)
        while not error_queue.empty():
            logging.info("alive")
            time.sleep(10)
        raise error


algorithms = {"NN": NN, "P2P": P2P, "RF": RF}


class Experiment:
    def __init__(self, client, secret, experiment_id):

        logging.info(f"Initialize worker...")

        self.experiment_id = experiment_id
        self.grpc_connection = grpc_util.GrpcConnector(client=client, secret=secret, experiment_id=experiment_id)

        # This is only needed until memory leak is fixed, to read memory usage
        self.process = psutil.Process(os.getpid())

        self.model, self.config = self.get_model_instance(client)

    def get_model_instance(self, client):
        _, responses = self.grpc_connection.get_grpc_connection(
            grpc_function='fetch_model_request',
            request=globalserver_pb2.DefaultRequest)

        for row in responses:  # pseudo stream
            config = row.model_definition
            # parameters = row.model_parameters
            protocol = row.protocol

            config = json.loads(config)
            config = self._set_custom_training_config(config, client)  # todo allow to change everything

            if 'dataset' in config["training"]:  # new data wrapper
                data_generator = NewWrapper(config, client)
            else:
                data_generator = OldWrapper(config, client)

            model = algorithms[protocol](config, data_generator)

            logging.info(f"GRPC Connection established...")

            return model, config

    @staticmethod
    def _set_custom_training_config(config, client):
        if "custom" not in config['training']:
            return config
        if client not in config['training']["custom"]:
            return config
        for key, value in config['training']["custom"][client].items():
            logging.debug(key)
            config['training'][key] = value
        return config

    def start_working(self):
        server_ok = True
        while server_ok:
            server_ok, response = self.grpc_connection.get_grpc_connection(grpc_function='get_task_request',
                                                                           request=globalserver_pb2.TaskRequest)

            # bad response from server
            if not response.ok:
                logging.error(f"Message <{response.message}>")
                raise Exception("Bad Response From Server")

            # good response from server
            elif response.task in config['VALID_TASKS']:
                self.run_task(response.task, task_id=response.task_id)

            # No task to do or unknown task
            else:
                # logging.debug(f"Nothing to do with <{response.task}> ...")
                time.sleep(config['CONTROLLER_IDLE_SLEEP'])

    def run_task(self, task, task_id):
        logging.info(f"Task <{task} - {self.experiment_id}> received...")
        self.memory_check(self.process)  # todo temporary "fix" for the memory leak (shutdown)
        error=[]
        for i in range(config['TASK_TRIALS']):
            try:
                task_method = getattr(self, task)
                task_method(task_id)
                break
            except Exception as e:
                error.append(e)
                time.sleep(config['TASK_ERROR_SLEEP'])
                if task != "fetch_model":
                    self.model.reset_model()
                logging.debug(f"{i}. try to do task {task} failed")
        if i == config['TASK_TRIALS'] - 1:
            raise Exception(error)
        return True

    def fetch_model(self, task_id):
        logging.info(f"Parsing Model...%s", self.experiment_id)

        _, responses = self.grpc_connection.get_grpc_connection(
            grpc_function='fetch_model_request',
            request=partial(globalserver_pb2.DefaultRequest, task_id=task_id))
        for row in responses:  # pseudo stream
            # todo split into load config and compile model. In load config set custom config!
            self.model.load_model(model=row)

        logging.info(f"Model parsed...%s", self.experiment_id)
        return True

    def train_model(self, task_id):
        self.model.train_model()
        self.grpc_connection.get_grpc_connection(grpc_function='train_model_response',
                                                 request=partial(globalserver_pb2.DefaultRequest, task_id=task_id))
        self.model.global_model = self.model.model
        logging.info("Training finished. %s", self.experiment_id)

    def send_model(self, task_id, ):
        logging.info("Sending model...%s", self.experiment_id)
        model_update = self.model.get_model_update()  # todo change stream und add function
        iterator = partial(self.stream_model_updates, model_updates=model_update, task_id=task_id)
        self.grpc_connection.get_grpc_connection(grpc_function='send_model_update_response',
                                                 request=iterator)
        logging.info("Sending model finished%s", self.experiment_id)

    def send_validation_loss(self, task_id):
        self._send_loss(task_id, data_type='validation')

    def send_test_loss(self, task_id, ):
        self._send_loss(task_id, data_type='test')

    def send_training_loss(self, task_id):
        self._send_loss(task_id, data_type='train')

    def _send_loss(self, task_id, data_type):
        performance = self.model.get_loss(data_type)

        logging.info(f"Client: {performance}, %s", self.experiment_id)

        self.grpc_connection.get_grpc_connection(grpc_function=f'send_{data_type}_loss_response',
                                                 request=partial(globalserver_pb2.Loss,
                                                                 task_id=task_id,
                                                                 loss=json.dumps(performance,
                                                                                 default=self.default)))
        logging.info("Loss sent...%s", self.experiment_id)
        return True

    @staticmethod
    def stream_model_updates(experiment_id, model_updates, client, secret, task_id):
        yield globalserver_pb2.ModelUpdate(client=client, secret=secret, experiment_id=experiment_id, task_id=task_id)
        yield globalserver_pb2.ModelUpdate(model_update=model_updates)

    @staticmethod
    def memory_check(process):
        if process.memory_info().rss > config['MAX_MEMORY_USAGE']:
            raise Exception('Too much memory used', 'To avoid overflow from memory leak we shutdown the worker')

    @staticmethod
    def default(obj):
        if type(obj).__module__ == np.__name__:
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj.item()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)

        raise TypeError('Unknown type:', type(obj))
