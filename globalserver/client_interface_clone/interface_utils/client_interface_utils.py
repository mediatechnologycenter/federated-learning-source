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

import json
import time
import logging

import os
import grpc
from api.utils.api_utils import client_is_valid

import sys
from api.utils import globalserver_pb2_grpc, globalserver_pb2

config = json.load(open(os.getenv("STATIC_VARIABLES_FILE_PATH", "static_variables.json"), 'r'))

SERVER_PORT = os.getenv('SERVER_PORT')


def get_task_from_experiment(db, experiment_id, client, db_session):
    if not db['task_name'] or db['task_name'] == '' :
        return None,db['protocol']
    return {"task_name": db['task_name'], '_id': db['task_name']}, db['protocol']


def add_validation_result_to_experiment(db, loss, task_id, experiment_id, client):
    db['validation_results'] = loss  # Caution! This changes the variable in client_interface_controller by intention

def add_test_result_to_experiment(db, loss, task_id, experiment_id, client):
    db['test_results'] = loss  # Caution! This changes the variable in client_interface_controller by intention


def task_completion(db, task_id, experiment_id, client, result, db_session):
    if db['task_name'] == str(task_id):
        db['task_name'] = None  # Caution! This changes the variable in client_interface_controller by intention
        return True

    return False


def get_db_connection():
    return {'protocol': '',
            'task_name': '',
            'result': ''}, ''


def stop_clients(clients):
    stop_task = {'task': config['STOP_WORD'], 'model_name': ''}
    for client in clients:
        json.dump([stop_task], open(f"{config['QUEUE_PATH']}{client}.json", 'w'))


def get_grpc_connection(grpc_function, request, server_port=SERVER_PORT,
                        max_retries=config["GRPC_CONNECTION_RETRIES"],
                        delay=config["GRPC_CONNECTION_RETRY_DELAY"],
                        sleep_on_error=config["SLEEP_ON_ERROR"],
                        timeout=config["GRPC_TIMEOUT"]):
    """
    Occasionally, the server is busy and the request fails. We retry 5 times to get reach the server.
    """
    retries = 0
    # Startup routine, open grpc channel, define messaging queues, start worker subprocess
    while retries < max_retries:
        try:
            channel = grpc.insecure_channel(os.getenv('SERVER_ADDRESS') + f":{server_port}")
            stub = globalserver_pb2_grpc.TaskControllerStub(channel)
            logging.debug(grpc_function)
            method = getattr(stub, grpc_function)
            response = method(request, timeout=timeout)
            return True, stub, response
        except Exception as error:
            retries = retries + 1
            logging.warning(f"GRPC Connection failed, retry... attempt {retries}/{max_retries}")
            print(error)
            time.sleep(delay)

    time.sleep(sleep_on_error)
    return False, None, None
