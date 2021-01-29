#   Copyright 2021, ETH Zurich, Media Technology Center
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import logging
import copy
import time
import json
import traceback
import os

import grpc


if int(os.getenv('SERVER', 1)):
    from api.utils import globalserver_pb2 as globalserver_pb2
    from api.utils.globalserver_pb2_grpc import TaskControllerStub as TaskController
else:
    from client_interface_clone.interface_utils import interface_pb2 as globalserver_pb2
    from client_interface_clone.interface_utils.interface_pb2_grpc import InterfaceControllerStub as TaskController

SERVER_PORT = os.getenv('SERVER_PORT')
config = json.load(open(os.getenv("STATIC_VARIABLES_FILE_PATH", "static_variables.json"), 'r'))
class GrpcConnector():
    def __init__(self,client,secret,experiment_id):
        self.client=client
        self.secret=secret
        self.stub=None
        self.experiment_id=experiment_id

        self.get_grpc_connection(grpc_function='test_connection',
                                                        request= globalserver_pb2.DefaultRequest)

    def get_grpc_connection(self,grpc_function, request, server_port=SERVER_PORT, max_retries=config['GRPC_CONNECTION_RETRIES'],
                            delay=config['GRPC_CONNECTION_RETRY_DELAY'], timeout=config['GRPC_TIMEOUT'], experiment_id=None):
        """
        Occasionally, the server is busy and the request fails. We retry 5 times to get reach the server.
        """
        retries = 0
        # Startup routine, open grpc channel, define messaging queues, start worker subprocess

        options = [('grpc.max_send_message_length', 512 * 1024 * 1024),
                   ('grpc.max_receive_message_length', 512 * 1024 * 1024)]
        while retries < max_retries:

            try:
                if self.stub is None:
                    channel = grpc.insecure_channel(os.getenv('SERVER_ADDRESS') + f":{server_port}", options=options)
                    self.stub = TaskController(channel)
                logging.debug(f"calling {grpc_function}")
                method = getattr(self.stub, grpc_function)

                response = method(request(client=self.client,secret=self.secret,experiment_id=self.experiment_id), timeout=timeout)
                return True,response
            except Exception as error:
                retries = retries + 1
                logging.warning(f"GRPC Connection failed for {grpc_function}, retry... attempt {retries}/{max_retries}")
                logging.warning(traceback.format_exc())
                time.sleep(delay * (retries + 1) ** 3)
        raise Exception