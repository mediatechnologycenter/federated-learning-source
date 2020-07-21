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