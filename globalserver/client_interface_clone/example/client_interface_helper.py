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

"""The Python implementation of the GRPC globalserver.Greeter client."""
import time
import os
import grpc
import json
import logging
import numpy as np

import tensorflow as tf
import interface_pb2, interface_pb2_grpc


class ClientInterface():
    def __init__(self, server_port=os.getenv('SERVER_PORT'),
                 server_address=os.getenv('SERVER_ADDRESS', '0.0.0.0')):
        self.server_port = server_port
        self.server_address = server_address
        try:
            channel = grpc.insecure_channel(f"{self.server_address}:{self.server_port}")
            self.stub = interface_pb2_grpc.InterfaceControllerStub(channel)
            fetch_model2 = self.do_task("")
        except:
            time.sleep(1)
            channel = grpc.insecure_channel(f"{self.server_address}:{self.server_port}")
            self.stub = interface_pb2_grpc.InterfaceControllerStub(channel)
            fetch_model2 = self.do_task("")

    def do_task(self, task, params={}, timeout=30):
        logging.info(task)
        task_submitted = self.stub.do_task(
            interface_pb2.InterfaceTaskSubmit(task=task, client='INTERFACE', protocol='NN',
                                              model_id=params.get('model_id', '')))
        timer = 0
        while True:
            worker_idle = self.stub.do_task(interface_pb2.InterfaceTaskSubmit(task='', client='INTERFACE'))
            if worker_idle.ok:
                break
            else:
                time.sleep(1)

            timer = timer + 1
            if timer > timeout:
                raise Exception("timed out for task {task}")

        task_response = self.stub.get_last_response(interface_pb2.InterfaceTaskSubmit(client='INTERFACE'))
        return task_response

    def get_available_models(self):
        response = self.stub.get_models(interface_pb2.DefaultRequest(client='INTERFACE'))
        return json.loads(response.response)

    def array_from_bytes(self, bytes_array):
        array = json.loads(bytes_array)
        for layer_index, layer in enumerate(array):
            for cell_index, _ in enumerate(layer):
                array[layer_index][cell_index] = np.array(array[layer_index][cell_index])
        return array

    def set_weights(self, model, weights, normalize=False):
        for layer_index, layer in enumerate(model.layers):
            cell_weights = []
            for cell_index, _ in enumerate(layer.weights):
                if normalize != 0:
                    # normalize weight
                    norm = np.linalg.norm(weights[layer_index][cell_index])
                    normalized_weigths = weights[layer_index][cell_index] / max([norm / normalize, 1])
                    cell_weights.append(normalized_weigths)
                else:
                    cell_weights.append(weights[layer_index][cell_index])
            layer.set_weights(cell_weights)
        return model

    def get_model(self, model_id):
        response = self.stub.fetch_model_request(
            interface_pb2.DefaultRequest(client='INTERFACE', task_id='fetch_task', model_id=model_id))

        for row in response:
            config = json.loads(row.model_definition)
            model = tf.keras.models.model_from_json(json.dumps(config['model']))
            model.compile(loss=tf.losses.get(config['compile']['loss']),
                          optimizer=tf.optimizers.get(config['compile']['optimizer']),
                          metrics=[config['compile']['metrics']],
                          loss_weights=config['compile'].get('loss_weights', None),
                          sample_weight_mode=config['compile'].get('sample_weight_mode', None),
                          weighted_metrics=config['compile'].get('weighted_metrics', None),
                          target_tensors=config['compile'].get('target_tensors', None)
                          )

            global_weights = self.array_from_bytes(row.model_parameters)
            model = self.set_weights(model, global_weights)
        return model,config
