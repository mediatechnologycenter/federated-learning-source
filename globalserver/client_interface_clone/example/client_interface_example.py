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

from client_interface_helper import ClientInterface
server_port='port you started the server with (CLIENT_INTERFACE_PORT)'
server_address = 'ip address of the host where you ran the client interface docker'
ClientInterfaceInstance = ClientInterface(server_address=server_address, server_port=server_port)

print(f"get_available_models returns a list of all available models and their meta data.")
models = ClientInterfaceInstance.get_available_models()

print(f"We pick one model and fetch it with get_model which returns the compiled keras model and its configuration.")
model_id = models[0]['_id']
model, config = ClientInterfaceInstance.get_model(model_id)

print(f"We tell the Node Worker to fetch the model, train it and return the loss:")
response = ClientInterfaceInstance.do_task("fetch_model", params={"model_id": model_id})
response = ClientInterfaceInstance.do_task("train_model", timeout=300)
response = ClientInterfaceInstance.do_task("send_validation_loss", timeout=300)
