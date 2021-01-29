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
"""
The Testing class is used to help you start and shutdown the server.

There are many environement variables that can be set. Most of them are defined in the envs.key file.
os.environ['SERVER_ADDRESS'] = ip of the global server (normally 0.0.0.0)
os.environ['DATA_WRAPPER_URL'] = optional url to the data wrapper
os.environ['CLIENT_SECRET'] = secret to use for the clients. this is validated on the global server
os.environ['SERVER_PORT'] = port for the global server
os.environ['CLIENT_INTERFACE_PORT'] = port for the client interface server
os.environ['STATIC_VARIABLES_FILE_PATH'] = path from root directory to the static_variables_file
os.environ['PATH_TO_GLOBALSERVER'] = path from root directory to the global server
os.environ['TESTING'] = whether we are testing or not (global server will accept all clients
os.environ['LOGGING_LEVEL'] = level of debug

On Initialization you have several options:__init__(self,
clients=['c1', 'c2'], this defines which client nodes we are working with.
data_source='1', this defines whether we get the data locally from the disk '1' or from the wrapper '0'
clear_logs=False, toggle to clear the testing.log file
as_docker=False,  toggle to start the nodes as docker or not
clear_db=False, toggle to clear the db/filesystem from all experiments with flag testing=True
start_servers=False, toggle to start the servers. This will first kill all existing running servers.
interface=False, toggle to start the client interface

With the class instance you can use following functions:
def clear_logs(self): clears the logs
def clear_db(self): clear the db/filesystem from all experiments with flag testing=True
def kill_global_server(self): kills the global server (reliable)
def kill_client_interface_node(self): kills the node used in the client interface (not reliable)
def kill_client_interface(self): kills the client interface (not reliable)
def kill_node_servers(self): kills the client node servers (not reliable)
def kill_servers(self): kills all the servers (reliable)
def start_servers(self, clients, interface): kills all the servers and starts the global server, nodes(, interface)
def start_node_servers(self): starts all the node servers
def start_client_interface(self, ): starts the client interface
def start_global_server(self): starts the global server

"""
import subprocess
import os
import tensorflow as tf
import logging
import sys
import time
import traceback
import json

sys.path.append(os.getcwd())

envs = json.load(open("envs.key", "r"))

os.environ['STATIC_VARIABLES_FILE_PATH'] = "globalserver/static_variables.json"
os.environ['PATH_TO_GLOBALSERVER'] = "globalserver/api/"
os.environ['TESTING'] = os.getenv('TESTING', '1')
os.environ['LOGGING_LEVEL'] = "0"
os.environ['SERVER_ADDRESS'] = os.getenv('SERVER_ADDRESS', envs['SERVER_ADDRESS'])
os.environ['DATA_WRAPPER_URL'] = os.getenv('DATA_WRAPPER_URL', envs['DATA_WRAPPER_URL'])
os.environ['CLIENT_SECRET'] = os.getenv('CLIENT_SECRET', envs['CLIENT_SECRET'])
os.environ['SERVER_PORT'] = os.getenv('SERVER_PORT', envs['SERVER_PORT'])
os.environ['CLIENT_INTERFACE_PORT'] = os.getenv('CLIENT_INTERFACE_PORT', envs['CLIENT_INTERFACE_PORT'])

from multiprocessing import Queue

tf.keras.backend.clear_session()
logging.basicConfig(
    level=int(os.getenv('LOGGING_LEVEL', 0)),
    format=f"%(asctime)s [%(processName)-12.12s] [%(levelname)-5.5s] [test] [%(filename)s / %(funcName)s / %(lineno)d] %(message)s")
import json
from pymongo import MongoClient

import shutil
import psutil


#
# os.environ['GRPC_VERBOSITY']="DEBUG"


class Testing():

    def __init__(self, clients=['c1', 'c2'], data_source='1', clear_logs=False, as_docker=False, clear_db=False,
                 start_servers=False,
                 interface=False):
        self.config = json.load(open("globalserver/static_variables.json", 'r'))
        self.config['PATH_TO_GLOBALSERVER'] = 'globalserver/api/'
        self.as_docker = as_docker
        self.error_queue = Queue()
        self.clients = clients
        if clear_db:
            self.clear_db()
        if clear_logs:
            self.clear_logs()
        self.global_server = None
        self.node_server = {}
        self.client_interface = None
        self.client_interface_node = None

        os.environ['DATA_SOURCE'] = data_source
        if start_servers:
            self.start_servers(clients, interface)

    def clear_logs(self):
        try:
            os.remove("testing/testing.log")
            time.sleep(1)
            log = open("testing/testing.log", 'a+')
            log.close()
        except:
            pass

    def clear_db(self):
        db_config = json.load(open(self.config['PATH_TO_GLOBALSERVER'] + self.config['DB_CONFIG_FILE'], 'r'))
        client = MongoClient(port=int(db_config['port']), username=db_config['user'], password=db_config['password'])
        fl_db = client.federated_learning
        db_session = client.start_session()
        with db_session.start_transaction():
            test_models = list(fl_db.model.find({"testing": True}, {"_id": 1}))
            test_experiments = list(fl_db.experiment.find({"testing": True}, {"_id": 1}))
            for model in test_models:
                model_parameters_path = f"{self.config['PATH_TO_GLOBALSERVER']}{self.config['GLOBAL_MODELS_PATH']}{str(model['_id'])}.json"

                try:
                    os.remove(model_parameters_path)
                except FileNotFoundError:
                    print(f"Model not found to delete {model_parameters_path}")
                    pass
            for experiment in test_experiments:
                model_updates_path = f"{self.config['PATH_TO_GLOBALSERVER']}{self.config['LOCAL_MODELS_PATH']}{str(experiment['_id'])}"
                try:
                    shutil.rmtree(model_updates_path)
                except FileNotFoundError:
                    print(f"Model not found to delete {model_updates_path}")
                    pass

            result = fl_db.model.remove({"testing": True})
            result = fl_db.experiment.remove({"testing": True})
            result = fl_db.task.remove({"testing": True})

            # model_parameters_path = utils.get_model_parameter_path(model_id)
            # os.remove()

    def kill_global_server(self):
        logging.info("KILLING GLOBAL SERVER")
        if self.global_server:
            self.global_server.kill()
            self.global_server.communicate()
            os.system('pkill -9 -f "globalserver_task_controller.py C88A33B946"')

    def kill_client_interface_node(self):
        logging.info("KILLING CLIENT INTERFACE NODE")

        if self.as_docker:
            os.system('docker kill $(docker ps -q --filter="name=c88a33b946")')
        if self.client_interface_node:
            self.client_interface_node.kill()
            self.client_interface_node.communicate()

    #todo kill specific process
    def kill_client_interface(self):
        logging.info("KILLING CLIENT INTERFACE")

        if self.as_docker:
            os.system('docker kill $(docker ps -q --filter="name=c88a33b946")')
        if self.client_interface:
            self.client_interface.kill()
            self.client_interface.communicate()

    def kill_node_servers(self):

        logging.info("KILLINGCLIENTS")

        if self.as_docker:
            os.system('docker kill $(docker ps -q --filter="name=c88a33b946")')
        else:
            for client, process in self.node_server.items():
                self.node_server[client].kill()
                self.node_server[client].communicate()
            os.system('pkill -9 -f "node_task_controller.py C88A33B946"')

    def kill_servers(self):
        logging.info("KILLING SERVERS")
        self.kill_global_server()
        self.kill_client_interface_node()
        self.kill_client_interface()
        self.kill_node_servers()
        os.system('pkill -9 -f C88A33B946')
        os.system('docker kill $(docker ps -q --filter="name=c88a33b946")')

        for proc in psutil.process_iter():  # for windows users
            # check whether the process name matches
            if proc.name() == "python.exe":
                if "C88A33B946" in proc.cmdline():
                    logging.info(proc.cmdline())
                    proc.kill()
        time.sleep(3)

    # todo start datawrapper if local_data=1
    def start_servers(self, clients, interface):
        self.kill_servers()
        # todo clear test models build test models
        # Define the Clients name (these are the same as you used in startup.sh script)

        self.start_global_server()
        self.clients = clients
        self.start_node_servers()

        # todo skipped
        if interface:
            self.start_client_interface()

    def start_node_servers(self):
        logging.info("STARTING UP NODES")
        node_server = {}
        for client in self.clients:

            if self.as_docker:
                node_server[client] = self.start_node_server_as_docker(client)
            else:
                node_server[client] = self.start_node_server(client)
        self.node_server = node_server

    def start_client_interface(self, ):
        logging.info("STARTING UP Client interface")

        if self.as_docker:
            client_interface, client_interface_node = self.start_client_interface_as_docker()
        else:
            client_interface = self.start_client_interface_process(self.clients[0])
            client_interface_node = self.start_node_server_for_client()
        self.client_interface = client_interface
        self.client_interface_node = client_interface_node

    # @concurrent.process
    def start_node_server_as_docker(self, client):
        log = open("testing/testing.log", 'a+')
        import os
        os.chdir('nodeserver/worker')
        proc = subprocess.Popen(["docker-compose", '--project-name', f"C88A33B946_{client}", 'up'],
                                env=dict(os.environ, PARTNER_NAME=client,
                                         STATIC_VARIABLES_FILE_PATH="static_variables.json",
                                         PATH_TO_GLOBALSERVER="api/"), )

        print("the commandline is {}".format(proc.args))
        # log.close()
        # error_queue.put("node_server failed")
        # raise Exception

        os.chdir('../..')
        return proc

    def start_node_server(self, client):
        log = open("testing/testing.log", 'a+')
        import os
        os.chdir('nodeserver/worker')
        proc = subprocess.Popen([sys.executable, "node_task_controller.py"] + [client] + ['C88A33B946'],
                                env=dict(os.environ, PARTNER_NAME=client,
                                         STATIC_VARIABLES_FILE_PATH="static_variables.json",
                                         PATH_TO_GLOBALSERVER="api/"), stdout=log, stderr=log)

        print("the commandline is {}".format(proc.args))
        # log.close()
        # error_queue.put("node_server failed")
        # raise Exception

        os.chdir('../..')
        return proc

    # @concurrent.process
    def start_client_interface_process(self, client):
        log = open("testing/testing.log", 'a+')
        import os
        os.chdir('globalserver')
        proc = subprocess.Popen(
            [sys.executable, "client_interface_clone/client_interface_controller.py"] + ['C88A33B946'],
            env=dict(os.environ, SERVER="0", PARTNER_NAME=client,
                     STATIC_VARIABLES_FILE_PATH="static_variables.json",
                     PATH_TO_GLOBALSERVER="api/"), stdout=log, stderr=log)

        print("the commandline is {}".format(proc.args))
        # log.close()
        # error_queue.put("node_client_server failed")
        # raise Exception
        os.chdir('..')
        return proc

    def start_client_interface_as_docker(self, ):
        log = open("testing/testing.log", 'a+')
        import os
        os.chdir('globalserver/client_interface_clone')
        proc = subprocess.Popen(["docker-compose", '--project-name', f"C88A33B946_INTERFACE", 'up'],
                                env=dict(os.environ, PARTNER_NAME="INTERFACE",
                                         CLIENT_INTERFACE_SERVER_ADDRESS="client_interface", SERVER="0",
                                         STATIC_VARIABLES_FILE_PATH="static_variables.json",
                                         PATH_TO_GLOBALSERVER="api/"))

        print("the commandline is {}".format(proc.args))
        # log.close()
        # error_queue.put("node_server failed")
        # raise Exception

        os.chdir('../..')
        return proc, proc

    # @concurrent.process
    def start_node_server_for_client(self):
        log = open("testing/testing.log", 'a+')
        import os
        os.chdir('nodeserver/worker')
        proc = subprocess.Popen([sys.executable, "node_task_controller.py"] + ['C88A33B946'],
                                env=dict(os.environ, PARTNER_NAME="INTERFACE",
                                         SERVER_PORT=os.getenv("CLIENT_INTERFACE_PORT"), SERVER="0",
                                         STATIC_VARIABLES_FILE_PATH="static_variables.json",
                                         PATH_TO_GLOBALSERVER="api/"), stdout=log, stderr=log)

        print("the commandline is {}".format(proc.args))
        # log.close()
        # error_queue.put("node_server failed")
        # raise Exception
        os.chdir('../..')
        return proc

    # @concurrent.process
    def start_global_server(self):
        logging.info("STARTING UP GLOBAL SERVER")
        log = open("testing/testing.log", 'a+')
        import os
        os.chdir('globalserver')
        proc = subprocess.Popen([sys.executable, "api/globalserver_task_controller.py"] + ['C88A33B946'],
                                env=dict(os.environ, STATIC_VARIABLES_FILE_PATH="static_variables.json",
                                         PATH_TO_GLOBALSERVER="api/"), stdout=log, stderr=log)

        print("the commandline is {}".format(proc.args))
        # log.close()
        # error_queue.put("global_server failed")
        # raise Exception
        os.chdir('..')
        self.global_server = proc


import importlib


class Tests(Testing):

    def __init__(self, clients=['c1', 'c2'], data_source='1', clear_logs=False, as_docker=False, clear_db=False,
                 start_servers=False,
                 interface=False):

        super().__init__(clients=clients, as_docker=as_docker, data_source=data_source, clear_logs=clear_logs,
                         clear_db=clear_db,
                         start_servers=start_servers,
                         interface=interface)

    # define the model

    def check_processes(self):
        error = False
        error_message = b''
        if self.global_server and self.global_server.poll():
            # self.global_server = self.start_global_server(self.error_queue)
            error = True
            log_message, error_message = self.global_server.communicate()
            # logging.info(log_message.decode("utf-8"))
            return error, error_message
        if self.client_interface_node and self.client_interface_node.poll():
            # self.client_interface_node = self.start_node_server_for_client(self.error_queue)
            error = True
            log_message, error_message = self.client_interface_node.communicate()
            # logging.info(log_message.decode("utf-8"))
            return error, error_message
        if self.client_interface and self.client_interface.poll():
            # self.client_interface = self.start_client_interface_process(self.error_queue)
            error = True
            log_message, error_message = self.client_interface.communicate()
            # logging.info(log_message.decode("utf-8"))
            return error, error_message
        for client, process in self.node_server.items():
            if process.poll():
                # self.node_server[client] = self.start_node_server(client, self.error_queue)
                error = True
                log_message, error_message = process.communicate()
                # logging.info(log_message.decode("utf-8"))
                return error, error_message
        return error, error_message

    def check_test_status(self, test_process):
        while not test_process.poll() and test_process.poll() != 0:
            error, error_message = self.check_processes()
            if error:
                # todo error message
                logging.error(error_message.decode("utf-8"))
                logging.info("FAAAAAIASLELTRLGE")
                test_process.terminate()

                super().__init__(self.clients)

            time.sleep(5)
        logging.info(test_process.poll())
        if test_process.poll() == 0:
            log_message, error_message = test_process.communicate()
            logging.info(log_message)
            logging.info(error_message)
            return True
        else:
            log_message, error_message = test_process.communicate()
            logging.info(log_message)
            logging.info(error_message)

        raise Exception("Test failed")

    def run_all_tests(self, test_cases=None):

        test_cases = test_cases if test_cases else [test_case for test_case in os.listdir('testing/test_wrapper') if
                                                    'test' in test_case]

        result = {}
        test_cases.sort()
        for test_case in test_cases:
            if os.getenv('SERVER_ADDRESS') == json.load(open("envs.key", "r")).get(
                ""
                "REMOTE_SERVER_ADDRESS",os.getenv('SERVER_ADDRESS')) and test_case in [
                '30_30_expermients_as_docker_test_wrapper.py', '11_10_round_test_wrapper.py']:
                continue
            logging.info(f"running {test_case}")
            try:
                importlib.import_module("test_wrapper." + test_case[:-3])
                logging.info(f"Sucesssssssss: {test_case[:-3]}")
                result[test_case] = True
            except Exception as error:
                logging.error(traceback.format_exc())
                logging.warning(error)
                logging.warning("Test Failed")
                result[test_case] = False
                raise Exception
            logging.info(result[test_case])
        return result

    def run_test_case(self, test_case, arg="{}"):

        log = open("testing/testing.log", 'a+')
        import os
        proc = subprocess.Popen([sys.executable, "testing/test_cases/" + test_case] + ['C88A33B946'],
                                env=dict(os.environ, arg=arg), stdout=log, stderr=log)

        print("the commandline is {}".format(proc.args))

        return proc
