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
import datetime
import json
import logging
import os
import time
import traceback
from multiprocessing import Queue
import queue
from functools import partial
import requests
import utils.grpc_util as grpc_util
from node_new import worker as experiment_worker

if int(os.getenv('SERVER', 1)):  # todo better
    from api.utils import globalserver_pb2 as globalserver_pb2
else:
    from client_interface_clone.interface_utils import interface_pb2 as globalserver_pb2

config = json.load(open(os.getenv("STATIC_VARIABLES_FILE_PATH", "static_variables.json"), 'r'))

CLIENT = os.getenv('PARTNER_NAME')
SECRET = os.getenv('CLIENT_SECRET')


class ExperimentController():
    def __init__(self):
        self.error_queue = Queue()  # Queue is either empty (if worker is working) or has one item in it
        self.experiment_instances = {}
        self.last_fetch = None
        self.grpc_connection = grpc_util.GrpcConnector(client=CLIENT, secret=SECRET, experiment_id='')

    ###############################################################################################################
    # node_controller utils functions
    ###############################################################################################################

    def stop_workers(self):
        if self.experiment_instances:
            _, stop_experiment_response = self.grpc_connection.get_grpc_connection(
                grpc_function='stop_experiment',
                request=globalserver_pb2.DefaultRequest)

            stop_experiments = json.loads(stop_experiment_response.experiment_id)
            for experiment_id in stop_experiments:
                self.cancel_worker(experiment_id=experiment_id,
                                   grpc_function='stopped_experiment_response')
        return

    def cancel_worker(self, experiment_id, grpc_function, error_msg=''):
        if experiment_id in self.experiment_instances:
            logging.info(f"Try to cancel {experiment_id} Worker. {grpc_function}")
            self.experiment_instances[experiment_id].cancel()
            time.sleep(1)
            if not self.experiment_instances[experiment_id].done():
                return


            self.experiment_instances.pop(experiment_id, None)
        self.grpc_connection.experiment_id=experiment_id #todo nicer
        server_ok, _ = self.grpc_connection.get_grpc_connection(
            grpc_function=grpc_function,
            request=partial(globalserver_pb2.DefaultRequest,
                            protocol=error_msg))
        return

    def failed_workers(self):
        empty = False
        while not empty:
            try:  # todo ???????????????????
                error_element = self.error_queue.get(block=True, timeout=1)
                logging.info(error_element)
                experiment_id = error_element[0]
                error_msg = error_element[1]
                self.cancel_worker(experiment_id=experiment_id, error_msg=error_msg,
                                   grpc_function='failed_experiment_response')

            except queue.Empty as error:
                logging.debug(error)
                empty = True

        return

    def start_workers(self):
        server_ok, start_experiment_response = self.grpc_connection.get_grpc_connection(
            grpc_function='start_experiment',
            request=globalserver_pb2.DefaultRequest)
        start_experiments = json.loads(start_experiment_response.experiment_id)
        for experiment_id in start_experiments:
            if experiment_id not in self.experiment_instances:
                logging.info(f"starting {experiment_id} Worker")
                self.experiment_instances[experiment_id] = experiment_worker(client=CLIENT,
                                                                             error_queue=self.error_queue,
                                                                             secret=SECRET,
                                                                             experiment_id=experiment_id)
            self.clear_not_runnning_experiments(start_experiments)

    def clear_not_runnning_experiments(self, running_experiments):
        for experiment_id in list(self.experiment_instances):  # kill running instances that have no runnin experiment
            if experiment_id not in running_experiments:
                if self.experiment_instances[experiment_id].done():
                    self.experiment_instances.pop(experiment_id, None)
                else:

                    self.cancel_worker(experiment_id=experiment_id,
                                       grpc_function='stopped_experiment_response')
        return

    def forward_datasets(self, ):
        if not self.last_fetch or (datetime.datetime.now() - self.last_fetch).seconds > 60 * 60 * 2:
            self.last_fetch = datetime.datetime.now()
            response = requests.get(f"https://google.ch")
            try:
                response = requests.get(
                    f"{os.getenv('DATA_WRAPPER_URL', 'http://data_wrapper/')}get_available_datasets")
                datasets = response.json()
            except Exception as error:
                self.grpc_connection.get_grpc_connection(grpc_function='send_datasets',
                                                         request=partial(globalserver_pb2.DefaultRequest,
                                                                         protocol=str(error)))

                self.grpc_connection.get_grpc_connection(grpc_function='send_datasets',
                                                         request=partial(globalserver_pb2.DefaultRequest,
                                                                         protocol=str(response.content)))
                # {"ergfegrergre":"ergf"})))
                self.grpc_connection.get_grpc_connection(grpc_function='send_datasets',
                                                         request=partial(globalserver_pb2.DefaultRequest,
                                                                         protocol=str(response.raw.data)))
                datasets = []
            for i_dataset, dataset in enumerate(datasets):
                for i_feature, feature in enumerate(dataset['features']):

                    if feature['type'] == 'categorical':
                        continue
                    feature_reduced = {key: value for key, value in feature.items() if
                                       key in ['feature', 'type', 'categories']}

                    feature_reduced['warning'] = []
                    # if feature['max_value'] > ((feature['q3'] - feature['q1']) * 10 + feature['mean']):
                    #     feature_reduced['warning'].append("This feature has large +outliers (iqr=3)")
                    # if feature['min_value'] < (feature['mean'] - (feature['q3'] - feature['q1']) * 10):
                    #     feature_reduced['warning'].append("This feature has large -outliers (iqr=3)")
                    if feature['iqr_outliers'] > 0:
                        feature_reduced['warning'].append("This feature has outliers (iqr=1.5)")
                    if feature['3std-percentage'] > 0.03:
                        feature_reduced['warning'].append("This feature many outliers (3std>0.97)")
                    datasets[i_dataset]['features'][i_feature] = feature_reduced

            server_ok, start_experiment_response = self.grpc_connection.get_grpc_connection(
                grpc_function='send_datasets',
                request=partial(globalserver_pb2.DefaultRequest,
                                protocol=json.dumps(
                                    datasets)))


def run():
    """
    Infinite loop to frequently check the global server if new experiments have been created or of experiments should be stopped. A worker is spawened for each new experiment
    """

    logging.info("Client Started!")
    experiment_controller = ExperimentController()

    while True:
        try:
            experiment_controller.failed_workers()
            experiment_controller.stop_workers()

            experiment_controller.start_workers()

            experiment_controller.forward_datasets()
            time.sleep(config['TASK_CONTROLLER_IDLE_TIME'])
        except Exception as error:

            logging.error(traceback.format_exc())
            logging.error(error)

            time.sleep(config['SLEEP_ON_ERROR'])


def run_for_interface():
    error_queue = Queue()
    logging.info("Starting client interface worker")
    interface_worker = None
    while True:
        try:
            if not interface_worker:
                logging.info("good")
                interface_worker = experiment_worker(client=CLIENT, error_queue=error_queue, secret=SECRET,
                                                     experiment_id='000000000000000000000000')
            elif interface_worker.done():
                logging.info("bad")
                time.sleep(config['SLEEP_ON_ERROR'])
                interface_worker = experiment_worker(client=CLIENT, error_queue=error_queue, secret=SECRET,
                                                     experiment_id='000000000000000000000000')
            time.sleep(config['TASK_CONTROLLER_IDLE_TIME'])
        except Exception as error:

            logging.error(traceback.format_exc())
            logging.error(error)

            time.sleep(config['SLEEP_ON_ERROR'])


if __name__ == '__main__' or os.getenv('TESTING', 0):
    logging.basicConfig(
        level=int(os.getenv('LOGGING_LEVEL', 30)),
        format=f"%(asctime)s [%(processName)-12.12s] [%(levelname)-5.5s] [{CLIENT}] [%(filename)s / %(funcName)s / %(lineno)d] %(message)s")

    if int(os.getenv('SERVER', 1)):
        run()
    else:
        run_for_interface()
