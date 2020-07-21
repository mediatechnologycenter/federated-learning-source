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
import requests
import copy
import logging
import json
import os
import gc
import time
import psutil

# import tensorflow_addons as tfa

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import pickle
import dill
import xgboost as xgb

import numpy as np
from pebble import concurrent
import optuna

from api.utils import globalserver_pb2_grpc, globalserver_pb2
import utils.worker_utils as utils
import utils.grpc_util as grpc_util

import utils.RandomForest.forest as RandomForest
import utils.RandomForest.tree as DecisionTree
import utils.RandomForest.histogram as Histogram

# from tensorflow_privacy.privacy.optimizers.dp_optimizer import DPGradientDescentOptimizer,DPAdamOptimizer

config = json.load(open(os.getenv("STATIC_VARIABLES_FILE_PATH", "static_variables.json"), 'r'))

config['CONTROLLER_IDLE_SLEEP'] = config['CONTROLLER_IDLE_SLEEP'] if int(os.getenv('LOGGING_LEVEL', 30)) >= 20 else 1

# todo
import sys
import numpy

numpy.set_printoptions(threshold=sys.maxsize)


def get_NN_optimizer(config):
    if config["training"].get("differential_privacy", {}).get("method", 'before') == 'before':
        return tf.optimizers.get(config['compile']['optimizer'])
    # elif config["training"].get("differential_privacy", {}).get("method", 'before') == 'after':
    #     #todo this is not working unless tf fixes this https://github.com/tensorflow/community/pull/234
    #     # loss needs to be not reduced= per mircobatch!
    #     logging.debug(config["training"].get("differential_privacy", {}).get("optimizer", "SGD"))
    #     if config["training"].get("differential_privacy", {}).get("optimizer", "SGD")=="SGD":
    #
    #         dp_sum_query = gaussian_query.GaussianSumQuery(4.0, 10.0)
    #         dp_sum_query = privacy_ledger.QueryWithLedger(dp_sum_query, 2400, 1 / 2400)
    #         optimizer= DPGradientDescentOptimizer(
    #             dp_sum_query=dp_sum_query,
    #             # l2_norm_clip=config["training"].get("differential_privacy", {}).get("l2_norm_clip", 1.0),
    #             # noise_multiplier=config["training"].get("differential_privacy", {}).get("noise_multiplier", 1.1),
    #             num_microbatches=config["training"].get("differential_privacy", {}).get("num_microbatches", 250),
    #             learning_rate=config["training"].get("differential_privacy", {}).get("learning_rate", 0.15))
    #     elif config["training"].get("differential_privacy", {}).get("optimizer", "SGD")=="Adam":
    #         optimizer = DPAdamOptimizer(
    #
    #             dp_sum_query=gaussian_query.GaussianSumQuery(1.0, 1000.0),
    #             # l2_norm_clip=config["training"].get("differential_privacy", {}).get("l2_norm_clip", 1.0),
    #             # noise_multiplier=config["training"].get("differential_privacy", {}).get("noise_multiplier", 1.1),
    #             num_microbatches=config["training"].get("differential_privacy", {}).get("num_microbatches", 250),
    #             learning_rate=config["training"].get("differential_privacy", {}).get("learning_rate", 0.15))
    #     return optimizer
    raise Exception("wrong differential_privacy set")


@concurrent.process
def worker(client, secret, error_queue, experiment_id, node=None):
    """
    Separate process for each experiment. It just creates a worker and starts it. the error_queue is shared with the
    node_task_controller.py to tell the controller if a process failed.
    """
    try:
        # Initialization
        node = Node(client, secret, experiment_id)
        node.start_working()

    # Enhanced Error logging:
    except Exception as error:
        logging.info(error)
        error_msg = utils.pretty_print_error(error, node)
        error_queue.put([node.experiment_id, error_msg])
        while error_queue.empty():
            time.sleep(0.1)
        while not error_queue.empty():
            logging.info("alive")
            time.sleep(10)
        raise error


class Node:
    def __init__(self, client, secret, experiment_id):

        logging.info(f"Initialize worker...")
        self.experiment_id = experiment_id
        self.client = client
        self.secret = secret
        self.model = None
        self.global_weights = None
        self.config = {}
        self.data_generator = None
        self.preprocessing_function = None
        # This is only needed until memory leak is fixed, to read memory usage
        self.process = psutil.Process(os.getpid())
        self.batch = None  # todo ugly
        _, self.stub, _ = grpc_util.get_grpc_connection(grpc_function='test_connection',
                                                    request=globalserver_pb2.DefaultRequest(
                                                        client=client, secret=secret, experiment_id=experiment_id))
        self.dataset_metadata = {}
        logging.info(f"GRPC Connection established...")

    # todo fix protocol experiment id
    def start_working(self):
        server_ok = True
        while server_ok:
            server_ok, self.stub, response = grpc_util.get_grpc_connection(grpc_function='get_task_request',
                                                                       request=globalserver_pb2.TaskRequest(
                                                                           client=self.client,
                                                                           secret=self.secret,
                                                                           experiment_id=self.experiment_id),
                                                                       stub=self.stub)

            # bad response from server
            if not response.ok:
                logging.error(f"Message <{response.message}>")
                raise Exception("Bad Response From Server")

            # good response from server
            elif response.task in config['VALID_TASKS']:
                self.run_task(response.task, experiment_id=self.experiment_id, task_id=response.task_id,
                              protocol=response.protocol)

            # No task to do or unknown task
            else:
                # logging.debug(f"Nothing to do with <{response.task}> ...")
                time.sleep(config['CONTROLLER_IDLE_SLEEP'])

    def run_task(self, task, experiment_id, task_id, protocol):
        logging.info(f"Task <{task} - {experiment_id}> received...")
        utils.memory_check(self.process)  # todo temporary "fix" for the memory leak (shutdown)
        task_method = getattr(self, task)
        task_method(experiment_id, task_id, protocol)
        return True

    def fetch_model(self, experiment_id, task_id, protocol):
        logging.info(f"Parsing Model...%s", experiment_id)

        _, self.stub, responses = grpc_util.get_grpc_connection(
            grpc_function='fetch_model_request',
            request=globalserver_pb2.DefaultRequest(client=self.client, task_id=task_id, secret=self.secret,
                                                    experiment_id=experiment_id))
        for row in responses:  # pseudo stream
            load_model = getattr(self, protocol + "_load_model") #todo split into load config and compile model. In load config set custom config!
            load_model(model=row)

        logging.info(f"Model parsed...%s", experiment_id)

        self._set_dataset()
        self._set_custom_training_config()#todo allow to change everything

        self._set_preprocessing()
        tf.keras.backend.clear_session()
        gc.collect()
        return True

    def train_model(self, experiment_id, task_id, protocol):
        train_model_method = getattr(self, f"{protocol}_train_model")
        train_model_method(experiment_id, task_id)

    def send_model(self, experiment_id, task_id, protocol):
        send_model_method = getattr(self, f"{protocol}_send_model")
        send_model_method(experiment_id, task_id)

    def send_validation_loss(self, experiment_id, task_id, protocol):
        self._send_loss(experiment_id, task_id, protocol=protocol, data_type='validation')

    def send_test_loss(self, experiment_id, task_id, protocol):
        self._send_loss(experiment_id, task_id, protocol=protocol, data_type='test')

    def send_training_loss(self, experiment_id, task_id, protocol):
        self._send_loss(experiment_id, task_id, protocol=protocol, data_type='train')

    def _set_preprocessing(self):
        if os.getenv("ALLOW_PREPROCESSING", True):
            if ('custom_preprocessing_function' in self.config['preprocessing']) and (
                    self.client in self.config['preprocessing']['custom_preprocessing_function']):
                self.preprocessing_function = dill.loads(
                    self.config['preprocessing']['custom_preprocessing_function'][self.client].encode('latin_1'))
            elif 'preprocessing_function' in self.config['preprocessing']:
                self.preprocessing_function = dill.loads(
                    self.config['preprocessing']['preprocessing_function'].encode('latin_1'))
            else:
                self.preprocessing_function = None

    def _set_dataset(self):
        if 'dataset' in self.config["training"]:  # new data wrapper
            self._set_dataset_metadata()
            logging.info("New Data Wrapper is used.")
            self.data_generator = self.__generator_new
        else:
            logging.info("Old Data Wrapper is used.")
            self.data_generator = self.__generator

    def _set_dataset_metadata(self):
        if not self.dataset_metadata:  # first fetch

            if os.getenv('DATA_SOURCE', '0') == '0':
                self.dataset_metadata = utils.get_dataset(identifier=self.config["training"].get('dataset', None))
            else:
                if "dataset_metadata" in self.config['training']:
                    self.dataset_metadata = self.config['training']['dataset_metadata']
                else:
                    # todo make nicer
                    self.dataset_metadata = utils.get_metadata(self.client, self.config)

                self.dataset_metadata['features'] = {feature['feature']: feature for feature in
                                                     self.dataset_metadata['features']}

        self.config['dataset_metadata'] = self.dataset_metadata  # todo make nicer

    def _set_custom_training_config(self):
        if "custom" not in self.config['training']:
            return
        if self.client not in self.config['training']["custom"]:
            return
        for key, value in self.config['training']["custom"][self.client].items():
            logging.debug(key)
            self.config['training'][key] = value

    def _send_loss(self, experiment_id, task_id, protocol, data_type):

        predict = getattr(self, f"{protocol}_predict")
        y_pred, y_true = predict(data_type)
        performance = utils.get_loss(y_pred=y_pred, y_true=y_true,
                                     tf_metrics=self.config['training'].get('tfmetrics', []),
                                     sk_metrics=self.config['training'].get('skmetrics', []))
        logging.info(f"Client: {performance}, %s", self.experiment_id)

        _, self.stub, _ = grpc_util.get_grpc_connection(grpc_function=f'send_{data_type}_loss_response',
                                                    request=globalserver_pb2.Loss(client=self.client,
                                                                                  secret=self.secret,
                                                                                  task_id=task_id,
                                                                                  experiment_id=experiment_id,
                                                                                  loss=json.dumps(performance,
                                                                                                  default=utils.default)))
        logging.info("Loss sent...%s", self.experiment_id)
        tf.keras.backend.clear_session()
        gc.collect()
        return True

    # todo split in classes
    def NN_load_model(self, model):
        self.config = json.loads(model.model_definition)

        if self.config["training"].get("differential_privacy", {}).get("method", 'before') not in ['after', 'before']:
            raise Exception("Bad differential privacy method set")
        self.model = tf.keras.models.model_from_json(json.dumps(self.config['model']))

        self.model.compile(loss=tf.losses.get(self.config['compile']['loss']),
                           optimizer=get_NN_optimizer(self.config),
                           metrics=[getattr(utils.import_from_string(metric['module']),
                                            metric['class_name']).from_config(metric["config"]) for
                                    metric in self.config['compile']['metrics']],
                           loss_weights=self.config['compile'].get('loss_weights', None),
                           sample_weight_mode=self.config['compile'].get('sample_weight_mode', None),
                           weighted_metrics=self.config['compile'].get('weighted_metrics', None),
                           target_tensors=self.config['compile'].get('target_tensors', None)
                           )

        self.global_weights = utils.array_from_bytes(model.model_parameters)
        self.model = utils.set_weights(self.model, self.global_weights,
                                       normalize=self.config['compile'].get("normalize", 0), )

    def NN_train_model(self, experiment_id, task_id):
        logging.info("Training...%s", self.experiment_id)

        self.model.fit(
            self.data_generator("train", preprocessing=self.preprocessing_function, config=self.config,
                                client=self.client),
            epochs=self.config['training'].get("epochs", 1),
            verbose=self.config['training'].get("verbose", 0),
            callbacks=self.config['training'].get("callback", []),
            shuffle=self.config['training'].get("shuffle", True),
            class_weight={int(key): value for key, value in
                          self.config['training'].get("class_weight").items()} if
            self.config['training'].get("class_weight", None) else None,
            initial_epoch=self.config['training'].get("initial_epoch", 0),
            steps_per_epoch=self.config['training'].get("steps_per_epoch", 12),
            max_queue_size=self.config['training'].get("max_queue_size", 10),
            workers=1,  # self.config['training'].get("workers", 1),
            use_multiprocessing=self.config['training'].get("use_multiprocessing", False),
        )

        _, self.stub, _ = grpc_util.get_grpc_connection(grpc_function='train_model_response',
                                                    request=globalserver_pb2.DefaultRequest(client=self.client,
                                                                                            secret=self.secret,
                                                                                            task_id=task_id,
                                                                                            experiment_id=experiment_id))
        logging.info("Training finished. %s", self.experiment_id)
        tf.keras.backend.clear_session()
        gc.collect()
        return True

    def NN_send_model(self, experiment_id, task_id):
        logging.info("Sending model...%s", self.experiment_id)

        iterator = utils.stream_model(experiment_id, self.model, self.global_weights, self.client, self.secret,
                                      task_id=task_id)
        _, self.stub, _ = grpc_util.get_grpc_connection(grpc_function='send_model_update_response',
                                                    request=iterator)
        logging.info("Sending model finished%s", self.experiment_id)
        gc.collect()
        return True

    def NN_predict(self, data_type):
        generator = self.data_generator(data_type, preprocessing=self.preprocessing_function, config=self.config,
                                        client=self.client)
        y_pred = []
        y_true = []
        for step in range(
                self.config['training'].get(f"{data_type}_steps", self.config['training'].get("validation_steps", 12))):
            try:
                validation_batch = next(generator)
            except StopIteration:
                y_pred = [y[0] for y in y_pred]
                return y_pred, y_true
            y_true.extend(validation_batch[-1])
            y_pred.extend(self.model.predict_on_batch(validation_batch, ))
        y_pred = [y[0] for y in y_pred]
        return y_pred, y_true

    def RF_load_model(self, model):
        self.config = json.loads(model.model_definition)

        self.model = RandomForest.RandomForestClassifier.from_json(self.config['model'])
        self._set_dataset()
        self._set_custom_training_config()

        self._set_preprocessing()
        if self.batch is None:
            generator = self.data_generator("train", preprocessing=self.preprocessing_function, config=self.config,
                                            client=self.client)
            batch = next(generator)
            batch = np.concatenate((batch[0], batch[1].reshape((self.config['training']['batch_size'], 1))), axis=1)

            np.random.shuffle(batch)

            self.batch = batch[:self.config['training'].get('bootstrap_size', 1000)]
        dict_forest = json.loads(model.model_parameters)
        for tree in dict_forest['forest']:
            self.model.forest.append(DecisionTree.DecisionTreeClassifier.from_json(tree))

    def RF_train_model(self, experiment_id, task_id):
        """Computes local histogram data for given information. Assumes RF_fetch_model is previously called
        and that the following fields have been set by the server process in the model-configuration-file:
        - current_condition_list
        - current_feature_list
        - random_state
        This function then writes the result into the local model under the attribute model_update

        NOTE: Function assumes positive-label=1, negative-label=0, need to incorporate how we can pass this information to the worker.
        """
        logging.info("Training...%s", self.experiment_id)

        batch = self.batch
        histograms = utils.RF_create_histograms(batch, self.config, self.model)

        self.model.model_update = histograms  # store as string

        _, self.stub, _ = grpc_util.get_grpc_connection(grpc_function='train_model_response',
                                                    request=globalserver_pb2.DefaultRequest(client=self.client,
                                                                                            secret=self.secret,
                                                                                            task_id=task_id,
                                                                                            experiment_id=experiment_id))
        logging.info("Training finished. %s", self.experiment_id)
        gc.collect()
        return True

    def RF_send_model(self, experiment_id, task_id):
        # Send the model update to the server
        logging.info("Sending model...%s", self.experiment_id)

        iterator = utils.stream_model_RF(experiment_id, self.model, self.client, self.secret, task_id=task_id)
        _, self.stub, _ = grpc_util.get_grpc_connection(grpc_function='send_model_update_response',
                                                    request=iterator)
        logging.info("Sending model finished%s", self.experiment_id)
        gc.collect()
        return True

    def RF_predict(self, data_type):
        config = copy.copy(self.config)
        if data_type == 'validation':
            config['training']['batch_size'] = config['training'].get("batch_size_validation",
                                                                      config['training']['batch_size'])
        if data_type == 'test':
            config['training']['batch_size'] = config['training'].get("batch_size_test",
                                                                      config['training']['batch_size'])
        generator = self.data_generator(data_type, preprocessing=self.preprocessing_function,
                                        config=config,
                                        client=self.client)
        train_X, train_y = next(generator)

        y_pred = self.model.predict(train_X, )
        if config['training'].get('cast_to_probabilities', False):
            y_pred = 1.0 / (1.0 + np.exp(-y_pred))
        return y_pred, train_y

    def P2P_load_model(self, model):
        self.config = json.loads(model.model_definition)
        global_model = pickle.loads(eval(json.loads(model.model_parameters)['pickle']))
        self.model = global_model  # Booster object

    def P2P_train_model(self, experiment_id, task_id):

        logging.info("Training...%s", self.experiment_id)
        generator = self.data_generator("train", preprocessing=self.preprocessing_function, config=self.config,
                                        client=self.client)

        train_X, train_y = next(generator)

        train_data_local = xgb.DMatrix(train_X, label=train_y)
        train_params_dict = self.config['compile']['model_params'].copy()

        train_params_dict['nthread'] = self.config['training'].get('nthread', -1)
        train_params_dict['verbosity'] = self.config['training'].get('verbosity', 0)

        self.model = xgb.train(train_params_dict, train_data_local,
                               num_boost_round=self.config['training']['client_steps_per_round'],
                               xgb_model=self.model)

        _, self.stub, _ = grpc_util.get_grpc_connection(grpc_function='train_model_response',
                                                    request=globalserver_pb2.DefaultRequest(client=self.client,
                                                                                            secret=self.secret,
                                                                                            task_id=task_id,
                                                                                            experiment_id=experiment_id))
        logging.info("Training finished. %s", self.experiment_id)
        gc.collect()
        return True

    def P2P_send_model(self, experiment_id, task_id):
        logging.info("Sending model...%s", self.experiment_id)

        iterator = utils.stream_model_P2P(experiment_id, self.model, self.client, self.secret, task_id=task_id)
        _, self.stub, _ = grpc_util.get_grpc_connection(grpc_function='send_model_update_response',
                                                    request=iterator)
        logging.info("Sending model finished%s", self.experiment_id)
        gc.collect()
        return True

    def P2P_predict(self, data_type):
        config = copy.copy(self.config)
        if data_type == 'validation':
            config['training']['batch_size'] = config['training'].get("batch_size_validation",
                                                                      config['training']['batch_size'])
        if data_type == 'test':
            config['training']['batch_size'] = config['training'].get("batch_size_test",
                                                                      config['training']['batch_size'])
        generator = self.data_generator(data_type, preprocessing=self.preprocessing_function,
                                        config=config,
                                        client=self.client)
        train_X, train_y = next(generator)
        validation_data_local = xgb.DMatrix(train_X, label=train_y)

        yhat_probs = self.model.predict(validation_data_local)

        if config['training'].get('cast_to_probabilities', False):
            yhat_probs = 1.0 / (1.0 + np.exp(-yhat_probs))

        y_true = validation_data_local.get_label()

        return yhat_probs, y_true

    @staticmethod
    def __generator_new(data_type, preprocessing, config, client=None):
        if data_type not in ['train', 'test', 'validation']:
            raise Exception("Wrong dataset type.")

        empty_batch = {key: [] for key in config['dataset_metadata']['features']}
        if "label" not in empty_batch:
            empty_batch.pop(list(empty_batch.keys())[-1])
            empty_batch['label'] = []

        if os.getenv('DATA_SOURCE', '0') == '0':
            data_generator_instance = utils.data_wrapper_generator
            url = f"get_dataset?identifier={config['training']['dataset']}&type={data_type}"

        elif os.getenv('DATA_SOURCE', '0') == '1':
            data_generator_instance = utils.local_data_generator
            url = f"../../datasets/{config['training']['dataset']}{data_type}_{client}.jsonl"  # todo make this nicer ( is in fetch model as well)

        else:
            assert "bad source"

        for i in range(config['training'].get("epochs", 1)):
            data_generator = data_generator_instance(empty_batch, data_type, preprocessing, config, client, url)
            while data_generator:
                batch = next(data_generator)
                # raise Exception(batch)
                yield batch


        return

    @staticmethod
    def __generator(data_type, preprocessing, config, client=None):
        if data_type not in ['train', 'test', 'validation'] and os.getenv('DATA_SOURCE', '0') == '0':
            raise Exception("Wrong dataset type.")

        # backward compatability
        url = data_type

        if "features" in config['preprocessing']:
            empty_batch = {key: [] for key in config['preprocessing']['features']}
            if "label" not in empty_batch:
                empty_batch.pop(list(empty_batch.keys())[-1])
                empty_batch['label'] = []
        else:
            empty_batch = None

        for i in range(config['training'].get("epochs", 1)):
            if os.getenv('DATA_SOURCE', '0') == '0':
                with requests.get(f"{os.getenv('DATA_WRAPPER_URL', 'http://data_wrapper/')}{url}",
                                  stream=True) as response:
                    batch = copy.copy(empty_batch)
                    for chunk in response.iter_lines():
                        if chunk:  # filter out keep-alive new chunks
                            processed_row = utils.process_row_new(chunk, config)
                            logging.info(type(processed_row))
                            if processed_row is None:
                                continue
                            if not empty_batch and not batch:  # no features define in config
                                batch = {key: [value] for key, value in processed_row.items()}
                                if "label" not in batch:
                                    label = batch.pop(list(batch.keys())[-1])
                                    batch["label"] = [label]
                            else:
                                for key in batch:
                                    batch[key].append(processed_row.get(key, None))
                            if len(batch['label']) >= config['training'].get("batch_size", 512):
                                features, labels = utils.process_batch_old(batch, data_type, preprocessing, config)
                                yield features, labels

                                batch = copy.copy(empty_batch)
            elif os.getenv('DATA_SOURCE', '0') == '1':
                with open(f"../../datasets/{url}_{client}.jsonl") as response:
                    batch = copy.copy(empty_batch)
                    for chunk in response:
                        processed_row = utils.process_row_new(chunk, config)
                        if not empty_batch and not batch:  # no features define in config
                            batch = {key: [value] for key, value in processed_row.items()}
                            if "label" not in batch:
                                label = batch.pop(list(batch.keys())[-1])
                                batch["label"] = [label]
                        else:
                            for key in batch:
                                batch[key].append(processed_row.get(key, None))
                        if len(batch['label']) >= config['training'].get("batch_size", 512):
                            features, labels = utils.process_batch_old(batch, data_type, preprocessing, config)
                            # raise Exception((features, labels))
                            yield features, labels

                            batch = copy.copy(empty_batch)

            #if not enough data streamed, call preprocessing to debug
            if preprocessing:
                preprocessing({})

        return
