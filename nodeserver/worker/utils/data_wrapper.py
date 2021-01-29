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
import dill

import logging
import copy
import json
import traceback
import os

import numpy as np
import requests
import random


class DataWrapper:
    def __init__(self, config, client):

        self.dataset_metadata = {}
        self.config = copy.deepcopy(config)
        self.client = client
        self.preprocessing_function = False

        self._set_preprocessing()
        self.config['training']['batch_size']={}
        self.config['training']['batch_size']['train'] = config['training']['batch_size']
        self.config['training']['batch_size']['validation'] = config['training'].get("batch_size_validation",
                                                                       config['training']['batch_size'])
        self.config['training']['batch_size']['test'] = config['training'].get("batch_size_test", config['training']['batch_size'])

        logging.debug(self.config['training']['batch_size'])

    @staticmethod
    def get_dataset(identifier):
        if not identifier:
            return {}
        logging.info(f"{os.getenv('DATA_WRAPPER_URL', 'http://data_wrapper/')}get_available_datasets")
        response = requests.get(f"{os.getenv('DATA_WRAPPER_URL', 'http://data_wrapper/')}get_available_datasets")
        try:
            for dataset_dict in response.json():
                if dataset_dict['identifier'] == identifier:
                    dataset_dict['features'] = {feature['feature']: feature for feature in dataset_dict['features']}
                    return dataset_dict
        except Exception as e:
            error_msg = traceback.format_exc()
            raise Exception(f"{e} \n {error_msg} \n {response.raw.data} \n {response.content}")

        logging.warning("Dataset not found")
        return {}


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


    def _set_dataset_metadata(self):
        if not self.dataset_metadata:  # first fetch

            if os.getenv('DATA_SOURCE', '0') == '0':
                self.dataset_metadata = self.get_dataset(identifier=self.config["training"].get('dataset', None))

            else:
                if "dataset_metadata" in self.config['training']:
                    self.dataset_metadata = self.config['training']['dataset_metadata']
                else:
                    # todo make nicer
                    self.dataset_metadata = self.get_metadata()

                self.dataset_metadata['features'] = {feature['feature']: feature for feature in
                                                     self.dataset_metadata['features']}

        self.config['dataset_metadata'] = self.dataset_metadata  # todo make nicer


    @staticmethod
    def is_float(val):
        try:
            num = [float(row) for row in val]
        except ValueError:
            return False
        return True


    # todo thisis a duplicated
    # todo consider less then 30 values as categorical if they are floats

    def create_feature_metadata_json(self, data):
        feature_jsons = []
        columns = data[0].keys()
        for column in columns:
            values = [row[column] for row in data if column in row]
            uniques = list(set(values))
            if not self.is_float(values) and len(uniques) < 30:
                feature_json = {"feature": column,
                                "type": "categorical",
                                "categories": list(uniques)}
            elif not self.is_float(values):
                feature_json = {"feature": column,
                                "type": "categorical",
                                "categories": ['too_many_features']}
            else:
                q1 = np.quantile(values, 0.25)
                q3 = np.quantile(values, 0.75)
                iqr = 1.5 * (q3 - q1)
                mean = np.mean(values)
                std = np.std(values)

                feature_json = {"feature": column,
                                "type": "continuous",
                                "min_value": min(values),
                                "max_value": max(values),
                                "q1": q1,
                                "q3": q3,
                                "iqr_outliers": float(
                                    len([x for x in values if x < (q1 - iqr) or x > (q3 + iqr)])) / len(
                                    values),
                                "3std-percentage": float(
                                    len([x for x in values if x < (mean - 3 * std) or x > (mean + 3 * std)])) / len(
                                    values),
                                "mean": mean,
                                "std": std
                                }
            feature_jsons.append(feature_json)

        return feature_jsons


    def get_metadata(self):
        train = []
        with open(f"../../datasets/{self.config['training']['dataset']}train_{self.client}.jsonl") as fp:
            for len_train, line in enumerate(fp):
                if len_train < 10000:
                    train.append(json.loads(line))
        with open(f"../../datasets/{self.config['training']['dataset']}validation_{self.client}.jsonl") as fp:
            for len_validation, l in enumerate(fp):
                pass
        with open(f"../../datasets/{self.config['training']['dataset']}test_{self.client}.jsonl") as fp:
            for len_test, l in enumerate(fp):
                pass
        features = self.create_feature_metadata_json(train)
        metadata = {"identifier": "1",
                    "description": "this is the first dataset",
                    "samples_num": [len_train, len_validation, len_test],
                    "creation_date": "2020-02-25T08:40:44.000Z",
                    'features': features}
        # todo memory problem wtff
        return metadata


    @staticmethod
    def process_row(chunk):
        try:
            chunk = json.loads(chunk)
        except:
            logging.info("traceback chunk was bad")
            return None
        if "label" not in chunk:
            chunk['label'] = chunk.pop(list(chunk.keys())[-1])
        return chunk


class NewWrapper(DataWrapper):
    def __init__(self, config, client):
        super().__init__(config, client)
        logging.info("New Data Wrapper is used.")
        self._set_dataset_metadata()

    def local_data_generator(self, empty_batch, data_type, url):
        with open(url) as response:
            batch_generator = self.create_batch(response, empty_batch, data_type)
            while batch_generator:
                yield next(batch_generator)

    def data_wrapper_generator(self, empty_batch, data_type, url):
        logging.info(self.config['training'].get("batch_size", 512))
        with requests.get(f"{os.getenv('DATA_WRAPPER_URL', 'http://data_wrapper/')}{url}",
                          stream=True) as response:
            batch_generator = self.create_batch(response.iter_lines(), empty_batch, data_type)
            while batch_generator:
                yield next(batch_generator)

    def create_batch(self, response, empty_batch, data_type):
        batch = copy.copy(empty_batch)

        logging.debug("start generator")
        for chunk in response:

            if chunk:  # filter out keep-alive new chunks
                processed_row = self.process_row(chunk)
                if processed_row:
                    for key in batch:
                        batch[key].append(processed_row.get(key, None))
                    if len(batch['label']) >= self.config['training']['batch_size'][data_type]:

                        features, labels = self.process_batch(batch, data_type)
                        yield features, labels
                        batch = copy.copy(empty_batch)


    def process_batch(self, batch, data_type):
        # logging.debug(batch)
        for key in batch:
            if key == 'label':
                continue
            # set default value

            batch[key] = [value if value is not None else self.config['dataset_metadata']['features'][key].get('mean',
                                                                                                               self.config[
                                                                                                                   'preprocessing'].get(
                                                                                                                   'default_value',
                                                                                                                   '(not set)'))
                          for value in batch[key]]

            # add noise
            if data_type == 'train' and (
                    self.config["training"].get("differential_privacy", {}).get("method",
                                                                                'before') == 'before' or os.getenv(
                'DATA_SOURCE', '0') == '0'):
                batch[key] = self.diff_privacy_noise_first(
                    np.array(batch[key]),
                    epsilon=self.config['preprocessing']['noise']['epsilon'],
                    delta=self.config['preprocessing']['noise']['delta'],
                    data_type=self.config['dataset_metadata']['features'][key]['type'],
                    categories=self.config['dataset_metadata']['features'][key].get('categories', []),
                    min_value=self.config['dataset_metadata']['features'][key].get('min_value', 0),
                    max_value=self.config['dataset_metadata']['features'][key].get('max_value', 0))
                # batch[key] = diff_privacy_noise_first_old(
                #     np.array(batch[key]),
                #     epsilon=self.config['preprocessing'].get('noise', {}).get('epsilon', 1000),
                #     delta=self.config['preprocessing'].get('noise', {}).get('delta', 1),
                # )
                # todo
                pass

        # logging.debug(batch)
        if self.preprocessing_function:
            try:  # todo better
                batch = self.preprocessing_function(batch, self.config)
            except:
                batch = self.preprocessing_function(batch)

        # logging.debug(batch)
        labels = np.array(batch.pop('label'))
        features = np.array(list(batch.values())).T

        return features, labels

    @staticmethod
    def diff_privacy_noise_first(X, epsilon, delta, data_type='categorical', categories=None, min_value=0,
                                 max_value=0,
                                 exponential_sampling="uniform"):
        """
        Add diff. private noise to the data directly. For continuous features Laplacian mechanism is used,
        for categorical mechanism a variant of Exponential mechanism is used.
        This function should be called prior to binning of the data. For equations and mathematical proofs see
        examples 4.9 and 4.10 in:
        https://repozitorij.uni-lj.si/Dokument.php?id=112731&lang=slv
        :param X: data (Pandas dataframe or numpy array)
        :param epsilon: epsilon parameter of differential privacy
        :param delta: delta parameter of differential privacy
        :param exponential_sampling: wheter to use uniform sampling or sampling w.r.t. empirical distribution in
        exponential mechanism. Possible values: uniform, empirical
        :param types: list of feature types. If Pandas dataframe is passed, this can be omitted. Can also be omitted
        for numpy array, in this case threshold rule will be applied. If passed as a list, write 'int' for categorical
        features and 'float' for continuous features

        :return:
        """

        epsilon = min(epsilon, os.getenv('DP_EPSILON', 1000))
        delta = min(delta, os.getenv('DP_DELTA', 1))
        try:
            if data_type == 'categorical':  # categorical feature, exponential mechanism #todo doublecheck
                if len(categories) > 100:
                    categories = random.sample(categories, k=100)
                m = len(categories) - 1
                p = (1 - delta) / (m + np.exp(epsilon))
                for i in range(len(X)):
                    # if not missing_mask[i]:
                    bernoulli_draw = np.random.binomial(1, 1 - m * p)
                    if bernoulli_draw == 1:  # keep feature value unchanged
                        continue
                    else:  # change feature value, sample from other features
                        value = X[i]
                        unique_ = [category for category in categories if category != value]
                        if exponential_sampling == "uniform":
                            X[i] = np.random.choice(unique_)
                        else:  # todo
                            X[i] = np.random.choice(unique_, p=0)

            else:  # continuous feature, Laplacian mechanism
                diameter = max_value - min_value
                b = diameter / (epsilon - np.log(1 - delta))
                noise_vector = np.random.laplace(0, b, len(X))
                np.add(X, noise_vector, out=X, casting="unsafe")
        except:
            X = [0 for i in range(len(X))]
        return X

    def generator(self, data_type):
        if data_type not in ['train', 'test', 'validation']:
            raise Exception("Wrong dataset type.")
        empty_batch = {key: [] for key in self.config['dataset_metadata']['features']}
        if "label" not in empty_batch:
            empty_batch.pop(list(empty_batch.keys())[-1])
            empty_batch['label'] = []

        if os.getenv('DATA_SOURCE', '0') == '0':
            data_generator_instance = self.data_wrapper_generator
            url = f"get_dataset?identifier={self.config['training']['dataset']}&type={data_type}"

        elif os.getenv('DATA_SOURCE', '0') == '1':
            data_generator_instance = self.local_data_generator
            url = f"../../datasets/{self.config['training']['dataset']}{data_type}_{self.client}.jsonl"  # todo make this nicer ( is in fetch model as well)

        else:
            assert "bad source"

        for i in range(self.config['training'].get("epochs", 1)):
            data_generator = data_generator_instance(empty_batch, data_type, url)
            while data_generator:
                batch = next(data_generator)
                # raise Exception(batch)
                yield batch

        if self.preprocessing_function:
            try:  # todo better
                self.preprocessing_function({}, {})
            except:
                batch = self.preprocessing_function({})
        return


class OldWrapper(DataWrapper):

    def __init__(self, config, client):
        super().__init__(config, client)
        logging.info("Old Data Wrapper is used.")

    def process_batch(self, batch, data_type):
        # logging.debug(batch)
        for key in batch:
            if key == 'label':
                continue
            # set default value

            batch[key] = [value if value else 0
                          for value in batch[key]]

            # add noise
            if data_type == 'train' and self.config["training"].get("differential_privacy", {}).get("method",
                                                                                                    'before') == 'before':
                batch[key] = self.diff_privacy_noise_first(
                    np.array(batch[key]),
                    epsilon=self.config['preprocessing'].get('noise', {}).get('epsilon', 1000),
                    delta=self.config['preprocessing'].get('noise', {}).get('delta', 1),
                )
            else:
                batch[key] = np.array(batch[key])
        # logging.debug(batch)
        if self.preprocessing_function:
            try:  # todo better
                batch = self.preprocessing_function(batch, self.config)
            except:
                batch = self.preprocessing_function(batch)

        labels = np.array(batch.pop('label'))
        features = np.array(list(batch.values())).T
        return features, labels

    def diff_privacy_noise_first(self, X, epsilon, delta, exponential_sampling="uniform"):
        """
        Add diff. private noise to the data directly. For continuous features Laplacian mechanism is used,
        for categorical mechanism a variant of Exponential mechanism is used.
        This function should be called prior to binning of the data. For equations and mathematical proofs see
        examples 4.9 and 4.10 in:
        https://repozitorij.uni-lj.si/Dokument.php?id=112731&lang=slv
        :param X: data (Pandas dataframe or numpy array)
        :param epsilon: epsilon parameter of differential privacy
        :param delta: delta parameter of differential privacy
        :param exponential_sampling: wheter to use uniform sampling or sampling w.r.t. empirical distribution in
        exponential mechanism. Possible values: uniform, empirical
        :param types: list of feature types. If Pandas dataframe is passed, this can be omitted. Can also be omitted
        for numpy array, in this case threshold rule will be applied. If passed as a list, write 'int' for categorical
        features and 'float' for continuous features

        :return:
        """

        epsilon = min(epsilon, os.getenv('DP_EPSILON', 1000))
        delta = min(delta, os.getenv('DP_DELTA', 1))
        try:  # todo improve
            categories = list(set(X))
            if not self.is_float(X) and len(
                    categories) < 30:  # categorical feature, exponential mechanism #todo doublecheck
                m = len(categories) - 1
                p = (1 - delta) / (m + np.exp(epsilon))
                for i in range(len(X)):
                    # if not missing_mask[i]:
                    bernoulli_draw = np.random.binomial(1, 1 - m * p)
                    if bernoulli_draw == 1:  # keep feature value unchanged
                        continue
                    else:  # change feature value, sample from other features
                        value = X[i]
                        unique_ = [category for category in categories if category != value]
                        if exponential_sampling == "uniform":
                            X[i] = np.random.choice(unique_)
                        else:  # todo
                            X[i] = np.random.choice(unique_, p=0)
            elif not self.is_float(X):  # too many categories for string
                X = [0 for i in range(len(X))]

            else:  # continuous feature, Laplacian mechanism
                X = X.astype(float)
                diameter = abs(max(X) - min(X))
                b = diameter / (epsilon - np.log(1 - delta))
                noise_vector = np.random.laplace(0, b, len(X))
                np.add(X, noise_vector, out=X, casting="unsafe")
        except:
            X = [0 for i in range(len(X))]
        return X

    def generator(self, data_type):
        if data_type not in ['train', 'test', 'validation'] and os.getenv('DATA_SOURCE', '0') == '0':
            raise Exception("Wrong dataset type.")

        # backward compatability
        url = data_type
        if "features" in self.config['preprocessing']:
            empty_batch = {key: [] for key in self.config['preprocessing']['features']}
            if "label" not in empty_batch:
                empty_batch.pop(list(empty_batch.keys())[-1])
                empty_batch['label'] = []
        else:
            empty_batch = None

        for i in range(self.config['training'].get("epochs", 1)):
            if os.getenv('DATA_SOURCE', '0') == '0':
                with requests.get(f"{os.getenv('DATA_WRAPPER_URL', 'http://data_wrapper/')}{url}",
                                  stream=True) as response:
                    batch = copy.copy(empty_batch)
                    for chunk in response.iter_lines():
                        if chunk:  # filter out keep-alive new chunks
                            processed_row = self.process_row(chunk)
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
                            if len(batch['label']) >= self.config['training']['batch_size'][data_type] :
                                features, labels = self.process_batch(batch, data_type)
                                yield features, labels

                                batch = copy.copy(empty_batch)
            elif os.getenv('DATA_SOURCE', '0') == '1':
                with open(f"../../datasets/{url}_{self.client}.jsonl") as response:
                    batch = copy.copy(empty_batch)
                    for chunk in response:
                        processed_row = self.process_row(chunk)
                        if not empty_batch and not batch:  # no features define in config
                            batch = {key: [value] for key, value in processed_row.items()}
                            if "label" not in batch:
                                label = batch.pop(list(batch.keys())[-1])
                                batch["label"] = [label]
                        else:
                            for key in batch:
                                batch[key].append(processed_row.get(key, None))
                        if len(batch['label']) >= self.config['training']['batch_size'][data_type]:
                            features, labels = self.process_batch(batch, data_type)
                            # raise Exception((features, labels))
                            yield features, labels

                            batch = copy.copy(empty_batch)

            # if not enough data streamed, call preprocessing to debug
        if self.preprocessing_function:
            try:  # todo better
                self.preprocessing_function({}, {})
            except:
                batch = self.preprocessing_function({})

        return
