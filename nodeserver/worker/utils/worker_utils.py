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

from sklearn.utils import resample

from math import isclose
import grpc
import datetime
import numpy as np
from multiprocessing import Queue
import requests
import random
# import pandas as pd #todo
import xgboost as xgb
import pickle
from sklearn.metrics import accuracy_score
from diffprivlib.mechanisms import GeometricTruncated
from sys import maxsize
from collections import namedtuple

from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn import metrics as skmetrics

from tensorflow import metrics as tfmetrics


if int(os.getenv('SERVER', 1)):
    from api.utils import globalserver_pb2 as globalserver_pb2
    from api.utils.globalserver_pb2_grpc import TaskControllerStub as TaskController
else:
    from client_interface_clone.interface_utils import interface_pb2 as globalserver_pb2
    from client_interface_clone.interface_utils.interface_pb2_grpc import InterfaceControllerStub as TaskController

SERVER_PORT = os.getenv('SERVER_PORT')
config = json.load(open(os.getenv("STATIC_VARIABLES_FILE_PATH", "static_variables.json"), 'r'))



def set_weights(model, weights, normalize=False):
    for layer_index, layer in enumerate(model.layers):
        cell_weights = []
        for cell_index, _ in enumerate(layer.weights):
            # if normalize != 0:
            #     # normalize weight
            #     norm = np.linalg.norm(weights[layer_index][cell_index])
            #     normalized_weigths = weights[layer_index][cell_index] / max([norm / normalize, 1])
            #     cell_weights.append(normalized_weigths)
            # else:
            cell_weights.append(weights[layer_index][cell_index])
        layer.set_weights(cell_weights)
    return model


def array_to_bytes(array):
    array_copy = copy.deepcopy(array)
    for layer_index, layer in enumerate(array_copy):
        for cell_index, _ in enumerate(layer):
            array_copy[layer_index][cell_index] = array_copy[layer_index][cell_index].tolist()
    return json.dumps(array_copy).encode('utf-8')


def array_from_bytes(bytes_array):
    array = json.loads(bytes_array)
    for layer_index, layer in enumerate(array):
        for cell_index, _ in enumerate(layer):
            array[layer_index][cell_index] = np.array(array[layer_index][cell_index])
    return array


def get_weights(model, normalize=False):
    weights = []
    for layer_index, layer in enumerate(model.layers):
        layer_weights = []
        for cell_index, cell_weights in enumerate(layer.get_weights()):
            # if normalize is True:
            #     # normalize weight
            #     norm = np.linalg.norm(cell_weights)
            #     normalized_weights = cell_weights / max([norm / normalize, 1])
            #     layer_weights.append(normalized_weights)
            # else:
            layer_weights.append(cell_weights)
        weights.append(layer_weights)
    return weights


def get_gradient(gradient, global_weights):
    for layer_index, _ in enumerate(gradient):
        for cell_index, _ in enumerate(gradient[layer_index]):
            try:
                gradient[layer_index][cell_index] = gradient[layer_index][cell_index] - global_weights[layer_index][
                    cell_index]
            except IndexError:
                logging.warning(f"No inital weights found in global model. Set to 0.")
                gradient[layer_index][cell_index] = gradient[layer_index][cell_index]

    return gradient


def stream_model(experiment_id, model, global_weights, client, secret, task_id):
    if int(os.getenv('SERVER', 1)):
        gradient = get_gradient(get_weights(model), global_weights)
        yield globalserver_pb2.ModelUpdate(client=client, secret=secret, experiment_id=experiment_id, task_id=task_id)
        yield globalserver_pb2.ModelUpdate(model_update=array_to_bytes(gradient))
    else:
        yield globalserver_pb2.ModelUpdate(client=client, secret=secret, experiment_id=experiment_id, task_id=task_id)
        yield globalserver_pb2.ModelUpdate(model_update=array_to_bytes(global_weights))


def get_loss(y_pred, y_true, tf_metrics, sk_metrics):
    performance = {}
    for metric in sk_metrics:
        try:
            performance[metric] = getattr(skmetrics, metric)(y_pred=y_pred, y_true=y_true)
        except TypeError:
            performance[metric] = getattr(skmetrics, metric)(y_score=y_pred, y_true=y_true)
        except ValueError:
            y_pred_rounded = []
            for value in y_pred:
                y_pred_rounded.append(1 if value > 0.5 else 0)
            performance[metric] = getattr(skmetrics, metric)(y_pred=y_pred_rounded, y_true=y_true)
    for metric in tf_metrics:  # todo add params
        m = getattr(tfmetrics, metric)()
        m.update_state(y_true=y_true, y_pred=y_pred)
        performance[metric] = m.result().numpy()

    return performance


def stream_model_P2P(experiment_id, model, client, secret, task_id):
    model_dict = dict()

    # save trees for visualization of the model
    model_dict['trees'] = str(model.get_dump())

    # save model object itself, str format. pickle returns bytes format, we transform it to str
    model_dict['pickle'] = str(pickle.dumps(model))

    yield globalserver_pb2.ModelUpdate(client=client, secret=secret, experiment_id=experiment_id, task_id=task_id)
    yield globalserver_pb2.ModelUpdate(model_update=json.dumps(model_dict).encode('utf-8'))


def stream_model_RF(experiment_id, model, client, secret, task_id):
    # the RF_train_model filled this attribute with a json-like string containing histogram data
    model_update = json.dumps(model.model_update)

    yield globalserver_pb2.ModelUpdate(client=client, secret=secret, experiment_id=experiment_id, task_id=task_id)
    yield globalserver_pb2.ModelUpdate(model_update=model_update.encode('utf-8'))


def get_dataset(identifier):
    if not identifier:
        return {}

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




def local_data_generator(empty_batch, data_type, preprocessing, config, client, url):
    with open(url) as response:
        batch_generator = create_batch(response, empty_batch, data_type, preprocessing, config)
        while batch_generator:
            yield next(batch_generator)


def data_wrapper_generator(empty_batch, data_type, preprocessing, config, client, url):
    logging.info(config['training'].get("batch_size", 512))
    with requests.get(f"{os.getenv('DATA_WRAPPER_URL', 'http://data_wrapper/')}{url}",
                      stream=True) as response:
        batch_generator = create_batch(response.iter_lines(), empty_batch, data_type, preprocessing, config)
        while batch_generator:
            yield next(batch_generator)


def create_batch(response, empty_batch, data_type, preprocessing, config):
    batch = copy.copy(empty_batch)
    for chunk in response:

        if chunk:  # filter out keep-alive new chunks
            processed_row = process_row_new(chunk, config)
            if process_row:
                for key in batch:
                    batch[key].append(processed_row.get(key, None))
                if len(batch['label']) >= config['training'].get("batch_size", 512):
                    features, labels = process_batch_new(batch, data_type, preprocessing, config)
                    yield features, labels
                    batch = copy.copy(empty_batch)

def is_float(val):
    try:
        num = [float(row) for row in val]
    except ValueError:
        return False
    return True

# todo thisis a duplicated
#todo consider less then 30 values as categorical if they are floats
def create_feature_metadata_json(data):
    feature_jsons = []
    columns = data[0].keys()
    for column in columns:
        values = [row[column] for row in data if column in row]
        uniques = list(set(values))
        if not is_float(values) and len(uniques) < 30:
            feature_json = {"feature": column,
                            "type": "categorical",
                            "categories": list(uniques)}
        elif not is_float(values):
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
                            "iqr_outliers": float(len([x for x in values if x < (q1 - iqr) or x > (q3 + iqr)])) / len(
                                values),
                            "3std-percentage": float(
                                len([x for x in values if x < (mean - 3 * std) or x > (mean + 3 * std)])) / len(values),
                            "mean": mean,
                            "std": std
                            }
        feature_jsons.append(feature_json)

    return feature_jsons


def get_metadata(client, config):
    train = []
    with open(f"../../datasets/{config['training']['dataset']}train_{client}.jsonl") as fp:
        for len_train, line in enumerate(fp):
            if len_train < 10000:
                train.append(json.loads(line))
    with open(f"../../datasets/{config['training']['dataset']}validation_{client}.jsonl") as fp:
        for len_validation, l in enumerate(fp):
            pass
    with open(f"../../datasets/{config['training']['dataset']}test_{client}.jsonl") as fp:
        for len_test, l in enumerate(fp):
            pass

    features = create_feature_metadata_json(train)
    metadata = {"identifier": "1",
                "description": "this is the first dataset",
                "samples_num": [len_train, len_validation, len_test],
                "creation_date": "2020-02-25T08:40:44.000Z",
                'features': features}
    # todo memory problem wtff
    return metadata


def process_row_new(chunk, config):
    try:
        chunk = json.loads(chunk)
    except:
        logging.info("traceback chunk was bad")
        return None
    if "label" not in chunk:
        chunk['label'] = chunk.pop(list(chunk.keys())[-1])
    return chunk


def process_batch_new(batch, data_type, preprocessing, config):
    # logging.debug(batch)
    for key in batch:
        if key == 'label':
            continue
        # set default value

        batch[key] = [value if value is not None else config['dataset_metadata']['features'][key].get('mean',
                                                                                          config['preprocessing'].get(
                                                                                              'default_value',
                                                                                              '(not set)'))
                      for value in batch[key]]

        # add noise
        if data_type == 'train' and (config["training"].get("differential_privacy", {}).get("method", 'before') =='before' or os.getenv('DATA_SOURCE', '0') == '0'):
            batch[key] = diff_privacy_noise_first_new(
                np.array(batch[key]),
                epsilon=config['preprocessing']['noise']['epsilon'],
                delta=config['preprocessing']['noise']['delta'],
                data_type=config['dataset_metadata']['features'][key]['type'],
                categories=config['dataset_metadata']['features'][key].get('categories', []),
                min_value=config['dataset_metadata']['features'][key].get('min_value', 0),
                max_value=config['dataset_metadata']['features'][key].get('max_value', 0))
            # batch[key] = diff_privacy_noise_first_old(
            #     np.array(batch[key]),
            #     epsilon=config['preprocessing'].get('noise', {}).get('epsilon', 1000),
            #     delta=config['preprocessing'].get('noise', {}).get('delta', 1),
            # )
            #todo
            pass


    # logging.debug(batch)
    if preprocessing:
        batch = preprocessing(batch)

    # logging.debug(batch)
    labels = np.array(batch.pop('label'))
    features = np.array(list(batch.values())).T

    return features, labels


def process_batch_old(batch, data_type, preprocessing, config):
    # logging.debug(batch)
    for key in batch:
        if key == 'label':
            continue
        # set default value

        batch[key] = [value if value else 0
                      for value in batch[key]]

        # add noise
        if data_type == 'train' and config["training"].get("differential_privacy", {}).get("method", 'before')=='before':
            batch[key] = diff_privacy_noise_first_old(
                np.array(batch[key]),
                epsilon=config['preprocessing'].get('noise', {}).get('epsilon', 1000),
                delta=config['preprocessing'].get('noise', {}).get('delta', 1),
            )
        else:
            batch[key]=np.array(batch[key])
    # logging.debug(batch)
    if preprocessing:
        batch = preprocessing(batch)

    labels = np.array(batch.pop('label'))
    features = np.array(list(batch.values())).T
    return features, labels


def default(obj):
    if type(obj).__module__ == np.__name__:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj.item()
    raise TypeError('Unknown type:', type(obj))


def process_row(chunk, config):
    chunk = json.loads(chunk)
    label = chunk.pop('label', None)
    if not label:
        label = chunk.pop(list(chunk.keys())[-1])

    if "feature_selection" in config['preprocessing']:
        chunk = [v for k, v in chunk.items() if k in config['preprocessing']["feature_selection"]]
    else:
        chunk = list(chunk.values())
    assert len(chunk) > 0, "Your feature selection resulted in no features"
    chunk.append(label)
    return chunk


def process_batch(batch, preprocessing, config):
    if preprocessing:
        batch = preprocessing(batch)
    return batch


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


def import_from_string(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def memory_check(process):
    if process.memory_info().rss > config['MAX_MEMORY_USAGE']:
        raise Exception('Too much memory used', 'To avoid overflow from memory leak we shutdown the worker')


def pretty_print_error(error, node):
    error_msg = traceback.format_exc()
    logging.error(error_msg)
    logging.error("Variable information for the previous error: ")
    try:
        error_msg = error_msg + str(error)
        logging.error(error)
        node.config.pop('dataset_metadata', None)
        logging_vars = json.dumps({"model_config": node.config,
                                   "experiment_id": node.experiment_id})
        error_msg = error_msg + logging_vars
        logging.error(logging_vars)
        return error_msg

    except Exception as error:
        logging.error(traceback.format_exc())
        return error_msg


def get_grpc_connection(grpc_function, request, server_port=SERVER_PORT, max_retries=config['GRPC_CONNECTION_RETRIES'],
                        delay=config['GRPC_CONNECTION_RETRY_DELAY'],
                        sleep_on_error=config['SLEEP_ON_ERROR'], timeout=config['GRPC_TIMEOUT'], stub=None):
    """
    Occasionally, the server is busy and the request fails. We retry 5 times to get reach the server.
    """
    retries = 0
    # Startup routine, open grpc channel, define messaging queues, start worker subprocess

    options = [('grpc.max_send_message_length', 512 * 1024 * 1024),
               ('grpc.max_receive_message_length', 512 * 1024 * 1024)]
    while retries < max_retries:

        try:
            if stub is None:
                channel = grpc.insecure_channel(os.getenv('SERVER_ADDRESS') + f":{server_port}", options=options)
                stub = TaskController(channel)
            logging.debug(f"calling {grpc_function}")
            method = getattr(stub, grpc_function)

            response = method(request, timeout=timeout)
            return True, stub, response
        except Exception as error:
            retries = retries + 1
            logging.warning(f"GRPC Connection failed for {grpc_function}, retry... attempt {retries}/{max_retries}")
            logging.warning(traceback.format_exc())
            time.sleep(delay * (retries + 1) ** 3)
    raise Exception


def objective_optuna_P2P(trial, objective, dtrain, dtest, model=None):
    """
    Objective function for hyperparameter optimization in P2P protocol.

    Args:
        trial:
        objective:
        dtrain:
        dtest:
        model:

    Returns:

    """

    param = {
        'silent': 1,
        'objective': objective,
        'lambda': trial.suggest_loguniform('lambda', 1e-8, 1.0),
        'alpha': trial.suggest_loguniform('alpha', 1e-8, 1.0),
        'max_depth': trial.suggest_int('max_depth', 1, 9),
        # 'eta': trial.suggest_loguniform('eta', 1e-8, 1.0),
        'gamma': trial.suggest_loguniform('gamma', 1e-8, 1.0),
        'colsample_bytree': trial.suggest_discrete_uniform('colsample_bytree', 0.0, 1.0, 0.1)
        # 'scale_pos_weight': trial.suggest_discrete_uniform('scale_pos_weight', 0.5, 3.0, 0.5)
    }

    bst = xgb.train(param, dtrain, xgb_model=model)
    preds = bst.predict(dtest)
    # if objective is logitraw, need to apply sigmoid to obtain probabilities
    if 'raw' in param['objective']:
        preds = 1.0 / (1.0 + np.exp(-preds))
    pred_labels = np.rint(preds)
    accuracy = accuracy_score(dtest.get_label(), pred_labels)
    return accuracy

def diff_privacy_noise_first_new(X, epsilon, delta, data_type='categorical', categories=None, min_value=0, max_value=0,
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
            if len(categories)>100:
                categories=random.sample(categories,k=100)
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


def is_float(val):
    try:
        num = [float(row) for row in val]
    except ValueError:
        return False
    return True


def diff_privacy_noise_first_old(X, epsilon, delta, exponential_sampling="uniform"):
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
    try: #todo improve
        categories = list(set(X))
        if not is_float(X) and len(categories) < 30:  # categorical feature, exponential mechanism #todo doublecheck
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
        elif not is_float(X):  # too many categories for string
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


def diff_privacy_noise_first(X, epsilon, delta, exponential_sampling="uniform", types=None):
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
    # if isinstance(X, pd.DataFrame):
    #     if types is None:
    #         types = [str(x) for x in list(X.dtypes)]
    #     X = X.values

    if isinstance(X, np.ndarray):
        if types is None:
            THRESHOLD = 0.02
            fun = lambda x: round(len(np.unique(x)) / len(x), 2) if type(x[0]) != str else 0
            threshold_fun = lambda x: 'int' if x <= THRESHOLD else 'float'
            types = np.apply_along_axis(fun, axis=0, arr=X)

            types = list(map(lambda x: threshold_fun(x), list(types)))

    assert exponential_sampling in ["uniform", 'emipirical'], "Wrong exponential sampling parameter!"

    # for numpy safety (when slicing, numpy does not create new dataframe)
    X_ = X.copy()

    # iterate over features
    for f_idx in range(X_.shape[1]):
        # ignore missing values when adding noise
        col_data = X_[:, f_idx].copy()
        # missing_mask = np.isnan(col_data)
        n = len(col_data)

        # check whether the feature is categorical or continuous
        distinct_values = np.unique(col_data)
        if 'int' in types[f_idx]:  # categorical feature, exponential mechanism
            m = len(distinct_values) - 1
            p = (1 - delta) / (m + np.exp(epsilon))
            unique, counts = np.unique(col_data, return_counts=True)
            assert m + 1 == len(unique)
            for i in range(len(col_data)):
                # if not missing_mask[i]:
                bernoulli_draw = np.random.binomial(1, 1 - m * p)
                if bernoulli_draw == 1:  # keep feature value unchanged
                    continue
                else:  # change feature value, sample from other features
                    value = col_data[i]
                    value_ind = np.where(unique == value)[0][0]
                    unique_, counts_ = np.delete(unique, value_ind), np.delete(counts, value_ind)
                    if exponential_sampling == "uniform":
                        col_data[i] = np.random.choice(unique_)
                    else:
                        counts_ = counts_ / sum(counts_)
                        col_data[i] = np.random.choice(unique_, p=counts_)

        else:  # continuous feature, Laplacian mechanism
            diameter = abs(np.max(col_data) - np.min(col_data))
            b = diameter / (epsilon - np.log(1 - delta))
            noise_vector = np.random.laplace(0, b, n)
            col_data += noise_vector

        X_[:, f_idx] = col_data

    return X_




def _dp_histograms(hist_dict, feature_list, epsilon):
    # NOTE: functiona assumes epsilon is a valid float value (>= 0)
    hists = copy.deepcopy(hist_dict)

    dp_mech = GeometricTruncated().set_epsilon(epsilon).set_sensitivity(1).set_bounds(0, maxsize)

    # iterate over all histograms and make them differentially private
    for f_idx in feature_list:
        for i in range(len(hist_dict[f"{f_idx}"])):
            hists[f"{f_idx}"][i]["n_pos"] = dp_mech.randomise(int(hist_dict[f"{f_idx}"][i]["n_pos"]))
            hists[f"{f_idx}"][i]["n_neg"] = dp_mech.randomise(int(hist_dict[f"{f_idx}"][i]["n_neg"]))

        # and filter out empty bins
        hists[f"{f_idx}"] = list(filter(lambda x: not (x["n_pos"] == 0 and x["n_neg"] == 0), hists[f"{f_idx}"]))

    return hists


def RF_create_histograms(batch,config,model):
    histograms = {}
    # batch = np.concatenate((batch[0], batch[1].reshape((config['training']['batch_size'], 1))), axis=1)

    if config["training"].get("balanced_subsample", "no") == "yes":
        batch_0 = batch[batch[:,-1]==0]
        batch_1 = batch[batch[:,-1]==1]

        # resample from both batches the same amount of samples and concatenate the two bootstrap samples
        n_bootstrap_samples = int((len(batch_0) + len(batch_1)) / 2)
        batch_0 = np.array(batch_0)
        batch_1 = np.array(batch_1)
        batch_0_btstrp = resample(batch_0, replace=True, n_samples=n_bootstrap_samples)
        batch_1_btstrp = resample(batch_1, replace=True, n_samples=n_bootstrap_samples)
        batch = np.append(batch_0_btstrp, batch_1_btstrp, axis=0)

    else:
        batch = resample(batch, replace=True, stratify=batch[:, -1])


    if model.current_condition_list != [[]]:
        for el in model.current_condition_list:
            batch = batch[((batch[:, el['feature_index']] <= el['threshold']) == el['condition'])]
    for feature_idx in model.current_feature_list:
        histograms[f"{feature_idx}"] = []

    unique_values = {}
    for f_idx in model.current_feature_list:
        if model.feature_information.get(f"col{feature_idx}", True) == False:
            unique_values[f"{f_idx}"] = []

    for el in batch:
        for f_idx in model.current_feature_list:
            if model.feature_information.get(f"col{feature_idx}", True) == False:
                # handle the case that the feature is categorical
                r_i = el[f_idx]
                # change r_i if having differentially private data
                # if config["training"]["differential_privacy"] == "data":
                #     dp_el = _make_datapoint_DP(el[0], config)
                #     r_i = dp_el[f_idx]
                y_i = el[-1]
                p_i = 0
                n_i = 0
                if y_i == 1:
                    p_i = 1
                elif y_i == 0:
                    n_i = 1
                else:
                    # There must be a mistake in the initialization, either we have more than 3 labels,
                    # or the labels must have been initialized wrong!
                    assert (False)
                # add to existing bin if possible
                if r_i in unique_values.get(f"{f_idx}", []):
                    extended = False
                    for bin_ in histograms[f"{f_idx}"]:
                        if bin_['bin_identifier'] == r_i:
                            bin_['n_pos'] = bin_['n_pos'] + p_i
                            bin_['n_neg'] = bin_['n_neg'] + n_i
                            extended = True
                            break
                    # make sure that the bin has been extended
                    assert (extended is True)
                # else create new bin to append to the histogram
                else:
                    curr_bin = {
                        'bin_identifier': r_i,
                        'n_pos': p_i,
                        'n_neg': n_i,
                    }
                    histograms[f"{f_idx}"].append(curr_bin)
                    histograms[f"{f_idx}"].sort(key=lambda x: x['bin_identifier'])
                    unique_values[f"{f_idx}"].append(r_i)

            else:
                # handle the case that the feature is continuous
                r_i = el[f_idx]
                # change r_i if having differentially private data
                # if config["training"]["differential_privacy"] == "data":
                #     dp_el = _make_datapoint_DP(el[0], config)
                #     r_i = dp_el[f_idx]
                y_i = el[-1]
                p_i = 0
                n_i = 0
                if y_i == 1:
                    p_i = 1
                elif y_i == 0:
                    n_i = 1
                else:
                    # There must be a mistake in the initialization, either we have more than 3 labels,
                    # or the labels must have been initialized wrong!
                    assert (False)
                current_bin = {
                    'bin_identifier': r_i,
                    'n_pos': p_i,
                    'n_neg': n_i,
                }
                # try to add current information to existing bin if possible
                extended = False
                for bin_ in histograms[f"{f_idx}"]:
                    if isclose(bin_['bin_identifier'], r_i, rel_tol=1e-5):
                        bin_['bin_identifier'] = r_i
                        bin_['n_pos'] = bin_['n_pos'] + p_i
                        bin_['n_neg'] = bin_['n_neg'] + n_i
                        extended = True
                        break
                if not extended:
                    histograms[f"{f_idx}"].append(current_bin)
                    histograms[f"{f_idx}"].sort(key=lambda x: x['bin_identifier'])
                # compress histogram by combining bins if needed
                while (len(histograms[f"{f_idx}"]) > config.get("max_bins", 100)):
                    assert (config.get("max_bins", 100) >= 2)
                    # find two closest bins
                    idx_right = 1
                    min_dist = abs(histograms[f"{f_idx}"][1]['bin_identifier'] - histograms[f"{f_idx}"][0][
                        'bin_identifier'])
                    for j in range(2, len(histograms[f"{f_idx}"])):
                        curr_dist = abs(
                            histograms[f"{f_idx}"][j]['bin_identifier'] - histograms[f"{f_idx}"][j - 1][
                                'bin_identifier'])
                        if curr_dist < min_dist:
                            min_dist = curr_dist
                            idx_right = j
                    # combine two closest bins
                    right_bin = histograms[f"{f_idx}"].pop(idx_right)
                    r_l = histograms[f"{f_idx}"][idx_right - 1]['bin_identifier']
                    p_l = histograms[f"{f_idx}"][idx_right - 1]['n_pos']
                    n_l = histograms[f"{f_idx}"][idx_right - 1]['n_neg']
                    r_r = right_bin['bin_identifier']
                    p_r = right_bin['n_pos']
                    n_r = right_bin['n_neg']
                    histograms[f"{f_idx}"][idx_right - 1]['bin_identifier'] = ((p_l + n_l) * r_l + (
                            p_r + n_r) * r_r) / (p_l + p_r + n_l + n_r)
                    histograms[f"{f_idx}"][idx_right - 1]['n_pos'] = p_l + p_r
                    histograms[f"{f_idx}"][idx_right - 1]['n_neg'] = n_l + n_r
    if config["training"].get("differential_privacy",{}).get("method",'before')=='after':

        epsilon = config['preprocessing'].get('noise', {}).get('epsilon', 1000),
        return _dp_histograms(histograms, model.current_feature_list, epsilon)

    return histograms



###############################################################################################################
# node_controller utils functions
###############################################################################################################

def stop_workers(worker_instances, stub, client, secret):
    if worker_instances:
        _, stub, stop_experiment_response = get_grpc_connection(stub=stub,
                                                                grpc_function='stop_experiment',
                                                                request=globalserver_pb2.DefaultRequest(
                                                                    client=client,
                                                                    secret=secret))

        stop_experiments = json.loads(stop_experiment_response.experiment_id)
        for experiment_id in stop_experiments:
            worker_instances, stub = cancel_worker(worker_instances=worker_instances, experiment_id=experiment_id,
                                                   stub=stub, client=client, secret=secret,
                                                   grpc_function='stopped_experiment_response')
    return worker_instances, stub


def cancel_worker(worker_instances, experiment_id, stub, client, secret, grpc_function, error_msg=''):
    if experiment_id in worker_instances:
        logging.info(f"Try to cancel {experiment_id} Worker. {grpc_function}")
        worker_instances[experiment_id].cancel()
        time.sleep(1)
        if not worker_instances[experiment_id].done():
            return worker_instances, stub

    server_ok, stub, response = get_grpc_connection(stub=stub,
                                                    grpc_function=grpc_function,
                                                    request=globalserver_pb2.DefaultRequest(
                                                        experiment_id=experiment_id,
                                                        protocol=error_msg,
                                                        client=client,
                                                        secret=secret))
    return worker_instances, stub


def failed_workers(worker_instances, error_queue, stub, client, secret):
    empty = False
    while not empty:
        try:
            error_element = error_queue.get(block=True, timeout=1)
            logging.info(error_element)
            experiment_id = error_element[0]
            error_msg = error_element[1]
            worker_instances, stub = cancel_worker(worker_instances=worker_instances, experiment_id=experiment_id,
                                                   stub=stub, client=client, secret=secret, error_msg=error_msg,
                                                   grpc_function='failed_experiment_response')

        except:
            empty = True

    return worker_instances, stub


def forward_datasets(stub, client, secret, last_fetch):
    if not last_fetch or (datetime.datetime.now() - last_fetch).seconds > 60 * 60 * 2:
        last_fetch = datetime.datetime.now()
        response=requests.get(f"https://google.ch")
        try:
            response = requests.get(f"{os.getenv('DATA_WRAPPER_URL', 'http://data_wrapper/')}get_available_datasets")
            datasets = response.json()
        except Exception as error:
            server_ok, stub, start_experiment_response = get_grpc_connection(stub=stub,
                                                                             grpc_function='send_datasets',
                                                                             request=globalserver_pb2.DefaultRequest(
                                                                                 client=client,
                                                                                 secret=secret,
                                                                                 protocol=str(error)))

            server_ok, stub, start_experiment_response = get_grpc_connection(stub=stub,
                                                                             grpc_function='send_datasets',
                                                                             request=globalserver_pb2.DefaultRequest(
                                                                                 client=client,
                                                                                 secret=secret,
                                                                                 protocol=str(response.content)))
            server_ok, stub, start_experiment_response = get_grpc_connection(stub=stub,
                                                                             grpc_function='send_datasets',
                                                                             request=globalserver_pb2.DefaultRequest(
                                                                                 client=client,
                                                                                 secret=secret,
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

        server_ok, stub, start_experiment_response = get_grpc_connection(stub=stub,
                                                                         grpc_function='send_datasets',
                                                                         request=globalserver_pb2.DefaultRequest(
                                                                             client=client,
                                                                             secret=secret,
                                                                             protocol=json.dumps(datasets)))

    return stub, last_fetch


def start_workers(worker, worker_instances, error_queue, stub, client, secret):
    server_ok, stub, start_experiment_response = get_grpc_connection(stub=stub,
                                                                     grpc_function='start_experiment',
                                                                     request=globalserver_pb2.DefaultRequest(
                                                                         client=client,
                                                                         secret=secret))
    start_experiments = json.loads(start_experiment_response.experiment_id)
    for experiment_id in start_experiments:
        if experiment_id not in worker_instances:
            logging.info(f"starting {experiment_id} Worker")
            worker_instances[experiment_id] = worker(client=client, error_queue=error_queue, secret=secret,
                                                     experiment_id=experiment_id)

    for experiment_id in list(worker_instances):  # kill running instances that have no runnin experiment
        if experiment_id not in start_experiments:
            if worker_instances[experiment_id].done():
                worker_instances.pop(experiment_id)
            else:

                worker_instances, stub = cancel_worker(worker_instances=worker_instances, experiment_id=experiment_id,
                                                       stub=stub, client=client, secret=secret,
                                                       grpc_function='stopped_experiment_response')
    return worker_instances, stub


