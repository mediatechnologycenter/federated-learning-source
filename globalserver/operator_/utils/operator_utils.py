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

import numpy as np
import random
import math
import json
import dill
import os
import logging
import datetime
import xgboost as xgb
import pickle
import time
from tensorflow import optimizers
# from tensorflow_privacy.privacy.optimizers.dp_optimizer import DPGradientDescentGaussianOptimizer

config = json.load(open(os.getenv("STATIC_VARIABLES_FILE_PATH", "static_variables.json"), 'r'))
config['PATH_TO_GLOBALSERVER'] = os.getenv("PATH_TO_GLOBALSERVER", config['DEFAULT_GLOBAL_SERVER_PATH'])


def build_model_document(model_id, model, model_name, model_description, protocol, git_version, is_running, testing):
    model_document = {"model_name": model_name,
                      "model_description": model_description,
                      "is_running": is_running,
                      "protocol": protocol,
                      "model": model,
                      "git_version": git_version,
                      "timestamp": datetime.datetime.utcnow(), "testing": testing
                      }
    if model_id:
        model_document['_id'] = model_id
    return model_document


def set_preprocessing_from_setupdict(setup_dict):
    preprocessing = setup_dict.get('preprocessing', {}).copy()
    if setup_dict['preprocessing'].get('preprocessing_function', None):
        preprocessing['preprocessing_function'] = dill.dumps(
            setup_dict['preprocessing']['preprocessing_function']).decode('latin_1')
    return preprocessing


def set_tasks_from_setupdict(setup_dict):
    round = setup_dict.get("round",
                           ["fetch_model", "train_model", "send_model", "send_validation_loss", "aggregate"])
    tasks = []
    for i in range(setup_dict['rounds']):  # number of rounds
        tasks.extend(round)

    tasks.extend(setup_dict.get("final_round", ["fetch_model", "send_test_loss", "send_validation_loss"]))
    return tasks


def get_git_version(setup_dict):
    try:
        import git
        repo = git.Repo(search_parent_directories=True)
        git_version = repo.head.object.hexsha
    except:
        git_version = setup_dict.get('git_version', "not defined")


def build_task_document(experiment_id, task_name, task_order, clients, testing):
    task_document = {"experiment_id": experiment_id,
                     "task_name": task_name,
                     "task_order": task_order,
                     "clients": {client: {"status": config["NOT_SCHEDULED_STOPWORD"], "result": []} for client in
                                 clients},
                     "timestamp": datetime.datetime.utcnow(),
                     "testing": testing
                     }
    return task_document


def get_current_task(experiment_document, last_task_order):
    for current_task in experiment_document["task_list"]:
        if current_task["task_status"] != config['TASK_DONE']:
            break

    if last_task_order != current_task['task_order']:
        logging.debug(f"Working on {current_task}")
    return current_task, current_task['task_order']


def build_task_list_document(task_id, task_order, task_name, task_status):
    task_list_document = {"task_id": task_id, "task_order": task_order, "task_name": task_name,
                          "task_status": task_status}
    return task_list_document


def get_model(db, model_id):
    model_document = list(db.model.find({"_id": model_id}))[0]
    parameters = get_parameters(model_id)
    return model_document, parameters


def valid_experiment(db, experiment_id):
    experiment_documents = list(db.experiment.find({"_id": experiment_id}).limit(1))

    if len(experiment_documents) == 0:
        logging.info(f"Experiment tried to start does not exist. {experiment_id}")
        return False
    elif experiment_documents[0]['is_running']:  # todo is that ok?
        logging.info(f"Experiment already started. {experiment_id}")
        return True
    elif experiment_documents[0].get('is_finished', False):
        logging.info(f"Experiment already finished. {experiment_id}")
        return False
    elif experiment_documents[0].get('has_failed', False):
        logging.info(f"Experiment failed on last run - reset experiment before starting it again. {experiment_id}")
        return False
    return True


def build_experiment_document(experiment_id, start_model_id,
                              experiment_state_model_id,
                              training_config, task_list,
                              clients, git_version, protocol,
                              experiment_description,
                              experiment_name, is_running, testing):
    experiment_document = {"start_model_id": start_model_id,
                           "experiment_state_model_id": experiment_state_model_id,
                           "training_config": training_config, "task_list": task_list,
                           "clients": clients, "git_version": git_version, "protocol": protocol,
                           "is_running": is_running,
                           "experiment_description": experiment_description,
                           "experiment_name": experiment_name,
                           "timestamp": datetime.datetime.utcnow(), "testing": testing}
    if experiment_id:
        experiment_document['_id'] = experiment_id
    return experiment_document


def json_dump_numpy_array(obj):
    if type(obj).__module__ == np.__name__:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj.item()
    raise TypeError('Unknown type:', type(obj))


def get_parameters(model_id):
    model_parameters_path = get_model_parameter_path(model_id)
    with open(model_parameters_path, 'r') as reader:
        model_parameters = reader.read().encode('utf-8')
    return json.loads(model_parameters)['parameters']


def get_model_parameter_path(model_id):
    model_parameters_path = f"{config['PATH_TO_GLOBALSERVER']}{config['GLOBAL_MODELS_PATH']}{model_id}.json"
    os.makedirs(os.path.dirname(model_parameters_path), exist_ok=True)
    return model_parameters_path


def aggregate_loss(results):
    if results and "aggregated_metric" not in list(results.values())[-1] and len(list(results.values())[0]) == len(
            list(results.values())[-1]):
        result = list(results.values())[-1]
        if type(json.loads(list(result.values())[0])) == dict:
            metrics = {}
            for client_metrics in result.values():
                client_metrics = json.loads(client_metrics)
                for metric, value in client_metrics.items():
                    if metric in metrics:
                        metrics[metric].append(value)
                    else:
                        metrics[metric] = [value]
                    try:
                        np.mean(np.array(metrics[metric]), axis=0)
                    except TypeError:
                        metrics[metric] = [0]
            average_metric = {metric: np.mean(np.array(values), axis=0).tolist() for metric, values in metrics.items()}
        else:
            metrics = []
            for client_metrics in result.values():
                metric_row = []
                client_metrics = json.loads(client_metrics)
                for metric in client_metrics:
                    if type(metric) == list:
                        metric_row.extend(metric)
                    else:
                        metric_row.append(metric)
                    metrics.append(metric_row)

            average_metric = np.mean(np.array(metrics), axis=0).tolist()

        logging.info(f"Average loss of {average_metric} reached in this round.")
        return average_metric
    return []


def save_model_parameters(model_id, model_document, parameters, send_model_task_id=None, overwrite=False):
    model_document['parameters'] = parameters
    model_document['_id'] = str(model_id)
    model_document['timestamp'] = str(model_document['timestamp'])
    model_parameters_path = get_model_parameter_path(model_id)
    if (not os.path.exists(model_parameters_path)) or overwrite:
        if os.getenv('BACKUP', config['BACKUP_ALL_INBETWEEN_MODELS']) and send_model_task_id:
            os.rename(model_parameters_path,
                      f"{model_parameters_path.replace(config['GLOBAL_MODELS_PATH'], config['BACKUP_PATH'])}_{send_model_task_id}.bak")

        with open(model_parameters_path, 'wb') as reader:
            reader.write(json.dumps(model_document, default=json_dump_numpy_array).encode('utf-8'))



    else:
        raise Exception(f"there is already a model stored with this path {model_parameters_path}")


def get_params(protocol, model):
    if protocol == 'NN':
        return get_weights(model)

    if protocol == "P2P":
        return get_P2P_params(model)

    if protocol == "RF":
        return model[1]


def jsonify_model_definition(model, protocol, meta_model_params=None, metrics=None):
    if protocol == 'NN':
        model_definition = {
            'model': json.loads(model.to_json()),
            'compile': {
                "loss": model.loss_functions[0].get_config()['name'],
                "metrics": [
                    {'module': metric.__module__.replace(".python", ""), 'class_name': metric.__class__.__name__,
                     'config': metric.get_config()} for metric in model.metrics],
                'optimizer': {
                    'class_name': model.optimizer.get_config()['name'],
                    'config': {k: v for k, v in model.optimizer.get_config().items() if
                               k != 'name'}
                },
            },
            'preprocessing': {}
        }
    # for P2P jsonifying function is not even necessary, could work directlz with model params dict.
    # Decided to work with it anyway, just to more closely follow pipeline for NN.
    if protocol == "P2P":
        model_definition = {
            'model': {'class_name': {}, 'config': {}, 'xgboost_version': xgb.__version__, 'backend': {}},
            'compile': {'model_params': meta_model_params['params']}
        }

    if protocol == "RF":
        model_definition = model[0]
    return model_definition


def load_clients_model_updates(experiment_id, send_model_task_id, clients):
    clients_model_updates = {}
    for client in clients:
        model_updates_path = f"{config['PATH_TO_GLOBALSERVER']}{config['LOCAL_MODELS_PATH']}{experiment_id}/{client}/{send_model_task_id}.json"
        with open(model_updates_path, 'r') as reader:
            model_updated = reader.read().encode('utf-8')
        try:
            clients_model_updates[client] = json.loads(model_updated)['model_update']
        except json.decoder.JSONDecodeError:
            time.sleep(5)
            clients_model_updates[client] = json.loads(model_updated)['model_update']

    return clients_model_updates


def NN_aggregate_model_updates(clients_model_updates, global_model_parameters, update_rate, verbose, aggregation_type):
    """
    For each layer average the gradient and add it to the global model layer with some update_rate
    """
    agg_weights = []
    clients_model_updates_copy = list(clients_model_updates.values())
    for layer_num, _ in enumerate(clients_model_updates_copy[0]):
        agg_cells = []
        for cell_num, _ in enumerate(clients_model_updates_copy[0][layer_num]):
            weight = np.array(global_model_parameters[layer_num][cell_num])
            # if aggregation_type == 'metric':
            #     contribution = [client_metric[1] for client_metric in clients_metric]
            #     contribution = [c / sum(contribution) for c in contribution]
            #     gradient = np.sum(
            #         [contribution[i] * (clients_weights[i][layer_num][cell_num]) for i in
            #          range(len(clients_weights))],
            #         axis=0)
            # else:  # aggregation_type == 'mean':
            if aggregation_type == 'mean':
                gradient = np.mean(
                    [np.array(client_weights[layer_num][cell_num]) for client_weights in clients_model_updates_copy],
                    axis=0)
            weight = weight + update_rate * gradient
            norm = np.linalg.norm(np.array(weight))
            if norm > 40 and verbose > 0:
                logging.info(f"Warning: Weight matrix is getting big: {norm}, {layer_num}/{cell_num}")
            elif norm > 100:
                logging.info(f"Warning: Weight matrix is getting big: {norm}, {layer_num}/{cell_num}")
            agg_cells.append(weight)
        agg_weights.append(agg_cells)
    return agg_weights


# duplicate
# NOTE to christian: I use this function in the RF_aggregate function to mark all tasks as completed once
# the tree is fully built. So when this function will be deleted, one needs to update this function call there.
def task_completion(db, task_id, experiment_id, client):
    task = list(db.task.find({"_id": task_id}).limit(1))
    if len(task) == 0:
        logging.warning(f"client {client} sent a task completion with invalid {task_id}")
        return False
    task_update = db.task.update_one({"_id": task_id},
                                     {"$set": {f"clients.{client}.status": config["TASK_DONE"]}})
    if all(client_status["status"] == 'done' for client_key, client_status in task[0]['clients'].items() if
           client_key != client):
        task_list_update = db.experiment.update_one({"_id": experiment_id, "task_list.task_id": task_id},
                                                    {"$set": {f"task_list.$.task_status": config["TASK_DONE"]}})
    return True


def get_weights(model, normalize=False):
    weights = []
    for layer_index, layer in enumerate(model.layers):
        layer_weights = []
        for cell_index, cell_weights in enumerate(layer.get_weights()):
            # if normalize != 0:
            #     # normalize weight
            #     norm = np.linalg.norm(cell_weights)
            #     normalized_weights = cell_weights / max([norm / normalize, 1])
            #     layer_weights.append(normalized_weights)
            # else:
            layer_weights.append(cell_weights)
        weights.append(layer_weights)
    return weights


def get_P2P_params(model):
    params = dict()

    # save trees for visualization of the model
    params['trees'] = str(model.get_dump())

    # save model object itself, str format. pickle returns bytes format, we transform it to str
    params['pickle'] = str(pickle.dumps(model))
    return params


# todo duplicate
def array_from_bytes(bytes_array):
    array = bytes_array
    for layer_index, layer in enumerate(array):
        for cell_index, _ in enumerate(layer):

            array[layer_index][cell_index] = np.array(array[layer_index][cell_index])
    return array


# todo duplicate
def set_weights(model, weights, normalize=False):
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


# Util functions to setup the Random Forest according to given specifications to be able to start building a RF
def RF_get_compiled_model(specification):
    """This function takes in a specification dictionary and sets up a random forest model.
    This function then returns a json_model_definition object that has to be passed to
    operator.define_model under the model parameter, as well as an empty list that should
    be passed to the parameters-parameter.
    ::
    Parameters
    ==========
    specifications: dict
        The following fields can be specified, when not specified, the default values
        will be chosen to build up the Forest:
        - n_features: int
            Number of features in the input data (necessary)
        - feature_information: dict
            Dictionary containing for each feature if it's values are continuous (True)
            or categorical (False).
            Naming convention for columns are ``f"col{i}"`` where i is the index of the
            respective column in the input data (index starting at 0).
            When no value is given, the feature will be assumed to have continuous values.
        - n_estimators: int (default=128)
            Number of trees to build in the random-forest
        - max_depth: int or None (default=50)
            Maximum depth of any tree.
        - max_features: String, int (default="sqrt")
            Maximum number of features to consider when looking for the best split.
            Values:
            - instance of int: consider max_features many features
            - "sqrt": consider sqrt(n_features) features
            - "log2": consider log_2(n_features) features
        - max_bins: int (default=100)
            Maximum number of bins allowed for continuous features while building
            histograms during the tree-build.
        - pos_label: int (default=1)
            Positive label of the data.
        - neg_label: int (default=0)
            Negative label of the data.
        - minimal_information_gain: float (default=0.0)
            Minimal information gain to not insert a leaf at the tree building process.
        - metrics: list of strings (default=['log_loss'])
            List of metrics that the model should be evaluated on when sending loss
            back. Possible values are 'log_loss', 'accuracy', 'f1_score'
    Returns
    =======
    Python Dictionary containing all necessary specifications for the random forest, and
    a python dictionary representing a forest with one empty, non-trained tree.
    """
    # construction of all information corresponding to the full forest
    model_dict = {'n_features': specification['n_features'],
                  'feature_information': specification.get('feature_information', {}),
                  'n_estimators': specification.get('n_estimators', 128),
                  'max_depth': specification.get('max_depth', 50),
                  'max_features': specification.get('max_features', 'sqrt'),
                  'max_bins': specification.get('max_bins', 100), 'pos_label': specification.get('pos_label', 1),
                  'neg_label': specification.get('neg_label', 0),
                  'minimal_information_gain': specification.get('minimal_information_gain', 0.0),
                  'metrics': specification.get('metrics', ['log_loss']),
                  'model_update': {},
                  'current_condition_list': ([],)}#todo ???
    # (internal state for training)
    if specification.get('max_features', 'sqrt') == 'sqrt':
        n_subfeatures = int(math.floor(math.sqrt(specification['n_features'])))
        model_dict['current_feature_list'] = random.sample([i for i in range(specification['n_features'])],
                                                           n_subfeatures)
    elif specification.get('max_features', 'sqrt') == 'log2':
        n_subfeatures = int(math.floor(math.log(specification['n_features'], 2)))
        model_dict['current_feature_list'] = random.sample([i for i in range(specification['n_features'])],
                                                           n_subfeatures)
    elif isinstance(specification.get('max_features', 'sqrt'), int):
        n_subfeatures = specification.get('max_features', 'sqrt')
        model_dict['current_feature_list'] = random.sample([i for i in range(specification['n_features'])],
                                                           n_subfeatures)
    model_dict['random_state'] = random.randint(0, 999999999)

    # construction of a forest having only one tree, since can only build one tree per experiment
    empty_forest_dict = {
        'forest': [
            {
                'n_features': specification['n_features'],
                'max_features': specification.get('max_features', 'sqrt'),
                'feature_information': specification.get('feature_information', {}),
                'max_depth': specification.get('max_depth', 50),
                'max_bins': specification.get('max_bins', 100),
                'minimal_information_gain': specification.get('minimal_information_gain', 0.0),
                'tree': "None"
            }
        ]
    }
    return {"model": model_dict, "preprocessing": specification.get('preprocessing', {})}, empty_forest_dict


# Util function used in the RF_aggregate at the operator
def RF_aggregate_model_updates(clients_model_updates, global_model_document, global_model_parameters, clients):
    """Parameter Annotations:
    client_model_updates: dict
        Indexable by client in clients, contains a dictionary as value which is indexable by the feature_index
        (as string)
    global_model_document: dict
    global_model_parameters: List
        List with one tree (dictionary)
    """
    # per worker, we have received one dictionary, indexable by the model-feature-indices (as string)
    # each feature has a list associated with it, corresponding to a list of bins (dicts)
    model_specification = global_model_document["model"]["model"]
    model_parameters = global_model_parameters['forest']
    # extract necessary information from the model specification
    max_depth = model_specification.get("max_depth", 50)  # int
    max_bins = model_specification.get("max_bins", 100)  # int
    feature_information = model_specification[
        "feature_information"]  # dict, indexable by f"col{i}" with i the feature index
    minimal_information_gain = model_specification.get("minimal_information_gain", 0.0)  # float
    current_condition_list = model_specification["current_condition_list"]  # float
    current_feature_list = model_specification["current_feature_list"]  # float
    # aggregate the information to a new dictionary
    # logging.debug(clients_model_updates)
    # logging.debug((model_specification['current_condition_list'],model_specification['current_feature_list']))
    # logging.debug(global_model_parameters['forest'][0]['tree'])
    # logging.debug(global_model_parameters.get('parameters',[{'tree':None}])[0]['tree'])
    aggregated_histogram = {}
    for f_i in current_feature_list:
        aggregated_histogram[f"{f_i}"] = []

    for client in clients:
        for f_i in current_feature_list:
            aggregated_histogram[f"{f_i}"] = _merge_histogram_lists(
                hist_1=aggregated_histogram[f"{f_i}"],
                hist_2=clients_model_updates[client][f"{f_i}"],
                is_cont=feature_information.get(f"col{f_i}", True),
                n_bins=max_bins
            )
    # make sure histograms are sorted
    for f_i in current_feature_list:
        aggregated_histogram[f"{f_i}"] = sorted(aggregated_histogram[f"{f_i}"], key=lambda k: k['bin_identifier'])

    # now all information is stored in aggregated_histogram
    # Now find the optimal split
    best_gain = 0.0
    best_feature_index = -1
    best_threshold = 0
    for f_i in current_feature_list:
        for h_i in range(len(aggregated_histogram[f"{f_i}"]) - 1):
            threshold = float(aggregated_histogram[f"{f_i}"][h_i]['bin_identifier'] +
                              aggregated_histogram[f"{f_i}"][h_i + 1]['bin_identifier']) / 2.0
            gain = _histogram_information_gain(
                histogram=aggregated_histogram[f"{f_i}"],
                threshold=threshold
            )
            if gain >= best_gain:
                best_gain = gain
                best_feature_index = f_i
                best_threshold = threshold

    # get number of samples for current node
    n_samples = 0
    for f_i in current_feature_list:
        for bin_ in aggregated_histogram[f"{f_i}"]:
            n_samples = n_samples + bin_['n_pos'] + bin_['n_neg']
    n_samples = n_samples / len(current_feature_list)
    # check if new node should be a leaf or not (if max_depth, )
    is_final_leaf = False
    if (len(current_condition_list) >= max_depth) or best_gain < minimal_information_gain or (
            n_samples <= 1) or (best_feature_index == -1):
        is_final_leaf = True
    new_tree_node = None
    if is_final_leaf:
        # get mode for label in current data
        mode_y = model_specification.get("pos_label", 1)
        n_pos = 0
        n_neg = 0
        for f_i in current_feature_list:
            for bin_ in aggregated_histogram[f"{f_i}"]:
                n_pos = n_pos + bin_['n_pos']
                n_neg = n_neg + bin_['n_neg']
        if n_neg > n_pos:
            mode_y = model_specification.get("neg_label", 0)
        # setup node
        new_tree_node = {
            'feature_index': "None",
            'threshold': "None",
            'depth': len(current_condition_list),
            'is_final_leaf': True,
            'y': mode_y,
            'left_child': "None",
            'right_child': "None",
        }
    else:
        new_tree_node = {
            'feature_index': best_feature_index,
            'threshold': best_threshold,
            'depth': len(current_condition_list),
            'is_final_leaf': False,
            'y': "None",
            'left_child': "None",
            'right_child': "None",
        }
    # insert current node into model_parameter dict
    updated_tree_trunk = _RF_aggregate_insert_node(model_parameters[0]['tree'], current_condition_list, new_tree_node)
    updated_tree = model_parameters[0]
    updated_tree['tree'] = updated_tree_trunk
    global_model_parameters["parameters"] = [updated_tree]

    # get information for next node, i.e. condition_list

    model_specification['current_condition_list'], wtf_bool = _RF_aggregate_get_next_node(model_parameters[0]['tree'],
                                                                                          [])
    # change feature-list for next node
    if model_specification.get('max_features', 'sqrt') == 'sqrt':
        n_subfeatures = int(math.floor(math.sqrt(model_specification['n_features'])))
        model_specification['current_feature_list'] = random.sample(
            [i for i in range(model_specification['n_features'])], n_subfeatures)
    elif model_specification.get('max_features', 'sqrt') == 'log2':
        n_subfeatures = int(math.floor(math.log(model_specification['n_features'], 2)))
        model_specification['current_feature_list'] = random.sample(
            [i for i in range(model_specification['n_features'])], n_subfeatures)
    elif isinstance(model_specification.get('max_features', 'sqrt'), int):
        n_subfeatures = model_specification.get('max_features', 'sqrt')
        model_specification['current_feature_list'] = random.sample(
            [i for i in range(model_specification['n_features'])], n_subfeatures)

    # reset model_update
    model_specification['model_update'] = {}
    # logging.debug((model_specification['current_condition_list'],model_specification['current_feature_list']))
    # logging.debug(global_model_parameters['forest'][0]['tree'])
    # logging.debug(global_model_parameters.get('parameters',{'tree':None})[0]['tree'])
    # update the changes to the global model document
    global_model_document["model"]["model"] = model_specification

    return global_model_document, global_model_parameters, not wtf_bool


def _entropy_histogram(histogram):
    """ Histogram is a list of dictionaries """
    n_pos = 0
    n_neg = 0

    for bin_ in histogram:
        n_pos += bin_['n_pos']
        n_neg += bin_['n_neg']
    n_total = n_pos + n_neg

    if (n_pos is 0) or (n_neg is 0) or (n_total is 0):
        return 0.0

    s = 0.0
    probability_pos = float(n_pos) / n_total
    probability_neg = float(n_neg) / n_total
    s += probability_pos * np.log(probability_pos)
    s += probability_neg * np.log(probability_neg)

    return -s


def _histogram_information_gain(histogram, threshold):
    """ Histogram is a list of dictionaries """
    E_y = _entropy_histogram(histogram)
    # compute E_y_true and E_y_false
    n_pos_lower = 0
    n_neg_lower = 0
    n_pos_higher = 0
    n_neg_higher = 0
    for bin_ in histogram:
        if bin_['bin_identifier'] <= threshold:
            n_pos_lower += bin_['n_pos']
            n_neg_lower += bin_['n_neg']
        else:
            n_pos_higher += bin_['n_pos']
            n_neg_higher += bin_['n_neg']
    n_total = float(n_pos_lower + n_neg_lower + n_pos_higher + n_neg_higher)
    n_lower = float(n_pos_lower + n_neg_lower)
    n_higher = float(n_pos_higher + n_neg_higher)

    # to avoid division by 0, return if any n_* is 0
    if (n_total is 0) or (n_lower is 0) or (n_higher is 0) or (n_pos_lower is 0) or (n_neg_lower is 0) or (
            n_pos_higher is 0) or (n_neg_higher is 0):
        return 0.0

    s_low = 0.0
    s_high = 0.0

    p_pos_low = float(n_pos_lower) / n_lower
    p_neg_low = float(n_neg_lower) / n_lower
    p_pos_high = float(n_pos_higher) / n_higher
    p_neg_high = float(n_neg_higher) / n_higher

    s_low += (p_pos_low * np.log(p_pos_low))
    s_low += (p_neg_low * np.log(p_neg_low))
    s_high += (p_pos_high * np.log(p_pos_high))
    s_high += (p_neg_high * np.log(p_neg_high))

    E_y_true = - s_low
    E_y_false = - s_high
    return E_y - float((E_y_true * n_lower) + (E_y_false * n_higher)) / n_total


def _merge_histogram_lists(hist_1, hist_2, is_cont, n_bins=100):
    merged_histogram = []
    # merge two histograms with merge-sort approach
    hist_1.sort(key=lambda x: x['bin_identifier'])
    hist_2.sort(key=lambda x: x['bin_identifier'])
    l = 0
    r = 0
    while (l < len(hist_1) and r < len(hist_2)):
        if hist_1[l]['bin_identifier'] == hist_2[r]['bin_identifier']:
            merged_histogram.append({
                'bin_identifier': hist_1[l]['bin_identifier'],
                'n_pos': hist_1[l]['n_pos'] + hist_2[r]['n_pos'],
                'n_neg': hist_1[l]['n_neg'] + hist_2[r]['n_neg'],
            })
            l = l + 1
            r = r + 1
        elif hist_1[l]['bin_identifier'] < hist_2[r]['bin_identifier']:
            merged_histogram.append({
                'bin_identifier': hist_1[l]['bin_identifier'],
                'n_pos': hist_1[l]['n_pos'],
                'n_neg': hist_1[l]['n_neg'],
            })
            l = l + 1
        elif hist_1[l]['bin_identifier'] > hist_2[r]['bin_identifier']:
            merged_histogram.append({
                'bin_identifier': hist_2[r]['bin_identifier'],
                'n_pos': hist_2[r]['n_pos'],
                'n_neg': hist_2[r]['n_neg'],
            })
            r = r + 1
    while (l < len(hist_1)):
        merged_histogram.append({
            'bin_identifier': hist_1[l]['bin_identifier'],
            'n_pos': hist_1[l]['n_pos'],
            'n_neg': hist_1[l]['n_neg'],
        })
        l = l + 1
    while (r < len(hist_2)):
        merged_histogram.append({
            'bin_identifier': hist_2[r]['bin_identifier'],
            'n_pos': hist_2[r]['n_pos'],
            'n_neg': hist_2[r]['n_neg'],
        })
        r = r + 1
    # if corresponding feature is continuous, then summarize histogram in n_bins
    if is_cont:
        while (len(merged_histogram) > n_bins):
            assert (n_bins >= 2)
            # find two closest bins
            idx_right = 1
            min_dist = abs(merged_histogram[1]['bin_identifier'] - merged_histogram[0]['bin_identifier'])
            for j in range(2, len(merged_histogram)):
                curr_dist = abs(merged_histogram[j]['bin_identifier'] - merged_histogram[j - 1]['bin_identifier'])
                if curr_dist < min_dist:
                    min_dist = curr_dist
                    idx_right = j
            # combine two closest bins
            right_bin = merged_histogram.pop(idx_right)
            r_l = merged_histogram[idx_right - 1]['bin_identifier']
            p_l = merged_histogram[idx_right - 1]['n_pos']
            n_l = merged_histogram[idx_right - 1]['n_neg']
            r_r = right_bin['bin_identifier']
            p_r = right_bin['n_pos']
            n_r = right_bin['n_neg']
            merged_histogram[idx_right - 1]['bin_identifier'] = ((p_l + n_l) * r_l + (p_r + n_r) * r_r) / (
                    p_l + p_r + n_l + n_r)
            merged_histogram[idx_right - 1]['n_pos'] = p_l + p_r
            merged_histogram[idx_right - 1]['n_neg'] = n_l + n_r

    return merged_histogram


def _RF_aggregate_get_next_node(tree_dict, current_condition_list):
    if tree_dict == "None":  # if leaf case in current tree
        # return condition-list and return True to indicate that one has found a leaf to extend
        return current_condition_list, True
    else:  # normal tree-node case
        # check if current node is a final leaf, then return False
        if tree_dict['is_final_leaf']:
            return current_condition_list, False
        l = current_condition_list + [
            {'feature_index': tree_dict['feature_index'], 'threshold': tree_dict['threshold'], 'condition': True}]
        # check in left subtree if can extend
        list_subtree, extendable = _RF_aggregate_get_next_node(tree_dict['left_child'], l)

        if extendable:
            return list_subtree, True
        r = current_condition_list + [
            {'feature_index': tree_dict['feature_index'], 'threshold': tree_dict['threshold'], 'condition': False}]
        # check in right subtree if can extend
        list_subtree, extendable = _RF_aggregate_get_next_node(tree_dict['right_child'], r)

        if extendable:
            return list_subtree, True
        # when here, cannot extend subtree, so return
        return None, False


def _RF_aggregate_insert_node(node_dict, current_condition_list, node):
    if node_dict == "None" and (current_condition_list== [[]] or len(current_condition_list)==0):  # case when directly calling on root-node or nothing left
        return node
    elif node_dict == "None":
        # this case should never occur!
        assert (False)
    else:
        # i have a tree-node as node_dict
        # assert that we have the right node
        assert (current_condition_list[0]['feature_index'] == node_dict['feature_index'])
        assert (current_condition_list[0]['threshold'] == node_dict['threshold'])

        # if condition is True, update left_child by recursion, else right_child
        if current_condition_list[0]['condition']:
            node_dict['left_child'] = _RF_aggregate_insert_node(node_dict['left_child'], current_condition_list[1:],
                                                                node)
        else:
            node_dict['right_child'] = _RF_aggregate_insert_node(node_dict['right_child'], current_condition_list[1:],
                                                                 node)
        # return updated node
        return node_dict


def is_float(val):
    try:
        num = [float(row) for row in val]
    except ValueError:
        return False
    return True


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


def get_metadata(client, path):
    train = []
    with open(f"{path}train_{client}.jsonl") as fp:
        for len_train, line in enumerate(fp):
            if len_train < 10000:
                train.append(json.loads(line))
    with open(f"{path}validation_{client}.jsonl") as fp:
        for len_validation, l in enumerate(fp):
            pass
    with open(f"{path}test_{client}.jsonl") as fp:
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


# todo these are duplicates
def _get_model_parameters(fl_db, model_id=None, experiment_id=None, ):
    if model_id:
        model_parameters_path = get_model_parameter_path(model_id)
        with open(model_parameters_path, 'r') as reader:  # todo load config from db?
            model_parameters = reader.read().encode('utf-8')
        model_parameters = json.loads(model_parameters)
    elif experiment_id:
        model_parameters = load_global_model(db=fl_db, experiment_id=experiment_id)

    return model_parameters


def load_global_model(db, experiment_id):
    experiment_document = list(db.experiment.find({"_id": experiment_id}).limit(1))
    if len(experiment_document) == 0:
        return None

    model_parameters_path = get_model_parameter_path(str(experiment_document[0]['experiment_state_model_id']))
    with open(model_parameters_path, 'r') as reader:
        model_parameters = reader.read().encode('utf-8')


    return json.loads(model_parameters)


def get_NN_optimizer(config):
    if config["training"].get("differential_privacy", {}).get("method", 'before') == 'before':
        return optimizers.get(config['compile']['optimizer'])
    elif config["training"].get("differential_privacy", {}).get("method", 'before') == 'after':
        return DPGradientDescentGaussianOptimizer(
            l2_norm_clip=config["training"].get("differential_privacy", {}).get("l2_norm_clip", 1.0),
            noise_multiplier=config["training"].get("differential_privacy", {}).get("noise_multiplier", 1.1),
            num_microbatches=config["training"].get("differential_privacy", {}).get("num_microbatches", 250),
            learning_rate=config["training"].get("differential_privacy", {}).get("learning_rate", 0.15))
    raise Exception("wrong differential_privacy set")


def import_from_string(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod
