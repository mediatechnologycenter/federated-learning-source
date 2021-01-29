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
import gc

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import utils.RandomForest.forest as RandomForest
import utils.RandomForest.tree as DecisionTree

import logging
import copy
import json
import os

from sklearn.utils import resample

from math import isclose
import numpy as np
import xgboost as xgb
import pickle
from diffprivlib.mechanisms import GeometricTruncated
from sys import maxsize

from sklearn import metrics as skmetrics

from tensorflow import metrics as tfmetrics


class Model:

    def __init__(self, config, wrapper):
        self.config = config

        self.data_generator = wrapper

        self.model = None

        self.global_model = None

    def get_loss(self, data_type):
        y_pred, y_true = self.predict(data_type)
        performance = self.calculate_loss(y_pred=y_pred, y_true=y_true,
                                          tf_metrics=self.config['training'].get('tfmetrics', []),
                                          sk_metrics=self.config['training'].get('skmetrics', []))
        return performance

    @staticmethod
    def calculate_loss(y_pred, y_true, tf_metrics, sk_metrics):
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
            try:
                m.update_state(y_true=y_true, y_pred=y_pred)
            except tf.errors.InvalidArgumentError:
                temp=np.array([y_true,y_pred]).T
                temp=temp[~np.isnan(temp[:, 1])]
                m.update_state(y_true=temp[:,0], y_pred=temp[:,1])
            performance[metric] = m.result().numpy()

        return performance

    def reset_model(self):
        self.model = self.global_model

    def load_model(self, model):
        self.set_model(model)
        self.global_model = self.model

    def predict(self, data_type):
        return 0, 0

    def set_model(self, model):
        return {}


class RF(Model):

    def __init__(self, config, wrapper):
        super().__init__(config, wrapper)
        self.batch = None  # todo ugly

    def get_model_update(self):
        model_update = json.dumps(self.model.model_update)
        return model_update.encode('utf-8')

    def set_model(self, model):

        config = json.loads(model.model_definition)
        self.model = RandomForest.RandomForestClassifier.from_json(config['model'])

        # self._set_dataset()##todo ????????????why
        # self._set_custom_training_config()
        #
        # self._set_preprocessing()
        if self.batch is None:
            generator = self.data_generator.generator("train")
            batch = next(generator)
            batch = np.concatenate((batch[0], batch[1].reshape((self.config['training']['batch_size'], 1))), axis=1)
            import datetime
            np.random.seed(datetime.datetime.now().microsecond)
            np.random.shuffle(batch)

            batch = batch[:self.config['training'].get('bootstrap_size', 1000)]

            if self.config["training"].get("balanced_subsample", "no") == "yes":
                batch_0 = batch[batch[:, -1] == 0]
                batch_1 = batch[batch[:, -1] == 1]

                # resample from both batches the same amount of samples and concatenate the two bootstrap samples
                n_bootstrap_samples = int((len(batch_0) + len(batch_1)) / 2)
                batch_0 = np.array(batch_0)
                batch_1 = np.array(batch_1)
                batch_0_btstrp = resample(batch_0, replace=True, n_samples=n_bootstrap_samples)
                batch_1_btstrp = resample(batch_1, replace=True, n_samples=n_bootstrap_samples)
                self.batch  = np.append(batch_0_btstrp, batch_1_btstrp, axis=0)

            else:
                self.batch  =batch # resample(self.batch, replace=True, stratify=self.batch[:, -1])

        dict_forest = json.loads(model.model_parameters)
        for tree in dict_forest['forest']:
            self.model.forest.append(DecisionTree.DecisionTreeClassifier.from_json(tree))

    def train_model(self):
        """Computes local histogram data for given information. Assumes RF_fetch_model is previously called
        and that the following fields have been set by the server process in the model-configuration-file:
        - current_condition_list
        - current_feature_list
        - random_state
        This function then writes the result into the local model under the attribute model_update

        NOTE: Function assumes positive-label=1, negative-label=0, need to incorporate how we can pass this information to the worker.
        """
        histograms = self.RF_create_histograms()

        self.model.model_update = histograms  # store as string
        gc.collect()
        return True

    def send_model(self, ):
        # Send the model update to the server

        return self.model

    def predict(self, data_type):

        generator = self.data_generator.generator(data_type)
        train_X, train_y = next(generator)

        y_pred = self.model.predict(train_X, )
        if self.config['training'].get('cast_to_probabilities', False):
            y_pred = 1.0 / (1.0 + np.exp(-y_pred))
        return y_pred, train_y

    @staticmethod
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

    def RF_create_histograms(self):

        histograms = {}
        # batch = np.concatenate((batch[0], batch[1].reshape((config['training']['batch_size'], 1))), axis=1)

        batch=self.batch
        if self.model.current_condition_list != [[]]:
            for el in self.model.current_condition_list:
                batch = batch[((batch[:, el['feature_index']] <= el['threshold']) == el['condition'])]
        for feature_idx in self.model.current_feature_list:
            histograms[f"{feature_idx}"] = []

        unique_values = {}
        for f_idx in self.model.current_feature_list:
            if self.model.feature_information.get(f"col{feature_idx}", True) == False:
                unique_values[f"{f_idx}"] = []

        for el in batch:
            for f_idx in self.model.current_feature_list:
                if self.model.feature_information.get(f"col{feature_idx}", True) == False:
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
                    for bin_ in histograms[f"{f_idx}"]:#changed this
                        if isclose(bin_['bin_identifier'], r_i, rel_tol=1e-10):
                            bin_['bin_identifier'] = r_i
                            bin_['n_pos'] = bin_['n_pos'] + p_i
                            bin_['n_neg'] = bin_['n_neg'] + n_i
                            extended = True
                            break
                    if not extended:
                        histograms[f"{f_idx}"].append(current_bin)
                        histograms[f"{f_idx}"].sort(key=lambda x: x['bin_identifier'])
                    # compress histogram by combining bins if needed
                    while (len(histograms[f"{f_idx}"]) > self.model.max_bins):
                        assert (self.model.max_bins >= 2)
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
        if self.config["training"].get("differential_privacy", {}).get("method", 'before') == 'after':
            epsilon = self.config['preprocessing'].get('noise', {}).get('epsilon', 1000),
            return self._dp_histograms(histograms, self.model.current_feature_list, epsilon)

        return histograms


class P2P(Model):
    def __init__(self, config, wrapper):

        super().__init__(config, wrapper)
        self.batch = None  # todo ugly

    def get_model_update(self):
        model_dict = dict()

        # save trees for visualization of the model
        model_dict['trees'] = str(self.model.get_dump())

        # save model object itself, str format. pickle returns bytes format, we transform it to str
        model_dict['pickle'] = str(pickle.dumps(self.model))

        return json.dumps(model_dict).encode('utf-8')

    def set_model(self, model):
        # self.config = json.loads(model.model_definition)
        global_model = pickle.loads(eval(json.loads(model.model_parameters)['pickle']))
        self.model = global_model  # Booster object

    def train_model(self):

        generator = self.data_generator.generator("train")

        train_X, train_y = next(generator)

        train_data_local = xgb.DMatrix(train_X, label=train_y)
        train_params_dict = self.config['compile']['model_params'].copy()

        train_params_dict['nthread'] = self.config['training'].get('nthread', -1)
        train_params_dict['verbosity'] = self.config['training'].get('verbosity', 0)

        self.model = xgb.train(train_params_dict, train_data_local,
                               num_boost_round=self.config['training']['client_steps_per_round'],
                               xgb_model=self.model)

        gc.collect()
        return True

    def send_model(self):

        return self.model

    def predict(self, data_type):
        generator = self.data_generator.generator(data_type)
        train_X, train_y = next(generator)
        validation_data_local = xgb.DMatrix(train_X, label=train_y)

        yhat_probs = self.model.predict(validation_data_local)

        if self.config['training'].get('cast_to_probabilities', False):
            yhat_probs = 1.0 / (1.0 + np.exp(-yhat_probs))

        y_true = validation_data_local.get_label()

        return yhat_probs, y_true


class NN(Model):
    def __init__(self, config, wrapper):

        super().__init__(config, wrapper)

        self.global_weights = None

    def get_model_update(self):

        if int(os.getenv('SERVER', 1)):
            gradient = self.get_gradient(self.get_weights(self.model), self.global_weights)
        else:
            gradient = self.global_weights

        return self.array_to_bytes(gradient)

    def set_model(self, model):
        # self.config = json.loads(model.model_definition) #todo allow always changing config? if yes then split data/train config

        if self.config["training"].get("differential_privacy", {}).get("method", 'before') not in ['after', 'before']:
            raise Exception("Bad differential privacy method set")
        self.model = tf.keras.models.model_from_json(json.dumps(self.config['model']))

        self.model.compile(loss=tf.losses.get(self.config['compile']['loss']),
                           optimizer=self.get_NN_optimizer(),
                           metrics=[getattr(self.import_from_string(metric['module']),
                                            metric['class_name']).from_config(metric["config"]) for
                                    metric in self.config['compile']['metrics']],
                           loss_weights=self.config['compile'].get('loss_weights', None),
                           sample_weight_mode=self.config['compile'].get('sample_weight_mode', None),
                           weighted_metrics=self.config['compile'].get('weighted_metrics', None),
                           target_tensors=self.config['compile'].get('target_tensors', None)
                           )

        self.global_weights = self.array_from_bytes(model.model_parameters)
        self.model = self.set_weights(self.model, self.global_weights,
                                      normalize=self.config['compile'].get("normalize", 0), )

    def train_model(self):

        self.model.fit(
            self.data_generator.generator("train"),
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

        tf.keras.backend.clear_session()
        gc.collect()
        return True

    def send_model(self):

        return self.model

    def predict(self, data_type):
        generator = self.data_generator.generator(data_type)
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

    @staticmethod
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

    @staticmethod
    def array_to_bytes(array):
        array_copy = copy.deepcopy(array)
        for layer_index, layer in enumerate(array_copy):
            for cell_index, _ in enumerate(layer):
                array_copy[layer_index][cell_index] = array_copy[layer_index][cell_index].tolist()
        return json.dumps(array_copy).encode('utf-8')

    @staticmethod
    def array_from_bytes(bytes_array):
        array = json.loads(bytes_array)
        for layer_index, layer in enumerate(array):
            for cell_index, _ in enumerate(layer):
                array[layer_index][cell_index] = np.array(array[layer_index][cell_index])
        return array

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def import_from_string(name):
        components = name.split('.')
        mod = __import__(components[0])
        for comp in components[1:]:
            mod = getattr(mod, comp)
        return mod

    def get_NN_optimizer(self):
        if self.config["training"].get("differential_privacy", {}).get("method", 'before') == 'before':
            return tf.optimizers.get(self.config['compile']['optimizer'])
        raise Exception("wrong differential_privacy set")
