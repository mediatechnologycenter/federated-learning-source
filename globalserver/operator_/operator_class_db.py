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

The Python implementation of the MTC Federated Learning Operator.

define_and_start_experiment:    Creates a new model as well as a new experiment and starts the experiment.
define_model:                   Creates a new model and store it in the db.
                                The model parameters are stored in the folder "db/global_models"
define_experiment:              Creates an experiment on top of a model. The experiment its own model "experiment_state_model".
                                The experiment and with that this model and the tasks are stored in the db.
start_experiment:               Changes the experiment to "is_running". This means that the global_task_controller starts to
                                schedule the tasks. Here, we check the db from time to time to see the status. When the task
                                "aggregate" is scheduled we aggregate the the model_updates. When all tasks are done we set the
                                experiment to "is_finished".
reset_experiment:               If the experiments fails, this function resets it to initial state.
set_as_final_model:             sets a model as a final model
objective:                      objective function to run hyperparameter optimization. Runs define_and_start_experiment
                                for every constellation.
"""

import json
import time
import datetime
import os
import contextlib
import logging
import numpy as np
import dill
import random
from bson.objectid import ObjectId
from multiprocessing import Queue
from multiprocessing import Process
from pebble import concurrent
import globalserver.operator_.utils.operator_utils as utils
from pymongo import MongoClient
import subprocess

import globalserver.operator_.utils.RandomForest.forest as RandomForest
import globalserver.operator_.utils.RandomForest.tree as DecisionTree
import pickle

config = json.load(open(os.getenv("STATIC_VARIABLES_FILE_PATH", "static_variables.json"), 'r'))
config['PATH_TO_GLOBALSERVER'] = os.getenv("PATH_TO_GLOBALSERVER", config['DEFAULT_GLOBAL_SERVER_PATH'])
db_config = json.load(open(config['PATH_TO_GLOBALSERVER'] + config['DB_CONFIG_FILE'], 'r'))

logging.basicConfig(
    level=int(os.getenv('LOGGING_LEVEL', 0)),
    format=f"%(asctime)s [%(processName)-12.12s] [%(levelname)-5.5s] [OPERATOR] [%(filename)s / %(funcName)s / %(lineno)d] %(message)s")

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf


class Operator:
    def __init__(self):
        client = MongoClient(host=db_config['host'], port=int(db_config['port']), username=db_config['user'],
                             password=db_config['password'])
        self.fl_db = client.federated_learning
        self.db_session = client.start_session()

    def define_model(self, model, protocol, model_name, model_description, git_version, is_running, parameters,
                     new_transaction=True, testing=False, model_id=None):
        """
        dict model: model as json/dict,
        string protocol: the alorithm/protocol we should use,
        string model_name, model_description: strings to help identify the model
        string git_version: the last commit hash from git used in the script
        bool is_running: if the model is the experiment_state of a not-finished experiment, this should be True
        dict parameters: the model parameters that will be stored in the filesystem
        bool testing: if the model is just for testing set to True
        bool new_transaction: This is for mongodb. If creating the model is part of another transaction set it to False
        """

        model_document = utils.build_model_document(model_id=model_id, model=model,
                                                    protocol=protocol,
                                                    model_description=model_description,
                                                    model_name=model_name,
                                                    git_version=git_version, is_running=is_running, testing=testing)
        result = self.fl_db.model.insert_one(model_document, session=self.db_session)
        model_id = result.inserted_id

        result = self.fl_db.model.update_one({"_id": model_id},
                                             {"$set": {
                                                 "parameters_path": utils.get_model_parameter_path(model_id)}})

        utils.save_model_parameters(model_id=model_id, model_document=model_document, parameters=parameters)

        return model_id

    def define_experiment(self, start_model_id, training_config,
                          tasks, clients, git_version,
                          experiment_name,
                          experiment_description, model_name=None, experiment_id=None, experiment_state_model_id=None,
                          model_description=None, testing=False):
        """
        object start_model_id:      id of the model we start your experiment with,
        dict training_config:       Configuration json on how to train the model. Is attached to the model_config of the
                                    experiment_state_model,
        list tasks:                 list of the task to perform in correct order
        list clients                list of clients to run the experiment with
        string git_version:         the last commit hash from git used in the script
        string experiment_name, experiment_description: strings to help identify the experiment
        string model_name, model_description:   strings to help identify the experiment state model if not set
                                                inherit from experiment
        bool testing: if the model is just for testing set to True
        """
        # with self.db_session.start_transaction():
        start_model_document, parameters = utils.get_model(db=self.fl_db, model_id=start_model_id)
        if start_model_document['is_running']:
            raise Exception("start_model is state model of other experiment")
        start_model_document['model']['training'] = training_config

        experiment_state_model_id = self.define_model(model_id=experiment_state_model_id,
                                                      model=start_model_document['model'],
                                                      parameters=parameters,
                                                      protocol=start_model_document['protocol'],
                                                      model_name=model_name if model_name else experiment_name,
                                                      model_description=model_description if model_description else experiment_description,
                                                      git_version=git_version, is_running=True,
                                                      new_transaction=False, testing=testing)

        experiment_document = utils.build_experiment_document(experiment_id=experiment_id,
                                                              start_model_id=start_model_id,
                                                              experiment_state_model_id=experiment_state_model_id,
                                                              training_config=training_config, task_list=[],
                                                              is_running=False,
                                                              protocol=start_model_document['protocol'],
                                                              clients=clients, git_version=git_version,
                                                              experiment_description=experiment_description,
                                                              experiment_name=experiment_name, testing=testing)

        result = self.fl_db.experiment.insert_one(experiment_document, session=self.db_session)
        experiment_id = result.inserted_id

        task_list = self.__define_task_list(experiment_id=experiment_id, tasks=tasks, clients=clients,
                                            testing=testing, protocol=start_model_document['protocol'])

        result = self.fl_db.experiment.update_one({"_id": experiment_id}, {"$set": {"task_list": task_list}})

        return experiment_id

    def start_experiment(self, experiment_id, idle_time=5, stop_function=None, upkeep_function=None, timeout=None):
        """
        stop_function:  callable function that is run in every aggregate step to store additional metrics
                        and early stop an experiment
                        Input: self, experiment_document, current_task
                        Output: True if experiment reached stop criterion, otherwise False; additional_metrics
        upkeep_function:callable function that runs in every aggregate step for whatever reason (ex. reshuffle data)
                        Input: self, experiment_document, current_task
                        Output: -
        """
        if not utils.valid_experiment(db=self.fl_db, experiment_id=experiment_id):
            logging.info(f"Experiment {experiment_id} not started.")
            return False
        logging.info(experiment_id)
        experiment_start_time = datetime.datetime.utcnow()
        result = self.fl_db.experiment.update_one({"_id": experiment_id}, {
            "$set": {"is_running": True, "experiment_start_time": experiment_start_time}})
        metrics = self.work_on_experiment(experiment_id, idle_time=idle_time, stop_function=stop_function,
                                          upkeep_function=upkeep_function, timeout=timeout)
        return metrics

    def start_experiments_RF(self, experiment_id, experiment_id_list, operator_list, idle_time=5, n_workers=10):
        """Start all experiments, creating one tree per experiment. This function spawns multiple processes in a
        local process-pool to execute all experiments. Once all experiments are finished, i.e. we have all trees
        built up, this function then aggregates all trees into one forest and writes them down in the corresponding
        file.

        Parameters
        ==========
        experiment_id
            Same experiment_id as for any experiment, but for the overhead experiment that builds up the whole
            forest and not just a single tree.
        experiment_id_list
            List of experiment_id's, each corresponding to one tree that should be built.
        idle_time: int
            Time the child-processes wait in between tasks to be finished.
        n_workers: int
            Number of processes to spawn in this overhead process.

        NOTE: (TODO) find a way to push start_experiment onto process-pool
        Current-problem: cannot picke object, and somehow does not work when initializing with different operator object
        # """
        # assert (len(experiment_id_list) == len(operator_list))
        # todo
        self.__build_RF_trees(experiment_id_list, operator_list, idle_time)

        self.wait_for_experiments_to_finish(experiment_id_list, idle_time)

        # once all experiments are finished, gather all individual trees together and put them into the
        # parameter json corresponding to the overhead experiment_id
        self.__RF_add_trees_to_random_forest(experiment_id, experiment_id_list)
        # mark current overhead experiment as finished

        # self.wait_for_experiments_to_finish([experiment_id], idle_time)
        # experiment_document = list(self.fl_db.experiment.find({"_id": experiment_id}))[0]

        # after whole forest is constructed, start the task-list of the overhead process
        # to gather all losses from the individual agents
        operator = Operator()
        operator.start_experiment(experiment_id, idle_time)

        return True

    # todo experiment_id in attribut und klassen aufsplitten
    def work_on_experiment(self, experiment_id, idle_time=5, stop_function=None, upkeep_function=None, timeout=None):

        # todo make get last task nicer
        metrics = []
        last_task_order = -1
        current_task={'task_status':""}
        experiment_document = self.fl_db.experiment.find_one({"_id": experiment_id})
        while experiment_document.get("is_running", False):
            current_task, last_task_order = utils.get_current_task(experiment_document, last_task_order)

            if current_task['task_name'] == config['AGGREGATE_TASK'] and current_task['task_status'] != config[
                'TASK_DONE']:
                metric = self.aggregate(experiment_document=experiment_document,
                                        current_task=current_task,
                                        stop_function=stop_function, upkeep_function=upkeep_function)
                metrics.append(metric)

            experiment_document = self._running_experiment_upkeep(current_task=current_task,
                                                                  experiment_id=experiment_id,
                                                                  experiment_document=experiment_document,
                                                                  idle_time=idle_time, timeout=timeout)

        metrics = self._save_aggregated_metrics(experiment_document, stop_function)
        self._finish_up_experiment(experiment_id, experiment_document, current_task, metrics, idle_time)
        return metrics

    def aggregate(self, experiment_document, current_task, stop_function, upkeep_function):
        """
            Runs the aggregate function of the corrsponding algorithm.
            Aggregates the metrics.
            Runs the stop function to and potentially early stops the experiment
            Runs the upkeep function
            Sets the aggregate task to complete
        """
        logging.info(f"Aggregating...")
        for latest_send_task in reversed(experiment_document["task_list"]):
            if latest_send_task["task_status"] == config['TASK_DONE'] and \
                    latest_send_task['task_name'] == 'send_model':
                aggregate_method = getattr(self, experiment_document['protocol'] + "_" + "aggregate")
                clients = experiment_document['clients']

                aggregate_method(experiment_id=str(experiment_document['_id']),
                                 global_model_id=experiment_document['experiment_state_model_id'],
                                 send_model_task_id=latest_send_task["task_id"],
                                 clients=clients,
                                 aggregation_config=experiment_document['training_config'],
                                 round_num=latest_send_task['task_order'])
                break

        return self._aggregation_step_upkeep(experiment_document=experiment_document, current_task=current_task,
                                             latest_send_task=latest_send_task, stop_function=stop_function,
                                             upkeep_function=upkeep_function)

    def define_and_start_experiment(self, setup_dict, additional_description='',start_model_id=None):
        experiment_id = self.setup_up_experiment_by_dict(setup_dict, additional_description,start_model_id)
        metrics = self.start_experiment(experiment_id, stop_function=setup_dict.get('stop_function', None),
                                        upkeep_function=setup_dict.get('upkeep_function', None))

        return experiment_id, metrics

    def setup_up_experiment_by_dict(self, setup_dict, additional_description='',start_model_id=None):
        """
                    Defines a model and an experiment and runs the experiment.
                """
        git_version = utils.get_git_version(setup_dict)
        if start_model_id:
            model_id=start_model_id
        else:
            model = setup_dict['model_function']['function'](setup_dict['model_function']['parameters'])
            model_parameters = utils.get_params(setup_dict['protocol'], model)  # todo for non NN
            model = utils.jsonify_model_definition(model=model, protocol=setup_dict['protocol'],
                                                   meta_model_params=setup_dict['model_function']['parameters'])
            model['preprocessing'] = utils.set_preprocessing_from_setupdict(setup_dict)
            # define model
            model_id = self.define_model(model=model, parameters=model_parameters, protocol=setup_dict['protocol'],
                                         model_name=setup_dict['model_name'],
                                         model_description=setup_dict['model_description'] + additional_description,
                                         git_version=git_version, is_running=False,
                                         testing=setup_dict['testing'])

        tasks = utils.set_tasks_from_setupdict(setup_dict)

        experiment_id = self.define_experiment(start_model_id=model_id, training_config=setup_dict['training_config'],
                                               tasks=tasks, clients=setup_dict['clients'],
                                               git_version=git_version,
                                               experiment_name=setup_dict['experiment_name'],
                                               experiment_description=setup_dict[
                                                                          'experiment_description'] + additional_description,
                                               testing=setup_dict['testing'])
        return experiment_id

    def wait_for_experiments_to_finish(self, experiment_id_list, idle_time):
        # wait for all experiments to finish
        for exp_id in experiment_id_list:
            experiment_document = list(self.fl_db.experiment.find({"_id": exp_id}))[0]
            while experiment_document.get("is_running", False):
                time.sleep(idle_time)
                experiment_document = list(self.fl_db.experiment.find({"_id": exp_id}))[0]

    def reset_experiment(self, experiment_id):
        with self.db_session.start_transaction():

            experiment_documents = list(self.fl_db.experiment.find({"_id": experiment_id}).limit(1))
            if len(experiment_documents) == 0:
                logging.info(f"The experiment you are trying to reset does not exist. {experiment_id}")
                return False
            if experiment_documents[0].get('is_finished', False):
                logging.info(f"You cannot reset a finished experiment {experiment_id}")
                return False
            _, parameters = utils.get_model(db=self.fl_db, model_id=experiment_documents[0]['start_model_id'])

            experiment_state_document, _ = utils.get_model(db=self.fl_db,
                                                           model_id=experiment_documents[0][
                                                               'experiment_state_model_id'])

            utils.save_model_parameters(model_id=experiment_state_document['_id'],
                                        model_document=experiment_state_document, parameters=parameters,
                                        overwrite=True)

            self.__reset_task_list(experiment_document=experiment_documents[0])

            result = self.fl_db.experiment.update_one({"_id": experiment_id},
                                                      {"$set": {"is_running": False, "is_finished": False,
                                                                "has_failed": False}}, session=self.db_session)
        return True

    def set_as_final_model(self, model_id=None, experiment_id=None):
        """
        either experiment_id or model_id must be set
        """
        if experiment_id:
            experiment_documents = list(self.fl_db.experiment.find({"_id": experiment_id}).limit(1))
            if len(experiment_documents) == 0:
                logging.info(f"The experiment you are trying to set as final model does not exist. {experiment_id}")
                return False
            model_id = experiment_documents[0]['experiment_state_model_id']

        logging.info(f"setting {model_id} as final model")
        task_update = self.fl_db.model.update_one(
            {"_id": model_id},
            {"$set": {f"is_final_model": True}})

    def objective(self, trial, setup_dict, tunable_parameters, trial_loss_function):
        """
        Does a hyperparameter optimisation with the tunable parameters given. Runs an experiment for each constellation

        Inputformat of tunable_parameters:
        [
            {
                'parameter_key_list': list of keys to navigate through the setup_dict (ex. ['model_function','parameters','lr']
                'function_name': function to pick the parameter value (ex. 'suggest_uniform', 'suggest_categorical' ) https://optuna.readthedocs.io/en/stable/reference/trial.html
                'function_arguments': arguments of the above function 'function_name'
            },
        ...
        ]
        """

        logging.info(f"TEST WITH {json.dumps(tunable_parameters)}")

        # replace the parameters with the tuneable parameters
        tuning = []
        for tunable_parameter_config in tunable_parameters:
            parameter_function = getattr(trial, tunable_parameter_config['function_name'])
            if len(tunable_parameter_config['parameter_key_list']) == 1:
                setup_dict[tunable_parameter_config['parameter_key_list'][0]] = parameter_function(
                    **tunable_parameter_config['function_arguments'])
                tuning.append(setup_dict[tunable_parameter_config['parameter_key_list'][0]])
            else:  # navigate through the dict with pointers
                sub_dict_pointer = setup_dict.setdefault(tunable_parameter_config['parameter_key_list'][0], {})
                for key in tunable_parameter_config['parameter_key_list'][1:-1]:
                    sub_dict_pointer = sub_dict_pointer.setdefault(key, {})
                sub_dict_pointer[tunable_parameter_config['parameter_key_list'][-1]] = \
                    parameter_function(**tunable_parameter_config['function_arguments'])
                tuning.append(sub_dict_pointer[tunable_parameter_config['parameter_key_list'][-1]])
        experiment_id, metrics = self.define_and_start_experiment(setup_dict, additional_description='')

        return trial_loss_function(metrics)

    def get_compiled_model(self, protocol, experiment_id=None, model_id=None):
        load_model = getattr(self, f"{protocol}_load_model")
        model_parameters = utils._get_model_parameters(self.fl_db, experiment_id=experiment_id,
                                                       model_id=model_id)

        model = load_model(model_parameters=model_parameters['parameters'],
                           model_definition=json.dumps(model_parameters['model']))
        return model

    def NN_aggregate(self, experiment_id, global_model_id, send_model_task_id, clients, aggregation_config, round_num):
        clients_model_updates = utils.load_clients_model_updates(experiment_id=experiment_id,
                                                                 send_model_task_id=send_model_task_id,
                                                                 clients=clients)

        global_model_document, global_model_parameters = utils.get_model(db=self.fl_db,
                                                                         model_id=global_model_id)

        update_rate = aggregation_config.get("update_rate", 1)
        update_decay = aggregation_config.get("update_decay", 0)
        update_rate = update_rate - round_num * update_decay / aggregation_config.get("total_rounds", 1)  # todo better
        logging.info(update_rate)
        updated_model_parameters = utils.NN_aggregate_model_updates(clients_model_updates=clients_model_updates,
                                                                    global_model_parameters=global_model_parameters,
                                                                    update_rate=update_rate,
                                                                    verbose=aggregation_config.get("verbose",
                                                                                                   0),
                                                                    aggregation_type=aggregation_config.get(
                                                                        "aggregation_type", 'mean'))

        utils.save_model_parameters(model_id=global_model_document['_id'], model_document=global_model_document,
                                    parameters=updated_model_parameters, send_model_task_id=send_model_task_id,
                                    overwrite=True)

    def P2P_aggregate(self, experiment_id, global_model_id, send_model_task_id, clients, aggregation_config, round_num):

        # in case of P2P, only one client at a time should be passed to method below
        # work with clients field in task collection
        task_dict = list(self.fl_db.task.find({"_id": send_model_task_id}))[0]
        clients = list(task_dict['clients'].keys())
        clients_model_updates = utils.load_clients_model_updates(experiment_id=experiment_id,
                                                                 send_model_task_id=send_model_task_id,
                                                                 clients=clients)

        # here we retrieve dict object, with fields pickle and trees
        clients_model_updates = clients_model_updates[clients[0]]

        logging.info("NUMBER OF TREES: {}. Last tree built on client {}".format(len
                                                                                (eval(clients_model_updates["trees"])),
                                                                                clients[0]))

        global_model_document, _ = utils.get_model(db=self.fl_db, model_id=global_model_id)

        # in P2P global model is just overwritten by the model sent from the client
        utils.save_model_parameters(model_id=global_model_document['_id'], model_document=global_model_document,
                                    parameters=clients_model_updates, send_model_task_id=send_model_task_id,
                                    overwrite=True)

    def RF_aggregate(self, experiment_id, global_model_id, send_model_task_id, clients, aggregation_config, round_num):
        """

        """
        # get updates
        clients_model_updates = utils.load_clients_model_updates(experiment_id=experiment_id,
                                                                 send_model_task_id=send_model_task_id,
                                                                 clients=clients)
        # per worker, we have received one dictionary, indexable by the model-feature-indices (as string)
        # each feature has a list associated with it, corresponding to a list of bins (dicts)
        # get current model and parameters (forest with one tree)
        global_model_document, global_model_parameters = utils.get_model(db=self.fl_db,
                                                                         model_id=global_model_id)

        # aggregate all histograms (dictionaries) per feature together
        # find optimal split in given information
        # update model parameters by inserting a new tree-node into the json-tree
        # Also done: Update the current_condition_list under global_model_document
        updated_model_document, updated_model_parameters, finished = utils.RF_aggregate_model_updates(
            clients_model_updates=clients_model_updates,
            global_model_document=global_model_document,
            global_model_parameters=global_model_parameters,
            clients=clients)

        # save new model parameters by writing down the json file
        utils.save_model_parameters(model_id=updated_model_document['_id'], model_document=updated_model_document,
                                    parameters=updated_model_parameters, send_model_task_id=send_model_task_id,
                                    overwrite=True)
        updated_model_document['_id'] = ObjectId(updated_model_document['_id'])
        result = self.fl_db.model.replace_one({"_id": updated_model_document['_id']}, updated_model_document)
        # if the tree is finished, mark all tasks to this tree as finished
        if finished:

            task_list_update = self.fl_db.experiment.update_many({"_id": ObjectId(experiment_id)},
                                                                 {"$set": {"task_list.$[].task_status": config[
                                                                     'TASK_DONE']}})

            logging.info("Aggregated and finished a tree%s", experiment_id)
        else:
            logging.info("Aggregated histograms%s", experiment_id)

    # todo these are duplicates
    def NN_load_model(self, model_definition, model_parameters):
        config = json.loads(model_definition)
        if config["training"].get("differential_privacy", {}).get("method", 'before') not in ['after', 'before']:
            raise Exception("Bad differential privacy method set")
        model = tf.keras.models.model_from_json(json.dumps(config['model']))
        model.compile(loss=tf.losses.get(config['compile']['loss']),
                      optimizer=utils.get_NN_optimizer(config),
                      metrics=[getattr(utils.import_from_string(metric['module']),
                                       metric['class_name']).from_config(metric["config"]) for
                               metric in config['compile']['metrics']],
                      loss_weights=config['compile'].get('loss_weights', None),
                      sample_weight_mode=config['compile'].get('sample_weight_mode', None),
                      weighted_metrics=config['compile'].get('weighted_metrics', None),
                      target_tensors=config['compile'].get('target_tensors', None)
                      )

        global_weights = utils.array_from_bytes(model_parameters)
        model = utils.set_weights(model, global_weights,
                                  normalize=config['compile'].get("normalize", 0), )

        return model

    def RF_load_model(self, model_definition, model_parameters):
        config = json.loads(model_definition)
        model = RandomForest.RandomForestClassifier.from_json(config['model'])
        dict_forest = model_parameters
        print(dict_forest)
        for tree in dict_forest['forest']:
            model.forest.append(DecisionTree.DecisionTreeClassifier.from_json(tree))

        return model

    # todo, the model_rparam is a dict but in tzhe node its a json
    def P2P_load_model(self, model_definition, model_parameters):
        config = json.loads(model_definition)
        global_model = pickle.loads(eval(model_parameters['pickle']))
        model = global_model  # Booster object

        return model

    def _running_experiment_upkeep(self, current_task, experiment_id, experiment_document, idle_time, timeout):
        if self._experiment_stop_condition(current_task, experiment_id, experiment_document, timeout):
            experiment_document['is_running'] = False
        else:
            time.sleep(idle_time)
            experiment_document = self.fl_db.experiment.find_one({"_id": experiment_id})
        return experiment_document

    def _experiment_stop_condition(self, current_task, experiment_id, experiment_document, timeout):
        if current_task['task_status'] == config['TASK_DONE']:
            logging.info(f"Experiment {experiment_id} finished")
            return True
        elif timeout and (datetime.datetime.now() - experiment_document['experiment_start_time']) > timeout:
            logging.info(f"Experiment {experiment_id} timed out.")
            test = self.fl_db.experiment.update_one({"_id": experiment_id}, {"$set": {"has_failed": True}})
            return True
        return False

    def _save_aggregated_metrics(self, experiment_document, stop_function):
        aggregated_metric = utils.aggregate_loss(experiment_document.get('test_results', None))
        additional_metrics = []
        if stop_function:
            logging.info("Please make sure that the stop function is described in the experiment")
            stop_experiment, additional_metrics = stop_function(self, experiment_document, aggregated_metric)
        metrics = {
            'aggregated_metric': aggregated_metric,
            'additional_metrics': additional_metrics}
        task_list_update = self.fl_db.experiment.update({"_id": ObjectId(experiment_document['_id'])},
                                                        {"$set": {
                                                            f"test_results.final.aggregated": metrics}})
        return metrics

    def _finish_up_experiment(self, experiment_id, experiment_document, current_task, metrics, idle_time):
        with self.db_session.start_transaction():  # finishup experiment
            if current_task['task_status'] == config['TASK_DONE']:
                test = self.fl_db.experiment.update_one({"_id": experiment_id}, {"$set": {"is_finished": True}},
                                                        session=self.db_session)
            self.fl_db.model.update_one({"_id": ObjectId(experiment_document['experiment_state_model_id'])},
                                        {"$set": {"is_running": False}}, session=self.db_session)

        while experiment_document.get("is_running", False):
            time.sleep(idle_time)
            experiment_document = self.fl_db.experiment.find_one({"_id": experiment_id})
        logging.info("Experiment finished!!")

        time.sleep(3)
        return metrics

    def _aggregation_step_upkeep(self, experiment_document, current_task, latest_send_task, stop_function,
                                 upkeep_function):
        aggregated_metric = utils.aggregate_loss(experiment_document.get('validation_results', None))
        additional_metrics = self._early_experiment_stop(experiment_document, aggregated_metric, stop_function)

        if upkeep_function:
            logging.info("Please make sure that the upkeep function is described in the experiment")
            upkeep_function(self, experiment_document)

        task_list_update = self.fl_db.experiment.update({"_id": ObjectId(experiment_document['_id'])},
                                                        {"$set": {
                                                            f"validation_results.{latest_send_task['task_order'] + 2}": {
                                                                'aggregated_metric': aggregated_metric,
                                                                'additional_metrics': additional_metrics}}})

        utils.task_completion(self.fl_db, current_task["task_id"], experiment_document['_id'],
                              config['AGGREGATOR_NAME'])
        return {'aggregated_metric': aggregated_metric, 'additional_metrics': additional_metrics}

    def _early_experiment_stop(self, experiment_document, aggregated_metric, stop_function):
        additional_metrics = []
        if stop_function:
            logging.info("Please make sure that the stop function is described in the experiment")
            stop_experiment, additional_metrics = stop_function(self, experiment_document, aggregated_metric)
            if stop_experiment:  # set all bat last two tasks (fetch+send_test_loss) to done
                task_list_update = self.fl_db.experiment.update_many(
                    {"_id": ObjectId(experiment_document['_id'])},
                    {"$set": {"task_list.$[i].task_status": config['TASK_DONE']}},
                    array_filters=[{"i.task_order": {"$lt": (len(experiment_document['task_list']) - 3)}}])
        return additional_metrics

    def __build_RF_trees(self, experiment_id_list, operator_list, idle_time):
        process_list = []

        for i in range(len(experiment_id_list)):
            p = Process(target=Operator._start_experiment_RF_helper,
                        args=([experiment_id_list[i], operator_list[i], idle_time]))
            p.start()
            process_list.append(p)

            if i % 8 == 7:
                for process in process_list:
                    process.join()
                process_list = []
                time.sleep(idle_time * 5)

        for process in process_list:
            process.join()
        time.sleep(idle_time * 5)

    def __RF_add_trees_to_random_forest(self, experiment_id, experiment_id_list):
        experiment_document = list(self.fl_db.experiment.find({"_id": experiment_id}))[0]
        global_model_id = experiment_document['experiment_state_model_id']
        global_model_document, global_model_parameters = utils.get_model(db=self.fl_db, model_id=global_model_id)
        global_model_parameters['forest'] = []
        # update the global_model_parameters by grouping all trees together in the list
        for exp_id in experiment_id_list:
            exp_doc = list(self.fl_db.experiment.find({"_id": exp_id}))[0]
            exp_model_id = exp_doc['experiment_state_model_id']
            _, exp_model_params = utils.get_model(db=self.fl_db, model_id=exp_model_id)
            global_model_parameters['forest'].append(exp_model_params["forest"][0])
        # save global forest
        utils.save_model_parameters(model_id=global_model_document['_id'], model_document=global_model_document,
                                    parameters=global_model_parameters, send_model_task_id=None,
                                    overwrite=True)

    @staticmethod
    def _start_experiment_RF_helper(experiment_id, operator_i, idle_time=5):
        """Function is used to spawn processes that actually start the processes for the random forest
        procedure. This function is mapped onto a process-pool.
        ::
        Note: This function should not be called!
        """
        operator_i.start_experiment(experiment_id, idle_time)
        return True

    def __define_task_list(self, experiment_id, tasks, clients, testing, protocol):
        task_list = []
        for task_order, task in enumerate(tasks):
            if protocol == 'P2P':  # in P2P protocol tasks are represented as tuples
                task, clients = task[0], [task[1]]
            if task in config['VALID_TASKS']:
                task_id = self.__define_task(experiment_id=experiment_id, task=task, task_order=task_order,
                                             clients=clients, testing=testing)
            elif task in config['VALID_SERVER_TASKS']:
                task_id = self.__define_task(experiment_id=experiment_id, task=task, task_order=task_order,
                                             clients=[config['AGGREGATOR_NAME']], testing=testing)
            else:
                logging.info(f"Bad Task given!!!!!!!!!!! Ignored {task}")
                break
            task_list.append(utils.build_task_list_document(task_id=task_id, task_order=task_order, task_name=task,
                                                            task_status=config['NOT_SCHEDULED_STOPWORD']))
        return task_list

    def __define_task(self, experiment_id, task, task_order, clients, testing):
        task_document = utils.build_task_document(experiment_id=experiment_id, task_name=task, task_order=task_order,
                                                  clients=clients,
                                                  testing=testing)
        result = self.fl_db.task.insert_one(task_document)
        task_id = result.inserted_id
        return task_id

    def __reset_task_list(self, experiment_document):
        for task in experiment_document["task_list"]:
            if task['task_name'] in config['VALID_TASKS']:
                clients = experiment_document['clients']
            if task['task_name'] in config['VALID_SERVER_TASKS']:
                clients = ['aggregator']
            task_update = self.fl_db.task.update_one(
                {"_id": task['task_id']},
                {"$set": {f"clients.{client}.status": "not_scheduled" for client in clients}},
                session=self.db_session)
            task_list_update = self.fl_db.experiment.update_one(
                {"_id": experiment_document['_id'], "task_list.task_id": task['task_id']},
                {"$set": {f"task_list.$.task_status": config['NOT_SCHEDULED_STOPWORD']}}, session=self.db_session)

    def powerset(self, iterable):

        from itertools import chain, combinations

        """
        Returns a generator that returns the subsets of iterable as lists.
        Ensures that the resulting subsets are sorted.
        :param iterable: an iterable object
        :return: generator for which returns subsets of the iterator's elements as lists.
        """
        s = sorted(list(iterable))
        for subset in chain.from_iterable(combinations(s, r) for r in range(len(s) + 1)):
            yield list(subset)

    def shapley_values(self, clients, setup_dict, metrics,empty_set=1):
        # todo zero score
        import bisect
        from scipy.special import comb
        logging.info("Computing values...")
        res = {client: {metric:0 for metric,score_type in metrics} for client in clients}
        combination_cache = {}
        for client in clients:
            remaining = [c for c in clients if c != client]
            client_combinations = self.powerset(remaining)
            for client_combination in client_combinations:
                # As list is not hashable we need to make them to tuples.
                key_s = tuple(client_combination)

                if key_s not in combination_cache:
                    combination_cache[key_s] = self._compute_players(client_combination, setup_dict,empty_set)

                bisect.insort(client_combination, client)
                key_si = tuple(client_combination)

                if key_si not in combination_cache:
                    combination_cache[key_si] = self._compute_players(client_combination, setup_dict,empty_set)
                for metric, score_type in metrics:
                    if score_type == "high":
                        res[client][metric] += (combination_cache[key_si][metric] - combination_cache[key_s][metric]) / comb(
                            len(clients) - 1, len(client_combination) - 1)
                    elif score_type == "low":
                        res[client][metric] += (combination_cache[key_s][metric] - combination_cache[key_si][metric]) / comb(
                            len(clients) - 1, len(client_combination) - 1)
        print(combination_cache)
        print(res)

        setup_dict['rounds']=0

        setup_dict['final_round']=['aggregate']
        setup_dict['clients']=clients
        setup_dict['experiment_name']='shapley'
        experiment_id = self.setup_up_experiment_by_dict(setup_dict, additional_description='shapley')
        result=   {
        "2" :
            {client: json.dumps({metric: value/len(clients) for metric, value in metrics.items()}) for client, metrics in res.items()}
           }
        self.fl_db.experiment.update_one({"_id": experiment_id}, { "$set": {"test_results": result}})
        return {client: {metric: value/len(clients) for metric,value in metrics.items()} for client, metrics in res.items()}

    def _compute_players(self, players, setup_dict, empty_set=1):
        setup_dict['clients'] = players
        setup_dict['experiment_name'] = str(players)

        # Acquire losses
        # TODO Find out how to deal with multiple metrics if needed

        if len(players) == 0:
            return empty_set

        experiment_id, metrics = self.define_and_start_experiment(setup_dict)
        return metrics['aggregated_metric']
        #
        # losses = np.transpose(np.array(self.get_clients_loss(players)))[1]
        #
        # return hf.harmonic_mean(losses)
