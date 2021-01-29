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
The Python implementation of the MTC Federated Learning Example Operator.

This is an example how to use the Operator Class.

Look at operator_class_db.py for all callable functions.

The example creates a model and runs 10 rounds of send_model_to_nodes->training_on_nodes->
send_back_to_server->aggregate_gradients_to_get_new_global_models

"""
import os
import sys
sys.path.append(os.getcwd())
os.environ['STATIC_VARIABLES_FILE_PATH'] = "globalserver/static_variables.json"
os.environ['PATH_TO_GLOBALSERVER'] = "globalserver/api/"
from globalserver.operator_.operator_class_db import Operator
from testing.test_class import Testing

import tensorflow as tf
import xgboost as xgb
import pandas as pd
import logging

from globalserver.operator_.utils import operator_utils
import json

tf.keras.backend.clear_session()

# Define the Clients name (these are the same as you used in startup.sh script)
clients = ['c0', 'c1']
# clients = ['c1']

# Define model use (NN or P2P):
model_type = 'P2P'
assert model_type in ["NN", "P2P"], "Model type unknown!"

logging.info("Starting Pipeline...")
operator = Operator()
TestSetup = Testing(clients, start_servers=True, clear_logs=True, clear_db=False, interface=False)
# ignore the interface=False it would be to test the client interface
logging.info("Pipeline started. If you only have to do this (TestSetup = Testing(...) once. ")
logging.info("The servers will run until they fail. To stop them run TestSetup.kill_servers().")
logging.info("To clear entries with the 'testing' flag run TestSetup.clean_db().")
logging.info("To clear the logs run TestSetup.clean_logs().")

# define xgboost model
xgboost_model_params = {'max_depth': 8, 'subsample': 0.5, 'eta': 0.01, 'max_delta_step': 0,
                        'scale_pos_weight': 1.5, 'objective': 'binary:logitraw',
                        'tree_method': 'hist', 'max_bin': 250, 'colsample_bytree': 1}
metrics = ['accuracy', 'f1score', 'log_loss']


def get_compiled_model_P2P(params, example):
    """

    Args:
        params: training parameters for xgboost, for list of params see
        https://xgboost.readthedocs.io/en/latest/parameter.html
        example: DMatrix object, contains one datapoint with all feature names

    Returns: Booster object (training model object in xgboost)

    """

    model = xgb.Booster(params, [example])
    return model


def split_half():
    import random
    with open('datasets/train_kkbox.jsonl') as f:
        train = f.readlines()

    with open('datasets/test_kkbox.jsonl') as f:
        test = f.readlines()
    random.shuffle(train)
    train_c1 = train[:70000]
    train_c2 = train[70000:140000]
    validation_c1 = train[140000:160000]
    validation_c2 = train[160000:180000]

    with open(f'datasets/train_{clients[0]}.jsonl', 'w+') as f:
        f.writelines(train_c1)
    with open(f'datasets/validation_{clients[0]}.jsonl', 'w+') as f:
        f.writelines(validation_c1)
    with open(f'datasets/test_{clients[0]}.jsonl', 'w+') as f:
        f.writelines(test)

    with open(f'datasets/train_{clients[1]}.jsonl', 'w+') as f:
        f.writelines(train_c2)
    with open(f'datasets/validation_{clients[1]}.jsonl', 'w+') as f:
        f.writelines(validation_c2)
    with open(f'datasets/test_{clients[1]}.jsonl', 'w+') as f:
        f.writelines(test)


# split_half()
with open('datasets/test_c0.jsonl') as f:
    line = list(json.loads(f.readline()).values())

X_example = [line[:-1], line[:-1]]
y_example = [line[-1], line[-1]]
example = xgb.DMatrix(X_example, label=y_example)
model = get_compiled_model_P2P(xgboost_model_params, example)
model_parameters = operator_utils.get_P2P_params(model)
model = operator_utils.jsonify_model_definition(model=None, protocol=model_type,
                                                meta_model_params=xgboost_model_params,
                                                metrics=metrics)


def preprocessing(batch):
    # batch = batch[:, -4:]
    batch = batch.astype('float64')
    for row_i, row in enumerate(batch):
        for cell_i, cell in enumerate(row):
            batch[row_i, cell_i] = float(cell)

    return batch

import dill

model["preprocessing"] = {
    "noise": {
        "epsilon": 1,
        "delta": 0.2
    },
    "cast_to_float": "",
    "feature_selection": [f"column{i}" for i in range(5,50)],
    "preprocessing_function": dill.dumps(preprocessing).decode('latin_1')
}
# define model
git_version = 'e9339081b76ad3a89b1862bd38d8af26f0541f1c'
model_id = operator.define_model(model=model, parameters=model_parameters, protocol=model_type, model_name="test_model",
                                 model_description="this model is just to test the db",
                                 git_version=git_version, is_running=False, testing=True)

logging.info("Base Model created. You can look at the entry in the DB.")
logging.info("This is a test model which will be cleared when you call TestSetup.clean_db(). ")
logging.info("To run a persistent model/experiment set  testing=False")

training_config = {
    "verbosity": 1,
    "epochs": 1,
    "batch_size": 10000,
    "nthread": -1,
    "client_steps_per_round": 1,
    "nr_clients": len(clients),
    "skmetrics": ["f1_score","accuracy_score"],
    "tfmetrics": ["AUC", "Accuracy"],

}

# here we specify rounds a bit differently, since clients need to perform tasks one after another
round = []
# for i in range(len(clients)):
for client in clients:
    round.extend([("fetch_model", client), ("train_model", client), ("send_model", client),
                  ("send_validation_loss", client), ("aggregate", 0)])

tasks = []
NR_ROUNDS = 2  # in P2P nr. of trees equals nr. of clients * NR_ROUNDS
for i in range(NR_ROUNDS):
    tasks.extend(round)

experiment_id = operator.define_experiment(start_model_id=model_id, training_config=training_config,
                                           tasks=tasks, clients=clients, git_version=git_version,
                                           experiment_name="test_experiment",
                                           experiment_description="this is a test experiment", testing=True)

logging.info("Experiment created. A new Model representing the experiment state was created. ")
logging.info("You can look at the entry in the DB.")
logging.info("Note that all the parameters of the model are stored in globalserver/db/global_models/<model_id>")
# start experiment
logging.info("We start the experiment...")
logging.info(
    "!!!!!!!!!!! You have to add the data sets in the folder datasets in the format of json lines (keynames are ignored).")
logging.info("!!!!!!!!!!! you need train, validation and test data for each client.")
logging.info("example: train_c1.jsonl,train_c2.jsonl,validation_c1.jsonl, test_c1.jsonl,...")
logging.info("To see whats going on on the server, look into the logs of the servers in testing/testing.log.")
logging.info("When you do not need the servers anymore run TestSetup.kill_servers() in the script.")
operator.start_experiment(experiment_id)
