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
import tensorflow as tf
import xgboost as xgb
import pandas as pd
import logging
from globalserver.operator_.operator_class_db import Operator
from testing.test_class import Testing
from globalserver.operator_.utils import operator_utils
import json

tf.keras.backend.clear_session()

# Define the Clients name (these are the same as you used in startup.sh script)

# define xgboost model


def get_compiled_model_P2P(param_dict):

    model = xgb.Booster(param_dict['params'], [param_dict['example']])
    return model

with open('datasets/test_c0.jsonl') as f:
    line = list(json.loads(f.readline()).values())

X_example = [line[:-1], line[:-1]]
y_example = [line[-1], line[-1]]
example = xgb.DMatrix(X_example, label=y_example)
param_dict={'params':{'max_depth': 8, 'subsample': 0.5, 'eta': 0.01, 'max_delta_step': 0,
                        'scale_pos_weight': 1.5, 'objective': 'binary:logitraw',
                        'tree_method': 'hist', 'max_bin': 250, 'colsample_bytree': 1},'example':example}


def preprocessing(batch):
    batch['label'] = [int(value) for value in batch['label']]
    for key in batch:
        batch[key] = [0 if value == '(not_set)' else float(value) for value in batch[key]]

    return batch


clients = ['c0', 'c1']

# here we specify rounds a bit differently, since clients need to perform tasks one after another
round = []
# for i in range(len(clients)):
for client in clients:
    round.extend([("fetch_model", client), ("train_model", client), ("send_model", client),
                  ("send_validation_loss", client), ("aggregate", 0)])



TestSetup = Testing(clients, start_servers=True, clear_logs=True, clear_db=False, interface=False)

# train, test = get_data_kdd()
# split_half(train, test, clients)

setup_dict = {"model_function": {
    "function": get_compiled_model_P2P,
    "parameters": param_dict,
    },
        "git_version": 'e9339081b76ad3a89b1862bd38d8af26f0541f1c',
        "protocol": 'P2P',
        "model_name": "test_model",
        "model_description": "this model is just to test the db",
        "testing": True,
        "training_config": {
        "verbosity": 1,
        "epochs": 1,
        "batch_size": 10000,
        "nthread": -1,
        "client_steps_per_round": 1,
        "nr_clients": len(clients),
        "skmetrics": ["f1_score","accuracy_score"],
        "tfmetrics": ["AUC", "Accuracy"],

        "dataset": "",

    },
    "rounds": 2,
    "round":round,
    "final_round":[],
    "clients": clients,
    "experiment_name": "kkbox",
    "experiment_description": f"desc if nice experiment",
    "preprocessing": {
    "noise": {
        "epsilon": 1,
        "delta": 0.2
    },
    "cast_to_float": "",
    "feature_selection": [f"column{i}" for i in range(5,50)],
    "preprocessing_function": preprocessing
}
}


from globalserver.operator_.operator_class_db import Operator

operator = Operator()
operator.define_and_start_experiment(setup_dict)
