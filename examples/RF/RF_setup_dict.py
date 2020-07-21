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
"""
The Python implementation of the MTC Federated Learning Example Operator.

This is an example how to use the Operator Class for Random Forests.

Look at operator_class.py for all callable functions.

The example creates a model and creates a Random Forest according to
the given specifications.
"""
import subprocess
import os

import sys

sys.path.append(os.getcwd())
os.environ['STATIC_VARIABLES_FILE_PATH'] = "globalserver/static_variables.json"

os.environ['PATH_TO_GLOBALSERVER'] = "globalserver/api/"
from globalserver.operator_.operator_class_db import Operator
from testing.test_class import Testing
import time
import json
from globalserver.operator_.utils import operator_utils

# Define the Clients name (these are the same as you used in startup.sh script)
clients = ['c1', 'c2']
# Setting up all the components
operator = Operator()
############!!!! Start your second operator with start_servers=False. To clear logs start with clear_logs=True.
# To clear test entries in db and folderstructure clear_db=False.
# ignore the interface=False it would be to test the client interface
TestSetup = Testing(clients, start_servers=True, clear_logs=True, clear_db=False, interface=False)

"""
NOTE: One has to only change the specification-dict and the training_config-dict in order to change
the random forest.
"""

# specification for the random forest
specification = {
    'n_features': 39,  # for given test data
    'feature_information': {  # can list up categorical features and initialize them with False
        'col0': False,
    },
    'n_estimators': 2,  # number of trees in the final forest
    'max_depth': 5,  # maximum depth parameter
    'max_features': "sqrt",  # maximum features to consider at each node for splitting
    'max_bins': 20,  # maximum number of bins to use for continuous data in the histograms
    'pos_label': 1,  # positive label in data
    'neg_label': 0,  # negative label in data
    'minimal_information_gain': 0.0,  # minimal information gain to not add a leaf
    'metrics': ['log_loss'],
    "preprocessing": {
        "noise": {
            "epsilon": 1,
            "delta": 0.2
        }
    }
}

get_compiled_model_RF = operator_utils.RF_get_compiled_model

setup_dict = {"model_function": {
    "function": get_compiled_model_RF,
    "parameters": specification,
},
    "protocol": 'RF',
    "model_name": "test_model",
    "model_description": "this model is just to test the db",
    "testing": True,
    "training_config": {"differential_privacy": "data",  # can be "data" or "hist"
                        "diff_p_epsilon": 10.0,
                        "diff_p_delta": 0.9,
                        "steps_per_epoch": 1,
                        "batch_size": 10000,
                        "dataset": "",
                        "skmetrics": ["f1_score", "accuracy_score"],
                        "tfmetrics": ["AUC", "Accuracy"]},
    "rounds": pow(2, (specification['max_depth'] + 1)) - 1,
    "round": ["fetch_model", "train_model", "send_model", "aggregate"],
    "final_round": [],
    "clients": clients,
    "experiment_name": "kkbox",
    "experiment_description": f"desc if nice experiment",
    "preprocessing": {
        "noise": {
            "epsilon": 1,
            "delta": 0.2
        }
    }
}
# setup all experient_id's, each corresponding to building one tree of the forest
experiment_id_list = []
operator_list = []
for _ in range(specification['n_estimators']):
    # it's important that each tree is specified individually!
    # model_dict, empty_forest_dict = operator_utils.RF_get_compiled_model(specification)
    operator_i = Operator()
    operator_list.append(operator_i)
    experiment_id = operator.setup_up_experiment_by_dict(setup_dict, additional_description='')
    experiment_id_list.append(experiment_id)

# initialize same for overhead process
setup_dict['rounds']=0
setup_dict['final_round']=["fetch_model", "send_training_loss", "send_validation_loss", "send_test_loss"]
experiment_id = operator.setup_up_experiment_by_dict(setup_dict, additional_description='')

"""
Overhead process:
- constructs the whole random forest
- after construction is finished, the losses from all agents get acquired and stored in the database
"""
# start experiment
operator.start_experiments_RF(
    experiment_id=experiment_id,
    experiment_id_list=experiment_id_list,
    operator_list=operator_list,
    idle_time=1,
    n_workers=1
)

