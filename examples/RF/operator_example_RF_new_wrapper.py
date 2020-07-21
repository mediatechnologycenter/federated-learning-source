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


# Setting up all the components
operator = Operator()
############!!!! Start your second operator with start_servers=False. To clear logs start with clear_logs=True.
# To clear test entries in db and folderstructure clear_db=False.
# ignore the interface=False it would be to test the client interface
TestSetup = Testing(clients, start_servers=True, clear_logs=True, clear_db=False, interface=False)
clients = ['c1', 'c2']
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

git_version = 'e9339081b76ad3a89b1862bd38d8af26f0541f1c'

# define new experiment
training_config = {
    "differential_privacy": "data",  # can be "data" or "hist"
    "diff_p_epsilon": 10.0,
    "diff_p_delta": 0.9,
    "steps_per_epoch": 1,
    "batch_size": 1000,
    "dataset": "",
    "skmetrics": ["f1_score","accuracy_score"],
    "tfmetrics": ["AUC", "Accuracy"],

}
# differential_privacy noise is either added directly to the data after reading it ("data" parameter)
# or noise is added after histograms have been generated ("hist" parameter)

# need to have one task-round per tree-node, thus need ``2^(max_depth+1) - 1`` many rounds per tree
node_round = ["fetch_model", "train_model", "send_model", "aggregate"]
tasks = []
max_nodes = pow(2, (specification['max_depth'] + 1)) - 1
for i in range(max_nodes):
    tasks.extend(node_round)

# setup all experient_id's, each corresponding to building one tree of the forest
experiment_id_list = []
operator_list = []
for _ in range(specification['n_estimators']):
    # it's important that each tree is specified individually!
    model_dict, empty_forest_dict = operator_utils.RF_get_compiled_model(specification)

    operator_i = Operator()
    operator_list.append(operator_i)

    model_id = operator_i.define_model(model=model_dict, parameters=empty_forest_dict, protocol='RF',
                                       model_name="test_model",
                                       model_description="this model is just to test the db",
                                       git_version=git_version, is_running=False, testing=True)

    experiment_id = operator_i.define_experiment(start_model_id=model_id, training_config=training_config,
                                                 tasks=tasks, clients=clients, git_version=git_version,
                                                 experiment_name="test_experiment",
                                                 experiment_description="this is a test experiment", testing=True)

    experiment_id_list.append(experiment_id)

# from bson import ObjectId
# experiment_id_list = [ObjectId("5e7cbc3c5030dab8429160c4"),ObjectId("5e7cbc3c5030dab842916084")]
# initialize same for overhead process
model_dict, empty_forest_dict = operator_utils.RF_get_compiled_model(specification)
# update empty_forest_dict to contain just a list (i.e. no empty tree)
# when all trees are built, the forest will be aggregated into this list
empty_forest_dict['forest'] = []
model_id = operator.define_model(model=model_dict, parameters=empty_forest_dict, protocol='RF', model_name="test_model",
                                 model_description="this model is just to test the db",
                                 git_version=git_version, is_running=False, testing=True)

overhead_tasks = ["fetch_model", "send_training_loss", "send_validation_loss", "send_test_loss"]

experiment_id = operator.define_experiment(start_model_id=model_id, training_config=training_config,
                                           tasks=overhead_tasks, clients=clients, git_version=git_version,
                                           experiment_name="test_experiment",
                                           experiment_description="this is a test experiment", testing=True)

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



############!!!! call this to kill all servers when no more needed
# TestSetup.kill_servers()
####### Call this to purge the db/folders from all testing experiment/models/tasks...
# operator.clean_testing_experiments()
