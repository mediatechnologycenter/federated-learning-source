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

This is an example how to use the Operator Class.

Look at operator_class_db.py for all callable functions.

The example creates a model and runs 10 rounds of send_model_to_nodes->training_on_nodes->
send_back_to_server->aggregate_gradients_to_get_new_global_models

"""
import os

os.environ['STATIC_VARIABLES_FILE_PATH'] = "globalserver/static_variables.json"

os.environ['PATH_TO_GLOBALSERVER'] = "globalserver/api/"
import tensorflow as tf
import logging
from globalserver.operator_.operator_class_db import Operator
from testing.test_class import Testing
from globalserver.operator_.utils import operator_utils

tf.keras.backend.clear_session()

# Define the Clients name (these are the same as you used in startup.sh script)

clients = ['c1']
logging.info("Starting Pipeline...")
operator = Operator()
TestSetup = Testing(clients, start_servers=True, clear_logs=True, clear_db=False, interface=False)
# ignore the interface=False it would be to test the client interface
logging.info("Pipeline started. If you only have to do this (TestSetup = Testing(...) once. ")
logging.info("The servers will run until they fail. To stop them run TestSetup.kill_servers().")
logging.info("To clear entries with the 'testing' flag run TestSetup.clean_db().")
logging.info("To clear the logs run TestSetup.clean_logs().")


# define NN model
def get_compiled_model(input_dim=61):
    # First Layer needs input_dim!
    # otherwise define whatever you want.
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(72, activation=tf.nn.relu, input_dim=input_dim, name='first'),
        tf.keras.layers.Dense(2, activation=tf.nn.softmax, name='last')
    ])

    # Set any compiler you want
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss='sparse_categorical_crossentropy',
                  metrics=[])
    return model


model = get_compiled_model(61)

logging.info("Turing the compiled Keras model into a json.")
# jsonify model
model_parameters = operator_utils.get_weights(model)
# returns a 2d-array with all NN weights
model = operator_utils.jsonify_model_definition(model=model, protocol='NN')

# define model
git_version = 'e9339081b76ad3a89b1862bd38d8af26f0541f1c'
model_id = operator.define_model(model=model, parameters=model_parameters, protocol='NN', model_name="test_model",
                                 model_description="this model is just to test the db",
                                 git_version=git_version, is_running=False, testing=True)

logging.info("Base Model created. You can look at the entry in the DB.")
logging.info("This is a test model which will be cleared when you call TestSetup.clean_db(). ")
logging.info("To run a persistent model/experiment set  testing=False")

# define new experiment
training_config = {
    'epochs': 1,
    'verbose': 1,
    'batch_size': 1,
    "validation_steps": 20,
    "steps_per_epoch": 10  # in the non-testing environment the data is streamed and thus we do not
    # know how many steps per epoch we can do. This number should be len(training)/batch_size
}
round = ["fetch_model", "train_model", "send_validation_loss", "send_model", "aggregate"]
tasks = []
for i in range(2):  # number of rounds
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
