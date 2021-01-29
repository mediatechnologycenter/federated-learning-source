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
Follow these instructions to run a federated learning simulation
1. Install tf (I used 2.*. See requirements.txt for full list of my setup)
2. Copy federated learning folder
3. Define a function that returns your compiled model.  !important: set input_dim
4. Define number of clients, rounds, epochs, batch size
5. Define training/test data as numpy arrays
6. Initialize clients
7. Initialize server
8. Run rounds
"""
import tensorflow as tf
import os
import sys
import numpy as np
import random
import json

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# import tensorflow_addons as tfa
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D

os.environ['STATIC_VARIABLES_FILE_PATH'] = "globalserver/static_variables.json"

os.environ['PATH_TO_GLOBALSERVER'] = "globalserver/api/"
sys.path.append(os.getcwd())

from testing.test_class import Testing

os.getcwd()


def default(obj):
    if type(obj).__module__ == np.__name__:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj.item()
    raise TypeError('Unknown type:', type(obj))


tf.keras.backend.clear_session()


def get_data_kdd():
    with open(f'../datasets/train.txt', 'r') as f:
        train = f.readlines()
    with open(f'../datasets/train.txt', 'r') as f:
        test = f.readlines()
    return train, test


def split_half(train, test, clients):
    random.shuffle(train)
    train_c1 = train[:20]
    train_c2 = train[:20]
    validation_c1 = train[20:33]
    validation_c2 = train[20:33]

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


def kknox_nn(param_dict):
    model = Sequential()
    model.add(Dense(50, activation=tf.nn.relu, input_dim=61))
    model.add(Dense(50, activation=tf.nn.relu))
    model.add(Dense(1, activation=tf.nn.sigmoid))

    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=param_dict['lr']),  # momentum=mt),
                  loss='binary_crossentropy',
                  metrics=[])
    model.summary()
    return model


def upkeep_function(self, experiment_document):
    import random
    with open(f'datasets/train_kkbox0.jsonl', 'r') as f:
        train_c1 = f.readlines()
    random.shuffle(train_c1)
    with open(f'datasets/train_kkbox0.jsonl', 'w+') as f:
        f.writelines(train_c1)

    with open(f'datasets/train_kkbox1.jsonl', 'r') as f:
        train_c2 = f.readlines()
    random.shuffle(train_c2)
    with open(f'datasets/train_kkbox1.jsonl', 'w+') as f:
        f.writelines(train_c2)


def stop_function(self, experiment_document, aggregated_metric):
    experiment_document['validation_results']=experiment_document.get('validation_results',[])
    if len(experiment_document['validation_results']) > 2:
        return True, [len(experiment_document['validation_results'])]
    else:
        return False, [len(experiment_document['validation_results'])]


def preprocessing(batch):
    # batch=batch[-28:]
    return batch


clients = [f"kkbox{i}" for i in range(2)]

TestSetup = Testing(clients, start_servers=True, clear_logs=True, clear_db=False, interface=False)

train, test = get_data_kdd()
split_half(train, test, clients)

setup_dict = {"model_function": {
    "function": kknox_nn,
    "parameters": {
        "lr": 0.1
    }
},
    "git_version": 'e9339081b76ad3a89b1862bd38d8af26f0541f1c',
    "protocol": 'NN',
    "model_name": "test_model",
    "model_description": "this model is just to test the db",
    "testing": True,
    "training_config": {
        'epochs': 1,
        'verbose': 0,
        'batch_size': 1,
        "validation_steps": 10,
        "test_steps": 10,
        "steps_per_epoch": 10,
        "train_url": "train",
        "test_url": "test",
        "validation_url": "validation",
        "skmetrics":["f1_score"],
        "tfmetrics":["AUC","Accuracy"],

        # in the non-testing environment the data is streamed and thus we do not
        # know how many steps per epoch we can do. This number should be len(training)/batch_size

    },
    "rounds": 5,
    "clients": clients,
    "experiment_name": "kkbox",
    "experiment_description": f"desc if nice experiment",
    "stop_function": stop_function,
    "upkeep_function": upkeep_function,
    "preprocessing": {
        "noise": {
            "epsilon": 1,
            "delta": 0.2
        },
        # "cast_to_float": "",
        "feature_selection": ['active_days', 'afternoon', 'avg_session_start_hour',
                              'C1_Anrede', 'cookiesNumber', 'days_since_registration', 'evening',
                              'isMobile', 'langNew', 'morning', 'night', 'noon', 'notWeekend_active_days',
                              'notWeekend_sessions', 'num_display_articles', 'num_of_sessions',
                              'num_read_articles', 'number_of_newsletters', 'operatingSystemNew',
                              'paygate_impressions', 'popularCategoriesNew', 'preferredTimeOfDay',
                              'publishPathNew', 'refElemNew', 'time_from_last_session',
                              'weekend_active_days', 'weekend_sessions', 'avg_time_from_session',
                              'last_churn_before_registration_flag'],
        "preprocessing_function": preprocessing
    },
}

tunable_parameters = [
]


def trial_loss_function(metrics):
    return metrics[-1]['aggregated_metric'][0]


from globalserver.operator_.operator_class_db import Operator

operator = Operator()
operator.define_and_start_experiment(setup_dict)
