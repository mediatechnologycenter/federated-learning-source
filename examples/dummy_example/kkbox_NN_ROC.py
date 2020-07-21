import os
import sys
os.environ['STATIC_VARIABLES_FILE_PATH'] = "globalserver/static_variables.json"
os.environ['PATH_TO_GLOBALSERVER'] = "globalserver/api/"
sys.path.append(os.getcwd())

import json

# Importing the required Keras modules containing model and layers
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense



from examples.dummy_example.utils import get_data,save_data_as_json,plot_data



def kkbox_nn(parameters):
    model = Sequential()
    layers = 5
    nodes = 16
    lr = 0.01

    for i in range(layers):
        if i == 0:
            model.add(Dense(nodes, activation=tf.nn.relu, input_shape=(2,)))
        else:
            model.add(Dense(nodes, activation=tf.nn.relu))

    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=tf.keras.optimizers.SGD(lr=lr, momentum=0.9),
                  loss='binary_crossentropy',
                  metrics=[tf.metrics.AUC()])
    model.summary()
    return model


clients = ["r1","r0"]


def preprocessing(batch):
    import numpy as np

    batch['label'] = [int(value) for value in batch['label']]

    return batch


setup_dict = {"model_function": {
    "function": kkbox_nn,
    "parameters": {
    }
},
    "git_version": 'e9339081b76ad3a89b1862bd38d8af26f0541f1c',
    "protocol": 'NN',
    "model_name": "test_model",
    "model_description": "this model is just to test the db",
    "testing": True,
    "training_config": {
        'epochs':  100,
        'verbose': 2,
        'batch_size': 100,
        "validation_steps": 40,
        "dataset":'',
        "test_steps": 40,
        "steps_per_epoch": 20,#int(14679/1000),
        "skmetrics": [],
        "tfmetrics": ["AUC", "Accuracy"],
        "differential_privacy": {"method": 'before',
                                 },

        "custom": {"r1": {"class_weight": {"0": 1, "1": 1},
                                'epochs': 1,
                                "steps_per_epoch": 1,#int(14679/1000),

                                },
                   "r2": {"class_weight": {"0": 1, "1": 1},
                            'epochs': 8,
                            "steps_per_epoch": 1,#int(14679/1000),

                            },
                   },
        # in the non-testing environment the data is streamed and thus we do not
        # know how many steps per epoch we can do. This number should be len(training)/batch_size

    },
    "rounds": 2,
    "round": ["fetch_model", "train_model", "send_model", "send_training_loss", "send_test_loss", "aggregate"],
    "final_round": ["fetch_model","send_test_loss", "send_training_loss"],
    "clients": clients,
    "experiment_name": "kkbox",
    "experiment_description": f"desc if nice experiment",
    "stop_function": None,
    "upkeep_function": None,
    "preprocessing": {
        "noise": {
            "epsilon": 10000,
            "delta": 1
        },
        "preprocessing_function": preprocessing
    },
}


from globalserver.operator_.operator_class_db import Operator
from testing.test_class import Testing

TestSetup = Testing(clients, start_servers=True, clear_logs=True, clear_db=True, interface=False)
operator = Operator()

#
# training_data, client1_data_final,client2_data_final,y=get_data(exp=3)
# test_data, _,_,_=get_data(seed=10,exp=3)
# save_data_as_json(client1_data_final,client2_data_final,test_data)

for clients in [["r1","r0"],["r1"],["r0"]]:
    setup_dict["clients"]= clients
    experiment_id,_=operator.define_and_start_experiment(setup_dict)
    #
    # model=operator.get_compiled_model(protocol='NN', experiment_id=experiment_id)
    # model.evaluate(x=test_data[:,0:2],y=test_data[:,2], verbose=2)
    # y_pred=model.predict(x=test_data[:,:2])
    # test_data[:, 2]=[1 if y[0]>0.5 else 0 for y in y_pred]
    #
    # plot_data([test_data[test_data[:,2]>0],test_data[test_data[:,2]<1],y])
