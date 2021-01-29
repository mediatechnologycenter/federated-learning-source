Our framework allows a low number of stakeholders to train a model in a federated way. It is easy to setup for the stakeholders as well as for the orchistrator. It incorperates three types of algorithms: standard federated averaging via neural networks and two novel tree based federated learning approaches. The stakeholders control the amount on differential privacy they want to ensure. The algorithms are based on these papers [[1]](#1).

# Documentation
Look into the [documentation](documentation/README.md) for detailed description of the pipeline.

# Testing Pipeline
Global Server and nodes run locally. The files in the Github repository are used to create the nodes.

## Installation
1. Install [Mongodb](https://docs.mongodb.com/manual/installation/)

    2. Start Daemon (`mongod --port <your_port> --dbpath /home/schneech/mongodb  --replSet rs0`)
    3. Open a mongo shell (`mongo --port <your_port>`) and initialize Replica set in mongo shell with `rs.initiate(`)
    4. [Enable Access Control](https://docs.mongodb.com/manual/tutorial/enable-authentication/): In the mongo shell:
        
         `   use admin
        db.createUser(
          {
            user: "myUserAdmin",
            pwd: passwordPrompt(), // or cleartext password
            roles: [ { role: "userAdminAnyDatabase", db: "admin" }, "readWriteAnyDatabase" ]
          }
        )`
        
    5. Stop Daemon
    6. Start the mongodb with `mongod --port <your_port> --dbpath /home/schneech/mongodb  --replSet rs0 --auth  --fork`.
    7. Create a database with a database named `federated_learning` (mongoshell: `use federated_learning`) and insert empty documents in collection model, experiment and task (mongoshell: `db.model.insert({})`,`db.experiment.insert({})`,`db.task.insert({})`).
    8. Add a credential file `db_conf.key` in the globalserver directory with the form:
`{"port": "port","host":"0.0.0.0","user": "user","password": "pw"}`.
9. Clone git repo:<br/>
`git clone https://gitlab.ethz.ch/mtc-federated-learning/Federated-Learning.git`
10. Install required python libraries (python3):<br/>`pip install --requirement "requirements.txt"`
11. Create a folder "datasets" in root directory and copy training data into it with the following naming.<br/>
`train_<clientname>.jsonl, test_<clientname>.jsonl`<br/>
The format should be [json-lines](http://jsonlines.org/) (column-names are ignored, but there needs to be a key "label"):<br/>
`{"col1":value1_1,"col2":value1_2,"label":1}`<br/>
`{"col1":value2_1,"col2":value2_2,"label":0}`
12. Add a file `envs.key` in root directory. SERVER_ADDRESS, CLIENT_SECRET and SERVER_PORT are required:

`{"SERVER_ADDRESS" : "your address or 0.0.0.0",
"CLIENT_SECRET": "this can be anything",
"SERVER_PORT" :"port for example 50000",
"CLIENT_INTERFACE_PORT" :"optional - needed if run with client interface",
"DATA_WRAPPER_URL":"optional - needed if run in docker mode"}`

Add a file `valid_clients.json` in the globalserver directory. Specifies the allowed clients in docker mode. For local setup, just copy the dummy entries below.

`{ "VALID_CLIENTS": [
    "Clientname1",
    "Clientname2",
    "INTERFACE"
  ]}`
  
## Test Installation:
Start Jupiter Notebook (`jupyter notebook`) and navigate to the examples/dummy_example folder. Try to run dummy_example.ipynb.

**The logs are written into `testing.testing.log`**
  
## Dashboard:
For the dashboard look into its [documentation](dashboard/README.md)



## Run
### Toyexamples:
All the examples start the global server and nodes automatically. To shut them down run `shutdown.sh` or use the Testing class described below.
* NN Basic Example: Run `python3 examples/operator_example.py` in the root directory of the repository.
* NN Setupdict: Run `python3 examples/operator_example_setupdict.py` in the root directory of the repository.
* NN Notebook old wrapper optuna:	Run `python3 examples/notebook_example_old_wrapper_optuna.ipynb` in the root directory of the repository.
* NN Notebook new wrapper:	Run `python3 examples/notebook_example_new_wrapper.ipynb` in the root directory of the repository.
* NN New Data Wrapper:	Run `python3 examples/setup_dict_new_wrapper.py` in the root directory of the repository.
* P2P Basic:	Run `python3 examples/operator_example_p2p.py` in the root directory of the repository.
* P2P new wrapper:	Run `python3 examples/operator_example_p2p_new_wrapper.py` in the root directory of the repository.
* RF Basic:	Run `python3 examples/operator_example_rf.py` in the root directory of the repository.
* RF new wrapper:	Run `python3 examples/operator_example_rf_new_wrapper.py` in the root directory of the repository.

**The logs are written into `testing.testing.log`**

### Usage description
#### Testing Class
We constructed a helper class for running experiments locally `class Testing in testing/test_class.py`.
Whenever you use this class, make sure that your current working dir is the root directory and that the root directory is in the sys path (see examples on how to do that).

The `Testing` class is used to start up and shutdown the global server/nodes, to clear the db and filesystem from all experiments with flag `testing=True` and to clear the logs. 
When you open a new instance you can already start the servers,clear logs/db:\
`TestSetup = Testing(clients, start_servers=True, clear_logs=True, clear_db=True)`\

**The logs are written into `testing.testing.log`**

In the end you can shut them down with `TestSetup.kill_servers()`. Please look into the class for more the documentation of all the functionalities.

## Setup dict
We implemented three Algorithms: Neural Networks (NN), Peer2Peer XGBoost (P2P) and Random Forest(RF). For each of these algorithms there is a set of parameters you can set to optimize the model architecture and the training process. There are also some parameters which are federated learning specific. These can be used in every algorithm:
* clients:  `list` List of clients/nodes to perform the tasks
* protocol: `string` the protocol to use. one of NN,RF,P2P
* round: `list` optional - List of tasks to perform in each round. Each task is one of "fetch_model", "train_model", "send_training_loss", "send_validation_loss","send_test_loss",  "send_model", "aggregate". Default is `["fetch_model", "train_model", "send_model", "send_validation_loss", "aggregate"]`
* rounds: `int` number of rounds to perform.
* final_round: `list` optional - List of tasks to perform after all rounds have finished. Default is `["fetch_model", "send_test_loss", "send_validation_loss"]`

* model_function.function: `function` This function needs to have exactle one parameter called `param_dict`. We describe the expected return in in the sections for the target algorithm.
* model_function.parameters:  `dict` the parameters that the model_function is called with.
* stop_function: `function` optional - function which gives the possibility to give conditions to early stop the experiment and to return additional metrics. The function is called in each aggregation step. It needs three input parameters: `stop_function(self, experiment_document, aggregated_metric)`. `self` is used to access all class variables from the operator class. `experiment_document` is the mongodb entry of the current experiment. `aggregated_metric` is the list of aggregated metrics calculated in the aggregation step. The function needs to return two values, a `bool` indicating whether to stop the experiment (True) or not. And a `list` containing additional metrics. See setup_dict_new_wrapper.py.
* upkeep_function: `function` optional - This function is called in each aggregation step. It can for example be used to shuffle the data or print the current performance.It needs two inputs `upkeep_function(self, experiment_document)`. `self` gives access the operator class variables and `experiment_document` contains the current experiment mongodb entry (see stop_function). It has no return value. See setup_dict_new_wrapper.py as an example.

* testing: `bool` whether the experiment is only for testing
* preprocessing.preprocessing_function:  `function` optional -   this function is called in the generator on the node after noisifying the data. It has input `preprocessing(batch)` and returns a batch. A batch has following format: `{"feature1": [sample1_feature1,sample2_feature1,....], "feature2":[...], ... "label":[sample1_label,sample2_label]}`
* preprocessing.default_value: `string` optional -  the fallback default value if there is no default value set in metadata_description
* preprocessing.noise.epsilon: `int` optional for local setup - differential privacy epsilon value ]0,inf[. The larger the less dp. (ratio how far the probability of two random samples after adding noise can be appart. i.e. if we have samples a,b: p(f(a))/p(f(b))<e^epsilon)
* preprocessing.noise.delta: `int` optional for local setup - differential privacy delta value [0,1]. The larger the less dp. (this value tells to how many samples can violate the dp condition)
* preprocessing.features: `list` optional - only used for old datawrapper. List of features we work with. If not set it works with all the features we get from the first row.

* (new) training_config.custom: Whenever the clients need separat configurations you can give them here. For example if you need seperate batch_size and epochs this should be `{'client1`:{'batch_size':100,'epochs':4},'client2':{'batch_size':200,'epochs':6}}` 
* training_config.skmetrics: `list` List of metrics as string from sklearn library to calculate in the loss calculation task. The node will try to getattr the function, i.e. `getattr(skmetrics, <metric_string>)(y_pred, y_true)`
* training_config.tfmetrics: `list` List of metrics as string from tf.metrics. The node will try to `m = getattr(tf.metrics, <metric_string>)() m.update_state(y_true=y_true, y_pred=y_pred)`
* training_config.dataset: `string` identifier of the dataset. In local mode this is just the path to the datasets. The files should be stored in  `datasets/<dataset>train_<client>.jsonl`
* training_config.dataset_metadata: `dict` optional -  Only needed in the local setup (if not set it will be calculated on the node). The metadata to add noise to define the features and add noise. It can be constructed from the dataset by calling the function`get_metadata(clientname, path_to_{train_client.jsonl}-file` in operator_utils.py. If it is not set, then the worker will call the function in the first fetch_model task.)
* training_config.differential_privacy.method: optional -  One of 'before' or 'after'. If it is before, than the noise (cf. preprocessing.noise) is added directly to the data. After is only implented for RF. If applies noise to the histogram.
* We have metadatas variables: experiment_name,experiment_description,model_name,model_description,git_version

### Neural Networks (NN)
* model_function.function: `function` For NN this function should return a compiled tf2 model. Any model can be used. Any non-custom optimizer can be used to compile the model and all tf2.copmlie options are available.
* model_function.parameters:  `dict` Of you want to parametrize anything (for example learning rate) you can do it with this dict.
* (new) training_config.update_rate: optional - the weight applied to the aggregated weight updates when applied to the global model's weights
* (new) training_config.update_decay: optional - a decay rate to reduce the update_rate over time. (update_rate = update_rate- current_round/total_rounds*update_decay)
* (new) training_config.total_rounds: optional - total number of rounds. for the decay
* training_config.*: any tf.fit parameter can be given in the training_config. Remember that we us a generator to stream the data - set steps_per_epoch. for example:
	* training_config.epochs: `int` (see keras.fit)
	* training_config.batch_size: `int` 2000 (see keras.fit)
	* training_config.steps_per_epoch: `int` normally dataset-size/batch_size (see keras.fit)
	* training_config.class_weight: `dict`(see keras.fit)
	* training_config.verbose: `int` 1 (see keras.fit)

* training_config.validation_steps: `int` the steps_per_epoch when looking at the validation data
* training_config.test_steps: `int` the steps_per_epoch when looking at the test data


### Random Forest (RF)
* training_config.batch_size: The number of samples that should be drawn to pick the bootstrap from. If small data then the whole dataset.
* training_config.bootstrap_size: The number of random samples that should be draw from the batch and build the tree on.
* training_config.cast_to_probabilities: optional - Whether or not to softmax the model predictions into probabilities.
* training_config.batch_size_validation: batch_size for validation set
* training_config.batch_size_test:  batch_size for test set
* training_config.balanced_subsample: whether the bootstraped samples should be balanced or not  
* model_function.function: This is our own implementation. Always use  -> operator_utils.RF_get_compiled_model:
* model_function.parameters: You can set these parameters:
    - n_features: Number of features in the input data 
    - feature_information: `dict`            Dictionary containing for each feature if it's values are continuous (True)            or categorical (False).            Naming convention for columns are ``f"col{i}"`` where i is the index of the   respective column in the input data (index starting at 0).  When no value is given, the feature will be assumed to have continuous values.
    - n_estimators: int (default=128)            Number of trees to build in the random-forest. The algorithm builds 7 trees in parallel and then the next 7,....
    - max_depth: int or None (default=50)            Maximum depth of any tree.
    - max_features: String, int (default="sqrt")            Maximum number of features to consider when looking for the best split.            Values:
        - instance of int: consider max_features many features
        - "sqrt": consider sqrt(n_features) features
        - "log2": consider log_2(n_features) features
    - max_bins: int (default=100)            Maximum number of bins allowed for continuous features while building            histograms during the tree-build.
    - pos_label: int (default=1)            Positive label of the data.
    - neg_label: int (default=0)            Negative label of the data.
    - minimal_information_gain: float (default=0.0)            Minimal information gain to not insert a leaf at the tree building process.
    - metrics: list of strings (default=['log_loss'])  (obsolete)          List of metrics that the model should be evaluated on when sending loss            back. Possible values are 'log_loss', 'accuracy', 'f1_score'



### XGBoost (P2P)
* model_function.function: xgb.Booster Object. Normally just return xgb.Booster(param_dict['params'], [param_dict['example']])
* model_function.parameters: Normally just the parameters needed for creating a booster object: 
	- {'params':{'max_depth': 8, 'subsample': 0.5, 'eta': 0.01, 'max_delta_step': 0,
                        'scale_pos_weight': 1.5, 'objective': 'binary:logitraw',
                        'tree_method': 'hist', 'max_bin': 250, 'colsample_bytree': 1},'example':example}
* training_config.client_steps_per_round: How many trees/boosting rounds should be done in one traing step
* training_config.cast_to_probabilities: optional - Whether or not to softmax the model predictions into probabilities.
* training_config.batch_size: number of samples to build the trees on
* training_config.batch_size_validation: number of samples to evaluet from
* training_config.batch_size_test: number of samples to evaluet from
* training_config.verbosity: Verbosity of the training run
* training_config.nthread: tbd

# Production Pipeline
Global Server runs on server and nodes run on clients as Docker containers fetch from the docker hub.


## References

<a id="1">[1]</a>
Keith Bonawitz, Hubert Eichner, Wolfgang Grieskamp, Dzmitry Huba, Alex Ingerman, Vladimir Ivanov, Chloe Kiddon, Jakub Kone?ný, Stefano Mazzocchi, H. Brendan McMahan, Timon Van Overveldt, David Petrou, Daniel Ramage, Jason Roselander. Towards Federated Learning at Scale: System Design
https://arxiv.org/abs/1902.01046
