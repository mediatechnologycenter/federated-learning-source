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

import numpy as np
from scipy.stats import mode
import json
import math
import random
import logging

from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.metrics import f1_score

from . import tree
from . import histogram
from . import utilities

class RandomForestClassifier():
    """A random forest classifier implementation that is partly used in the implementation
    of the RF (Random Forest) Protocol inside the Federated-Learning Framework.
    This implementation allows the Client to get out a finished random forest from the
    framework and operate on it.
    ::
    Note that this algorithm is built to solve 2-class classification.
    ::
    Parameters
    ==========
    n_features: integer
        Number of features in the underlying dataset. Information is needed for the tree-building
        process.
    feature_information: dict (default={})
        Dictionary representing whether the underlying feature a continuous feature or a 
        categorical feature. The naming convention for the feature with index i (starting at 0)
        is ``("col%s" % i)``, with the value being a Boolean. When the value is True, 
        then the feature is considered as continuous, if False the feature is considered as 
        categorical. When no dict is given, all features are considered as continuous.
        This information is important for the Histogram-Merging process during the Training
        of the trees.
    n_estimators: integer (default=128)
        Number of trees in the forest.
    max_depth: integer or None (default=50)
        Maximum depth of any tree in the forest. If None, then the trees get expanded until 
        all leaves are pure or until minimal_information_gain criterion is met.
    max_features: String, int (default="sqrt")
        Maximum number of features to consider when looking for the best split while building
        a tree-node.
        Values:
        - instance of int: consider max_features many features
        - "sqrt": consider sqrt(n_features) features
        - "log2": consider log_2(n_features) features
    max_bins: integer (default=100)
        Maximum number of bins for continuous features while building a local histogram and
        while merging multiple histograms.
    pos_label: integer (default=1)
        Positive Label in the dataset.
    neg_label: integer (default=0)
        Negative Label in the dataset.
    minimal_information_gain: float (default=0.0)
        Minimal information gain to allow the tree build to go further down. Once the optimal
        information gain at the current tree-node is below (<=) the minimal_information_gain
        threshold a new node gets added and the current node will not become a leaf.
    metrics: list of strings (default=['log_loss'])
        List of metrics that the model should be evaluated on when sending the loss back
        in the Federated-Learning Framework.
    Attributes
    ==========
    forest: List of DecisionTreeClassifiers
        List of all trees in the current random forest.
    model_update: Histogram (default=None)
        Internal update field used during the training of the trees.
    current_condition_list: List of Conditions
        List of Conditions (3-Tuples) in json-format such that all Workers in the Federated Learning
        Framework get the same condition list to form the local histograms. The Conditions are implemented
        using python dictionaries. Gets updated by RF_aggregate after a tree-node has been generated.
    current_feature_list: List of int
        List of features to make the histograms of, during the training process of the federated
        random forest. Gets updated by RF_aggregate after a tree-node has been generated.
    random_state: int
        Integer representing the global random state such that the bootstrap samples at the workers can 
        be replicated. Is initialized at the forest setup. The random_state is importand for the tree-build
        in the FL Framework to ensure we always get the same bootstrap samples.
    Example
    =======
        ...TODO...
    """
    def __init__(self, n_features, feature_information={}, n_estimators=128, max_depth=50, max_features="sqrt", max_bins=100, pos_label=1, neg_label=0, minimal_information_gain=0.0, metrics=['log_loss']):
        # check input
        assert((max_features in ['sqrt', 'log2']) or (isinstance(max_features, int)))

        # assign values
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.max_bins = max_bins
        self.pos_label = pos_label
        self.neg_label = neg_label
        self.minimal_information_gain = minimal_information_gain
        self.n_features = n_features
        self.feature_information = feature_information
        self.metrics = metrics # List of metrics to return during send_loss

        self.forest = []
        # internal attribute needed to store current histogram (in json format) in Training of current Tree
        self.model_update = {}
        self.current_condition_list = [] # gets updated by RF_aggregate procedure!
        # internal attribute, needed to say worker for what features he should produce histograms
        self.current_feature_list = []
        # to initialize this feature list, use the below code:
        # range_list = [i for i in range(self.n_features)]
        # n_subfeatures = int(math.floor(math.sqrt(n_features))) # init as default value
        # if (self.max_features == 'sqrt'):
        #     n_subfeatures = int(math.floor(math.sqrt(n_features)))
        # elif (self.max_features == 'log2'):
        #     n_subfeatures = int(math.floor(math.log(n_features, 2)))
        # elif (isinstance(self.max_features, int)):
        #     n_subfeatures = self.max_features
        # self.current_feature_list = random.sample(range_list, n_subfeatures)

        # random state has to be set at initialization, but never changed anymore after this.
        # the random state ensures that the worker gets the same bootstrap sample every time it's built again
        self.random_state = random.randint(0, 999999999)

    def to_json(self):
        """Function returns a dict, containing all information of this model.

        To store this information as a json, one has to call ``json.dumps(result)``
        on the result of this function, to get a json-like string.

        To restore this information, one has to call ``json.loads(json_string)``
        and then pass the result of this to ``RandomForestClassifier.from_json(result)``.

        This function returns a python dictionary.
        """
        model_dict = {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'max_features': self.max_features,
            'max_bins': self.max_bins,
            'pos_label': self.pos_label,
            'neg_label': self.neg_label,
            'n_features': self.n_features,
            'feature_information': self.feature_information,
            'minimal_information_gain': self.minimal_information_gain,
            'metrics': self.metrics,
            'current_condition_list': [],
            'current_feature_list': [],
            'random_state': self.random_state,
            'forest': [],
        }
        # add all trees from the forest to the dictionary under forest
        for tree in self.forest:
            model_dict['forest'].append(tree.to_json())
        return model_dict

    @staticmethod
    def from_json(json_object):
        """Function to take a json-object (python dictionary) which returns a full random forest 
        from the given information.
        ::
        Note: it is assumed that json.loads(json_string) is already called and thus the argument
        to this function is a python dictionary.
        ::
        Parameters
        ==========
        json_object: dict
            Python dictionary that contains all information that is stored in the .to_json(self)
            function.
        """

        model = RandomForestClassifier(
            n_features = json_object['n_features'],
            feature_information = json_object['feature_information'],
            n_estimators = json_object['n_estimators'],
            max_depth = json_object['max_depth'],
            max_features = json_object['max_features'],
            max_bins = json_object['max_bins'],
            pos_label = json_object['pos_label'],
            neg_label = json_object['neg_label'],
            minimal_information_gain = json_object['minimal_information_gain'],
            metrics = json_object['metrics']
        )
        # handle internal state
        model.current_condition_list = json_object['current_condition_list']
        model.current_feature_list = json_object['current_feature_list']
        model.random_state = json_object['random_state']
        # append all decision trees from the forest to the model
        tree_list = json_object.get('forest', [])
        for tree in tree_list:
            model.forest.append(tree.from_json())
        
        return model

    def get_parameters(self):
        """Return a python dict, with key "forest", under which is a list stored of
        python dicts, corresponding to all trees in the current forest.

        To get a json-file, one has to call ``json.dumps(result)`` to get the corresponding
        string to store.
        """
        tree_dict = {
            'forest': []
        }
        for tree in self.forest:
            tree_dict['forest'].append(tree.to_json())
        return tree_dict

    def fit(self, X, Y):
        """Function to fit a random forest to local data.
        Note that this function does not take any bootstrap sample from the given data.
        If one expects to use a bootstrap sample, one has to pass the bootstrap sample to
        this function as arguments.
        Parameters
        ==========
        X: Numpy Array of Numpy Arrays (shape: (n_samples, n_features))
            Data representing all features
        Y: Numpy Array of Numpy Arrays (shape: (n_samples, 1))
            Data representing the corresponding labels
        """
        assert((X is not None) and (Y is not None))
        assert(X.shape[0] == Y.shape[0])

        for _ in range(self.n_estimators):
            decision_tree = tree.DecisionTreeClassifier(
                n_features = self.n_features,
                feature_information = self.feature_information,
                max_features = self.max_features,
                max_depth = self.max_depth,
                max_bins = self.max_bins,
                minimal_information_gain = self.minimal_information_gain
            )
            decision_tree.fit(X, Y)
            self.forest.append(decision_tree)

    def predict(self, X):
        """Predict the class of each sample in X.
        ::
        Parameters
        ==========
        X: Numpy array of numpy arrays, shape (n_samples, n_features)
        Returns
        =======
        numpy.ndarray of shape (n_samples,)
        """
        n_samples = X.shape[0]
        n_trees = len(self.forest)
        predictions = np.empty([n_trees, n_samples])
        for i in range(n_trees):
            predictions[i] = self.forest[i].predict(X)
        return mode(predictions)[0][0]

    def accuracy_score(self, X, Y):
        """Return accuracy score of prediction of X compared to reference vector Y.
        Parameters
        ==========
        X: Numpy.ndarray of shape (n_samples, n_features)
        Y: Numpy.ndarray of shape (n_samples,)
        Returns
        =======
        float representing the accuracy score
        """
        assert(Y.shape[0] == X.shape[0])
        y_pred = self.predict(X)
        return accuracy_score(Y, y_pred)


    def f1_score(self, X, Y):
        """Return f1 score of prediction of X compared to reference vector Y.
        Parameters
        ==========
        X: Numpy.ndarray of shape (n_samples, n_features)
        Y: Numpy.ndarray of shape (n_samples,)
        Returns
        =======
        float representing the accuracy score
        """
        assert(Y.shape[0] == X.shape[0])
        y_pred = self.predict(X)
        return f1_score(Y, y_pred, pos_label = self.pos_label)

    def logloss_score(self, X, Y):
        """Return log-loss score of prediction of X compared to reference vector Y.
        Parameters
        ==========
        X: Numpy.ndarray of shape (n_samples, n_features)
        Y: Numpy.ndarray of shape (n_samples,)
        Returns
        =======
        float representing the accuracy score
        """
        assert(Y.shape[0] == X.shape[0])
        y_pred = self.predict(X)
        return log_loss(Y, y_pred)