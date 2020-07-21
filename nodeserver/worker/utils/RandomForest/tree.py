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

import math
import numpy
import json
import random
from scipy.stats import mode

from . import utilities
from . import histogram

class DecisionTreeClassifier():
    """A decision tree implementation for 2-class classification. This implementation is party used in the 
    Federated-Learning Framework. However the implementation should mainly allow the user of this framework
    to operate on the trees after the training process.
    ::
    Note that this algorithm is built to solve 2-class classification.
    ::
    Parameters
    ==========
    n_features: integer
        Number of features in the underlying dataset.
    feature_information: dict (default={})
        Contains information whether a given feature contains continuous or categorical values.
        This information is used in the Federated Learning Framework during the training process, but
        is not necessary for local evaluations using this class.
    max_features: string, int (default="sqrt")
        Maximum number of features to consider when looking for the best split while building
        a tree-node.
        Values:
        - instance of int: consider max_features many features
        - "sqrt": consider sqrt(n_features) features
        - "log2": consider log_2(n_features) features
    max_depth: integer or None (default=50)
        Maximum depth of any tree in the forest. If None, then the trees get expanded until 
        all leaves are pure or until minimal_information_gain criterion is met.
    max_bins: integer (default=100)
        Maximum number of bins for continuous features while building a local histogram and
        while merging multiple histograms.
    minimal_information_gain: float (default=0.0)
        Minimal information gain to not add a leaf to the decision tree.
    """
    def __init__(self, n_features, feature_information={}, max_features="sqrt", max_depth=50, max_bins=100, minimal_information_gain=0.0):
        # check input
        assert(isinstance(n_features, int))
        assert(isinstance(feature_information, dict))
        assert((max_features in ['sqrt', 'log2']) or (isinstance(max_features, int)))
        assert(isinstance(max_depth, int) or max_depth is None)
        assert(isinstance(max_bins, int))

        # assign values
        self.n_features = n_features
        self.feature_information = feature_information
        if (max_features == 'sqrt'):
            self.max_features = int(math.floor(math.sqrt(n_features)))
        elif (max_features == 'log2'):
            self.max_features = int(math.floor(math.log(n_features, 2)))
        self.max_depth = max_depth
        self.max_bins = max_bins
        self.minimal_information_gain = minimal_information_gain

        # initialize tree as empty tree
        self.tree = None

    def to_json(self):
        """Function to return the current tree as a dict.
        Note: One still has to call json.dumps(result) on the result!
        """
        tree_dict = {
            'n_features': self.n_features,
            'max_features': self.max_features,
            'feature_information': self.feature_information,
            'max_depth': self.max_depth,
            'max_bins': self.max_bins,
            'minimal_information_gain': self.minimal_information_gain,
            'tree': "None"
        }
        if self.tree is not None:
            tree_dict['tree'] = self.tree.to_json()
        return tree_dict

    @staticmethod
    def from_json(json):

        decision_tree = DecisionTreeClassifier(
            n_features = json['n_features'],
            feature_information = json['feature_information'],
            max_features = json['max_features'],
            max_depth = json['max_depth'],
            max_bins = json['max_bins'],
            minimal_information_gain = json['minimal_information_gain']
        )
        if (json['max_features'] == 'sqrt'):
            decision_tree.max_features = int(math.floor(math.sqrt(json['n_features'])))
        elif (json['max_features'] == 'log2'):
            decision_tree.max_features = int(math.floor(math.log(json['n_features'], 2)))

        if json['tree'] != "None":
            decision_tree.tree = DecisionTreeNode.from_json( json['tree'], None )
        return decision_tree

    def get_parameters(self):
        """Return the trees in json format. I.e. do the same as in to_json, but without the
        model specifications.
        """
        if self.tree is not None:
            return json.dumps( self.tree.to_json() )
        else:
            return None

    def fit(self, X, Y):
        """Function to recursively build up a tree on the local data.
        """
        assert((X is not None) and (Y is not None))
        # get the bootstrap sample
        X_bootstrap, Y_bootstrap = utilities.bootstrap_sample(X, Y)
        # start buiding the tree recursively
        self.tree = self._build_node(
            depth = 0,
            parent = None,
            X = X_bootstrap,
            Y = Y_bootstrap
        )
        return

    def _build_node(self, depth, parent, X, Y):
        """Recursively build up a decision tree
        """
        # subsample the features to look at
        range_list = [i for i in range(self.n_features)]
        feature_indices = random.sample(range_list, self.max_features)
        # add a leaf if reached maximum depth or uniform data distribution
        entropy_Y = utilities.entropy(Y)
        if (depth is self.max_depth) or (entropy_Y is 0):
            return DecisionTreeNode(
                feature_index = None,
                threshold = None,
                depth = depth,
                is_final_leaf = True,
                left_child = None,
                right_child = None,
                parent_node = parent,
                y = mode(Y)[0][0]
            )
        # find optimal split
        feature_index, threshold = DecisionTreeClassifier.find_split(X, Y, feature_indices)
        # split the data according to the found decision
        # Data that satisfies the condition goes left, other data goes right down the tree
        X_True, Y_True, X_False, Y_False = utilities.split(X, Y, feature_index, threshold)
        # If any returned data is empty, return a leaf, or if information gain is too small

        if (Y_True.shape[0] is 0) or (Y_False.shape[0] is 0) or (utilities.information_gain(Y, Y_True, Y_False) < self.minimal_information_gain):
            return DecisionTreeNode(
                feature_index = None,
                threshold = None,
                depth = depth,
                is_final_leaf = True,
                left_child = None,
                right_child = None,
                parent_node = parent,
                y = mode(Y)[0][0]
            )
        # recursively add the children nodes to the tree
        current_node = DecisionTreeNode(
            feature_index = feature_index,
            threshold = threshold,
            depth = depth,
            is_final_leaf = False,
            left_child = None,
            right_child = None,
            parent_node = parent,
            y = None
        )
        current_node.left_child = self._build_node(
            depth = depth + 1,
            parent = current_node,
            X = X_True,
            Y = Y_True
        )
        current_node.right_child = self._build_node(
            depth = depth + 1,
            parent = current_node,
            X = X_False,
            Y = Y_False
        )
        return current_node

    @staticmethod
    def find_split(X, Y, feature_indices):
        """Find the best split rule for the given data and return this rule.
        The rule is of the form feature_index, threshold
        """
        n_features = X.shape[0]
        best_gain = 0.0
        best_feature_index = 0
        best_threshold = 0
        for feature_index in feature_indices:
            values = sorted(set(X[:, feature_index]))
            for j in range(len(values) - 1):
                threshold = float(values[j] + values[j+1]) / 2.0
                X_True, Y_True, X_False, Y_False = utilities.split(X, Y, feature_index, threshold)
                gain = utilities.information_gain(Y, Y_True, Y_False)

                if gain > best_gain:
                    best_gain = gain
                    best_feature_index = feature_index
                    best_threshold = threshold
        return best_feature_index, best_threshold

    def predict(self, X):
        """Predict the class of each sample in X.
        """
        n_samples = X.shape[0]
        Y = numpy.empty(n_samples)
        for j in range(n_samples):
            curr_node = self.tree

            while curr_node._is_leaf() is False:
                if X[j, curr_node.feature_index] <= curr_node.threshold:
                    curr_node = curr_node.left_child
                else:
                    curr_node = curr_node.right_child

            Y[j] = curr_node.y
        return Y

class DecisionTreeNode:
    def __init__(self, feature_index, threshold, depth, is_final_leaf, left_child, right_child, parent_node, y=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.depth = depth
        self.is_final_leaf = is_final_leaf
        self.left_child = left_child
        self.right_child = right_child
        self.parent_node = parent_node
        self.y = y
    
    def to_json(self):
        """Returns a dict correspondint to all information in the object.
        One has to call json.dumps(result) on the result to get a json-like string.
        """
        node_dict = {
            'feature_index': "None",
            'threshold': "None",
            'depth': self.depth,
            'is_final_leaf': self.is_final_leaf,
            'y': "None",
            'left_child': "None",
            'right_child': "None",
        }
        if self.feature_index is not None:
            node_dict['feature_index'] = self.feature_index
        if self.threshold is not None:
            node_dict['threshold'] = self.threshold
        if self.y is not None:
            node_dict['y'] = self.y
        if self.left_child is not None:
            node_dict['left_child'] = self.left_child.to_json()
        if self.right_child is not None:
            node_dict['right_child'] = self.right_child.to_json()
        return node_dict

    @staticmethod
    def from_json(json, parent_node=None):
        node = DecisionTreeNode(
            feature_index = json['feature_index'],
            threshold = json['threshold'],
            depth = json['depth'],
            is_final_leaf = json['is_final_leaf'],
            y = None,
            left_child = None,
            right_child = None,
            parent_node = parent_node
        )
        if json['feature_index'] != "None":
            node.feature_index = json['feature_index']
        if json['threshold'] != "None":
            node.threshold = json['threshold']
        if json['y'] != "None":
            node.y = json['y']
        if json['left_child'] != "None":
            node.left_child = DecisionTreeNode.from_json(json['left_child'], node)
        if json['right_child'] != "None":
            node.right_child = DecisionTreeNode.from_json(json['right_child'], node)
        return node

    def _is_root(self):
        if self.parent_node is None:
            return True
        else:
            return False

    def _is_leaf(self):
        if (self.is_final_leaf or (self.y is not None)):
            assert(self.is_final_leaf)
            assert(self.y is not None)
            assert(self.left_child is None)
            assert(self.right_child is None)
            return True
        else:
            return False

    @staticmethod
    def get_condition_list(node):
        if node is None or node._is_root():
            return []
        else:
            parent_list = DecisionTreeNode.get_condition_list(node.parent_node)
            result_b = True
            if node.parent_node.right_child is node:
                result_b = False
            parent_list.append( {
                'feature_index': node.parent_node.feature_index,
                'threshold': node.parent_node.threshold,
                'condition': result_b,
            } )
            return parent_list