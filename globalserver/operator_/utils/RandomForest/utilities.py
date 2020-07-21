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
import numpy
from collections import namedtuple
from collections import Counter
from sklearn.utils import resample

def split(feature_index, threshold, X, Y):
    # Data for left child
    X_True = []
    Y_True = []
    # Data for right child
    X_False = []
    Y_False = []
    # split the data
    for j in range(Y.shape[0]):
        if X[j][feature_index] <= threshold:
            X_True.append(X[j])
            Y_True.append(Y[j])
        else:
            X_False.append(X[j])
            Y_False.append(Y[j])
    X_True = numpy.array(X_True)
    Y_True = numpy.array(Y_True)
    X_False = numpy.array(X_False)
    Y_False = numpy.array(Y_False)

    return X_True, Y_True, X_False, Y_False    


def bootstrap_sample(X, Y):
    X_bootstrap, Y_bootstrap = resample(X, Y, replace=True, stratify=Y)
    return X_bootstrap, Y_bootstrap

def entropy(Y):
    distribution = Counter(Y)
    s = 0.0
    total = float(Y.shape[0])
    for _, num_y in distribution.items():
        probability_y = (float(num_y) / total)
        s += (probability_y) * numpy.log(probability_y)
    return -s

def entropy_of_histogram(histogram):
    n_pos = 0
    n_neg = 0

    for bin_ in histogram.hist:
        n_pos += bin_['n_pos']
        n_neg += bin_['n_neg']
    n_total = n_pos + n_neg

    if (n_pos is 0) or (n_neg is 0) or (n_total is 0):
        return 0.0

    s = 0.0
    probability_pos = float(n_pos) / n_total
    probability_neg = float(n_neg) / n_total
    s += probability_pos * numpy.log(probability_pos)
    s += probability_neg * numpy.log(probability_neg)

    return -s

def information_gain(Y_all, Y_true, Y_false):
    E_y = entropy(Y_all)
    E_y_true = entropy(Y_true)
    E_y_false = entropy(Y_false)
    return E_y - float((E_y_true * Y_true.shape[0]) + (E_y_false * Y_false.shape[0])) / Y_all.shape[0]

def information_gain_from_histogram(histogram, threshold):
    """Function computes the information gain of having a specific treshold for a given histogram.
    ::
    Note: A threshold should be taken such that it is in the middle of two bins (when all bins are in a
    sorted order)
    """
    E_y = entropy_of_histogram(histogram)
    # compute E_y_true and E_y_false
    n_pos_lower = 0
    n_neg_lower = 0
    n_pos_higher = 0
    n_neg_higher = 0
    for bin_ in histogram.hist:
        if bin_['bin_identifier'] <= threshold:
            n_pos_lower += bin_['n_pos']
            n_neg_lower += bin_['n_neg']
        else:
            n_pos_higher += bin_['n_pos']
            n_neg_higher += bin_['n_neg']
    n_total = float(n_pos_lower + n_neg_lower + n_pos_higher + n_neg_higher)
    n_lower  = float(n_pos_lower + n_neg_lower)
    n_higher = float(n_pos_higher + n_neg_higher)

    # to avoid division by 0, return if any n_* is 0
    if (n_total is 0 ) or (n_lower is 0) or (n_higher is 0) or (n_pos_lower is 0) or (n_neg_lower is 0) or (n_pos_higher is 0) or (n_neg_higher is 0):
        return 0.0
 
    s_low = 0.0
    s_high = 0.0

    p_pos_low = float(n_pos_lower) / n_lower
    p_neg_low = float(n_neg_lower) / n_lower
    p_pos_high = float(n_pos_higher) / n_higher
    p_neg_high = float(n_neg_higher) / n_higher

    s_low += (p_pos_low * numpy.log(p_pos_low))
    s_low += (p_neg_low * numpy.log(p_neg_low))
    s_high += (p_pos_high * numpy.log(p_pos_high))
    s_high += (p_neg_high * numpy.log(p_neg_high))

    E_y_true = - s_low
    E_y_false = - s_high

    return E_y - float((E_y_true * n_lower) + (E_y_false * n_higher)) / n_total

