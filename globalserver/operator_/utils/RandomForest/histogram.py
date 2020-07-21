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

import json
import math


class Histogram:
    """
    Histogram represents the given input data as a histogram where for categorical data, each value has a unique bin,
    and for continuous data one has at maximum n_bins. When there are too many bins some bins get merged together.
    This Implementation is used in the Federated Learning Framework at the workers to 'summarize' the local data,
    this information is then sent to the global server.
    ::
    Note that there is one histogram object per feature.
    ::
    Note that a bin is represented by a dictionary, for easy conversion to json and back.
    ::
    A bin has the following fields:
    - bin_identifier: the value (categorical or continuous) that this bin represents
    - n_pos: number of positive samples in this bin
    - n_neg: number of negative samples in this bin
    ::
    Note that n_bins must be >= 2
    """

    def __init__(self, feature_index, gen=None, X=None, Y=None, steps_per_epoch=1, pos_label=1, neg_label=0,
                 is_cont=True, n_bins=100, hist=None):
        self.feature_index = feature_index
        self.pos_label = pos_label
        self.neg_label = neg_label
        self.is_cont = is_cont
        self.n_bins = n_bins
        # build the histogram, if a generator is provided, use this one, else use the data X,Y
        # use dict as bin, because can be updated and easily transformed to json
        assert ((gen is not None) or (X is not None and Y is not None) or (hist is not None))
        if gen is not None:
            self.hist = self._build_hist_from_gen(gen, steps_per_epoch, neg_label, pos_label, is_cont, n_bins)
        elif X is not None and Y is not None:
            self.hist = self._build_hist_from_XY(X, Y, neg_label, pos_label, is_cont, n_bins)
        elif hist is not None:
            self.hist = hist

    def to_json(self):
        hist_dict = {
            'feature_index': self.feature_index,
            'pos_label': self.pos_label,
            'neg_label': self.neg_label,
            'is_cont': self.is_cont,
            'n_bins': self.n_bins,
            'hist': self.hist,
        }
        return json.dumps(hist_dict)

    @staticmethod
    def from_json(json):
        return Histogram(
            feature_index=json['feature_index'],
            gen=None,
            X=None,
            Y=None,
            pos_label=json['pos_label'],
            neg_label=json['neg_label'],
            is_cont=json['is_cont'],
            n_bins=json['n_bins'],
            hist=json['hist']
        )

    def _build_hist_from_gen(self, gen, steps_per_epoch=1, neg_label=0, pos_label=1, is_cont=True, n_bins=100):
        histogram = []
        # For categorical data
        if self.is_cont is False:
            # gather all unique values in here
            unique_values = []
            # fill up bins and create bins on the fly

            for step in range(steps_per_epoch):
                el = next(gen())

                r_i = el[0][self.feature_index]
                y_i = el[1]
                p_i = 0
                n_i = 0
                if y_i == self.pos_label:
                    p_i = 1
                elif y_i == self.neg_label:
                    n_i = 1
                else:
                    # There must be a mistake in the initialization, either we have more than 3 labels,
                    # or the labels must have been initialized wrong!
                    assert (False)
                # add to existing bin if possible
                if r_i in unique_values:
                    extended = False
                    for bin_ in histogram:
                        if bin_['bin_identifier'] == r_i:
                            bin_['n_pos'] = bin_['n_pos'] + p_i
                            bin_['n_neg'] = bin_['n_neg'] + n_i
                            extended = True
                            break
                    # make sure that the bin has been extended
                    assert (extended is True)
                # else create new bin to append to the histogram
                else:
                    curr_bin = {
                        'bin_identifier': r_i,
                        'n_pos': p_i,
                        'n_neg': n_i,
                    }
                    histogram.append(curr_bin)
                    histogram.sort(key=lambda x: x['bin_identifier'])
            return histogram
        # for continuous data
        else:
            for step in range(steps_per_epoch):
                el = next(gen())
                r_i = el[0][self.feature_index]
                y_i = el[1]
                p_i = 0
                n_i = 0
                if y_i == self.pos_label:
                    p_i = 1
                elif y_i == self.neg_label:
                    n_i = 1
                else:
                    # There must be a mistake in the initialization, either we have more than 3 labels,
                    # or the labels must have been initialized wrong!
                    assert (False)
                current_bin = {
                    'bin_identifier': r_i,
                    'n_pos': p_i,
                    'n_neg': n_i,
                }
                # try to add current information to existing bin if possible
                extended = False
                for bin_ in histogram:
                    if math.isclose(bin_['bin_identifier'], r_i, rel_tol=1e-10):
                        bin_['bin_identifier'] = r_i
                        bin_['n_pos'] = bin_['n_pos'] + p_i
                        bin_['n_neg'] = bin_['n_neg'] + n_i
                        extended = True
                        break
                if not extended:
                    histogram.append(current_bin)
                    histogram.sort(key=lambda x: x['bin_identifier'])
                # compress histogram by combining bins if needed
                while (len(histogram) > self.n_bins):
                    assert (self.n_bins >= 2)
                    # find two closest bins
                    idx_right = 1
                    min_dist = abs(histogram[1]['bin_identifier'] - histogram[0]['bin_identifier'])
                    for j in range(2, len(histogram)):
                        curr_dist = abs(histogram[j]['bin_identifier'] - histogram[j - 1]['bin_identifier'])
                        if curr_dist < min_dist:
                            min_dist = curr_dist
                            idx_right = j
                    # combine two closest bins
                    right_bin = histogram.pop(idx_right)
                    r_l = histogram[idx_right - 1]['bin_identifier']
                    p_l = histogram[idx_right - 1]['n_pos']
                    n_l = histogram[idx_right - 1]['n_neg']
                    r_r = right_bin['bin_identifier']
                    p_r = right_bin['n_pos']
                    n_r = right_bin['n_neg']
                    histogram[idx_right - 1]['bin_identifier'] = ((p_l + n_l) * r_l + (p_r + n_r) * r_r) / (
                                p_l + p_r + n_l + n_r)
                    histogram[idx_right - 1]['n_pos'] = p_l + p_r
                    histogram[idx_right - 1]['n_neg'] = n_l + n_r

            return histogram

    def _build_hist_from_XY(self, X, Y, neg_label=0, pos_label=1, is_cont=True, n_bins=100):
        n_samples = X.shape[0]
        assert (n_samples == Y.shape[0])

        histogram = []
        # For categorical data
        if self.is_cont is False:
            # get all unique values for this feature and make one bin per value
            unique_values = sorted(set(X[:, self.feature_index]))
            # initialize empty bins and index structure
            index = dict()
            for i in range(len(unique_values)):
                curr_bin = {
                    'bin_identifier': unique_values[i],
                    'n_pos': 0,
                    'n_neg': 0,
                }
                histogram.append(curr_bin)
                index[unique_values[i]] = i
            # fill up bins
            for i in range(n_samples):
                r_i = X[i, self.feature_index]
                y_i = Y[i]
                p_i = 0
                n_i = 0
                if y_i == self.pos_label:
                    p_i = 1
                elif y_i == self.neg_label:
                    n_i = 1
                else:
                    # There must be a mistake in the initialization, either we have more than 3 labels,
                    # or the labels must have been initialized wrong!
                    assert (False)
                assert (histogram[index[r_i]]['bin_identifier'] == r_i)
                # add values to histogram
                histogram[index[r_i]]['n_pos'] = histogram[index[r_i]]['n_pos'] + p_i
                histogram[index[r_i]]['n_neg'] = histogram[index[r_i]]['n_neg'] + n_i
            return histogram
        # for continuous data
        else:
            for i in range(n_samples):
                r_i = X[i, self.feature_index]
                y_i = Y[i]
                p_i = 0
                n_i = 0
                if y_i == self.pos_label:
                    p_i = 1
                elif y_i == self.neg_label:
                    n_i = 1
                else:
                    # There must be a mistake in the initialization, either we have more than 3 labels,
                    # or the labels must have been initialized wrong!
                    assert (False)
                current_bin = {
                    'bin_identifier': r_i,
                    'n_pos': p_i,
                    'n_neg': n_i,
                }
                # try to add current information to existing bin if possible
                extended = False
                for bin_ in histogram:
                    if math.isclose(bin_['bin_identifier'], r_i, rel_tol=1e-10):
                        bin_['bin_identifier'] = r_i
                        bin_['n_pos'] = bin_['n_pos'] + p_i
                        bin_['n_neg'] = bin_['n_neg'] + n_i
                        extended = True
                        break
                if not extended:
                    histogram.append(current_bin)
                    histogram.sort(key=lambda x: x['bin_identifier'])
            # compress histogram by combining bins if needed
            while (len(histogram) > self.n_bins):
                assert (self.n_bins >= 2)
                # find two closest bins
                idx_right = 1
                min_dist = abs(histogram[1]['bin_identifier'] - histogram[0]['bin_identifier'])
                for j in range(2, len(histogram)):
                    curr_dist = abs(histogram[j]['bin_identifier'] - histogram[j - 1]['bin_identifier'])
                    if curr_dist < min_dist:
                        min_dist = curr_dist
                        idx_right = j
                # combine two closest bins
                right_bin = histogram.pop(idx_right)
                r_l = histogram[idx_right - 1]['bin_identifier']
                p_l = histogram[idx_right - 1]['n_pos']
                n_l = histogram[idx_right - 1]['n_neg']
                r_r = right_bin['bin_identifier']
                p_r = right_bin['n_pos']
                n_r = right_bin['n_neg']
                histogram[idx_right - 1]['bin_identifier'] = ((p_l + n_l) * r_l + (p_r + n_r) * r_r) / (
                            p_l + p_r + n_l + n_r)
                histogram[idx_right - 1]['n_pos'] = p_l + p_r
                histogram[idx_right - 1]['n_neg'] = n_l + n_r

            return histogram

    def merge_with_histogram(self, histogram_to_merge, B_bins=None):
        """Current Object gets updated by adding all bins together from other histogram.
        ::
        B_bins is by default set to self.n_bins
        """
        if B_bins is None:
            B_bins = self.n_bins
        assert (self.feature_index == histogram_to_merge.feature_index)
        # for categorical histogram
        if self.is_cont is False:
            # add bins from histogram_to_merge to 
            # sort the two histograms to apply in-situ merge-sort approach to merging the histograms
            self.hist.sort(key=lambda x: x['bin_identifier'])
            histogram_to_merge.hist.sort(key=lambda x: x['bin_identifier'])
            left_index = 0  # index for self histogram list
            right_index = 0  # index for histogram_to_merge histograms
            new_bin_list = []
            while right_index < len(histogram_to_merge.hist) and left_index < len(self.hist):
                if histogram_to_merge.hist[right_index]['bin_identifier'] == self.hist[left_index]['bin_identifier']:
                    self.hist[left_index]['n_pos'] = self.hist[left_index]['n_pos'] + \
                                                     histogram_to_merge.hist[right_index]['n_pos']
                    self.hist[left_index]['n_neg'] = self.hist[left_index]['n_neg'] + \
                                                     histogram_to_merge.hist[right_index]['n_pos']
                    left_index = left_index + 1
                    right_index = right_index + 1
                elif histogram_to_merge.hist[right_index]['bin_identifier'] < self.hist[left_index]['bin_identifier']:
                    new_bin = {
                        'bin_identifier': histogram_to_merge.hist[right_index]['bin_identifier'],
                        'n_pos': histogram_to_merge.hist[right_index]['n_pos'],
                        'n_neg': histogram_to_merge.hist[right_index]['n_neg'],
                    }
                    new_bin_list.append(new_bin)
                    right_index = right_index + 1
                elif histogram_to_merge.hist[right_index]['bin_identifier'] > self.hist[left_index]['bin_identifier']:
                    left_index = left_index + 1
            while right_index < len(histogram_to_merge.hist):
                new_bin = {
                    'bin_identifier': histogram_to_merge.hist[right_index]['bin_identifier'],
                    'n_pos': histogram_to_merge.hist[right_index]['n_pos'],
                    'n_neg': histogram_to_merge.hist[right_index]['n_neg'],
                }
                new_bin_list.append(new_bin)
                right_index = right_index + 1

            # extend the local histogram by the new_bin_list
            self.hist.extend(new_bin_list)

        # for continuous histogram
        else:
            # try to extend existing bins, if not possible, add a new bin
            for bin_ in histogram_to_merge.hist:
                extended = False
                for extend_bin_ in self.hist:
                    if math.isclose(extend_bin_['bin_identifier'], bin_['bin_identifier'], rel_tol=1e-10):
                        extend_bin_['bin_identifier'] = ((bin_['n_pos'] + bin_['n_neg']) * bin_['bin_identifier'] + (
                                    extend_bin_['n_pos'] + extend_bin_['n_neg']) * extend_bin_['bin_identifier']) / (
                                                                    bin_['n_pos'] + bin_['n_neg'] + extend_bin_[
                                                                'n_pos'] + extend_bin_['n_neg'])
                        extend_bin_['n_pos'] = extend_bin_['n_pos'] + bin_['n_pos']
                        extend_bin_['n_neg'] = extend_bin_['n_neg'] + bin_['n_neg']
                        extended = True
                        break
                if not extended:
                    self.hist.append(bin_)
                    self.hist.sort(key=lambda x: x['bin_identifier'])
            # compress histogram by combining bins if needed
            while (len(self.hist) > self.n_bins):
                assert (self.n_bins >= 2)
                # find two closest bins
                idx_right = 1
                min_dist = abs(self.hist[1]['bin_identifier'] - self.hist[0]['bin_identifier'])
                for j in range(2, len(self.hist)):
                    curr_dist = abs(self.hist[j]['bin_identifier'] - self.hist[j - 1]['bin_identifier'])
                    if curr_dist < min_dist:
                        min_dist = curr_dist
                        idx_right = j
                # combine two closest bins
                right_bin = self.hist.pop(idx_right)
                r_l = self.hist[idx_right - 1]['bin_identifier']
                p_l = self.hist[idx_right - 1]['n_pos']
                n_l = self.hist[idx_right - 1]['n_neg']
                r_r = right_bin['bin_identifier']
                p_r = right_bin['n_pos']
                n_r = right_bin['n_neg']
                self.hist[idx_right - 1]['bin_identifier'] = ((p_l + n_l) * r_l + (p_r + n_r) * r_r) / (
                            p_l + p_r + n_l + n_r)
                self.hist[idx_right - 1]['n_pos'] = p_l + p_r
                self.hist[idx_right - 1]['n_neg'] = n_l + n_r

    @staticmethod
    def merge_histograms(histogram_list, B_bins=100):
        """Function returns a new histogram object that contains all merged information.
        ::
        Note that B_bins is only used for categorical data.
        """
        assert (len(histogram_list) >= 1)
        # make sure all histograms match
        feature_idx = histogram_list[0].feature_index
        is_continuous = histogram_list[0].is_cont
        for h in histogram_list:
            assert (h.feature_index == feature_idx)
            assert (h.is_cont == is_continuous)

        # create empty histogram to then merge with all other histograms
        histogram = Histogram(
            feature_index=feature_idx,
            pos_label=histogram_list[0].pos_label,
            neg_label=histogram_list[0].neg_label,
            is_cont=is_continuous,
            n_bins=B_bins,
            hist=[]
        )
        # continuously merge this histogram with all histograms from the histogram_list
        for h in histogram_list:
            histogram.merge_with_histogram(h, B_bins)
        return histogram

    @staticmethod
    def get_mode_from_histograms(hist_dict, feature_indices, pos_label, neg_label):
        """Function that takes a histogram_dictionary, that is indexable by
        ``f"{f_i}"``, where ``f_i``is the index of the feature, and all indices
        given by ``feature_indices`` are present in this dictionary, and
        returns the label (pos_label or neg_label) that occurs the most in
        all presented histograms.
        ::
        Same functionality is used in RF_aggregate.
        """
        n_pos = 0
        n_neg = 0
        for f_i in feature_indices:
            for bin_ in hist_dict[f"{f_i}"].hist:
                n_pos = n_pos + bin_['n_pos']
                n_neg = n_neg + bin_['n_neg']

        if n_pos > n_neg:
            return pos_label
        else:
            return neg_label
