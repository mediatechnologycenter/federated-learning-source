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
Dummy Data Wrapper to stream training data to the node client

This scripts provides an endpoint callable by all containers in the docker network "webnet" via tcp.
On startup, it fetches the dummy train & test data from an S3 bucket and transforms it into jsonlines.
The the callable routes '/train' & '/test' stream the fetched data line by line (i.e. json by json).
"""

from flask import Response, Flask, jsonify, request
import json
import os
import boto3
import io
import random

import codecs


def get_s3_file(client, url):
    obj = client.get_object(Bucket='federated-learning-dummy-data', Key=url)
    body = obj['Body']

    data = []
    i = 0
    for ln in codecs.getreader('utf-8')(body):
        if i < 100000:
            data.append(json.loads(ln))
        i = i + 1
    return data


def is_float(val):
    try:
        num = [float(row) for row in val]
    except ValueError:
        return False
    return True


import numpy as np


def create_feature_metadata_json(data):
    feature_jsons = []
    columns = data[0].keys()
    for column in columns:
        values = [row[column] for row in data if column in row]

        uniques = list(set(values))
        if not is_float(values) or len(uniques) < 30:
            feature_json = {"feature": column,
                            "type": "categorical",
                            "categories": list(uniques)}
        else:
            q1 = np.quantile(values, 0.25)
            q3 = np.quantile(values, 0.75)
            iqr = 1.5 * (q3 - q1)
            mean = np.mean(values)
            std = np.std(values)

            feature_json = {"feature": column,
                            "type": "continuous",
                            "min_value": min(values),
                            "max_value": max(values),
                            "q1": q1,
                            "q3": q3,
                            "iqr_outliers": float(len([x for x in values if x < (q1 - iqr) or x > (q3 + iqr)]) )/ len(values),
                            "3std-percentage": float(len([x for x in values if x < (mean - 3 * std) or x > (mean + 3 * std)])) / len(values),
                            "mean": mean,
                            "std": std
                            }
        feature_jsons.append(feature_json)

    return feature_jsons


app = Flask(__name__)

s3 = boto3.client('s3', aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                  aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'), region_name='eu-central-1')

# Fetch Data
train = get_s3_file(s3, 'new/train_kkbox.jsonl')
train, validation = train[:int(len(train) * 0.9)], train[int(len(train) * 0.9):]

test = get_s3_file(s3, 'new/test_kkbox.jsonl')

metadata = {"identifier": "1",
            "description": "this is the first dataset",
            "samples_num": [len(train), len(validation), len(test)],
            "creation_date": "2020-02-25T08:40:44.000Z",
            'features': create_feature_metadata_json(train)}

available_data_sets_metadata = [metadata]
available_data_sets = {'1': {'train': train, 'test': test, 'validation': validation}}


# Route to stream the training data
@app.route('/get_available_datasets')
def stream_train_data():
    return jsonify(available_data_sets_metadata)


# Route to stream the training data
@app.route('/get_dataset', methods=['GET'])
def get_dataset():
    identifier = request.args.get('identifier')
    type = request.args.get('type')
    if identifier in available_data_sets and type in available_data_sets[identifier]:
        random.shuffle(available_data_sets[identifier][type])

        def generate():
            for row in available_data_sets[identifier][type]:
                yield json.dumps(row) + '\n'

        return Response(generate(), mimetype='text/json')

    return f"Dataset {identifier} not found", 400


if __name__ == '__main__':
    app.run(host='0.0.0.0')
