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

from flask import Response, Flask
import json
import os
import boto3
import io
import random

app = Flask(__name__)

s3 = boto3.client('s3', aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                  aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'), region_name='eu-central-1')

import codecs


def get_s3_file(client, url):
    obj = client.get_object(Bucket='federated-learning-dummy-data', Key=url)
    body = obj['Body']

    data = []
    i = 0
    for ln in codecs.getreader('utf-8')(body):
        if i < 100000:
            data.append(ln)
        i=i+1
    return data



# Fetch Data
train = get_s3_file(s3, 'new/train_kkbox.jsonl')
train, validation = train[:int(len(train) * 0.9)], train[int(len(train) * 0.9):]

test = get_s3_file(s3, 'new/test_kkbox.jsonl')

# Route to stream the training data
@app.route('/train')
def stream_train_data():
    random.shuffle(train)

    def generate():
        for row in train:
            yield row + '\n'

    return Response(generate(), mimetype='text/json')


@app.route('/validation')
def stream_validation_data():
    def generate():
        for row in validation:
            yield row + '\n'

    return Response(generate(), mimetype='text/json')


# Route to stream the test data
@app.route('/test')
def stream_test_data():
    def generate():
        for row in test:
            yield row + '\n'

    return Response(generate(), mimetype='text/json')

if __name__ == '__main__':
    if os.getenv('local',False):
        app.run(host='0.0.0.0',port=8989)
    else:

        app.run(host='0.0.0.0')

