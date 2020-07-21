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

import json
import logging

import os

config = json.load(open(os.getenv("STATIC_VARIABLES_FILE_PATH", "static_variables.json"), 'r'))
try:
    secret_config = json.load(open(os.getenv("VALID_CLIENTS_FILE_PATH", "valid_clients.key"), 'r'))
except FileNotFoundError:
    secret_config={ "VALID_CLIENTS": [    "INTERFACE"  ]}
config['PATH_TO_GLOBALSERVER'] = os.getenv("PATH_TO_GLOBALSERVER", config['DEFAULT_GLOBAL_SERVER_PATH'])


# task utils


def get_task_from_experiment(db, experiment_id, client, db_session):
    from bson.objectid import ObjectId
    experiment_documents = list(db.experiment.find({"_id": experiment_id}).limit(1))
    if len(experiment_documents) == 0 or not experiment_documents[0]['is_running'] \
            or experiment_documents[0].get('is_finished', False):
        return None, ''

    experiment_document = experiment_documents[0]
    for task_fk in experiment_document["task_list"]:
        if task_fk["task_status"] != config['TASK_DONE']:
            break

    if task_fk['task_name'] == config['AGGREGATE_TASK'] or task_fk["task_status"] == config['TASK_DONE']:
        return None, experiment_document["protocol"]
    task_fk['task_id'] = ObjectId(task_fk['task_id'])

    task_document = get_task_from_id(db=db, task_id=task_fk['task_id'])
    if task_fk['task_status'] == config['NOT_SCHEDULED_STOPWORD']:
        schedule_task(db=db, task=task_fk, experiment_id=experiment_document['_id'],
                      clients=task_document['clients'].keys(), db_session=db_session)

    if client in task_document['clients'] and task_document and task_document['clients'][client]['status'] == config[
        'SCHEDULED_STOPWORD']:
        return task_document, experiment_document["protocol"]

    return None, experiment_document["protocol"]


def get_latest_models(db):
    model_documents = list(db.model.find({"is_final_model": True}))
    if len(model_documents) == 0:
        return ''
    else:
        for i, model_document in enumerate(model_documents):
            model_document.pop('model', None)
            model_document['_id'] = str(model_document['_id'])
            model_document['timestamp'] = str(model_document['timestamp'])
            model_documents[i] = model_document
        return model_documents


def load_global_model(db, experiment_id):
    experiment_document = list(db.experiment.find({"_id": experiment_id}).limit(1))
    if len(experiment_document) == 0:
        return None

    model_parameters_path = get_model_parameter_path(str(experiment_document[0]['experiment_state_model_id']))
    with open(model_parameters_path, 'r') as reader:
        model_parameters = reader.read().encode('utf-8')
    return json.loads(model_parameters)


def task_completion(db, task_id, experiment_id, client, result, db_session):
    task = list(db.task.find({"_id": task_id}).limit(1))
    if len(task) == 0:
        logging.warning(f"client {client} sent a task completion with invalid {task_id}")

        task_list_update = db.experiment.update({"_id": experiment_id},
                                                {"$set": {f"has_failed": True}})
        return False

    if client not in task[0]['clients'] or task[0]['clients'][client]['status'] != config["SCHEDULED_STOPWORD"]:
        logging.warning(f"client {client} finished wrong task: {task_id}")

        task_list_update = db.experiment.update({"_id": experiment_id},
                                                {"$set": {f"has_failed": True}})
        return False

    task_update = db.task.update({"_id": task_id},
                                 {"$set": {f"clients.{client}.status": config['TASK_DONE'],
                                           f"clients.{client}.result": result}})

    task = list(db.task.find({"_id": task_id}).limit(1))
    if all(client_status["status"] == config['TASK_DONE'] for client_key, client_status in task[0]['clients'].items() if
           client_key != client):
        task_list_update = db.experiment.update({"_id": experiment_id, "task_list.task_id": task_id},
                                                {"$set": {f"task_list.$.task_status": config['TASK_DONE']}})
    return True


def get_model_parameter_path(model_id):
    model_parameters_path = f"{config['PATH_TO_GLOBALSERVER']}{config['GLOBAL_MODELS_PATH']}{model_id}.json"
    backup_path = f"{config['PATH_TO_GLOBALSERVER']}{config['BACKUP_PATH']}{model_id}.json"
    os.makedirs(os.path.dirname(model_parameters_path), exist_ok=True)
    os.makedirs(os.path.dirname(backup_path), exist_ok=True)
    return model_parameters_path


def get_clients_response_path(experiment_id, task_id, client):
    model_parameters_path = f"{config['PATH_TO_GLOBALSERVER']}{config['LOCAL_MODELS_PATH']}{experiment_id}/{client}/{task_id}.json"
    os.makedirs(os.path.dirname(model_parameters_path), exist_ok=True)
    return model_parameters_path


# db functions
def get_db_connection():
    from pymongo import MongoClient
    db_config = json.load(open(config['PATH_TO_GLOBALSERVER'] + config['DB_CONFIG_FILE'], 'r'))
    client = MongoClient(port=int(db_config['port']), username=db_config['user'], password=db_config['password'])
    return client.federated_learning, client.start_session


def schedule_task(db, task, experiment_id, clients, db_session):
    from pymongo.errors import OperationFailure
    session = db_session()
    try:
        with session.start_transaction():

            task_list_update = db.experiment.update_one({"_id": experiment_id},
                                                        {"$set": {
                                                            f"task_list.{task['task_order']}.task_status": config[
                                                                "SCHEDULED_STOPWORD"]}},
                                                        session=session)
            if task['task_name'] in config['VALID_SERVER_TASKS']:
                clients = [config['AGGREGATOR_NAME']]

            task_update = db.task.update_one({"_id": task['task_id']},
                                             {"$set": {f"clients.{client}.status": config["SCHEDULED_STOPWORD"] for
                                                       client in clients}},
                                             session=session)
    except OperationFailure as error:
        logging.info("WriteConcern on Schedule_task")


def get_task_from_id(db, task_id):
    task = next(db.task.find({"_id": task_id}).limit(1), None)
    return task


def add_validation_result_to_experiment(db, loss, task_id, experiment_id, client):
    task_order = db.task.find_one({"_id": task_id})['task_order']

    task_list_update = db.experiment.update({"_id": experiment_id},
                                            {"$set": {f"validation_results.{task_order}.{client}": loss}})


def add_training_result_to_experiment(db, loss, task_id, experiment_id, client):
    task_order = db.task.find_one({"_id": task_id})['task_order']

    task_list_update = db.experiment.update({"_id": experiment_id},
                                            {"$set": {f"training_results.{task_order}.{client}": loss}})


def add_test_result_to_experiment(db, loss, task_id, experiment_id, client):
    task_order = db.task.find_one({"_id": task_id})['task_order']

    task_list_update = db.experiment.update({"_id": experiment_id},
                                            {"$set": {f"test_results.{task_order}.{client}": loss}})


# # other utils
def client_is_valid(client, secret):
    if (secret != os.getenv('CLIENT_SECRET', '')) and os.getenv('TESTING', '0') != '1':
        logging.warning(f"Client <{client}> tried to access with wrong secret!")
        return False
    if client == '' or (client not in secret_config['VALID_CLIENTS'] and os.getenv('TESTING', '0') != '1'):
        logging.warning(f"Client <{client}> tried to access!")
        return False
    return True
