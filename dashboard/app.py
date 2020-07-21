# Import required libraries
import pickle
import copy
import pathlib
import dash
import dash_auth
import math
import datetime as dt
import pandas as pd
from dash.dependencies import Input, Output, State, ClientsideFunction
import dash_core_components as dcc
import dash_html_components as html
from bson import ObjectId
import json
import logging
from flask import request
import dash_table
from pymongo import MongoClient

logging.basicConfig(
    level=0,
    format=f"%(asctime)s [%(processName)-12.12s] [%(levelname)-5.5s] [test] [%(filename)s / %(funcName)s / %(lineno)d] %(message)s")

# get relative data folder


db_config = json.load(
    open("globalserver/db_conf.key", 'r'))
client = MongoClient(port=int(db_config['port']), username=db_config['user'], password=db_config['password'])
fl_db = client.federated_learning


def get_experiments():
    result = fl_db.experiment.find({"task_list": {"$exists": True}})
    experiments = pd.DataFrame(list(result))
    experiments=experiments[experiments['task_list'].apply(lambda x: len(x))>0]
    experiments['working_task'] = experiments['task_list'].apply(
        lambda x: next((task for task in x if task["task_status"] in ['scheduled', 'not_scheduled']), x[-1]))
    experiments['number_of_tasks'] = experiments['task_list'].apply(lambda x: len(x))
    experiments = experiments.drop(columns=['task_list'])
    return experiments


def get_datatable(experiments):
    # table
    working_task = experiments['working_task'].apply(pd.Series)
    dataframe = experiments.copy()
    dataframe['working_task'] = working_task['task_name']
    dataframe['progress'] = (working_task['task_order'] + 1).map(str) + "/" + dataframe['number_of_tasks'].map(str)
    try:
        dataframe['is_finished'] = dataframe['is_finished'].fillna(False)
    except KeyError:
        dataframe['is_finished'] = dataframe.apply(lambda x: False)

    try:
        dataframe['has_failed'] = dataframe['has_failed'].fillna(False)
    except KeyError:
        dataframe['has_failed'] = dataframe.apply(lambda x: False)

    try:
        dataframe['is_running'] = dataframe['is_running'].fillna(False)
    except KeyError:
        dataframe['is_running'] = dataframe.apply(lambda x: False)
    try:
        dataframe['experiment_start_time'] = dataframe['experiment_start_time']
    except KeyError:
        dataframe['experiment_start_time'] = dataframe['timestamp']
    try:

        dataframe['status'] = dataframe.apply(
            lambda x: 'failed' if x['has_failed'] else 'finished' if x['is_finished'] else 'running' if x[
                'is_running'] else 'not_started', axis=1)

        dataframe['has_validation_results'] = dataframe['validation_results'].apply(
            lambda x: True if type(x) == dict and len(list(list(x.values())[-1].values())[0]) > 0 else False)
        dataframe['has_training_results'] = dataframe['training_results'].apply(
            lambda x: True if type(x) == dict and len(list(list(x.values())[-1].values())[0]) > 0 else False)
        dataframe['has_test_results'] = dataframe['test_results'].apply(
            lambda x: True if type(x) == dict and len(list(list(x.values())[-1].values())[0]) > 0 and (
                    type(list(list(x.values())[-1].values())[0]) != dict or len(
                list(list(x.values())[-1].values())[0].get('aggregated_metric', [1, 2, 3])) > 0) else False)
        dataframe['has_results'] = dataframe.apply(lambda x: True if (
                x['has_validation_results'] or x['has_training_results'] or x['has_test_results']) else False, axis=1)
    except Exception as error:
        print("Aggregating results failed")
        print(error)
        dataframe['has_results'] = dataframe.apply(lambda x: False)

    dataframe["id"] = dataframe['_id']
    dataframe.set_index('id', inplace=True, drop=False)
    dataframe = dataframe[
        ['id', 'experiment_name', 'experiment_description', 'protocol', 'working_task', 'progress', 'status',
         'has_results',
         'experiment_start_time', ]].applymap(str)
    dataframe = dataframe.sort_values('experiment_start_time', ascending=False)
    return dataframe


app = dash.Dash(
    __name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}]
)
server = app.server

try:
    VALID_USERNAME_PASSWORD_PAIRS = json.load(open("dashboard/credentials.cred"))

    auth = dash_auth.BasicAuth(
        app,
        VALID_USERNAME_PASSWORD_PAIRS
    )
except FileNotFoundError:
    print("no authentification activated")
# Create global chart template
mapbox_access_token = "pk.eyJ1IjoiamFja2x1byIsImEiOiJjajNlcnh3MzEwMHZtMzNueGw3NWw5ZXF5In0.fk8k06T96Ml9CLGgKmk81w"
experiments = get_experiments()
dataframe = get_datatable(experiments)


# Create app layout

def serve_layout():
    body = html.Div(
        [
            dcc.Store(id="aggregated_data"),
            dcc.Interval(
                id='interval-component',
                interval=5 * 1000,  # in milliseconds
                n_intervals=0
            ),
            # empty Div to trigger javascript file for graph resizing
            html.Div(id="output-clientside"),
            html.Div(
                [
                    html.Div(
                        [
                            html.Img(
                                src=app.get_asset_url("MTC_Wide.png"),
                                id="plotly-image",
                                style={
                                    "height": "60px",
                                    "width": "auto",
                                    "margin-bottom": "25px",
                                },
                            )
                        ],
                        className="one-third column",
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H3(
                                        "Federated Learning at MTC",
                                        style={"margin-bottom": "0px"},
                                    ),
                                    html.H5(
                                        "Experiment Overview and Learning Curves", style={"margin-top": "0px"}
                                    ),
                                ]
                            )
                        ],
                        className="one-half column",
                        id="title",
                    ),
                    html.Div(
                        [
                            html.A(
                                html.Button("Learn More", id="learn-more-button"),
                                href="https://mtc.ethz.ch/",
                            )
                        ],
                        className="one-third column",
                        id="button",
                    ),
                ],
                id="header",
                className="row flex-display",
                style={"margin-bottom": "25px"},
            ),

            html.Div(
                [
                    dash_table.DataTable(
                        id='experiments_table',
                        columns=[{"name": i, "id": i} for i in dataframe.columns],
                        data=dataframe.to_dict('records'),
                        filter_action="native",
                        sort_action="native",
                        page_action='native',
                        filter_query='{has_results} = True',
                        page_current=0,
                        page_size=10,
                        style_table={'overflowX': 'scroll'},
                        style_cell_conditional=[
                            {
                                'if': {'column_id': c},
                                'textAlign': 'left'
                            } for c in ['Date', 'Region']
                        ],
                        style_data_conditional=[
                            {
                                'if': {'row_index': 'odd'},
                                'backgroundColor': 'rgb(248, 248, 248)'
                            }
                        ],
                        style_header={
                            'backgroundColor': '#F1E6DC',
                            'borderColor': '#ED365B',
                            'fontWeight': 'bold'
                        }
                    ),

                ],
                id="table-container",
                className="row flex-display", ),
            html.Div([
                html.H5(
                    "Final Results", style={"margin-top": "0px"}
                ),
                html.Div(id="final-graph-container",
                         style={'display': 'flex',
                                "justify-content": "center",

                                "margin": "-30px",
                                "flex-wrap": "wrap",
                                'flex-direction': 'row'},
                         className="results row flex-display",
                         ),
                html.H5(
                    "Results per Iteration", style={"margin-top": "0px"}
                ),
                html.Div(id="training-graph-container",
                         style={'display': 'flex',
                                "justify-content": "center",

                                "margin": "-30px",
                                "flex-wrap": "wrap",
                                'flex-direction': 'row'},
                         className="row flex-display",
                         ),
            ],
                id="resultsContainter",
                className="results column flex-display"
            )
        ],
        id="mainContainer",
        style={"display": "flex", "flex-direction": "column"},
    )
    return body


app.layout = serve_layout


@app.callback(
    Output("experiments_table", "data"),
    [Input('interval-component', 'n_intervals')])
def update_metrics(n):
    global dataframe
    global experiments
    experiments = get_experiments()
    dataframe = get_datatable(experiments)
    return dataframe.to_dict('records')


def get_results(experiment_id, experiments):
    experiment = experiments[experiments['_id'] == ObjectId(experiment_id)].iloc[0]

    try:
        user = request.authorization['username']
    except RuntimeError:
        user = ''

    results = {}
    if len(experiment) > 0:
        clients = experiment["clients"] + ["aggregated_metric"]
        for data_type in ['training', 'validation', 'test']:
            for client in clients:

                if user != client and user != 'admin':
                    continue

                elif client == "aggregated_metric":
                    results[f'{data_type}_results_{client}'] = pd.DataFrame() if type(
                        experiment[f'{data_type}_results']) != dict else pd.DataFrame(
                        {task_id: result[client] for task_id, result in
                         experiment[f'{data_type}_results'].items() if
                         client in result and result[client] != []}).T

                else:
                    results[f'{data_type}_results_{client}'] = pd.DataFrame() if type(
                        experiment[f'{data_type}_results']) != dict else pd.DataFrame(
                        {task_id: json.loads(result[client]) for task_id, result in
                         experiment[f'{data_type}_results'].items() if
                         client in result}).T

    return experiment, results


def get_final_result(experiment_id, experiments):
    experiment = experiments[experiments['_id'] == ObjectId(experiment_id)].iloc[0]

    try:
        user = request.authorization['username']
    except RuntimeError:
        user = ''

    results = {}
    if len(experiment) > 0:
        clients = experiment["clients"] + ["aggregated_metric"]
        for data_type in ['training', 'validation', 'test']:
            for client in clients:

                if user != client and user != 'admin':
                    continue
                if client == "aggregated_metric":

                    results[f'{data_type}_results_{client}'] = [] if type(
                        experiment[f'{data_type}_results']) != dict else [result[client] for task_id, result in
                                                                          experiment[f'{data_type}_results'].items() if
                                                                          client in result]
                else:
                    results[f'{data_type}_results_{client}'] = [] if type(
                        experiment[f'{data_type}_results']) != dict else [json.loads(result[client]) for task_id, result
                                                                          in
                                                                          experiment[f'{data_type}_results'].items() if
                                                                          client in result]
    for client, result in results.items():
        if len(result) > 0:
            results[client] = results[client][-1]
        else:

            results[client] = 0
    return experiment, results


layout = dict(
    autosize=True,

    width=600,
    automargin=True,
    margin=dict(l=30, r=30, b=20, t=40),
    hovermode="closest",

    plot_bgcolor="#FFFFFF",
    paper_bgcolor="#FFFFFF",
    # legend=dict(font=dict(size=10), orientation="h"),
    legend=dict(orientation="h",
                x=0,
                y=-0.25,
                traceorder="normal",
                font=dict(
                    size=10,
                ),
                # bordercolor="Black",
                # borderwidth=2
                ),
    title="Satellite Overview",
    xaxis={"automargin": True,
           "title": 'Round'},
    yaxis={"automargin": True,
           "title": 'Metric Score'},
    textfont=dict(
        color="#343D3A"
    ),
    mapbox=dict(
        accesstoken=mapbox_access_token,
        style="light",
        center=dict(lon=-78.05, lat=42.54),
        zoom=7,
    ),
)


# Selectors, main graph -> aggregate graph
@app.callback(
    Output("final-graph-container", "children"),
    [
        Input('experiments_table', 'active_cell'),
    ],
)
def make_aggregate_figure(active_cell, ):
    if active_cell:
        experiments_id = active_cell.get('row_id', None)
    else:
        return None

    experiment, results = get_final_result(experiments_id, experiments)
    data = {}
    metrics = experiment['training_config'].get('skmetrics', []) + experiment['training_config'].get('tfmetrics', [])
    graphs = []
    colors = ["#4e79a7", "#59a14f", "#e15759", "#76b7b2", "#59a14f", "#edc949", "#af7aa1", "#ff9da7", "#9c755f",
              "#bab0ab", "#f28e2c","#4e79a7", "#59a14f", "#e15759", "#76b7b2", "#59a14f", "#edc949", "#af7aa1", "#ff9da7",
              "#9c755f", "#bab0ab","#f28e2c"]
    # heatmap=["rgb(186, 176, 171)","rgb(237,201,73)","rgb(175,122,161)","rgb(118,183,178)"]
    line_types = ["", "dash", "dot"]
    line_zip = {key: line_types[i] for i, key in enumerate(["test", "validation", "training"])}
    color_zip = {key: colors[i] for i, key in enumerate(experiment['clients'] + ["aggregated"])}
    for metric in metrics:
        data[metric] = []
        data_clients = []
        for result_type, result in results.items():
            if type(result) == dict and metric in result:

                if metric == 'roc_curve' and type(result[metric]) == list:
                    client = result_type.split("_")[2]
                    line_type = result_type.split("_")[0]

                    line_dict = dict(
                        type="scatter",
                        mode="lines",

                        marker=dict(color=color_zip[client]),
                        name=result_type,
                        x=result[metric][0],
                        y=result[metric][1],
                        line=dict(shape="spline", smoothing="0", dash=line_zip[line_type]),
                    )
                    data[metric].append(line_dict)
                elif metric == 'confusion_matrix':

                    data[metric].append((result_type,result[metric]))
                else:
                    client = result_type.split("_")[2]
                    if client in data_clients:
                        showlegend = False
                    else:
                        data_clients.append(client)
                        showlegend = True

                    line_dict = dict(
                        type="bar",
                        name=client,
                        showlegend=showlegend,

                        marker=dict(color=color_zip[client]),
                        x=[result_type.split("_")[0]],
                        y=[result[metric]],
                    )
                    data[metric].append(line_dict)

        layout_aggregate = copy.deepcopy(layout)
        layout_aggregate["title"] = f"Aggregate: {metric}"
        if metric == 'roc_curve':

            layout_aggregate['xaxis'] = {"automargin": True,
                                         "title": 'fpr'}
            layout_aggregate['yaxis'] = {"automargin": True,
                                         "title": 'tpr', "scaleanchor": "x", "scaleratio": 1}

            layout_aggregate['height'] = 700

        elif metric == 'confusion_matrix':
            for client in experiment['clients']:
                layout_aggregate["title"]=f"Confusion Matrix {client}"
                layout_aggregate["xaxis"] = {"automargin": True, "title": "Predicted value"}
                layout_aggregate["yaxis"] = {"automargin": True, "title": "Real value"}
                layout_aggregate["annotations"] = [
                    {
                        "x": "0",
                        "y": "0",
                        "font": {
                            "color": "white"
                        },
                        "text": "\n".join([f"{x[0].split('_')[0]}: {x[1][0][0]}<br>" for x in data[metric] if client in x[0]]),
                        "xref": "x1",
                        "yref": "y1",
                        "showarrow": False
                    },
                    {
                        "x": "1",
                        "y": "0",
                        "font": {
                            "color": "white"
                        },
                        "text": "\n".join([f"{x[0].split('_')[0]}: {x[1][1][0]}<br>" for x in data[metric] if client in x[0]]),
                        "xref": "x1",
                        "yref": "y1",
                        "showarrow": False
                    },
                    {
                        "x": "0",
                        "y": "1",
                        "font": {
                            "color": "white"
                        },
                        "text": "\n".join([f"{x[0].split('_')[0]}: {x[1][0][1]}<br>" for x in data[metric]if client in x[0]]),
                        "xref": "x1",
                        "yref": "y1",
                        "showarrow": False
                    },
                    {
                        "x": "1",
                        "y": "1",
                        "font": {
                            "color": "white"
                        },
                        "text":"\n".join([f"{x[0].split('_')[0]}: {x[1][1][1]}<br>" for x in data[metric] if client in x[0]]),
                        "xref": "x1",
                        "yref": "y1",
                        "showarrow": False
                    },
                ]
                import colorsys
                value = color_zip[client].lstrip('#')
                amount=1.1
                lv = len(value)
                c =tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))
                c = colorsys.rgb_to_hls(c[0],c[1],c[2])
                c=colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])
                heatmap = [{
                    "type": "heatmap",
                    "x": ["class 0", "class 1"],
                    "y": ["class 0", " class 1"],
                    "z": [[1, 0], [0, 1]],
                    "colorscale": [

                        [0, f"rgb{c}"],

                        [1, color_zip[client]]
                    ],
                    "showscale": False,

                }]


                if client != experiment['clients'][-1]:
                    figure = dict(data=heatmap, layout=layout_aggregate)

                    # print(figure)
                    graphs.append(html.Div([dcc.Graph(figure=figure,
                                                      style={"border": "1px solid #343D3A", "margin": "30px",
                                                             "padding": "5px"})],
                                           style={}))
            data[metric]=heatmap
        elif metric in  ['Accuracy']:
            continue


        figure = dict(data=data[metric], layout=layout_aggregate)

        # print(figure)
        graphs.append(html.Div([dcc.Graph(figure=figure,
                                          style={"border": "1px solid #343D3A", "margin": "30px", "padding": "5px"})],
                               style={}))

    # logging.debug(data)

    # make beautyful
    return graphs


# Selectors, main graph -> aggregate graph
@app.callback(
    Output("training-graph-container", "children"),
    [
        Input('experiments_table', 'active_cell'),
    ],
)
def make_aggregate_figure(active_cell, ):
    if active_cell:
        experiments_id = active_cell.get('row_id', None)
    else:
        return None

    experiment, results = get_results(experiments_id, experiments)
    data = {}
    metrics = experiment['training_config'].get('skmetrics', []) + experiment['training_config'].get('tfmetrics', [])
    graphs = []
    colors = ["#4e79a7", "#59a14f", "#e15759", "#76b7b2", "#59a14f", "#edc949", "#af7aa1", "#ff9da7", "#9c755f",
              "#bab0ab", "#f28e2c","#4e79a7", "#59a14f", "#e15759", "#76b7b2", "#59a14f", "#edc949", "#af7aa1", "#ff9da7",
              "#9c755f", "#bab0ab","#f28e2c"]
    line_types = ["", "dash", "dot"]
    symbol_types = ["circle", "cross", "triangle-up"]
    line_zip = {key: line_types[i] for i, key in enumerate(["test", "validation", "training"])}
    color_zip = {key: colors[i] for i, key in enumerate(experiment['clients'] + ["aggregated"])}
    symbol_zip = {key: symbol_types[i] for i, key in enumerate(["test", "validation", "training"])}
    # print(results)
    for metric in metrics:
        data[metric] = []
        for result_type, result in results.items():
            if metric in result:

                if metric == 'confusion_matrix':

                    line_dict = dict(
                        type="scatter",
                        mode="lines",
                        name=result_type + " TN",
                        x=result.index.to_list(),
                        y=[x[0][0] for x in result[metric].to_list()],
                        line=dict(shape="spline", smoothing="0"),
                    )
                    data[metric].append(line_dict)
                    line_dict = dict(
                        type="scatter",
                        mode="lines",
                        name=result_type + " FN",
                        x=result.index.to_list(),
                        y=[x[0][1] for x in result[metric].to_list()],
                        line=dict(shape="spline", smoothing="0"),
                    )
                    data[metric].append(line_dict)
                    line_dict = dict(
                        type="scatter",
                        mode="lines",
                        name=result_type + " FP",
                        x=result.index.to_list(),
                        y=[x[1][0] for x in result[metric].to_list()],
                        line=dict(shape="spline", smoothing="0"),
                    )
                    data[metric].append(line_dict)
                    line_dict = dict(
                        type="scatter",
                        mode="lines",
                        name=result_type + " TP",
                        x=result.index.to_list(),
                        y=[x[1][1] for x in result[metric].to_list()],
                        line=dict(shape="spline", smoothing="0"),
                    )
                    data[metric].append(line_dict)
                elif metric in ['roc_curve',"Accuracy"]:
                    pass
                elif len(result) > 1:

                    client = result_type.split("_")[2]
                    line_type = result_type.split("_")[0]

                    line_dict = dict(
                        type="scatter",
                        mode="lines",

                        marker=dict(color=color_zip[client]),
                        name=result_type,
                        x=result.index.to_list(),
                        y=result[metric].to_list(),
                        line=dict(shape="spline", smoothing="0", dash=line_zip[line_type]),
                    )
                    data[metric].append(line_dict)

                else:

                    client = result_type.split("_")[2]
                    line_type = result_type.split("_")[0]

                    line_dict = dict(
                        type="scatter",

                        marker=dict(color=color_zip[client], symbol=symbol_zip[line_type]),
                        name=result_type,
                        x=result.index.to_list(),
                        y=result[metric].to_list(),
                    )
                    data[metric].append(line_dict)

        layout_aggregate = copy.deepcopy(layout)
        layout_aggregate["title"] = f"Aggregate: {metric}"

        figure = dict(data=data[metric], layout=layout_aggregate)

        if metric in ['roc_curve', "Accuracy"]:
            continue
        graphs.append(html.Div([dcc.Graph(figure=figure,
                                          style={})],
                               style={"border": "1px solid #343D3A", "margin": "30px", "padding": "5px"}))

    # logging.debug(data)

    # make beautyful
    return graphs


# Main
if __name__ == "__main__":
    app.run_server(host= '0.0.0.0',debug=True)
