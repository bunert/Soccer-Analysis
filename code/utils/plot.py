import os
import argparse
import soccer
from os import listdir
from os.path import isfile, join, exists
import pandas
from scipy.linalg import block_diag, norm
import numpy as np
from filterpy.common import Q_discrete_white_noise
import camera as cam_utils
import draw2 as draw

import datetime
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly
from dash.dependencies import Input, Output

all_players_3d =

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.layout = html.Div(
    html.Div([
        html.H4('Soccer Analysis'),
        dcc.Graph(id='live-update-graph'),
        dcc.Interval(
            id='interval-component',
            interval=1*1000, # in milliseconds
            n_intervals=0
        )
    ])
)


# Multiple components can update everytime interval gets fired.
@app.callback(Output('live-update-graph', 'figure'),
              [Input('interval-component', 'n_intervals')])
def update_graph_live(n):
    # plot the field
    plot_data = []
    draw.plot_field(plot_data)

    rgb_color = 'rgb(0, 0, 0)'
    for i in players_3d:
        if (i == 11):
            rgb_color = 'rgb(255, 8, 0)' #different color for the different teams
        plot_player(players_3d[i], plot_data, rgb_color)

    # layout parameters
    layout = dict(
        width=1500,
        height=750,
        plot_bgcolor='rgb(0,0,0)',
        autosize=False,
        title='camera location',
        showlegend=False,
        margin=dict(
            r=0, l=10,
            b=0, t=30),
        scene=dict(
            xaxis=dict(
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(230, 230,230)',
                range=[-100, 100]
            ),
            yaxis=dict(
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(230, 230,230)',
                range=[0, 50]
            ),
            zaxis=dict(
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(230, 230,230)',
                range=[-100, 100]
            ),
            camera=dict(
                up=dict(
                    x=0,
                    y=1,
                    z=0
                ),
                eye=dict(
                    x=1.2,
                    y=0.7100,
                    z=1.2,
                )
            ),
            aspectratio=dict(x=1, y=0.25, z=1),
            aspectmode='manual'
        ),
    )

    fig = dict(data=plot_data, layout=layout)
    return fig

def plot_player(players, plot_data, rgb_color):
    limps = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [1, 11], [11, 12], [12, 13], [1, 8],
            [8, 9], [9, 10], [14, 15], [16, 17], [0, 14], [0, 15], [14, 16], [15, 17], [8, 11], [2, 8], [5, 11]])

    players = np.asmatrix(players)

    for i in range(len(limps)):
        plot_data.append(go.Scatter3d(x=[players[limps[i][0]][0,0], players[limps[i][1]][0,0]], y=[players[limps[i][0]][0,1], players[limps[i][1]][0,1]],
             z=[players[limps[i][0]][0,2], players[limps[i][1]][0,2]], mode='lines', line=dict(color=rgb_color, width=3)))



# # Multiple components can update everytime interval gets fired.
# @app.callback(Output('live-update-graph', 'figure'),
#               [Input('interval-component', 'n_intervals')])
# def update_graph_live2(n):
#     satellite = Orbital('TERRA')
#     data = {
#         'time': [],
#         'Latitude': [],
#         'Longitude': [],
#         'Altitude': []
#     }
#
#     # Collect some data
#     for i in range(180):
#         time = datetime.datetime.now() - datetime.timedelta(seconds=i*20)
#         lon, lat, alt = satellite.get_lonlatalt(
#             time
#         )
#         data['Longitude'].append(lon)
#         data['Latitude'].append(lat)
#         data['Altitude'].append(alt)
#         data['time'].append(time)
#
#     # Create the graph with subplots
#     fig = plotly.tools.make_subplots(rows=2, cols=1, vertical_spacing=0.2)
#     fig['layout']['margin'] = {
#         'l': 30, 'r': 10, 'b': 30, 't': 10
#     }
#     fig['layout']['legend'] = {'x': 0, 'y': 1, 'xanchor': 'left'}
#
#     fig.append_trace({
#         'x': data['time'],
#         'y': data['Altitude'],
#         'name': 'Altitude',
#         'mode': 'lines+markers',
#         'type': 'scatter'
#     }, 1, 1)
#
#
#     return fig


if __name__ == '__main__':
    app.run_server(debug=True)
