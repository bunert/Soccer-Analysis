import numpy as np
import os
from os import listdir
from os.path import isfile, join, exists
import soccer
import argparse
import utils.io as io
import utils.camera as cam_utils
from tqdm import tqdm
import plotly as py
import utils.draw2 as draw
from plotly.tools import FigureFactory as FF
import scipy

# CMD Line arguments
parser = argparse.ArgumentParser(description='Estimate the poses')
# --path_to_data: where the images are
parser.add_argument('--path_to_data', default='/home/bunert/Data/Left', help='path')

opt, _ = parser.parse_known_args()

# load corresponding metadata
db = soccer.SoccerVideo(opt.path_to_data)
db.digest_metadata()

# ------------------------------------------------------------------------------


# plot the field
data = []
draw.plot_field(data)

# plot the cameras (extension needed - hardcoded))
draw.plot_camera(data, db)

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

fig = dict(data=data, layout=layout)
py.offline.plot(fig, filename='camera.html')
