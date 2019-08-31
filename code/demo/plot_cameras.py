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
import utils.draw as draw
from plotly.tools import FigureFactory as FF
import scipy
import plotly.graph_objs as go

################################################################################
# run: python3 plot_cameras.py --path_to_data ~/path/to/openpose
################################################################################



# CMD Line arguments
parser = argparse.ArgumentParser(description='Estimate the poses')
# --path_to_data: where the images are
parser.add_argument('--path_to_data', default='~/path/to/openpose', help='path')

opt, _ = parser.parse_known_args()

# load corresponding metadata
db_K1 = soccer.SoccerVideo(os.path.join(opt.path_to_data, 'K1'))
db_K8 = soccer.SoccerVideo(os.path.join(opt.path_to_data, 'K8'))
# db_Left = soccer.SoccerVideo(os.path.join(opt.path_to_data, 'Left'))
# db_Right = soccer.SoccerVideo(os.path.join(opt.path_to_data, 'Right'))
db_K9 = soccer.SoccerVideo(os.path.join(opt.path_to_data, 'K9'))

db_K1.digest_metadata()
db_K8.digest_metadata()
# db_Left.digest_metadata()
# db_Right.digest_metadata()
db_K9.digest_metadata()

# plot the field
data = []

# K1:
cam = cam_utils.Camera("K1", db_K1.calib[db_K1.frame_basenames[0]]['A'], db_K1.calib[db_K1.frame_basenames[0]]
                       ['R'], db_K1.calib[db_K1.frame_basenames[0]]['T'], db_K1.shape[0], db_K1.shape[1])
vec = np.array([0,0,1])
dir1 = cam.R.T.dot(vec.T)
dir1 *= 10 # cam.A[0][0] # 10 good value
dir1 = dir1+[cam.get_position().item(0, 0), cam.get_position().item(1, 0), cam.get_position().item(2, 0)]
cam_vec1 = np.array([cam.get_position().item(0, 0), cam.get_position().item(1, 0), cam.get_position().item(2, 0)])
data.append(go.Scatter3d(x=[dir1[0], cam_vec1[0]], y=[dir1[1], cam_vec1[1]], z=[dir1[2], cam_vec1[2]],
                         mode='lines', line=dict(color='rgb(0,0,0)', width=5)))

# K8:
cam = cam_utils.Camera("K8", db_K8.calib[db_K8.frame_basenames[0]]['A'], db_K8.calib[db_K8.frame_basenames[0]]
                       ['R'], db_K8.calib[db_K8.frame_basenames[0]]['T'], db_K8.shape[0], db_K8.shape[1])
vec = np.array([0,0,1])
dir2 = cam.R.T.dot(vec.T)
dir2 *= 10 # cam.A[0][0] # 10 good value
dir2 = dir2+[cam.get_position().item(0, 0), cam.get_position().item(1, 0), cam.get_position().item(2, 0)]
cam_vec2 = np.array([cam.get_position().item(0, 0), cam.get_position().item(1, 0), cam.get_position().item(2, 0)])
data.append(go.Scatter3d(x=[dir2[0], cam_vec2[0]], y=[dir2[1], cam_vec2[1]], z=[dir2[2], cam_vec2[2]],
                         mode='lines', line=dict(color='rgb(0,0,0)', width=5)))

# K9:
cam = cam_utils.Camera("K9", db_K9.calib[db_K9.frame_basenames[0]]['A'], db_K9.calib[db_K9.frame_basenames[0]]
                       ['R'], db_K9.calib[db_K9.frame_basenames[0]]['T'], db_K9.shape[0], db_K9.shape[1])
vec = np.array([0,0,1])
dir3 = cam.R.T.dot(vec.T)
dir3 *= 10 # cam.A[0][0] # 10 good value
dir3 = dir3+[cam.get_position().item(0, 0), cam.get_position().item(1, 0), cam.get_position().item(2, 0)]
cam_vec3 = np.array([cam.get_position().item(0, 0), cam.get_position().item(1, 0), cam.get_position().item(2, 0)])
data.append(go.Scatter3d(x=[dir3[0], cam_vec3[0]], y=[dir3[1], cam_vec3[1]], z=[dir3[2], cam_vec3[2]],
                        mode='lines', line=dict(color='rgb(0,0,0)', width=5)))
# ---------------------------------------------------------------

draw.plot_field(data)

# plot the cameras (extension needed - hardcoded))
draw.plot_camera(data, db_K1, "K1")
draw.plot_camera(data, db_K8, "K8")
# draw.plot_camera(data, db_Left, "Left")
# draw.plot_camera(data, db_Right, "Right")
draw.plot_camera(data, db_K9, "K9")


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
            gridcolor='rgb(180, 180, 180)',
            zerolinecolor='rgb(180, 180, 180)',
            showbackground=True,
            spikesides = False,
            backgroundcolor='rgb(230, 230,230)',
            range=[-60, 60]
        ),
        yaxis=dict(
            gridcolor='rgb(180, 180, 180)',
            zerolinecolor='rgb(180, 180, 180)',
            showbackground=True,
            backgroundcolor='rgb(230, 230,230)',
            range=[0, 30]
        ),
        zaxis=dict(
            gridcolor='rgb(180, 180, 180)',
            zerolinecolor='rgb(180, 180, 180)',
            showbackground=True,
            backgroundcolor='rgb(230, 230,230)',
            range=[-40, 60]
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
        aspectratio=dict(x=1, y=0.25, z=0.83333333333),
        aspectmode='manual'
    ),
)


fig = dict(data=data, layout=layout)
py.offline.plot(fig, filename='/home/bunert/Data/results/camera.html', validate=False)
