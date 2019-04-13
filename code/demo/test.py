import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import soccer
from os import listdir
from os.path import isfile, join, exists
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Arc, Rectangle, ConnectionPatch
import utils.io as io
import utils.camera as cam_utils
from tqdm import tqdm


# CMD Line arguments
parser = argparse.ArgumentParser(description='Estimate the poses')
# --path_to_data: where the images are
parser.add_argument('--path_to_data', default='/home/bunert/Data', help='path')
# --cameras: number of cameras
parser.add_argument('--cameras', default=1, type=int, help='path')

opt, _ = parser.parse_known_args()

# load corresponding metadata
db = []
for i in range(opt.cameras):
    db.append(soccer.SoccerVideo(join(opt.path_to_data, 'camera{0}'.format(i))))
    db[i].digest_metadata()

# ------------------------------------------------------------------------------
W, H = 104.73, 67.74


def make_field_circle(r=9.15, nn=1):
    """
    Returns points that lie on a circle on the ground
    :param r: radius
    :param nn: points per arc?
    :return: 3D points on a circle with y = 0
    """
    d = 2 * np.pi * r
    n = int(nn * d)
    return [(np.cos(2 * np.pi / n * x) * r, 0, np.sin(2 * np.pi / n * x) * r) for x in range(0, n + 1)]


def get_field_points():

    outer_rectangle = np.array([[-W / 2., 0, H / 2.],
                                [-W / 2., 0, -H / 2.],
                                [W / 2., 0, -H / 2.],
                                [W / 2., 0, H / 2.]])

    mid_line = np.array([[0., 0., H / 2],
                         [0., 0., -H / 2]])

    left_big_box = np.array([[-W / 2., 0, 40.32/2.],
                             [-W / 2., 0, -40.32 / 2.],
                             [-W / 2. + 16.5, 0, -40.32 / 2.],
                             [-W/2.+16.5, 0, 40.32/2.]])

    right_big_box = np.array([[W/2.-16.5, 0, 40.32/2.],
                              [W/2., 0, 40.32/2.],
                              [W/2., 0, -40.32/2.],
                              [W/2.-16.5, 0, -40.32/2.]])

    left_small_box = np.array([[-W/2., 0, 18.32/2.],
                               [-W / 2., 0, -18.32 / 2.],
                               [-W / 2. + 5.5, 0, -18.32 / 2.],
                               [-W/2.+5.5, 0, 18.32/2.]])

    right_small_box = np.array([[W/2.-5.5, 0, 18.32/2.],
                                [W/2., 0, 18.32/2.],
                                [W/2., 0, -18.32/2.],
                                [W/2.-5.5, 0, -18.32/2.]])

    central_circle = np.array(make_field_circle(r=9.15, nn=1))

    left_half_circile = np.array(make_field_circle(9.15))
    left_half_circile[:, 0] = left_half_circile[:, 0] - W / 2. + 11.0
    index = left_half_circile[:, 0] > (-W / 2. + 16.5)
    left_half_circile = left_half_circile[index, :]

    right_half_circile = np.array(make_field_circle(9.15))
    right_half_circile[:, 0] = right_half_circile[:, 0] + W / 2. - 11.0
    index = right_half_circile[:, 0] < (W / 2. - 16.5)
    right_half_circile = right_half_circile[index, :]

    return [outer_rectangle, left_big_box, right_big_box, left_small_box, right_small_box,
            left_half_circile, right_half_circile, central_circle, mid_line]


# get camera position
cam = cam_utils.Camera('camera0', db[0].calib['00118']['A'], db[0].calib['00118']
                       ['R'], db[0].calib['00118']['T'], db[0].shape[0], db[0].shape[1])

# demo field for x-y plane
field = io.imread('./demo/data/demo_field.png')

field_list = get_field_points()
for i in range(len(field_list)):
    print(field_list[i])
    continue
    # test:
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# camera0
x = cam.get_position().item(0, 0)
y = cam.get_position().item(1, 0)
z = cam.get_position().item(2, 0)
#ax.scatter(x, y, z, label='camera0', marker='^')

# set axis names
ax.set_xlabel('X-axes')
ax.set_ylabel('Y-axes')
ax.set_zlabel('Z-axes')

# plot details
plt.title('soccer field with cameras')
plt.legend(loc=2)
plt.show()
