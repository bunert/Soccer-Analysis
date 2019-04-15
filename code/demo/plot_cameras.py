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
from matplotlib.patches import Circle, PathPatch
import matplotlib.patches as mpatches
import mpl_toolkits.mplot3d.art3d as art3d
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


def make_field_circle(r=9.15, nn=5):
    """
    Returns points that lie on a circle on the ground
    :param r: radius
    :param nn: points per arc?
    :return: 3D points on a circle with y = 0
    """
    d = 2 * np.pi * r
    n = int(nn * d)
    return [(np.cos(2 * np.pi / n * x) * r, np.sin(2 * np.pi / n * x) * r) for x in range(0, n + 1)]


def get_field_points():

    outer_rectangle = np.array([[-W / 2., 0, H / 2.],
                                [-W / 2., 0, -H / 2.],
                                [W / 2., 0, -H / 2.],
                                [W / 2., 0, H / 2.]])

    left_big_box = np.array([[-W / 2., 0, 40.32/2.],
                             [-W / 2., 0, -40.32 / 2.],
                             [-W / 2. + 16.5, 0, -40.32 / 2.],
                             [-W / 2.+16.5, 0, 40.32/2.]])

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

    mid_line = np.array([[0., 0., H / 2],
                         [0., 0., -H / 2]])

    return [outer_rectangle, left_big_box, right_big_box, left_small_box, right_small_box, mid_line]


def draw_box(b):
    ax.plot([b.item(0),b.item(3)],[0,0],zs=[b.item(2),b.item(5)],color='black')
    ax.plot([b.item(0),b.item(9)],[0,0],zs=[b.item(2),b.item(11)],color='black')
    ax.plot([b.item(3),b.item(6)],[0,0],zs=[b.item(5),b.item(8)],color='black')
    ax.plot([b.item(6),b.item(9)],[0,0],zs=[b.item(8),b.item(11)],color='black')

def draw_line(b):
    ax.plot([b.item(0),b.item(3)],[0,0],zs=[b.item(2),b.item(5)],color='black')

def draw_middlecircle():
    p = Circle((0, 0), 9.15, fill=False)
    ax.add_patch(p)
    art3d.pathpatch_2d_to_3d(p, z=0, zdir="y")

def draw_lefthalf_circle():

    left_half_circile = np.array(make_field_circle(9.15))
    left_half_circile[:, 0] = left_half_circile[:, 0] - W / 2. + 11.0
    index = left_half_circile[:, 0] > (-W / 2. + 16.5)
    left_half_circile = left_half_circile[index, :]
    poly = mpatches.Polygon(left_half_circile, fill=False)
    ax.add_patch(poly)
    art3d.pathpatch_2d_to_3d(poly, z=0, zdir="y")


def draw_righthalf_circle():
    right_half_circile = np.array(make_field_circle(9.15))
    right_half_circile[:, 0] = right_half_circile[:, 0] + W / 2. - 11.0
    index = right_half_circile[:, 0] < (W / 2. - 16.5)
    right_half_circile = right_half_circile[index, :]
    poly = mpatches.Polygon(right_half_circile, fill=False)
    ax.add_patch(poly)
    art3d.pathpatch_2d_to_3d(poly, z=0, zdir="y")

def plot_field():
    field_list = get_field_points()
    for i in range(len(field_list)):
        if (i<=4):
            draw_box(field_list[i])
        elif (i==5):
            draw_line(field_list[i])
    draw_lefthalf_circle()
    draw_righthalf_circle()
    draw_middlecircle()

# get camera position
cam = cam_utils.Camera('camera0', db[0].calib['00118']['A'], db[0].calib['00118']
                  ['R'], db[0].calib['00118']['T'], db[0].shape[0], db[0].shape[1])

# initialize figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


# camera0
x = cam.get_position().item(0, 0)
y = cam.get_position().item(1, 0)
z = cam.get_position().item(2, 0)
ax.scatter(x, y, z, label='camera0', marker='o')

# demo field for x-y plane
field = io.imread('./demo/data/demo_field.png')



plot_field()

# set axis names
ax.set_xlabel('X', fontsize=25, rotation=0)
ax.set_ylabel('Y', fontsize=25, rotation=0)
ax.set_zlabel('Z', fontsize=25, rotation=0)

# change axis location (naming)
# ax.xaxis._axinfo['juggled'] = (0,0,0)
# ax.yaxis._axinfo['juggled'] = (1,1,1)
# ax.zaxis._axinfo['juggled'] = (2,2,2)

# scale
ax.set_xlim3d(-100,100)
ax.set_ylim3d(0,200)
ax.set_zlim3d(-100,100)


#print(ax.get_xaxis().get_navigate)
ax.get_xaxis().set_label_position('top')
ax.view_init(azim=-90, elev=120)
ax.xaxis._axinfo['juggled'] = (0,0,0)
ax.yaxis._axinfo['juggled'] = (1,1,1)
ax.zaxis._axinfo['juggled'] = (2,2,2)

# plot details
plt.title('soccer field with cameras')
plt.legend(loc=2)

# ax.disable_mouse_rotation()
# ax.mouse_init(rotate_btn=1, zoom_btn=3)


plt.show()
