import numpy as np
from shapely.geometry import LineString, Polygon
import plotly.graph_objs as go
import utils.camera as cam_utils
import cv2

import matplotlib
import os
from os import listdir
from os.path import isfile, join, exists
import soccer
import argparse
import utils.io as io
from tqdm import tqdm
import plotly as py
from plotly.tools import FigureFactory as FF
import scipy

W, H = 104.73, 67.74


################################################################################
# plots all players from arguments in plotly
# arguments: list of all 22 players and their keypoints in 3D (0-10 Denmark, 11-21 Swiss)
################################################################################
def plot_all_players(players_3d):
    # plot the field
    plot_data = []
    plot_field(plot_data)

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
    py.offline.plot(fig, filename='/home/bunert/Data/results/players.html')

def plot_player(players, plot_data, rgb_color):
    limps = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [1, 11], [11, 12], [12, 13], [1, 8],
            [8, 9], [9, 10], [14, 15], [16, 17], [0, 14], [0, 15], [14, 16], [15, 17], [8, 11], [2, 8], [5, 11]])

    players = np.asmatrix(players)

    for i in range(len(limps)):
        plot_data.append(go.Scatter3d(x=[players[limps[i][0]][0,0], players[limps[i][1]][0,0]], y=[players[limps[i][0]][0,1], players[limps[i][1]][0,1]],
             z=[players[limps[i][0]][0,2], players[limps[i][1]][0,2]], mode='lines', line=dict(color=rgb_color, width=3)))


# higher nn lead to preciser cicles but longer computation
def make_field_circle(r=9.15, nn=7):
    """
    Returns points that lie on a circle on the ground
    :param r: radius
    :param nn: points per arc?
    :return: 3D points on a circle with y = 0
    """
    d = 2 * np.pi * r
    n = int(nn * d)
    return [(np.cos(2 * np.pi / n * x) * r, 0, np.sin(2 * np.pi / n * x) * r) for x in range(0, n + 1)]

# sort the polygon to plot the convex hull of the polygon
def PolygonSort(corners):
    n = len(corners)
    cx = float(sum(x for x, y in corners)) / n
    cy = float(sum(y for x, y in corners)) / n
    cornersWithAngles = []
    for x, y in corners:
        an = (np.arctan2(y - cy, x - cx) + 2.0 * np.pi) % (2.0 * np.pi)
        cornersWithAngles.append((x, y, an))
    cornersWithAngles.sort(key=lambda tup: tup[2])
    return map(lambda tuple: (tuple[0], tuple[1]), cornersWithAngles)

# return the points for the straight lines of the field
def get_field_points_limited():

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

# draw a box out of 4 data points
def draw_box(b, data):
    data.append(go.Scatter3d(x=[b.item(0), b.item(3)], y=[0, 0], z=[b.item(2), b.item(5)],
                             mode='lines', line=dict(color='rgb(0,0,0)', width=3)))
    data.append(go.Scatter3d(x=[b.item(0), b.item(9)], y=[0, 0], z=[b.item(2), b.item(11)],
                             mode='lines', line=dict(color='rgb(0,0,0)', width=3)))
    data.append(go.Scatter3d(x=[b.item(3), b.item(6)], y=[0, 0], z=[b.item(5), b.item(8)],
                             mode='lines', line=dict(color='rgb(0,0,0)', width=3)))
    data.append(go.Scatter3d(x=[b.item(6), b.item(9)], y=[0, 0], z=[b.item(8), b.item(11)],
                             mode='lines', line=dict(color='rgb(0,0,0)', width=3)))


# draw a single line
def draw_line(b, data):
    data.append(go.Scatter3d(x=[b.item(0), b.item(3)], y=[0, 0], z=[b.item(2), b.item(5)],
                             mode='lines', line=dict(color='rgb(0,0,0)', width=3)))

def connect_points(p1, p2, data):
    data.append(go.Scatter3d(x=[p1[0], p2[0]], y=[p1[1], p2[1]], z=[p1[2], p2[2]],
                             mode='lines', line=dict(color='rgb(0,0,0)', width=3)))


# middle circle
def draw_middlecircle(data):
    corners = np.array(make_field_circle(9.15))
    tuples = []
    for i in range(len(corners)):
        tuples.append((corners[i, 0], corners[i, 2]))
    corners_sorted = list(PolygonSort(tuples))
    x = [corner[0] for corner in corners_sorted]
    z = [corner[1] for corner in corners_sorted]
    draw_circle(data, x, z)

# draw a cicle
def draw_circle(data, x, z):
    for i in range(len(x)-1):
        data.append(go.Scatter3d(x=[x[i], x[i+1]], y=[0, 0], z=[z[i], z[i+1]],
                                 mode='lines', line=dict(color='rgb(0,0,0)', width=3)))
    data.append(go.Scatter3d(x=[x[len(x)-1], x[0]], y=[0, 0], z=[z[len(z)-1], z[0]],
                             mode='lines', line=dict(color='rgb(0,0,0)', width=3)))

# left halft circle
def draw_lefthalf_circle(data):
    left_half_circile = np.array(make_field_circle(9.15))
    left_half_circile[:, 0] = left_half_circile[:, 0] - W / 2. + 11.0
    index = left_half_circile[:, 0] > (-W / 2. + 16.5)
    left_half_circile = left_half_circile[index, :]
    tuples = []
    for i in range(len(left_half_circile)):
        tuples.append((left_half_circile[i, 0], left_half_circile[i, 2]))
    corners_sorted = list(PolygonSort(tuples))
    x = [corner[0] for corner in corners_sorted]
    z = [corner[1] for corner in corners_sorted]
    draw_circle(data, x, z)

# right half circle
def draw_righthalf_circle(data):
    right_half_circile = np.array(make_field_circle(9.15))
    right_half_circile[:, 0] = right_half_circile[:, 0] + W / 2. - 11.0
    index = right_half_circile[:, 0] < (W / 2. - 16.5)
    right_half_circile = right_half_circile[index, :]
    tuples = []
    for i in range(len(right_half_circile)):
        tuples.append((right_half_circile[i, 0], right_half_circile[i, 2]))
    corners_sorted = list(PolygonSort(tuples))
    x = [corner[0] for corner in corners_sorted]
    z = [corner[1] for corner in corners_sorted]
    draw_circle(data, x, z)

# plot the whole field
def plot_field(data):
    field_list = get_field_points_limited()
    for i in range(len(field_list)):
        if (i <= 4):
            draw_box(field_list[i], data)
        elif (i == 5):
            draw_line(field_list[i], data)
    draw_lefthalf_circle(data)
    draw_righthalf_circle(data)
    draw_middlecircle(data)

# plot one camera
# needs to get extended:
# - frame number hardcoded
# - iterate through all cameras available
def plot_camera(data, db, name):
    cam = cam_utils.Camera(name, db.calib[db.frame_basenames[0]]['A'], db.calib[db.frame_basenames[0]]
                           ['R'], db.calib[db.frame_basenames[0]]['T'], db.shape[0], db.shape[1])

    data.append(go.Scatter3d(go.Scatter3d(
        x=[cam.get_position().item(0, 0)],
        y=[cam.get_position().item(1, 0)],
        z=[cam.get_position().item(2, 0)],
        text=cam.name,
        mode='markers',
        marker=dict(
            sizemode='diameter',
            sizeref=750,  # info on sizeref: https://plot.ly/python/reference/#scatter-marker-sizeref
            size=6,
            color='rgb(30, 70, 180)',
        ))))

################################################################################
# Original draw file:
################################################################################


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


def project_field_to_image(camera):

    field_list = get_field_points()

    field_points2d = []
    for i in range(len(field_list)):
        tmp, depth = camera.project(field_list[i])

        behind_points = (depth < 0).nonzero()[0]
        tmp[behind_points, :] *= -1
        # if len(behind_points) > 0:
        #     new_points = camera.

        # Check if point begind the camera
        # center = camera.get_position()
        # dir = camera.get_direction()
        # for j in range(field_list[i].shape[0]):
        #     point_dir = field_list[i][j, :] - center
        #     point_dir /= np.linalg.norm(point_dir)
        #
        #     dot_prod = np.dot(dir, point_dir)
        #

        field_points2d.append(tmp)

    return field_points2d


def draw_field(camera):

    field_points2d = project_field_to_image(camera)
    h, w = camera.height, camera.width
    # Check if the entities are 7
    assert len(field_points2d) == 9

    img_polygon = Polygon([(0, 0), (w - 1, 0), (w - 1, h - 1), (0, h - 1)])

    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    mask = np.zeros((h, w, 3), dtype=np.uint8)

    # Draw the boxes
    for i in range(5):

        # And make a new image with the projected field
        linea = LineString([(field_points2d[i][0, :]),
                            (field_points2d[i][1, :])])

        lineb = LineString([(field_points2d[i][1, :]),
                            (field_points2d[i][2, :])])

        linec = LineString([(field_points2d[i][2, :]),
                            (field_points2d[i][3, :])])

        lined = LineString([(field_points2d[i][3, :]),
                            (field_points2d[i][0, :])])

        if i == 0:
            polygon0 = Polygon([(field_points2d[i][0, :]),
                                (field_points2d[i][1, :]),
                                (field_points2d[i][2, :]),
                                (field_points2d[i][3, :])])

            intersect0 = img_polygon.intersection(polygon0)
            if not intersect0.is_empty:
                pts = np.array(list(intersect0.exterior.coords), dtype=np.int32)
                pts = pts[:, :].reshape((-1, 1, 2))
                cv2.fillConvexPoly(mask, pts, (255, 255, 255))

        intersect0 = img_polygon.intersection(linea)
        if not intersect0.is_empty:
            pts = np.array(list(list(intersect0.coords)), dtype=np.int32)
            cv2.line(canvas, (pts[0, 0], pts[0, 1]), (pts[1, 0], pts[1, 1]), (255, 255, 255))

        intersect0 = img_polygon.intersection(lineb)
        if not intersect0.is_empty:
            pts = np.array(list(list(intersect0.coords)), dtype=np.int32)
            if pts.shape[0] < 2:
                continue
            cv2.line(canvas, (pts[0, 0], pts[0, 1]), (pts[1, 0], pts[1, 1]), (255, 255, 255))

        intersect0 = img_polygon.intersection(linec)
        if not intersect0.is_empty:
            pts = np.array(list(list(intersect0.coords)), dtype=np.int32)
            if pts.shape[0] == 2:
                cv2.line(canvas, (pts[0, 0], pts[0, 1]), (pts[1, 0], pts[1, 1]), (255, 255, 255))

        intersect0 = img_polygon.intersection(lined)
        if not intersect0.is_empty:
            pts = np.array(list(list(intersect0.coords)), dtype=np.int32)
            cv2.line(canvas, (pts[0, 0], pts[0, 1]), (pts[1, 0], pts[1, 1]), (255, 255, 255))

    # Mid line
    line1 = LineString([(field_points2d[8][0, :]),
                        (field_points2d[8][1, :])])

    intersect1 = img_polygon.intersection(line1)
    if not intersect1.is_empty:
        pts = np.array(list(list(intersect1.coords)), dtype=np.int32)
        pts = pts[:, :].reshape((-1, 1, 2))
        cv2.fillConvexPoly(canvas, pts, (255, 255, 255), )

    # Circles
    for ii in range(5, 8):
        for i in range(field_points2d[ii].shape[0] - 1):
            line2 = LineString([(field_points2d[ii][i, :]),
                                (field_points2d[ii][i + 1, :])])
            intersect2 = img_polygon.intersection(line2)
            if not intersect2.is_empty:
                pts = np.array(list(list(intersect2.coords)), dtype=np.int32)
                pts = pts[:, :].reshape((-1, 1, 2))
                cv2.fillConvexPoly(canvas, pts, (255, 255, 255), )

    return canvas[:, :, 0] / 255., mask[:, :, 0] / 255.


def draw_skeleton(keypoints, h, w):

    limps = np.array(
        [[0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [1, 11], [11, 12], [12, 13], [1, 8],
         [8, 9], [9, 10], [14, 15], [16, 17], [0, 14], [0, 15], [14, 16], [15, 17], [8, 11], [2, 8], [5, 11]])

    fg_label = 1

    output = np.zeros((h, w, 3), dtype=np.float32)

    for k in range(limps.shape[0]):
        kp1, kp2 = limps[k, :].astype(int)
        bone_start = keypoints[kp1, :]
        bone_end = keypoints[kp2, :]
        bone_start[0] = np.maximum(np.minimum(bone_start[0], w - 1), 0.)
        bone_start[1] = np.maximum(np.minimum(bone_start[1], h - 1), 0.)

        bone_end[0] = np.maximum(np.minimum(bone_end[0], w - 1), 0.)
        bone_end[1] = np.maximum(np.minimum(bone_end[1], h - 1), 0.)

        if bone_start[2] > 0.0:
            output[int(bone_start[1]), int(bone_start[0])] = 1
            cv2.circle(output, (int(bone_start[0]), int(bone_start[1])), 2, (fg_label, 0, 0), -1)

        if bone_end[2] > 0.0:
            output[int(bone_end[1]), int(bone_end[0])] = 1
            cv2.circle(output, (int(bone_end[0]), int(bone_end[1])), 2, (fg_label, 0, 0), -1)

        if bone_start[2] > 0.0 and bone_end[2] > 0.0:
            cv2.line(output, (int(bone_start[0]), int(bone_start[1])),
                     (int(bone_end[0]), int(bone_end[1])), (fg_label, 0, 0), 1)

    return output[:, :, 0]


def draw_skeleton_on_image(img, poses, cmap_fun, one_color=False, pose_color=None):
    limps = np.array(
        [[0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [1, 11], [11, 12], [12, 13], [1, 8],
         [8, 9], [9, 10], [14, 15], [16, 17], [0, 14], [0, 15], [14, 16], [15, 17], [8, 11], [2, 8], [5, 11]])

    for i in range(len(poses)):
        if poses[i] is None:
            continue

        if one_color:
            clr = cmap_fun(0.5)
        else:
            if pose_color is None:
                clr = cmap_fun(i / float(len(poses)))
            else:
                clr = cmap_fun(pose_color[i])

        for k in range(limps.shape[0]):
            kp1, kp2 = limps[k, :].astype(int)
            x1, y1, s1 = poses[i][kp1, :]
            x2, y2, s2 = poses[i][kp2, :]

            if s1 == 0 or s2 == 0:
                continue

            cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)),
                     (int(clr[0]*255), int(clr[1]*255), int(clr[2]*255)), 3)

def draw_skeleton_on_image_2dposes(img, poses, cmap_fun, one_color=False, pose_color=None):
    limps = np.array(
        [[0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [1, 11], [11, 12], [12, 13], [1, 8],
         [8, 9], [9, 10], [14, 15], [16, 17], [0, 14], [0, 15], [14, 16], [15, 17], [8, 11], [2, 8], [5, 11]])

    for i in range(len(poses)):
        if poses[i] is None:
            continue

        if one_color:
            clr = cmap_fun(0.5)
        else:
            if pose_color is None:
                clr = cmap_fun(i / float(len(poses)))
            else:
                clr = cmap_fun(pose_color[i])

        for k in range(limps.shape[0]):
            kp1, kp2 = limps[k, :].astype(int)
            x1, y1 = poses[kp1][0]
            x2, y2 = poses[kp2][0]


            cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)),
                     (int(clr[0]*255), int(clr[1]*255), int(clr[2]*255)), 3)

################################################################################
# Project all players on the frame number image of the given db_cam
# -> get the players_3d array as argument
################################################################################
def project_3d_players_on_image(db_cam, players_3d, frame):
    frame_name = db_cam.frame_basenames[frame]
    camera = cam_utils.Camera("Cam", db_cam.calib[frame_name]['A'], db_cam.calib[frame_name]['R'], db_cam.calib[frame_name]['T'], db_cam.shape[0], db_cam.shape[1])

    cmap = matplotlib.cm.get_cmap('hsv')
    img = db_cam.get_frame(frame, dtype=np.uint8)
    for k in players_3d:
        points2d = []
        for i in range(len(players_3d[k])):
            tmp, depth = camera.project(players_3d[k][i])
            behind_points = (depth < 0).nonzero()[0]
            tmp[behind_points, :] *= -1
            points2d.append(tmp)
        draw_skeleton_on_image_2dposes(img, points2d, cmap, one_color=True)

    cv2.imwrite('/home/bunert/Data/results/'+ db_cam.name +  '_3d' + '.jpg',np.uint8(img[:, :, (2, 1, 0)]))


################################################################################
# Project the players on the frame number image of the given db_cam
################################################################################
def draw_2d_players_on_image(db_cam, players_2d, frame, player=False):
    cmap = matplotlib.cm.get_cmap('hsv')
    img = db_cam.get_frame(frame, dtype=np.uint8)

    # if just one player: place in array to get draw working
    if (player):
        draw_skeleton_on_image_2dposes(img, players_2d[player], cmap, one_color=True)
    else:
        for i in range (len(players_2d)):
            draw_skeleton_on_image_2dposes(img, players_2d[i], cmap, one_color=True)

    cv2.imwrite('/home/bunert/Data/results/'+ db_cam.name + '_2d' + '.jpg',np.uint8(img[:, :, (2, 1, 0)]))


################################################################################
# Draw all players (2D dictionary) on the first (frame 0) image from one Kamera db_cam:
# players_2d_dict: [[x,y,prec]]
################################################################################
def draw_openpose_on_image(db_cam, players_2d_dict, frame, player=False):
    # if just one player: place in array to get draw working
    cmap = matplotlib.cm.get_cmap('hsv')
    img = db_cam.get_frame(frame, dtype=np.uint8)
    if (player):
        print("Just one player projected")
        draw_skeleton_on_image(img, [players_2d_dict[player]], cmap, one_color=True)
    else:
        players_2d = [ v for v in players_2d_dict.values() ]
        draw_skeleton_on_image(img, players_2d, cmap, one_color=True)
    cv2.imwrite('/home/bunert/Data/results/'+ db_cam.name + '_openpose' + '.jpg',np.uint8(img[:, :, (2, 1, 0)]))
