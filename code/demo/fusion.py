import os
import argparse
import soccer
import filter
from os import listdir
from os.path import isfile, join, exists
import pandas
import cv2
import numpy as np
import matplotlib
import plotly as py
import utils.draw2 as draw
import utils.camera as cam_utils
import utils.draw2 as draw_utils
from tqdm import tqdm


import math
import sys
import glog

# to print full matrices
np.set_printoptions(threshold=sys.maxsize)

# hardcoded number of frame (-> Soccer.n_frames)
number_of_frames = 328
number_of_keypoints = 18
number_of_cameras = 5


################################################################################
# run: python3 demo/fusion.py --path_to_data ~/Data/
################################################################################


# CMD Line arguments
parser = argparse.ArgumentParser(description='Estimate the poses')
# --path_to_data: where the images are
parser.add_argument('--path_to_data', default='/home/bunert/Data/', help='path')

opt, _ = parser.parse_known_args()

################################################################################
################################################################################
# initialization of the data
################################################################################
################################################################################


################################################################################
# initialize databases for all cameras and load the data (COCO)
# names: db_K1, db_K8, db_K9, db_Left, db_Right
# example: print(db_K1.poses["00000009"][0])
################################################################################
def init_soccerdata(mylist):
    # load corresponding metadata
    db_K1 = soccer.SoccerVideo(os.path.join(opt.path_to_data, 'K1'))
    db_K1.name = "K1"
    db_K8 = soccer.SoccerVideo(os.path.join(opt.path_to_data, 'K8'))
    db_K8.name = "K8"
    db_K9 = soccer.SoccerVideo(os.path.join(opt.path_to_data, 'K9'))
    db_K9.name = "K9"
    db_Left = soccer.SoccerVideo(os.path.join(opt.path_to_data, 'Left'))
    db_Left.name = "Left"
    db_Right = soccer.SoccerVideo(os.path.join(opt.path_to_data, 'Right'))
    db_Right.name = "Right"

    data_dict = {}
    if 0 in mylist:
        db_K1.digest_metadata()
        db_K1.refine_poses(keypoint_thresh=7, score_thresh=0.4, neck_thresh=0.4)
        data_dict.update({0:db_K1})
    if 1 in mylist:
        db_K8.digest_metadata()
        db_K8.refine_poses(keypoint_thresh=7, score_thresh=0.4, neck_thresh=0.4)
        data_dict.update({1:db_K8})
    if 2 in mylist:
        db_K9.digest_metadata()
        db_K9.refine_poses(keypoint_thresh=7, score_thresh=0.4, neck_thresh=0.4)
        data_dict.update({2:db_K9})
    if 3 in mylist:
        db_Left.digest_metadata()
        db_Left.refine_poses(keypoint_thresh=7, score_thresh=0.4, neck_thresh=0.4)
        data_dict.update({3:db_Left})
    if 4 in mylist:
        db_Right.digest_metadata()
        db_Right.refine_poses(keypoint_thresh=7, score_thresh=0.4, neck_thresh=0.4)
        data_dict.update({4:db_Right})

    return data_dict


################################################################################
# access: players[x].iloc[row][column]
# column: 0=time, 1=x, 2=y -> y is later z in 3D
# TODO: verschiebung von Feldgrösse genau berechnen (csv field ungenau)
################################################################################
def init_csv():
    players = []

    # DANMARK
    # 0 - Dan_1: Schmeichel
    players.append(pandas.read_csv('/home/bunert/Data/Smash/Switzerland_-_Denmark_Fitness_RAW_data_Denmark/lib/approximation_data/Schmeichel_Kasper.csv',
            sep = ';', decimal=",", skiprows=5, usecols=[0,1,2], nrows=505, names=['time', 'x', 'y']))
    # 1 - Dan_4: Kjar
    players.append(pandas.read_csv('/home/bunert/Data/Smash/Switzerland_-_Denmark_Fitness_RAW_data_Denmark/lib/approximation_data/Kjar_Simon.csv',
            sep = ';', decimal=",", skiprows=5, usecols=[0,1,2], nrows=505, names=['time', 'x', 'y']))
    # 2 - Dan_8: Delaney
    players.append(pandas.read_csv('/home/bunert/Data/Smash/Switzerland_-_Denmark_Fitness_RAW_data_Denmark/lib/approximation_data/Delaney_Thomas.csv',
            sep = ';', decimal=",", skiprows=5, usecols=[0,1,2], nrows=505, names=['time', 'x', 'y']))
    # 3 - Dan_9: Jorgensen Nicolai
    players.append(pandas.read_csv('/home/bunert/Data/Smash/Switzerland_-_Denmark_Fitness_RAW_data_Denmark/lib/approximation_data/Jorgensen_Nicolai.csv',
            sep = ';', decimal=",", skiprows=5, usecols=[0,1,2], nrows=505, names=['time', 'x', 'y']))
    # 4 - Dan_10: Eriksen
    players.append(pandas.read_csv('/home/bunert/Data/Smash/Switzerland_-_Denmark_Fitness_RAW_data_Denmark/lib/approximation_data/Christian_Eriksen.csv',
            sep = ';', decimal=",", skiprows=5, usecols=[0,1,2], nrows=505, names=['time', 'x', 'y']))
    # 5 - Dan_11 Braithwaite
    players.append(pandas.read_csv('/home/bunert/Data/Smash/Switzerland_-_Denmark_Fitness_RAW_data_Denmark/lib/approximation_data/Braithwaite_Christensen_Martin.csv',
            sep = ';', decimal=",", skiprows=5, usecols=[0,1,2], nrows=505, names=['time', 'x', 'y']))
    # 6 - Dan_13: Jorgensen Mathias aka Zanka
    players.append(pandas.read_csv('/home/bunert/Data/Smash/Switzerland_-_Denmark_Fitness_RAW_data_Denmark/lib/approximation_data/Zanka.csv',
            sep = ';', decimal=",", skiprows=5, usecols=[0,1,2], nrows=505, names=['time', 'x', 'y']))
    # 7 - Dan_14: Dalsgaard
    players.append(pandas.read_csv('/home/bunert/Data/Smash/Switzerland_-_Denmark_Fitness_RAW_data_Denmark/lib/approximation_data/Dalsgaard_Henrik.csv',
            sep = ';', decimal=",", skiprows=5, usecols=[0,1,2], nrows=505, names=['time', 'x', 'y']))
    # 8 - Dan_17: Larsen
    players.append(pandas.read_csv('/home/bunert/Data/Smash/Switzerland_-_Denmark_Fitness_RAW_data_Denmark/lib/approximation_data/Stryger-Larsen_Jens.csv',
            sep = ';', decimal=",", skiprows=5, usecols=[0,1,2], nrows=505, names=['time', 'x', 'y']))
    # 9 - Dan_19: Schöne
    players.append(pandas.read_csv('/home/bunert/Data/Smash/Switzerland_-_Denmark_Fitness_RAW_data_Denmark/lib/approximation_data/Schone_Lasse.csv',
            sep = ';', decimal=",", skiprows=5, usecols=[0,1,2], nrows=505, names=['time', 'x', 'y']))
    # 10 - Dan_20: Poulsen
    players.append(pandas.read_csv('/home/bunert/Data/Smash/Switzerland_-_Denmark_Fitness_RAW_data_Denmark/lib/approximation_data/Poulsen_Yussuf.csv',
            sep = ';', decimal=",", skiprows=5, usecols=[0,1,2], nrows=505, names=['time', 'x', 'y']))

    # SWITZERLAND
    # 11 - CH_1: Sommer
    players.append(pandas.read_csv('/home/bunert/Data/Smash/Switzerland_-_Denmark_Fitness_RAW_data_Switzerland/lib/approximation_data/Sommer_Yann.csv',
            sep = ';', decimal=",", skiprows=5, usecols=[0,1,2], nrows=505, names=['time', 'x', 'y']))
    # 12 - CH_4: Elvedi
    players.append(pandas.read_csv('/home/bunert/Data/Smash/Switzerland_-_Denmark_Fitness_RAW_data_Switzerland/lib/approximation_data/Elvedi_Nico.csv',
            sep = ';', decimal=",", skiprows=5, usecols=[0,1,2], nrows=505, names=['time', 'x', 'y']))
    # 13 - CH_5: Akanji
    players.append(pandas.read_csv('/home/bunert/Data/Smash/Switzerland_-_Denmark_Fitness_RAW_data_Switzerland/lib/approximation_data/Akanji_Manuel_Obafemi.csv',
            sep = ';', decimal=",", skiprows=5, usecols=[0,1,2], nrows=505, names=['time', 'x', 'y']))
    # 14 - CH_7: Embolo
    players.append(pandas.read_csv('/home/bunert/Data/Smash/Switzerland_-_Denmark_Fitness_RAW_data_Switzerland/lib/approximation_data/Embolo_Breel-Donald.csv',
            sep = ';', decimal=",", skiprows=5, usecols=[0,1,2], nrows=505, names=['time', 'x', 'y']))
    # 15 - CH_8: Freuler
    players.append(pandas.read_csv('/home/bunert/Data/Smash/Switzerland_-_Denmark_Fitness_RAW_data_Switzerland/lib/approximation_data/Freuler_Remo.csv',
            sep = ';', decimal=",", skiprows=5, usecols=[0,1,2], nrows=505, names=['time', 'x', 'y']))
    # 16 - CH_9: Ajeti
    players.append(pandas.read_csv('/home/bunert/Data/Smash/Switzerland_-_Denmark_Fitness_RAW_data_Switzerland/lib/approximation_data/Ajeti_Albian.csv',
            sep = ';', decimal=",", skiprows=5, usecols=[0,1,2], nrows=505, names=['time', 'x', 'y']))
    # 17 - CH_10: Xhaka
    players.append(pandas.read_csv('/home/bunert/Data/Smash/Switzerland_-_Denmark_Fitness_RAW_data_Switzerland/lib/approximation_data/Xhaka_Granit.csv',
            sep = ';', decimal=",", skiprows=5, usecols=[0,1,2], nrows=505, names=['time', 'x', 'y']))
    # 18 - CH_13: Rodriguez
    players.append(pandas.read_csv('/home/bunert/Data/Smash/Switzerland_-_Denmark_Fitness_RAW_data_Switzerland/lib/approximation_data/Rodriguez_Ricardo.csv',
            sep = ';', decimal=",", skiprows=5, usecols=[0,1,2], nrows=505, names=['time', 'x', 'y']))
    # 19 - CH_14: Zuber
    players.append(pandas.read_csv('/home/bunert/Data/Smash/Switzerland_-_Denmark_Fitness_RAW_data_Switzerland/lib/approximation_data/Zuber_Steven.csv',
            sep = ';', decimal=",", skiprows=5, usecols=[0,1,2], nrows=505, names=['time', 'x', 'y']))
    # 20 - CH_17: Zakaria
    players.append(pandas.read_csv('/home/bunert/Data/Smash/Switzerland_-_Denmark_Fitness_RAW_data_Switzerland/lib/approximation_data/Zakaria_Denis.csv',
            sep = ';', decimal=",", skiprows=5, usecols=[0,1,2], nrows=505, names=['time', 'x', 'y']))
    # 21 - CH_23: Mbabu
    players.append(pandas.read_csv('/home/bunert/Data/Smash/Switzerland_-_Denmark_Fitness_RAW_data_Switzerland/lib/approximation_data/Mbabu_Kevin.csv',
            sep = ';', decimal=",", skiprows=5, usecols=[0,1,2], nrows=505, names=['time', 'x', 'y']))

    for i in range(len(players)):
        players[i]['y'] -= 32.75
        players[i]['x'] *= 1.03693
        players[i]['y'] *= -1.03419847328

    return players

################################################################################
# Blickrichtung: initial in the positive X direction
################################################################################
def init_3d_players(x,z,alpha):
    # W/X = 104.73, H/Y = 67.74 Meter

    body = []

    #  0: neck
    body.append([0. ,1.6875 ,0.])

    #  1: middle shoulder
    body.append([0. ,1.5075 , 0.])

    #  2: right shoulder
    body.append([0. ,1.5075 ,0.15])

    #  3: right ellbow
    body.append([0.,1.125 ,0.325])

    #  4: right hand
    body.append([0.2, 0.8 ,0.3])

    #  5: left shoulder
    body.append([0. ,1.5075 , -0.15])

    #  6: left ellbow
    body.append([0.,1.125 , -0.325])

    #  7: left hand
    body.append([0.2, 0.8, -0.3])

    #  8: right hip
    body.append([0., 0.9, 0.15])

    #  9: right knee
    body.append([0., 0.45, 0.2])

    #  10: right feet
    body.append([0., 0., 0.25])

    #  11: left hip
    body.append([0., 0.9, -0.15])

    #  12: left knee
    body.append([0., 0.45, -0.2])

    #  13: left feet
    body.append([0., 0., -0.25])

    #  14: right head
    body.append([0., 1.7375, 0.075])

    #  15: left head
    body.append([0., 1.7375, -0.075])

    #  16: right ear
    body.append([0., 1.6875, 0.075])

    #  17: left ear
    body.append([0., 1.6875, -0.075])

    body = np.asmatrix(body)

    # building rotation matrix if necessary:
    theta = np.radians(alpha)
    c = np.cos(theta)
    s = np.sin(theta)
    R = np.matrix([[c, 0., s],[0., 1., 0.], [-s,0.,  c]])

    # person rotated about alpha degrees
    body = R.dot(body.T).T

    # translate with x and z
    body[:,0] = body[:,0] + x
    body[:,2] = body[:,2] + z

    return body

################################################################################
# project each csv player on the field
# return: array of all player, each player is a matrix
################################################################################
def init_all_3d_players(csv_players, frame):
    players = {}
    for i in range(len(csv_players)):
        #Blickrichtung,
        if (i <= 10):
            players.update({i:init_3d_players(csv_players[i].iloc[frame][1],csv_players[i].iloc[frame][2],180)})
        else:
            players.update({i:init_3d_players(csv_players[i].iloc[frame][1],csv_players[i].iloc[frame][2],0)})

    return players

################################################################################
# Project all players on the 2D plane from the camera db_cam
################################################################################
def project_players_2D(db_cam, players_3d, frame):
    frame_name = db_cam.frame_basenames[frame]
    camera = cam_utils.Camera("Cam", db_cam.calib[frame_name]['A'], db_cam.calib[frame_name]['R'], db_cam.calib[frame_name]['T'], db_cam.shape[0], db_cam.shape[1])

    players_2d = {}
    cmap = matplotlib.cm.get_cmap('hsv')
    img = db_cam.get_frame(frame, dtype=np.uint8)
    for k in players_3d:
        points2d = []
        for i in range(len(players_3d[k])):
            tmp, depth = camera.project(players_3d[k][i])
            behind_points = (depth < 0).nonzero()[0]
            tmp[behind_points, :] *= -1
            points2d.append(tmp)
        players_2d.update({k:points2d})

    return players_2d

################################################################################
# Get all poses from openpose for a specific frame
################################################################################
def get_actual_2D_keypoints(data_dict, frame):
    actual_keypoint_dict = {}
    for i in data_dict:
        frame_name = data_dict[i].frame_basenames[frame]
        actual_keypoint_dict.update({i: data_dict[i].poses[frame_name]})
    return actual_keypoint_dict

################################################################################
# Return dictionary for all cameras, with all the poses (players_3d projected on the camera screen)
################################################################################
def project_players_allCameras_2D(data_dict, players_3d, frame):
    projected_players_2d_dict = {}
    for i in data_dict:
        projected_players_2d_dict.update({i:project_players_2D(data_dict[i], players_3d, frame)})

    return projected_players_2d_dict


################################################################################
################################################################################
# Metric logic
################################################################################
################################################################################

################################################################################
# return nearest player of openpose poses to all the projected players
################################################################################
def nearest_player(keypoints, projected_players_2d):
    minimal_distance = sys.float_info.max
    number = False
    distance = 0.
    for i in projected_players_2d:
        distance = 0.
        for k in range(len(keypoints)):
            x1, y1 = keypoints[k][0], keypoints[k][1]
            x2, y2 = projected_players_2d[i][k][0][0], projected_players_2d[i][k][0][1]
            distance += math.sqrt(math.pow(x1-x2,2)+ math.pow(y1-y2,2))

        # update if distance in smaller
        if (distance < minimal_distance):
            minimal_distance = distance
            number = i
            pose = keypoints

    return number, pose

################################################################################
# Return dictionary for all cameras, with all the poses as tuples with corresponding player numer (0-10 Danmark, 11-21 Swiss)
################################################################################
def assign_player_to_poses(projected_players_2d_dict, keypoint_dict):
    players_2d_dict = {}
    for i in keypoint_dict:
        players_2d = {}
        for k in range(len(keypoint_dict[i])):
            number,pose = nearest_player(keypoint_dict[i][k], projected_players_2d_dict[i])
            players_2d.update({number:pose})
        players_2d_dict.update({i:players_2d})

    return players_2d_dict


################################################################################
# Dumping Videos
################################################################################
def dump_csv_video_poses(data_dict, csv_players, vidtype, fps=25.0, scale=1, mot_tracks=None, one_color=True):
    if vidtype not in ['test']:
        raise Exception('Uknown video format')

    glog.info('Dumping {0} video'.format(vidtype))

    for i in data_dict:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4V
        out_file = join('/home/bunert/Data/results', data_dict[i].name +'_{0}.mp4'.format(vidtype))
        # FPS: 5.0
        out = cv2.VideoWriter(out_file, fourcc, fps,
                              (data_dict[i].shape[1] // scale, data_dict[i].shape[0] // scale))

        font = cv2.FONT_HERSHEY_SIMPLEX
        cmap = matplotlib.cm.get_cmap('hsv')
        if mot_tracks is not None:
            n_tracks = max(np.unique(mot_tracks[:, 1]))

        for frame, basename in enumerate(tqdm(data_dict[i].frame_basenames)):
            players_3d = init_all_3d_players(csv_players, frame)
            projected_players_2d_dict = project_players_allCameras_2D(data_dict, players_3d, frame)

            img = data_dict[i].get_frame(frame, dtype=np.uint8)

            # draw_skeleton_on_image_2dposes(img, poses, cmap_fun, one_color=False, pose_color=None)

            if vidtype == 'test':
                # Pose (db_cam, players_2d, frame, player=False)
                for k in range (len(projected_players_2d_dict[i])):
                    draw.draw_skeleton_on_image_2dposes(img, projected_players_2d_dict[i][k], cmap, one_color=True)

            if vidtype == 'kalman':
                # Pose
                for k in range (len(projected_players_2d_dict[i])):
                    draw.draw_skeleton_on_image_2dposes(img, projected_players_2d_dict[i][k], cmap, one_color=True)


            img = cv2.resize(img, (data_dict[i].shape[1] // scale, data_dict[i].shape[0] // scale))
            out.write(np.uint8(img[:, :, (2, 1, 0)]))

        # Release everything if job is finished
        out.release()
        cv2.destroyAllWindows()

def dump_video_poses(data_dict, all_players_3d, vidtype, fps=25.0, scale=1, mot_tracks=None, one_color=True):
    if vidtype not in ['kalman']:
        raise Exception('Uknown video format')

    glog.info('Dumping {0} video'.format(vidtype))

    for i in data_dict:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4V
        out_file = join('/home/bunert/Data/results', data_dict[i].name +'_{0}.mp4'.format(vidtype))
        # FPS: 10.0
        out = cv2.VideoWriter(out_file, fourcc, fps,
                              (data_dict[i].shape[1] // scale, data_dict[i].shape[0] // scale))

        font = cv2.FONT_HERSHEY_SIMPLEX
        cmap = matplotlib.cm.get_cmap('hsv')
        if mot_tracks is not None:
            n_tracks = max(np.unique(mot_tracks[:, 1]))

        for frame, basename in enumerate(tqdm(data_dict[i].frame_basenames)):
            if (frame > 100):
                break
            players_3d_dict = state_to_player_3d_dict(all_players_3d[frame])
            projected_players_2d_dict = project_players_allCameras_2D(data_dict, players_3d_dict, frame)

            img = data_dict[i].get_frame(frame, dtype=np.uint8)

            # draw_skeleton_on_image_2dposes(img, poses, cmap_fun, one_color=False, pose_color=None)

            if vidtype == 'test':
                # Pose (db_cam, players_2d, frame, player=False)
                for k in range (len(projected_players_2d_dict[i])):
                    draw.draw_skeleton_on_image_2dposes(img, projected_players_2d_dict[i][k], cmap, one_color=True)

            if vidtype == 'kalman':
                # Pose
                for k in range (len(projected_players_2d_dict[i])):
                    draw.draw_skeleton_on_image_2dposes(img, projected_players_2d_dict[i][k], cmap, one_color=True)


            img = cv2.resize(img, (data_dict[i].shape[1] // scale, data_dict[i].shape[0] // scale))
            out.write(np.uint8(img[:, :, (2, 1, 0)]))

        # Release everything if job is finished
        out.release()
        cv2.destroyAllWindows()


################################################################################
################################################################################
# Kalman Logic
################################################################################
################################################################################

def get_actual_z_vector(data_dict, players_2d_dict, player_number, projected_new_z):
    actual_z = np.zeros((number_of_keypoints * (2*number_of_cameras),1))
    #print("player number: ", player_number)

    for camera in data_dict:

        camera_offset = camera*2

        # openpose keypoint available
        if player_number in players_2d_dict[camera].keys():
            #print("Openpose keypoint for camera ", camera, " exists")

            for index in range (number_of_keypoints):

                actual_index = index*(2*number_of_cameras) + camera_offset

                if (players_2d_dict[camera][player_number][index,2] == 0): #confidence = 0
                    actual_z[actual_index]   = projected_new_z[actual_index]
                    actual_z[actual_index+1] = projected_new_z[actual_index+1]
                else:
                    actual_z[actual_index]   = players_2d_dict[camera][player_number][index, 0]
                    actual_z[actual_index+1] = players_2d_dict[camera][player_number][index, 1]

        # no openpose keypoint exists
        else:
            #print("No Openpose keypoint for camera", camera)
            for index in range (number_of_keypoints):
                actual_index = index*(2*number_of_cameras) + camera_offset
                actual_z[actual_index]   = projected_new_z[actual_index]
                actual_z[actual_index+1] = projected_new_z[actual_index+1]

    return actual_z

def state_to_player_3d_dict(all_players_3d):
    players = {}
    for numb in range(len(all_players_3d)):
        body = []
        for index in range(number_of_keypoints):
            x, y, z = all_players_3d[numb][index*6,0], all_players_3d[numb][(index*6)+1,0], all_players_3d[numb][(index*6)+2,0]
            body.append([x,y,z])

        body = np.asmatrix(body)
        players.update({numb:body})


    return players





################################################################################
################################################################################
# Main Function
################################################################################
################################################################################
def main():

    # Read camera data
    data_dict = init_soccerdata([0,1,2,3,4])

    # Read data from csv files
    csv_players = init_csv()

    # dict of all player in 3D (0-10 Denmar, 11-21 Swiss)
    players_3d_dict = init_all_3d_players(csv_players, 0)


    ################################################################################
    # Kalman Filter:
    ################################################################################
    # list to save all state vectors x
    # all_players_3d[frame][player]
    all_players_3d = []

    # prepare a kalman filter for every player
    kalman_dict = {}
    for i in players_3d_dict:
        kalman_dict.update({i:filter.Kalman(5, 18)})
        kalman_dict[i].initialize_state(players_3d_dict[i])

    for actual_frame in range (number_of_frames):
        if (actual_frame > 100):
            break
        print("Kalman iteration for frame: ", actual_frame)
        # get dict for every player with the 3D points
        actual_players_3d_dict = init_all_3d_players(csv_players, actual_frame)

        # dictionary for every camera, project the 3D players into camera coordinates
        projected_players_2d_dict = project_players_allCameras_2D(data_dict, actual_players_3d_dict, actual_frame)

        # unordered openpose keypoints
        keypoint_dict = get_actual_2D_keypoints(data_dict, actual_frame)

        # assigned the unordered openposes to player numbers (smallest distance)
        players_2d_dict = assign_player_to_poses(projected_players_2d_dict, keypoint_dict)
        players = []
        for player_number in kalman_dict:
            # predict next state
            new_state = kalman_dict[player_number].predict()

            # project the new state vector x into measurement space (2D)
            projected_new_z = kalman_dict[player_number].Hx(new_state, data_dict)

            # get actual measurement vector from openpose or if not available from prediction
            actual_z = get_actual_z_vector(data_dict, players_2d_dict, player_number, projected_new_z)

            # update the EKF
            kalman_dict[player_number].update(actual_z, data_dict)

            # add state vector x to player list
            players.append(kalman_dict[player_number].filter.x)

        # add all state vectors for one iteration to the list
        all_players_3d.append(players)

    # dump video with the state vectors
    dump_video_poses(data_dict, all_players_3d, 'kalman', fps=5.0)


    ################################################################################
    # initialization:
    ################################################################################
    # Read camera data
    # data_dict = {0:db_K1, 1:db_K8, 2:db_K9, 3:db_Left, 4:db_Right}
    # data_dict = init_soccerdata([0])
    # data_dict = init_soccerdata([0,1,2,3,4])

    # Read data from csv files
    # csv_players = init_csv()

    # dict of all player in 3D (0-10 Denmar, 11-21 Swiss) -> matrix of the keypoints
    # initialized from x-z positions from csv files
    # players_3d[player_number][keypoint]
    # players_3d_dict = init_all_3d_players(csv_players, 0)


    # Project the 3D players to get dictionaries with the player numbers as key (0-10 Danmark, 11-21 Swiss)
    # projected_players_2d_dict[camera][player_number][keypoint]: [[x,y]]
    # projected_players_2d_dict = project_players_allCameras_2D(data_dict, players_3d_dict, 0)


    # Openposes (from SoccerVideo DB) in it for a specific frame number -> unordered
    # keypoint_dict[camera][person][keypoint]: [x, y, prec.]
    # keypoint_dict = get_actual_2D_keypoints(data_dict, 0)


    # Keypoints from openpose assigned to player_number as dictionary (0-10 Danmark, 11-21 Swiss)
    # players_2d_dict[camera][player_number][keypoint]: [x,y,prec]
    # players_2d_dict = assign_player_to_poses(projected_players_2d_dict, keypoint_dict)

    ################################################################################
    # To plot the actual 3D players in 3D with plotly
    ################################################################################
    # draw.plot_all_players(players_3d_dict)


    ################################################################################
    # Project the 3D players on the image (frame)
    ################################################################################
    # for i in data_dict:
    #     draw.project_3d_players_on_image(data_dict[i], players_3d_dict, 0)

    ###############################################################################
    # Draw all players (2D dictionary) [x,y,prec] on the image (frame 0) from one Kamera data_dict[0]:
    # if just one player, last argument specifies the players number (default=False -> all players)
    ################################################################################
    # for i in data_dict:
    #     draw.draw_openpose_on_image(data_dict[i], players_2d_dict[i], 0, player=False)


    ###############################################################################
    # Draw all players (2D dictionary) [x,y] on the image (frame 0) from one Kamera data_dict[0]:
    # if just one player, last argument specifies the players number (default=False -> all players)
    ################################################################################
    # for i in data_dict:
    #    draw.draw_2d_players_on_image(data_dict[i], projected_players_2d_dict[i], 0, player=False)

    ###############################################################################
    # Dumps a Video for every Camera with the generic skeletons from CSV location projected
    ###############################################################################
    # dump_csv_video_poses(data_dict, csv_players, 'test', fps=5.0)

if __name__ == "__main__":
    main()
