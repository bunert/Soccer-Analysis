import os
import argparse
import soccer
import filter
from os import listdir
from os.path import isfile, join, exists
import pandas
import cv2
import pickle
import numpy as np
import matplotlib
import plotly as py
import utils.draw as draw
import utils.camera as cam_utils
from tqdm import tqdm


import math
import sys
import glog

# to print full matrices
np.set_printoptions(threshold=sys.maxsize)


number_of_frames =  215  # set manually or take the n_frames value from the Soccer class
number_of_keypoints = 18
number_of_cameras = 2


################################################################################
# run: python3 demo/fusion.py --path_to_data ~/path/to/data
################################################################################


# CMD Line arguments
parser = argparse.ArgumentParser(description='Estimate the poses')
# --path_to_data: where the images are
parser.add_argument('--path_to_data', default='~/path/to/data/', help='path')

opt, _ = parser.parse_known_args()

################################################################################
# initialization of the data
################################################################################

# initialize databases for all cameras and load the data (COCO)
# modify for an other dataset
def init_soccerdata(mylist):
    # load corresponding metadata
    data_dict = {}
    if 0 in mylist:
        db_K1 = soccer.SoccerVideo(os.path.join(opt.path_to_data, 'K1'))
        db_K1.name = "K1"
        db_K1.digest_metadata()
        db_K1.refine_poses(keypoint_thresh=7, score_thresh=0.4, neck_thresh=0.4)
        data_dict.update({0:db_K1})
    if 1 in mylist:
        db_K8 = soccer.SoccerVideo(os.path.join(opt.path_to_data, 'K8'))
        db_K8.name = "K8"
        db_K8.digest_metadata()
        db_K8.refine_poses(keypoint_thresh=7, score_thresh=0.4, neck_thresh=0.4)
        data_dict.update({1:db_K8})
    if 2 in mylist:
        db_K9 = soccer.SoccerVideo(os.path.join(opt.path_to_data, 'K9'))
        db_K9.name = "K9"
        db_K9.digest_metadata()
        db_K9.refine_poses(keypoint_thresh=7, score_thresh=0.4, neck_thresh=0.4)
        data_dict.update({2:db_K9})
    if 3 in mylist:
        db_Left = soccer.SoccerVideo(os.path.join(opt.path_to_data, 'Left'))
        db_Left.name = "Left"
        db_Left.digest_metadata()
        db_Left.refine_poses(keypoint_thresh=7, score_thresh=0.4, neck_thresh=0.4)
        data_dict.update({3:db_Left})
    if 4 in mylist:
        db_Right = soccer.SoccerVideo(os.path.join(opt.path_to_data, 'Right'))
        db_Right.name = "Right"
        db_Right.digest_metadata()
        db_Right.refine_poses(keypoint_thresh=7, score_thresh=0.4, neck_thresh=0.4)
        data_dict.update({4:db_Right})

    return data_dict

# initialize the csv data
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

    ball = pandas.read_csv('/home/bunert/Data/Smash/Switzerland_-_Denmark_Fitness_RAW_data_Switzerland/lib/approximation_data/Ball.csv',
            sep = ';', decimal=",", skiprows=5, usecols=[0,1,2], names=['time', 'x', 'y'])

    ball_min = ball.min()
    ball_max = ball.max()
    W = abs(ball_max[1]- ball_min[1])
    H = abs(ball_max[2]- ball_min[2])

    for i in range(len(players)):
        players[i]['y'] -= 33.5
        # players[i]['x'] *= 1.03693
        players[i]['y'] *= -1

    return players

# Hardcoded uniform skeletons
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

# project each csv player on the field
def init_all_3d_players(csv_players, frame):
    players = {}
    for i in range(len(csv_players)):
        #according to the view direction for the two different teams
        if (i <= 10):
            players.update({i:init_3d_players(csv_players[i].iloc[frame][1],csv_players[i].iloc[frame][2],180)})
        else:
            players.update({i:init_3d_players(csv_players[i].iloc[frame][1],csv_players[i].iloc[frame][2],0)})

    return players

# Project all players to screen coordinates for the camera db_cam
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

# Get all poses from openpose for a specific frame
def get_actual_2D_keypoints(data_dict, frame):
    actual_keypoint_dict = {}
    for i in data_dict:
        frame_name = data_dict[i].frame_basenames[frame]
        actual_keypoint_dict.update({i: data_dict[i].poses[frame_name]})
    return actual_keypoint_dict

# Return dictionary for all cameras, with all the poses (players_3d projected on the camera screen)
def project_players_allCameras_2D(data_dict, players_3d, frame):
    projected_players_2d_dict = {}
    for i in data_dict:
        projected_players_2d_dict.update({i:project_players_2D(data_dict[i], players_3d, frame)})

    return projected_players_2d_dict

# return nearest player of openpose poses to all the projected players
def nearest_player(keypoints, projected_players_2d):
    minimal_distance = sys.float_info.max
    number = False
    distance = 0.
    for i in projected_players_2d:
        numb = 0
        for k in range(len(keypoints)):
            if (keypoints[k][2] != 0):
                numb += 1
                x1, y1 = keypoints[k][0], keypoints[k][1]
                x2, y2 = projected_players_2d[i][k][0][0], projected_players_2d[i][k][0][1]
                distance += math.sqrt(math.pow(x1-x2,2)+ math.pow(y1-y2,2))

        distance = distance / numb
        # update if distance in smaller
        if (distance < minimal_distance):
            minimal_distance = distance
            number = i
            pose = keypoints

    return number, pose

# Return dictionary for all cameras, with all the poses as tuples with corresponding player numbering
def assign_player_to_poses(projected_players_2d_dict, keypoint_dict):
    players_2d_dict = {}
    for i in keypoint_dict:
        players_2d = {}
        for k in range(len(keypoint_dict[i])):
            number,pose = nearest_player(keypoint_dict[i][k], projected_players_2d_dict[i])
            players_2d.update({number:pose})
        players_2d_dict.update({i:players_2d})

    return players_2d_dict

# Dump a video with the poses of the uniform skeletons
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

# Dump a video with the posses from all_players_3d_array for one KF
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
            if (frame >= len(all_players_3d)):
                break
            players_3d_dict = state_to_player_3d_dict(all_players_3d[frame])
            projected_players_2d_dict = project_players_allCameras_2D(data_dict, players_3d_dict, frame)

            img = data_dict[i].get_frame(frame, dtype=np.uint8)

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

# Dump a video with the posses from all_players_3d_array for multiple KF
def dump_video_multiple_poses(data_dict, all_players_3d_array, vidtype, rq,  fps=25.0, scale=1, mot_tracks=None, one_color=True):
    if vidtype not in ['kalman']:
        raise Exception('Uknown video format')

    glog.info('Dumping {0} video'.format(vidtype))

    length = len(all_players_3d_array[0])

    for i in data_dict:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4V
        out_file = join('/home/bunert/Data/results', data_dict[i].name +'_{0}.mp4'.format(vidtype))
        out = cv2.VideoWriter(out_file, fourcc, fps,
                              (data_dict[i].shape[1] // scale, data_dict[i].shape[0] // scale))

        font = cv2.FONT_HERSHEY_SIMPLEX
        cmap = matplotlib.cm.get_cmap('hsv')
        if mot_tracks is not None:
            n_tracks = max(np.unique(mot_tracks[:, 1]))

        # rgb(30, 30, 250)
        # rgb(60, 180, 30)
        # rgb(250, 10, 10)

        listOfColors = [(30, 30, 250), (250, 10, 10)]
        for frame, basename in enumerate(tqdm(data_dict[i].frame_basenames)):
            if (frame >= length):
                break

            img = data_dict[i].get_frame(frame, dtype=np.uint8)

            for j in range (len(all_players_3d_array)):
                players_3d_dict = state_to_player_3d_dict(all_players_3d_array[j][frame])
                projected_players_2d_dict = project_players_allCameras_2D(data_dict, players_3d_dict, frame)

                if (len(all_players_3d_array)==1):
                    if vidtype == 'kalman':
                        # Pose
                        for k in range (len(projected_players_2d_dict[i])):
                            draw.draw_skeleton_on_image_2dposes(img, projected_players_2d_dict[i][k], cmap, one_color=True)

                # draw_skeleton_on_image_2dposes(img, poses, cmap_fun, one_color=False, pose_color=None)
                else:
                    if vidtype == 'kalman':
                        # Pose
                        for k in range (len(projected_players_2d_dict[i])):
                            draw.draw_skeleton_on_image_2dposes_color(img, projected_players_2d_dict[i][k], listOfColors[j], 5-3*j)


            font                   = cv2.FONT_HERSHEY_SIMPLEX
            fontScale              = 1
            fontColor              = (255,255,255)
            lineType               = 2
            cv2.putText(img,'frame={0}'.format(frame), (data_dict[i].shape[1] -450 ,50), font, fontScale,fontColor, lineType)
            cv2.putText(img,'Poses1: R={0}, Q={1}'.format(rq[0], rq[1]), (data_dict[i].shape[1] -450 ,100), font, fontScale,listOfColors[0], lineType)
            if (len(rq) > 3):
                cv2.putText(img,'Poses2: R={0}, Q={1}'.format(rq[2], rq[3]), (data_dict[i].shape[1] -450 ,150), font, fontScale,listOfColors[1], lineType)

            img = cv2.resize(img, (data_dict[i].shape[1] // scale, data_dict[i].shape[0] // scale))
            out.write(np.uint8(img[:, :, (2, 1, 0)]))

        # Release everything if job is finished
        out.release()
        cv2.destroyAllWindows()

# Get the actual z vector for the KF -> workaround to get all poses
def get_actual_z_vector(data_dict, players_2d_dict, player_number, projected_new_z, kalman_dict, frame):
    actual_z = np.zeros((number_of_keypoints * (2*number_of_cameras),1))

    for camera in data_dict:
        if (number_of_cameras==2):
            camera_offset = (camera-1)*2
        else:
            camera_offset = (camera)*2
        # openpose keypoint available
        if player_number in players_2d_dict[camera].keys():
            #print("Openpose key exists")
            head_x, head_y  = players_2d_dict[camera][player_number][0,0], players_2d_dict[camera][player_number][0,1]
            left_shoulder_x, left_shoulder_y, left_shoulder_conf = players_2d_dict[camera][player_number][5,0], players_2d_dict[camera][player_number][5,1], players_2d_dict[camera][player_number][5,2]
            right_shoulder_x, right_shoulder_y, right_shoulder_conf = players_2d_dict[camera][player_number][2,0], players_2d_dict[camera][player_number][2,1], players_2d_dict[camera][player_number][2,2]
            neck_x, neck_y, neck_conf = players_2d_dict[camera][player_number][1,0], players_2d_dict[camera][player_number][1,1], players_2d_dict[camera][player_number][1,2]

            # set the shoulder keypoint when one is missing
            if (left_shoulder_conf == 0 and right_shoulder_conf == 0):
                # print("Both shoulders conf == 0")
            elif (left_shoulder_conf == 0):
                # print("Left shoulder set")
                if (neck_x >= right_shoulder_x):
                    left_shoulder_x = neck_x + (neck_x -right_shoulder_x)
                else:
                    left_shoulder_x = neck_x - (right_shoulder_x - neck_x)
                if (neck_y >= right_shoulder_y):
                    left_shoulder_y = neck_y + (neck_y - right_shoulder_y)
                else:
                    left_shoulder_y = neck_y - (right_shoulder_y - neck_y)
            elif (right_shoulder_conf == 0):
                # print("Right shoulder set")
                if (neck_x >= left_shoulder_x):
                    right_shoulder_x = neck_x + (neck_x - left_shoulder_x)
                else:
                    right_shoulder_x = neck_x - (left_shoulder_x - neck_x)
                if (neck_y >= left_shoulder_y):
                    right_shoulder_y = neck_y + (neck_y - left_shoulder_y)
                else:
                    right_shoulder_y = neck_y - (left_shoulder_y - neck_y)

            for index in range (number_of_keypoints):
                actual_index = index*(2*number_of_cameras) + camera_offset
                if (players_2d_dict[camera][player_number][index,2] != 0): # confidence != 0
                    #print("Method: normal openpose keypoint taken ", index)
                    actual_z[actual_index]   = players_2d_dict[camera][player_number][index, 0]
                    actual_z[actual_index+1] = players_2d_dict[camera][player_number][index, 1]

                else: #confidence == 0

                    if (neck_conf == 0):
                        #print ("NECK CONF == 0!")

                    # Head Keypoint set
                    if (index == 0):
                        right_hip_x, right_hip_y, right_hip_conf = players_2d_dict[camera][player_number][8,0], players_2d_dict[camera][player_number][8,1], players_2d_dict[camera][player_number][8,2]
                        left_hip_x, left_hip_y, left_hip_conf = players_2d_dict[camera][player_number][11,0], players_2d_dict[camera][player_number][11,1], players_2d_dict[camera][player_number][11,2]
                        right_hand_x, right_hand_y, right_hand_conf = players_2d_dict[camera][player_number][4,0], players_2d_dict[camera][player_number][4,1], players_2d_dict[camera][player_number][4,2]
                        left_hand_x, left_hand_y, left_hand_conf = players_2d_dict[camera][player_number][7,0], players_2d_dict[camera][player_number][7,1], players_2d_dict[camera][player_number][7,2]

                        if (right_hip_conf != 0):
                            dist =  math.sqrt(math.pow(neck_x-right_hip_x,2)+ math.pow(neck_y-right_hip_y,2))
                            dist = 4./10. * dist
                        elif (left_hip_conf != 0):
                            dist =  math.sqrt(math.pow(neck_x-left_hip_x,2)+ math.pow(neck_y-left_hip_y,2))
                            dist = 4./10. * dist
                        elif (right_hand_conf != 0):
                            dist =  math.sqrt(math.pow(neck_x-right_hand_x,2)+ math.pow(neck_y-right_hand_y,2))
                            dist = 3./10. * dist
                        elif (left_hand_conf != 0):
                            dist =  math.sqrt(math.pow(neck_x-left_hand_x,2)+ math.pow(neck_y-left_hand_y,2))
                            dist = 3./10. * dist
                        else:
                            #print("HIP FAILED")
                            dist = 10
                        actual_z[actual_index] = neck_x
                        head_x = neck_x
                        actual_z[actual_index+1] = neck_y - dist
                        head_y = neck_y - dist

                    # Shoulders already set
                    elif (index == 2):
                        actual_z[actual_index] = right_shoulder_x
                        actual_z[actual_index+1] = right_shoulder_y
                    elif (index == 5):
                        actual_z[actual_index] = left_shoulder_x
                        actual_z[actual_index+1] = left_shoulder_y

                    # if keypoint 14,15,16,17 (head) take hardcoded values (proportional to shoulder)
                    elif (index in [14,15,16,17] and (left_shoulder_conf != 0 or right_shoulder_conf != 0)):
                        #print("Method: head Keypoint hardcoded ", index)
                        if (index == 14):
                            if (right_shoulder_x <= head_x):
                                actual_z[actual_index] = head_x + ((head_x - right_shoulder_x)/10.)
                            else:
                                actual_z[actual_index] = head_x - ((right_shoulder_x - head_x)/10.)

                            actual_z[actual_index+1] = head_y - (head_y - right_shoulder_y)/10.

                        elif (index == 15):
                            if (left_shoulder_x <= head_x):
                                actual_z[actual_index] = head_x - ((head_x - left_shoulder_x)/10.)
                            else:
                                actual_z[actual_index] = head_x + ((left_shoulder_x - head_x)/10.)

                            actual_z[actual_index+1] = head_y - (head_y - left_shoulder_y)/10.

                        elif (index == 16):
                            actual_z[actual_index]   = (head_x + right_shoulder_x)/2.
                            actual_z[actual_index+1] = head_y

                        elif (index == 17):
                            actual_z[actual_index]   = (head_x + left_shoulder_x)/2.
                            actual_z[actual_index+1] = head_y

                        else: # should not enter this section
                            print("Head point not updated for actual measurement vector.")
                            exit()


                    else: # confidence == 0 but not a head point or shoulder&head also not found
                        if (frame > 0):
                            #print("Method: take z from last iteration ", index)
                            actual_z[actual_index] = kalman_dict.filter.z[actual_index]
                            actual_z[actual_index+1] = kalman_dict.filter.z[actual_index+1]
                        else:
                            #print("Method: take predicted keypoint ", index)
                            actual_z[actual_index]   = projected_new_z[actual_index]
                            actual_z[actual_index+1] = projected_new_z[actual_index+1]

        # no openpose keypoint exists
        else:
            #print("Camera: "+  str(camera) + " - Openpose not available for player: "+ str(player_number))
            for index in range (number_of_keypoints):
                actual_index = index*(2*number_of_cameras) + camera_offset
                actual_z[actual_index]   = projected_new_z[actual_index]
                actual_z[actual_index+1] = projected_new_z[actual_index+1]

    return actual_z

# player dictionary to state vector
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

# Kalman Filter iterations:
def iterate_kalman(data_dict, csv_players, players_3d_dict, R, Q, number_of_iterations):
    all_players_3d = []

    # prepare a kalman filter for every player
    R_std = R
    Q_var = Q
    kalman_dict = {}
    for i in players_3d_dict:
        kalman_dict.update({i:filter.Kalman(number_of_cameras, 18, R_std=R_std, Q_var=Q_var)})
        kalman_dict[i].initialize_state(players_3d_dict[i])


    for actual_frame in range (number_of_frames):
        if (actual_frame > number_of_iterations):
            break
        print("\n Kalman iteration for frame: ", actual_frame)
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
            actual_z = get_actual_z_vector(data_dict, players_2d_dict, player_number, projected_new_z, kalman_dict[player_number], actual_frame)

            # update the EKF
            kalman_dict[player_number].update(actual_z, data_dict)

            # add state vector x to player list
            players.append(kalman_dict[player_number].filter.x)

        # add all state vectors for one iteration to the list
        all_players_3d.append(players)

        # to store the the data i a pickle file
        store_data = join('/home/bunert/Data/results/', 'R={0}_Q={1}_data.p'.format(R_std,Q_var))
        with open(store_data, 'wb') as f:
            pickle.dump(all_players_3d, f)


    return all_players_3d

################################################################################
# Main Function
################################################################################
def main():

    # Read camera data
    data_dict = init_soccerdata([1,2]) #([0,1,2])

    # Read data from csv files
    csv_players = init_csv()

    # dict of all player in 3D (0-10 Denmar, 11-21 Swiss)
    players_3d_dict = init_all_3d_players(csv_players, 0)

    # values for R_std and Q_var for the first EKF
    # R= .1. Q= .5
    r1, q1 = .1, .5

    # values for R_std and Q_var for the second EKF
    r2, q2 = .1, .5

    # number of frames to iterate
    number_of_iterations = 50


    test1 = iterate_kalman(data_dict, csv_players, players_3d_dict, r1, q1, number_of_iterations)
    #test2 = iterate_kalman(data_dict, csv_players, players_3d_dict, r2, q2, number_of_iterations)

    all_players_3d_array = [test1]#, test2]
    rq = [r1, q1]#, r2, q2]

    # to load the pickle data:
    # store_data = join('/home/bunert/Data/results/', 'Two.p')
    # with open(store_data, 'rb') as f:
    #     test1 = pickle.load(f)
    # store_data = join('/home/bunert/Data/results/', 'Three.p')
    # with open(store_data, 'rb') as f:
    #     test2 = pickle.load(f)
    # all_players_3d_array = [test1,test2]

    # dump video with the state vectors
    # dump_video_poses(data_dict, test1, 'kalman', fps=2.0)
    # dump_video_multiple_poses(data_dict, all_players_3d_array, 'kalman', rq=rq, fps=5.0)



if __name__ == "__main__":
    main()
