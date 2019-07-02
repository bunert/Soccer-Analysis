import os
import argparse
import soccer
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



################################################################################
# run: python3 demo/fusion.py --path_to_data ~/Data/
################################################################################


# CMD Line arguments
parser = argparse.ArgumentParser(description='Estimate the poses')
# --path_to_data: where the images are
parser.add_argument('--path_to_data', default='/home/bunert/Data/', help='path')

opt, _ = parser.parse_known_args()

################################################################################
# initialization of the data
################################################################################


################################################################################
# initialize databases for all cameras and load the data (COCO)
# names: db_K1, db_K8, db_K9, db_Left, db_Right
# example: print(db_K1.poses["00000009"][0])
################################################################################
def init_soccerdata():
    # load corresponding metadata
    db_K1 = soccer.SoccerVideo(os.path.join(opt.path_to_data, 'K1'))
    db_K8 = soccer.SoccerVideo(os.path.join(opt.path_to_data, 'K8'))
    db_K9 = soccer.SoccerVideo(os.path.join(opt.path_to_data, 'K9'))
    db_Left = soccer.SoccerVideo(os.path.join(opt.path_to_data, 'Left'))
    db_Right = soccer.SoccerVideo(os.path.join(opt.path_to_data, 'Right'))


    db_K1.digest_metadata()
    db_K8.digest_metadata()
    db_K9.digest_metadata()
    db_Left.digest_metadata()
    db_Right.digest_metadata()

    db_K1.refine_poses(keypoint_thresh=7, score_thresh=0.4, neck_thresh=0.4)
    db_K8.refine_poses(keypoint_thresh=7, score_thresh=0.4, neck_thresh=0.4)
    db_K9.refine_poses(keypoint_thresh=7, score_thresh=0.4, neck_thresh=0.4)
    db_Left.refine_poses(keypoint_thresh=7, score_thresh=0.4, neck_thresh=0.4)
    db_Right.refine_poses(keypoint_thresh=7, score_thresh=0.4, neck_thresh=0.4)
    return db_K1, db_K8, db_K9, db_Left, db_Right

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
# Blickrichtung: positive X
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
def init_all_3d_players():
    # Read data from csv files
    csv_players = init_csv()

    players = []

    for i in range(len(csv_players)):
        #Blickrichtung,
        if (i <= 10):
            players.append(init_3d_players(csv_players[i].iloc[0][1],csv_players[i].iloc[0][2],180))
        else:
            players.append(init_3d_players(csv_players[i].iloc[0][1],csv_players[i].iloc[0][2],0))

    return players

################################################################################
# Project all players on the first image from one Kamera
################################################################################
def project_players(db_cam, players_3d, frame):
    frame_name = db_cam.frame_basenames[frame]
    camera = cam_utils.Camera("Cam", db_cam.calib[frame_name]['A'], db_cam.calib[frame_name]['R'], db_cam.calib[frame_name]['T'], db_cam.shape[0], db_cam.shape[1])

    cmap = matplotlib.cm.get_cmap('hsv')
    img = db_cam.get_frame(frame, dtype=np.uint8)
    for k in range(len(players_3d)):
        points2d = []
        for i in range(len(players_3d[k])):
            tmp, depth = camera.project(players_3d[k][i])
            behind_points = (depth < 0).nonzero()[0]
            tmp[behind_points, :] *= -1
            points2d.append(tmp)
        draw_utils.draw_skeleton_on_image_2dposes(img, points2d, cmap, one_color=True)

    cv2.imwrite('/home/bunert/Data/test.png',np.uint8(img[:, :, (2, 1, 0)]))

################################################################################
# Metric logic
################################################################################




def main():

    ################################################################################
    # initialization
    ################################################################################
    # Read camera data
    # db_K1, db_K8, db_K9, db_Left, db_Right = init_soccerdata()

    data_dict = {0: db_K1, 1: db_K8, 2: db_K9, 3: db_Left, 4: db_Right}

    print(data_dict[1])
    exit()

    # Testing with only one camera:
    # TODO: remove later
    db_K1 = soccer.SoccerVideo(os.path.join(opt.path_to_data, 'K1'))
    db_K1.digest_metadata()
    db_K1.refine_poses(keypoint_thresh=7, score_thresh=0.4, neck_thresh=0.4)

    # all 3D player positions from csv x-z location
    # use: players_3d[x] -> matrix of player number x with each row a keypoint
    players_3d = init_all_3d_players()

    ################################################################################
    # To plot the acutal 3D Player List in 3D
    ################################################################################
    # draw.plot_all_players(players_3d)


    ################################################################################
    # Project all players on the first (frame 0) image from one Kamera
    ################################################################################
    # project_players(db_K1, players_3d, 0)


    print(db_K1.poses["00000009"][0][0])







if __name__ == "__main__":
    main()
