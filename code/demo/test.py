import os
import argparse
import soccer
from os import listdir
from os.path import isfile, join, exists
import pandas
import pickle
from scipy.linalg import block_diag, norm
import numpy as np
from filterpy.common import Q_discrete_white_noise
import utils.camera as cam_utils
import scipy

# CMD Line arguments
parser = argparse.ArgumentParser(description='Estimate the poses')
# --path_to_data: where the images are
parser.add_argument('--path_to_data', default='/home/bunert/Data/', help='path')

opt, _ = parser.parse_known_args()

db = soccer.SoccerVideo(os.path.join(opt.path_to_data, 'K1'))
db.digest_metadata()

cam = cam_utils.Camera('k1', db.calib[db.frame_basenames[30]]['A'], db.calib[db.frame_basenames[30]]
                       ['R'], db.calib[db.frame_basenames[30]]['T'], db.shape[0], db.shape[1])
print(cam.A)
