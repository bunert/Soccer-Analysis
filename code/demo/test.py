import os
import argparse
import soccer
from os import listdir
from os.path import isfile, join, exists
import pandas
from scipy.linalg import block_diag, norm
import numpy as np
from filterpy.common import Q_discrete_white_noise
import utils.camera as cam_utils
import scipy

q = Q_discrete_white_noise(dim=2, dt=0.04 , var=0.3)
block = np.matrix([[q[0,0], 0., 0., q[0,1], 0., 0.],
                  [0., q[0,0], 0., 0., q[0,1], 0.],
                  [0., 0., q[0,0], 0., 0., q[0,1]],
                  [q[1,0], 0., 0., q[1,1], 0., 0.],
                  [0., q[1,0], 0., 0., q[1,1], 0.],
                  [0., 0., q[1,0], 0., 0., q[1,1]]])
matrix_list = []
for i in range (18):
    matrix_list.append(block)
Q = scipy.linalg.block_diag(*matrix_list)

print(Q.shape)
