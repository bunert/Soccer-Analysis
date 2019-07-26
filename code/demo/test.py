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

index = 16

print(index in [16,17,18,19])
