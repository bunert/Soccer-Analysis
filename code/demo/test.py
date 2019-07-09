import os
import argparse
import soccer
from os import listdir
from os.path import isfile, join, exists
import pandas
from scipy.linalg import block_diag, norm
import numpy as np
from filterpy.common import Q_discrete_white_noise
q = Q_discrete_white_noise(dim=3, dt=0.04, var=0.3)
print(q)
