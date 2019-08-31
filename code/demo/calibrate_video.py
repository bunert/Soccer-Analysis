import os
import argparse

import soccer
import utils.files as file_utils
from os.path import isfile, join, exists

################################################################################
# run: python3 calibrate_video.py --path_to_data ~/path/to/data
################################################################################

parser = argparse.ArgumentParser(description='Calibrate a soccer video')
# --path_to_data: where the images are
parser.add_argument('--path_to_data', default='~/path/to/data', help='path')
opt, _ = parser.parse_known_args()

db = soccer.SoccerVideo(opt.path_to_data)
db.gather_detectron()
db.digest_metadata()
file_utils.mkdir(os.path.join(db.path_to_dataset, 'calib'))
db.calibrate_camera()
db.dump_video('calib')
