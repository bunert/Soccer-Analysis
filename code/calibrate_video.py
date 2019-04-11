import os
import argparse

import soccer
import utils.files as file_utils
from os.path import isfile, join, exists

################################################################################
# run: python3 calibrate_video.py --cameras 1
################################################################################

parser = argparse.ArgumentParser(description='Calibrate a soccer video')
# --path_to_data: where the images are
parser.add_argument('--path_to_data', default='/home/bunert/Data', help='path')
# --cameras: number of cameras
parser.add_argument('--cameras', default=8, type=int, help='path')
opt, _ = parser.parse_known_args()


db = []
for i in range(opt.cameras):
    db.append(soccer.SoccerVideo(join(opt.path_to_data, 'camera{0}'.format(i))))
    db[i].gather_detectron()
    db[i].digest_metadata()
    file_utils.mkdir(os.path.join(db[i].path_to_dataset, 'calib'))
    db[i].calibrate_camera()
    db[i].dump_video('calib')
