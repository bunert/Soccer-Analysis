import os
import argparse
import soccer
from os import listdir
from os.path import isfile, join, exists


################################################################################
# run: python3 refine_openpose.py --path_to_data ~/path/to/data/
################################################################################


# CMD Line arguments
parser = argparse.ArgumentParser(description='Estimate the poses')
# --path_to_data: where the images are
parser.add_argument('--path_to_data', default='~/path/to/data', help='path')

opt, _ = parser.parse_known_args()

# initialize SoccerVideo for every camera
db = soccer.SoccerVideo(opt.path_to_data)
db.gather_detectron()
db.digest_metadata()

db.refine_poses(keypoint_thresh=7, score_thresh=0.4, neck_thresh=0.4)
db.dump_video('poses')
