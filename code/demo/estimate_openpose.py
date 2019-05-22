import os
import argparse
import soccer
from os import listdir
from os.path import isfile, join, exists


################################################################################
# run: python3 estimate_openpose.py --cameras 1
################################################################################


# CMD Line arguments
parser = argparse.ArgumentParser(description='Estimate the poses')
# --path_to_data: where the images are
parser.add_argument('--path_to_data', default='/home/bunert/Data/camera0', help='path')
# --openpose_dir: where the openpose directory is (./build/examples/openpose/openpose.bin)
parser.add_argument('--openpose_dir', default='/home/bunert/installations/openpose', help='path')

opt, _ = parser.parse_known_args()

# initialize SoccerVideo for every camera
db = soccer.SoccerVideo(opt.path_to_data)
# what exactly do those?
db.gather_detectron()
db.digest_metadata()
db.get_boxes_from_detectron()
db.dump_video('detections')

db.estimate_openposes(openpose_dir=opt.openpose_dir)
#db.refine_poses(keypoint_thresh=7, score_thresh=0.4, neck_thresh=0.4)
db.dump_video('poses')
