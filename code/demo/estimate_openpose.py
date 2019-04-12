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
parser.add_argument('--path_to_data', default='/home/bunert/Data', help='path')
# --openpose_dir: where the openpose directory is (./build/examples/openpose/openpose.bin)
parser.add_argument('--openpose_dir', default='/home/bunert/installations/openpose', help='path')
# --cameras: number of cameras
parser.add_argument('--cameras', default=8, type=int, help='path')

opt, _ = parser.parse_known_args()

# initialize SoccerVideo for every camera
db = []

for i in range(opt.cameras):
    db.append(soccer.SoccerVideo(join(opt.path_to_data, 'camera{0}'.format(i))))

    # what exactly do those?
    db[i].gather_detectron()
    db[i].digest_metadata()
    db[i].get_boxes_from_detectron()
    db[i].dump_video('detections')

    db[i].estimate_openposes(openpose_dir=opt.openpose_dir)
    db[i].refine_poses(keypoint_thresh=7, score_thresh=0.4, neck_thresh=0.4)
    db[i].dump_video('poses')
