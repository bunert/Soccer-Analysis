import numpy as np
import os
from os.path import isfile, join, exists
from os import listdir
from tqdm import tqdm
import json
import cv2
import pickle
import glog
import yaml
import matplotlib
import pycocotools.mask as mask_util
# import utils
#
# import utils.io as io
# import utils.misc as misc_utils
# import utils.camera as cam_utils
# import utils.draw as draw_utils
# import utils.files as file_utils
#
# from utils.nms.nms_wrapper import nms


class SoccerVideo:
    def __init__(self, path_to_dataset):
        image_extensions = ['jpg', 'png']

        # Load images
        self.path_to_dataset = path_to_dataset

        # all images name
        self.frame_basenames = [f for f in listdir(join(path_to_dataset, 'images'))
                                if isfile(join(path_to_dataset, 'images', f)) and any(i in f for i in image_extensions)]

        self.frame_fullnames = [join(path_to_dataset, 'images', f) for f in self.frame_basenames]

        # remove '.jpg' or '.png' ending
        self.frame_basenames = [f[:-4] for f in self.frame_basenames]

        self.frame_basenames.sort()
        self.frame_fullnames.sort()

        # save the poses from openpose
        self.poses = {f: None for f in self.frame_basenames}

    # ---------------------------------------------------------------------------
    # customized from tabletop
    # estimates the poses with openpose and saves them in class SoccerVideo.poses per frame
    def estimate_openpose(self, openpose_dir='/path/to/openpose'):
        openposebin = './build/examples/openpose/openpose.bin'

        # tmp directory to store the output of openpose
        tmp_dir = join(self.path_to_dataset, 'tmp')
        if not exists(tmp_dir):
            os.mkdir(tmp_dir)

        for i, basename in enumerate(tqdm(self.frame_basenames)):
            # Remove previous stored files
            previous_files = [f for f in os.listdir(tmp_dir)]
            for f in previous_files:
                os.remove(join(tmp_dir, f))

            cwd = os.getcwd()
            os.chdir(openpose_dir)
            # openpose demo which gets executed with arguments as specified
            # model_pose COCO or default?
            # --maximize_positives: lower threshold -> more detections but less correct
            command = '{0} --model_pose COCO --image_dir {1} --write_json {2} --maximize_positives'.format(
                openposebin, join(self.path_to_dataset, 'images'), tmp_dir)
            os.system(command)
            os.chdir(cwd)

            # achtung: format of output file?
            poses = []
            with open(join(join(self.path_to_dataset, 'tmp'), '{0}_keypoints.json'.format(basename))) as data_file:
                data_json = json.load(data_file)

                if len(data_json['people']) == 0:
                    continue
                n_persons = len(data_json['people'])

                # extract the x,y for the keypoints for all persons
                for k in range(n_persons):
                    keypoints_ = np.array(data_json['people'][k]
                                          ['pose_keypoints_2d']).reshape((18, 3))
                    # keypoints_[:, 0] += x1
                    # keypoints_[:, 1] += y1
                    poses.append(keypoints_)
            self.poses[basename] = poses
        return 0

    # ---------------------------------------------------------------------------
    # copied from tabletop
    # - removes all poses with less keypoints than keypoint_thresh
    # - removes all poses where the neck doesn't pass the nms test (utils.nms.nms_wrapper)
    #   https://github.com/rbgirshick/fast-rcnn/blob/master/lib/utils/nms.py
    # - (comment out) remove poses outside of field
    def refine_poses(self, keypoint_thresh=10, score_thresh=0.5, neck_thresh=0.59, margin=0.0):
        # W, H = 104.73, 67.74

        for i, basename in enumerate(tqdm(self.frame_basenames)):
            poses = self.poses[basename]

            # remove the poses with few keypoints or they
            keep = []
            for ii in range(len(poses)):
                keypoints = poses[ii]
                valid = (keypoints[:, 2] > 0.).nonzero()[0]
                score = np.sum(keypoints[valid, 2])

                if len(valid) > keypoint_thresh and score > score_thresh and keypoints[1, 2] > neck_thresh:
                    keep.append(ii)

            poses = [poses[ii] for ii in keep]

            root_part = 1
            root_box = []
            for ii in range(len(poses)):
                root_tmp = poses[ii][root_part, :]
                valid_keypoints = (poses[ii][:, 2] > 0).nonzero()
                root_box.append(
                    [root_tmp[0] - 10, root_tmp[1] - 10, root_tmp[0] + 10, root_tmp[1] + 10,
                     np.sum(poses[ii][valid_keypoints, 2])])
            root_box = np.array(root_box)

            # Perform Neck NMS
            if len(root_box.shape) == 1:
                root_box = root_box[None, :]
                keep2 = [0]
            else:
                keep2 = nms(root_box.astype(np.float32), 0.1)

            poses = [poses[ii] for ii in keep2]

            # # Remove poses outside of field (camera not implemented)
            # keep3 = []
            # cam_mat = self.calib[basename]
            # cam = cam_utils.Camera(basename, cam_mat['A'], cam_mat['R'], cam_mat['T'], self.shape[0], self.shape[1])
            # for ii in range(len(poses)):
            #     kp3 = misc_utils.lift_keypoints_in_3d(cam, poses[ii])
            #     if (-W / 2. - margin) <= kp3[1, 0] <= (W / 2. + margin) and (-H / 2. - margin) <= kp3[1, 2] <= (H / 2. + margin):
            #         keep3.append(ii)
            #
            # poses = [poses[ii] for ii in keep3]

            self.poses[basename] = poses
