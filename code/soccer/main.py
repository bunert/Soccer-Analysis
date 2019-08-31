import numpy as np
import os
from os.path import isfile, join, exists
from os import listdir
from tqdm import tqdm
from soccer import calibration
import json
import cv2
import pickle
import glog
import yaml
import matplotlib
import pycocotools.mask as mask_util
import utils
import utils.io as io
import utils.misc as misc_utils
import utils.camera as cam_utils
import utils.draw as draw_utils
import utils.files as file_utils

from utils.nms.nms_wrapper import nms

## SEMESTER PROJECT ##
from soccer.calibration.semesterproject_calibration import filter_parameters


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
        self.estimated_poses = {f: None for f in self.frame_basenames}

        # number of frames
        self.n_frames = len(self.frame_basenames)
        self.ext = self.frame_fullnames[0][-3:]

        # bbox from detectron
        self.bbox = {f: None for f in self.frame_basenames}
        # mask from detectron
        self.mask = {f: None for f in self.frame_basenames}
        # camera calibration infos
        self.calib = {f: None for f in self.frame_basenames}
        # detectron output
        self.detectron = {f: None for f in self.frame_basenames}
        # can use those infos for ball?
        self.ball = {f: None for f in self.frame_basenames}
        self.tracks = None
        self.name = None

        # Make the txt file
        txt_file = join(path_to_dataset, 'youtube.txt')
        if not exists(txt_file):
            np.savetxt(txt_file, self.frame_fullnames, fmt='%s')

        if not exists(join(path_to_dataset, 'metadata')):
            os.mkdir(join(path_to_dataset, 'metadata'))

        # save the shape of the images
        img_ = self.get_frame(0)
        self.shape = img_.shape

    # ---------------------------------------------------------------------------

    def _load_metadata(self, filename, attr):
        if exists(filename):
            with open(filename, 'rb') as f:
                setattr(self, attr, pickle.load(f))
        glog.info('{0}: {1}\tfrom {2}'.format(attr, exists(
            filename), file_utils.extract_basename(filename)[0]))

    def digest_metadata(self):

        calib_file = join(self.path_to_dataset, 'metadata', 'calib.p')
        self._load_metadata(calib_file, 'calib')

        pose_coarse_file = join(self.path_to_dataset, 'metadata', 'poses.p')
        self._load_metadata(pose_coarse_file, 'poses')

        detectron_file = join(self.path_to_dataset, 'metadata', 'detectron.p')
        self._load_metadata(detectron_file, 'detectron')

    def get_frame(self, frame_number, dtype=np.float32, sfactor=1.0, image_type='rgb'):
        return io.imread(self.frame_fullnames[frame_number], dtype=dtype, sfactor=sfactor, image_type=image_type)

    def get_frame_index(self, frame_name):
        return self.frame_basenames.index(frame_name)

    def calibrate_camera(self, vis_every=-1):
        if not exists(join(self.path_to_dataset, 'calib')):
            os.mkdir(join(self.path_to_dataset, 'calib'))

        calib_file = join(self.path_to_dataset, 'metadata', 'calib.p')
        if exists(calib_file):
            glog.info('Loading coarse detections from: {0}'.format(calib_file))
            with open(calib_file, 'rb') as f:
                self.calib = pickle.load(f)

        else:

            if not self.file_lists_match(listdir(join(self.path_to_dataset, 'calib'))):

                # The first frame is estimated by manual clicking
                manual_calib = join(self.path_to_dataset, 'calib',
                                    '{0}.npy'.format(self.frame_basenames[0]))
                if exists(manual_calib):
                    calib_npy = np.load(manual_calib).item()
                    A, R, T = calib_npy['A'], calib_npy['R'], calib_npy['T']
                else:
                    img = self.get_frame(0)
                    coarse_mask = self.get_mask_from_detectron(0)
                    A, R, T = calibration.calibrate_by_click(img, coarse_mask)

                if A is None:
                    glog.error('Manual calibration failed!')
                else:
                    np.save(join(self.path_to_dataset, 'calib', '{0}'.format(self.frame_basenames[0])),
                            {'A': A, 'R': R, 'T': T})
                    for i in tqdm(range(1, self.n_frames)):
                        # glog.info('Calibrating frame {0} ({1}/{2})'.format(self.frame_basenames[i], i, self.n_frames))
                        img = self.get_frame(i)
                        coarse_mask = self.get_mask_from_detectron(i)

                        if i % vis_every == 0:
                            vis = True
                        else:
                            vis = False
                        A, R, T, __ = calibration.calibrate_from_initialization(
                            img, coarse_mask, A, R, T, vis)

                        np.save(join(self.path_to_dataset, 'calib', '{0}'.format(self.frame_basenames[i])),
                                {'A': A, 'R': R, 'T': T})

                ## SEMESTER PROJECT ##
                # Call function that applies a filter on the estimated parameter. Possibly filter_type are 'mean' and 'median',
                # default is 'mean'. The kernel size defines the number of estimations to consider during filtering, default 5.
                filter_parameters(join(self.path_to_dataset, 'calib'), filter_type='mean', kernel_size=5)
                ## END ##

            for i, basename in enumerate(tqdm(self.frame_basenames)):
                calib_npy = np.load(join(self.path_to_dataset, 'calib',
                                         '{0}.npy'.format(basename))).item()
                A, R, T = calib_npy['A'], calib_npy['R'], calib_npy['T']
                self.calib[basename] = {'A': A, 'R': R, 'T': T}

            with open(calib_file, 'wb') as f:
                pickle.dump(self.calib, f)

    # ---------------------------------------------------------------------------
    # customized slightly from tabletop project
    # estimates the poses with openpose and saves them in class soccer.poses per frame
    def estimate_openposes(self, redo=False, openpose_dir='~/installations/openpose', pad=150):

        pose_file_coarse = join(self.path_to_dataset, 'metadata', 'poses.p')
        if exists(pose_file_coarse) and not redo:
            glog.info('Loading fine detections from: {0}'.format(pose_file_coarse))
            with open(pose_file_coarse, 'rb') as f:
                self.poses = pickle.load(f)
        else:
            h, w = self.shape[:2]
            openposebin = './build/examples/openpose/openpose.bin'
            tmp_dir = join(self.path_to_dataset, 'tmp')
            if not exists(tmp_dir):
                os.mkdir(tmp_dir)

            for i, basename in enumerate(tqdm(self.frame_basenames)):

                # Remove previous files
                previous_files = [f for f in os.listdir(tmp_dir)]
                for f in previous_files:
                    os.remove(join(tmp_dir, f))

                img = self.get_frame(i)
                bbox = self.bbox[basename]

                # save the crops in a temp file
                for j in range(bbox.shape[0]):
                    x1, y1, x2, y2 = bbox[j, 0:4]
                    x1, y1 = int(np.maximum(np.minimum(x1 - pad, w - 1), 0)), int(
                        np.maximum(np.minimum(y1 - pad, h - 1), 0))
                    x2, y2 = int(np.maximum(np.minimum(x2 + pad, w - 1), 0)), int(
                        np.maximum(np.minimum(y2 + pad, h - 1), 0))
                    crop = img[y1:y2, x1:x2, :]

                    # Save crop
                    cv2.imwrite(join(self.path_to_dataset, 'tmp',
                                     '{0}.jpg'.format(j)), crop[:, :, (2, 1, 0)] * 255)

                exit()

                cwd = os.getcwd()
                os.chdir(openpose_dir)
                # ./build/examples/openpose/openpose.bin --model_pose COCO --image_dir ~/Data/K1/images --write_json ~/Data/K1/images --write_images ~/Data/K1/results --display 0 --render_pose 0

                # openpose command
                # display & render_pose disable video output
                command = '{0} --model_pose COCO --image_dir {1} --write_json {2} --display 0 --render_pose 0'.format(
                    openposebin, tmp_dir, tmp_dir)

                os.system(command)
                os.chdir(cwd)

                poses = []
                for j in range(bbox.shape[0]):
                    x1, y1, x2, y2 = bbox[j, 0:4]
                    x1, y1 = int(np.maximum(np.minimum(x1 - pad, w - 1), 0)), int(
                        np.maximum(np.minimum(y1 - pad, h - 1), 0))

                    with open(join(join(self.path_to_dataset, 'tmp'), '{0}_keypoints.json'.format(j))) as data_file:
                        # for iii in range(2):
                        #     _ = data_file.readline()
                        data_json = json.load(data_file)

                        if len(data_json['people']) == 0:
                            continue
                        # sz = data_json['sizes']
                        n_persons = len(data_json['people'])
                        # keypoints = np.array(data_json['data']).reshape(sz)

                        for k in range(n_persons):
                            keypoints_ = np.array(
                                data_json['people'][k]['pose_keypoints_2d']).reshape((18, 3))
                            keypoints_[:, 0] += x1
                            keypoints_[:, 1] += y1
                            poses.append(keypoints_)

                self.poses[basename] = poses

            with open(pose_file_coarse, 'wb') as f:
                pickle.dump(self.poses, f)

        return 0

    def estimate_openpose_without_detectron(self, openpose_dir='/path/to/openpose'):
        openposebin = './build/examples/openpose/openpose.bin'

        # tmp directory to store the output of openpose
        tmp_dir = join(self.path_to_dataset, 'tmp')
        if not exists(tmp_dir):
            os.mkdir(tmp_dir)

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
        for i, basename in enumerate(tqdm(self.frame_basenames)):
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


    def refine_poses(self, keypoint_thresh=10, score_thresh=0.5, neck_thresh=0.59, margin=0.0):
        W, H = 103.0, 67.0

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

            if (len(keep) == 0):
                continue

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

            # Remove poses outside of field
            keep3 = []
            cam_mat = self.calib[basename]
            cam = cam_utils.Camera(
                basename, cam_mat['A'], cam_mat['R'], cam_mat['T'], self.shape[0], self.shape[1])
            for ii in range(len(poses)):
                kp3 = misc_utils.lift_keypoints_in_3d(cam, poses[ii])
                if (-W / 2. - margin) <= kp3[1, 0] <= (W / 2. + margin) and (-H / 2. - margin) <= kp3[1, 2] <= (H / 2. + margin):
                    keep3.append(ii)

            poses = [poses[ii] for ii in keep3]

            self.poses[basename] = poses

    def file_lists_match(self, list2):
        list2 = [file_utils.extract_basename(f)[0] for f in list2]
        hash_table = dict.fromkeys(list2)
        all_included = True
        for i in self.frame_basenames:
            if i not in hash_table:
                all_included = False
                break
        return all_included

    def dump_video(self, vidtype, scale=1, mot_tracks=None, one_color=True):
        if vidtype not in ['calib', 'poses', 'detections', 'tracks', 'mask']:
            raise Exception('Uknown video format')

        if vidtype == 'tracks' and mot_tracks is None:
            raise Exception('No MOT tracks provided')

        glog.info('Dumping {0} video'.format(vidtype))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4V
        out_file = join(self.path_to_dataset, '{0}.mp4'.format(vidtype))
        # 25 FPS
        out = cv2.VideoWriter(out_file, fourcc, 25.0,
                              (self.shape[1] // scale, self.shape[0] // scale))

        font = cv2.FONT_HERSHEY_SIMPLEX
        cmap = matplotlib.cm.get_cmap('hsv')
        if mot_tracks is not None:
            n_tracks = max(np.unique(mot_tracks[:, 1]))

        for i, basename in enumerate(tqdm(self.frame_basenames)):

            img = self.get_frame(i, dtype=np.uint8)

            if vidtype == 'poses':
                # Pose
                poses = self.poses[basename]
                draw_utils.draw_skeleton_on_image(img, poses, cmap, one_color=one_color)

            if vidtype == 'calib':
                # Calib
                cam = cam_utils.Camera('tmp', self.calib[basename]['A'], self.calib[basename]
                                       ['R'], self.calib[basename]['T'], self.shape[0], self.shape[1])
                canvas, mask = draw_utils.draw_field(cam)
                canvas = cv2.dilate(canvas.astype(np.uint8), np.ones(
                    (15, 15), dtype=np.uint8)).astype(float)
                img = img * (1 - canvas)[:, :, None] + np.dstack((canvas *
                                                                  255, np.zeros_like(canvas), np.zeros_like(canvas)))

            elif vidtype == 'detections':
                # Detection
                bbox = self.bbox[basename].astype(np.int32)
                if self.ball[basename] is not None:
                    ball = self.ball[basename].astype(np.int32)
                else:
                    ball = np.zeros((0, 4), dtype=np.int32)

                for j in range(bbox.shape[0]):
                    cv2.rectangle(img, (bbox[j, 0], bbox[j, 1]),
                                  (bbox[j, 2], bbox[j, 3]), (255, 0, 0), 10)
                for j in range(ball.shape[0]):
                    cv2.rectangle(img, (ball[j, 0], ball[j, 1]),
                                  (ball[j, 2], ball[j, 3]), (0, 255, 0), 10)

            elif vidtype == 'tracks':
                # Tracks
                cur_id = mot_tracks[:, 0] - 1 == i
                current_boxes = mot_tracks[cur_id, :]

                for j in range(current_boxes.shape[0]):
                    track_id, x, y, w, h = current_boxes[j, 1:6]
                    clr = cmap(track_id / float(n_tracks))
                    cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)),
                                  (clr[0] * 255, clr[1] * 255, clr[2] * 255), 10)
                    cv2.putText(img, str(int(track_id)), (int(x), int(y)),
                                font, 2, (255, 255, 255), 2, cv2.LINE_AA)

            elif vidtype == 'mask':
                # Mask
                mask = self.get_mask_from_detectron(i)*255
                img = np.dstack((mask, mask, mask))

            img = cv2.resize(img, (self.shape[1] // scale, self.shape[0] // scale))
            out.write(np.uint8(img[:, :, (2, 1, 0)]))

        # Release everything if job is finished
        out.release()
        cv2.destroyAllWindows()

    def gather_detectron(self):
        glog.info('Gathering Detectron')

        if not exists(join(self.path_to_dataset, 'detectron')):
            os.mkdir(join(self.path_to_dataset, 'detectron'))

        detectron_file = join(self.path_to_dataset, 'metadata', 'detectron.p')
        if exists(detectron_file):
            glog.info('Loading coarse detections from: {0}'.format(detectron_file))
            with open(detectron_file, 'rb') as f:
                self.detectron = pickle.load(f)

        else:

            for i, basename in enumerate(tqdm(self.frame_basenames)):
                with open(join(self.path_to_dataset, 'detectron', '{0}.yml'.format(basename)), 'rb') as stream:
                    #data = yaml.load(stream)
                    data = yaml.unsafe_load(stream)
                boxes, classes, segms = data['boxes'], data['classes'], data['segms']

                self.detectron[basename] = {'boxes': boxes,
                                            'segms': segms, 'keyps': None, 'classes': classes}

            with open(detectron_file, 'wb') as f:
                pickle.dump(self.detectron, f)

    def get_number_of_players(self):

        players_in_frame = np.zeros((self.n_frames,))
        for i, basename in enumerate(self.frame_basenames):
            players_in_frame[i] = len(self.bbox[basename])

        return players_in_frame

    def get_boxes_in_2d(self):
        boxes2d = []
        for i, basename in enumerate(self.frame_basenames):
            bbox = self.bbox[basename]
            boxes2d.append(bbox[:, :4].reshape(bbox.shape[0], 2, 2))
        return boxes2d

    def get_keypoints_in_2d(self):
        keypoints = []
        for i, basename in enumerate(self.frame_basenames):
            kp = self.poses[basename]
            keypoints.append(kp)
        return keypoints

    def get_boxes_in_3d(self):
        boxes3d = []
        for i, basename in enumerate(self.frame_basenames):
            bbox = self.bbox[basename]
            cam_mat = self.calib[basename]
            cam = cam_utils.Camera(
                basename, cam_mat['A'], cam_mat['R'], cam_mat['T'], self.shape[0], self.shape[1])
            bbox3d = misc_utils.lift_box_in_3d(cam, bbox)
            boxes3d.append(bbox3d)
        return boxes3d

    def get_mask_from_detectron(self, frame_number):
        return io.imread(join(self.path_to_dataset, 'detectron', self.frame_basenames[frame_number]+'.png'))[:, :, 0]

    def get_ball_from_detectron(self, thresh=0.0, nms_thresh=0.5):
        for i, basename in enumerate(tqdm(self.frame_basenames)):
            data = self.detectron[basename]
            boxes, segms, keyps, classes = data['boxes'], data['segms'], data['keyps'], data['classes']
            valid = (boxes[:, 4] > thresh)*([j == 33 for j in classes])
            boxes = boxes[valid, :]

            valid_nms = nms(boxes.astype(np.float32), nms_thresh)
            boxes = boxes[valid_nms, :]

            self.ball[basename] = boxes

    def get_color_from_detections(self, frame_number):
        basename = self.frame_basenames[frame_number]
        img = self.get_frame(frame_number)
        boxes = self.bbox[basename]
        n_boxes = boxes.shape[0]
        box_color = np.zeros((n_boxes, 3))
        segms = self.detectron[basename]['segms']

        for i in range(n_boxes):
            masks = mask_util.decode(segms[i])
            II, JJ = (masks > 0).nonzero()
            crop = img[II, JJ, :].reshape((-1, 3))
            box_color[i, :] = np.mean(crop, axis=0)
        return box_color

    def refine_detectron(self, basename, score_thresh=0.9, nms_thresh=0.5, min_height=0.0, min_area=200):

        data = self.detectron[basename]
        boxes, segms, keyps, classes = data['boxes'], data['segms'], data['keyps'], data['classes']

        valid = (boxes[:, 4] > score_thresh) * ([j == 1 for j in classes])
        valid = (valid == True).nonzero()[0]
        boxes = boxes[valid, :]
        segms = [segms[i] for i in valid]
        classes = [classes[i] for i in valid]

        cam_mat = self.calib[basename]
        cam = cam_utils.Camera(
            basename, cam_mat['A'], cam_mat['R'], cam_mat['T'], self.shape[0], self.shape[1])

        # indices of the boxes to keep
        keep, __ = misc_utils.putting_objects_in_perspective(cam, boxes, min_height=min_height)
        boxes = boxes[keep, :]
        segms = [segms[i] for i in keep]
        classes = [classes[i] for i in keep]

        valid_nms = nms(boxes.astype(np.float32), nms_thresh)
        boxes = boxes[valid_nms, :]
        segms = [segms[i] for i in valid_nms]
        classes = [classes[i] for i in valid_nms]

        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        valid_area = (areas > min_area).nonzero()[0]
        boxes = boxes[valid_area, :]
        segms = [segms[i] for i in valid_area]
        classes = [classes[i] for i in valid_area]

        return boxes, segms, keyps, classes

    def get_boxes_from_detectron(self, score_thresh=0.9, nms_thresh=0.5, min_height=0.0, min_area=200):

        for i, basename in enumerate(tqdm(self.frame_basenames)):

            boxes, segms, keyps, classes = self.refine_detectron(basename, score_thresh=score_thresh,
                                                                 nms_thresh=nms_thresh, min_height=min_height,
                                                                 min_area=min_area)
            self.bbox[basename] = boxes
