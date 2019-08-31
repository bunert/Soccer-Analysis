# Soccer Analysis

------------------------------------------------------------
The application of Computer Vision in the field of sports is increasing in numerous areas. The Swiss national soccer team approached the Computer Vision and Geometry Lab at the ETH to explore the possibilities of technologies to analyze soccer games using the existing TV cameras. In the following paper, we present a system that estimates three-dimensional human poses based on TV camera recordings and two-dimensional tracking data of the soccer game. The work is based on existing state-of-the-art algorithms for object detection and human 2D pose estimations. We present an Extended Kalman Filter to fuse the poses from different camera perspectives into world coordinates. The results confirm the feasibility of setting up such a system without large investments compared to other systems which are installed explicitly for this purpose. To achieve the desired accuracy, many points would have to be improved upon.

![Overview](https://github.com/bunert/BachelorThesis/blob/master/overview.jpg)

------------------------------------------------------------


## Prerequisite
There are several prerequisites and dependencies which are required, for more details check the following three pages and install them.
* [Detectron](https://github.com/facebookresearch/Detectron)
* [Openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)
* [Soccer on your Tabletop](https://github.com/krematas/soccerontable)

#### Pythonpath remark
The local `utils` directory is used for the python3 commands so include this github directory to your python3 path. While detectron uses python2 but also got a `Detectron/detectron/utils` directory which leads to import errors when the `Detectron` directory is also added to the python3 path.


## Directory Structure
The directory structure should look like this before starting the pipeline. Where the data directory is located does not matter, but the structure afterwards have to match to this exmple. During the workflow the subroutines will create more subdirectories for each camera with the intermediate results.
```bash
DATA
  ├── camera1
    ├── images
      ├── 00000.jpg
      ├── 00001.jpg
      ├── ...
  ├── camera2
    ├── images
      ├── 00000.jpg
      ├── 00001.jpg
      ├── ...

```


## Pipeline
As you can see in the Overview picture above, there are several steps which are performed per camera, only the last step for the EKF is performed once for all cameras together. So repeat the pipeline except the last step (fusion) for each camera and then start the final step. Most steps are adapted from the project [Soccer on your Tabletop](https://github.com/krematas/soccerontable), so for more information visit their GitHub page. 
```bash
git clone git@github.com:bunert/BachelorThesis.git $PROJ
```


### Detect bounding Boxes [(Detectron github)](https://github.com/facebookresearch/Detectron)
To obtain the desired bounding boxes we run Detectron on the input images.
```bash
DATADIR=/path/to/camera1
mkdir $DATADIR/detectron
DETECTRON=/path/to/clone/detectron
cp utils/thirdpartyscripts/infer_subimages.py ./$DETECTRON/tools/
cd $DETECTRON

python2 tools/infer_subimages.py \
  --cfg configs/12_2017_baselines/e2e_mask_rcnn_R-50-FPN_2x.yaml \
  --output-dir $DATADIR/detectron \
  --image-ext jpg \
  --wts https://dl.fbaipublicfiles.com/detectron/35859007/12_2017_baselines/e2e_mask_rcnn_R-50-FPN_2x.yaml.01_49_07.By8nQcCH/output/train/coco_2014_train%3Acoco_2014_valminusminival/generalized_rcnn/model_final.pkl \
  $DATADIR/images/

```


### Calibrate Cameras
To run the calibration step, you have to manually select four pairs of correspondences. Good practice is to take the intersection of the middle line and the outside line and additionally a point on the outside line. After placing in each image four points close the window. Now choose the better estimation (Save cv - left, Save opt - right) or discard and try again.

```bash
cd $PROJ

# expand $DATA/ with actual path
python3 demo/calibrate_video.py \
  --path_to_data $DATADIR
```

### Estimate Poses [(openpose github)](https://github.com/CMU-Perceptual-Computing-Lab/openpose)
Run the `openpose.bin` on the boxes obtained from detectron. The demo from openpose runs with the following arguments (can be changed in main.py):
```
--model_pose COCO --image_dir {1} --write_json {2} --display 0 --render_pose 0
```
* --model_pose: used model (affects number of keypoints)
* {1} and {2}: temporary directory in the `$DATADIR` folder.
* --display 0 --render_pose 0: disables the video output


```bash
# set path variables (adjust for own system)
OPENPOSE=$HOME/installations/openpose

python3 demo/estimate_openpose.py \
  --openpose_dir $OPENPOSE \
  --path_to_data $DATADIR
```

### Run the Extended Kalman Filter
Information to adjust for your purposes:

`init_soccerdata`: loads the corresponding data for the cameras which should be used for the filter. So adjust the directory names for your cameras.

`init_csv`: loads the corresponding tracking data, individual for each game.

```bash
# set path variable DATA to the path where all cameras are placed in
DATA=/path/to/data/
python3 demo/fusion.py --path_to_data $DATA
```
