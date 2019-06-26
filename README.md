# Soccer Analysis

------------------------------------------------------------


## Problems

### Todo
* synchronize Part
* implement Kalman Filter


### Meeting

#### Detectron so slow:
When running inference on your own high-resolution images, Mask R-CNN may be slow simply because substantial time is spent upsampling the predicted masks to the original image resolution (this has not been optimized). You can diagnose this issue if the misc_mask time reported by tools/infer_simple.py is high (e.g., much more than 20-90ms). The solution is to first resize your images such that the short side is around 600-800px (the exact choice does not matter) and then run inference on the resized image.

#### Inputs
- frames extrahieren, verschiedene frame rates problem???

#### questions

next:
* Kalman Filter vorbereiten/implementieren?
* synchronisierung vorbereiten soweit wie möglich
* Euler

Visualisierung:
* später die 3D scene visualisieren, tipps für library?
* zum Testen des Kalman Filter erstmal mit plotly...

------------------------------------------------------------

## Notizen:
* alternative Kalman Filter: particle filtering
* SMPL realistic 3D model of human body

#### calibration parameters:

```python
cam = cam_utils.Camera(
    'tmp',
    self.calib[basename]['A'],
    self.calib[basename]['R'],
    self.calib[basename]['T'],
    self.shape[0], self.shape[1])

class Camera:
    def __init__(self, name=None, A=None, R=None, T=None, h=None, w=None):

        self.name = name
        self.A = np.eye(3, 3)
        self.A_i = np.eye(3, 3)
        self.R = np.eye(3, 3)
        self.T = np.zeros((3, 1))
```

#### Kalman-Filter

#### implementation
[pykalman - python implementation](https://pykalman.github.io/)

implement: [guide](http://www.kostasalexis.com/the-kalman-filter.html)

implement Kalman Filter (for all players -> player tracking needed?)
* problems:
  * players entering and leaving the screen
  * player in a camera but not in the other cameras how to handle
* remarks:
  * using the existing tracking system?



### Display result
Is the goal to display the 3D-keypoints on a virtual soccer field?

### evaluation/statistics
further questions:
* to evaluate the visualization necessary?
* how to track the ball?
* how to track the player's identity


------------------------------------------------------------


## Prerequisite
* [Detectron](https://github.com/facebookresearch/Detectron)
* [Openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)

#### Pythonpath remark
The local `utils` directory is used for the python3 commands so include this github directory to your python3 path. While detectron uses python2 but also got a `Detectron/detectron/utils` directory which leads to import errors when the `Detectron` directory is also added to the python3 path.

## Pipeline
1. Synchronize the videos_
2. Prepare Keypoints
  * Detectron for bounding boxes
  * Calibrate cameras
  * Estimate poses


### 1. Synchronize the videos
How?

options:
* [synchronization for multi-perspective videos in the wild](http://www.cs.cmu.edu/~poyaoh/data/ICASSP_2017.pdf)
* [Elan](https://www.mpi.nl/corpus/html/elan/ch01s02s04.html)

### 2. Prepare Keypoints
#### Detect bounding Boxes [(Detectron github)](https://github.com/facebookresearch/Detectron)
remark: adjust for multiple cameras

Using an end-to-end trained Mask R-CNN model with a ResNet-50-FPN backbone from the [model zoo](https://github.com/facebookresearch/Detectron/blob/master/MODEL_ZOO.md). Detectron should automatically download the model from the URL specified by the --wts argument. The --cfg arguments corresponds to the configuration of the baseline, all those configurations are located in the detectron project directory `configs/12_2017_baselines`.

```bash
# set path variables (adjust for own system)
DETECTRON=$HOME/installations/Detectron

# add path from folder which you want
DATA=$HOME/Data/camera0


mkdir $DATA/detectron
cd $DETECTRON
###
python2 tools/infer_subimages.py \
  --cfg configs/12_2017_baselines/e2e_mask_rcnn_R-50-FPN_2x.yaml \
  --output-dir $DATA/detectron \
  --image-ext jpg \
  --wts https://dl.fbaipublicfiles.com/detectron/35859007/12_2017_baselines/e2e_mask_rcnn_R-50-FPN_2x.yaml.01_49_07.By8nQcCH/output/train/coco_2014_train%3Acoco_2014_valminusminival/generalized_rcnn/model_final.pkl \
  $DATA/images/
```


#### Calibrate cameras
To run the calibration step, you have to manually select four pairs of correspondences. Good practice is to take the intersection of the middle line and the outside line and additionally a point on the outside line. After placed in each image four points close the window. Now choose the better estimation (Save cv - left, Save opt - right) or discard and try again.

```bash
# set path variables (adjust for own system)
PROJ=$HOME/Studium/BachelorThesis/code

# set your folder
DATA=$HOME/Data/...
cd $PROJ

# expand $DATA/ with actual path
python3 demo/calibrate_video.py \
  --path_to_data $DATA
```

#### Estimate Poses [(openpose github)](https://github.com/CMU-Perceptual-Computing-Lab/openpose)
Run the `openpose.bin` on the boxes obtained from detectron. The demo from openpose runs with the following arguments (can be changed in main.py):
```
--model_pose COCO --image_dir {1} --write_json {2} --display 0 --render_pose 0
```
* --model_pose: used model (affects number of keypoints)
* {1} and {2}: temporary directory in the `$DATA` folder.
* --display 0 --render_pose 0: disables the video output


```bash
# set path variables (adjust for own system)
OPENPOSE=$HOME/installations/openpose

python3 demo/estimate_openpose.py \
  --openpose_dir $OPENPOSE \
  --path_to_data $DATA
```

and the refine of the openposes:
```bash
cd $HOME/Studium/BachelorThesis/code
# add desired data path
python3 demo/refine_openpose.py --path_to_data ~/Data/.../

```
