################################################
# Cluster setup:
################################################

### Anaconda
https://docs.anaconda.com/anaconda/install/linux/

### Caffe2
```bash
module load  cuda/9.0.176 cudnn/7.0 opencv/3.4.3 python_gpu/2.7.14 nccl/2.3.7-1 libpng/1.6.27 openblas/0.2.19 jpeg/9b

conda create --name detectron2 python=2.7
conda activate detectron2
conda install pytorch-nightly -c pytorch
conda install protobuf
pip install opencv-python --user

# add to .bashrc
export PATH=/cluster/scratch/bunert/anaconda2/envs/detectron2/lib/python2.7:$PATH
export PATH=/cluster/home/bunert/.local/lib/python2.7/site-packages:$PATH
export PYTHONPATH=/cluster/home/bunert/python/lib64/python2.7/site-packages:$PYTHONPATH
export PATH=/cluster/scratch/bunert/anaconda2/envs/detectron2/bin:$PATH


# test works
bsub -R "rusage[ngpus_excl_p=1]" python -c 'from caffe2.python import workspace; print(workspace.NumCudaDevices())'
```


### COCOAPI
```bash
git clone https://github.com/cocodataset/cocoapi.git
cd PythonAPI/
make install
python setup.py install --user
```

### Detectron
```bash
git clone https://github.com/facebookresearch/detectron
cd detectron
conda install pyyaml matplotlib cython mock scipy opencv

make

# TEST:
bsub -R "rusage[ngpus_excl_p=1,mem=2048]" python $SCRATCH/installations/detectron/detectron/tests/test_spatial_narrow_as_op.py
```

### soccerontable
```bash
git clone https://github.com/krematas/soccerontable
cd soccerontable
cp utils/thirdpartyscripts/infer_subimages.py ../detectron/tools/
```

### openpose
```bash
module remove python_gpu/2.7.14
module load python_gpu/3.7.1 cuda/10.0.130 boost/1.69.0 openblas hdf5/1.10.1 leveldb/1.20 protobuf glog snappy gflags lmdb

conda create -n openpose python=3.7.1
conda activate openpose
cd $SCRATCH/installations/soccerontable
while read requirement; do conda install --yes $requirement; done < requirements.txt
pip install glumpy glog visdom --user

git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose
cd openpose
mkdir build && cd build && cmake .. && make -j`nproc`

```
openpose install:
https://github.com/CMU-Perceptual-Computing-Lab/openpose/issues/949

### caffe
```bash
git clone https://github.com/BVLC/caffe
for req in $(cat caffe/python/requirements.txt); do conda install $req; done
cd caffe
cp Makefile.config.example Makefile.config

vim Makefile.config
# edit:
BLAS := open
BLAS_INCLUDE := /cluster/apps/gcc-4.8.5/openblas-0.2.19-w25ydbabttcfa6g76gejjkthcm3xcuv3/

```

where:
calibrate K1 done


################################################
# Cluster workflow:
################################################


#### Detectron
``` bash
module load  cuda/9.0.176 cudnn/7.0 opencv/3.4.3 python_gpu/2.7.14 nccl/2.3.7-1 libpng/1.6.27 openblas/0.2.19 jpeg/9b

conda activate detectron2
source .bashrc
conda activate detectron2
# set path variables (adjust for own system)
DETECTRON=$SCRATCH/installations/detectron
# desired camera folger:
CAM=$SCRATCH/Data/camera0


cd $CAM
mkdir detectron
cd $DETECTRON
bsub -n 8 -W 8:00 -R "rusage[ngpus_excl_p=8,mem=4096,scratch=4096]" python tools/infer_subimages.py --cfg configs/12_2017_baselines/e2e_mask_rcnn_R-50-FPN_2x.yaml --output-dir $CAM/detectron --image-ext jpg --wts models/model_final.pkl $CAM/images/

```

##### normal
```bash
bsub -n 8 -W 2:00 -o $SCRATCH/outputs/parallel_n8_g8 -R "rusage[ngpus_excl_p=8,mem=4096]" python tools/infer_subimages.py --cfg configs/12_2017_baselines/e2e_mask_rcnn_R-50-FPN_2x.yaml --output-dir $CAM/detectron --image-ext jpg --wts models/model_final.pkl $CAM/images/
```
##### Parallel - not working
```bash
bsub -n 1 -W 2:00 -o $SCRATCH/outputs/parallel_n1_g1 -R "rusage[ngpus_excl_p=1,mem=4096]" python tools/infer_parallel.py --cfg configs/12_2017_baselines/e2e_mask_rcnn_R-50-FPN_2x.yaml --output-dir $CAM/detectron --image-ext jpg --wts models/model_final.pkl $CAM/images/
```

#### Calibrate camera
Do local at home tower (need interaction):
``` bash
# set path variables (adjust for own system)
PROJ=$HOME/Studium/BachelorThesis/code
DATA=$HOME/Data
cd $PROJ

# set --cameras to the number of different cameras
python3 demo/calibrate_video.py \
  --path_to_data $DATA \
  --cameras 1
```


################################################
# linux commands:
################################################

# crop a video (ss=start [s], -t= duration [s])
ffmpeg -ss 420 -i Right-2019-03-26-20-40-02.mp4 -t 30 -acodec copy test.mp4

# extract frames
ffmpeg -i test.mp4 test%08d.jpg -hide_banner

# VNC remote access port 200 or 5900
run on tower:
/usr/lib/vino/vino-server

# ssh:
ssh bunert@178.195.249.45 -p 200

# remmina:
test2

# scp - tower to notebook
scp -P 200 bunert@178.195.249.45:/home/bunert/Downloads/videos/* /home/bunert/Data

# scp - cluster to tower
scp -P 200 -r K9/ bunert@178.195.249.45:/home/bunert/Data

# scp - tower to cluster
scp -P 200 bunert@178.195.249.45:/home/bunert/Data $SCRATCH/Data/

# scp - notebook to leonhard cluster
scp Right-2019-03-26-20-40-02.mp4 bunert@login.leonhard.ethz.ch:/cluster/scratch/bunert/Data/camera0/images/

# sshfs - mount remote directory
sshfs bunert@178.195.249.45:/home/bunert /home/bunert/remote/ -C -p 200
fusermount -u /home/bunert/remote
