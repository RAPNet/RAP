## Installation

The installation is exactly the **same** as maskrcnn-benchmark dose.  I only install it on Linux as follows. Installation on other platforms can be referred to [here](https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/INSTALL.md).

### Requirements:

- PyTorch==1.0.0.dev20190328
- torchvision==0.2.2
- cocoapi
- yacs
- matplotlib
- GCC >= 4.9
- OpenCV==3.4.3


### Step-by-step installation

```bash
# first, make sure that your conda is setup properly with the right environment
# for that, check that `which conda`, `which pip` and `which python` points to the
# right path. From a clean conda env, this is what you need to do

conda create --name RAP
conda activate RAP

# this installs the right pip and dependencies for the fresh python
conda install ipython

# maskrcnn_benchmark and coco api dependencies
pip install ninja yacs cython matplotlib tqdm opencv-python

# follow PyTorch installation in https://pytorch.org/get-started/locally/
# we give the instructions for CUDA 9.0
conda install -c pytorch pytorch-nightly torchvision cudatoolkit=9.0

export INSTALL_DIR=$PWD

# install pycocotools
cd $INSTALL_DIR
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install

# install apex
cd $INSTALL_DIR
git clone https://github.com/NVIDIA/apex.git
cd apex
python setup.py install --cuda_ext --cpp_ext

# install PyTorch Panoptic Segmentation
cd $INSTALL_DIR
git clone https://github.com/CVPR2020-RAP/RAP.git
cd RAP

# the following will install the lib with
# symbolic links, so that you can modify
# the files if you want and won't need to
# re-build it
python setup.py build develop

unset INSTALL_DIR
```
