# Installation

Two options:
 1. use the NVIDIA GPU Cloud pre-created container: `docker run --runtime=nvidia -it -d -v $PWD:/workspace nvcr.io/ea-nvaitc/toolkit:20.11-py3`
 1. create your own environment using the following requirements list
 
## Requirements
- Python >= 3.6
- PyTorch >= 1.3
- [dali](https://github.com/NVIDIA/DALI#installing-or-building-dali) that matches the PyTorch installation
- [apex](https://github.com/NVIDIA/apex) that matches the PyTorch installation
- [trtorch](https://github.com/NVIDIA/TRTorch) >= 0.1.0
- GCC >= 4.9
- OpenCV: `pip install opencv-python`
- torchvision: `pip install torchvision` or `conda install torchvision -c pytorch`
- tensorboard: `pip install tensorboard`
- tensorboardX: `pip install tensorboardX`
- horovod: `pip install horovod`


You can find most of the software in a preinstalled [NVIDIA GPU Cloud](ngc.nvidia.com) PyTorch container in version 19.10 and later or use the `requirements.txt`.

