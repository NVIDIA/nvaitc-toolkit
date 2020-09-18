# Introduction

- The NVAITC (NVIDIA AI Technology Centre) toolkit is a Python codebase that showcases the interoperability of CUDA-X AI software stack in multi-GPU environments.

- The goal of this project is to provide researchers a reference framework to build new projects on.

- The codebase might be subject to further changes as new versions of used libraries become available or new functionalities requested. It requests the availability of ImageNet to demonstrate how to train a network (ResNet) against a well known dataset.



## Environment setup

This repository contains Dockerfile which extends the PyTorch NGC container and encapsulates some dependencies. Aside from these dependencies, ensure you have the following components:

* [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker)
* [PyTorch 20.08-py3 NGC container](https://ngc.nvidia.com/registry/nvidia-pytorch) or newer
* Supported GPUs:
    * [NVIDIA Volta architecture](https://www.nvidia.com/en-us/data-center/volta-gpu-architecture/)
    * [NVIDIA Turing architecture](https://www.nvidia.com/en-us/geforce/turing/)
    * [NVIDIA Ampere architecture](https://www.nvidia.com/en-us/data-center/nvidia-ampere-gpu-architecture/)

## Data Preparation 

This codes operates on ImageNet 1k, a widely popular image classification dataset from the ILSVRC challenge.

To train your model using mixed or TF32 precision with Tensor Cores or using FP32, perform the following steps using the default parameters of the resnet50 model on the ImageNet dataset.

1. [Download the images](http://image-net.org/download-images).

2. Extract the training data:
  ```bash
  mkdir train && mv ILSVRC2012_img_train.tar train/ && cd train
  tar -xvf ILSVRC2012_img_train.tar && rm -f ILSVRC2012_img_train.tar
  find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
  cd ..
  ```

3. Extract the validation data and move the images to subfolders:
  ```bash
  mkdir val && mv ILSVRC2012_img_val.tar val/ && cd val && tar -xvf ILSVRC2012_img_val.tar
  wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
  ```

The directory in which the `train/` and `val/` directories are placed, is referred to as `<path to imagenet>` in this document.

## Training Procedure

Run `python run_training_dali.py <ops1> <ops2> ` using the following arguments:
Run `python run_training_torchvision.py <ops1> <ops2> `
Run `python run_inference.py <ops1> <ops2> `


| Argument | Possible values |
|----------|----------|
| `--log-dir` | tensorboard log directory |
| `--epochs` | Number of epochs (default: 90) |
| `--base-lr` | Learning rate for a single GPU (default: 0.0125) |
