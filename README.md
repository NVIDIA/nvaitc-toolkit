# Introduction

- The NVAITC (NVIDIA AI Technology Centre) toolkit is a Python codebase that showcases the use of most relevant NVIDIA deep learning libraries (DALI, AMP, TensorRT) and Horovod, a framework for distributed training. 

- The purpose of this project, which includes code snippets taken from other NVIDIA Deep Learning examples, i.e. https://github.com/NVIDIA/apex/blob/master/examples/imagenet/main_amp.py, https://github.com/NVIDIA/DALI/blob/master/docs/examples/use_cases/pytorch/resnet50/main.py, is to illustrate how to combine these libraries together for achieving better training and inference performance. 

- The codebase might be subject to further changes as new versions of used libraries become available or new functionalities requested. It requests the availability of ImageNet to demonstrate how to train a network (ResNet50) against a well known dataset.

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

Run `python model <ops1> train <ops2> ` using the following arguments:

| Argument | Possible values |
|----------|----------|
| `--log-dir` | tensorboard log directory |
| `--epochs` | Number of epochs (default: 90) |
| `--base-lr` | Learning rate for a single GPU (default: 0.0125) |
