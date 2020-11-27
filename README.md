**NVIDIA AI Technology Centre (NVAITC)**
===============

# Toolkit 

## Introduction

- The NVAITC toolkit is a Python codebase that showcases the interoperability of CUDA-X AI software stack in multi-GPU environments. It collects code snippets from other NVIDIA repos.

- The goal of this project is to provide researchers a reference framework to build new projects on.

- The codebase might be subject to further changes as new versions of used libraries become available or new functionalities requested. It requests the availability of ImageNet to demonstrate how to train a network (ResNet[18/50/101]) against a well known dataset.

## Clone repo

```
git clone -b toolkit --single-branch https://github.com/nvidia/nvaitc-toolkit.git
```


## Getting Started

Please find details and installation instructions in [README.md](https://github.com/NVIDIA/nvaitc-toolkit/blob/toolkit/README.md).


# cuAugment

## Introduction

cuAugment is a CUDA-accelerated 1D/2D/3D/4D augmenter library that utilizes a just-in-time compiler to transform a cascade of coordinate transformation into a single monolithic kernel to avoid unnecessary accesses to global memory.

## Clone repo

```
git clone -b cuaugment --single-branch https://github.com/nvidia/nvaitc-toolkit.git
```

## Getting Started

Please find details and installation instructions in [README.md](https://github.com/NVIDIA/nvaitc-toolkit/blob/cuaugment/README.md). 



