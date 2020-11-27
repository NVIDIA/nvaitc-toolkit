**NVIDIA AI Technology Centre (NVAITC)**
===============

# Toolkit 

### Introduction

Python codebase to showcase the interoperability of CUDA-X AI software stack in multi-GPU environments. The goal of this project is to provide researchers a reference framework to build new projects on. It requests the availability of ImageNet to demonstrate how to train a network (ResNet[18/50/101]) against a well known dataset.

### Clone repo

```
git clone -b toolkit --single-branch https://github.com/nvidia/nvaitc-toolkit.git toolkit
```

### Getting Started

Please find details and installation instructions in [README.md](https://github.com/NVIDIA/nvaitc-toolkit/blob/toolkit/README.md).


# cuAugment

### Introduction

cuAugment is a CUDA-accelerated 1D/2D/3D/4D augmenter library that utilizes a just-in-time compiler to transform a cascade of coordinate transformation into a single monolithic kernel to avoid unnecessary accesses to global memory.

### Clone repo

```
git clone -b cuaugment --single-branch https://github.com/nvidia/nvaitc-toolkit.git cuaugment
```

### Getting Started

Please find details and installation instructions in [README.md](https://github.com/NVIDIA/nvaitc-toolkit/blob/cuaugment/README.md).
