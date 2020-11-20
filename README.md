**NVIDIA AI Technology Centre (NVAITC) Toolkit**
===============

test branch

## Introduction

- The NVAITC toolkit is a Python codebase that showcases the interoperability of CUDA-X AI software stack in multi-GPU environments. It collects code snippets from other NVIDIA repos.

- The goal of this project is to provide researchers a reference framework to build new projects on.

- The codebase might be subject to further changes as new versions of used libraries become available or new functionalities requested. It requests the availability of ImageNet to demonstrate how to train a network (ResNet[18/50/101]) against a well known dataset.

## License

The NVAITC Toolkit is released under the [MIT License](LICENSE).


## Installation

Please find installation instructions in [INSTALL.md](INSTALL.md). You may follow the instructions in [DATASET.md](DATASET.md) to prepare the datasets.


# Getting Started

## Train ResNet model from scratch using DALI

```
python run_training_dali.py [-h] [--log-dir LOG_DIR] [--epochs EPOCHS]
                            [--base-lr BASE_LR] [--momentum MOMENTUM]
                            [--lr LR]
                            [--batches-per-allreduce BATCHES_PER_ALLREDUCE]
                            [--use-adasum] [-b N] [-j N]
                            [--print-freq PRINT_FREQ] [--sync_bn]
                            [--weight-decay W] [--warmup-epochs WARMUP_EPOCHS]
                            [--start-epoch N] [--resume PATH] [--amp]
                            [--opt-level OPT_LEVEL]
                            [--keep-batchnorm-fp32 KEEP_BATCHNORM_FP32]
                            [--loss-scale LOSS_SCALE]
                            [--channels-last CHANNELS_LAST] [-ar ARCH]
                            [--deterministic] [--fp16-allreduce] [--dali_cpu]
                            [--prof PROF]
                            DATA PATH
```


## Train ResNet model from scratch using Torchvision

```
python run_training_torchvision.py [-h] [--log-dir LOG_DIR] [--epochs EPOCHS]
                                   [--base-lr BASE_LR] [--momentum MOMENTUM]
                                   [--lr LR]
                                   [--batches-per-allreduce BATCHES_PER_ALLREDUCE]
                                   [--use-adasum] [-b N] [-j N]
                                   [--print-freq PRINT_FREQ] [--sync_bn]
                                   [--weight-decay W]
                                   [--warmup-epochs WARMUP_EPOCHS]
                                   [--start-epoch N] [--resume PATH] [--amp]
                                   [--opt-level OPT_LEVEL]
                                   [--keep-batchnorm-fp32 KEEP_BATCHNORM_FP32]
                                   [--loss-scale LOSS_SCALE]
                                   [--channels-last CHANNELS_LAST] [-ar ARCH]
                                   [--deterministic] [--fp16-allreduce]
                                   [--prof PROF]
                                   DATA PATH

```

## Resume from an Existing Checkpoint

You can resume your training from an existing checkpoint using the following option in the command line:

```
--resume <path-to-checkpoint>
```

## Perform Inference

```
python run_inference.py [-h] [--tensorrt] [--half] [--log-dir LOG_DIR] [-b N]
                        [-j N] [--print-freq PRINT_FREQ] [--data-dir DATA_DIR]
                        [-ar ARCH] [--deterministic] [--dali_cpu]
                        PATH

```
