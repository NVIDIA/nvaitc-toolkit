# The MIT License (MIT)

# Copyright (c) 2020 NVIDIA CORPORATION.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import os
import random
import shutil
import time
import warnings
import sys

from argparse import ArgumentParser
from torchvision import datasets, transforms, models


import re
import torch
import torch.optim as optim
import trtorch

from launchers.dali import DALITrainer
from launchers.torchvision_ddp import TVTrainer, AverageMeter
from loaders.pipe import ImageNetTrainPipe, ImageNetValPipe
from util import timeme

from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss


import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import torch.backends.cudnn as cudnn

import torch.multiprocessing as mp

import numpy as np

from torch.utils.tensorboard import SummaryWriter

import pprint
import torch.distributed as dist

try:
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator
    from nvidia.dali.pipeline import Pipeline
    import nvidia.dali.ops as ops
    import nvidia.dali.types as types
except ImportError:
    raise ImportError("Please install DALI from https://www.github.com/NVIDIA/DALI to run this example.")




ap = ArgumentParser(description='New Image Classifier')

ap.add_argument('checkpoint', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
ap.add_argument('--tensorrt', default=False, action='store_true',
                help='Runs using tensorrt')
ap.add_argument('--half', default=False, action='store_true',
                help='Runs using half precision')
ap.add_argument('--log-dir', default='./logs', 
        help='tensorboard log directory')
ap.add_argument('-b', '--batch-size', default=256, type=int,
                metavar='N', help='mini-batch size per process (default: 256)')
ap.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                help='number of data loading workers (default: 4)')                    
ap.add_argument('--print-freq', type=int, default=10,
                help='output frequency')
ap.add_argument('--data-dir', default='/workspace/imagenet', 
                help='data loading path')
ap.add_argument('-ar', '--arch', type=str, default="resnet50")
ap.add_argument('--deterministic', action='store_true')
ap.add_argument('--dali_cpu', action='store_true', 
                help='Runs CPU based version of DALI pipeline.')
#Pytorch Distributed
parser.add_argument('--num_nodes', type=int, default=1,
                help='Number of available nodes/hosts')
parser.add_argument('--node_id', type=int, default=0,
                help='Unique ID to identify the current node/host')
parser.add_argument('--num_gpus', type=int, default=1,
                help='Number of GPUs in each node')

args = parser.parse_args()

WORLD_SIZE = args.num_gpus * args.num_nodes
os.environ['MASTER_ADDR'] = 'localhost' 
os.environ['MASTER_PORT'] = '9957' 

args = ap.parse_args()



def to_python_float(t):
    if hasattr(t, 'item'):
        return t.item()
    else:
        return t[0]

def correct(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return correct.sum()

def validate(args, loader, model):
    batch_time = AverageMeter()
    cor_count = 0
    tot_count = 0

    print("Warm up ...")
    for i, data in enumerate(loader):
        input = data[0]["data"]
        if args.half:
            input = input.half()

        features = model(input)
        if i == 5: break
    torch.cuda.synchronize()

    for i, data in enumerate(loader):
        input = data[0]["data"]
        if args.half:
            input = input.half()

        target = data[0]["label"].squeeze().cuda().long()
        loader_len = int(loader._size / args.batch_size)

        # compute output
        with torch.no_grad():
            start = time.time()
            output = model(input)
            torch.cuda.synchronize()
            end = time.time()
            batch_time.update(end - start)

        # measure accuracy and record loss
        n = correct(output.data, target)

        if args.distributed:
            n = dist.all_reduce(n, op=dist.Sum)
        cor_count += n.cpu().numpy()
        tot_count += args.batch_size * args.world_size
        
        if args.local_rank == 0 and i % args.print_freq == 0:
            print(('Test: [{0}/{1}]\t'
                + 'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                + 'Speed {2:.3f} ({3:.3f})\t'
                + 'Accuracy {acc:.3f}\t').format(
                i, loader_len,
                args.world_size * args.batch_size / batch_time.val,
                args.world_size * args.batch_size / batch_time.avg,
                batch_time=batch_time,
                acc=cor_count/tot_count))

    print('Accuracy {acc:.3f}'.format(acc=cor_count/tot_count)) 


def run(args):
    crop_size = 224
    val_size = 256

    valdir = os.path.join(args.data_dir, 'val')

    # Loader
    pipe = ImageNetValPipe(batch_size=args.batch_size,
                        num_threads=args.workers,
                        device_id=args.local_rank,
                        data_dir=valdir,
                        crop=crop_size,
                        size=val_size,
                        shard_id=args.local_rank,
                        num_shards=args.world_size)
    pipe.build()
    loader = DALIClassificationIterator(pipe, reader_name="Reader", fill_last_batch=True)        

    # Network
    if args.arch == 'resnet50':
        network = models.resnet50(pretrained=False)
    elif args.arch == 'resnet101':
        network = models.resnet101(pretrained=False)
    elif args.arch == 'resnet18':
        network = models.resnet18(pretrained=False)
    else:
        if args.local_rank == 0:
            print('No network specified')
        sys.exit()

    network = network.cuda().eval()

    checkpoint = torch.load(args.checkpoint, map_location = lambda storage, loc: storage.cuda(args.gpu))
    network.load_state_dict(checkpoint['state_dict'])

    if args.tensorrt:
        traced_network = torch.jit.trace(network, [torch.randn((args.batch_size, 3, 224, 224)).to("cuda")])
        traced_network = traced_network.cuda()
        compiled_network = trtorch.compile(traced_network, {
            "input_shapes": [(args.batch_size, 3, 224, 224)],
            "op_precision": torch.half if args.half else torch.float32,
            "max_batch_size": args.batch_size
        })
        network = compiled_network

    validate(args, loader, network)


def worker(local_rank,args):
    args.no_cuda = False
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.local_rank = local_rank
    args.world_size = WORLD_SIZE
    args.global_rank = args.node_id * args.num_gpus + local_rank
    if args.cuda:
        torch.cuda.set_device(args.local_rank)
    args.local_rank = local_rank
    args.gpu = args.local_rank
    args.world_size = WORLD_SIZE

    args.distributed = args.world_size > 1

    if args.distributed:
        torch.cuda.set_device(args.gpu)

    # Enable cudnn rk
    cudnn.benchmark = True

    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.manual_seed(args.local_rank)
        torch.set_printoptions(precision=10)

    args.total_batch_size = args.world_size * args.batch_size

    run(args)

if __name__ == '__main__': 
    torch.multiprocessing.spawn(worker, nprocs=args.num_gpus, args=(args,))