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

from launchers.dali import DALITrainer
from loaders.pipe import image_net_train_pipe, image_net_val_pipe
from util import timeme

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import torch.backends.cudnn as cudnn

import torch.multiprocessing as mp

import numpy as np

from torch.utils.tensorboard import SummaryWriter

import pprint

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

try:
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy
    import nvidia.dali.types as types
except ImportError:
    raise ImportError("Please install DALI from https://www.github.com/NVIDIA/DALI to run this example.")

parser = ArgumentParser(description='NVAITC Toolkit Classification using DALI')

parser.add_argument('datadir', default='', help='data loading path', metavar='DATA PATH')
parser.add_argument('--log-dir', default='./logs', 
        help='tensorboard log directory')
parser.add_argument('--epochs', type=int, default=90,
                help='number of epochs to train')
parser.add_argument('--base-lr', type=float, default=0.0125,
                help='learning rate for a single GPU')     
parser.add_argument('--momentum', type=float, default=0.9,
                help='SGD momentum')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='Initial learning rate.  Will be scaled by <global batch size>/256: args.lr = args.lr*float(args.batch_size*args.world_size)/256.  A warmup schedule will also be applied over the first 5 epochs.')
parser.add_argument('--batches-per-allreduce', type=int, default=1,
                help=('number of batches processed locally before '
                        + 'executing allreduce across workers; it multiplies '
                        + 'total batch size.'))       
parser.add_argument('--use-adasum', action='store_true', default=False,
                help='use adasum algorithm to do reduction')       
parser.add_argument('-b', '--batch-size', default=256, type=int,
                metavar='N', help='mini-batch size per process (default: 256)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                help='number of data loading workers (default: 4)')                    
parser.add_argument('--print-freq', type=int, default=10,
                help='output frequency')
parser.add_argument('--sync_bn', action='store_true',
                help='enabling apex sync BN.')        
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                metavar='W', help='weight decay (default: 1e-4)')           

parser.add_argument('--warmup-epochs', type=float, default=5,
                    help='number of warmup epochs')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')

parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')


# Mixed precision
parser.add_argument('--amp', '-a', action='store_true')
parser.add_argument('--opt-level', type=str, default="O1")
parser.add_argument('--keep-batchnorm-fp32', type=str, default=None)
parser.add_argument('--loss-scale', type=str, default=None)
parser.add_argument('--channels-last', type=bool, default=False)

# Arch
parser.add_argument('-ar', '--arch', type=str, default="resnet50")

# Deterministic runtime
parser.add_argument('--deterministic', action='store_true')

parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                help='use fp16 compression during allreduce')

# CPU Based DALI pipeline
parser.add_argument('--dali_cpu', action='store_true', help='Runs CPU based version of DALI pipeline.')
# DALI auto_augmentation policy
parser.add_argument('--dali_auto_augment', action='store_true', default=False, help='Runs auto augmentation policy for ImageNet')

# Profiling NVTX
parser.add_argument('--prof', default=-1, type=int,  help='Only run 10 iterations for profiling.')

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

print('The code will most likely crash if you don"t have DALI >= 1.26')
print('You can substitute line 36 in loaders/pipe.py with "@pipeline_def()" to still use the code without updating DALI & without auto augment')
if args.dali_auto_augment:
    print('Make sure you have at least DALI 1.26 to use auto augmentations')
    
def worker(local_rank,args):
    args.no_cuda = False
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.local_rank = local_rank
    args.world_size = WORLD_SIZE
    args.global_rank = args.node_id * args.num_gpus + local_rank
    if args.cuda:
        torch.cuda.set_device(args.local_rank)
    
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "0"
    
    args.distributed = args.world_size > 1
    args.gpu = args.local_rank
    torch.cuda.set_device(args.gpu)

    # Enable cudnn rk
    cudnn.benchmark = True

    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.manual_seed(args.local_rank)
        torch.set_printoptions(precision=10)

    assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."

    args.total_batch_size = args.world_size * args.batch_size
    args.allreduce_batch_size = args.batch_size * args.batches_per_allreduce    

    crop_size = 224
    val_size = 256
    
    # Scale learning rate based on global batch size
    args.lr = args.lr * float(args.batch_size*args.world_size)/256.
    torch.distributed.init_process_group(backend="nccl", world_size=args.world_size, 
            rank=args.global_rank)

    traindir = os.path.join(args.datadir, 'train')
    valdir = os.path.join(args.datadir, 'val')

    if args.arch == 'resnet50':
        network = models.resnet50()
    elif args.arch == 'resnet101':
        network = models.resnet101()
    elif args.arch == 'resnet18':
        network = models.resnet18()
    else:
        if args.local_rank == 0:
            print('No network specified')
        sys.exit()

    if args.local_rank == 0:
        print("= Start training =")
        print("=> Arch '{}'".format(args.arch))

    if args.sync_bn:
        print("using apex synced BN")
        model = parallel.convert_syncbn_model(model)

    if hasattr(torch, 'channels_last') and  hasattr(torch, 'contiguous_format'):
        if args.channels_last:
            memory_format = torch.channels_last
        else:
            memory_format = torch.contiguous_format
        network = network.cuda().to(memory_format=memory_format)
    else:
        network = network.cuda()


    if args.local_rank == 0:
        print("=> world size '{}'".format(args.world_size))

    optimizer = optim.SGD(network.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    args.device = torch.device("cuda:{}".format(args.local_rank))
    network = network.to(args.device)
    network = DDP(network, device_ids=[args.local_rank], output_device=args.local_rank)
    optimizer = optim.SGD(network.parameters(), args.lr,
                            momentum=args.momentum,
                            weight_decay=args.weight_decay)
    resume_from_epoch = args.start_epoch

    args.best_prec1 = 0
    
    if args.resume:
    # Use a local scope to avoid dangling references
        def resume():
            if os.path.isfile(args.resume):
                print("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume, map_location = lambda storage, loc: storage.cuda(args.gpu))
                args.start_epoch = checkpoint['epoch']
                args.best_prec1 = checkpoint['best_prec1']
                print("=> best precision '{}'".format(args.best_prec1))
                network.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                print("=> loaded checkpoint '{}' (epoch {})"
                        .format(args.resume, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))
        resume()
    
    if args.global_rank == 0:
        print('Using DALI as data loader')

    train_pipe = image_net_train_pipe(batch_size=args.batch_size, 
        num_threads=args.workers,
        device_id=args.local_rank,
        seed=12 + args.local_rank,
        data_dir=traindir,
        crop=crop_size,
        dali_cpu=args.dali_cpu,
        shard_id=args.local_rank,
        num_shards=args.world_size,
        dali_auto_augment=args.dali_auto_augment)
    train_pipe.build()

    train_loader = DALIClassificationIterator(train_pipe, 
                    reader_name="Reader", last_batch_policy=LastBatchPolicy.DROP)

    val_pipe = image_net_val_pipe(batch_size=args.batch_size,
                        num_threads=args.workers,
                        device_id=args.global_rank,
                        seed=12 + args.local_rank,
                        data_dir=valdir,
                        crop=crop_size,
                        size=val_size,
                        dali_cpu=args.dali_cpu,
                        shard_id=args.global_rank,
                        num_shards=args.world_size)
    val_pipe.build()
    
    val_loader = DALIClassificationIterator(val_pipe, reader_name="Reader", last_batch_policy=LastBatchPolicy.DROP)        

    launcher = DALITrainer(args, train_loader, val_loader, network, optimizer)
    launcher.run()


if __name__ == '__main__': 
    torch.multiprocessing.spawn(worker, nprocs=args.num_gpus, args=(args,))


