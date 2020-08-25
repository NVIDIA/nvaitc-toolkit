import os
import random
import shutil
import time
import warnings

from apex import amp
from argparse import ArgumentParser
from torch2trt import torch2trt, TRTModule
from torchvision import datasets, transforms, models

import horovod.torch as hvd
import re
import torch
import torch.optim as optim

from model.launchers import Tester, Trainer
from model.loader import get_loader, loaders
from model.network import get_network
from model.util import timeme

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import torch.backends.cudnn as cudnn

import torch.multiprocessing as mp

from torch.utils.tensorboard import SummaryWriter

from nvidia.dali.plugin.pytorch import DALIGenericIterator

import pprint

try:
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator
    from nvidia.dali.pipeline import Pipeline
    import nvidia.dali.ops as ops
    import nvidia.dali.types as types
except ImportError:
    raise ImportError("Please install DALI from https://www.github.com/NVIDIA/DALI to run this example.")


# @timeme
def process_train(args, cfg=None):
    # Since we start from scratch set start epoch as 1
    args.epstart = 1

    # Scale learning rate based on global batch size
    args.lr = args.lr*float(args.batch_size*args.world_size)/256.

 
    kwargs = {'num_threads' : args.workers}

    traindir = os.path.join(args.data_dir, 'train')
    valdir = os.path.join(args.data_dir, 'val')

    # Get a network object and push it onto device
    # network = get_network(args.arch)

    network = models.resnet50()

    # TODO 
    if args.sync_bn:
        print("using apex synced BN")
        model = parallel.convert_syncbn_model(model)

    # FIXME: understand if needed
    if hasattr(torch, 'channels_last') and  hasattr(torch, 'contiguous_format'):
        if args.channels_last:
            memory_format = torch.channels_last
        else:
            memory_format = torch.contiguous_format
        network = network.cuda().to(memory_format=memory_format)
    else:
        network = network.cuda()

    # Instantiate distributed SGD optimizer
    optimizer = optim.SGD(network.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # Horovod: (optional) compression algorithm.
    # FIXME
    # compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none
    compression = hvd.Compression.none
    # Horovod: wrap optimizer with DistributedOptimizer.

    print('World Size', args.world_size)
    if args.world_size > 1:
        #optimizer = hvd.DistributedOptimizer(
        #    optimizer, named_parameters=network.named_parameters(),
        #    compression=compression,
        #    backward_passes_per_step=args.batches_per_allreduce,
        #    op=hvd.Adasum if args.use_adasum else hvd.Average)

        optimizer = hvd.DistributedOptimizer(
                optimizer, named_parameters=network.named_parameters()
            )            
    
    # optimizer = optim.Adam(network.parameters())
    # optimizer = hvd.DistributedOptimizer(
    #    optimizer, named_parameters=network.named_parameters()
    # )

    # Optionally resume from a checkpoint
    if args.resume:
        # Use a local scope to avoid dangling references
        def resume():
            if os.path.isfile(args.resume):
                print("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume, map_location = lambda storage, loc: storage.cuda(args.gpu))
                args.start_epoch = checkpoint['epoch']
                best_prec1 = checkpoint['best_prec1']
                network.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.resume, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))
        resume()

    # Option for Apex/AMP multiprecision training
    if args.amp:
        network, optimizer = amp.initialize(network, optimizer)

    # Instantiate a trainer launcher and run!

    # Horovod: broadcast parameters & optimizer state.
    hvd.broadcast_parameters(network.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    # TODO: fix data loader torchvision
    if args.loader == 'dali':
        train = get_loader('train_loader', args.batch_size, hvd.local_rank(), hvd.size(), traindir, **kwargs)    
        train.build()
        train_loader = DALIClassificationIterator(train, reader_name="Reader", fill_last_batch=False)

        val = get_loader('val_loader', args.batch_size, hvd.local_rank(), hvd.size(), valdir, **kwargs)    
        val.build()
        val_loader = DALIClassificationIterator(val, reader_name="Reader", fill_last_batch=False)

    elif args.loader == 'torchvision':

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

        # Horovod: use DistributedSampler to partition data among workers. Manually specify
        # `num_replicas=hvd.size()` and `rank=hvd.rank()`.
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=hvd.size(), rank=hvd.rank())

        '''
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=allreduce_batch_size,
            sampler=train_sampler, **kwargs)
        '''

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.allreduce_batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler)

        val_dataset = \
            datasets.ImageFolder(valdir,
                                transform=transforms.Compose([
                                    transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                        std=[0.229, 0.224, 0.225])
                                ]))
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, num_replicas=hvd.size(), rank=hvd.rank())
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
                                                sampler=val_sampler, num_workers=args.workers, pin_memory=True)            
  
    launcher = Trainer(args, train_loader, val_loader, network, optimizer)
    launcher.run()

def process_restart(args, cfg):
    cfg['amp_on'] = 1 if args.amp else None
    # Since we restart get start epoch from the file name
    cfg['epstart'] = int(re.search("-(\d+)", args.state).group(1))
    # cfg['load_path'] = cfg['train_path']
    cfg['load_path'] = './img/labels.txt'

    hvd.init()
    torch.cuda.set_device(hvd.local_rank())

    loader = get_loader(cfg, hvd.local_rank(), hvd.size(), split='train')

    # Load a network from file
    network = torch.load(args.state)
    network.cuda()

    optimizer = optim.Adam(network.parameters())
    optimizer = hvd.DistributedOptimizer(
        optimizer, named_parameters=network.named_parameters()
    )

    if args.amp:
        network, optimizer = amp.initialize(network, optimizer)

    launcher = Trainer(cfg, loader, network, optimizer)
    launcher.run()


def process_test(args, cfg):
    cfg['load_path'] = cfg['test_path']

    hvd.init()
    torch.cuda.set_device(hvd.local_rank())

    # One shard
    loader = get_loader(cfg, hvd.local_rank(), 1, split='test')

    if args.state.endswith('.trt'):
        network = TRTModule()
        network.load_state_dict(torch.load(args.state))
    else:
        network = torch.load(args.state)
        network.eval()
    network.cuda()

    launcher = Tester(cfg, loader, network, None)
    launcher.run()


def process_cam(args, cfg):
    cfg['load_path'] = cfg['test_path']
    cfg['batch_size'] = 1

    hvd.init()
    torch.cuda.set_device(hvd.local_rank())

    loader = get_loader(cfg, hvd.local_rank(), 1, split='test')

    network = torch.load(args.state)
    network.eval()

    optimizer = optim.Adam(network.parameters())

    launcher = Tester(cfg, loader, network, optimizer)
    launcher.cam()


def process_convert(args, cfg):
    state = torch.load(args.state)
    path = args.state.replace('.pt', '.trt')
    bs = cfg['batch_size']

    # Three channels
    dummy_input = torch.zeros(bs, 3, *cfg['image_size'])
    dummy_input = dummy_input.type(torch.cuda.FloatTensor)

    state_trt = torch2trt(state, [dummy_input], max_batch_size=bs)
    torch.save(state_trt.state_dict(), path)


def main():


    # parser = argparse.ArgumentParser(description='Deployment tool')
    # subparsers = parser.add_subparsers()

    # add_p = subparsers.add_parser('add')
    # add_p.add_argument("name")
    # add_p.add_argument("--web_port")

    # upg_p = subparsers.add_parser('upgrade')
    # upg_p.add_argument("name")

    ap = ArgumentParser(description='New Image Classifier')
    sp = ap.add_subparsers(dest='cmd')

    # Run from scratch
    ap_run = sp.add_parser('train')
    ap_run.add_argument('--log-dir', default='./logs', 
            help='tensorboard log directory')
    ap_run.add_argument('--epochs', type=int, default=90,
                    help='number of epochs to train')
    ap_run.add_argument('--base-lr', type=float, default=0.0125,
                    help='learning rate for a single GPU')     
    ap_run.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
    ap_run.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='Initial learning rate.  Will be scaled by <global batch size>/256: args.lr = args.lr*float(args.batch_size*args.world_size)/256.  A warmup schedule will also be applied over the first 5 epochs.')
    ap_run.add_argument('--batches-per-allreduce', type=int, default=1,
                    help='number of batches processed locally before '
                         'executing allreduce across workers; it multiplies '
                         'total batch size.')       
    ap_run.add_argument('--use-adasum', action='store_true', default=False,
                    help='use adasum algorithm to do reduction')       
    ap_run.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size per process (default: 256)')
    ap_run.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')                    
    ap_run.add_argument('--print-freq', type=int, default=10,
                    help='output frequency')
    ap_run.add_argument('--sync_bn', action='store_true',
                    help='enabling apex sync BN.')        
    ap_run.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')           
    ap_run.add_argument('--data-dir', default='/workspace/imagenet', 
                    help='data loading path')
    ap_run.add_argument('--loss-scale', type=str, default=None)     
    ap_run.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')    
    ap_run.add_argument('--warmup-epochs', type=float, default=5,
                        help='number of warmup epochs')

    ap_run.set_defaults(process=process_train)

    # Restart from state
    ap_restart = sp.add_parser('restart')
    ap_restart.add_argument('state', help='state file')
    ap_restart.set_defaults(process=process_restart)

    # Test model
    ap_restart = sp.add_parser('test')
    ap_restart.add_argument('state', help='state file')
    ap_restart.set_defaults(process=process_test)

    # Run CAM
    ap_restart = sp.add_parser('cam')
    ap_restart.add_argument('state', help='state file')
    ap_restart.set_defaults(process=process_cam)

    # Convert to TensorRT
    ap_convert = sp.add_parser('convert')
    ap_convert.add_argument('state', help='state file')
    ap_convert.set_defaults(process=process_convert)
    
    # Mixed precision
    ap.add_argument('--amp', '-a', action='store_true')

    # Deterministic runtime
    ap.add_argument('--deterministic', action='store_true')

    # CPU Based DALI pipeline
    ap.add_argument('--dali_cpu', action='store_true', 
        help='Runs CPU based version of DALI pipeline.')

    # FIXME
    ap.add_argument('--channels-last', type=bool, default=False)        

    args = ap.parse_args()

    args.no_cuda = False

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # print(args.log_dir)

    # TODO
    args.seed = 1

    hvd.init()
    torch.manual_seed(args.seed)

    # Horovod: limit # of CPU threads to be used per worker.
    torch.set_num_threads(4)

    if args.cuda:
        # Horovod: pin GPU to local rank.
        torch.cuda.set_device(hvd.local_rank())
        torch.cuda.manual_seed(args.seed)

    args.local_rank = hvd.local_rank()
    args.gpu = args.local_rank
    args.world_size = hvd.size()

    # Enable cudnn benchmark
    cudnn.benchmark = True
    best_prec1 = 0
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.manual_seed(args.local_rank)
        torch.set_printoptions(precision=10)

    assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."

    args.total_batch_size = args.world_size * args.batch_size

    args.allreduce_batch_size = args.batch_size * args.batches_per_allreduce    

    # Horovod: print logs on the first worker.
    # verbose = 1 if hvd.rank() == 0 else 0

    args.loader = 'dali'
    args.arch = 'resnet50'
    args.image_size = (224, 224)

    
    cfg = {'save_nsteps': 5,
           'test_path':
           './data/img',
           'save_path': './trained/state.pt',
           'cam_dir': './cam',
           #'image_size': (96, 96),
           'image_size': (224, 224)}

    #
    # FIXME
    # When supported, use 'forkserver' to spawn dataloader workers instead of 'fork' to prevent
    # issues with Infiniband implementations that are not fork-safe
    if (cfg.get('num_workers', 0) > 0 and hasattr(mp, '_supports_context') and
            mp._supports_context and 'forkserver' in mp.get_all_start_methods()):
        cfg['multiprocessing_context'] = 'forkserver'

    # lr = args.lr*float(args.batch_size*args.world_size)/256.

    if hasattr(args, 'process'):
        args.process(args, cfg)
    else:
        ap.print_help()


if __name__ == '__main__': 
    main()
