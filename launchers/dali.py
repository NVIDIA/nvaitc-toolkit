
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

from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
import numpy as np
import torch
from util import timeme
import pprint
import math
import shutil
import time
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import datetime
import torch.distributed as dist

class DALITrainer:
    def __init__(self, args, train_loader, val_loader, model, optimizer):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model
        self.optimizer = optimizer
        self.criterion = CrossEntropyLoss().cuda()
        self.args = args
        self.experiment_name = '{0}-{1}_{2}_{3}_{4}'.format(datetime.datetime.now().strftime("%Y%m%d"), args.arch, 'dali', args.batch_size, 'amp' if args.amp else 'noamp')
        # Horovod: write TensorBoard logs on first worker.
        self.log_writer = SummaryWriter(log_dir=args.log_dir + '/' + self.experiment_name, comment=self.experiment_name) if args.global_rank == 0 else None
        if args.amp:
            self.scaler = torch.cuda.amp.GradScaler()
    def save(self, e):
        if self.args.global_rank == 0:
            path = self.cfg['save_path']
            torch.save(self.network, path.replace('.pt', '-{0}.pt'.format(e)))

    def run(self):
        total_time = AverageMeter()
        best_prec1 = self.args.best_prec1

        for epoch in range(self.args.start_epoch, self.args.epochs):
            batch_time = AverageMeter()
            losses = AverageMeter()
            top1 = AverageMeter()
            top5 = AverageMeter()

            # switch to train mode
            self.model.train()
            end = time.time()

            for i, data in enumerate(self.train_loader):
                input = data[0]["data"]
                target = data[0]["label"].squeeze().cuda().long()
                train_loader_len = int(math.ceil(
                    self.train_loader._size / self.args.batch_size))

                if self.args.prof >= 0 and i == self.args.prof:
                    print("Profiling begun at iteration {}".format(i))
                    torch.cuda.cudart().cudaProfilerStart()

                if self.args.prof >= 0: torch.cuda.nvtx.range_push("Body of iteration {}".format(i))

                self.adjust_learning_rate(epoch, i, train_loader_len)

                # compute output
                if self.args.prof >= 0: torch.cuda.nvtx.range_push("forward")
                if self.args.amp:
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        output = self.model(input)
                        loss = self.criterion(output, target)
                else:
                    output = self.model(input)
                    loss = self.criterion(output, target)
                if self.args.prof >= 0: torch.cuda.nvtx.range_pop()

                # compute gradient and do SGD step
                # self.optimizer.zero_grad()
                # self.model.zero_grad()
                # model.zero_grad() and optimizer.zero_grad() are the same if all model parameters are in that optimizer

                # more efficient way to zero gradients
                for param in self.model.parameters():
                    param.grad = None

                if self.args.prof >= 0: torch.cuda.nvtx.range_push("backward")

                if self.args.amp:
                    if self.args.prof >= 0: torch.cuda.nvtx.range_push("backward pass with mixed precision")
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    if self.args.prof >= 0: torch.cuda.nvtx.range_pop()
                else:
                    if self.args.prof >= 0: torch.cuda.nvtx.range_push("backward pass w/o mixed precision")
                    loss.backward()
                    self.optimizer.step()
                    if self.args.prof >= 0: torch.cuda.nvtx.range_pop()

                if i%self.args.print_freq == 0:
                    # Every print_freq iterations, check the loss, accuracy, and speed.
                    # For best performance, it doesn't make sense to print these metrics every
                    # iteration, since they incur an allreduce and some host<->device syncs.

                    # Measure accuracy
                    prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

                    # Average loss and accuracy across processes for logging
                    if self.args.distributed:
                        reduced_loss = reduce_tensor(loss.data)
                        prec1 = reduce_tensor(prec1)
                        prec5 = reduce_tensor(prec5)
                    else:
                        reduced_loss = loss.data

                    # to_python_float incurs a host<->device sync
                    losses.update(to_python_float(reduced_loss), input.size(0))
                    top1.update(to_python_float(prec1), input.size(0))
                    top5.update(to_python_float(prec5), input.size(0))

                    torch.cuda.synchronize()
                    batch_time.update((time.time() - end)/self.args.print_freq)
                    end = time.time()

                    if self.args.local_rank == 0:
                        print(('Epoch: [{0}][{1}/{2}]\t'
                            + 'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                            + 'Speed {3:.3f} ({4:.3f})\t'
                            + 'Loss {loss.val:.10f} ({loss.avg:.4f})\t'
                            + 'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                            + 'Prec@5 {top5.val:.3f} ({top5.avg:.3f})').format(
                            epoch, i, train_loader_len,
                            self.args.world_size*self.args.batch_size/batch_time.val,
                            self.args.world_size*self.args.batch_size/batch_time.avg,
                            batch_time=batch_time,
                            loss=losses, top1=top1, top5=top5))

                        if self.log_writer:
                            self.log_writer.add_scalar('train/loss', losses.avg, epoch)
                            self.log_writer.add_scalar('train/accuracy1', top1.avg, epoch)
                            self.log_writer.add_scalar('train/accuracy5', top5.avg, epoch)

                # Pop range "Body of iteration {}".format(i)
                if self.args.prof >= 0: torch.cuda.nvtx.range_pop()

                if self.args.prof >= 0 and i == self.args.prof + 10:
                    print("Profiling ended at iteration {}".format(i))
                    torch.cuda.cudart().cudaProfilerStop()
                    quit()

            avg_train_time = batch_time.avg

            total_time.update(avg_train_time)

            # evaluate on validation set
            [prec1, prec5] = self.validate(epoch)

            # remember best prec@1 and save checkpoint
            if self.args.local_rank == 0:
                is_best = prec1 > best_prec1
                best_prec1 = max(prec1, best_prec1)
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': self.args.arch,
                    'state_dict': self.model.state_dict(),
                    'best_prec1': best_prec1,
                    'optimizer' : self.optimizer.state_dict(),
                }, is_best)
                if epoch == self.args.epochs - 1:
                    print(('##Top-1 {0}\n'
                        + '##Top-5 {1}\n'
                        + '##Perf  {2}').format(
                        prec1,
                        prec5,
                        int(self.args.total_batch_size / total_time.avg)))

            self.train_loader.reset()
            self.val_loader.reset()

    # Horovod: using `lr = base_lr * hvd.size()` from the very beginning leads to worse final
    # accuracy. Scale the learning rate `lr = base_lr` ---> `lr = base_lr * hvd.size()` during
    # the first five epochs. See https://arxiv.org/abs/1706.02677 for details.
    # After the warmup reduce learning rate by 10 on the 30th, 60th and 80th epochs.
    def adjust_learning_rate(self, epoch, step, len_epoch):
        """LR schedule that should yield 76% converged accuracy with batch size 256"""
        factor = epoch // 30

        if epoch >= 80:
            factor = factor + 1

        lr = self.args.lr*(0.1**factor)

        """Warmup"""
        if epoch < 5:
            lr = lr*float(1 + step + epoch*len_epoch)/(5.*len_epoch)

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        if self.log_writer:
            self.log_writer.add_scalar('lr/lr', lr, epoch)

    def validate(self, epoch):
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        # switch to evaluate mode
        self.model.eval()

        end = time.time()

        for i, data in enumerate(self.val_loader):
            input = data[0]["data"]
            target = data[0]["label"].squeeze().cuda().long()
            val_loader_len = int(self.val_loader._size / self.args.batch_size)

            # compute output
            with torch.no_grad():
                output = self.model(input)
                loss = self.criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

            if self.args.distributed:
                reduced_loss = reduce_tensor(loss.data)
                prec1 = reduce_tensor(prec1)
                prec5 = reduce_tensor(prec5)
            else:
                reduced_loss = loss.data

            losses.update(to_python_float(reduced_loss), input.size(0))
            top1.update(to_python_float(prec1), input.size(0))
            top5.update(to_python_float(prec5), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # TODO:  Change timings to mirror train().
            if self.args.local_rank == 0 and i % self.args.print_freq == 0:
                print(('Test: [{0}/{1}]\t'
                    + 'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    + 'Speed {2:.3f} ({3:.3f})\t'
                    + 'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    + 'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    + 'Prec@5 {top5.val:.3f} ({top5.avg:.3f})').format(
                    i, val_loader_len,
                    self.args.world_size * self.args.batch_size / batch_time.val,
                    self.args.world_size * self.args.batch_size / batch_time.avg,
                    batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))
                    
                if self.log_writer:
                    self.log_writer.add_scalar('val/loss', losses.avg, epoch)
                    self.log_writer.add_scalar('val/accuracy1', top1.avg, epoch)
                    self.log_writer.add_scalar('val/accuracy5', top5.avg, epoch)

        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
            .format(top1=top1, top5=top5))

        return [top1.avg, top5.avg]

def reduce_tensor(tensor):
    '''
    rt = tensor.clone()
    # Switch to horovod
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= args.world_size
    return rt
    '''

    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.AVG)
    return rt

def to_python_float(t):
    if hasattr(t, 'item'):
        return t.item()
    else:
        return t[0]

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')        

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

# Horovod: average metrics from distributed training.
class Metric:
    def __init__(self, name):
        self.name = name
        self.sum = torch.tensor(0.)
        self.n = torch.tensor(0.)

    def update(self, val):
        self.sum += dist.all_reduce(val.detach().cpu(), name=self.name)
        self.n += 1

    @property
    def avg(self):
        return self.sum / self.n

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count