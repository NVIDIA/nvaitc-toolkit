from apex import amp
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss


import cv2
import horovod.torch as hvd
import numpy as np
import torch

from model.util import timeme

import pprint
import math
import shutil
import time

from torch.utils.tensorboard import SummaryWriter

class AverageMeter(object):
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


class Trainer(object):
    def __init__(self, args, train_loader, val_loader, network, optimizer):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.network = network
        self.optimizer = optimizer
        self.lossfunc = CrossEntropyLoss().cuda()
        self.args = args

        # FIXME: comulative log writer
        # Horovod: write TensorBoard logs on first worker.
        self.log_writer = SummaryWriter(args.log_dir) if hvd.rank() == 0 else None

    def save(self, e):
        if hvd.local_rank() == 0:
            path = self.cfg['save_path']
            torch.save(self.network, path.replace('.pt', '-{0}.pt'.format(e)))


    @timeme
    def run(self):

        best_prec1 = 0

        for e in range(self.args.epstart, self.args.epochs + 1):
            batch_time = AverageMeter()
            losses = AverageMeter()
            top1 = AverageMeter()
            top5 = AverageMeter()

            self.network.train()

            end = time.time()


            # Torchvision
            
            for i, (images, target) in enumerate(self.train_loader):
                # measure data loading time
                # data_time.update(time.time() - end)
                images = images.cuda(device=self.args.gpu, non_blocking=True)
                target = target.cuda(device=self.args.gpu, non_blocking=True)
            
            # DALI
            
            # for i, data in enumerate(self.train_loader):
            #    # print(type(data), len(data), type(data[0]))
            #    images = data[0]['data']
            #    target = data[0]['label'].squeeze().cuda().long()

            #    images = images.cuda()
            #    target = target.cuda()
            #    train_loader_len = int(math.ceil(self.train_loader._size / self.args.batch_size))
            #             
                train_loader_len = int(math.ceil(self.train_loader.__len__() / self.args.batch_size))

                adjust_learning_rate(self.optimizer, e, i, train_loader_len, self.args.lr)
                output = self.network(images)
                loss = self.lossfunc(output, target)

                # compute gradient and do SGD step

                self.optimizer.zero_grad()
                if self.args.amp:
                    with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                        if self.args.world_size > 1:
                            self.optimizer.synchronize()
                    if self.args.world_size > 1:
                        with self.optimizer.skip_synchronize():
                            self.optimizer.step()
                    else: 
                        self.optimizer.step()
                else:
                    loss.backward()
                    self.optimizer.step()

                if i % self.args.print_freq == 0:
                    # Every print_freq iterations, check the loss, accuracy, and speed.
                    # For best performance, it doesn't make sense to print these metrics every
                    # iteration, since they incur an allreduce and some host<->device syncs.

                    # Measure accuracy
                    prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

                    # Average loss and accuracy across processes for logging
                    # TODO: fix with horovod
                    if self.args.world_size > 1:
                        reduced_loss = reduce_tensor(loss.data)
                        prec1 = reduce_tensor(prec1)
                        prec5 = reduce_tensor(prec5)
                    else:
                        reduced_loss = loss.data

                    # to_python_float incurs a host<->device sync
                    losses.update(to_python_float(reduced_loss), images.size(0))
                    top1.update(to_python_float(prec1), images.size(0))
                    top5.update(to_python_float(prec5), images.size(0))

                    # torch.cuda.synchronize()

                    batch_time.update((time.time() - end)/self.args.print_freq)                   
                    end = time.time()

                    if hvd.local_rank() == 0:
                        print('Epoch: [{0}][{1}/{2}]\t'
                            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                            'Speed {3:.3f} ({4:.3f})\t'
                            'Loss {loss.val:.10f} ({loss.avg:.4f})\t'
                            'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                            'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                            e, i, train_loader_len,
                            hvd.size()*self.args.batch_size/batch_time.val,
                            hvd.size()*self.args.batch_size/batch_time.avg,
                            batch_time=batch_time,
                            loss=losses, top1=top1, top5=top5))
        
            # evaluate on validation set
            [prec1, prec5] = self.validate(self.val_loader, self.network, self.lossfunc)

            # remember best prec@1 and save checkpoint
            if self.args.local_rank == 0:
                is_best = prec1 > best_prec1
                best_prec1 = max(prec1, best_prec1)
                # FIXME: save checkpoint
                save_checkpoint({
                    'epoch': e + 1,
                    'arch': self.args.arch,
                    'state_dict': self.network.state_dict(),
                    'best_prec1': best_prec1,
                    'optimizer' : self.optimizer.state_dict(),
                }, is_best)
                
                if e == self.args.epochs - 1:
                    print('##Top-1 {0}\n'
                        '##Top-5 {1}\n'
                        '##Perf  {2}'.format(
                        prec1,
                        prec5,
                        self.args.batch_size / total_time.avg))

            if self.args.loader == 'dali':
                self.train_loader.reset()
                self.val_loader.reset()

            if self.log_writer:
                # self.log_writer.add_scalar('train/loss', train_loss.avg, epoch)
                self.log_writer.add_scalar('train/loss', losses.avg, e)
                self.log_writer.add_scalar('train/accuracy', top1.avg, e)
                # self.log_writer.add_scalar('train/accuracy', train_accuracy.avg, epoch)

    def validate(self, val_loader, network, criterion):
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        # switch to evaluate mode
        self.network.eval()

        end = time.time()

        i = 1

        # Torchvision
        for input, target in val_loader:
            input, target = input.cuda(), target.cuda()
            val_loader_len = int(val_loader.__len__() / self.args.batch_size)

        # DALI
        #for i, data in enumerate(val_loader):
        #    input = data[0]["data"]
        #    target = data[0]["label"].squeeze().cuda().long()
        #    val_loader_len = int(val_loader._size / self.args.batch_size)

            # compute output
            with torch.no_grad():
                output = network(input)
                loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

            if self.args.world_size > 1:
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
                print('Test: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Speed {2:.3f} ({3:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, val_loader_len,
                    self.args.world_size * self.args.batch_size / batch_time.val,
                    self.args.world_size * self.args.batch_size / batch_time.avg,
                    batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))

            i += 1
            
        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
            .format(top1=top1, top5=top5))

        return [top1.avg, top5.avg]


                
class Tester(object):
    def __init__(self, cfg, loader, network, optimizer):
        self.iterator = loader.get()
        self.network = network
        self.optimizer = optimizer
        # self.lossfunc = BCEWithLogitsLoss()
        self.cfg = cfg
            # Horovod: write TensorBoard logs on first worker.
        self.log_writer = SummaryWriter(cfg['log_dir']) if hvd.rank() == 0 else None


    def run(self):
        accum = 0
        total = 0

        for i, batch in enumerate(self.iterator):
            for j, data in enumerate(batch):
                indata = data['data'].type(torch.cuda.FloatTensor)
                label = data['label'].type(torch.cuda.FloatTensor)

                outputs = self.network(indata)

                outputs = outputs.cpu().data
                outputs[outputs < 0.0] = 0.0
                outputs[outputs > 0.0] = 1.0
                corr = (outputs == label.cpu().data)
                corr = torch.sum(corr.all(axis=1))

                accum += corr
                total += self.cfg['batch_size']

        print('Accuracy: ', accum.float()/total)

    @timeme
    def cam(self):

        # https://snappishproductions.com/blog/2018/01/03/class-activation-mapping-in-pytorch.html.html
        
        for i, batch in enumerate(self.iterator):
            for j, data in enumerate(batch):
                print('Processing img ', i)

                indata = data['data'].type(torch.cuda.FloatTensor)
                label = data['label'].type(torch.cuda.FloatTensor)

                self.optimizer.zero_grad()

                outputs = self.network(indata)

                # CAM via hook
                gradients = self.network.get_activations_gradient()
                pooled_grad = torch.mean(gradients, dim=[0, 2, 3])
                activations = self.network.get_activations(indata)
                activations = activations.cpu().detach()
                for k in range(self.network.avg_pool_dim):
                    activations[:, i, :, :] *= pooled_grad[k]

                heatmap = torch.mean(activations, dim=1).squeeze()
                heatmap = np.maximum(heatmap, 0)

                # Normalise
                heatmap /= torch.max(heatmap)
                heatmap = heatmap.data.numpy()
                
                # Postproc and write
                img = 255*indata[0].permute(1, 2, 0).cpu().numpy()
                heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
                heatmap = np.uint8(255*heatmap)
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                superimposed_img = heatmap*0.7 + img

                out = float(outputs.cpu().data)
                lab = int(label.cpu().data)
                cdir = self.cfg['cam_dir']
                cv2.imwrite(
                    '{0}/L{1}_O{2:.2f}_i{3}.jpg'.format(cdir,lab, out, i),
                    superimposed_img
                )

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def reduce_tensor(tensor):
    '''
    rt = tensor.clone()
    # Switch to horovod
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= args.world_size
    return rt
    '''

    rt = tensor.clone()
    hvd.allreduce(rt, op=hvd.Sum)
    rt /= hvd.size()
    return rt

# item() is a recent addition, so this helps with backward compatibility.
def to_python_float(t):
    if hasattr(t, 'item'):
        return t.item()
    else:
        return t[0]

def adjust_learning_rate(optimizer, epoch, step, len_epoch, lr):
    """LR schedule that should yield 76% converged accuracy with batch size 256"""
    factor = epoch // 30

    if epoch >= 80:
        factor = factor + 1
        
    lr = lr*(0.1**factor)

    """Warmup"""
    if epoch < 5:
        lr = lr*float(1 + step + epoch*len_epoch)/(5.*len_epoch)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')        