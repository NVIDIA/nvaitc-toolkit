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
import torch.nn.functional as F

from tqdm import tqdm

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
        resume_from_epoch = 0
        verbose = True
        
        for epoch in range(resume_from_epoch, self.args.epochs):
            self.network.train()

            train_loss = Metric('train_loss')
            train_accuracy = Metric('train_accuracy')

            end = time.time()

            with tqdm(total=self.train_loader._size,
            # with tqdm(total=len(self.train_loader),
                    #  with tqdm(total=len(train_loader),
                    desc='Train Epoch #{}'.format(epoch + 1),
                    disable=not verbose) as t:

                # Torchvision
                # for i, (data, target) in enumerate(self.train_loader):
                for i, dt in enumerate(self.train_loader):
                    data = dt[0]['data']
                    target = dt[0]['label'].squeeze().cuda().long()
                    
                    batch_idx = i
                    self.adjust_learning_rate(epoch, batch_idx)

                    if self.args.cuda:
                        data, target = data.cuda(), target.cuda()

                    self.optimizer.zero_grad()
                    # Split data into sub-batches of size batch_size
                    for i in range(0, len(data), self.args.batch_size):
                        data_batch = data[i:i + self.args.batch_size]
                        target_batch = target[i:i + self.args.batch_size]
                        output = self.network(data_batch)
                        train_accuracy.update(accuracy(output, target_batch))
                        loss = F.cross_entropy(output, target_batch)
                        train_loss.update(loss)
                        # Average gradients among sub-batches
                        loss.div_(math.ceil(float(len(data)) / self.args.batch_size))

                        if self.args.amp:
                            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                                scaled_loss.backward()
                        else:
                            loss.backward()

                        # loss.backward()
                    # Gradient is applied across all ranks

                    if self.args.amp and self.args.world_size > 1:
                            self.optimizer.synchronize()
                            with self.optimizer.skip_synchronize():
                                self.optimizer.step()
                    else:
                        self.optimizer.step()      

                    # self.optimizer.step()

                    t.set_postfix({'loss': train_loss.avg.item(),
                                'accuracy': 100. * train_accuracy.avg.item()})
                    t.update(1)

            if log_writer:
                log_writer.add_scalar('train/loss', train_loss.avg, epoch)
                log_writer.add_scalar('train/accuracy', train_accuracy.avg, epoch)

    # Horovod: using `lr = base_lr * hvd.size()` from the very beginning leads to worse final
    # accuracy. Scale the learning rate `lr = base_lr` ---> `lr = base_lr * hvd.size()` during
    # the first five epochs. See https://arxiv.org/abs/1706.02677 for details.
    # After the warmup reduce learning rate by 10 on the 30th, 60th and 80th epochs.
    def adjust_learning_rate(self, epoch, batch_idx):
        if epoch < self.args.warmup_epochs:
            epoch += float(batch_idx + 1) / self.train_loader._size
            # epoch += float(batch_idx + 1) / len(self.train_loader)
            lr_adj = 1. / hvd.size() * (epoch * (hvd.size() - 1) / self.args.warmup_epochs + 1)
        elif epoch < 30:
            lr_adj = 1.
        elif epoch < 60:
            lr_adj = 1e-1
        elif epoch < 80:
            lr_adj = 1e-2
        else:
            lr_adj = 1e-3
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.args.base_lr * hvd.size() * self.args.batches_per_allreduce * lr_adj

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
        #for input, target in val_loader:
        #    input, target = input.cuda(), target.cuda()
        #    val_loader_len = int(len(val_loader)/ self.args.batch_size)

        # DALI
        for i, data in enumerate(val_loader):
            input = data[0]["data"]
            target = data[0]["label"].squeeze().cuda().long()
            val_loader_len = int(val_loader._size / self.args.batch_size)

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




# Horovod: average metrics from distributed training.
class Metric(object):
    def __init__(self, name):
        self.name = name
        self.sum = torch.tensor(0.)
        self.n = torch.tensor(0.)

    def update(self, val):
        self.sum += hvd.allreduce(val.detach().cpu(), name=self.name)
        self.n += 1

    @property
    def avg(self):
        return self.sum / self.n

'''
def accuracy(output, target):
    # get the index of the max log-probability
    pred = output.max(1, keepdim=True)[1]
    return pred.eq(target.view_as(pred)).cpu().float().mean()
'''
