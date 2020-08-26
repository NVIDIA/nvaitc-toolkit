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
from torch.utils.tensorboard import SummaryWriter

class DALITrainer(object):
    def __init__(self, args, train_loader, val_loader, model, optimizer):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model
        self.optimizer = optimizer
        self.criterion = CrossEntropyLoss().cuda()
        self.args = args

        # Horovod: write TensorBoard logs on first worker.
        self.log_writer = SummaryWriter(args.log_dir) if hvd.rank() == 0 else None

    def save(self, e):
        if hvd.local_rank() == 0:
            path = self.cfg['save_path']
            torch.save(self.network, path.replace('.pt', '-{0}.pt'.format(e)))

    def run(self):
        total_time = AverageMeter()
        best_prec1 = 0

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
                train_loader_len = int(math.ceil(self.train_loader._size / self.args.batch_size))

                # if args.prof >= 0 and i == args.prof:
                #    print("Profiling begun at iteration {}".format(i))
                #    torch.cuda.cudart().cudaProfilerStart()

                # if args.prof >= 0: torch.cuda.nvtx.range_push("Body of iteration {}".format(i))

                self.adjust_learning_rate(epoch, i, train_loader_len)
                #if args.test:
                #    if i > 10:
                #        break

                # compute output
                #if args.prof >= 0: torch.cuda.nvtx.range_push("forward")
                output = self.model(input)
                #if args.prof >= 0: torch.cuda.nvtx.range_pop()
                loss = self.criterion(output, target)

                # compute gradient and do SGD step
                if self.args.distributed:
                    self.optimizer.synchronize()
                self.optimizer.zero_grad()

                #if args.prof >= 0: torch.cuda.nvtx.range_push("backward")

                if self.args.amp is not None:
                    with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                
                #if args.prof >= 0: torch.cuda.nvtx.range_pop()

                #if args.prof >= 0: torch.cuda.nvtx.range_push("optimizer.step()")
                if self.args.distributed:
                    with self.optimizer.skip_synchronize():
                        self.optimizer.step()
                else:
                    self.optimizer.step()
                #if args.prof >= 0: torch.cuda.nvtx.range_pop()

                if i%self.args.print_freq == 0:
                    # Every print_freq iterations, check the loss, accuracy, and speed.
                    # For best performance, it doesn't make sense to print these metrics every
                    # iteration, since they incur an allreduce and some host<->device syncs.

                    # Measure accuracy
                    prec1, prec5 = self.accuracy(output.data, target, topk=(1, 5))

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
                        print('Epoch: [{0}][{1}/{2}]\t'
                            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                            'Speed {3:.3f} ({4:.3f})\t'
                            'Loss {loss.val:.10f} ({loss.avg:.4f})\t'
                            'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                            'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                            epoch, i, train_loader_len,
                            self.args.world_size*self.args.batch_size/batch_time.val,
                            self.args.world_size*self.args.batch_size/batch_time.avg,
                            batch_time=batch_time,
                            loss=losses, top1=top1, top5=top5))

                # Pop range "Body of iteration {}".format(i)
                #if args.prof >= 0: torch.cuda.nvtx.range_pop()

                #if args.prof >= 0 and i == args.prof + 10:
                #    print("Profiling ended at iteration {}".format(i))
                #    torch.cuda.cudart().cudaProfilerStop()
                #    quit()

            avg_train_time = batch_time.avg

            total_time.update(avg_train_time)
            #if args.test:
            #    break

            # evaluate on validation set
            [prec1, prec5] = self.validate()

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
                    print('##Top-1 {0}\n'
                        '##Top-5 {1}\n'
                        '##Perf  {2}'.format(
                        prec1,
                        prec5,
                        self.args.total_batch_size / total_time.avg))

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

    def accuracy(self, output, target, topk=(1,)):
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

    def validate(self):
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
            prec1, prec5 = self.accuracy(output.data, target, topk=(1, 5))

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

        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
            .format(top1=top1, top5=top5))

        return [top1.avg, top5.avg]

class TVTrainer(object):

    def __init__(self, args, train_loader, train_sampler, val_loader, model, optimizer):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model
        self.optimizer = optimizer
        self.criterion = CrossEntropyLoss().cuda()
        self.train_sampler = train_sampler
        self.args = args

        # FIXME: comulative log writer
        # Horovod: write TensorBoard logs on first worker.
        self.log_writer = SummaryWriter(self.args.log_dir) if hvd.rank() == 0 else None
    
    def run(self):
        #for epoch in range(args.start_epoch, args.epochs):
        for epoch in range(0, self.args.epochs):
            if self.args.distributed:
                self.train_sampler.set_epoch(epoch)

            batch_time = AverageMeter()
            losses = AverageMeter()
            top1 = AverageMeter()
            top5 = AverageMeter()

            # switch to train mode
            self.model.train()
            end = time.time()

            prefetcher = self.data_prefetcher(self.train_loader)
            input, target = prefetcher.next()
            i = 0
            while input is not None:
                i += 1
                #if args.prof >= 0 and i == args.prof:
                #    print("Profiling begun at iteration {}".format(i))
                #    torch.cuda.cudart().cudaProfilerStart()

                #if args.prof >= 0: torch.cuda.nvtx.range_push("Body of iteration {}".format(i))

                self.adjust_learning_rate(epoch, i, len(self.train_loader))

                # compute output
                #if args.prof >= 0: torch.cuda.nvtx.range_push("forward")
                output = self.model(input)
                #if args.prof >= 0: torch.cuda.nvtx.range_pop()
                loss = self.criterion(output, target)

                # compute gradient and do SGD step
                self.optimizer.synchronize()
                self.optimizer.zero_grad()

                #if args.prof >= 0: torch.cuda.nvtx.range_push("backward")
                if self.args.amp is not None:
                    with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                #if args.prof >= 0: torch.cuda.nvtx.range_pop()

                # for param in model.parameters():
                #     print(param.data.double().sum().item(), param.grad.data.double().sum().item())

                #if args.prof >= 0: torch.cuda.nvtx.range_push("optimizer.step()")
                with self.optimizer.skip_synchronize():
                    self.optimizer.step()
                #if args.prof >= 0: torch.cuda.nvtx.range_pop()

                if i%self.args.print_freq == 0:
                    # Every print_freq iterations, check the loss, accuracy, and speed.
                    # For best performance, it doesn't make sense to print these metrics every
                    # iteration, since they incur an allreduce and some host<->device syncs.

                    # Measure accuracy
                    prec1, prec5 = self.accuracy(output.data, target, topk=(1, 5))

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
                        print('Epoch: [{0}][{1}/{2}]\t'
                            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                            'Speed {3:.3f} ({4:.3f})\t'
                            'Loss {loss.val:.10f} ({loss.avg:.4f})\t'
                            'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                            'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                            epoch, i, len(self.train_loader),
                            self.args.world_size*self.args.batch_size/batch_time.val,
                            self.args.world_size*self.args.batch_size/batch_time.avg,
                            batch_time=batch_time,
                            loss=losses, top1=top1, top5=top5))
                #if args.prof >= 0: torch.cuda.nvtx.range_push("prefetcher.next()")
                input, target = prefetcher.next()
                #if args.prof >= 0: torch.cuda.nvtx.range_pop()

                # Pop range "Body of iteration {}".format(i)
                #if args.prof >= 0: torch.cuda.nvtx.range_pop()

                #if args.prof >= 0 and i == args.prof + 10:
                #    print("Profiling ended at iteration {}".format(i))
                #    torch.cuda.cudart().cudaProfilerStop()
                #    quit()

            # evaluate on validation set
            prec1 = self.validate()

            # remember best prec@1 and save checkpoint
            if args.local_rank == 0:
                is_best = prec1 > best_prec1
                best_prec1 = max(prec1, best_prec1)
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'best_prec1': best_prec1,
                    'optimizer' : optimizer.state_dict(),
                }, is_best)

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

    def accuracy(self, output, target, topk=(1,)):
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

    def validate(self):
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        # switch to evaluate mode
        self.model.eval()

        end = time.time()

        prefetcher = self.data_prefetcher(self.val_loader)
        input, target = prefetcher.next()
        i = 0
        while input is not None:
            i += 1

            # compute output
            with torch.no_grad():
                output = self.model(input)
                loss = self.criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = self.accuracy(output.data, target, topk=(1, 5))

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
                print('Test: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Speed {2:.3f} ({3:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(self.val_loader),
                    self.args.world_size * self.args.batch_size / batch_time.val,
                    self.args.world_size * self.args.batch_size / batch_time.avg,
                    batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))

            input, target = prefetcher.next()

        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
            .format(top1=top1, top5=top5))

        return top1.avg

    class data_prefetcher():
        def __init__(self, loader):
            self.loader = iter(loader)
            self.stream = torch.cuda.Stream()
            self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1,3,1,1)
            self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1,3,1,1)
            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.mean = self.mean.half()
            #     self.std = self.std.half()
            self.preload()

        def preload(self):
            try:
                self.next_input, self.next_target = next(self.loader)
            except StopIteration:
                self.next_input = None
                self.next_target = None
                return
 
            with torch.cuda.stream(self.stream):
                self.next_input = self.next_input.cuda(non_blocking=True)
                self.next_target = self.next_target.cuda(non_blocking=True)
                self.next_input = self.next_input.float()
                self.next_input = self.next_input.sub_(self.mean).div_(self.std)

        def next(self):
            torch.cuda.current_stream().wait_stream(self.stream)
            input = self.next_input
            target = self.next_target
            if input is not None:
                input.record_stream(torch.cuda.current_stream())
            if target is not None:
                target.record_stream(torch.cuda.current_stream())
            self.preload()
            return input, target
                
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

def to_python_float(t):
    if hasattr(t, 'item'):
        return t.item()
    else:
        return t[0]

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