from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIGenericIterator as PyTorchIterator
from torch.utils.data import DataLoader
from torchvision.datasets import STL10
from PIL import Image

import numpy as np
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import torchvision

import horovod.torch as hvd

import os
import sys

from random import shuffle

class ImageNetTrainPipe(Pipeline):

    name = 'train_loader'
    
    #{'num_threads': 10, 'file_list': './STL10/train_filelist.txt', 'train_path': './STL10', 'num_shards': 1}

    # (batch_size, rank, ngpu, **kwargs)

    '''
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop,
                 shard_id, num_shards, dali_cpu=False):
    '''

    def __init__(self, batch_size, device_id, num_shards, data_path, num_threads, dali_cpu=False):

        print("Calling DALI Loader ImageNet")

        print('Device ID', device_id)
        print('Num Shards', num_shards)
        print('Num Threads', num_threads)

        crop = 224

        super(ImageNetTrainPipe, self).__init__(batch_size,
                                              num_threads,
                                              device_id,
                                              seed=12 + device_id)
        self.input = ops.FileReader(file_root=data_path,
                                    shard_id=device_id,
                                    num_shards=num_shards,
                                    random_shuffle=True,
                                    pad_last_batch=True)
        #let user decide which pipeline works him bets for RN version he runs
        dali_device = 'cpu' if dali_cpu else 'gpu'
        decoder_device = 'cpu' if dali_cpu else 'mixed'
        # This padding sets the size of the internal nvJPEG buffers to be able to handle all images from full-sized ImageNet
        # without additional reallocations
        device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
        host_memory_padding = 140544512 if decoder_device == 'mixed' else 0
        self.decode = ops.ImageDecoderRandomCrop(device=decoder_device, output_type=types.RGB,
                                                 device_memory_padding=device_memory_padding,
                                                 host_memory_padding=host_memory_padding,
                                                 random_aspect_ratio=[0.8, 1.25],
                                                 random_area=[0.1, 1.0],
                                                 num_attempts=100)
        self.res = ops.Resize(device=dali_device,
                              resize_x=crop,
                              resize_y=crop,
                              interp_type=types.INTERP_TRIANGULAR)
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(crop, crop),
                                            mean=[0.485 * 255,0.456 * 255,0.406 * 255],
                                            std=[0.229 * 255,0.224 * 255,0.225 * 255])
        self.coin = ops.CoinFlip(probability=0.5)
        print('DALI "{0}" variant'.format(dali_device))

    def define_graph(self):
        rng = self.coin()
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images.gpu(), mirror=rng)
        return [output, self.labels]

class ImageNetValPipe(Pipeline):

    name = 'val_loader'

    # Add Crop and Val
    # TODO: drop file list

    # val = get_loader('val_loader', cfg['batch_size'], hvd.local_rank(), hvd.size(), cfg['load_path'] + '/val', **kwargs)

    def __init__(self, batch_size, device_id, num_shards, data_path, num_threads):

        crop_size = 224
        val_size = 256
        
        super(ImageNetValPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)

        self.input = ops.FileReader(file_root=data_path,
                                    shard_id=device_id,
                                    num_shards=num_shards,
                                    random_shuffle=False,
                                    pad_last_batch=True)
        self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)
        self.res = ops.Resize(device="gpu",
                              resize_shorter=val_size,
                              interp_type=types.INTERP_TRIANGULAR)
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            #dtype=types.FLOAT,
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(crop_size, crop_size),
                                            mean=[0.485 * 255,0.456 * 255,0.406 * 255],
                                            std=[0.229 * 255,0.224 * 255,0.225 * 255])

    def define_graph(self):
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images)
        return [output, self.labels]       


class TorchVisionPipeline(object):

    def __init__(self, batch_size, device_id, num_threads, num_shards, file_list, 
                 split):

        print("Calling new TorchVision Loader")

        class Wrapdict(object):
            def __getitem__(self, index):
                if self.labels is not None:
                    img, target = self.data[index], int(self.labels[index])
                else:
                    img, target = self.data[index], None

                img = Image.fromarray(np.transpose(img, (1, 2, 0)))

                if self.transform is not None:
                    img = self.transform(img)

                # Transform to STL-2 (vehicle vs. animal)
                if self.target_transform is not None:
                    # airplane, bird, car, cat, deer,
                    # dog, horse, monkey, ship, truck
                    vehindx = [0, 2, 8, 9]
                    toh = np.array([1]) if target in vehindx else np.array([0])

                return [{'data': img, 'label': toh}]

        ModDS = type('', (Wrapdict, STL10), {})

        # input_dir = '/newt/data/STL10'

        input_dir = file_list

        dataset = ModDS(
            root=input_dir, transform=torchvision.transforms.ToTensor(),
            target_transform=True, download=True, split=split
        )

        self.loader = DataLoader(dataset, batch_size=batch_size,
                                 num_workers=1, shuffle=True)

    def get(self):
        return self.loader

