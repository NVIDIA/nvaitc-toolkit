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

try:
    from nvidia.dali.pipeline import pipeline_def
    import nvidia.dali.types as types
    import nvidia.dali.fn as fn
    from nvidia.dali.auto_aug import auto_augment
except ImportError:
    raise ImportError("Please install DALI from https://www.github.com/NVIDIA/DALI to run this example.")

import os
import sys

from random import shuffle


@pipeline_def(enable_conditionals=True) #batch_size, num_threads, device_id, seed, 
def image_net_train_pipe(data_dir, crop, dali_cpu, shard_id, num_shards, dali_auto_augment):

    mirror = fn.random.coin_flip(probability=0.5)

    images, labels = fn.readers.file(file_root=data_dir,
                                shard_id=shard_id,
                                num_shards=num_shards,
                                random_shuffle=True,
                                pad_last_batch=True,
                                name="Reader")
    #let user decide which pipeline works him bets for RN version he runs
    dali_device = 'cpu' if dali_cpu else 'gpu'
    decoder_device = 'cpu' if dali_cpu else 'mixed'
    # This padding sets the size of the internal nvJPEG buffers to be able to 
    # handle all images from full-sized ImageNet without additional reallocations
    device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
    host_memory_padding = 140544512 if decoder_device == 'mixed' else 0
    images = fn.decoders.image_random_crop(images,
                                                device=decoder_device,
                                                output_type=types.RGB,
                                                device_memory_padding=device_memory_padding,
                                                host_memory_padding=host_memory_padding,
                                                random_aspect_ratio=[0.8, 1.25],
                                                random_area=[0.1, 1.0],
                                                num_attempts=100)
    images = fn.resize(images,
                            device=dali_device,
                            resize_x=crop,
                            resize_y=crop, 
                            interp_type=types.INTERP_LINEAR,
                            antialias=True)
    
    images = images.gpu()
    images = fn.flip(images, horizontal=mirror)

    if dali_auto_augment:
        images = auto_augment.auto_augment_image_net(images, shape=[crop, crop])
    
    images = fn.crop_mirror_normalize(images, 
                                        dtype=types.FLOAT, 
                                        output_layout=types.NCHW,
                                        crop=(crop, crop),
                                        mean=[0.485 * 255,0.456 * 255,0.406 * 255],
                                        std=[0.229 * 255,0.224 * 255,0.225 * 255])
    
    # print('DALI "{0}" variant'.format(dali_device))
    return images, labels


@pipeline_def #batch_size, num_threads, device_id, seed, 
def image_net_val_pipe(data_dir, crop, size, dali_cpu, shard_id, num_shards): #world_size

    images, labels = fn.readers.file(file_root=data_dir,
                                shard_id=shard_id,
                                num_shards=num_shards,
                                random_shuffle=True,
                                pad_last_batch=True,
                                name="Reader")
    #let user decide which pipeline works him bets for RN version he runs
    dali_device = 'cpu' if dali_cpu else 'gpu'
    decoder_device = 'cpu' if dali_cpu else 'mixed'
    # This padding sets the size of the internal nvJPEG buffers to be able to 
    # handle all images from full-sized ImageNet without additional reallocations
    device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
    host_memory_padding = 140544512 if decoder_device == 'mixed' else 0
    images = fn.decoders.image(images,
                                device=decoder_device,
                                output_type=types.RGB)

    images = fn.resize(images,
                        device=dali_device,
                        size=size, 
                        mode="not_smaller",
                        interp_type=types.INTERP_LINEAR,
                        antialias=True)
    
    mirror = False
    images = fn.crop_mirror_normalize(images.gpu(), 
                                        dtype=types.FLOAT, 
                                        output_layout=types.NCHW,
                                        crop=(crop, crop),
                                        mean=[0.485 * 255,0.456 * 255,0.406 * 255],
                                        std=[0.229 * 255,0.224 * 255,0.225 * 255],
                                        mirror=mirror)
    
    # print('DALI "{0}" variant'.format(dali_device))
    labels = labels.gpu()
    return images, labels
