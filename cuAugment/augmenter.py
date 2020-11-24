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

import sys
import numba
import datetime
import importlib
import numpy as np
from jinja2 import Environment, BaseLoader

from .transformations.base import SpatialTransformation, ColorTransformation
from .templates.augmenter_template import payload

sys.path.append('/tmp')

__all__ = ["JITAugmenter"]

class JITAugmenter:
    
    def __init__(self, dim, channel_mode='CF', gpu=0):
        
        # set up the template engine        
        self.env = Environment(loader=BaseLoader)
        self.env.trim_blocks = True
        self.env.lstrip_blocks = True
        
        self.gpu = gpu
        self.dim = dim
        self.channel_mode = channel_mode
        
        assert isinstance(gpu, int)
        assert 0 < self.dim < 5 and type(self.dim) == int
        assert self.channel_mode in ['CF2CF', 'CL2CL', 'CL2CF', 'CF2CL']
        
        self.config = {
            'year' : datetime.datetime.now().year,
            'dim'  : self.dim,
            'channel_mode': self.channel_mode,
            'spatial_ops': [],
            'color_ops': []
        }
        
        self.libaug = None
        self.string = None
        
    def add_spatial_op(self, spatial_op):
        
        assert isinstance(spatial_op, SpatialTransformation), 'Argument not a spatial transformation'
        assert spatial_op.min_dims() <= self.dim, 'The op issued has a too high dimension'
        
        self.libaug = None
        self.string = None
                
        identifier = len(self.config['spatial_ops'])
        spatial_param  = "spatial_param%s" % identifier
        spatial_params = "spatial_params%s" % identifier
        
        self.config['spatial_ops'].append((spatial_op, spatial_param, spatial_params))

        
    def add_color_op(self, color_op):
        
        assert isinstance(color_op, ColorTransformation), 'Argument not a color transformation'
        assert color_op.min_dims() <= self.dim, 'The op issued has a too high dimension'
        
        self.libaug = None
        self.string = None
                
        identifier = len(self.config['color_ops'])
        color_param  = "color_param%s" % identifier
        color_params = "color_params%s" % identifier
        
        self.config['color_ops'].append((color_op, color_param, color_params))

        
    def compile(self, verbose=False):
        
        self.string = self.env.from_string(payload).render(self.config)

        hash_value = hash(self.string)
        module_name = 'module_%s' % (hash_value if hash_value > 0 else 2**64-hash_value)

        with open("/tmp/"+module_name+'.py', "w") as f:
            f.write(self.string)    
        
        self.libaug = importlib.import_module(module_name)

        if verbose:
            print("STATUS: augmentor code generated in", module_name+".py")
    
    def __call__(self, batch_in, batch_out, empty_values, seed=None, verbose=False):
        
        if isinstance(seed, int):
            np.random.seed(seed)
        
        # get the JIT-compiled kernel
        assert self.libaug != None, 'compile the kernel first!'
        kernel = self.libaug.augment
        
        # channel order for different formats
        CF = {'B':0, 'X':2, 'Y':3, 'Z':4, 'T':5, 'C':1}
        CL = {'B':0, 'X':1, 'Y':2, 'Z':3, 'T':4, 'C':self.dim+1}
        
        # fix input and output channel order
        if self.channel_mode == 'CF2CF':
            dim_in, dim_out = CF, CF
        if self.channel_mode == 'CF2CL':
            dim_in, dim_out = CF, CL
        if self.channel_mode == 'CL2CF':
            dim_in, dim_out = CL, CF
        if self.channel_mode == 'CL2CL':
            dim_in, dim_out = CL, CL
        
        # ensure color channels are compatible
        assert(batch_in.shape[dim_in['C']] == batch_out.shape[dim_out['C']])
        assert(batch_in.shape[dim_in['C']] == empty_values.shape[0])
        
        if verbose:
            if batch_in.shape[dim_in['B']] < batch_out.shape[dim_out['B']]:
                print('STATUS: fewer images in input than in output - sampling in cyclic fashion.')
            if batch_in.shape[dim_in['B']] > batch_out.shape[dim_out['B']]:
                print('STATUS: fewer images output than in input - sampling the first in the batch.')
    
        # TODO: Tune grid later for optimal performance
        if self.dim == 1:
            block = (64, 1, 1)
            grid = (batch_out.shape[dim_out['X']]//64, 1, 1)
        if self.dim == 2:
            block = (8, 8, 1)
            grid = (batch_out.shape[dim_out['Y']]//8, 
                    batch_out.shape[dim_out['X']]//8, 1)
        if self.dim == 3:
            block = (4, 4, 4)
            grid = (batch_out.shape[dim_out['Z']]//4, 
                    batch_out.shape[dim_out['Y']]//4, 
                    batch_out.shape[dim_out['X']]//4)
        if self.dim == 4:
            block = (4, 4, 4)
            grid = (batch_out.shape[dim_out['T']]//4, 
                    batch_out.shape[dim_out['Z']]//4, 
                    batch_out.shape[dim_out['Y']]//4)
        
        # build the spatial parameter list
        spatial_params_list = []
        for op, _, __ in self.config['spatial_ops']:
            if len(op.shape_params()):                
                shape = [batch_out.shape[dim_out['B']]]+op.shape_params()
                spatial_params_list.append(op.function(shape))

        # build the color parameter list
        color_params_list = []
        for op, _, __ in self.config['color_ops']:
            if len(op.shape_params()):
                shape = [batch_out.shape[dim_out['B']]]+op.shape_params()
                color_params_list.append(op.function(shape))
                
        # call the kernel and synchronize it
        numba.cuda.select_device(self.gpu)
        kernel[grid, block](batch_in, batch_out, empty_values, *spatial_params_list, *color_params_list)
        numba.cuda.synchronize()
        
        return batch_out