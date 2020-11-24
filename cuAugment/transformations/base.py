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

import numpy as np
from cuAugment.distributions import BaseDistribution

class BaseTransformation:
    
    def __init__(self,  distribution=None):
        '''base transformation constructor'''
        
        if distribution == None:
            self.function = np.zeros            
        else:
            assert isinstance(distribution, BaseDistribution)
            self.function = distribution.sample_map()
    
    def has_params(self):
        '''base has_params'''
        return len(self.shape_params()) > 0
    
    def inline(self):
        '''base inline'''        
        raise NotImplementedError('You must override self.inline')

    def shape_params(self):
        '''base shape_params'''        
        raise NotImplementedError('You must override self.shape_params')

    def min_dims(self):
        '''base min_dims'''
        raise NotImplementedError('You must override self.min_dims')

class SpatialTransformation(BaseTransformation):
    
    def __init__(self, distribution):
        '''constructor'''
        
        super().__init__(distribution)
        
class ColorTransformation(BaseTransformation):
    
    def __init__(self, distribution):
        '''constructor'''
        
        super().__init__(distribution)
