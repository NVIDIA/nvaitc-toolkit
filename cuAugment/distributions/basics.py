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
from .base import BaseDistribution

__all__ = ['UniformIntDistribution',
           'UniformDistribution', 
           'NormalDistribution',
           'LogNormalDistribution',
           'DiracDistribution', 
           'BernoulliDistribution']

class UniformIntDistribution(BaseDistribution):
    
    def __init__(self, lower=0, upper=2**64):
        '''Uniform integer distribution constructor'''
        
        super().__init__()
        
        self.lower = lower
        self.upper = upper
        
    def sample_map(self):
        '''Uniform integer distribution sample map'''
        
        return lambda shape : np.random.randint(self.lower, self.upper, shape)

class UniformDistribution(BaseDistribution):
    
    def __init__(self, lower=-1.0, upper=+1.0):
        '''Uniform distribution constructor'''
        
        super().__init__()
        
        self.lower = lower
        self.upper = upper
        
    def sample_map(self):
        '''Uniform distribution sample map'''
        
        return lambda shape : np.random.uniform(self.lower, self.upper, shape)

class NormalDistribution(BaseDistribution):
    
    def __init__(self, avg=0.0, std=1.0):
        '''Normal distribution constructor'''
        
        super().__init__()
        
        self.avg = avg
        self.std = std
        
    def sample_map(self):
        '''Normal distribution sample map'''
        
        return lambda shape : np.random.normal(self.avg, self.std, shape)
    
class LogNormalDistribution(BaseDistribution):
    
    def __init__(self, avg=0.0, std=1.0):
        '''Log normal distribution constructor'''
        
        super().__init__()
        
        self.avg = avg
        self.std = std
        
    def sample_map(self):
        '''Log normal distribution sample map'''
        
        return lambda shape : np.random.lognormal(self.avg, self.std, shape)
    
    
class BernoulliDistribution(BaseDistribution):
    
    def __init__(self, probability):
        '''Bernoulli distribution constructor'''
        
        super().__init__()
        
        self.p = probability
        self.q = 1.0-probability
        
    def sample_map(self):
        '''Bernoulli distribution sample map'''
        
        return lambda shape : np.random.uniform(-self.q, +self.p, shape) > 0
    
class DiracDistribution(BaseDistribution):
    
    def __init__(self, array):
        '''Static distribution constructor'''
        
        super().__init__()
        
        if len(array.shape) == 1:
            array = np.expand_dims(array, 0)
        
        self.array = array
    
    def reshape(self, out_shape):
        '''reshape helper for cyclic stacking of data'''
        
        assert tuple(out_shape[1:]) == tuple(self.array.shape[1:]), 'input shape of static tensor incompatible '
        
        b_in, b_out = self.array.shape[0], out_shape[0]
        
        repeat = (b_out+b_in-1)//b_in
        out_data = np.vstack([self.array for _ in range(repeat)])
    
        return out_data[:b_out]
    
    def sample_map(self):
        '''Static distribution sample map'''
        
        return lambda shape : self.reshape(shape)
