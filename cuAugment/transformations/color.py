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

from cuAugment.transformations.base import ColorTransformation

__all__ = ['InvertColor', 
           'InvertColor01', 
           'InvertColor0255', 
           'AddColor', 
           'ScaleColor', 
           'ClipColor01', 
           'ClipColor0255']

class InvertColor(ColorTransformation):
    
    def __init__(self, distribution):
        super().__init__(distribution)
    
    def inline(self):
        '''invert color'''
        
        return ['v = -v if {param}[0] > 0 else v']
    
    def shape_params(self):
        return [1]
    
    def min_dims(self):
        return 0

class InvertColor01(ColorTransformation):
    
    def __init__(self, distribution):
        super().__init__(distribution)
    
    def inline(self):
        '''invert color'''
        
        return ['v = 1.0-v if {param}[0] > 0 else v']
    
    def shape_params(self):
        return [1]
    
    def min_dims(self):
        return 0
    
class InvertColor0255(ColorTransformation):
    
    def __init__(self, distribution):
        super().__init__(distribution)
    
    def inline(self):
        '''invert color'''
        
        return ['v = 255-v if {param}[0] > 0 else v']
    
    def shape_params(self):
        return [1]
    
    def min_dims(self):
        return 0

class AddColor(ColorTransformation):
    
    def __init__(self, distribution):
        super().__init__(distribution)
    
    def inline(self):
        '''add color'''
        
        return ['v += {param}[0]']
    
    def shape_params(self):
        return [1]
    
    def min_dims(self):
        return 0
    
class ScaleColor(ColorTransformation):
    
    def __init__(self, distribution):
        super().__init__(distribution)
    
    def inline(self):
        '''scale color'''
        
        return ['v *= {param}[0]']
    
    def shape_params(self):
        return [1]
    
    def min_dims(self):
        return 0
    
class ClipColor01(ColorTransformation):
    
    def __init__(self, distribution=None):
        super().__init__(distribution)
    
    def inline(self):
        '''clip color to [0, 1]'''
        
        return ['v = max(0.0, min(1.0, v))']
    
    def shape_params(self):
        return []
    
    def min_dims(self):
        return 0
    
class ClipColor0255(ColorTransformation):
    
    def __init__(self, distribution=None):
        super().__init__(distribution)
    
    def inline(self):
        '''clip color to [0, 255]'''
        
        return ['v = max(0.0, min(255.0, v))']
    
    def shape_params(self):
        return []
    
    def min_dims(self):
        return 0