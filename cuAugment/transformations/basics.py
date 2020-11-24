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

from cuAugment.transformations.base import SpatialTransformation

__all__ = ['CenterX', 'UncenterX', 
           'CenterY', 'UncenterY',
           'CenterZ', 'UncenterZ',
           'CenterT', 'UncenterT',                     
           'CenterXY', 'UncenterXY',
           'CenterXYZ', 'UncenterXYZ',
           'CenterXYZT', 'UncenterXYZT']

class CenterX(SpatialTransformation):
    
    def __init__(self, distribution=None):
        super().__init__(distribution)
    
    def inline(self):
        '''center x'''
        
        return ['x = 2.0*(x-0.5)']
    
    def shape_params(self):
        return []
    
    def min_dims(self):
        return 1

class UncenterX(SpatialTransformation):
    
    def __init__(self, distribution=None):
        super().__init__(distribution)
    
    def inline(self):
        '''uncenter x'''
        
        return ['x = 0.5*x+0.5']
    
    def shape_params(self):
        return []
    
    def min_dims(self):
        return 1
    
class CenterY(SpatialTransformation):
    
    def __init__(self, distribution=None):
        super().__init__(distribution)
    
    def inline(self):
        '''center y'''
        
        return ['y = 2.0*(x-0.5)']
    
    def shape_params(self):
        return []
    
    def min_dims(self):
        return 2

class UncenterY(SpatialTransformation):
    
    def __init__(self, distribution=None):
        super().__init__(distribution)
    
    def inline(self):
        '''uncenter y'''
        
        return ['y = 0.5*x+0.5']
    
    def shape_params(self):
        return []
    
    def min_dims(self):
        return 2

class CenterZ(SpatialTransformation):
    
    def __init__(self, distribution=None):
        super().__init__(distribution)
    
    def inline(self):
        '''center z'''
        
        return ['z = 2.0*(x-0.5)']
    
    def shape_params(self):
        return []
    
    def min_dims(self):
        return 3

class UncenterZ(SpatialTransformation):
    
    def __init__(self, distribution=None):
        super().__init__(distribution)
    
    def inline(self):
        '''uncenter z'''
        
        return ['z = 0.5*x+0.5']
    
    def shape_params(self):
        return []
    
    def min_dims(self):
        return 3

class CenterT(SpatialTransformation):
    
    def __init__(self, distribution=None):
        super().__init__(distribution)
    
    def inline(self):
        '''center t'''
        
        return ['t = 2.0*(x-0.5)']
    
    def shape_params(self):
        return []
    
    def min_dims(self):
        return 4

class UncenterT(SpatialTransformation):
    
    def __init__(self, distribution=None):
        super().__init__(distribution)
    
    def inline(self):
        '''uncenter t'''
        
        return ['t = 0.5*x+0.5']
    
    def shape_params(self):
        return []
    
    def min_dims(self):
        return 4
    
class CenterXY(SpatialTransformation):
    
    def __init__(self, distribution=None):
        super().__init__(distribution)
    
    def inline(self):
        '''center xy'''
        
        return ['x, y = 2.0*(x-0.5), 2.0*(y-0.5)']
    
    def shape_params(self):
        return []
    
    def min_dims(self):
        return 2

class UncenterXY(SpatialTransformation):
    
    def __init__(self, distribution=None):
        super().__init__(distribution)
    
    def inline(self):
        '''uncenter xy'''
        
        return ['x, y = 0.5*x+0.5, 0.5*y+0.5']
    
    def shape_params(self):
        return []
    
    def min_dims(self):
        return 2

class CenterXYZ(SpatialTransformation):
    
    def __init__(self, distribution=None):
        super().__init__(distribution)
    
    def inline(self):
        '''center xyz'''
        
        return ['x, y, z = 2.0*(x-0.5), 2.0*(y-0.5), 2.0*(z-0.5)']
    
    def shape_params(self):
        return []
    
    def min_dims(self):
        return 3

class UncenterXYZ(SpatialTransformation):
    
    def __init__(self, distribution=None):
        super().__init__(distribution)
    
    def inline(self):
        '''uncenter xyz'''
        
        return ['x, y, z = 0.5*x+0.5, 0.5*y+0.5, 0.5*z+0.5']
    
    def shape_params(self):
        return []
    
    def min_dims(self):
        return 3

class CenterXYZT(SpatialTransformation):
    
    def __init__(self, distribution=None):
        super().__init__(distribution)
    
    def inline(self):
        '''center xyzt'''
        
        return ['x, y, z, t = 2.0*(x-0.5), 2.0*(y-0.5), 2.0*(z-0.5), 2.0*(t-0.5)']
    
    def shape_params(self):
        return []
    
    def min_dims(self):
        return 4

class UncenterXYZT(SpatialTransformation):
    
    def __init__(self, distribution=None):
        super().__init__(distribution)
    
    def inline(self):
        '''uncenter xyzt'''
        
        return ['x, y, z, t = 0.5*x+0.5, 0.5*y+0.5, 0.5*z+0.5, 0.5*t+0.5']
    
    def shape_params(self):
        return []
    
    def min_dims(self):
        return 4
