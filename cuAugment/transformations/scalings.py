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

__all__ = ['ScaleX', 
           'ScaleY', 
           'ScaleZ', 
           'ScaleT', 
           'ScaleXY', 
           'ScaleXYZ', 
           'ScaleXYZT',
           'LinearX',
           'LinearXY',
           'LinearXYZ',
           'LinearXYZT']

class ScaleX(SpatialTransformation):
    
    def __init__(self, distribution):
        super().__init__(distribution)
    
    def inline(self):
        '''scale x'''
        
        return ['x /= {param}[0]']
    
    def shape_params(self):
        return [1]
    
    def min_dims(self):
        return 1

class ScaleY(SpatialTransformation):
    
    def __init__(self, distribution):
        super().__init__(distribution)
    
    def inline(self):
        '''scale y'''
        
        return ['y /= {param}[0]']
    
    def shape_params(self):
        return [1]
    
    def min_dims(self):
        return 2

class ScaleZ(SpatialTransformation):
    
    def __init__(self, distribution):
        super().__init__(distribution)
    
    def inline(self):
        '''scale z'''
        
        return ['z /= {param}[0]']
    
    def shape_params(self):
        return [1]
    
    def min_dims(self):
        return 3

class ScaleT(SpatialTransformation):
    
    def __init__(self, distribution):
        super().__init__(distribution)
    
    def inline(self):
        '''scale t'''
        
        return ['t /= {param}[0]']
    
    def shape_params(self):
        return [1]
    
    def min_dims(self):
        return 4

class ScaleXY(SpatialTransformation):
    
    def __init__(self, distribution):
        super().__init__(distribution)
    
    def inline(self):
        '''scale x and y with same factor'''
        
        return ['x /= {param}[0]',
                'y /= {param}[0]']
    
    def shape_params(self):
        return [1]
    
    def min_dims(self):
        return 2

class ScaleXYZ(SpatialTransformation):
    
    def __init__(self, distribution):
        super().__init__(distribution)
    
    def inline(self):
        '''scale x, y, and z with same factor'''
        
        return ['x /= {param}[0]',
                'y /= {param}[0]',
                'z /= {param}[0]']
    
    def shape_params(self):
        return [1]
    
    def min_dims(self):
        return 3

class ScaleXYZT(SpatialTransformation):
    
    def __init__(self, distribution):
        super().__init__(distribution)
    
    def inline(self):
        '''scale x, y, z, and t with same factor'''
        
        return ['x /= {param}[0]',
                'y /= {param}[0]',
                'z /= {param}[0]',
                't /= {param}[0]']
    
    def shape_params(self):
        return [1]
    
    def min_dims(self):
        return 4
    
class LinearXY(SpatialTransformation):
    
    def __init__(self, distribution):
        super().__init__(distribution)
    
    def inline(self):
        '''apply a generic linear transformation'''
        
        return ['x_, y_ = x, y',
                'x = {param}[0,0]*x_ + {param}[0,1]*y_',
                'y = {param}[1,0]*x_ + {param}[1,1]*y_']
    
    def shape_params(self):
        return [2, 2]
    
    def min_dims(self):
        return 2
    
LinearX = ScaleX
    
class LinearXY(SpatialTransformation):
    
    def __init__(self, distribution):
        super().__init__(distribution)
    
    def inline(self):
        '''apply a generic linear transformation'''
        
        return ['x_, y_ = x, y',
                'x = {param}[0,0]*x_ + {param}[0,1]*y_',
                'y = {param}[1,0]*x_ + {param}[1,1]*y_']
    
    def shape_params(self):
        return [2, 2]
    
    def min_dims(self):
        return 2
    
class LinearXYZ(SpatialTransformation):
    
    def __init__(self, distribution):
        super().__init__(distribution)
    
    def inline(self):
        '''apply a generic linear transformation'''
        
        return ['x_, y_, z_ = x, y, z',
                'x = {param}[0,0]*x_ + {param}[0,1]*y_ + {param}[0,2]*z_',
                'y = {param}[1,0]*x_ + {param}[1,1]*y_ + {param}[1,2]*z_',
                'z = {param}[2,0]*x_ + {param}[2,1]*y_ + {param}[2,2]*z_']
    
    def shape_params(self):
        return [3, 3]
    
    def min_dims(self):
        return 3
    
class LinearXYZT(SpatialTransformation):
    
    def __init__(self, distribution):
        super().__init__(distribution)
    
    def inline(self):
        '''apply a generic linear transformation'''
        
        return ['x_, y_, z_, t_ = x, y, z, t',
                'x = {param}[0,0]*x_ + {param}[0,1]*y_ + {param}[0,2]*z_ + {param}[0,3]*t_',
                'y = {param}[1,0]*x_ + {param}[1,1]*y_ + {param}[1,2]*z_ + {param}[1,3]*t_',
                'z = {param}[2,0]*x_ + {param}[2,1]*y_ + {param}[2,2]*z_ + {param}[2,3]*t_',
                't = {param}[3,0]*x_ + {param}[3,1]*y_ + {param}[3,2]*z_ + {param}[3,3]*t_']
    
    def shape_params(self):
        return [4, 4]
    
    def min_dims(self):
        return 4