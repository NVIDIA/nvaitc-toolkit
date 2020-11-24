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

__all__ = ['DeformX', 
           'DeformY', 
           'DeformZ', 
           'DeformT', 
           'DeformXY', 
           'DeformXYZ',
           'DeformXYZT']

class DeformX(SpatialTransformation):
    
    def __init__(self, distribution):
        super().__init__(distribution)
    
    def inline(self):
        '''locally deform x'''
        
        return ['x += {param}[0]*x + {param}[1]**2*x*x']
    
    def shape_params(self):
        return [2]
    
    def min_dims(self):
        return 1

class DeformY(SpatialTransformation):
    
    def __init__(self, distribution):
        super().__init__(distribution)
    
    def inline(self):
        '''locally deform y'''
        
        return ['y += {param}[0]*y + {param}[1]**2*y*y']
    
    def shape_params(self):
        return [2]
    
    def min_dims(self):
        return 2
    
class DeformZ(SpatialTransformation):
    
    def __init__(self, distribution):
        super().__init__(distribution)
    
    def inline(self):
        '''locally deform z'''
        
        return ['z += {param}[0]*z + {param}[1]**2*z*z']
    
    def shape_params(self):
        return [2]
    
    def min_dims(self):
        return 3
    
class DeformT(SpatialTransformation):
    
    def __init__(self, distribution):
        super().__init__(distribution)
    
    def inline(self):
        '''locally deform t'''
        
        return ['t += {param}[0]*t + {param}[1]**2*t*t']
    
    def shape_params(self):
        return [2]
    
    def min_dims(self):
        return 4
    
    
class DeformXY(SpatialTransformation):
    
    def __init__(self, distribution):
        super().__init__(distribution)
    
    def inline(self):
        '''locally deform x and y'''
        
        return ['x_, y_ = x, y',
                'x += {param}[0,0]*x_+{param}[0,1]*y_+{param}[0,2]**2*x_*y_+{param}[0,3]**2*x_*x_+{param}[0,4]*y_*y_',
                'y += {param}[1,0]*x_+{param}[1,1]*y_+{param}[1,2]**2*x_*y_+{param}[1,3]**2*x_*x_+{param}[1,4]*y_*y_',
               ]
    
    def shape_params(self):
        return [2, 5]
    
    def min_dims(self):
        return 2

class DeformXYZ(SpatialTransformation):
    
    def __init__(self, distribution):
        super().__init__(distribution)
    
    def inline(self):
        '''locally deform x, y and z'''
        
        return ['x_, y_, z_ = x, y, z',
                'x += {param}[0,0]*x_      +{param}[0,1]*y_      +{param}[0,2]*z_',
                'x += {param}[0,3]**2*x_*x_+{param}[0,4]**2*y_*y_+{param}[0,5]**2*z_*z_',
                'x += {param}[0,6]**2*x_*y_+{param}[0,7]**2*x_*z_+{param}[0,8]**2*y_*z_',
                'y += {param}[1,0]*x_      +{param}[1,1]*y_      +{param}[1,2]*z_',
                'y += {param}[1,3]**2*x_*x_+{param}[1,4]**2*y_*y_+{param}[1,5]**2*z_*z_',
                'y += {param}[1,6]**2*x_*y_+{param}[1,7]**2*x_*z_+{param}[1,8]**2*y_*z_',
                'z += {param}[2,0]*x_      +{param}[2,1]*y_      +{param}[2,2]*z_',
                'z += {param}[2,3]**2*x_*x_+{param}[2,4]**2*y_*y_+{param}[2,5]**2*z_*z_',
                'z += {param}[2,6]**2*x_*y_+{param}[2,7]**2*x_*z_+{param}[2,8]**2*y_*z_',
                ]
    
    def shape_params(self):
        return [3, 9]
    
    def min_dims(self):
        return 3
    
class DeformXYZT(SpatialTransformation):
    
    def __init__(self, distribution):
        super().__init__(distribution)
    
    def inline(self):
        '''locally deform x, y, z and t'''
        
        return ['x_, y_, z_, t_ = x, y, z, t',
                'x += {param}[0, 0]*x_      +{param}[0, 1]*y_',
                'x += {param}[0, 2]*z_      +{param}[0, 3]*t_',
                'x += {param}[0, 4]**2*x_*x_+{param}[0, 5]**2*y_*y_',
                'x += {param}[0, 6]**2*z_*z_+{param}[0, 7]**2*t_*t_',
                'x += {param}[0, 8]**2*x_*y_+{param}[0, 9]**2*x_*z_+{param}[0,10]**2*x_*t_',
                'x += {param}[0,11]**2*y_*z_+{param}[0,12]**2*y_*t_+{param}[0,13]**2*z_*t_',
                'y += {param}[1, 0]*x_      +{param}[1, 1]*y_',
                'y += {param}[1, 2]*z_      +{param}[1, 3]*t_',
                'y += {param}[1, 4]**2*x_*x_+{param}[1, 5]**2*y_*y_',
                'y += {param}[1, 6]**2*z_*z_+{param}[1, 7]**2*t_*t_',
                'y += {param}[1, 8]**2*x_*y_+{param}[1, 9]**2*x_*z_+{param}[1,10]**2*x_*t_',
                'y += {param}[1,11]**2*y_*z_+{param}[1,12]**2*y_*t_+{param}[1,13]**2*z_*t_',
                'z += {param}[2, 0]*x_      +{param}[2, 1]*y_',
                'z += {param}[2, 2]*z_      +{param}[2, 3]*t_',
                'z += {param}[2, 4]**2*x_*x_+{param}[2, 5]**2*y_*y_',
                'z += {param}[2, 6]**2*z_*z_+{param}[2, 7]**2*t_*t_',
                'z += {param}[2, 8]**2*x_*y_+{param}[2, 9]**2*x_*z_+{param}[2,10]**2*x_*t_',
                'z += {param}[2,11]**2*y_*z_+{param}[2,12]**2*y_*t_+{param}[2,13]**2*z_*t_',
                't += {param}[3, 0]*x_      +{param}[3, 1]*y_',
                't += {param}[3, 2]*z_      +{param}[3, 3]*t_',
                't += {param}[3, 4]**2*x_*x_+{param}[3, 5]**2*y_*y_',
                't += {param}[3, 6]**2*z_*z_+{param}[3, 7]**2*t_*t_',
                't += {param}[3, 8]**2*x_*y_+{param}[3, 9]**2*x_*z_+{param}[3,10]**2*x_*t_',
                't += {param}[3,11]**2*y_*z_+{param}[3,12]**2*y_*t_+{param}[3,13]**2*z_*t_',
                ]
    
    def shape_params(self):
        return [4, 14]
    
    def min_dims(self):
        return 3
  
    
