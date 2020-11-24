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

__all__ = ['RotateXY', 
           'RotateYZ', 
           'RotateZX', 
           'RotateXYZ',
           'RotateXYZOverTime']

class RotateXY(SpatialTransformation):
    
    def __init__(self, distribution):
        super().__init__(distribution)
    
    def inline(self):
        '''rotate in x-y-plane around origin (0,0)'''

        return ['smallest = min(batch_out.shape[dimX_out], batch_out.shape[dimY_out])',
                'scale_x = batch_out.shape[dimX_out]/smallest',
                'scale_y = batch_out.shape[dimY_out]/smallest',
                'x, y = x*scale_x, y*scale_y',
                'cphi, sphi = math.cos({param}[0]), math.sin({param}[0])',
                'x, y = +cphi*x-sphi*y, +sphi*x+cphi*y',
                'x, y = x/scale_x, y/scale_y']
    
    def shape_params(self):
        return [1]
    
    def min_dims(self):
        return 2
    
class RotateYZ(SpatialTransformation):
    
    def __init__(self, distribution):
        super().__init__(distribution)
    
    def inline(self):
        '''rotate in y-z-plane around origin (0,0,0)'''

        return ['smallest = min(batch_out.shape[dimY_out], batch_out.shape[dimZ_out])',
                'scale_y = batch_out.shape[dimY_out]/smallest',
                'scale_z = batch_out.shape[dimZ_out]/smallest',                
                'y, z = y*scale_y, z*scale_z',
                'cphi, sphi = math.cos({param}[0]), math.sin({param}[0])',
                'y, z = +cphi*y-sphi*z, +sphi*y+cphi*z',
                'y, z = y/scale_y, z/scale_z']
    
    def shape_params(self):
        return [1]
    
    def min_dims(self):
        return 3

class RotateZX(SpatialTransformation):
    
    def __init__(self, distribution):
        super().__init__(distribution)
    
    def inline(self):
        '''rotate in z-x-plane around origin (0,0,0)'''

        return ['smallest = min(batch_out.shape[dimZ_out], batch_out.shape[dimX_out])',
                'scale_z = batch_out.shape[dimZ_out]/smallest',
                'scale_x = batch_out.shape[dimX_out]/smallest',                
                'z, x = z*scale_z, x*scale_x',
                'cphi, sphi = math.cos({param}[0]), math.sin({param}[0])',
                'z, x = +cphi*z+sphi*x, -sphi*z+cphi*x',
                'z, x = z/scale_z, x/scale_x']
    
    def shape_params(self):
        return [1]
    
    def min_dims(self):
        return 3

class RotateXYZ(SpatialTransformation):

    def __init__(self, distribution):
        super().__init__(distribution)
    
    def inline(self):
        '''rotate in random plane around origin (0,0,0)'''

        return ['smallest = min(batch_out.shape[dimX_out], batch_out.shape[dimY_out])',
                'smallest = min(smallest, batch_out.shape[dimZ_out])',
                'scale_x = batch_out.shape[dimX_out]/smallest',
                'scale_y = batch_out.shape[dimY_out]/smallest',
                'scale_z = batch_out.shape[dimZ_out]/smallest',
                'nx, ny, nz = {param}[0], {param}[1], {param}[2]',                
                'phi = math.sqrt(nx*nx+ny*ny+nz*nz)',
                'cphi, sphi = math.cos(phi), math.sin(phi)',
                'nx, ny, nz = nx/phi, ny/phi, nz/phi',
                'x_, y_, z_ = x*scale_x, y*scale_y, z*scale_z',
                'x += sphi*(-nz*y_+ny*z_)+(1-cphi)*(-(ny*ny+nz*nz)*x_+nx*ny*y_+nx*nz*z_)',
                'y += sphi*(+nz*x_-nx*z_)+(1-cphi)*(+ny*nx*x_-(nz*nz+nx*nx)*y_+ny*nz*z_)',
                'z += sphi*(-ny*x_+nx*y_)+(1-cphi)*(+nz*nx*x_+nz*ny*y_-(nx*nx+ny*ny)*z_)',
                'x, y, z = x/scale_x, y/scale_y, z/scale_z']
    
    def shape_params(self):
        return [3]
    
    def min_dims(self):
        return 3

class RotateXYZOverTime(SpatialTransformation):

    def __init__(self, distribution):
        super().__init__(distribution)
    
    def inline(self):
        '''rotate in random plane around origin (0,0,0) over time'''

        return ['smallest = min(batch_out.shape[dimX_out], batch_out.shape[dimY_out])',
                'smallest = min(smallest, batch_out.shape[dimZ_out])',
                'scale_x = batch_out.shape[dimX_out]/smallest',
                'scale_y = batch_out.shape[dimY_out]/smallest',
                'scale_z = batch_out.shape[dimZ_out]/smallest',
                'nx, ny, nz = {param}[0], {param}[1], {param}[2]',                
                'phi = math.sqrt(nx*nx+ny*ny+nz*nz)',
                'cphi, sphi = math.cos(phi+t*3.1415), math.sin(phi+t*3.1415)',
                'nx, ny, nz = nx/phi, ny/phi, nz/phi',
                'x_, y_, z_ = x*scale_x, y*scale_y, z*scale_z',
                'x += sphi*(-nz*y_+ny*z_)+(1-cphi)*(-(ny*ny+nz*nz)*x_+nx*ny*y_+nx*nz*z_)',
                'y += sphi*(+nz*x_-nx*z_)+(1-cphi)*(+ny*nx*x_-(nz*nz+nx*nx)*y_+ny*nz*z_)',
                'z += sphi*(-ny*x_+nx*y_)+(1-cphi)*(+nz*nx*x_+nz*ny*y_-(nx*nx+ny*ny)*z_)',
                'x, y, z = x/scale_x, y/scale_y, z/scale_z']
    
    def shape_params(self):
        return [3]
    
    def min_dims(self):
        return 4