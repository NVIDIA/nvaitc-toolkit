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

from numba import cuda

__all__ = ['cudaTimer']

class cudaTimer:

    def __init__(self, label='', gpu=0):
    
        
        self.label = label
        self.gpu = gpu
        self.start = cuda.event()
        self.end   = cuda.event()
        cuda.select_device(self.gpu)
        self.start.record(),
   
    def __enter__(self):
        pass
    
    def __exit__(self, *args):
    
        cuda.select_device(self.gpu)
        suffix = 'ms ('+self.label+')' if self.label else 'ms'
        self.end.record()
        self.end.synchronize()
        time = cuda.event_elapsed_time(self.start, self.end)
        print('STATUS: elapsed time', int(time), suffix)

