FROM nvcr.io/nvidia/pytorch:21.03-py3

# install NCCL and Horovod
# you need to download the nccl repo from the NVIDIA devzone: https://developer.nvidia.com/nccl 

#ADD nccl-local-repo-ubuntu2004-2.9.6-cuda11.3_1.0-1_amd64.deb /tmp
#RUN dpkg -i /tmp/nccl-local-repo-ubuntu2004-2.9.6-cuda11.3_1.0-1_amd64.deb
#RUN apt update && apt install libnccl2 libnccl-dev -y 
#RUN MAKEFLAGS="-j1" HOROVOD_WITH_PYTORCH=1 HOROVOD_GPU_OPERATIONS=NCCL python -m pip install horovod 

RUN cd /tmp \
 && git clone --recursive -b v0.21.3 https://github.com/uber/horovod.git \
 && cd horovod \
 && export HOROVOD_GPU_ALLREDUCE=NCCL \
 && export HOROVOD_GPU_BROADCAST=NCCL \
 && export HOROVOD_NCCL_INCLUDE=/usr/include \
 && export HOROVOD_NCCL_LIB=/usr/lib/x86_64-linux-gnu \
 && export HOROVOD_NCCL_LINK=SHARED \
 && export HOROVOD_WITHOUT_TENSORFLOW=1 \
 && export HOROVOD_WITHOUT_MXNET=1 \
 && export HOROVOD_WITH_MPI=1 \
 && sed -i "s/avx_fma_flags =.*$/avx_fma_flags = ['-march=sandybridge', '-mtune=broadwell']/" setup.py \
 && ln -s /usr/local/cuda/lib64/stubs/libcuda.so ./libcuda.so.1 \
 && export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD \
 && python setup.py install \
 && python setup.py clean \
 && rm ./libcuda.so.1 \
 && rm -rf /tmp/horovod


# update numba, cupy

RUN pip install --upgrade pip
RUN conda update numba -y
RUN pip install cupy-cuda112

# tensorboardX, trtorch

RUN pip install tensorboardX
RUN pip install trtorch
RUN pip install cloudpickle

# set env variables

ENV OMPI_MCA_btl_vader_single_copy_mechanism=none

# clone the sub-projects of the nvaitc-toolkit

RUN cd /workspace && git clone -b toolkit --single-branch https://github.com/nvidia/nvaitc-toolkit.git toolkit
RUN cd /workspace && git clone -b cuaugment --single-branch https://github.com/nvidia/nvaitc-toolkit.git cuaugment
RUN pip install -e /workspace/cuaugment

# add the Dockerfile for documentation purposes

ADD Dockerfile /workspace/Dockerfile
