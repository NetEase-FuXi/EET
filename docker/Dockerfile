FROM nvcr.io/nvidia/cuda:11.0-cudnn8-devel-ubuntu18.04

LABEL maintainer="eet 2021"

WORKDIR /root/workspace

RUN apt-get update \
    && apt-get install -y \
       software-properties-common \
       cmake \
       vim 

RUN apt-get install -y \
    openssh-server

RUN apt-get install -y \
    python-dev \
    python3.7-dev 

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 1

RUN apt-get install -y python3-pip

RUN pip3 install --upgrade pip 
RUN pip3 install  torch===1.7.1+cu110 torchvision===0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html 
RUN pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple --no-cache-dir numpy

ENV PATH="/workspace/bin:${PATH}"

RUN ln -sf /usr/bin/python3.7 /usr/bin/python
RUN ln -sf /usr/bin/pip3.7 /usr/bin/pip

RUN pip install transformers==3.0.2
RUN pip installl fairseq==0.10.0

RUN git clone git@github.com:NetEase-FuXi/EET.git
WORKDIR EET
RUN pip install .