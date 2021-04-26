FROM nvcr.io/nvidia/cuda:11.0-cudnn8-devel-ubuntu18.04

LABEL maintainer="eet 2021"

# If you get an error executing apt-get update, you can try executing this command, and if it doesn't report an error, you can remove this line
RUN rm -rf  /etc/apt/sources.list.d/cuda.list

RUN apt-get update \
    && apt-get install -y \
       software-properties-common \
       cmake \
       vim \
       git 

RUN apt-get install -y \
    openssh-server

RUN apt-get install -y \
    python-dev \
    python3.7-dev 

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 1

RUN apt-get install -y python3-pip
RUN pip3 install --upgrade pip 
RUN ln -sf /usr/bin/python3.7 /usr/bin/python
RUN ln -sf /usr/bin/pip3.7 /usr/bin/pip

COPY . /workspace/EET

RUN pip install -r /workspace/EET/requirements.txt

WORKDIR /workspace
