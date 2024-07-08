FROM nvcr.io/nvidia/pytorch:23.04-py3
LABEL maintainer="eet 2.0"

RUN pip install transformers==4.31.0
WORKDIR /workspace/
RUN git clone https://github.com/NetEase-FuXi/EET.git
WORKDIR /workspace/EET/
RUN pip install .
WORKDIR /workspace
