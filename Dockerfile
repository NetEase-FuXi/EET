FROM nvcr.io/nvidia/pytorch:21.12-py3

LABEL maintainer="eet 2022"

COPY . /workspace/EET

RUN pip install transformers==4.22.0
RUN pip install fairseq==0.10.0

WORKDIR /workspace/EET

RUN pip install .

WORKDIR /workspace
