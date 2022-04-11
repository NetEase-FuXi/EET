FROM nvcr.io/nvidia/pytorch:21.12-py3

LABEL maintainer="eet 2022"

COPY . /workspace/EET

RUN pip install transformers
RUN pip install fairseq

WORKDIR /workspace/EET

RUN pip install .

WORKDIR /workspace