FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y build-essential git wget python3 python3-pip libgl1-mesa-glx libglib2.0-0 unzip && \
    rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python

COPY modelbased-rl/BMPO/requirements.txt .
COPY modelbased-rl/MBPO/ED2-MBPO/environment/requirements.txt .
COPY modelbased-rl/MuZero/requirements.txt .

RUN pip3 install --no-cache-dir -r modelbased-rl/BMPO/requirements.txt
RUN pip3 install --no-cache-dir -r modelbased-rl/MBPO/ED2-MBPO/environment/requirements.txt
RUN pip3 install --no-cache-dir -r modelbased-rl/MuZero/requirements.txt

COPY . /workspace/

WORKDIR /workspace/