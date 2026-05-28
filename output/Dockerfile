# Use the official NVIDIA CUDA base image to support GPU acceleration for PyTorch
FROM nvidia/cuda:12.0-cudnn8-devel-ubuntu20.04

# Set the working directory to /app
WORKDIR /app

# Install system dependencies
RUN apt update && apt install -y \
    build-essential \
    git \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Create a symlink for python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Install PyTorch with CUDA support
RUN pip3 install --find-links https://download.pytorch.org/whl/cu120 \
    torch==2.6.0+cu120 \
    torchvision==0.15.1+cu120

# Install other dependencies
RUN pip3 install \
    regex==2022.10.31 \
    ftfy==6.1.1 \
    tqdm==4.64.1 \
    huggingface_hub==0.10.1 \
    safetensors==0.3.0 \
    timm==0.6.12

# Create a clean working directory and copy project files
COPY . /app/

# Set the working directory to /app
WORKDIR /app