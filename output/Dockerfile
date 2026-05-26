# Use the official NVIDIA CUDA base image to support GPU acceleration
FROM nvidia/cuda:11.7.0-cudnn8-runtime-ubuntu20.04

# Set the working directory to /app
WORKDIR /app

# Install system dependencies commonly needed for ML
RUN apt update && apt install -y \
    build-essential \
    git \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Create a symlink so python points to python3
RUN ln -s /usr/bin/python3 /usr/bin/python

# Install pip dependencies
RUN pip3 install \
    torch==1.13.1+cu117 --find-links https://download.pytorch.org/whl/cu117 \
    torchvision==0.14.1+cu117 --find-links https://download.pytorch.org/whl/cu117 \
    pyyaml==6.0 \
    huggingface_hub==0.17.0 \
    safetensors==0.2.2 \
    numpy==1.23.4

# Copy all project files into the working directory
COPY . .

# Set the default command to None
# No CMD instruction as per the rules