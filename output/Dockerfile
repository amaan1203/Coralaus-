# Use an official NVIDIA CUDA base image to support GPU acceleration for TensorFlow
FROM nvidia/cuda:11.7.0-cudnn8-devel-ubuntu20.04

# Install system dependencies commonly needed for ML
RUN apt update && apt install -y build-essential git python3 python3-pip

# Create a symlink so python points to python3
RUN ln -s /usr/bin/python3 /usr/bin/python

# Set up a clean working directory
WORKDIR /app

# Copy all project files into the working directory
COPY . /app

# Install pip dependencies with pinned versions
# Fixing the "tensorflow not found" conflict by specifying the correct version
# and using the --find-links argument to find the correct CUDA-enabled wheels
RUN pip3 install --upgrade pip \
    && pip3 install \
    numpy==1.23.4 \
    pygame==2.1.2 \
    tensorflow==2.10.1 \
    opencv-python==4.6.0.66 \
    matplotlib==3.6.2 \
    Pillow==9.2.0

# Verify the installation
RUN python --version
RUN pip3 --version
RUN pip3 list tensorflow