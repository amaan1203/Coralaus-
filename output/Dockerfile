# Use the official NVIDIA CUDA base image to support GPU acceleration
FROM nvidia/cuda:11.0-cudnn8-runtime-ubuntu20.04

# Set the working directory to /app
WORKDIR /app

# Install system dependencies commonly needed for ML
RUN apt update && apt install -y build-essential git python3 python3-pip

# Create a symlink so python points to python3
RUN ln -s /usr/bin/python3 /usr/bin/python

# Install pip dependencies
# The specified version of torch (1.7.1+cu101) is not available, 
# so we will use the latest version compatible with CUDA 11.0
RUN pip3 install --find-links https://download.pytorch.org/whl/torch_stable.html torch==1.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
# Install other dependencies with pinned versions
RUN pip3 install transformers==3.3.1 spacy==3.4.4 tqdm==4.64.0 tensorboard==2.9.0 progress==1.6

# Copy all project files into the working directory
COPY . .

# Set the environment variable for the Python path
ENV PYTHONPATH /app

# Expose the port for tensorboard
EXPOSE 6006