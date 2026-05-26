# Use the official nvidia/cuda base image to support CUDA dependencies
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

# Set the working directory to /app
WORKDIR /app

# Install system dependencies commonly needed for ML
RUN apt update && apt install -y build-essential git python3 python3-pip

# Create a symlink so python points to python3
RUN ln -s /usr/bin/python3 /usr/bin/python

# Install pip dependencies
RUN pip3 install --find-links https://download.pytorch.org/whl/cu118 \
    torch==2.2.0+cu118 \
    transformers==4.40.0 \
    datasets==2.18.0 \
    tqdm==4.66.0 \
    numpy==1.26.0 \
    pandas==2.2.0 \
    matplotlib==3.8.0

# Copy all project files into the working directory
COPY . /app/

# Set the environment variable for the Python runtime
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/