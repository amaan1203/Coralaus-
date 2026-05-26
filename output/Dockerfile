# Use the official NVIDIA CUDA base image to support GPU acceleration
FROM nvidia/cuda:11.0-cudnn7-devel-ubuntu18.04

# Install system dependencies commonly needed for ML
RUN apt update && apt install -y \
    build-essential \
    git \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set up a clean working directory
WORKDIR /app

# Copy all project files into the working directory
COPY . /app

# Install pip dependencies with pinned versions
RUN pip3 install --upgrade pip \
    && pip3 install \
    tensorflow==1.11.0

# No conflicts to fix in this case
# If there were conflicts, they would be addressed here with a comment explaining the fix

# Verify the installation
RUN python --version
RUN pip3 --version
RUN pip3 list tensorflow
