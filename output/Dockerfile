FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential \
  git \
  wget \
  python3 \
  python3-pip \
  && rm -rf /var/lib/apt/lists/*

# Create python symlink
RUN ln -sf /usr/bin/python3 /usr/bin/python

# Set working directory
WORKDIR /workspace

# Copy requirements file
COPY requirements.txt .

# Install python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Pin packages to 2024 versions
RUN pip install torch==2.6.0 torchvision regex ftfy tqdm huggingface_hub safetensors timm

# Copy all source code
COPY . /workspace/

# Set environment variables
ENV PATH /usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Set working directory
WORKDIR /workspace