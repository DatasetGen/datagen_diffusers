FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu20.04

# Set up the environment
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and essential tools
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch with CUDA 11.6
RUN pip install --upgrade pip

# IF YOU WANT TO RUN WITH CUDA
# RUN pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --index-url https://download.pytorch.org/whl/cu116
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN pip install diffusers["torch"] transformers

# Set the default command to bash
CMD ["/bin/bash"]
