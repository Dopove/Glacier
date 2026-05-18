FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Set non-interactive to avoid timezone prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.10 and necessary build tools
# Ubuntu 22.04 defaults to gcc-11, which is perfectly compatible with CUDA 12.1
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-venv \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Alias python3 to python
RUN ln -s /usr/bin/python3.10 /usr/bin/python

WORKDIR /app

# Upgrade pip and install core build dependencies
RUN python -m pip install --upgrade pip
RUN pip install packaging ninja wheel setuptools

# Install PyTorch mapped to CUDA 12.1
RUN pip install torch==2.3.0+cu121 --index-url https://download.pytorch.org/whl/cu121

# Copy requirements and install
COPY requirements.txt .
# Install everything else, ensuring we don't isolate the build so it finds PyTorch
RUN pip install --no-build-isolation -r requirements.txt

# Copy the rest of the application
COPY . .

# Set default command to run the benchmark script
CMD ["python", "tests/test_glacier.py"]
