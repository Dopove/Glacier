#!/bin/bash

# Activate the virtual environment
source .venv/bin/activate

# Ensure pip is up to date
pip install --upgrade pip

# Install PyTorch first (often helps with build dependency resolution for CUDA packages)
echo "Installing PyTorch..."
pip install torch

# Install dependencies for Mamba
echo "Installing causal_conv1d..."
export MAMBA_FORCE_BUILD=TRUE
pip install causal_conv1d

echo "Installing mamba_ssm..."
pip install mamba_ssm

echo "Installing other requirements..."
pip install transformers
pip install -r requirements.txt

echo "Installation complete."