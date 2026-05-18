#!/bin/bash
set -e

# --- 1. Setup Clean Environment ---
echo "--- Setting up clean Python 3.12 virtual environment in ./.venv_real ---"
python3 -m venv .venv_real
source .venv_real/bin/activate

# --- 2. Install All Dependencies ---
echo "--- Installing all dependencies (torch, transformers, build tools...) ---"
pip install --upgrade pip
# Install torch for CUDA 12.1 first
pip install torch==2.3.0+cu121 --index-url https://download.pytorch.org/whl/cu121
# Install build tools and other dependencies
pip install packaging ninja wheel setuptools
# Install remaining requirements
pip install -r requirements.txt

# --- 3. Verify Environment ---
echo "--- Verifying environment ---"
python -c "import torch; print(f'Torch version: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"
python -c "import transformers; print('transformers imported successfully')"

# --- 4. Run Real-Model Tests ---
echo "--- Running GLACIER benchmarks against REAL Mamba model ---"
export MAMBA_FORCE_REAL="true"

# Execute tests and save ALL output to a log file
{
  pytest test_glacier.py && 
  pytest test_agentic_features.py
} | tee real_test_results.txt

echo "--- REAL-MODEL BENCHMARK COMPLETE ---"
echo "Full, verifiable output saved to real_test_results.txt"
