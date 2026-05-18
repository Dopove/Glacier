# Getting Started with GLACIER

This guide provides a robust walkthrough for setting up and running the GLACIER project, ensuring you can replicate our benchmark results with real models.

## 1. System Requirements

- **Python:** Python 3.12 is recommended.
- **GPU:** An NVIDIA GPU with CUDA Toolkit 12.1+ installed is required for real model inference. 6GB of VRAM is sufficient for the default 130m parameter Mamba model.
- **Compiler:** A compatible C++ compiler (`g++-12` or newer) is necessary for building the CUDA extensions.

## 2. Environment Setup

To avoid dependency conflicts, it is crucial to work within a dedicated virtual environment.

```bash
# Navigate to the project directory
cd Glacier_Release_Staging

# Create a virtual environment using Python 3.12
python3.12 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate
```

## 3. Installation

The installation process for Mamba's dependencies can be complex. The following steps have been verified to work correctly.

```bash
# 1. Upgrade pip and install core build tools
pip install --upgrade pip
pip install wheel ninja packaging

# 2. Install PyTorch with CUDA support
# This example uses CUDA 12.1. Adjust if your system differs.
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 3. Install Mamba dependencies WITHOUT build isolation
# The --no-build-isolation flag is critical. It ensures the build process
# can find the PyTorch version you just installed.
pip install causal-conv1d mamba_ssm --no-build-isolation

# 4. Install remaining tokenizer and application dependencies
pip install transformers sentencepiece tiktoken onnxruntime httpx tqdm gradio Pillow PyMuPDF pytest pytest-asyncio
```

## 4. Download the Embedding Model

ICE-Lite's memory engine requires a local ONNX model for semantic search.

**Steps:**

1.  **Create the directory:**
    ```bash
    mkdir -p ~/.cache/ice/models/
    ```

2.  **Download the model and tokenizer files:**
    ```bash
    # Download the ONNX model (quantized for efficiency)
    wget https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/main/onnx/model_quantized.onnx -O ~/.cache/ice/models/model.onnx

    # Download the tokenizer configuration
    wget https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/main/tokenizer.json -O ~/.cache/ice/models/tokenizer.json
    ```

## 5. Run the Verification Suites

With the environment fully configured, you can now run the tests.

**A. Full Benchmark Suite (Core Memory Logic)**
This runs the "Needle in a Haystack" test and other core benchmarks.

```bash
python test_glacier.py
```
*Note: This will download the `state-spaces/mamba-130m` model from HuggingFace on its first run.*

**B. Agentic Features Suite**
This verifies tool-use memory, ingestion, and long-form generation.
```bash
pytest test_agentic_features.py
```

## 6. Run the Gradio Demo

Launch the interactive web demo to see GLACIER in action.
```bash
python app.py
```
This will start a local web server and provide a URL to access the demo in your browser.
