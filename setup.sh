#!/usr/bin/env bash
set -e

echo ""
echo "  ============================================="
echo "     WatermarkRemover-AI Setup (Linux/macOS)"
echo "  ============================================="
echo ""

PYTHON_VERSION="3.12"

# China mirror configuration
CHINA_MODE=0
UV_INDEX_URL=""
HF_ENDPOINT=""

# Check if user is in China (for mirror selection)
echo "  [?] Are you in China? (y/n)"
echo "      This will use faster mirrors for downloads"
read -p "      " -n 1 -r china_choice
echo
if [[ $china_choice =~ ^[Yy]$ ]]; then
    CHINA_MODE=1
    UV_INDEX_URL="https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple"
    HF_ENDPOINT="https://hf-mirror.com"
    echo "  [OK] Using China mirrors (Tsinghua PyPI + HF-Mirror)"
else
    echo "  [OK] Using default mirrors"
fi
echo ""

# Detect OS
OS_TYPE="linux"
if [[ "$OSTYPE" == "darwin"* ]]; then
    OS_TYPE="macos"
    echo "  [*] Detected macOS"
else
    echo "  [*] Detected Linux"
fi

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "  [*] Installing uv package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Add to current session PATH
    export PATH="$HOME/.local/bin:$PATH"
    export PATH="$HOME/.cargo/bin:$PATH"

    if ! command -v uv &> /dev/null; then
        echo "  [X] Failed to install uv"
        echo "      Please install uv manually: https://docs.astral.sh/uv/getting-started/installation/"
        exit 1
    fi
    echo "  [OK] uv installed"
else
    echo "  [OK] uv found"
fi

# Install/verify Python
echo "  [*] Checking Python $PYTHON_VERSION..."
if ! uv python find $PYTHON_VERSION &> /dev/null; then
    echo "  [*] Installing Python $PYTHON_VERSION via uv..."
    uv python install $PYTHON_VERSION
fi
echo "  [OK] Python $PYTHON_VERSION ready"

# Install dependencies using uv sync
echo "  [*] Installing dependencies..."

# Set environment for China mirrors if enabled
if [ "$CHINA_MODE" == "1" ]; then
    export UV_INDEX_URL="$UV_INDEX_URL"
fi

# Detect GPU and set appropriate PyTorch index
if [ "$OS_TYPE" == "macos" ]; then
    echo "  [*] Installing for macOS (MPS support)..."
    # macOS uses default PyPI which includes MPS support
    UV_EXTRA_INDEX_URL="" uv sync --python $PYTHON_VERSION
elif command -v nvidia-smi &> /dev/null; then
    echo "  [*] NVIDIA GPU detected, using CUDA 12.4..."
    # Linux with NVIDIA: pyproject.toml already configured for CUDA
    uv sync --python $PYTHON_VERSION
else
    echo "  [*] No NVIDIA GPU detected, using CPU version..."
    # Linux without GPU: override to CPU index
    UV_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cpu" uv sync --python $PYTHON_VERSION
fi

echo "  [OK] Dependencies installed"

# Install iopaint separately with --no-deps
echo "  [*] Installing iopaint (no deps)..."
if [ "$CHINA_MODE" == "1" ]; then
    uv pip install iopaint --no-deps --index-url "$UV_INDEX_URL" -q
else
    uv pip install iopaint --no-deps -q
fi
echo "  [OK] iopaint installed"

# Verify installation
echo "  [*] Verifying installation..."
if ! uv run python -c "import torch; import transformers; import webview; import cv2; print('OK')" 2>/dev/null | grep -q "OK"; then
    echo "  [X] Verification failed"
    exit 1
fi
echo "  [OK] All dependencies verified"

# Download LaMA model
echo "  [*] Downloading LaMA model (~196MB)..."
LAMA_DIR="$HOME/.cache/torch/hub/checkpoints"
LAMA_FILE="$LAMA_DIR/big-lama.pt"
if [ ! -f "$LAMA_FILE" ]; then
    mkdir -p "$LAMA_DIR"
    curl -L -o "$LAMA_FILE" "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt" || echo "  [!] LaMA download failed, will retry on first use"
    echo "  [OK] LaMA model downloaded"
else
    echo "  [OK] LaMA model already exists"
fi

# Download Florence-2 model
echo "  [*] Downloading Florence-2 model (~1.5GB)..."
if [ "$CHINA_MODE" == "1" ]; then
    echo "      Using HF-Mirror for faster download in China"
    HF_ENDPOINT="$HF_ENDPOINT" uv run python -c "import os; os.environ['HF_ENDPOINT']='$HF_ENDPOINT'; from huggingface_hub import snapshot_download; snapshot_download('florence-community/Florence-2-large', local_dir_use_symlinks=False)" || echo "  [!] Florence-2 download failed, will retry on first use"
else
    uv run python -c "from huggingface_hub import snapshot_download; snapshot_download('florence-community/Florence-2-large', local_dir_use_symlinks=False)" || echo "  [!] Florence-2 download failed, will retry on first use"
fi
echo "  [OK] Florence-2 model ready"

echo ""
echo "  ============================================="
echo "     Setup complete!"
echo "  ============================================="
echo ""
echo "  To run the app:"
echo "    ./run.sh"
echo ""
echo "  Or directly:"
echo "    uv run python remwmgui.py"
echo ""
echo "  For CLI:"
echo "    uv run python remwm.py input.png output/"
echo ""

# Ask to launch
read -p "  Launch now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "  Starting WatermarkRemover-AI..."
    uv run python remwmgui.py
fi
