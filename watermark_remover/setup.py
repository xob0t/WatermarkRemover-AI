"""Setup utilities for WatermarkRemover-AI."""

import torch

from .download_models import run as download_models


def run_setup() -> None:
    """Download AI models and report system info."""
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    download_models()
