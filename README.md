# WatermarkRemover-AI

**AI-Powered Watermark Removal Tool using Florence-2 and LaMA Models**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

---

## Overview

`WatermarkRemover-AI` is a CLI tool that leverages AI models for precise watermark detection and seamless removal. Perfect for removing watermarks from AI-generated videos like Sora, Sora 2, Runway, and others.

It uses Florence-2 from Microsoft for watermark identification and LaMA for inpainting to fill in the removed regions naturally.

## Demo

https://github.com/user-attachments/assets/505be2a8-8eda-4def-90b6-5a4ceefee456

---

## Features

- **Smart Detection** - AI-powered watermark detection using Florence-2
- **Seamless Removal** - LaMA inpainting for natural-looking results
- **Video Support** - Process videos with two-pass detection and audio preservation
- **AI Video Ready** - Remove watermarks from Sora, Sora 2, Runway, and other AI-generated videos
- **Batch Processing** - Handle entire folders at once
- **Preview Mode** - Preview detected watermarks before processing
- **Fade In/Out Handling** - Extend masks for watermarks that fade in/out
- **GPU Acceleration** - CUDA support for faster processing

---

## Installation

### Prerequisites

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) (fast Python package manager):

```bash
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Linux / macOS
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Setup

```bash
uv tool install git+https://github.com/D-Ogi/WatermarkRemover-AI.git
watermark-remover setup
```

The setup command downloads AI models (~1.7GB total). Dependencies are installed automatically by uv.

### Optional: FFmpeg

Install FFmpeg to preserve audio when processing videos:

- **Windows**: Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH
- **Linux**: `sudo apt install ffmpeg`
- **macOS**: `brew install ffmpeg`

---

## Usage

```bash
# Basic usage
watermark-remover remove input.png output_folder/

# With options
watermark-remover remove ./images ./output --overwrite --max-bbox-percent=15 --force-format=PNG

# Process video with two-pass detection
watermark-remover remove video.mp4 ./output --detection-skip=3 --fade-in=0.5 --fade-out=0.5

# Preview mode (detect without processing)
watermark-remover remove input.png --preview

# Run setup wizard
watermark-remover setup
```

### Options

| Option               | Description                                                 |
| -------------------- | ----------------------------------------------------------- |
| `--overwrite`        | Overwrite existing files                                    |
| `--transparent`      | Make watermark regions transparent (images only)            |
| `--max-bbox-percent` | Max detection size as % of image (default: 10)              |
| `--force-format`     | Force output format (PNG, WEBP, JPG, MP4, AVI)              |
| `--detection-prompt` | Custom detection prompt (default: "watermark")              |
| `--detection-skip`   | Detect every N frames for videos (1-10, default: 1)         |
| `--fade-in`          | Extend mask backwards by N seconds (for fade-in watermarks) |
| `--fade-out`         | Extend mask forwards by N seconds (for fade-out watermarks) |
| `--preview`          | Preview detected watermarks without processing              |

---

## Video Processing

- **Supported formats:** MP4, AVI, MOV, MKV, FLV, WMV, WEBM
- **Audio preservation:** Requires FFmpeg installed
- **Two-pass mode:** Faster processing with `--detection-skip` > 1
- **Fade handling:** Use `--fade-in` / `--fade-out` for watermarks that appear/disappear gradually

---

## Tech Stack

- **Florence-2** - Microsoft's vision model for watermark detection
- **LaMA** - Large Mask Inpainting model
- **PyTorch** - Deep learning backend

---

## Contributing

Contributions are welcome! Feel free to:

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

