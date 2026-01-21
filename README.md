# WatermarkRemover-AI

**AI-Powered Watermark Removal Tool using Florence-2 and LaMA Models**

🇬🇧 English | 🇫🇷 Français | 🇨🇳 中文 | 🇯🇵 日本語 | 🇧🇷 Português | 🧠 Brainrot

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

---

## Overview

`WatermarkRemover-AI` is a cutting-edge application that leverages AI models for precise watermark detection and seamless removal. Perfect for removing watermarks from AI-generated videos like Sora, Sora 2, Runway, and others.

It uses Florence-2 from Microsoft for watermark identification and LaMA for inpainting to fill in the removed regions naturally. The software features a modern GUI built with PyWebview for an accessible and intuitive experience.

## Screenshot

![App Screenshot](assets/screenshot-preview.png)

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
- **Multi-Language UI** - Available in English, French, Chinese, Japanese, Portuguese, and more
- **Themes** - Multiple UI themes to choose from

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
git clone https://github.com/D-Ogi/WatermarkRemover-AI.git
cd WatermarkRemover-AI
./setup.sh      # Linux/macOS
.\setup.bat     # Windows
```

The setup wizard will:

- Install Python and dependencies via uv
- Download AI models (~1.7GB total)
- Offer to launch the app when complete

After setup, use `run.bat` (Windows) or `./run.sh` (Linux/macOS) to launch.

### Optional: FFmpeg

Install FFmpeg to preserve audio when processing videos:

- **Windows**: Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH
- **Linux**: `sudo apt install ffmpeg`
- **macOS**: `brew install ffmpeg`

---

## Usage

### GUI Mode

1. Run the app (`run.bat` on Windows, `./run.sh` on macOS/Linux)
2. Select your preferred language and theme from the top-right corner
3. Select your mode (Single File or Batch)
4. Set input and output paths
5. Configure settings as needed
6. Hit **Start Processing**

Your settings are automatically saved and restored on next launch.

### CLI Mode

```bash
# Basic usage
uv run watermark-remover remove input.png output_folder/

# With options
uv run watermark-remover remove ./images ./output --overwrite --max-bbox-percent=15 --force-format=PNG

# Process video with two-pass detection
uv run watermark-remover remove video.mp4 ./output --detection-skip=3 --fade-in=0.5 --fade-out=0.5

# Preview mode (detect without processing)
uv run watermark-remover remove input.png --preview

# Run setup wizard
uv run watermark-remover setup

# Launch GUI directly
uv run watermark-remover gui
```

### CLI Options

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
- **PyWebview** - Cross-platform webview wrapper
- **Alpine.js** - Lightweight JavaScript framework for UI
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

