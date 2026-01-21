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
- **AI Video Ready** - Remove watermarks from Sora, Sora 2, Runway, Veo, and other AI-generated videos
- **Fixed Coordinates Mode** - Skip AI detection with user-provided watermark positions
- **Presets** - Built-in presets for known watermarks (e.g., Veo)
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

### Install

```bash
# CPU (default)
uv tool install git+https://github.com/D-Ogi/WatermarkRemover-AI.git

# With CUDA (NVIDIA GPU)
uv tool install "watermarkremover-ai[cuda] @ git+https://github.com/D-Ogi/WatermarkRemover-AI.git"
```

AI models (~1.7GB total) are downloaded automatically on first use.

### Optional: FFmpeg

Install FFmpeg to preserve audio when processing videos:

- **Windows**: Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH
- **Linux**: `sudo apt install ffmpeg`
- **macOS**: `brew install ffmpeg`

---

## Usage

```bash
# Basic usage (AI detection)
watermark-remover remove input.png output_folder/

# With options
watermark-remover remove ./images ./output --overwrite --max-bbox-percent=15 --force-format=PNG

# Process video with two-pass detection
watermark-remover remove video.mp4 ./output --detection-skip=3 --fade-in=0.5 --fade-out=0.5

# Preview mode (detect without processing)
watermark-remover remove input.png --preview
```

### Fixed Coordinates Mode

Skip AI detection entirely by providing watermark coordinates directly. This is faster and more reliable when you know the exact watermark position.

```bash
# Use a preset (e.g., Veo watermark)
watermark-remover remove video.mp4 output.mp4 --preset veo

# Provide pixel coordinates (single region)
watermark-remover remove video.mp4 output.mp4 --coords "[100, 50, 300, 100]"

# Multiple regions
watermark-remover remove video.mp4 output.mp4 --coords "[[100,50,300,100], [10,800,150,850]]"

# Percentage-based coordinates (works across resolutions)
watermark-remover remove video.mp4 output.mp4 --coords-percent "[80, 90, 100, 100]"

# Load coordinates from JSON file
watermark-remover remove video.mp4 output.mp4 --coords-file coords.json
```

Coordinates format: `[x1, y1, x2, y2]` where (x1, y1) is top-left and (x2, y2) is bottom-right.

```
(0,0) ────────────────────────► X (width)
  │
  │     (x1, y1) ┌─────────┐
  │              │WATERMARK│
  │              └─────────┘ (x2, y2)
  ▼
  Y (height)
```

### Finding Watermark Coordinates

**Option 1: Use Preview Mode**

Let the AI detect the watermark and show you the coordinates:

```bash
watermark-remover remove video.mp4 --preview
```

This outputs JSON with detected bboxes you can copy to `--coords`.

**Option 2: Use an Image Editor**

1. Take a screenshot of a frame with the watermark
2. Open in any image editor (Paint, GIMP, Photoshop)
3. Note the cursor position (shown in status bar) at top-left and bottom-right corners

**Option 3: Common Percentage Positions**

| Position            | `--coords-percent`      |
| ------------------- | ----------------------- |
| Bottom-right corner | `[85, 90, 100, 100]`    |
| Bottom-left corner  | `[0, 90, 15, 100]`      |
| Top-right corner    | `[85, 0, 100, 10]`      |
| Center bottom       | `[40, 90, 60, 100]`     |

### Creating Custom Mask Images

For watermarks with complex shapes (logos, text), you can create a precise mask image instead of using rectangular coordinates. This results in better quality since only the exact watermark pixels are inpainted.

**How to create a mask:**

1. Take a screenshot of a frame with the watermark visible
2. Open in an image editor (Paint, GIMP, Photoshop, etc.)
3. Create a new layer or image with **black background**
4. Paint **white** over the watermark area (the exact shape you want removed)
5. Save as PNG in grayscale

**Mask format:**
- **White (255)** = watermark area (will be removed)
- **Black (0)** = keep as-is
- Any resolution works - it will be resized to match your video

**Example using Paint:**
1. Open your screenshot
2. Use the eyedropper to select white color
3. Use brush/pencil to paint over the watermark
4. Select All, then copy
5. Create new image, paste, save as PNG

**Using your custom mask:**

Place your mask in `watermark_remover/masks/` and update the preset in `core.py`, or use it directly:

```python
# In core.py, add or modify a preset:
WATERMARK_PRESETS = {
    "my-watermark": {
        "description": "My custom watermark",
        "mask_image": "my-watermark.png",
        "coords_percent": [[90, 90, 100, 100]],  # Fallback if mask not found
    },
}
```

Then use: `watermark-remover remove video.mp4 output.mp4 --preset my-watermark`

### Options

| Option               | Description                                                 |
| -------------------- | ----------------------------------------------------------- |
| `--preset`           | Use predefined coordinates (e.g., `veo`). Bypasses AI.      |
| `--coords`           | Watermark bbox as JSON. Bypasses AI detection.              |
| `--coords-percent`   | Coordinates as percentages (0-100). Bypasses AI detection.  |
| `--coords-file`      | JSON file with coordinates. Bypasses AI detection.          |
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
- **Fixed coordinates mode:** Fastest option when watermark position is known
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

