"""Core watermark removal functionality."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict

import cv2

# Monkey-patch: cached_download was removed in huggingface_hub 0.24, add compatibility shim
import huggingface_hub
import numpy as np
import tqdm
from loguru import logger
from PIL import Image, ImageDraw

if not hasattr(huggingface_hub, "cached_download"):
    huggingface_hub.cached_download = huggingface_hub.hf_hub_download

import torch
from iopaint.model_manager import ModelManager
from iopaint.schema import HDStrategy, LDMSampler
from iopaint.schema import InpaintRequest as Config
from transformers import AutoProcessor, Florence2ForConditionalGeneration

if TYPE_CHECKING:
    from numpy.typing import NDArray


# Type definitions
class DetectionResult(TypedDict):
    """Result from watermark detection."""

    bbox: list[int]
    area_percent: float
    accepted: bool


class WatermarkPreset(TypedDict):
    """Predefined watermark coordinates."""

    description: str
    coords_percent: list[list[float]]


class PreviewResult(TypedDict):
    """Result from preview mode."""

    image: str
    detections: list[DetectionResult]
    source: str
    source_type: str
    source_frame: int | None
    prompt_used: str
    max_bbox_percent: float


# Predefined watermark coordinates (percentage-based for resolution independence)
# Format: coords_percent is [[x1%, y1%, x2%, y2%], ...] where values are 0-100
WATERMARK_PRESETS: dict[str, WatermarkPreset] = {
    "veo": {
        "description": "Google Veo watermark (bottom-right corner)",
        "coords_percent": [[93.5, 93.2, 98.7, 97.4]],
    },
}


def download_lama_model() -> bool:
    """Download LaMA model using iopaint."""
    logger.info("Downloading LaMA model... (this may take a few minutes)")
    print("Downloading LaMA model (~196MB)... Please wait.")

    result = subprocess.run(
        [sys.executable, "-m", "iopaint", "download", "--model", "lama"],
        capture_output=False,  # Show download progress
        text=True,
    )

    if result.returncode != 0:
        logger.error("Failed to download LaMA model")
        return False

    logger.info("LaMA model downloaded successfully")
    print("LaMA model downloaded!")
    return True


def load_lama_model(device: torch.device | str) -> ModelManager:
    """Load LaMA model, downloading if necessary."""
    try:
        return ModelManager(name="lama", device=device)
    except NotImplementedError as e:
        if "Unsupported model: lama" in str(e):
            print("LaMA model not available, attempting to download...")
            if download_lama_model():
                # Re-import to refresh model registry
                import importlib

                import iopaint.model

                importlib.reload(iopaint.model)
                # Try again
                return ModelManager(name="lama", device=device)
            else:
                raise RuntimeError(
                    "Failed to download LaMA model. Please run manually: python -m iopaint download --model lama"
                ) from e
        raise


class TaskType(str, Enum):
    """Supported Florence-2 task types."""

    OPEN_VOCAB_DETECTION = "<OPEN_VOCABULARY_DETECTION>"
    """Detect bounding box for objects and OCR text."""


def identify(
    task_prompt: TaskType,
    image: Image.Image,
    text_input: str | None,
    model: Florence2ForConditionalGeneration,
    processor: AutoProcessor,
    device: str,
) -> dict[str, Any]:
    """Run Florence-2 inference for object detection."""
    if not isinstance(task_prompt, TaskType):
        raise ValueError(f"task_prompt must be a TaskType, but {task_prompt} is of type {type(task_prompt)}")

    prompt = task_prompt.value if text_input is None else task_prompt.value + text_input
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        do_sample=False,
        num_beams=1,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    return processor.post_process_generation(
        generated_text, task=task_prompt.value, image_size=(image.width, image.height)
    )


def get_watermark_mask(
    image: Image.Image,
    model: Florence2ForConditionalGeneration,
    processor: AutoProcessor,
    device: str,
    max_bbox_percent: float,
    detection_prompt: str = "watermark",
) -> Image.Image:
    """Detect watermarks and create a mask for inpainting.

    Args:
        image: PIL Image to analyze
        model: Florence-2 model for detection
        processor: Florence-2 processor
        device: Device to run inference on (cuda or cpu)
        max_bbox_percent: Maximum bbox size as percentage of image area
        detection_prompt: Text prompt for detection

    Returns:
        Grayscale mask image (255=watermark region, 0=keep)
    """
    task_prompt = TaskType.OPEN_VOCAB_DETECTION
    parsed_answer = identify(task_prompt, image, detection_prompt, model, processor, device)

    mask = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(mask)

    detection_key = "<OPEN_VOCABULARY_DETECTION>"
    if detection_key in parsed_answer and "bboxes" in parsed_answer[detection_key]:
        image_area = image.width * image.height
        for bbox in parsed_answer[detection_key]["bboxes"]:
            x1, y1, x2, y2 = map(int, bbox)
            bbox_area = (x2 - x1) * (y2 - y1)
            if (bbox_area / image_area) * 100 <= max_bbox_percent:
                draw.rectangle([x1, y1, x2, y2], fill=255)
            else:
                logger.warning(
                    f"Skipping large bounding box: {bbox} covering {bbox_area / image_area:.2%} of the image"
                )

    return mask


def detect_only(
    image: Image.Image,
    model: Florence2ForConditionalGeneration,
    processor: AutoProcessor,
    device: str,
    max_bbox_percent: float,
    detection_prompt: str = "watermark",
) -> list[DetectionResult]:
    """Detect watermarks and return bounding boxes without inpainting.

    Used for preview mode to show what would be detected.

    Returns:
        List of detection results with bbox, area_percent, and accepted status.
    """
    task_prompt = TaskType.OPEN_VOCAB_DETECTION
    parsed_answer = identify(task_prompt, image, detection_prompt, model, processor, device)

    results: list[DetectionResult] = []
    detection_key = "<OPEN_VOCABULARY_DETECTION>"

    if detection_key in parsed_answer and "bboxes" in parsed_answer[detection_key]:
        image_area = image.width * image.height
        for bbox in parsed_answer[detection_key]["bboxes"]:
            x1, y1, x2, y2 = map(int, bbox)
            bbox_area = (x2 - x1) * (y2 - y1)
            area_percent = (bbox_area / image_area) * 100
            accepted = area_percent <= max_bbox_percent

            results.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "area_percent": round(area_percent, 2),
                    "accepted": accepted,
                }
            )

    return results


def process_image_with_lama(
    image: NDArray[np.uint8],
    mask: NDArray[np.uint8],
    model_manager: ModelManager,
) -> NDArray[np.uint8]:
    """Apply LaMA inpainting to remove watermarked regions."""
    config = Config(
        ldm_steps=50,
        ldm_sampler=LDMSampler.ddim,
        hd_strategy=HDStrategy.CROP,
        hd_strategy_crop_margin=64,
        hd_strategy_crop_trigger_size=800,
        hd_strategy_resize_limit=1600,
    )
    result = model_manager(image, mask, config)

    if result.dtype in [np.float64, np.float32]:
        result = np.clip(result, 0, 255).astype(np.uint8)

    return result


def make_region_transparent(image: Image.Image, mask: Image.Image) -> Image.Image:
    """Make watermark regions transparent using vectorized NumPy operations."""
    img_array = np.array(image.convert("RGBA"))
    mask_array = np.array(mask.convert("L"))
    img_array[:, :, 3] = np.where(mask_array > 0, 0, 255)
    return Image.fromarray(img_array, mode="RGBA")


VIDEO_EXTENSIONS = frozenset({".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm"})
IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp"})

# Output format dispatch tables
FORMAT_TO_PIL: dict[str, str] = {"JPG": "JPEG", "JPEG": "JPEG", "PNG": "PNG", "WEBP": "WEBP"}
FORMAT_SUPPORTS_TRANSPARENCY: frozenset[str] = frozenset({"PNG", "WEBP"})


def is_video_file(file_path: str | Path) -> bool:
    """Check if the file is a video based on its extension."""
    return Path(file_path).suffix.lower() in VIDEO_EXTENSIONS


def is_image_file(file_path: str | Path) -> bool:
    """Check if the file is a supported image based on its extension."""
    return Path(file_path).suffix.lower() in IMAGE_EXTENSIONS


def get_media_dimensions(file_path: Path) -> tuple[int, int]:
    """Get width and height of an image or video file."""
    if is_video_file(file_path):
        cap = cv2.VideoCapture(str(file_path))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
    else:
        with Image.open(file_path) as img:
            width, height = img.size
    return width, height


def resolve_output_format(
    input_path: Path,
    force_format: str | None,
    transparent: bool,
) -> str:
    """Determine output format for image processing.

    Returns PIL-compatible format string (e.g., "JPEG", "PNG", "WEBP").
    """
    if force_format:
        fmt = force_format.upper()
    elif transparent:
        return "PNG"
    else:
        fmt = input_path.suffix[1:].upper()

    # Normalize and validate
    pil_format = FORMAT_TO_PIL.get(fmt, "PNG")

    # JPEG doesn't support transparency
    if transparent and pil_format not in FORMAT_SUPPORTS_TRANSPARENCY:
        logger.warning(f"{pil_format} doesn't support transparency. Using PNG.")
        return "PNG"

    return pil_format


def collect_media_files(directory: Path) -> list[Path]:
    """Collect all supported image and video files from a directory."""
    images = list(directory.glob("*.[jp][pn]g")) + list(directory.glob("*.webp"))
    videos = (
        list(directory.glob("*.mp4"))
        + list(directory.glob("*.avi"))
        + list(directory.glob("*.mov"))
        + list(directory.glob("*.mkv"))
    )
    return images + videos


def ensure_video_extension(output_path: Path, force_format: str | None) -> Path:
    """Ensure video output has a valid video extension."""
    if output_path.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv"]:
        return output_path
    if force_format and force_format.upper() in ["MP4", "AVI"]:
        return output_path.with_suffix(f".{force_format.lower()}")
    return output_path.with_suffix(".mp4")


def has_audio_stream(file_path: str | Path) -> bool:
    """Check if a video file has an audio stream using ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "a",
                "-show_entries",
                "stream=codec_type",
                "-of",
                "csv=p=0",
                str(file_path),
            ],
            capture_output=True,
            text=True,
        )
        return "audio" in result.stdout
    except (subprocess.SubprocessError, FileNotFoundError):
        # ffprobe not available, assume audio exists to be safe
        return True


def parse_coords(
    coords: str | None = None,
    coords_file: str | None = None,
    coords_percent: str | None = None,
    preset: str | None = None,
    width: int | None = None,
    height: int | None = None,
) -> list[list[int]]:
    """Parse user-provided watermark coordinates into a list of bboxes.

    Args:
        coords: JSON string of absolute pixel coordinates
        coords_file: Path to JSON file with coordinates
        coords_percent: JSON string of percentage-based coordinates (0-100)
        preset: Name of a predefined preset (e.g., "veo")
        width: Video/image width (required for percentage conversion)
        height: Video/image height (required for percentage conversion)

    Returns:
        List of bboxes: [[x1, y1, x2, y2], ...]

    Raises:
        ValueError: If coordinates are invalid or required dimensions are missing.
    """
    bboxes: list[list[int]] = []
    percent_coords: list[list[float]] | None = None

    if preset:
        if preset.lower() not in WATERMARK_PRESETS:
            available = ", ".join(WATERMARK_PRESETS.keys())
            raise ValueError(f"Unknown preset: '{preset}'. Available presets: {available}")
        percent_coords = WATERMARK_PRESETS[preset.lower()]["coords_percent"]

    elif coords:
        try:
            parsed = json.loads(coords)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Invalid JSON in --coords: {e.msg}. Expected: [x1,y1,x2,y2] or [[x1,y1,x2,y2],...]"
            ) from e
        if not isinstance(parsed, list) or not parsed:
            raise ValueError("--coords must be a non-empty list")
        if isinstance(parsed[0], (int, float)):
            bboxes = [list(map(int, parsed))]
        else:
            bboxes = [list(map(int, b)) for b in parsed]

    elif coords_file:
        with open(coords_file) as f:
            data = json.load(f)
        if isinstance(data, list):
            if data and isinstance(data[0], (int, float)):
                bboxes = [list(map(int, data))]
            else:
                bboxes = [list(map(int, b)) for b in data]
        elif isinstance(data, dict) and "regions" in data:
            bboxes = [list(map(int, r["bbox"])) for r in data["regions"]]
        elif isinstance(data, dict) and "coords_percent" in data:
            percent_coords = data["coords_percent"]

    elif coords_percent:
        parsed = json.loads(coords_percent)
        if parsed and isinstance(parsed[0], (int, float)):
            percent_coords = [parsed]
        else:
            percent_coords = parsed

    if percent_coords:
        if width is None or height is None:
            raise ValueError("Width and height required for percentage-based coordinates")
        for pct in percent_coords:
            x1 = int(pct[0] / 100 * width)
            y1 = int(pct[1] / 100 * height)
            x2 = int(pct[2] / 100 * width)
            y2 = int(pct[3] / 100 * height)
            bboxes.append([x1, y1, x2, y2])

    return bboxes


def validate_bboxes(
    bboxes: list[list[int | float]],
    width: int,
    height: int,
    max_bbox_percent: float | None = None,
) -> list[list[int]]:
    """Validate and sanitize bounding boxes.

    Args:
        bboxes: List of [x1, y1, x2, y2] coordinates
        width: Image/video width
        height: Image/video height
        max_bbox_percent: Optional max size check (rejects if exceeded)

    Returns:
        List of validated bboxes clamped to image bounds
    """
    validated: list[list[int]] = []
    image_area = width * height

    for bbox in bboxes:
        if len(bbox) != 4:
            logger.warning(f"Bbox must have 4 values, got {len(bbox)}: {bbox}")
            continue

        x1, y1, x2, y2 = map(int, bbox)

        # Ensure proper ordering
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1

        # Clamp to image bounds
        x1 = max(0, min(x1, width))
        x2 = max(0, min(x2, width))
        y1 = max(0, min(y1, height))
        y2 = max(0, min(y2, height))

        # Check for zero-area bbox
        if x1 >= x2 or y1 >= y2:
            logger.warning(f"Skipping zero-area bbox: {bbox}")
            continue

        # Optional size check
        if max_bbox_percent is not None:
            bbox_area = (x2 - x1) * (y2 - y1)
            area_pct = (bbox_area / image_area) * 100
            if area_pct > max_bbox_percent:
                logger.warning(f"Bbox too large ({area_pct:.1f}% > {max_bbox_percent}%): {bbox}")
                continue

        validated.append([x1, y1, x2, y2])

    return validated


def create_mask_from_bboxes(bboxes: list[list[int]], width: int, height: int) -> Image.Image:
    """Create a binary mask from bounding boxes.

    Args:
        bboxes: List of [x1, y1, x2, y2] coordinates
        width: Image width
        height: Image height

    Returns:
        PIL Image in "L" mode (255=watermark region, 0=keep)
    """
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)

    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        draw.rectangle([x1, y1, x2, y2], fill=255)

    return mask


def inpaint_image(
    image: Image.Image,
    mask: Image.Image,
    model_manager: ModelManager | None,
    transparent: bool,
) -> Image.Image:
    """Apply watermark removal to an image using mask.

    Args:
        image: Input PIL Image (RGB)
        mask: Mask image (L mode, 255=watermark)
        model_manager: LaMA model (required if transparent=False)
        transparent: If True, make regions transparent; else inpaint

    Returns:
        Processed PIL Image
    """
    if transparent:
        return make_region_transparent(image, mask)

    lama_result = process_image_with_lama(np.array(image), np.array(mask), model_manager)
    return Image.fromarray(cv2.cvtColor(lama_result, cv2.COLOR_BGR2RGB))


def save_image(
    image: Image.Image,
    output_path: Path,
    output_format: str,
) -> Path:
    """Save image with correct format and extension.

    Returns the actual path used (with corrected extension if needed).
    """
    # Ensure extension matches format
    ext = "jpg" if output_format == "JPEG" else output_format.lower()
    final_path = output_path.with_suffix(f".{ext}")
    image.save(final_path, format=output_format)
    return final_path


def _get_video_fourcc(output_format: str) -> int:
    """Get OpenCV fourcc code for output format."""
    codecs = {"MP4": "mp4v", "AVI": "XVID"}
    return cv2.VideoWriter_fourcc(*codecs.get(output_format.upper(), "mp4v"))


def _resolve_video_output_path(
    input_path: Path,
    output_path: Path,
    force_format: str | None,
) -> tuple[Path, str]:
    """Determine output path and format for video processing.

    Returns:
        Tuple of (output_file_path, output_format)
    """
    output_format = (force_format or "MP4").upper()
    if output_path.is_dir():
        output_file = output_path / f"{input_path.stem}_no_watermark.{output_format.lower()}"
    else:
        output_file = output_path.with_suffix(f".{output_format.lower()}")
    return output_file, output_format


def _merge_audio_to_video(
    temp_video: Path,
    input_path: Path,
    output_file: Path,
) -> bool:
    """Merge original audio with processed video using FFmpeg.

    Returns:
        True if merge successful, False if fallback to copy was used.
    """
    # Check if FFmpeg is available
    try:
        subprocess.check_output(["ffmpeg", "-version"], stderr=subprocess.STDOUT)
    except (subprocess.SubprocessError, FileNotFoundError):
        logger.warning("FFmpeg not available. Video will be produced without audio.")
        shutil.copy(str(temp_video), str(output_file))
        return False

    if not has_audio_stream(input_path):
        logger.info("Original video has no audio. Copying video only.")
        shutil.copy(str(temp_video), str(output_file))
        return True

    logger.info("Merging processed video with original audio...")
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(temp_video),
        "-i",
        str(input_path),
        "-c:v",
        "copy",
        "-c:a",
        "copy",
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-shortest",
        str(output_file),
    ]
    try:
        subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
        logger.info("Audio/video merge completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error during audio/video merge: {e}")
        shutil.copy(str(temp_video), str(output_file))
        return False


def process_video(
    input_path: Path | str,
    output_path: Path | str,
    florence_model: Florence2ForConditionalGeneration,
    florence_processor: AutoProcessor,
    model_manager: ModelManager,
    device: str,
    transparent: bool,
    max_bbox_percent: float,
    force_format: str | None,
    detection_prompt: str = "watermark",
    progress_offset: int = 0,
    progress_scale: int = 100,
) -> Path | None:
    """Process a video file by extracting frames, removing watermarks, and reconstructing."""
    input_path = Path(input_path)
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        logger.error(f"Error opening video file: {input_path}")
        return None

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Resolve output path and format
    output_file, output_format = _resolve_video_output_path(input_path, Path(output_path), force_format)

    # Create a temporary file for the video without audio
    temp_dir = tempfile.mkdtemp()
    temp_video_path = Path(temp_dir) / f"temp_no_audio.{output_format.lower()}"

    fourcc = _get_video_fourcc(output_format)
    out = cv2.VideoWriter(str(temp_video_path), fourcc, fps, (width, height))

    # Process each frame
    with tqdm.tqdm(total=total_frames, desc="Processing video frames") as pbar:
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert frame to PIL Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)

            # Get watermark mask
            mask_image = get_watermark_mask(
                pil_image, florence_model, florence_processor, device, max_bbox_percent, detection_prompt
            )

            # Process frame
            if transparent:
                # For video, we can't use transparency, so we'll fill with a color or background
                result_image = make_region_transparent(pil_image, mask_image)
                # Convert RGBA to RGB by filling transparent areas with white
                background = Image.new("RGB", result_image.size, (255, 255, 255))
                background.paste(result_image, mask=result_image.split()[3])
                result_image = background
            else:
                lama_result = process_image_with_lama(np.array(pil_image), np.array(mask_image), model_manager)
                result_image = Image.fromarray(cv2.cvtColor(lama_result, cv2.COLOR_BGR2RGB))

            # Convert back to OpenCV format and write to output video
            frame_result = cv2.cvtColor(np.array(result_image), cv2.COLOR_RGB2BGR)
            out.write(frame_result)

            # Update progress
            frame_count += 1
            pbar.update(1)
            local_progress = frame_count / total_frames
            progress = int(progress_offset + local_progress * progress_scale)
            print(f"Processing frame {frame_count}/{total_frames}, overall_progress:{progress}%")

    # Release resources
    cap.release()
    out.release()

    # Merge audio and cleanup
    _merge_audio_to_video(temp_video_path, input_path, output_file)
    try:
        os.remove(str(temp_video_path))
        os.rmdir(temp_dir)
    except OSError:
        pass

    final_progress = progress_offset + progress_scale
    logger.info(f"input_path:{input_path}, output_path:{output_file}, overall_progress:{final_progress}")
    return output_file


def process_video_two_pass(
    input_path: Path | str,
    output_path: Path | str,
    florence_model: Florence2ForConditionalGeneration,
    florence_processor: AutoProcessor,
    model_manager: ModelManager,
    device: str,
    transparent: bool,
    max_bbox_percent: float,
    force_format: str | None,
    detection_prompt: str = "watermark",
    detection_skip: int = 1,
    fade_in_sec: float = 0.0,
    fade_out_sec: float = 0.0,
    progress_offset: int = 0,
    progress_scale: int = 100,
) -> Path | None:
    """Two-pass video processing with frame skip detection and fade handling.

    Pass 1: Detect watermarks every N frames (sparse detection)
    Pass 2: Apply inpainting to all frames using interpolated masks

    More efficient for videos with static watermarks and handles fade in/out
    by extending the mask temporally.
    """
    input_path = Path(input_path)
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        logger.error(f"Error opening video file: {input_path}")
        return None

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Convert seconds to frames
    fade_in_frames = int(fade_in_sec * fps)
    fade_out_frames = int(fade_out_sec * fps)

    logger.info(
        f"Two-pass processing: {total_frames} frames, skip={detection_skip}, fade_in={fade_in_frames}f, fade_out={fade_out_frames}f"
    )

    # Resolve output path and format
    output_file, output_format = _resolve_video_output_path(input_path, Path(output_path), force_format)

    # ========== PASS 1: DETECTION (sparse) ==========
    logger.info("Pass 1: Detecting watermarks...")
    detections = {}  # frame_idx -> [bbox, bbox, ...]
    detection_frames = list(range(0, total_frames, detection_skip))

    with tqdm.tqdm(total=len(detection_frames), desc="Pass 1: Detection") as pbar:
        for frame_idx in detection_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            bboxes = detect_only(
                pil_image, florence_model, florence_processor, device, max_bbox_percent, detection_prompt
            )

            if bboxes:
                accepted_bboxes = [b["bbox"] for b in bboxes if b["accepted"]]
                if accepted_bboxes:
                    detections[frame_idx] = accepted_bboxes

            pbar.update(1)
            local_progress = (pbar.n / len(detection_frames)) * 0.5  # Pass 1 = 0-50% local
            progress = int(progress_offset + local_progress * progress_scale)
            print(f"Pass 1: frame {frame_idx}/{total_frames}, overall_progress:{progress}%")

    logger.info(f"Pass 1 complete: found watermarks in {len(detections)} detection points")

    # ========== TIMELINE EXPANSION ==========
    # Create frame->bbox mapping with fade in/out expansion
    frame_masks = {}  # frame_idx -> [bbox, ...]

    for det_frame, bboxes in detections.items():
        # Expand backwards (fade in) - watermark might be fading in before detection
        start_frame = max(0, det_frame - fade_in_frames)
        # Expand forwards (fade out) - continue masking after detection
        # Also include frames until next detection point
        end_frame = min(total_frames, det_frame + detection_skip + fade_out_frames)

        for f in range(start_frame, end_frame):
            if f not in frame_masks:
                frame_masks[f] = []
            # Add bboxes, avoiding duplicates
            for bbox in bboxes:
                if bbox not in frame_masks[f]:
                    frame_masks[f].append(bbox)

    logger.info(f"Timeline expanded: {len(frame_masks)} frames will have inpainting applied")

    # ========== PASS 2: INPAINTING ==========
    logger.info("Pass 2: Applying inpainting...")

    # Create temporary file for video without audio
    temp_dir = tempfile.mkdtemp()
    temp_video_path = Path(temp_dir) / f"temp_no_audio.{output_format.lower()}"

    fourcc = _get_video_fourcc(output_format)
    out = cv2.VideoWriter(str(temp_video_path), fourcc, fps, (width, height))

    # Reset video to beginning
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    with tqdm.tqdm(total=total_frames, desc="Pass 2: Inpainting") as pbar:
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx in frame_masks:
                # This frame needs inpainting
                pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                # Create mask from bboxes
                mask = Image.new("L", pil_image.size, 0)
                draw = ImageDraw.Draw(mask)
                for bbox in frame_masks[frame_idx]:
                    x1, y1, x2, y2 = bbox
                    draw.rectangle([x1, y1, x2, y2], fill=255)

                # Apply inpainting or transparency
                if transparent:
                    result_image = make_region_transparent(pil_image, mask)
                    background = Image.new("RGB", result_image.size, (255, 255, 255))
                    background.paste(result_image, mask=result_image.split()[3])
                    result_image = background
                else:
                    lama_result = process_image_with_lama(np.array(pil_image), np.array(mask), model_manager)
                    result_image = Image.fromarray(cv2.cvtColor(lama_result, cv2.COLOR_BGR2RGB))

                frame_result = cv2.cvtColor(np.array(result_image), cv2.COLOR_RGB2BGR)
            else:
                # No watermark detected for this frame, copy original
                frame_result = frame

            out.write(frame_result)
            frame_idx += 1
            pbar.update(1)
            local_progress = 0.5 + (frame_idx / total_frames) * 0.5  # Pass 2 = 50-100% local
            progress = int(progress_offset + local_progress * progress_scale)
            print(f"Pass 2: frame {frame_idx}/{total_frames}, overall_progress:{progress}%")

    cap.release()
    out.release()

    # Merge audio and cleanup
    _merge_audio_to_video(temp_video_path, input_path, output_file)
    try:
        os.remove(str(temp_video_path))
        os.rmdir(temp_dir)
    except OSError:
        pass

    final_progress = progress_offset + progress_scale
    logger.info(f"input_path:{input_path}, output_path:{output_file}, overall_progress:{final_progress}")
    return output_file


def process_video_fixed_coords(
    input_path: Path | str,
    output_path: Path | str,
    bboxes: list[list[int]],
    model_manager: ModelManager | None,
    transparent: bool = False,
    force_format: str | None = None,
    progress_offset: int = 0,
    progress_scale: int = 100,
) -> Path | None:
    """Process video with fixed watermark coordinates (no AI detection).

    Most efficient mode for videos with static watermarks:
    - Creates mask ONCE from provided coordinates
    - Applies identical mask to ALL frames
    - No Florence-2 model needed
    - Pipes raw frames directly to FFmpeg when available

    Args:
        input_path: Input video path
        output_path: Output video/directory path
        bboxes: List of [x1,y1,x2,y2] watermark coordinates (already validated)
        model_manager: LaMA model (can be None if transparent=True)
        transparent: Replace watermark with white (transparency not supported in video)
        force_format: Output format (MP4, AVI)
        progress_offset: Starting progress percentage
        progress_scale: Progress range for this operation

    Returns:
        Path to output video file, or None on failure
    """
    input_path = Path(input_path)
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        logger.error(f"Error opening video file: {input_path}")
        return None

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logger.info(f"Processing video (fixed coords): {width}x{height}, {fps:.2f}fps, {total_frames} frames")
    logger.info(f"Watermark regions: {len(bboxes)} bbox(es)")

    # CREATE MASK ONCE - this is the key optimization
    mask = create_mask_from_bboxes(bboxes, width, height)
    mask_array = np.array(mask)

    logger.info(f"Created static mask covering {np.sum(mask_array > 0)} pixels")

    # Resolve output path and format
    output_file, output_format = _resolve_video_output_path(input_path, Path(output_path), force_format)

    # Check if FFmpeg is available for direct piping
    ffmpeg_available = False
    try:
        subprocess.check_output(["ffmpeg", "-version"], stderr=subprocess.STDOUT)
        ffmpeg_available = True
    except (subprocess.SubprocessError, FileNotFoundError):
        logger.warning("FFmpeg not available. Falling back to OpenCV (lower quality, no audio).")

    if ffmpeg_available:
        # Direct pipe to FFmpeg - single encode, best quality
        logger.info("Using FFmpeg pipe for direct H.264 encoding (single encode, best quality)")

        # Check if original has audio to include in output
        has_audio = has_audio_stream(input_path)
        if has_audio:
            logger.info("Original has audio - will copy to output")
        else:
            logger.info("Original has no audio - video only output")

        # FFmpeg command to read raw video frames from stdin and encode with H.264
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "-",
        ]
        if has_audio:
            ffmpeg_cmd.extend(["-i", str(input_path)])
        ffmpeg_cmd.extend(
            [
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
            ]
        )
        if has_audio:
            ffmpeg_cmd.extend(["-c:a", "copy", "-map", "0:v:0", "-map", "1:a:0", "-shortest"])
        ffmpeg_cmd.append(str(output_file))

        # Start FFmpeg process
        # Note: stderr must go to DEVNULL to prevent buffer deadlock
        # (FFmpeg writes lots of progress info to stderr which fills the pipe buffer)
        ffmpeg_proc = subprocess.Popen(
            ffmpeg_cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

        # Process all frames with the SAME mask
        with tqdm.tqdm(total=total_frames, desc="Processing frames (fixed coords)") as pbar:
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Convert frame to PIL Image (BGR -> RGB)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)

                # Apply inpainting or transparency
                if transparent:
                    result_image = make_region_transparent(pil_image, mask)
                    background = Image.new("RGB", result_image.size, (255, 255, 255))
                    background.paste(result_image, mask=result_image.split()[3])
                    result_image = background
                else:
                    lama_result = process_image_with_lama(np.array(pil_image), mask_array, model_manager)
                    result_image = Image.fromarray(cv2.cvtColor(lama_result, cv2.COLOR_BGR2RGB))

                # Convert back to BGR and write to FFmpeg stdin
                frame_result = cv2.cvtColor(np.array(result_image), cv2.COLOR_RGB2BGR)
                ffmpeg_proc.stdin.write(frame_result.tobytes())

                # Update progress
                frame_count += 1
                pbar.update(1)
                local_progress = frame_count / total_frames
                progress = int(progress_offset + local_progress * progress_scale)
                print(f"Processing frame {frame_count}/{total_frames}, overall_progress:{progress}%")

        # Close FFmpeg stdin and wait for completion
        cap.release()
        ffmpeg_proc.stdin.close()
        ffmpeg_proc.wait()

        if ffmpeg_proc.returncode != 0:
            logger.error(f"FFmpeg encoding failed with return code {ffmpeg_proc.returncode}")
            return None

        logger.info("Video encoding completed successfully!")

    else:
        # Fallback: use OpenCV VideoWriter (lower quality, no audio)
        temp_dir = tempfile.mkdtemp()
        temp_video_path = Path(temp_dir) / f"temp_no_audio.{output_format.lower()}"

        if output_format.upper() == "MP4":
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        elif output_format.upper() == "AVI":
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
        else:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        out = cv2.VideoWriter(str(temp_video_path), fourcc, fps, (width, height))

        with tqdm.tqdm(total=total_frames, desc="Processing frames (fixed coords)") as pbar:
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)

                if transparent:
                    result_image = make_region_transparent(pil_image, mask)
                    background = Image.new("RGB", result_image.size, (255, 255, 255))
                    background.paste(result_image, mask=result_image.split()[3])
                    result_image = background
                else:
                    lama_result = process_image_with_lama(np.array(pil_image), mask_array, model_manager)
                    result_image = Image.fromarray(cv2.cvtColor(lama_result, cv2.COLOR_BGR2RGB))

                frame_result = cv2.cvtColor(np.array(result_image), cv2.COLOR_RGB2BGR)
                out.write(frame_result)

                frame_count += 1
                pbar.update(1)
                local_progress = frame_count / total_frames
                progress = int(progress_offset + local_progress * progress_scale)
                print(f"Processing frame {frame_count}/{total_frames}, overall_progress:{progress}%")

        cap.release()
        out.release()
        shutil.copy(str(temp_video_path), str(output_file))

        try:
            os.remove(str(temp_video_path))
            os.rmdir(temp_dir)
        except OSError:
            pass

    final_progress = progress_offset + progress_scale
    logger.info(f"input_path:{input_path}, output_path:{output_file}, overall_progress:{final_progress}")
    return output_file


def check_overwrite_safety(input_path: Path, output_path: Path, overwrite: bool) -> bool:
    """Check if it's safe to write to output path.

    Returns True if safe to proceed, False if should skip.
    """
    if input_path.resolve() == output_path.resolve():
        logger.error(f"Cannot overwrite input file: {input_path}. Choose a different output path.")
        print("ERROR: Cannot overwrite input file! Choose a different output folder.")
        return False

    if output_path.exists() and not overwrite:
        logger.info(f"Skipping existing file: {output_path}")
        return False

    return True


def handle_one(
    image_path: Path,
    output_path: Path,
    florence_model: Florence2ForConditionalGeneration,
    florence_processor: AutoProcessor,
    model_manager: ModelManager | None,
    device: str,
    transparent: bool,
    max_bbox_percent: float,
    force_format: str | None,
    overwrite: bool,
    detection_prompt: str = "watermark",
    detection_skip: int = 1,
    fade_in: float = 0.0,
    fade_out: float = 0.0,
    progress_offset: int = 0,
    progress_scale: int = 100,
) -> Path | None:
    """Process a single image or video file with AI detection."""
    if not check_overwrite_safety(image_path, output_path, overwrite):
        return None

    # Dispatch: video vs image processing
    if is_video_file(image_path):
        return _handle_video_detection(
            image_path,
            output_path,
            florence_model,
            florence_processor,
            model_manager,
            device,
            transparent,
            max_bbox_percent,
            force_format,
            detection_prompt,
            detection_skip,
            fade_in,
            fade_out,
            progress_offset,
            progress_scale,
        )

    return _handle_image_detection(
        image_path,
        output_path,
        florence_model,
        florence_processor,
        model_manager,
        device,
        transparent,
        max_bbox_percent,
        force_format,
        detection_prompt,
        progress_offset,
        progress_scale,
    )


def _handle_video_detection(
    image_path: Path,
    output_path: Path,
    florence_model: Florence2ForConditionalGeneration,
    florence_processor: AutoProcessor,
    model_manager: ModelManager | None,
    device: str,
    transparent: bool,
    max_bbox_percent: float,
    force_format: str | None,
    detection_prompt: str,
    detection_skip: int,
    fade_in: float,
    fade_out: float,
    progress_offset: int,
    progress_scale: int,
) -> Path | None:
    """Process video with AI watermark detection."""
    use_two_pass = detection_skip > 1 or fade_in > 0 or fade_out > 0

    process_fn = process_video_two_pass if use_two_pass else process_video
    kwargs = {
        "input_path": image_path,
        "output_path": output_path,
        "florence_model": florence_model,
        "florence_processor": florence_processor,
        "model_manager": model_manager,
        "device": device,
        "transparent": transparent,
        "max_bbox_percent": max_bbox_percent,
        "force_format": force_format,
        "detection_prompt": detection_prompt,
        "progress_offset": progress_offset,
        "progress_scale": progress_scale,
    }

    if use_two_pass:
        kwargs["detection_skip"] = detection_skip
        kwargs["fade_in_sec"] = fade_in
        kwargs["fade_out_sec"] = fade_out

    return process_fn(**kwargs)


def _handle_image_detection(
    image_path: Path,
    output_path: Path,
    florence_model: Florence2ForConditionalGeneration,
    florence_processor: AutoProcessor,
    model_manager: ModelManager | None,
    device: str,
    transparent: bool,
    max_bbox_percent: float,
    force_format: str | None,
    detection_prompt: str,
    progress_offset: int,
    progress_scale: int,
) -> Path | None:
    """Process image with AI watermark detection."""
    image = Image.open(image_path).convert("RGB")
    mask = get_watermark_mask(image, florence_model, florence_processor, device, max_bbox_percent, detection_prompt)

    result_image = inpaint_image(image, mask, model_manager, transparent)
    output_format = resolve_output_format(image_path, force_format, transparent)
    final_path = save_image(result_image, output_path, output_format)

    final_progress = progress_offset + progress_scale
    print(f"input_path:{image_path}, output_path:{final_path}, overall_progress:{final_progress}%")
    return final_path


# =============================================================================
# PROCESSING MODE HANDLERS
# =============================================================================


def _process_fixed_coords_single(
    input_path: Path,
    output_path: Path,
    coords: str | None,
    coords_file: str | None,
    coords_percent: str | None,
    preset: str | None,
    model_manager: ModelManager | None,
    transparent: bool,
    max_bbox_percent: float,
    force_format: str | None,
    overwrite: bool,
) -> Path | None:
    """Process a single file with fixed coordinates."""
    width, height = get_media_dimensions(input_path)

    try:
        bboxes = parse_coords(coords, coords_file, coords_percent, preset, width, height)
    except Exception as e:
        logger.error(f"Failed to parse coordinates: {e}")
        return None

    bboxes = validate_bboxes(bboxes, width, height, max_bbox_percent)
    if not bboxes:
        logger.error("No valid coordinates provided")
        return None

    logger.info(f"Using {len(bboxes)} watermark region(s): {bboxes}")

    output_file = output_path / input_path.name if output_path.is_dir() else output_path

    if not check_overwrite_safety(input_path, output_file, overwrite):
        return None

    if is_video_file(input_path):
        output_file = ensure_video_extension(output_file, force_format)
        return process_video_fixed_coords(input_path, output_file, bboxes, model_manager, transparent, force_format)

    # Process image
    image = Image.open(input_path).convert("RGB")
    mask = create_mask_from_bboxes(bboxes, width, height)
    result_image = inpaint_image(image, mask, model_manager, transparent)

    output_format = resolve_output_format(input_path, force_format, transparent)
    output_file = save_image(result_image, output_file, output_format)

    print(f"input_path:{input_path}, output_path:{output_file}, overall_progress:100")
    return output_file


def _process_fixed_coords_batch(
    input_dir: Path,
    output_dir: Path,
    coords: str | None,
    coords_file: str | None,
    coords_percent: str | None,
    preset: str | None,
    model_manager: ModelManager | None,
    transparent: bool,
    max_bbox_percent: float,
    force_format: str | None,
) -> None:
    """Process multiple files with fixed coordinates."""
    output_dir.mkdir(parents=True, exist_ok=True)

    files = collect_media_files(input_dir)
    total_files = len(files)

    for idx, file_path in enumerate(tqdm.tqdm(files, desc="Processing files (fixed coords)")):
        width, height = get_media_dimensions(file_path)

        try:
            bboxes = parse_coords(coords, coords_file, coords_percent, preset, width, height)
        except Exception as e:
            logger.warning(f"Failed to parse coordinates for {file_path}: {e}")
            continue

        bboxes = validate_bboxes(bboxes, width, height, max_bbox_percent)
        if not bboxes:
            logger.warning(f"No valid coordinates for {file_path}, skipping")
            continue

        progress_offset = int(idx / total_files * 100)
        progress_scale = int(100 / total_files)
        output_file = output_dir / file_path.name

        if is_video_file(file_path):
            output_file = ensure_video_extension(output_file, force_format)
            process_video_fixed_coords(
                file_path,
                output_file,
                bboxes,
                model_manager,
                transparent,
                force_format,
                progress_offset,
                progress_scale,
            )
        else:
            image = Image.open(file_path).convert("RGB")
            mask = create_mask_from_bboxes(bboxes, width, height)
            result_image = inpaint_image(image, mask, model_manager, transparent)

            output_format = resolve_output_format(file_path, force_format, transparent)
            output_file = save_image(result_image, output_file, output_format)
            print(
                f"input_path:{file_path}, output_path:{output_file}, overall_progress:{progress_offset + progress_scale}%"
            )


def _process_fixed_coords(
    input_path: Path,
    output_path: Path,
    coords: str | None,
    coords_file: str | None,
    coords_percent: str | None,
    preset: str | None,
    transparent: bool,
    max_bbox_percent: float,
    force_format: str | None,
    overwrite: bool,
) -> Path | None:
    """Process files with fixed watermark coordinates (no AI detection)."""
    logger.info("Using fixed coordinates mode (no AI detection)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model_manager = load_lama_model(device) if not transparent else None
    if model_manager:
        logger.info("LaMa model loaded")

    if input_path.is_dir():
        _process_fixed_coords_batch(
            input_path,
            output_path,
            coords,
            coords_file,
            coords_percent,
            preset,
            model_manager,
            transparent,
            max_bbox_percent,
            force_format,
        )
        return None

    return _process_fixed_coords_single(
        input_path,
        output_path,
        coords,
        coords_file,
        coords_percent,
        preset,
        model_manager,
        transparent,
        max_bbox_percent,
        force_format,
        overwrite,
    )


def _process_preview(
    input_path: Path,
    max_bbox_percent: float,
    detection_prompt: str,
) -> None:
    """Run preview mode - detect watermarks and output JSON."""
    import base64
    import json as json_module
    import random
    from io import BytesIO

    device = "cuda" if torch.cuda.is_available() else "cpu"
    florence_model = (
        Florence2ForConditionalGeneration.from_pretrained("florence-community/Florence-2-large").to(device).eval()
    )
    florence_processor = AutoProcessor.from_pretrained("florence-community/Florence-2-large")

    # Get sample image
    if input_path.is_dir():
        files = collect_media_files(input_path)
        if not files:
            print(json_module.dumps({"error": "No supported files found in directory"}))
            return
        sample_path = random.choice(files)
    else:
        sample_path = input_path

    # Load image (extract frame if video)
    if is_video_file(sample_path):
        cap = cv2.VideoCapture(str(sample_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print(json_module.dumps({"error": f"Could not read frame from video: {sample_path}"}))
            return
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        source_type = "video"
        source_frame = total_frames // 2
    else:
        pil_image = Image.open(sample_path).convert("RGB")
        source_type = "image"
        source_frame = None

    # Run detection
    detections = detect_only(pil_image, florence_model, florence_processor, device, max_bbox_percent, detection_prompt)

    # Draw bounding boxes
    draw = ImageDraw.Draw(pil_image)
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        color = (0, 255, 0) if det["accepted"] else (255, 0, 0)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        draw.text((x1, y1 - 15), f"{det['area_percent']:.1f}%", fill=color)

    # Convert to base64
    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    result: PreviewResult = {
        "image": img_base64,
        "detections": detections,
        "source": str(sample_path),
        "source_type": source_type,
        "source_frame": source_frame,
        "prompt_used": detection_prompt,
        "max_bbox_percent": max_bbox_percent,
    }
    print(json_module.dumps(result))


def _process_detection(
    input_path: Path,
    output_path: Path,
    transparent: bool,
    max_bbox_percent: float,
    force_format: str | None,
    overwrite: bool,
    detection_prompt: str,
    detection_skip: int,
    fade_in: float,
    fade_out: float,
) -> Path | None:
    """Process files with AI watermark detection."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    florence_model = (
        Florence2ForConditionalGeneration.from_pretrained("florence-community/Florence-2-large").to(device).eval()
    )
    florence_processor = AutoProcessor.from_pretrained("florence-community/Florence-2-large")
    logger.info("Florence-2 Model loaded")

    model_manager = load_lama_model(device) if not transparent else None
    if model_manager:
        logger.info("LaMa model loaded")

    # Batch processing
    if input_path.is_dir():
        output_path.mkdir(parents=True, exist_ok=True)
        files = collect_media_files(input_path)
        total_files = len(files)

        for idx, file_path in enumerate(tqdm.tqdm(files, desc="Processing files")):
            progress_offset = int(idx / total_files * 100)
            progress_scale = int(100 / total_files)
            output_file = output_path / file_path.name

            handle_one(
                file_path,
                output_file,
                florence_model,
                florence_processor,
                model_manager,
                device,
                transparent,
                max_bbox_percent,
                force_format,
                overwrite,
                detection_prompt,
                detection_skip,
                fade_in,
                fade_out,
                progress_offset,
                progress_scale,
            )
        return None

    # Single file processing
    output_file = output_path / input_path.name if output_path.is_dir() else output_path

    if is_video_file(input_path):
        output_file = ensure_video_extension(output_file, force_format)

    result = handle_one(
        input_path,
        output_file,
        florence_model,
        florence_processor,
        model_manager,
        device,
        transparent,
        max_bbox_percent,
        force_format,
        overwrite,
        detection_prompt,
        detection_skip,
        fade_in,
        fade_out,
    )
    print(f"input_path:{input_path}, output_path:{output_file}, overall_progress:100")
    return result


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================


def _validate_inputs(
    detection_skip: int,
    fade_in: float,
    fade_out: float,
    preset: str | None,
    coords: str | None,
    coords_file: str | None,
    coords_percent: str | None,
    preview: bool,
) -> tuple[int, float, float, bool] | None:
    """Validate and normalize input parameters.

    Returns (detection_skip, fade_in, fade_out, use_fixed_coords) or None if validation fails.
    """
    # Clamp values
    if detection_skip < 1 or detection_skip > 10:
        logger.warning(f"detection_skip must be 1-10, got {detection_skip}. Using 1.")
        detection_skip = max(1, min(10, detection_skip))

    fade_in = max(0.0, fade_in)
    fade_out = max(0.0, fade_out)

    # Check fixed coords mode
    coord_options = [preset, coords, coords_file, coords_percent]
    use_fixed_coords = any(coord_options)

    # Validate mutually exclusive options
    if sum(1 for opt in coord_options if opt) > 1:
        logger.error(
            "Cannot use multiple coordinate options together. Choose one of: --preset, --coords, --coords-file, --coords-percent"
        )
        return None

    if use_fixed_coords and preview:
        logger.error("Preview mode is not supported with fixed coordinates")
        return None

    return detection_skip, fade_in, fade_out, use_fixed_coords


def process(
    input_path: str,
    output_path: str | None,
    preview: bool = False,
    overwrite: bool = False,
    transparent: bool = False,
    max_bbox_percent: float = 10.0,
    force_format: str | None = None,
    detection_prompt: str = "watermark",
    detection_skip: int = 1,
    fade_in: float = 0.0,
    fade_out: float = 0.0,
    preset: str | None = None,
    coords: str | None = None,
    coords_file: str | None = None,
    coords_percent: str | None = None,
) -> Path | None:
    """Main processing function - called by CLI.

    Dispatches to appropriate handler based on mode:
    - Fixed coordinates mode: Uses preset or user-provided coordinates
    - Preview mode: Detects watermarks and outputs JSON
    - Detection mode: Uses AI to detect and remove watermarks
    """
    validated = _validate_inputs(
        detection_skip, fade_in, fade_out, preset, coords, coords_file, coords_percent, preview
    )
    if validated is None:
        return None

    detection_skip, fade_in, fade_out, use_fixed_coords = validated
    input_path_obj = Path(input_path)
    output_path_obj = Path(output_path) if output_path else None

    # Dispatch to appropriate handler
    if use_fixed_coords:
        return _process_fixed_coords(
            input_path_obj,
            output_path_obj,
            coords,
            coords_file,
            coords_percent,
            preset,
            transparent,
            max_bbox_percent,
            force_format,
            overwrite,
        )

    if preview:
        _process_preview(input_path_obj, max_bbox_percent, detection_prompt)
        return None

    return _process_detection(
        input_path_obj,
        output_path_obj,
        transparent,
        max_bbox_percent,
        force_format,
        overwrite,
        detection_prompt,
        detection_skip,
        fade_in,
        fade_out,
    )
