"""Utility functions for WatermarkRemover-AI."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image, ImageDraw

if TYPE_CHECKING:
    pass

# Color palette for visualization
COLORMAP: tuple[str, ...] = (
    "blue",
    "orange",
    "green",
    "purple",
    "brown",
    "pink",
    "gray",
    "olive",
    "cyan",
    "red",
    "lime",
    "indigo",
    "violet",
    "aqua",
    "magenta",
    "coral",
    "gold",
    "tan",
    "skyblue",
)


def draw_polygons(
    image: Image.Image,
    prediction: dict[str, Any],
    fill_mask: bool = False,
) -> Image.Image:
    """Draw segmentation masks with polygons on an image.

    Args:
        image: PIL Image to draw on
        prediction: Dict with 'polygons' and 'labels' keys
        fill_mask: Whether to fill the polygons

    Returns:
        Image with polygons drawn
    """
    draw = ImageDraw.Draw(image)
    for polygons, label in zip(prediction["polygons"], prediction["labels"], strict=True):
        color = random.choice(COLORMAP)
        fill_color = random.choice(COLORMAP) if fill_mask else None

        for polygon in polygons:
            pts = np.array(polygon).reshape(-1, 2)
            if len(pts) < 3:
                continue

            flat_pts = pts.reshape(-1).tolist()
            draw.polygon(flat_pts, outline=color, fill=fill_color)
            draw.text((flat_pts[0] + 8, flat_pts[1] + 2), label, fill=color)

    return image


def draw_ocr_bboxes(
    image: Image.Image,
    prediction: dict[str, Any],
) -> Image.Image:
    """Draw OCR bounding boxes on an image.

    Args:
        image: PIL Image to draw on
        prediction: Dict with 'quad_boxes' and 'labels' keys

    Returns:
        Image with bounding boxes drawn
    """
    draw = ImageDraw.Draw(image)
    bboxes = prediction["quad_boxes"]
    labels = prediction["labels"]

    for box, label in zip(bboxes, labels, strict=True):
        color = random.choice(COLORMAP)
        box_pts = np.array(box).tolist()
        draw.polygon(box_pts, width=3, outline=color)
        draw.text((box_pts[0] + 8, box_pts[1] + 2), str(label), align="right", fill=color)

    return image


def convert_bbox_to_relative(box: list[float], image: Image.Image) -> list[float]:
    """Convert bounding box pixel coordinates to relative coordinates (0-999 range)."""
    return [
        (box[0] / image.width) * 999,
        (box[1] / image.height) * 999,
        (box[2] / image.width) * 999,
        (box[3] / image.height) * 999,
    ]


def convert_relative_to_bbox(relative: list[float], image: Image.Image) -> list[float]:
    """Convert relative coordinates (0-999 range) to pixel coordinates."""
    return [
        (relative[0] / 999) * image.width,
        (relative[1] / 999) * image.height,
        (relative[2] / 999) * image.width,
        (relative[3] / 999) * image.height,
    ]


def convert_bbox_to_loc(box: list[float], image: Image.Image) -> str:
    """Convert bounding box pixel coordinates to position tokens."""
    relative = convert_bbox_to_relative(box, image)
    return "".join(f"<loc_{int(coord)}>" for coord in relative)
