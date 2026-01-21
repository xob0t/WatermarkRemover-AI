"""CLI commands for WatermarkRemover-AI."""

from __future__ import annotations

import click


@click.group()
def main() -> None:
    """WatermarkRemover-AI - Remove watermarks from images and videos using AI."""


@main.command()
def setup() -> None:
    """Download AI models and configure PyTorch for GPU acceleration."""
    from .setup import run_setup

    run_setup()


@main.command("remove")
@click.argument("input_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path(), required=False, default=None)
@click.option("--preview", is_flag=True, help="Preview mode: detect watermarks and output JSON with base64 image.")
@click.option("--overwrite", is_flag=True, help="Overwrite existing files in bulk mode.")
@click.option("--transparent", is_flag=True, help="Make watermark regions transparent instead of removing.")
@click.option("--max-bbox-percent", default=10.0, help="Maximum percentage of the image a bounding box can cover.")
@click.option(
    "--force-format",
    type=click.Choice(["PNG", "WEBP", "JPG", "MP4", "AVI"], case_sensitive=False),
    default=None,
    help="Force output format. Defaults to input format.",
)
@click.option(
    "--detection-prompt",
    default="watermark",
    help="Text prompt for watermark detection (e.g. 'watermark', 'watermark Sora logo').",
)
@click.option(
    "--detection-skip",
    default=1,
    type=int,
    help="Detect watermarks every N frames for videos (1-10). Higher = faster.",
)
@click.option("--fade-in", default=0.0, type=float, help="Extend mask backwards by N seconds for fade-in watermarks.")
@click.option("--fade-out", default=0.0, type=float, help="Extend mask forwards by N seconds for fade-out watermarks.")
@click.option(
    "--preset",
    type=click.Choice(["veo"], case_sensitive=False),
    default=None,
    help="Use predefined watermark coordinates for known services. Bypasses AI detection.",
)
@click.option("--coords", default=None, type=str, help="Watermark bbox as JSON: [x1,y1,x2,y2]. Bypasses AI detection.")
@click.option("--coords-file", default=None, type=click.Path(exists=True), help="JSON file with watermark coordinates.")
@click.option("--coords-percent", default=None, type=str, help="Coordinates as percentages (0-100): [x1,y1,x2,y2].")
def remove_cmd(
    input_path: str,
    output_path: str | None,
    preview: bool,
    overwrite: bool,
    transparent: bool,
    max_bbox_percent: float,
    force_format: str | None,
    detection_prompt: str,
    detection_skip: int,
    fade_in: float,
    fade_out: float,
    preset: str | None,
    coords: str | None,
    coords_file: str | None,
    coords_percent: str | None,
) -> None:
    """Remove watermarks from images or videos."""
    from .core import process

    process(
        input_path=input_path,
        output_path=output_path,
        preview=preview,
        overwrite=overwrite,
        transparent=transparent,
        max_bbox_percent=max_bbox_percent,
        force_format=force_format,
        detection_prompt=detection_prompt,
        detection_skip=detection_skip,
        fade_in=fade_in,
        fade_out=fade_out,
        preset=preset,
        coords=coords,
        coords_file=coords_file,
        coords_percent=coords_percent,
    )


if __name__ == "__main__":
    main()
