"""CLI commands for WatermarkRemover-AI."""

import click


@click.group()
def main():
    """WatermarkRemover-AI - Remove watermarks from images and videos using AI."""
    pass


@main.command()
@click.option("--china", is_flag=True, help="Use China mirrors (Tsinghua PyPI + HF-Mirror)")
def setup(china):
    """Run the setup wizard to install dependencies and download models."""
    from .setup_wizard import run_setup

    run_setup(china_mode=china)


@main.command()
def gui():
    """Launch the graphical user interface."""
    from .gui import main as gui_main

    gui_main()


@main.command("remove")
@click.argument("input_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path(), required=False, default=None)
@click.option("--preview", is_flag=True, help="Preview mode: detect watermarks and output JSON with base64 image (no processing).")
@click.option("--overwrite", is_flag=True, help="Overwrite existing files in bulk mode.")
@click.option("--transparent", is_flag=True, help="Make watermark regions transparent instead of removing.")
@click.option("--max-bbox-percent", default=10.0, help="Maximum percentage of the image that a bounding box can cover.")
@click.option("--force-format", type=click.Choice(["PNG", "WEBP", "JPG", "MP4", "AVI"], case_sensitive=False), default=None, help="Force output format. Defaults to input format.")
@click.option("--detection-prompt", default="watermark", help="Text prompt for watermark detection (e.g. 'watermark', 'watermark Sora logo', 'Getty Images').")
@click.option("--detection-skip", default=1, type=int, help="Detect watermarks every N frames for videos (1-10). Higher = faster but may miss brief watermarks.")
@click.option("--fade-in", default=0.0, type=float, help="Extend mask backwards by N seconds to handle fade-in watermarks.")
@click.option("--fade-out", default=0.0, type=float, help="Extend mask forwards by N seconds to handle fade-out watermarks.")
def remove_cmd(input_path, output_path, preview, overwrite, transparent, max_bbox_percent, force_format, detection_prompt, detection_skip, fade_in, fade_out):
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
    )


if __name__ == "__main__":
    main()
