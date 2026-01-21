#!/usr/bin/env python
"""Batch test script for edge-blend comparison.

Processes all videos in test_assets/ with fast quality and different edge-blend values.
Runs tests in parallel and logs runtimes.
"""

from __future__ import annotations

import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Configuration
INPUT_DIR = Path("./test_assets")
OUTPUT_DIR = Path("./tests_output")
BLEND_VALUES = [0, 4, 8]
QUALITY = "fast"
MAX_WORKERS = 3  # Process 3 videos in parallel


def process_video(input_path: Path, output_path: Path, edge_blend: int) -> dict:
    """Process a single video and return timing info."""
    start_time = time.time()

    cmd = [
        "uv", "run", "watermark-remover", "remove",
        str(input_path),
        str(output_path),
        "--preset", "veo",
        "--quality", QUALITY,
        "--edge-blend", str(edge_blend),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start_time

    return {
        "input": input_path.name,
        "output": output_path.name,
        "edge_blend": edge_blend,
        "elapsed_seconds": round(elapsed, 2),
        "success": result.returncode == 0,
        "error": result.stderr if result.returncode != 0 else None,
    }


def main():
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Find all video files
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    videos = [f for f in INPUT_DIR.iterdir() if f.suffix.lower() in video_extensions]

    if not videos:
        print(f"No videos found in {INPUT_DIR}")
        sys.exit(1)

    print(f"Found {len(videos)} videos in {INPUT_DIR}")
    print(f"Testing edge-blend values: {BLEND_VALUES}")
    print(f"Quality preset: {QUALITY}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Parallel workers: {MAX_WORKERS}")
    print("-" * 60)

    # Build task list
    tasks = []
    for video in videos:
        stem = video.stem
        for blend in BLEND_VALUES:
            output_name = f"{stem}_blend{blend}.mp4"
            output_path = OUTPUT_DIR / output_name
            tasks.append((video, output_path, blend))

    print(f"Total tasks: {len(tasks)}")
    print("-" * 60)

    # Process in parallel
    results = []
    start_total = time.time()

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_video, inp, out, blend): (inp, blend)
            for inp, out, blend in tasks
        }

        for future in as_completed(futures):
            inp, blend = futures[future]
            try:
                result = future.result()
                results.append(result)
                status = "OK" if result["success"] else "FAILED"
                print(f"[{status}] {result['input']} blend={blend}: {result['elapsed_seconds']}s")
            except Exception as e:
                print(f"[ERROR] {inp.name} blend={blend}: {e}")
                results.append({
                    "input": inp.name,
                    "edge_blend": blend,
                    "success": False,
                    "error": str(e),
                })

    total_elapsed = time.time() - start_total

    # Summary
    print("-" * 60)
    print("SUMMARY")
    print("-" * 60)

    # Group by video
    by_video = {}
    for r in results:
        name = r["input"]
        if name not in by_video:
            by_video[name] = {}
        by_video[name][r["edge_blend"]] = r

    # Print table
    print(f"{'Video':<50} | blend=0 | blend=4 | blend=8")
    print("-" * 80)
    for video_name in sorted(by_video.keys()):
        times = by_video[video_name]
        t0 = times.get(0, {}).get("elapsed_seconds", "N/A")
        t4 = times.get(4, {}).get("elapsed_seconds", "N/A")
        t8 = times.get(8, {}).get("elapsed_seconds", "N/A")
        short_name = video_name[:47] + "..." if len(video_name) > 50 else video_name
        print(f"{short_name:<50} | {t0:>6}s | {t4:>6}s | {t8:>6}s")

    print("-" * 80)
    successful = sum(1 for r in results if r["success"])
    print(f"Completed: {successful}/{len(tasks)} tasks")
    print(f"Total wall time: {round(total_elapsed, 2)}s")
    print(f"Output files in: {OUTPUT_DIR.absolute()}")


if __name__ == "__main__":
    main()
