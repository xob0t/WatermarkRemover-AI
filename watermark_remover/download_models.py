"""Download AI models for WatermarkRemover-AI."""

from __future__ import annotations

import sys
import threading
import time
import urllib.request
from pathlib import Path

# Model URLs
LAMA_MODEL_URL = "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt"
FLORENCE_MODEL_REPO = "florence-community/Florence-2-large"

# Fun facts and tips to show during download
TIPS: tuple[dict[str, str], ...] = (
    {"icon": "[i]", "color": "cyan", "text": "Florence-2 can detect watermarks in any language - even emojis!"},
    {"icon": "[?]", "color": "yellow", "text": "Tip: Use 'Transparent mode' to keep the original background visible"},
    {"icon": "[i]", "color": "cyan", "text": "The AI model was trained on millions of images to understand context"},
    {"icon": "[?]", "color": "yellow", "text": "Tip: Lower 'Max detection size' if the AI removes too much"},
    {"icon": "[i]", "color": "cyan", "text": "LaMA stands for 'Large Mask inpainting' - it fills gaps naturally"},
    {"icon": "[?]", "color": "yellow", "text": "Tip: GPU processing is 10-50x faster than CPU"},
    {"icon": "[i]", "color": "cyan", "text": "This tool works on both images AND videos!"},
    {"icon": "[?]", "color": "yellow", "text": "Tip: Batch mode can process entire folders at once"},
    {"icon": "[i]", "color": "cyan", "text": "The AI analyzes each frame independently for best results"},
    {"icon": "[?]", "color": "yellow", "text": "Tip: PNG format preserves quality, JPG saves space"},
    {"icon": "[i]", "color": "cyan", "text": "Florence-2 is Microsoft's latest vision AI model"},
    {"icon": "[?]", "color": "yellow", "text": "Tip: Install FFmpeg to keep audio in processed videos"},
    {"icon": "[i]", "color": "cyan", "text": "The inpainting AI 'imagines' what should be behind the watermark"},
    {"icon": "[?]", "color": "yellow", "text": "Tip: Works best on watermarks that cover less than 10% of image"},
    {"icon": "[i]", "color": "cyan", "text": "Processing 4K video? Get some snacks, it takes a while"},
    {"icon": "[?]", "color": "yellow", "text": "Tip: Check the logs if something goes wrong"},
    {"icon": "[i]", "color": "cyan", "text": "The AI can handle semi-transparent watermarks too!"},
    {"icon": "[?]", "color": "yellow", "text": "Tip: Your settings are saved automatically between sessions"},
)


class Colors:
    """ANSI color codes with Windows support."""

    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    GRAY = "\033[90m"
    WHITE = "\033[97m"
    RESET = "\033[0m"

    @classmethod
    def init(cls) -> None:
        """Enable ANSI colors on Windows."""
        if sys.platform == "win32":
            try:
                import ctypes

                kernel32 = ctypes.windll.kernel32
                kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
            except Exception:
                pass


def get_cache_dir() -> Path:
    """Get the torch hub checkpoints directory."""
    return Path.home() / ".cache" / "torch" / "hub" / "checkpoints"


def print_header(text: str) -> None:
    """Print a styled header."""
    print()
    print(f"  {Colors.CYAN}============================================={Colors.RESET}")
    print(f"  {Colors.CYAN}   {text}{Colors.RESET}")
    print(f"  {Colors.CYAN}============================================={Colors.RESET}")
    print()


def print_ok(text: str) -> None:
    """Print a success message."""
    print(f"  {Colors.GREEN}[OK]{Colors.RESET} {text}")


def print_warning(text: str) -> None:
    """Print a warning message."""
    print(f"  {Colors.YELLOW}[!]{Colors.RESET} {text}")


def print_info(text: str) -> None:
    """Print an info message."""
    print(f"  {Colors.CYAN}[*]{Colors.RESET} {text}")


def show_rotating_tips(stop_event: threading.Event, tip_index: list[int]) -> None:
    """Show rotating tips while a long operation runs."""
    while not stop_event.is_set():
        tip = TIPS[tip_index[0] % len(TIPS)]
        color = Colors.CYAN if tip["color"] == "cyan" else Colors.YELLOW
        line = f"      {color}{tip['icon']}{Colors.RESET} {tip['text']}"
        print(f"\r{line:<90}", end="", flush=True)
        tip_index[0] += 1
        for _ in range(50):  # 5 seconds
            if stop_event.is_set():
                break
            time.sleep(0.1)
    print("\r" + " " * 90 + "\r", end="", flush=True)


def download_with_tips(url: str, dest: Path) -> bool:
    """Download a file while showing rotating tips."""
    stop_event = threading.Event()
    tip_index = [0]
    success = [False]

    def download() -> None:
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(url, dest)
            success[0] = True
        except Exception:
            success[0] = False
        finally:
            stop_event.set()

    tip_thread = threading.Thread(target=show_rotating_tips, args=(stop_event, tip_index))
    download_thread = threading.Thread(target=download)

    tip_thread.start()
    download_thread.start()

    download_thread.join()
    tip_thread.join()

    return success[0]


def download_lama_model() -> bool:
    """Download the LaMA inpainting model."""
    lama_file = get_cache_dir() / "big-lama.pt"

    if lama_file.exists():
        print_ok("LaMA model already exists")
        return True

    print()
    print(f"      {Colors.GRAY}Did you know?{Colors.RESET}")
    print()

    if download_with_tips(LAMA_MODEL_URL, lama_file):
        if lama_file.exists():
            print_ok("LaMA model ready")
            return True

    print_warning("Could not download LaMA model")
    print(f"      {Colors.YELLOW}It will be downloaded on first use{Colors.RESET}")
    return False


def download_florence_model() -> bool:
    """Download the Florence-2 model via huggingface_hub."""
    print()
    print(f"      {Colors.GRAY}Did you know?{Colors.RESET}")
    print()

    stop_event = threading.Event()
    tip_index = [0]
    success = [False]

    def download() -> None:
        try:
            from huggingface_hub import snapshot_download

            snapshot_download(FLORENCE_MODEL_REPO, local_dir_use_symlinks=False)
            success[0] = True
        except Exception:
            success[0] = False
        finally:
            stop_event.set()

    tip_thread = threading.Thread(target=show_rotating_tips, args=(stop_event, tip_index))
    download_thread = threading.Thread(target=download)

    tip_thread.start()
    download_thread.start()

    download_thread.join()
    tip_thread.join()

    if success[0]:
        print_ok("Florence-2 model ready")
        return True

    print_warning("Could not download Florence-2 model")
    print(f"      {Colors.YELLOW}It will be downloaded on first use{Colors.RESET}")
    return False


def run() -> None:
    """Download AI models for WatermarkRemover-AI."""
    Colors.init()

    print_header("Downloading AI Models")

    print_info("Downloading LaMA model (~196MB)...")
    download_lama_model()

    print()
    print_info("Downloading Florence-2 model (~1.5GB)...")
    download_florence_model()

    print_header("Download complete!")
    print()


if __name__ == "__main__":
    run()
