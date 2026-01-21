"""Setup wizard for WatermarkRemover-AI - replaces setup.ps1 and setup.sh."""

import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

# China mirror configuration
CHINA_PYPI_MIRROR = "https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple"
CHINA_HF_MIRROR = "https://hf-mirror.com"

# Model URLs
LAMA_MODEL_URL = "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt"
FLORENCE_MODEL_REPO = "florence-community/Florence-2-large"

# Fun facts and tips to show during installation
TIPS = [
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
]


class Colors:
    """ANSI color codes with Windows support."""

    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    MAGENTA = "\033[95m"
    GRAY = "\033[90m"
    WHITE = "\033[97m"
    RESET = "\033[0m"

    @classmethod
    def init(cls):
        """Enable ANSI colors on Windows."""
        if sys.platform == "win32":
            try:
                import ctypes

                kernel32 = ctypes.windll.kernel32
                kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
            except Exception:
                pass


def detect_platform() -> str:
    """Detect the current operating system."""
    if sys.platform == "win32":
        return "windows"
    elif sys.platform == "darwin":
        return "macos"
    else:
        return "linux"


def detect_china_locale() -> bool:
    """Auto-detect if user is likely in China based on timezone or locale."""
    try:
        # Check timezone
        if time.timezone == -28800 or time.altzone == -28800:  # UTC+8
            # Could be China, but also Singapore, HK, etc.
            # Check locale for more confidence
            import locale
            lang = locale.getdefaultlocale()[0] or ""
            if lang.startswith("zh_CN"):
                return True
    except Exception:
        pass
    return False


def detect_nvidia_gpu() -> bool:
    """Detect if an NVIDIA GPU is available."""
    if detect_platform() == "macos":
        return False

    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return False

    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, timeout=10)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False


def get_cache_dir() -> Path:
    """Get the torch hub checkpoints directory."""
    home = Path.home()
    return home / ".cache" / "torch" / "hub" / "checkpoints"


def print_header(text: str):
    """Print a styled header."""
    print()
    print(f"  {Colors.CYAN}============================================={Colors.RESET}")
    print(f"  {Colors.CYAN}   {text}{Colors.RESET}")
    print(f"  {Colors.CYAN}============================================={Colors.RESET}")
    print()


def print_ok(text: str):
    """Print an OK message."""
    print(f"  {Colors.GREEN}[OK]{Colors.RESET} {text}")


def print_error(text: str):
    """Print an error message."""
    print(f"  {Colors.RED}[X]{Colors.RESET} {text}")


def print_warning(text: str):
    """Print a warning message."""
    print(f"  {Colors.YELLOW}[!]{Colors.RESET} {text}")


def print_info(text: str):
    """Print an info message."""
    print(f"  {Colors.CYAN}[*]{Colors.RESET} {text}")


def show_rotating_tips(stop_event: threading.Event, tip_index: list):
    """Show rotating tips while a long operation runs."""
    while not stop_event.is_set():
        tip = TIPS[tip_index[0] % len(TIPS)]
        color = Colors.CYAN if tip["color"] == "cyan" else Colors.YELLOW
        line = f"      {color}{tip['icon']}{Colors.RESET} {tip['text']}"
        print(f"\r{line:<90}", end="", flush=True)
        tip_index[0] += 1
        for _ in range(50):  # 5 seconds in 100ms increments
            if stop_event.is_set():
                break
            time.sleep(0.1)
    print("\r" + " " * 90 + "\r", end="", flush=True)


def run_command_with_tips(args: list, env: dict = None, cwd: str = None) -> int:
    """Run a command while showing rotating tips."""
    stop_event = threading.Event()
    tip_index = [0]

    tip_thread = threading.Thread(target=show_rotating_tips, args=(stop_event, tip_index))
    tip_thread.start()

    try:
        full_env = os.environ.copy()
        if env:
            full_env.update(env)

        process = subprocess.Popen(
            args,
            env=full_env,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        process.wait()
        return process.returncode
    finally:
        stop_event.set()
        tip_thread.join()


def download_with_tips(url: str, dest: Path) -> bool:
    """Download a file while showing rotating tips."""
    stop_event = threading.Event()
    tip_index = [0]
    success = [False]

    def download():
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


def sync_dependencies(china_mode: bool, use_cpu: bool) -> bool:
    """Sync dependencies using uv."""
    env = {}
    if china_mode:
        env["UV_INDEX_URL"] = CHINA_PYPI_MIRROR

    if use_cpu:
        env["UV_EXTRA_INDEX_URL"] = "https://download.pytorch.org/whl/cpu"

    args = ["uv", "sync"]
    return run_command_with_tips(args, env=env) == 0


def install_iopaint(china_mode: bool) -> bool:
    """Install iopaint with --no-deps."""
    args = ["uv", "pip", "install", "iopaint", "--no-deps"]
    if china_mode:
        args.extend(["--index-url", CHINA_PYPI_MIRROR])

    result = subprocess.run(args, capture_output=True)
    return result.returncode == 0


def verify_installation() -> bool:
    """Verify all required imports work."""
    result = subprocess.run(
        ["uv", "run", "python", "-c", "import torch; import transformers; import webview; import cv2; print('OK')"],
        capture_output=True,
        text=True,
    )
    return "OK" in result.stdout


def download_lama_model() -> bool:
    """Download the LaMA inpainting model."""
    cache_dir = get_cache_dir()
    lama_file = cache_dir / "big-lama.pt"

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


def download_florence_model(china_mode: bool) -> bool:
    """Download the Florence-2 model via huggingface_hub."""
    print()
    print(f"      {Colors.GRAY}Did you know?{Colors.RESET}")
    print()

    env = {}
    if china_mode:
        env["HF_ENDPOINT"] = CHINA_HF_MIRROR
        print(f"      {Colors.GRAY}Using HF-Mirror for faster download in China{Colors.RESET}")

    script = f"""
import os
{'os.environ["HF_ENDPOINT"] = "' + CHINA_HF_MIRROR + '"' if china_mode else ''}
from huggingface_hub import snapshot_download
snapshot_download('{FLORENCE_MODEL_REPO}', local_dir_use_symlinks=False)
print('FLORENCE_OK')
"""

    args = ["uv", "run", "python", "-c", script]
    returncode = run_command_with_tips(args, env=env)

    if returncode == 0:
        print_ok("Florence-2 model ready")
        return True

    print_warning("Could not download Florence-2 model")
    print(f"      {Colors.YELLOW}It will be downloaded on first use{Colors.RESET}")
    return False


def set_console_title(title: str):
    """Set console window title (Windows only)."""
    if detect_platform() == "windows":
        try:
            import ctypes

            ctypes.windll.kernel32.SetConsoleTitleW(title)
        except Exception:
            pass


def run_setup(china_mode: bool = False):
    """Run the full setup wizard."""
    Colors.init()
    set_console_title("WatermarkRemover-AI Setup")

    print_header("WatermarkRemover-AI Setup")

    platform = detect_platform()
    print_info(f"Detected platform: {platform}")

    # Auto-detect China locale if not explicitly set
    if not china_mode:
        china_mode = detect_china_locale()

    if china_mode:
        print_ok("Using China mirrors (Tsinghua PyPI + HF-Mirror)")
    else:
        print_ok("Using default mirrors")
    print()

    # Detect GPU
    use_cpu = False
    if platform == "macos":
        print_info("macOS detected - using MPS acceleration")
    elif detect_nvidia_gpu():
        print_info("NVIDIA GPU detected - using CUDA acceleration")
    else:
        print_info("No NVIDIA GPU detected - using CPU version")
        use_cpu = True

    # Install dependencies
    print()
    print_info("Installing dependencies...")
    print(f"      {Colors.MAGENTA}This may take a few minutes. Chill and learn something!{Colors.RESET}")
    print()
    print(f"      {Colors.GRAY}Did you know?{Colors.RESET}")
    print()

    if not sync_dependencies(china_mode, use_cpu):
        print()
        print_error("Failed to install dependencies")
        sys.exit(1)

    print_ok("Dependencies installed")

    # Install iopaint
    print_info("Installing iopaint (no deps)...")
    if not install_iopaint(china_mode):
        print_error("Failed to install iopaint")
        sys.exit(1)
    print_ok("iopaint installed")

    # Verify installation
    print_info("Verifying installation...")
    if not verify_installation():
        print_error("Verification failed")
        sys.exit(1)
    print_ok("All dependencies verified")

    # Download models
    print()
    print_info("Downloading LaMA model (~196MB)...")
    download_lama_model()

    print()
    print_info("Downloading Florence-2 model (~1.5GB)...")
    download_florence_model(china_mode)

    # Success
    print_header("Setup complete! Ready to go!")

    print(f"  To run the app: {Colors.WHITE}uv run watermark-remover gui{Colors.RESET}")
    print()
    print(f"  {Colors.MAGENTA}Have fun yeeting watermarks!{Colors.RESET}")
    print()


if __name__ == "__main__":
    run_setup()
