# WatermarkRemover-AI Setup Script (uv)
$Host.UI.RawUI.WindowTitle = "WatermarkRemover-AI Setup"

$PYTHON_VERSION = "3.12"

# China mirror configuration
$CHINA_MODE = $false
$UV_INDEX_URL = ""
$HF_ENDPOINT = ""

# Fun facts and tips to show during installation
$tips = @(
    @{icon="[i]"; color="Cyan"; text="Florence-2 can detect watermarks in any language - even emojis!"},
    @{icon="[?]"; color="Yellow"; text="Tip: Use 'Transparent mode' to keep the original background visible"},
    @{icon="[i]"; color="Cyan"; text="The AI model was trained on millions of images to understand context"},
    @{icon="[?]"; color="Yellow"; text="Tip: Lower 'Max detection size' if the AI removes too much"},
    @{icon="[i]"; color="Cyan"; text="LaMA stands for 'Large Mask inpainting' - it fills gaps naturally"},
    @{icon="[?]"; color="Yellow"; text="Tip: GPU processing is 10-50x faster than CPU"},
    @{icon="[i]"; color="Cyan"; text="This tool works on both images AND videos!"},
    @{icon="[?]"; color="Yellow"; text="Tip: Batch mode can process entire folders at once"},
    @{icon="[i]"; color="Cyan"; text="The AI analyzes each frame independently for best results"},
    @{icon="[?]"; color="Yellow"; text="Tip: PNG format preserves quality, JPG saves space"},
    @{icon="[i]"; color="Cyan"; text="Florence-2 is Microsoft's latest vision AI model"},
    @{icon="[?]"; color="Yellow"; text="Tip: Install FFmpeg to keep audio in processed videos"},
    @{icon="[i]"; color="Cyan"; text="The inpainting AI 'imagines' what should be behind the watermark"},
    @{icon="[?]"; color="Yellow"; text="Tip: Works best on watermarks that cover less than 10% of image"},
    @{icon="[i]"; color="Cyan"; text="Processing 4K video? Get some snacks, it takes a while"},
    @{icon="[?]"; color="Yellow"; text="Tip: Check the logs if something goes wrong"},
    @{icon="[i]"; color="Cyan"; text="The AI can handle semi-transparent watermarks too!"},
    @{icon="[?]"; color="Yellow"; text="Tip: Your settings are saved automatically between sessions"}
)

Write-Host ""
Write-Host "  =============================================" -ForegroundColor Cyan
Write-Host "     WatermarkRemover-AI Setup                 " -ForegroundColor Cyan
Write-Host "  =============================================" -ForegroundColor Cyan
Write-Host ""

# Check if user is in China (for mirror selection)
Write-Host "  [?] Are you in China? (y/n)" -ForegroundColor Yellow
Write-Host "      This will use faster mirrors for downloads" -ForegroundColor DarkGray
$chinaChoice = Read-Host "      "
if ($chinaChoice -eq "y" -or $chinaChoice -eq "Y") {
    $CHINA_MODE = $true
    $UV_INDEX_URL = "https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple"
    $HF_ENDPOINT = "https://hf-mirror.com"
    Write-Host "  [OK] Using China mirrors (Tsinghua PyPI + HF-Mirror)" -ForegroundColor Green
} else {
    Write-Host "  [OK] Using default mirrors" -ForegroundColor Green
}
Write-Host ""

# Check if uv is installed
$uvCmd = Get-Command uv -ErrorAction SilentlyContinue
if (-not $uvCmd) {
    Write-Host "  [*] Installing uv package manager..." -ForegroundColor Cyan
    try {
        Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression

        # Refresh PATH for current session
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")

        # Also check common install location
        $uvUserPath = "$env:USERPROFILE\.local\bin"
        if (Test-Path "$uvUserPath\uv.exe") {
            $env:Path = "$uvUserPath;$env:Path"
        }
        $uvCargoPath = "$env:USERPROFILE\.cargo\bin"
        if (Test-Path "$uvCargoPath\uv.exe") {
            $env:Path = "$uvCargoPath;$env:Path"
        }

        # Verify installation
        $uvCmd = Get-Command uv -ErrorAction SilentlyContinue
        if (-not $uvCmd) {
            throw "uv not found in PATH after installation"
        }

        Write-Host "  [OK] uv installed" -ForegroundColor Green
    }
    catch {
        Write-Host "  [X] Failed to install uv: $_" -ForegroundColor Red
        Write-Host "      Please install uv manually: https://docs.astral.sh/uv/getting-started/installation/" -ForegroundColor Yellow
        Read-Host "  Press Enter to exit"
        exit 1
    }
} else {
    Write-Host "  [OK] uv found" -ForegroundColor Green
}

# Install Python using uv
Write-Host "  [*] Checking Python $PYTHON_VERSION..." -ForegroundColor Cyan
$pythonCheck = & uv python find $PYTHON_VERSION 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "  [*] Installing Python $PYTHON_VERSION via uv..." -ForegroundColor Cyan
    & uv python install $PYTHON_VERSION
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  [X] Failed to install Python" -ForegroundColor Red
        Read-Host "  Press Enter to exit"
        exit 1
    }
}
Write-Host "  [OK] Python $PYTHON_VERSION ready" -ForegroundColor Green

Write-Host ""
Write-Host "  [*] Installing dependencies..." -ForegroundColor Cyan
Write-Host "      This may take a few minutes. Chill and learn something!" -ForegroundColor Magenta
Write-Host ""
Write-Host "      Did you know?" -ForegroundColor DarkGray
Write-Host ""

# Build uv sync arguments
$syncArgs = @("sync", "--python", $PYTHON_VERSION)
if ($CHINA_MODE) {
    $env:UV_INDEX_URL = $UV_INDEX_URL
}

# Start the sync process
$process = Start-Process -FilePath "uv" -ArgumentList $syncArgs -NoNewWindow -PassThru

# Show tips while installing
$lastTipTime = Get-Date
$currentTip = Get-Random -Maximum $tips.Count

while (-not $process.HasExited) {
    $now = Get-Date
    if (($now - $lastTipTime).TotalSeconds -ge 5) {
        $tip = $tips[$currentTip]
        $line = "      $($tip.icon) $($tip.text)"
        $line = $line.PadRight(90)
        Write-Host "`r$line" -ForegroundColor $tip.color -NoNewline

        $currentTip = ($currentTip + 1) % $tips.Count
        $lastTipTime = $now
    }
    Start-Sleep -Milliseconds 300
}

Write-Host "`r                                                                                              "

if ($process.ExitCode -ne 0) {
    Write-Host ""
    Write-Host "  [X] Failed to install dependencies" -ForegroundColor Red
    Read-Host "  Press Enter to exit"
    exit 1
}

Write-Host "  [OK] Dependencies installed" -ForegroundColor Green

# Install iopaint separately with --no-deps
Write-Host "  [*] Installing iopaint (no deps)..." -ForegroundColor Cyan
if ($CHINA_MODE) {
    & uv pip install iopaint --no-deps --index-url $UV_INDEX_URL 2>&1 | Out-Null
} else {
    & uv pip install iopaint --no-deps 2>&1 | Out-Null
}

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "  [X] Failed to install iopaint" -ForegroundColor Red
    Read-Host "  Press Enter to exit"
    exit 1
}
Write-Host "  [OK] iopaint installed" -ForegroundColor Green

# Verify installation
Write-Host "  [*] Verifying installation..." -ForegroundColor Cyan
$verifyResult = & uv run python -c "import torch; import transformers; import webview; import cv2; print('OK')" 2>&1
if ($verifyResult -notmatch "OK") {
    Write-Host "  [X] Verification failed" -ForegroundColor Red
    Write-Host "      $verifyResult" -ForegroundColor Yellow
    Read-Host "  Press Enter to exit"
    exit 1
}
Write-Host "  [OK] All dependencies verified" -ForegroundColor Green

# Download LaMA model
Write-Host ""
Write-Host "  [*] Downloading AI model (196MB)..." -ForegroundColor Cyan
Write-Host ""
Write-Host "      Did you know?" -ForegroundColor DarkGray
Write-Host ""

$lamaDir = Join-Path $env:USERPROFILE ".cache\torch\hub\checkpoints"
$lamaFile = Join-Path $lamaDir "big-lama.pt"
$lamaUrl = "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt"

if (-not (Test-Path $lamaFile)) {
    if (-not (Test-Path $lamaDir)) {
        New-Item -ItemType Directory -Path $lamaDir -Force | Out-Null
    }

    try {
        $job = Start-Job -ScriptBlock {
            param($url, $dest)
            Invoke-WebRequest -Uri $url -OutFile $dest -UseBasicParsing
        } -ArgumentList $lamaUrl, $lamaFile

        $lastTipTime = Get-Date
        while ($job.State -eq "Running") {
            $now = Get-Date
            if (($now - $lastTipTime).TotalSeconds -ge 5) {
                $tip = $tips[$currentTip]
                $line = "      $($tip.icon) $($tip.text)"
                $line = $line.PadRight(90)
                Write-Host "`r$line" -ForegroundColor $tip.color -NoNewline

                $currentTip = ($currentTip + 1) % $tips.Count
                $lastTipTime = $now
            }
            Start-Sleep -Milliseconds 300
        }

        Write-Host "`r                                                                                              "

        Receive-Job -Job $job | Out-Null
        Remove-Job -Job $job

        if (Test-Path $lamaFile) {
            Write-Host "  [OK] LaMA model ready" -ForegroundColor Green
        } else {
            Write-Host "  [!] Warning: Could not download LaMA model" -ForegroundColor Yellow
            Write-Host "      It will be downloaded on first use" -ForegroundColor Yellow
        }
    }
    catch {
        Write-Host "  [!] Warning: Could not download LaMA model" -ForegroundColor Yellow
        Write-Host "      It will be downloaded on first use" -ForegroundColor Yellow
    }
}
else {
    Write-Host "  [OK] LaMA model already exists" -ForegroundColor Green
}

# Download Florence-2 model
Write-Host ""
Write-Host "  [*] Downloading Florence-2 detection model (~1.5GB)..." -ForegroundColor Cyan
Write-Host ""
Write-Host "      Did you know?" -ForegroundColor DarkGray
Write-Host ""

if ($CHINA_MODE) {
    $env:HF_ENDPOINT = $HF_ENDPOINT
    Write-Host "      Using HF-Mirror for faster download in China" -ForegroundColor DarkGray
    $florenceScript = @"
import os
os.environ['HF_ENDPOINT'] = '$HF_ENDPOINT'
from huggingface_hub import snapshot_download
snapshot_download('florence-community/Florence-2-large', local_dir_use_symlinks=False)
print('FLORENCE_OK')
"@
} else {
    $florenceScript = @"
from huggingface_hub import snapshot_download
snapshot_download('florence-community/Florence-2-large', local_dir_use_symlinks=False)
print('FLORENCE_OK')
"@
}

$florenceProcess = Start-Process -FilePath "uv" -ArgumentList "run", "python", "-c", "`"$florenceScript`"" -NoNewWindow -PassThru

$lastTipTime = Get-Date
while (-not $florenceProcess.HasExited) {
    $now = Get-Date
    if (($now - $lastTipTime).TotalSeconds -ge 5) {
        $tip = $tips[$currentTip]
        $line = "      $($tip.icon) $($tip.text)"
        $line = $line.PadRight(90)
        Write-Host "`r$line" -ForegroundColor $tip.color -NoNewline

        $currentTip = ($currentTip + 1) % $tips.Count
        $lastTipTime = $now
    }
    Start-Sleep -Milliseconds 300
}

Write-Host "`r                                                                                              "

if ($florenceProcess.ExitCode -ne 0) {
    Write-Host "  [!] Warning: Could not download Florence-2 model" -ForegroundColor Yellow
    Write-Host "      It will be downloaded on first use" -ForegroundColor Yellow
}
else {
    Write-Host "  [OK] Florence-2 model ready" -ForegroundColor Green
}

Write-Host ""
Write-Host "  =============================================" -ForegroundColor Green
Write-Host "     Setup complete! Ready to go!              " -ForegroundColor Green
Write-Host "  =============================================" -ForegroundColor Green
Write-Host ""
Write-Host "  To run the app: Double-click " -ForegroundColor Cyan -NoNewline
Write-Host "run.bat" -ForegroundColor White
Write-Host ""

$launch = Read-Host "  Launch now? (y/n)"
if ($launch -eq "y" -or $launch -eq "Y") {
    Write-Host ""
    Write-Host "  Starting WatermarkRemover-AI..." -ForegroundColor Green
    & uv run python remwmgui.py
}

Write-Host ""
Write-Host "  Have fun yeeting watermarks!" -ForegroundColor Magenta
Write-Host ""
Read-Host "  Press Enter to exit"
