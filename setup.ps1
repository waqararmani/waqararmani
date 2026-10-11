# One-time setup for the Parakeet + Claude Code video editing workflow (Windows PowerShell).
#   powershell -ExecutionPolicy Bypass -File .\setup.ps1          (CPU)
#   powershell -ExecutionPolicy Bypass -File .\setup.ps1 -Gpu     (NVIDIA GPU)
param([switch]$Gpu)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Write-Host "==> Installing ffmpeg via winget"
    winget install --id Gyan.FFmpeg -e --accept-source-agreements --accept-package-agreements
    Write-Host "ffmpeg installed. Close and reopen PowerShell, then run setup.ps1 again."
    exit 0
}

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "==> Installing Python 3.12 via winget"
    winget install --id Python.Python.3.12 -e --accept-source-agreements --accept-package-agreements
    Write-Host "Python installed. Close and reopen PowerShell, then run setup.ps1 again."
    exit 0
}

Write-Host "==> Creating virtual environment in .venv"
python -m venv .venv
$py = ".\.venv\Scripts\python.exe"
& $py -m pip install --upgrade pip | Out-Null

if ($Gpu) {
    Write-Host "==> Installing onnx-asr with CUDA"
    & $py -m pip install "onnx-asr[gpu,hub]"
} else {
    Write-Host "==> Installing onnx-asr (CPU)"
    & $py -m pip install "onnx-asr[cpu,hub]"
}

Write-Host "==> Downloading the Parakeet model (first time only)"
& $py -c "import onnx_asr; onnx_asr.load_model('nemo-parakeet-tdt-0.6b-v2'); onnx_asr.load_vad('silero')"

Write-Host ""
Write-Host "Done. Put your raw clips in footage\ then run 'claude' in this folder and say: edit the footage in footage\"
