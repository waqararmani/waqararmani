#!/usr/bin/env bash
# One-time setup for the Parakeet + Claude Code video editing workflow (macOS / Linux).
#   ./setup.sh          auto-detect: Apple Silicon -> parakeet-mlx, NVIDIA GPU -> onnx-asr[gpu], else onnx-asr[cpu]
#   ./setup.sh --cpu    force the CPU backend
set -euo pipefail
cd "$(dirname "$0")"

FORCE_CPU=0
[[ "${1:-}" == "--cpu" ]] && FORCE_CPU=1

say() { printf '\n==> %s\n' "$*"; }

# 1. ffmpeg (cutting, audio extraction, captions)
if ! command -v ffmpeg >/dev/null 2>&1; then
  say "Installing ffmpeg"
  if [[ "$(uname)" == "Darwin" ]]; then
    command -v brew >/dev/null || { echo "Install Homebrew first: https://brew.sh"; exit 1; }
    brew install ffmpeg
  elif command -v apt-get >/dev/null; then
    sudo apt-get update && sudo apt-get install -y ffmpeg
  elif command -v dnf >/dev/null; then
    sudo dnf install -y ffmpeg
  else
    echo "Please install ffmpeg with your package manager, then re-run."; exit 1
  fi
fi
echo "ffmpeg: $(ffmpeg -version | head -1)"

# 2. Python 3.10+
PY="$(command -v python3 || true)"
[[ -z "$PY" ]] && { echo "Python 3.10+ is required (https://www.python.org/downloads/)"; exit 1; }
"$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' \
  || { echo "Python 3.10+ is required, found $("$PY" --version)"; exit 1; }

say "Creating virtual environment in .venv"
"$PY" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip >/dev/null

# 3. Parakeet backend
if [[ $FORCE_CPU -eq 0 && "$(uname)" == "Darwin" && "$(uname -m)" == "arm64" ]]; then
  say "Apple Silicon detected: installing parakeet-mlx"
  pip install parakeet-mlx
  MODEL_CHECK='from parakeet_mlx import from_pretrained; from_pretrained("mlx-community/parakeet-tdt-0.6b-v2")'
elif [[ $FORCE_CPU -eq 0 ]] && command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  say "NVIDIA GPU detected: installing onnx-asr with CUDA"
  pip install "onnx-asr[gpu,hub]"
  MODEL_CHECK='import onnx_asr; onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v2"); onnx_asr.load_vad("silero")'
  echo "Tip: pass --gpu to tools/transcribe.py to use it."
else
  say "Installing onnx-asr (CPU)"
  pip install "onnx-asr[cpu,hub]"
  MODEL_CHECK='import onnx_asr; onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v2"); onnx_asr.load_vad("silero")'
fi

# 4. Download the Parakeet model once (~600 MB-2.5 GB from Hugging Face)
say "Downloading the Parakeet model (first time only)"
python -c "$MODEL_CHECK"

say "Done. Next:"
echo "  1. Put your raw clips in footage/"
echo "  2. Run: claude   (in this folder) and say: \"edit the footage in footage/\""
echo "     or transcribe manually: .venv/bin/python tools/transcribe.py footage/*"
