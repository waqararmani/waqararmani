# Video editing with Claude Code + NVIDIA Parakeet

Free, local workflow for editing talking-head / YouTube footage:

1. **Parakeet** (NVIDIA's free speech-recognition model) listens to the raw footage and writes a transcript with timestamps.
2. **Claude Code** reads the transcript and decides which clips to keep, where to cut, and which flubs/repeats/pauses to remove.
3. **ffmpeg** renders the final video (optionally with captions).

## Install (once)

Requirements: Python 3.10+, ~3 GB free disk, internet for the first model download. ffmpeg is installed for you if missing.

**macOS / Linux**
```bash
git clone <this repo> video-editing && cd video-editing
./setup.sh
```
- Apple Silicon Macs get `parakeet-mlx` (fast, runs on the Mac GPU).
- Machines with an NVIDIA GPU get `onnx-asr` with CUDA; everything else gets the CPU version (still faster than real-time).

**Windows (PowerShell)**
```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1        # add -Gpu if you have an NVIDIA card
```

You also need Claude Code: https://code.claude.com (`npm install -g @anthropic-ai/claude-code`).

## Use

1. Drop raw clips into `footage/`.
2. Run `claude` in this folder and ask, e.g.:
   > Edit the footage in footage/ into one video. Cut flubs, repeated takes and long pauses. Add captions.
3. Claude transcribes, writes an edit plan to `edits/`, and renders to `exports/`. Ask for changes in plain English.

### Manual commands
```bash
.venv/bin/python tools/transcribe.py footage/*.mp4          # -> transcripts/<clip>.txt/.json/.srt
.venv/bin/python tools/render.py edits/final.json --preview # quick check
.venv/bin/python tools/render.py edits/final.json --burn    # final, captions burned in
```
(On Windows use `.venv\Scripts\python.exe`.)

## Models
| Model | Use |
|---|---|
| `nemo-parakeet-tdt-0.6b-v2` (default) | English, best accuracy |
| `nemo-parakeet-tdt-0.6b-v3` | 25 European languages |

Change with `--model`. See `CLAUDE.md` for how Claude makes edit decisions.
