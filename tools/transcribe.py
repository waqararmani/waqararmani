#!/usr/bin/env python3
"""Transcribe raw footage with NVIDIA Parakeet and write timestamped transcripts.

For each input file this writes, next to it in --out (default: transcripts/):

  <name>.json  - segments + word-level timestamps (machine readable, used by render.py)
  <name>.txt   - one line per segment as "[start - end] text", with long pauses marked;
                 this is the file Claude Code reads to decide cuts
  <name>.srt   - standard subtitles

Backends:
  onnx  - onnx-asr runtime (Windows / Linux / Intel Mac, CPU or NVIDIA GPU)  [default]
  mlx   - parakeet-mlx (Apple Silicon Macs, fastest there)
  auto  - mlx if installed, otherwise onnx
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

DEFAULT_ONNX_MODEL = "nemo-parakeet-tdt-0.6b-v2"  # English. Use ...-v3 for 25 European languages.
DEFAULT_MLX_MODEL = "mlx-community/parakeet-tdt-0.6b-v2"
PAUSE_MARK_SECONDS = 1.0


def extract_audio(src: Path, dst: Path) -> None:
    """Pull a 16 kHz mono WAV out of any video/audio file (what Parakeet expects)."""
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y", "-i", str(src), "-vn", "-ac", "1", "-ar", "16000", str(dst)],
        check=True,
    )


def transcribe_onnx(wav: Path, model_name: str, use_gpu: bool) -> list[dict]:
    import onnx_asr

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_gpu else ["CPUExecutionProvider"]
    model = onnx_asr.load_model(model_name, providers=providers)
    vad = onnx_asr.load_vad("silero", providers=providers)

    segments = []
    for seg in model.with_vad(vad).with_timestamps().recognize(str(wav)):
        words: list[dict] = []
        tokens = seg.tokens or []
        stamps = seg.timestamps or []
        # Tokens are SentencePiece pieces; a leading space starts a new word.
        # Token timestamps are relative to the segment start and mark token onset.
        for i, (tok, ts) in enumerate(zip(tokens, stamps)):
            t = round(seg.start + ts, 3)
            if tok.startswith(" ") or not words:
                if words:
                    words[-1]["end"] = t
                words.append({"word": tok.strip(), "start": t, "end": t})
            else:
                words[-1]["word"] += tok
        if words:
            words[-1]["end"] = round(seg.end, 3)
        words = [w for w in words if w["word"]]
        text = seg.text.strip()
        if text:
            segments.append({"start": round(seg.start, 3), "end": round(seg.end, 3), "text": text, "words": words})
    return segments


def transcribe_mlx(wav: Path, model_name: str) -> list[dict]:
    from parakeet_mlx import from_pretrained

    model = from_pretrained(model_name)
    result = model.transcribe(str(wav), chunk_duration=120.0, overlap_duration=15.0)
    segments = []
    for sent in result.sentences:
        words: list[dict] = []
        for tok in sent.tokens:
            piece = tok.text
            if piece.startswith(" ") or not words:
                words.append({"word": piece.strip(), "start": round(tok.start, 3), "end": round(tok.end, 3)})
            else:
                words[-1]["word"] += piece
                words[-1]["end"] = round(tok.end, 3)
        words = [w for w in words if w["word"]]
        text = sent.text.strip()
        if text:
            segments.append({"start": round(sent.start, 3), "end": round(sent.end, 3), "text": text, "words": words})
    return segments


def fmt_clock(t: float) -> str:
    m, s = divmod(t, 60)
    h, m = divmod(int(m), 60)
    return f"{h:d}:{m:02d}:{s:06.3f}" if h else f"{m:02d}:{s:06.3f}"


def fmt_srt(t: float) -> str:
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_outputs(src: Path, segments: list[dict], duration: float, out_dir: Path, backend: str, model: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / src.stem

    data = {
        "source": str(src.resolve()),
        "duration": duration,
        "backend": backend,
        "model": model,
        "segments": segments,
    }
    stem.with_suffix(".json").write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [f"# {src.name}  (duration {fmt_clock(duration)})"]
    prev_end = 0.0
    for seg in segments:
        gap = seg["start"] - prev_end
        if gap >= PAUSE_MARK_SECONDS:
            lines.append(f"            ... pause {gap:.1f}s ...")
        lines.append(f"[{fmt_clock(seg['start'])} - {fmt_clock(seg['end'])}] {seg['text']}")
        prev_end = seg["end"]
    if duration - prev_end >= PAUSE_MARK_SECONDS:
        lines.append(f"            ... pause {duration - prev_end:.1f}s (to end) ...")
    stem.with_suffix(".txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    srt = []
    for i, seg in enumerate(segments, 1):
        srt.append(f"{i}\n{fmt_srt(seg['start'])} --> {fmt_srt(seg['end'])}\n{seg['text']}\n")
    stem.with_suffix(".srt").write_text("\n".join(srt), encoding="utf-8")


def probe_duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return round(float(out), 3) if out else 0.0


def pick_backend(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import parakeet_mlx  # noqa: F401

        return "mlx"
    except ImportError:
        return "onnx"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", type=Path, help="video or audio files")
    ap.add_argument("--out", type=Path, default=Path("transcripts"), help="output folder (default: transcripts/)")
    ap.add_argument("--backend", choices=["auto", "onnx", "mlx"], default="auto")
    ap.add_argument("--model", help="override the Parakeet model name")
    ap.add_argument("--gpu", action="store_true", help="onnx backend: use an NVIDIA GPU (CUDA)")
    args = ap.parse_args()

    backend = pick_backend(args.backend)
    model = args.model or (DEFAULT_MLX_MODEL if backend == "mlx" else DEFAULT_ONNX_MODEL)

    for src in args.inputs:
        if not src.exists():
            print(f"skip: {src} not found", file=sys.stderr)
            continue
        print(f"transcribing {src} with Parakeet ({backend}: {model}) ...", file=sys.stderr)
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "audio.wav"
            extract_audio(src, wav)
            if backend == "mlx":
                segments = transcribe_mlx(wav, model)
            else:
                segments = transcribe_onnx(wav, model, args.gpu)
        write_outputs(src, segments, probe_duration(src), args.out, backend, model)
        print(f"  -> {args.out / src.stem}.txt / .json / .srt", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
