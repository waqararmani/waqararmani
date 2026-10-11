#!/usr/bin/env python3
"""Render an edit decision list (EDL) into a finished video with ffmpeg.

EDL format (JSON) - this is what Claude Code writes after reading the transcripts:

{
  "output": "exports/final.mp4",
  "clips": [
    {"source": "footage/take1.mp4", "start": 0.00, "end": 8.20, "note": "intro"},
    {"source": "footage/take1.mp4", "start": 14.75, "end": 31.10, "note": "skip flubbed line"}
  ]
}

Clips are joined in order. Each clip is re-encoded frame-accurately, scaled to the
first clip's resolution/frame rate, and given a tiny audio fade at the cut so there
are no clicks.

Options:
  --captions   also write an .srt for the edited timeline (word timestamps from transcripts/)
  --burn       burn those captions into the video
  --preview    fast low-quality render (for checking cuts)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

FADE = 0.012  # seconds of audio fade at every cut


def probe_video(path: Path) -> tuple[int, int, str]:
    out = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate", "-of", "json", str(path),
        ],
        check=True, capture_output=True, text=True,
    ).stdout
    stream = json.loads(out)["streams"][0]
    return int(stream["width"]), int(stream["height"]), stream["r_frame_rate"]


def fmt_srt(t: float) -> str:
    ms = int(round(max(t, 0) * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def build_captions(clips: list[dict], transcripts: Path, max_words: int = 7, max_len: float = 3.0) -> str:
    """Remap word timestamps from each source's transcript onto the edited timeline."""
    cache: dict[str, list[dict]] = {}
    words: list[dict] = []
    offset = 0.0
    for clip in clips:
        src = Path(clip["source"])
        if str(src) not in cache:
            tj = transcripts / f"{src.stem}.json"
            if not tj.exists():
                print(f"warning: no transcript {tj}; clip has no captions", file=sys.stderr)
                cache[str(src)] = []
            else:
                data = json.loads(tj.read_text(encoding="utf-8"))
                cache[str(src)] = [w for seg in data["segments"] for w in seg["words"]]
        start, end = clip["start"], clip["end"]
        for w in cache[str(src)]:
            mid = (w["start"] + w["end"]) / 2
            if start <= mid <= end:
                words.append({
                    "word": w["word"],
                    "start": offset + max(w["start"], start) - start,
                    "end": offset + min(w["end"], end) - start,
                })
        offset += end - start

    cues: list[list[dict]] = []
    for w in words:
        if (
            not cues
            or len(cues[-1]) >= max_words
            or w["end"] - cues[-1][0]["start"] > max_len
            or w["start"] - cues[-1][-1]["end"] > 0.6
        ):
            cues.append([])
        cues[-1].append(w)

    out = []
    for i, cue in enumerate(cues, 1):
        text = " ".join(w["word"] for w in cue)
        out.append(f"{i}\n{fmt_srt(cue[0]['start'])} --> {fmt_srt(cue[-1]['end'])}\n{text}\n")
    return "\n".join(out)


def render(edl: dict, output: Path, preview: bool, subs: Path | None) -> None:
    clips = edl["clips"]
    if not clips:
        sys.exit("EDL has no clips")
    width, height, fps = probe_video(Path(clips[0]["source"]))

    cmd = ["ffmpeg", "-loglevel", "error", "-stats", "-y"]
    filters = []
    for i, clip in enumerate(clips):
        start, end = float(clip["start"]), float(clip["end"])
        if end <= start:
            sys.exit(f"clip {i} has end <= start: {clip}")
        cmd += ["-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", clip["source"]]
        dur = end - start
        fade = min(FADE, dur / 4)
        filters.append(
            f"[{i}:v:0]scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={fps},setpts=PTS-STARTPTS[v{i}]"
        )
        filters.append(
            f"[{i}:a:0]aresample=48000,aformat=channel_layouts=stereo,asetpts=PTS-STARTPTS,"
            f"afade=t=in:d={fade:.3f},afade=t=out:st={dur - fade:.3f}:d={fade:.3f}[a{i}]"
        )
    joined = "".join(f"[v{i}][a{i}]" for i in range(len(clips)))
    filters.append(f"{joined}concat=n={len(clips)}:v=1:a=1[vcat][aout]")
    if subs:
        path = str(subs.resolve()).replace("\\", "/").replace(":", "\\:").replace("'", "\\'")
        filters.append(f"[vcat]subtitles='{path}'[vout]")
    else:
        filters.append("[vcat]null[vout]")

    cmd += ["-filter_complex", ";".join(filters), "-map", "[vout]", "-map", "[aout]"]
    if preview:
        cmd += ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "30"]
    else:
        cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", "18"]
    cmd += ["-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(output)]

    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, check=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("edl", type=Path, help="edit decision list JSON")
    ap.add_argument("-o", "--output", type=Path, help="output file (overrides EDL 'output')")
    ap.add_argument("--transcripts", type=Path, default=Path("transcripts"))
    ap.add_argument("--captions", action="store_true", help="write <output>.srt for the edited timeline")
    ap.add_argument("--burn", action="store_true", help="burn captions into the video (implies --captions)")
    ap.add_argument("--preview", action="store_true", help="fast, low-quality render")
    args = ap.parse_args()

    edl = json.loads(args.edl.read_text(encoding="utf-8"))
    output = args.output or Path(edl.get("output", "exports/final.mp4"))
    if args.preview and not args.output:
        output = output.with_name(output.stem + ".preview" + output.suffix)

    total = sum(c["end"] - c["start"] for c in edl["clips"])
    print(f"{len(edl['clips'])} clips, {total:.1f}s total -> {output}", file=sys.stderr)

    with tempfile.TemporaryDirectory() as tmp:
        subs = None
        if args.captions or args.burn:
            srt_text = build_captions(edl["clips"], args.transcripts)
            srt_path = output.with_suffix(".srt")
            srt_path.parent.mkdir(parents=True, exist_ok=True)
            srt_path.write_text(srt_text, encoding="utf-8")
            print(f"captions -> {srt_path}", file=sys.stderr)
            if args.burn:
                subs = Path(tmp) / "captions.srt"
                subs.write_text(srt_text, encoding="utf-8")
        render(edl, output, args.preview, subs)
    print(f"done: {output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
