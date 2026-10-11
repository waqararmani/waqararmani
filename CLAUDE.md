# Video editing workspace (Parakeet + Claude Code)

This folder turns raw talking-head footage into an edited video:
**Parakeet transcribes → Claude reads the transcript and decides the cuts → ffmpeg renders.**

Python lives in `.venv` — use `.venv/bin/python` (macOS/Linux) or `.venv\Scripts\python.exe` (Windows).
If `.venv` is missing, tell the user to run `./setup.sh` (or `setup.ps1` on Windows) first.

## Folders
- `footage/` – raw clips from the user (never modify or delete these)
- `transcripts/` – Parakeet output: `<clip>.txt` (read this), `<clip>.json` (word timestamps), `<clip>.srt`
- `edits/` – edit decision lists (EDL JSON) you write
- `exports/` – rendered videos

## Workflow when asked to edit footage
1. **Transcribe** every clip that has no transcript yet:
   `.venv/bin/python tools/transcribe.py footage/*.mp4` (add `--gpu` on an NVIDIA machine).
2. **Read** `transcripts/<clip>.txt`. Each line is `[start - end] text`; long silences are marked `... pause Ns ...`.
   For exact cut points inside a line, look up word timings in `transcripts/<clip>.json` (`segments[].words[]`).
3. **Decide the edit.** By default:
   - drop flubbed lines, false starts and repeated takes — keep the **last** clean take of a repeated line
   - drop filler-only lines ("um", "uh", "so yeah") and long pauses
   - keep the story in a logical order; drop anything off-topic only if the user asked for a tighter cut
   - start each clip ~0.10 s before the first word and end ~0.20 s after the last word so speech isn't clipped
   - merge clips that are less than ~0.3 s apart
4. **Write the EDL** to `edits/<name>.json`:
   ```json
   {
     "output": "exports/<name>.mp4",
     "clips": [
       {"source": "footage/take1.mp4", "start": 0.10, "end": 8.20, "note": "intro"},
       {"source": "footage/take1.mp4", "start": 14.75, "end": 31.10, "note": "skips repeated line at 9-14s"}
     ]
   }
   ```
   Every clip gets a short `note` explaining why it's there / what was cut around it.
5. **Show the user** a short summary table of kept clips and what was removed (with timestamps), and the
   new total runtime, before the final render if the edit is large or ambiguous.
6. **Render**: `.venv/bin/python tools/render.py edits/<name>.json`
   - `--preview` fast low-quality render for checking cuts
   - `--captions` also write an `.srt` for the edited timeline; `--burn` burns captions into the video
7. If the user asks for changes ("tighten the intro", "put back the bit about X"), edit the EDL and re-render.
   Don't re-transcribe unless the footage changed.

## Rules
- Never overwrite or delete files in `footage/`.
- Timestamps in the EDL are seconds in the **source** clip.
- Don't invent transcript content; if a transcript looks wrong, say so and offer to re-run with another model
  (`--model nemo-parakeet-tdt-0.6b-v3` for non-English European languages).
