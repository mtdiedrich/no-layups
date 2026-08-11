# No Layups

A golf swing 3D analyzer: footage in, interactive 3D skeleton + swing analysis
out.

Upload a short video of a golf swing (phone footage, static camera) and the
app extracts the golfer's body pose, reconstructs an approximate 3D skeleton,
detects address/top-of-backswing/impact, renders an interactive 3D
stick-figure viewer, and compares 8 swing metrics against a bundled reference
swing.

See `docs/spec/No_Layups_V1_Specification.pdf` for the full technical
specification this project implements.

## Requirements

- Python 3.11 or 3.12 (3.13 is not supported by the pinned MediaPipe version).
- [`uv`](https://docs.astral.sh/uv/) for dependency management.
- `ffmpeg` + `ffprobe` on the system `PATH` (used for video probing and the
  HEVC transcode fallback). `uv` does not manage this — install separately
  (e.g. `sudo apt install ffmpeg`, `brew install ffmpeg`).

## Setup

```bash
uv sync                # creates .venv, installs deps + the project (editable)
uv run uvicorn no_layups.main:app --host 127.0.0.1 --port 8000
# or:
uv run no-layups serve
```

The app is fully usable at `http://127.0.0.1:8000/` with no other services
running.

To run the pose-extraction pipeline directly on a video without the web UI:

```bash
uv run no-layups process path/to/swing.mp4 --handedness right -o out.swing.json
```

Run all commands (pytest, the CLI, uvicorn) via `uv run ...` from the repo
root — never invoke a system `python`/`pip` directly.

```bash
uv run pytest
```

## Known accuracy limits

- 3D is inferred from a single 2D camera view and is approximate. Absolute
  positions are unreliable, and the depth axis is the weakest channel —
  MediaPipe's frame-to-frame depth jitter runs 2-3x its in-image jitter.
- **Camera angle matters more than resolution.** Turn metrics are read from
  the depth axis, so they depend on the view having something to see. On a
  face-on clip the shoulders are edge-on to the camera exactly at the top of
  the backswing — the worst case — and shoulder turn reads ~12° where the
  truth is nearer 90°. The same swing from an angled view reads ~55°. Shoot
  from a view that shows the turn, and treat turn metrics from a dead face-on
  clip with suspicion.
- Because of that, the 8 metrics are rated against a bundled **reference swing
  processed through the same pipeline** (Section 9.2), so a systematic depth
  bias largely cancels. Compare the deltas, not the absolute numbers. A clip
  shot from a very different angle than the reference cancels less cleanly.
- `spine_tilt_address` is reported as `null` by design. The canonical frame
  defines "up" as the address trunk vector, so an absolute spine tilt at
  address is identically zero and cannot be measured; see `metrics.py`.
- The skeleton may appear mirrored depending on which side the camera is on;
  the viewer has a mirror toggle.
- Key-event detection (address / top of backswing / impact) is heuristic and
  will occasionally be off by a few frames; a manual override exists in the
  UI for this reason.
