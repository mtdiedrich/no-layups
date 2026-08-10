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

Run all commands (pytest, the CLI, uvicorn) via `uv run ...` from the repo
root — never invoke a system `python`/`pip` directly.

```bash
uv run pytest
```

## Known accuracy limits

- 3D is inferred from a single 2D camera view and is approximate. Absolute
  positions are unreliable; joint angles and rotations are the trustworthy
  output.
- The skeleton may appear mirrored depending on which side the camera is on;
  the viewer has a mirror toggle.
- Key-event detection (address / top of backswing / impact) is heuristic and
  will occasionally be off by a few frames; a manual override exists in the
  UI for this reason.
