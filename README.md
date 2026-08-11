# No Layups

A golf swing 3D analyzer: footage in, interactive 3D skeleton + swing analysis
out.

Upload a short video of a golf swing (phone footage, static camera) and the
app extracts the golfer's body pose, reconstructs an approximate 3D skeleton,
detects address/top-of-backswing/impact, renders an interactive 3D
stick-figure viewer, and compares 8 swing metrics against a bundled reference
swing.

## Shoot face-on

**V1 supports face-on footage only** — camera square to the golfer, so the
stance runs left-to-right across the frame. Down-the-line footage (camera
behind the golfer, looking along the target line) is out of scope and will
produce wrong numbers rather than an error, because the metrics assume the
swing is being viewed from the front.

This is not just a preference. The 8 metrics are rated as deltas against a
bundled reference swing, and that cancellation only works when the upload and
the reference carry the *same* systematic depth bias — which means the same
camera view. The bundled reference is face-on (measured stance azimuth 8.5 deg
off the image plane), so face-on uploads cancel against it and other views do
not. Supporting a second view would mean bundling a second reference.

Also keep the camera still. Handheld footage still processes, but appends a
`camera_moving` warning.

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
  positions are unreliable, and depth is by far the weakest channel —
  MediaPipe's frame-to-frame depth jitter measures 3-8x its in-image jitter.
- **MediaPipe's 3D output is not rigid.** This is the hard limit on everything
  below. Across frames of a single clip the upper arm's length varies by 92%
  of its own mean and the shoulder width by 79%, while the 2D image landmarks
  the 3D is derived from stay accurate. A body whose bones change length
  cannot yield trustworthy joint angles either, so treat *absolute* metric
  values as indicative only. It is why shoulder turn reads ~60° where the truth
  is nearer 90°, and why the rendered head sits well in front of the shoulders.
- Because of that, the 8 metrics are rated as **deltas against a bundled
  reference swing processed through the same pipeline** (Section 9.2), so the
  systematic part of the bias largely cancels. Compare the deltas, not the
  absolute numbers — and shoot face-on, or the cancellation does not hold (see
  above).
- Turn metrics are read from the depth axis, so a face-on view measures them
  in the worst case: the shoulders are edge-on to the camera exactly at the
  top of the backswing. This is a known cost of the face-on-only scope, and
  the reason turn values read low in absolute terms.
- The skeleton may appear mirrored depending on which side the camera is on;
  the viewer has a mirror toggle.
- Key-event detection (address / top of backswing / impact) is heuristic and
  will occasionally be off by a few frames; a manual override exists in the
  UI for this reason.
