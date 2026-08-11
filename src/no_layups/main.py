import json
from pathlib import Path
from uuid import uuid4

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles

from . import config, jobs, schemas
from .pipeline import metrics

app = FastAPI(title="No Layups")

STATIC_DIR = Path(__file__).parent / "static"


def _swing_path(swing_id: str) -> Path:
    return config.SWINGS_DIR / f"{swing_id}.swing.json"


def _load_swing(swing_id: str) -> dict:
    path = _swing_path(swing_id)
    if not path.exists():
        raise HTTPException(404, "swing not found")
    return json.loads(path.read_text())


def _frames_to_arrays(frame_list: list[dict], joints: list[str]) -> dict[str, np.ndarray]:
    """Rebuild the per-joint (N,3) arrays metrics.py expects from stored swing.json
    frames, with NaN rows for joints absent on a given frame."""
    n = len(frame_list)
    arrays = {joint: np.full((n, 3), np.nan) for joint in joints}
    for i, frame in enumerate(frame_list):
        for joint, p in frame["pos"].items():
            arrays[joint][i] = p
    return arrays


@app.post("/api/swings", status_code=202, response_model=schemas.UploadResponse)
async def create_swing(video: UploadFile = File(...), handedness: str = Form("right")):
    if handedness not in ("right", "left"):
        raise HTTPException(422, "handedness must be 'right' or 'left'")

    content = await video.read()
    if len(content) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(413, "video exceeds the 100 MB upload limit")

    job_id = uuid4().hex
    config.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = Path(video.filename or "").suffix or ".mp4"
    upload_path = config.UPLOADS_DIR / f"{job_id}{suffix}"
    upload_path.write_bytes(content)

    jobs.submit(job_id, upload_path, handedness)
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def get_job_status(job_id: str) -> dict:
    record = jobs.get_job(job_id)
    if record is None:
        raise HTTPException(404, "unknown job id")
    return record


@app.get("/api/swings/{swing_id}")
def get_swing(swing_id: str) -> dict:
    return _load_swing(swing_id)


@app.post("/api/swings/{swing_id}/keyframes")
def update_keyframes(swing_id: str, body: schemas.KeyframesUpdate) -> dict:
    swing = _load_swing(swing_id)
    frame_count = swing["meta"]["frame_count"]
    if not (0 <= body.address < body.top < body.impact < frame_count):
        raise HTTPException(
            422, "keyframes must satisfy 0 <= address < top < impact < frame_count"
        )

    keyframes = {"address": body.address, "top": body.top, "impact": body.impact}
    frames_by_joint = _frames_to_arrays(swing["frames"], swing["joints"])
    handedness = swing["meta"]["handedness"]

    # SPEC DEVIATION (Section 10): the endpoint table documents this response
    # as "updated metrics + comparison". compare.py and GET /api/reference are
    # M6 deliverables (Section 12) and don't exist yet, so this returns the
    # updated swing.json only; M6 adds the comparison payload alongside it.
    swing["keyframes"] = keyframes
    swing["metrics"] = metrics.compute_metrics(frames_by_joint, keyframes, handedness)
    swing["trajectories"] = metrics.compute_trajectories(frames_by_joint, keyframes, handedness)
    swing["meta"]["keyframe_source"] = "manual"

    _swing_path(swing_id).write_text(json.dumps(swing))
    return swing


# Static frontend (Section 11) must be mounted last: Starlette matches routes
# in registration order and a "/" mount would otherwise shadow every route above.
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
