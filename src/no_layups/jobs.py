import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from . import config
from .pipeline import PipelineError, run_pipeline

# Section 10: background work runs on a 2-worker in-process pool; job records
# persist to disk (data/jobs/{job_id}.json) so status survives restarts.
_executor = ThreadPoolExecutor(max_workers=2)
_lock = threading.Lock()


def _job_path(job_id: str) -> Path:
    return config.JOBS_DIR / f"{job_id}.json"


def _write_job(record: dict) -> None:
    path = _job_path(record["job_id"])
    with _lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record))


def get_job(job_id: str) -> Optional[dict]:
    """Section 5.4 job status payload, or None if job_id is unknown."""
    path = _job_path(job_id)
    if not path.exists():
        return None
    with _lock:
        return json.loads(path.read_text())


def submit(job_id: str, video_path: Path, handedness: str) -> None:
    """Enqueue a pipeline run for an already-saved upload. job_id == swing_id."""
    _write_job(
        {"job_id": job_id, "status": "queued", "step": None, "error": None, "swing_id": None}
    )
    _executor.submit(_run, job_id, video_path, handedness)


def _run(job_id: str, video_path: Path, handedness: str) -> None:
    def on_progress(step: str) -> None:
        _write_job(
            {
                "job_id": job_id,
                "status": "processing",
                "step": step,
                "error": None,
                "swing_id": None,
            }
        )

    try:
        swing = run_pipeline(video_path, handedness, on_progress)
    except PipelineError as exc:
        _write_job(
            {
                "job_id": job_id,
                "status": "error",
                "step": None,
                "error": {"code": exc.code, "message": exc.message},
                "swing_id": None,
            }
        )
        return
    except Exception as exc:  # pragma: no cover - defensive: pipeline should only raise PipelineError
        _write_job(
            {
                "job_id": job_id,
                "status": "error",
                "step": None,
                "error": {"code": "internal_error", "message": str(exc)},
                "swing_id": None,
            }
        )
        return

    config.SWINGS_DIR.mkdir(parents=True, exist_ok=True)
    swing_path = config.SWINGS_DIR / f"{job_id}.swing.json"
    swing_path.write_text(json.dumps(swing))
    _write_job(
        {"job_id": job_id, "status": "done", "step": None, "error": None, "swing_id": job_id}
    )
