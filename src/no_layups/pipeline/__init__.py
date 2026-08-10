from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np

from .. import config
from . import camera_check, canonical, filtering, pose, segment, video_io
from .errors import PipelineError

__all__ = ["run_pipeline", "PipelineError"]


def _build_frames(xyz_by_joint: dict, fps: float) -> list[dict]:
    n = len(next(iter(xyz_by_joint.values())))
    frames = []
    for i in range(n):
        pos = {}
        for joint, arr in xyz_by_joint.items():
            p = arr[i]
            if not np.isnan(p).any():
                pos[joint] = [float(p[0]), float(p[1]), float(p[2])]
        frames.append({"t": i / fps, "pos": pos})
    return frames


def run_pipeline(video_path: Path, handedness: str, on_progress: Callable[[str], None]) -> dict:
    """Returns the swing.json dict (Section 5.2). Raises PipelineError.

    M2 implements Stages 1-5 only: keyframes/metrics/trajectories stay null.
    Full key-event segmentation (Stage 6) and metrics (Stage 7) land in M4.
    """
    on_progress("validating")
    probe_result = video_io.probe(video_path)
    video_io.validate(probe_result)
    decodable_path = video_io.resolve_decodable_path(video_path)

    on_progress("camera_check")
    warnings = []
    if camera_check.detect_motion(video_io.frames(decodable_path)):
        warnings.append("camera_moving")

    on_progress("extracting_pose")
    raw = pose.extract(video_io.frames(decodable_path))

    on_progress("filtering")
    smoothed = filtering.process(raw, probe_result.fps)

    on_progress("segmenting")
    lead_wrist = config.lead("wrist", handedness)
    address_idx = segment.detect_address(smoothed[lead_wrist], probe_result.fps)
    canonical_frames = canonical.transform(smoothed, address_idx, handedness)

    return {
        "version": 1,
        "meta": {
            "fps": probe_result.fps,
            "frame_count": raw.frame_count,
            "source_width": probe_result.width,
            "source_height": probe_result.height,
            "handedness": handedness,
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "warnings": warnings,
            "keyframe_source": "auto",
        },
        "joints": list(config.JOINTS),
        "frames": _build_frames(canonical_frames, probe_result.fps),
        "keyframes": None,
        "metrics": None,
        "trajectories": None,
    }
