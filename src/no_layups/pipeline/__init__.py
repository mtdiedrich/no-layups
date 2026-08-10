from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np

from .. import config
from . import camera_check, canonical, filtering, metrics, pose, segment, video_io
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
    """Returns the swing.json dict (Section 5.2). Raises PipelineError."""
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
    lead_wrist_xyz = smoothed[config.lead("wrist", handedness)]
    try:
        keyframes = segment.detect_keyframes(lead_wrist_xyz, probe_result.fps)
        address_idx = keyframes["address"]
    except PipelineError as exc:
        if exc.code != "no_full_swing":
            raise
        keyframes = None
        warnings.append("no_full_swing")
        try:
            address_idx = segment.detect_address(lead_wrist_xyz, probe_result.fps)
        except PipelineError:
            address_idx = 0

    # Section 7.4 step 3: the tracking-quality gate covers address..impact,
    # falling back to the whole clip when segmentation could not place them.
    if keyframes is not None:
        filtering.check_quality(smoothed, keyframes["address"], keyframes["impact"])
    else:
        filtering.check_quality(smoothed, 0, raw.frame_count - 1)

    canonical_frames = canonical.transform(smoothed, address_idx, handedness)

    if keyframes is not None:
        on_progress("computing")
        metrics_dict = metrics.compute_metrics(canonical_frames, keyframes, handedness)
        trajectories_dict = metrics.compute_trajectories(canonical_frames, keyframes, handedness)
    else:
        metrics_dict = None
        trajectories_dict = None

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
        "keyframes": keyframes,
        "metrics": metrics_dict,
        "trajectories": trajectories_dict,
    }
