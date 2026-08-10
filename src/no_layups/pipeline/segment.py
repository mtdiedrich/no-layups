import numpy as np

from .. import config
from .errors import PipelineError


def wrist_speed(wrist_xyz: np.ndarray, fps: float) -> np.ndarray:
    """v(t) = ||p(t) - p(t-1)|| * fps, m/s. v(0) is defined as 0 (no prior frame)."""
    diffs = np.diff(wrist_xyz, axis=0)
    speed = np.linalg.norm(diffs, axis=1) * fps
    return np.concatenate([[0.0], speed])


def detect_takeaway_start(wrist_xyz: np.ndarray, fps: float) -> int:
    """Section 7.6 step 1: first frame in [0, 0.4N] where speed exceeds the
    threshold for 3 consecutive frames."""
    n = len(wrist_xyz)
    v = wrist_speed(wrist_xyz, fps)
    limit = int(config.TAKEAWAY_SEARCH_FRACTION * n)
    consecutive = config.TAKEAWAY_CONSECUTIVE_FRAMES
    for t in range(0, limit + 1):
        if t + consecutive <= n and np.all(v[t : t + consecutive] > config.TAKEAWAY_SPEED_THRESHOLD_MPS):
            return t
    raise PipelineError("no_full_swing", "no swing motion detected in the first part of the clip")


def detect_address(wrist_xyz: np.ndarray, fps: float) -> int:
    """Section 7.6 step 2: address = max(0, takeaway_start - round(0.4 * fps))."""
    takeaway_start = detect_takeaway_start(wrist_xyz, fps)
    return max(0, takeaway_start - round(config.ADDRESS_LOOKBACK_S * fps))
