import numpy as np
from scipy.signal import savgol_filter

from .. import config
from .errors import PipelineError
from .pose import PoseSeries


def _savgol_window(fps: float, n_frames: int) -> int:
    window = round(fps / 3)
    if window % 2 == 0:
        window += 1
    hi = n_frames if n_frames % 2 == 1 else n_frames - 1
    window = min(window, hi)
    window = max(window, 5)
    if window > hi:
        window = hi if hi % 2 == 1 else hi - 1
    return window


def _interpolate_short_gaps(values: np.ndarray, missing: np.ndarray, max_gap: int) -> np.ndarray:
    values = values.copy()
    n = len(values)
    idx = np.arange(n)
    i = 0
    while i < n:
        if not missing[i]:
            i += 1
            continue
        j = i
        while j < n and missing[j]:
            j += 1
        run_len = j - i
        if run_len <= max_gap and i > 0 and j < n:
            values[i:j] = np.interp(idx[i:j], [i - 1, j], [values[i - 1], values[j]])
        i = j
    return values


def process(raw: PoseSeries, fps: float) -> dict[str, np.ndarray]:
    """Section 7.4: gap-fill short dropouts then Savitzky-Golay smooth, per joint per axis."""
    n = raw.frame_count
    max_gap = round(config.GAP_FILL_MAX_FRACTION_OF_FPS * fps)
    window = _savgol_window(fps, n)

    out: dict[str, np.ndarray] = {}
    missing_fraction: dict[str, float] = {}

    for joint in config.JOINTS:
        vis = raw.vis[joint]
        xyz = raw.xyz[joint].copy()
        missing = (vis < config.VISIBILITY_THRESHOLD) | np.isnan(xyz).any(axis=1)
        xyz[missing] = np.nan

        for axis in range(3):
            xyz[:, axis] = _interpolate_short_gaps(xyz[:, axis], missing.copy(), max_gap)

        missing_fraction[joint] = float(np.isnan(xyz).any(axis=1).mean())

        if window >= config.SAVGOL_POLYORDER + 1:
            for axis in range(3):
                xyz[:, axis] = savgol_filter(
                    xyz[:, axis], window_length=window, polyorder=config.SAVGOL_POLYORDER, mode="interp"
                )

        out[joint] = xyz

    offending = [j for j, frac in missing_fraction.items() if frac > config.POOR_TRACKING_MAX_MISSING_FRACTION]
    if offending:
        raise PipelineError(
            "poor_tracking",
            f"poor tracking on joints: {', '.join(offending)} — keep the whole body in frame",
        )

    return out
