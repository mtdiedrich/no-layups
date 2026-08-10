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


def _fill_for_smoothing(values: np.ndarray) -> np.ndarray:
    """Temporarily fill every remaining NaN so savgol never sees one.

    Section 7.4's pitfall note ("Savitzky-Golay on a series containing NaN
    produces NaN everywhere") is handled by smoothing a fully-filled copy and
    re-applying the true missing mask afterwards. Filling only the short gaps
    and smoothing in place would let long gaps bleed NaN into ~window/2
    neighbouring frames that actually tracked fine.
    """
    missing = np.isnan(values)
    if not missing.any():
        return values
    if missing.all():
        return np.zeros_like(values)
    idx = np.arange(len(values))
    return np.interp(idx, idx[~missing], values[~missing])


def process(raw: PoseSeries, fps: float) -> dict[str, np.ndarray]:
    """Section 7.4 steps 1, 2 and 4: gap-fill short dropouts then Savitzky-Golay
    smooth, per joint per axis. Frames that stay unrecoverable are NaN.

    The step-3 quality gate is NOT applied here -- see check_quality(), which
    the caller runs over the address..impact span once keyframes are known.
    """
    n = raw.frame_count
    max_gap = round(config.GAP_FILL_MAX_FRACTION_OF_FPS * fps)
    window = _savgol_window(fps, n)

    out: dict[str, np.ndarray] = {}
    for joint in config.JOINTS:
        vis = raw.vis[joint]
        xyz = raw.xyz[joint].copy()
        missing = (vis < config.VISIBILITY_THRESHOLD) | np.isnan(xyz).any(axis=1)
        xyz[missing] = np.nan

        for axis in range(3):
            xyz[:, axis] = _interpolate_short_gaps(xyz[:, axis], missing.copy(), max_gap)

        still_missing = np.isnan(xyz).any(axis=1)

        if window >= config.SAVGOL_POLYORDER + 1:
            for axis in range(3):
                filled = _fill_for_smoothing(xyz[:, axis])
                xyz[:, axis] = savgol_filter(
                    filled,
                    window_length=window,
                    polyorder=config.SAVGOL_POLYORDER,
                    mode="interp",
                )

        xyz[still_missing] = np.nan
        out[joint] = xyz

    return out


def check_quality(smoothed: dict[str, np.ndarray], start: int, end: int) -> None:
    """Section 7.4 step 3: if any joint is still missing in more than 25% of
    the frames between address and impact, raise poor_tracking.

    Scoped to [start, end] because that is the span the spec cares about --
    tracking loss during the follow-through (very common once the arms swing
    across the body) says nothing about whether the swing itself was
    measurable, and gating on the whole clip rejects otherwise-good footage.
    """
    lo = max(0, start)
    hi = min(len(next(iter(smoothed.values()))) - 1, end)
    if hi < lo:
        return

    offending = []
    for joint in config.JOINTS:
        span = smoothed[joint][lo : hi + 1]
        fraction = float(np.isnan(span).any(axis=1).mean())
        if fraction > config.POOR_TRACKING_MAX_MISSING_FRACTION:
            offending.append(joint)

    if offending:
        raise PipelineError(
            "poor_tracking",
            f"poor tracking on joints: {', '.join(offending)} — keep the whole body in frame",
        )
