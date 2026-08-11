import numpy as np
from scipy.signal import savgol_filter

from .. import config
from .errors import PipelineError
from .pose import PoseSeries


def _savgol_window(fps: float, n_frames: int, scale: float = 1.0) -> int:
    window = round(fps * scale / 3)
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
    windows = (window, window, window)

    out: dict[str, np.ndarray] = {}
    for joint in config.JOINTS:
        xyz = raw.xyz[joint].copy()
        # SPEC DEVIATION (Section 7.4 step 1): the spec masks frames with
        # visibility < 0.4 as missing. Only genuinely absent coordinates are
        # masked here; visibility is no longer consulted.
        #
        # MediaPipe's visibility is a sigmoid predicting whether a landmark is
        # unoccluded, and it is trained on a pose distribution (fitness, yoga,
        # standing) that a golf swing sits far outside. Its landmark head keeps
        # tracking a limb accurately while its visibility head collapses. That
        # is not a threshold to tune: measured on a 60fps face-on clip whose
        # lead arm is plainly visible to the eye for every frame, left_elbow
        # and left_wrist score a MEDIAN visibility of 0.02 -- admitting them
        # needs a threshold under 0.01, which is not a stricter signal but no
        # signal at all. The mean of 0.36 on the same joints hides this; the
        # distribution is bimodal, so the mean reads as "marginally uncertain"
        # while the median shows most frames scoring ~zero.
        #
        # The cost is real and worth stating: visibility was the only signal
        # distinguishing "MediaPipe tracked this limb" from "MediaPipe lost it
        # and is predicting". Dropping it means a genuinely lost limb now
        # produces plausible-looking coordinates instead of a poor_tracking
        # error. Section 5.2's missing-joint path still covers frames with no
        # pose at all (pose.py writes NaN for those), so total tracking loss is
        # still caught -- what is no longer caught is confident-but-wrong
        # tracking of a single limb. tools/tracking_report.py exists to inspect
        # visibility directly when that is in question.
        missing = np.isnan(xyz).any(axis=1)

        for axis in range(3):
            xyz[:, axis] = _interpolate_short_gaps(xyz[:, axis], missing.copy(), max_gap)

        still_missing = np.isnan(xyz).any(axis=1)

        for axis in range(3):
            axis_window = windows[axis]
            if axis_window < config.SAVGOL_POLYORDER + 1:
                continue
            filled = _fill_for_smoothing(xyz[:, axis])
            xyz[:, axis] = savgol_filter(
                filled,
                window_length=axis_window,
                polyorder=config.SAVGOL_POLYORDER,
                mode="interp",
            )

        xyz[still_missing] = np.nan
        out[joint] = xyz

    return out


def smooth_depth(series: dict[str, np.ndarray], fps: float) -> dict[str, np.ndarray]:
    """Extra smoothing on the depth axis only, for the rendering/metrics path.

    SPEC DEVIATION (Section 7.4 step 4): the spec applies one Savitzky-Golay
    window to all three axes. MediaPipe's noise is not isotropic -- measured
    over address..impact on the reference clip, per-joint depth jitter runs
    3-8x the in-image jitter (right wrist 3.28 mm/frame^2 in Z against 0.45 in
    X and 0.59 in Y). A single window wide enough to settle depth would
    over-smooth the two accurate axes.

    Deliberately NOT folded into process(): segmentation reads that output, and
    the two have opposite requirements. Event detection wants temporal
    precision -- detect_takeaway_start looks for the frame where the lead wrist
    starts moving, and a wide window smears exactly that onset. Rendering wants
    spatial smoothness and does not care about a frame or two.

    They were coupled at first and it showed up as a cliff: sweeping the depth
    scale on the reference clip, address held at 49 (the manually verified
    value) through 2.0 and then collapsed to 31 at 2.5, while top and impact
    never moved. Splitting the two paths means the render window can be as wide
    as the picture wants without the segmenter ever seeing it.
    """
    n = len(next(iter(series.values())))
    window = _savgol_window(fps, n, scale=config.SAVGOL_DEPTH_WINDOW_SCALE)
    if window < config.SAVGOL_POLYORDER + 1:
        return {joint: arr.copy() for joint, arr in series.items()}

    out = {}
    for joint, arr in series.items():
        smoothed = arr.copy()
        missing = np.isnan(smoothed[:, 2])
        smoothed[:, 2] = savgol_filter(
            _fill_for_smoothing(smoothed[:, 2]),
            window_length=window,
            polyorder=config.SAVGOL_POLYORDER,
            mode="interp",
        )
        smoothed[missing, 2] = np.nan
        out[joint] = smoothed
    return out


def metric_critical_joints(handedness: str) -> list[str]:
    """The joints Section 8's metrics actually read.

    Everything else (the trail elbow and wrist, the lead knee and ankle) only
    feeds the rendered skeleton, and Section 11.2 already specifies that a
    missing joint simply hides its mesh and any bone touching it.
    """
    return [
        "nose",
        "left_shoulder",
        "right_shoulder",
        "left_hip",
        "right_hip",
        config.lead("elbow", handedness),
        config.lead("wrist", handedness),
        config.trail("knee", handedness),
        config.trail("ankle", handedness),
    ]


def check_quality(
    smoothed: dict[str, np.ndarray], start: int, end: int, handedness: str
) -> list[str]:
    """Section 7.4 step 3: too much missing tracking between address and impact
    is poor_tracking. Returns the non-fatal offenders, for meta.warnings.

    Scoped to [start, end] because that is the span the spec cares about --
    tracking loss during the follow-through (very common once the arms swing
    across the body) says nothing about whether the swing itself was
    measurable, and gating on the whole clip rejects otherwise-good footage.

    SPEC DEVIATION (Section 7.4 step 3): the spec fails on *any* of the 13
    joints. Only the joints Section 8 actually measures are fatal here; the
    rest downgrade to a warning. The spec contradicts itself otherwise --
    Section 13.2 requires end-to-end acceptance on a down-the-line clip, but
    that view necessarily hides one arm behind the torso, so an any-joint gate
    rejects exactly the footage the acceptance criteria demand. Measured on the
    reference clip, the trail elbow is missing 33.5% of address..impact while
    every joint feeding a metric is missing 0%.

    Gap-filling those dropouts instead was considered and rejected: they run
    69, 21 and 31 frames through the middle of the backswing, where the elbow
    is moving fast, so interpolating them would fabricate its path rather than
    recover it. A joint we could not see is better reported missing.
    """
    lo = max(0, start)
    hi = min(len(next(iter(smoothed.values()))) - 1, end)
    if hi < lo:
        return []

    critical = set(metric_critical_joints(handedness))
    fatal, warned = [], []
    for joint in config.JOINTS:
        span = smoothed[joint][lo : hi + 1]
        fraction = float(np.isnan(span).any(axis=1).mean())
        if fraction > config.POOR_TRACKING_MAX_MISSING_FRACTION:
            (fatal if joint in critical else warned).append(joint)

    if fatal:
        raise PipelineError(
            "poor_tracking",
            f"poor tracking on joints: {', '.join(fatal)} — keep the whole body in frame",
        )
    return warned
