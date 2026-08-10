import numpy as np

from .. import config
from .errors import PipelineError


def wrist_speed(wrist_xyz: np.ndarray, fps: float) -> np.ndarray:
    """v(t) = ||p(t) - p(t-1)|| * fps, m/s. v(0) is defined as 0 (no prior frame)."""
    diffs = np.diff(wrist_xyz, axis=0)
    speed = np.linalg.norm(diffs, axis=1) * fps
    return np.concatenate([[0.0], speed])


def wrist_height(wrist_xyz: np.ndarray) -> np.ndarray:
    """Height of the lead wrist, increasing upward.

    SPEC DEVIATION (Section 7.6 step 4): the spec takes argmax of the raw
    wrist Y for the top of the backswing, which assumes Y grows upward.
    MediaPipe's world landmarks grow DOWNWARD -- measured on real footage,
    the nose sits at y=-0.37 and the ankle at y=+0.73 at address. Segmentation
    runs on those raw values (Pass A of the Section 7.5 two-pass ordering,
    before the canonical transform exists), so Y is negated here to recover
    "higher is larger". The canonical transform is unaffected: it derives its
    own up axis from the shoulder-hip vector.
    """
    return -wrist_xyz[:, 1]


def _nan_safe(window: np.ndarray, reducer) -> int | None:
    """argmax/argmin ignoring NaN (tracking gaps inside the search window).
    None when the window is entirely NaN.

    Plain np.argmax/argmin mis-handle NaN: the NaN wins the comparison chain
    and is returned as the extreme, and because `NaN < threshold` is False any
    downstream threshold check silently passes. Section 7.6 assumes a gap-free
    series; filtering only bounds missing data over the swing span, so a given
    sub-window can still contain gaps.
    """
    if np.all(np.isnan(window)):
        return None
    return int(reducer(window))


def takeaway_speed_threshold(v: np.ndarray) -> float:
    """The speed a lead wrist must exceed to count as having started moving.

    SPEC DEVIATION (Section 7.6 step 1): see TAKEAWAY_SPEED_FRACTION_OF_PEAK.
    The spec's flat 0.4 m/s is unreachable on slow-motion footage, so the
    threshold is capped by a fraction of the clip's own peak speed. A wholly
    stationary clip has peak 0, giving threshold 0, which the strict `>`
    comparison still never passes -- so "nothing moved" remains no_full_swing.
    """
    finite = v[np.isfinite(v)]
    if len(finite) == 0:
        return config.TAKEAWAY_SPEED_THRESHOLD_MPS
    relative = config.TAKEAWAY_SPEED_FRACTION_OF_PEAK * float(finite.max())
    return min(config.TAKEAWAY_SPEED_THRESHOLD_MPS, relative)


def _net_displacement_floor(wrist_xyz: np.ndarray) -> float:
    """How far the lead wrist must actually travel for motion to count as a
    takeaway rather than a waggle, in metres.

    Scaled off the wrist's own total range along its widest axis, so it is
    invariant to both playback rate and the golfer's size/distance from camera.
    """
    extent = 0.0
    for axis in range(3):
        column = wrist_xyz[:, axis]
        if np.all(np.isnan(column)):
            continue
        extent = max(extent, float(np.nanmax(column) - np.nanmin(column)))
    return config.TAKEAWAY_MIN_NET_DISPLACEMENT_FRACTION * extent


def detect_takeaway_start(wrist_xyz: np.ndarray, fps: float) -> int:
    """Section 7.6 step 1: first frame in [0, 0.4N] where speed exceeds the
    threshold for 3 consecutive frames.

    SPEC DEVIATION: the speed run must additionally be confirmed by net
    displacement -- see TAKEAWAY_MIN_NET_DISPLACEMENT_FRACTION. Without it the
    lowered (playback-invariant) speed trigger fires on address waggle: on the
    reference clip it picked frame 1, roughly two seconds before the lead wrist
    left address, and because the turn metrics are differences measured against
    the address frame, that moved shoulder_turn_top by a factor of three.
    """
    n = len(wrist_xyz)
    v = wrist_speed(wrist_xyz, fps)
    threshold = takeaway_speed_threshold(v)
    limit = int(config.TAKEAWAY_SEARCH_FRACTION * n)
    consecutive = config.TAKEAWAY_CONSECUTIVE_FRAMES
    window = max(1, round(config.TAKEAWAY_CONFIRM_WINDOW_FRACTION * n))
    floor = _net_displacement_floor(wrist_xyz)

    for t in range(0, limit + 1):
        if t + consecutive > n or not np.all(v[t : t + consecutive] > threshold):
            continue
        net = wrist_xyz[min(n - 1, t + window)] - wrist_xyz[t]
        # Missing data cannot disconfirm; fall back to the speed run alone.
        if np.isnan(net).any() or float(np.linalg.norm(net)) >= floor:
            return t
    raise PipelineError("no_full_swing", "no swing motion detected in the first part of the clip")


def detect_address(wrist_xyz: np.ndarray, fps: float) -> int:
    """Section 7.6 step 2: address = max(0, takeaway_start - round(0.4 * fps))."""
    takeaway_start = detect_takeaway_start(wrist_xyz, fps)
    return max(0, takeaway_start - round(config.ADDRESS_LOOKBACK_S * fps))


def detect_top(height: np.ndarray, address: int) -> int:
    """Section 7.6 step 4: the lead wrist is highest at the top of the backswing.

    Searched over [address + 5, N - 1 - 3] rather than the spec's
    [address + 5, impact - 3]: see detect_keyframes for why top is now found
    before impact instead of after it.
    """
    start = address + config.TOP_WINDOW_ADDRESS_MARGIN
    end = len(height) - 1 - config.IMPACT_MIN_FRAMES_AFTER_TOP
    window = height[start : end + 1]
    if len(window) < config.MIN_ARGMAX_WINDOW_FRAMES:
        raise PipelineError("no_full_swing", "top-of-backswing search window is too small")
    local_idx = _nan_safe(window, np.nanargmax)
    if local_idx is None:
        raise PipelineError("no_full_swing", "top-of-backswing window has no usable tracking data")
    return start + local_idx


def detect_impact(height: np.ndarray, top: int) -> int:
    """Impact is where the lead wrist comes back down to its lowest point
    after the top -- the hands returning to the ball.

    SPEC DEVIATION (Section 7.6 step 3): the spec locates impact as the
    lead-wrist speed peak within a fixed [0.4N, 0.9N] slice of the clip. Two
    problems showed up on real footage: the fixed slice assumes the swing sits
    in a particular part of the clip (impact fell at 96% of the reference
    clip, outside the window), and the speed peak is easily won by tracking
    jitter during the fast blurred part of the backswing rather than by
    impact itself. Wrist height returning to its minimum is a geometric
    property of the swing -- independent of where the swing sits in the clip,
    of the playback rate, and of frame-to-frame speed noise.
    """
    start = top + config.IMPACT_MIN_FRAMES_AFTER_TOP
    window = height[start:]
    if len(window) < config.MIN_ARGMAX_WINDOW_FRAMES:
        raise PipelineError("no_full_swing", "impact search window is too small")
    local_idx = _nan_safe(window, np.nanargmin)
    if local_idx is None:
        raise PipelineError("no_full_swing", "impact window has no usable tracking data")
    return start + local_idx


def detect_keyframes(wrist_xyz: np.ndarray, fps: float) -> dict:
    """Section 7.6, full: address, top, impact. Raises PipelineError('no_full_swing', ...).

    Ordering deviates from the spec, which finds impact (step 3) before top
    (step 4) and uses impact to bound the top search. Top is the more robust
    of the two -- a clear height maximum -- so it is found first and used to
    bound impact, which removes the spec's dependence on a fixed-fraction
    search window. Physically the order is also the true one: top precedes
    impact.
    """
    n = len(wrist_xyz)
    height = wrist_height(wrist_xyz)

    address = detect_address(wrist_xyz, fps)
    top = detect_top(height, address)
    impact = detect_impact(height, top)

    address_height = np.nanmedian(height[max(0, address - 2) : address + 3])
    if np.isnan(address_height):
        raise PipelineError("no_full_swing", "no usable lead-wrist tracking at address")
    if height[top] - address_height < config.MIN_BACKSWING_RISE_M:
        raise PipelineError("no_full_swing", "no backswing detected: the lead wrist never rises")

    if not (address < top < impact):
        raise PipelineError("no_full_swing", "detected keyframes are out of order")
    return {"address": address, "top": top, "impact": impact}
