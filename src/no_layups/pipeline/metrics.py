import numpy as np

from .. import config

METRIC_UNITS = {
    "shoulder_turn_top": "deg",
    "hip_turn_top": "deg",
    "x_factor": "deg",
    "spine_tilt_address": "deg",
    "spine_tilt_impact": "deg",
    "lead_elbow_top": "deg",
    "trail_knee_top": "deg",
    "head_sway_top": "cm",
}


def angle_at(a, b, c) -> float:
    """Section 6.3: angle at joint B formed by A-B-C, 0..180 degrees."""
    a, b, c = np.asarray(a, dtype=float), np.asarray(b, dtype=float), np.asarray(c, dtype=float)
    v1 = a - b
    v2 = c - b
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos = min(1.0, max(-1.0, cos))
    return float(np.degrees(np.arccos(cos)))


def turn_angle(line_addr, line_top) -> float:
    """Section 8.1: rotation between two lines projected onto the ground plane (Y dropped)."""
    a = np.array([line_addr[0], line_addr[2]], dtype=float)
    b = np.array([line_top[0], line_top[2]], dtype=float)
    cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    cos = min(1.0, max(-1.0, cos))
    return float(np.degrees(np.arccos(cos)))


def spine_tilt(mid_shoulder, mid_hip) -> float:
    """Section 8.2: lean away from vertical in the frontal plane."""
    t = np.asarray(mid_shoulder, dtype=float) - np.asarray(mid_hip, dtype=float)
    return float(np.degrees(np.arctan2(abs(t[0]), t[1])))


def _get(frames: dict[str, np.ndarray], joint: str, idx: int):
    p = frames[joint][idx]
    return None if np.isnan(p).any() else p


def _mid(a, b):
    if a is None or b is None:
        return None
    return (a + b) / 2


def compute_metrics(frames: dict[str, np.ndarray], keyframes: dict, handedness: str) -> dict:
    """Section 8: the 8 swing metrics. A metric is null if a required joint
    is missing at the needed frame."""
    address, top, impact = keyframes["address"], keyframes["top"], keyframes["impact"]
    lead = lambda name: config.lead(name, handedness)  # noqa: E731
    trail = lambda name: config.trail(name, handedness)  # noqa: E731

    ls_a, rs_a = _get(frames, lead("shoulder"), address), _get(frames, trail("shoulder"), address)
    ls_t, rs_t = _get(frames, lead("shoulder"), top), _get(frames, trail("shoulder"), top)
    lh_a, rh_a = _get(frames, lead("hip"), address), _get(frames, trail("hip"), address)
    lh_t, rh_t = _get(frames, lead("hip"), top), _get(frames, trail("hip"), top)

    shoulder_turn = (
        turn_angle(ls_a - rs_a, ls_t - rs_t)
        if all(v is not None for v in (ls_a, rs_a, ls_t, rs_t))
        else None
    )
    hip_turn = (
        turn_angle(lh_a - rh_a, lh_t - rh_t)
        if all(v is not None for v in (lh_a, rh_a, lh_t, rh_t))
        else None
    )
    x_factor = None if shoulder_turn is None or hip_turn is None else shoulder_turn - hip_turn

    def spine_tilt_at(idx):
        ms = _mid(_get(frames, "left_shoulder", idx), _get(frames, "right_shoulder", idx))
        mh = _mid(_get(frames, "left_hip", idx), _get(frames, "right_hip", idx))
        return None if ms is None or mh is None else spine_tilt(ms, mh)

    # SPEC DEVIATION (Section 8.2): spine_tilt_address is reported as null, not
    # as a number. Section 6.1 defines canonical +Y *as* normalize(mid(shoulders)
    # - mid(hips)) at address, so in canonical space the address trunk vector is
    # exactly the up axis and Section 8.2's atan2(|t.x|, t.y) is identically 0.0
    # for every clip that has ever been or will ever be processed -- measured on
    # the reference clip, the address trunk is [-0.000000, 0.483705, 0.000000].
    #
    # Reporting 0.0 is worse than reporting nothing: Section 9.3 rates this
    # metric at |delta| <= 3 deg for "good", and since the reference is anchored
    # the same way its address tilt is also exactly 0.0, so the delta is
    # identically 0 and every golfer scores a permanent green dot on a quantity
    # that was never measured. Absolute spine tilt is simply not recoverable
    # from canonical coordinates; it would need a gravity vector, and Section 15
    # pitfall 9 explicitly warns against trusting MediaPipe's axes to supply one.
    #
    # null is an already-supported state: Section 8 says a metric whose inputs
    # are unavailable is null and its rating is omitted from the comparison, so
    # this propagates correctly through compare.py and the Section 11.5 metrics
    # panel without touching the Section 5.2 schema.
    #
    # spine_tilt_impact is kept and is genuinely informative -- in canonical
    # space it reads as the change in trunk lean between address and impact,
    # which is the coachable quantity (secondary tilt) rather than an absolute.
    spine_tilt_address = None

    ls_top, le_top, lw_top = (
        _get(frames, lead("shoulder"), top),
        _get(frames, lead("elbow"), top),
        _get(frames, lead("wrist"), top),
    )
    lead_elbow = (
        angle_at(ls_top, le_top, lw_top)
        if all(v is not None for v in (ls_top, le_top, lw_top))
        else None
    )

    rh_top, rk_top, ra_top = (
        _get(frames, trail("hip"), top),
        _get(frames, trail("knee"), top),
        _get(frames, trail("ankle"), top),
    )
    trail_knee = (
        angle_at(rh_top, rk_top, ra_top)
        if all(v is not None for v in (rh_top, rk_top, ra_top))
        else None
    )

    # SPEC DEVIATION (Section 8.3): the spec measures head sway as the full
    # horizontal displacement norm(nose_top.xz - nose_addr.xz). The canonical Z
    # term is depth, and monocular depth is the least reliable channel MediaPipe
    # produces -- on the reference clip it contributed 37.9 cm of a 39.4 cm
    # total, against 10.7 cm from X, and MediaPipe's per-frame Z jitter measures
    # 2-3x its X/Y jitter. Section 9.3 rates head_sway_top on *absolute*
    # thresholds (good <= 5 cm, ok <= 10 cm) rather than against the reference,
    # so unlike the turn metrics it gets no cancellation from the comparison and
    # a depth-inflated value lands on "attention" for every golfer regardless of
    # how still their head actually was.
    #
    # Only the X component is used. Canonical +X is derived from the body (the
    # address hip line, running roughly toward the target -- Section 6.1), not
    # from the camera, so this is the target-line direction that golf head-sway
    # actually refers to, and it stays the same axis whether the clip is shot
    # face-on or down-the-line.
    nose_a, nose_t = _get(frames, "nose", address), _get(frames, "nose", top)
    head_sway = (
        abs(float(nose_t[0] - nose_a[0])) * 100
        if nose_a is not None and nose_t is not None
        else None
    )

    values = {
        "shoulder_turn_top": shoulder_turn,
        "hip_turn_top": hip_turn,
        "x_factor": x_factor,
        "spine_tilt_address": spine_tilt_address,
        "spine_tilt_impact": spine_tilt_at(impact),
        "lead_elbow_top": lead_elbow,
        "trail_knee_top": trail_knee,
        "head_sway_top": head_sway,
    }
    return {
        key: (None if value is None else {"value": round(value, 6), "unit": METRIC_UNITS[key]})
        for key, value in values.items()
    }


def compute_trajectories(frames: dict[str, np.ndarray], keyframes: dict, handedness: str) -> dict:
    """Section 8.4: shoulder_turn_rel(t) and spine_tilt(t), resampled onto phase percent 0..100."""
    address, top, impact = keyframes["address"], keyframes["top"], keyframes["impact"]
    lead_s, trail_s = config.lead("shoulder", handedness), config.trail("shoulder", handedness)
    n = len(frames["nose"])
    last = n - 1

    line_addr = None
    la, ta = _get(frames, lead_s, address), _get(frames, trail_s, address)
    if la is not None and ta is not None:
        line_addr = la - ta

    shoulder_turn = np.full(n, np.nan)
    spine_tilt_series = np.full(n, np.nan)
    for t in range(n):
        if line_addr is not None:
            lt, tt = _get(frames, lead_s, t), _get(frames, trail_s, t)
            if lt is not None and tt is not None:
                shoulder_turn[t] = turn_angle(line_addr, lt - tt)
        ms = _mid(_get(frames, "left_shoulder", t), _get(frames, "right_shoulder", t))
        mh = _mid(_get(frames, "left_hip", t), _get(frames, "right_hip", t))
        if ms is not None and mh is not None:
            spine_tilt_series[t] = spine_tilt(ms, mh)

    phase_of_frame = np.interp(
        np.arange(n),
        [address, top, impact, last],
        [
            config.PHASE_PERCENT_ADDRESS,
            config.PHASE_PERCENT_TOP,
            config.PHASE_PERCENT_IMPACT,
            config.PHASE_PERCENT_LAST_FRAME,
        ],
    )
    grid = np.arange(101)

    def resample(series):
        valid = ~np.isnan(series)
        if valid.sum() < 2:
            return [None] * 101
        return [float(v) for v in np.interp(grid, phase_of_frame[valid], series[valid])]

    return {
        "phase_percent": grid.tolist(),
        "shoulder_turn": resample(shoulder_turn),
        "spine_tilt": resample(spine_tilt_series),
    }
