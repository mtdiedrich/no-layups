import numpy as np

from . import metrics

# Section 5.3 / 11.5 — human labels for the fixed set of 8 metrics, in display order.
METRIC_LABELS = {
    "shoulder_turn_top": "Shoulder turn at top",
    "hip_turn_top": "Hip turn at top",
    "x_factor": "X-factor",
    "spine_tilt_address": "Spine tilt at address",
    "spine_tilt_impact": "Spine tilt at impact",
    "lead_elbow_top": "Lead elbow at top",
    "trail_knee_top": "Trail knee at top",
    "head_sway_top": "Head sway at top",
}

# Section 9.3: (good, ok, basis). basis "delta" rates |user - reference|;
# head_sway_top's basis is "absolute" — it rates the user's own value, since
# Section 9.3 marks it "absolute, not vs reference".
RATING_THRESHOLDS = {
    "shoulder_turn_top": (10, 20, "delta"),
    "hip_turn_top": (10, 20, "delta"),
    "x_factor": (8, 15, "delta"),
    "spine_tilt_address": (3, 6, "delta"),
    "spine_tilt_impact": (4, 8, "delta"),
    "lead_elbow_top": (10, 20, "delta"),
    "trail_knee_top": (10, 15, "delta"),
    "head_sway_top": (5, 10, "absolute"),
}


def _rating(key: str, value, delta) -> str | None:
    if value is None:
        return None
    good, ok, basis = RATING_THRESHOLDS[key]
    if basis == "absolute":
        magnitude = abs(value)
    else:
        if delta is None:
            return None
        magnitude = abs(delta)
    if magnitude <= good:
        return "good"
    if magnitude <= ok:
        return "ok"
    return "attention"


def _metric_entries(user_metrics: dict, reference_metrics: dict) -> list[dict]:
    entries = []
    for key, label in METRIC_LABELS.items():
        user_entry = user_metrics.get(key)
        ref_entry = reference_metrics.get(key)
        user_value = user_entry["value"] if user_entry else None
        ref_value = ref_entry["value"] if ref_entry else None
        unit = (user_entry or ref_entry or {}).get("unit", metrics.METRIC_UNITS[key])
        delta = None if user_value is None or ref_value is None else user_value - ref_value
        entries.append(
            {
                "key": key,
                "label": label,
                "unit": unit,
                "user": user_value,
                "reference": ref_value,
                "delta": None if delta is None else round(delta, 6),
                "rating": _rating(key, user_value, delta),
            }
        )
    return entries


def _phase_to_frame(frame_count: int, keyframes: dict) -> list[int]:
    """Section 8.4 last paragraph: inverts phase_of_frame — for each phase
    percent 0..100, the nearest frame index."""
    phase = metrics.phase_of_frame(frame_count, keyframes)
    frame_index = np.arange(frame_count)
    grid = np.arange(101)
    return [int(round(v)) for v in np.interp(grid, phase, frame_index)]


def build_comparison(user_swing: dict, reference_swing: dict) -> dict:
    """Section 9.2: compare a completed user swing against the reference,
    producing the comparison.json structure (Section 5.3)."""
    user_traj = user_swing["trajectories"]
    ref_traj = reference_swing["trajectories"]

    return {
        "metrics": _metric_entries(user_swing["metrics"], reference_swing["metrics"]),
        "trajectories": {
            "phase_percent": user_traj["phase_percent"],
            "user": {
                "shoulder_turn": user_traj["shoulder_turn"],
                "spine_tilt": user_traj["spine_tilt"],
            },
            "reference": {
                "shoulder_turn": ref_traj["shoulder_turn"],
                "spine_tilt": ref_traj["spine_tilt"],
            },
        },
        "phase_to_frame": {
            "user": _phase_to_frame(user_swing["meta"]["frame_count"], user_swing["keyframes"]),
            "reference": _phase_to_frame(
                reference_swing["meta"]["frame_count"], reference_swing["keyframes"]
            ),
        },
    }
