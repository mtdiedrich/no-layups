from no_layups.pipeline import compare, metrics


def _metrics_dict(values: dict) -> dict:
    return {
        key: (None if v is None else {"value": v, "unit": metrics.METRIC_UNITS[key]})
        for key, v in values.items()
    }


def _swing(frame_count: int, keyframes: dict, metric_values: dict) -> dict:
    return {
        "meta": {"frame_count": frame_count},
        "keyframes": keyframes,
        "metrics": _metrics_dict(metric_values),
        "trajectories": {
            "phase_percent": list(range(101)),
            "shoulder_turn": [0.0] * 101,
            "spine_tilt": [0.0] * 101,
        },
    }


BASE_VALUES = {
    "shoulder_turn_top": 95.0,
    "hip_turn_top": 44.0,
    "x_factor": 46.0,
    "spine_tilt_address": None,
    "spine_tilt_impact": 28.0,
    "lead_elbow_top": 171.0,
    "trail_knee_top": 158.0,
    "head_sway_top": 4.0,
}


def _entry(comparison: dict, key: str) -> dict:
    return next(m for m in comparison["metrics"] if m["key"] == key)


def test_delta_basis_rating_bands():
    reference = _swing(100, {"address": 0, "top": 40, "impact": 65}, {**BASE_VALUES, "shoulder_turn_top": 100.0})

    good = _swing(100, {"address": 0, "top": 40, "impact": 65}, {**BASE_VALUES, "shoulder_turn_top": 95.0})
    ok = _swing(100, {"address": 0, "top": 40, "impact": 65}, {**BASE_VALUES, "shoulder_turn_top": 115.0})
    bad = _swing(100, {"address": 0, "top": 40, "impact": 65}, {**BASE_VALUES, "shoulder_turn_top": 125.0})

    assert _entry(compare.build_comparison(good, reference), "shoulder_turn_top")["rating"] == "good"
    assert _entry(compare.build_comparison(ok, reference), "shoulder_turn_top")["rating"] == "ok"
    assert _entry(compare.build_comparison(bad, reference), "shoulder_turn_top")["rating"] == "attention"


def test_delta_is_user_minus_reference():
    reference = _swing(100, {"address": 0, "top": 40, "impact": 65}, BASE_VALUES)
    user = _swing(100, {"address": 0, "top": 40, "impact": 65}, {**BASE_VALUES, "hip_turn_top": 50.0})

    entry = _entry(compare.build_comparison(user, reference), "hip_turn_top")
    assert entry["user"] == 50.0
    assert entry["reference"] == 44.0
    assert entry["delta"] == 6.0


def test_head_sway_rated_on_absolute_user_value_not_delta():
    """Section 9.3: head_sway_top's basis is absolute, not vs reference — a
    user who sways less than the (large) reference value should still be
    rated on their own absolute sway, not on how much smaller it is."""
    reference = _swing(100, {"address": 0, "top": 40, "impact": 65}, {**BASE_VALUES, "head_sway_top": 20.0})
    user = _swing(100, {"address": 0, "top": 40, "impact": 65}, {**BASE_VALUES, "head_sway_top": 7.0})

    entry = _entry(compare.build_comparison(user, reference), "head_sway_top")
    assert entry["delta"] == -13.0
    assert entry["rating"] == "ok"  # 7cm: good<=5, ok<=10 -- not "good" despite a large negative delta


def test_missing_value_on_either_side_yields_null_rating_but_keeps_the_row():
    reference = _swing(100, {"address": 0, "top": 40, "impact": 65}, BASE_VALUES)
    user = _swing(100, {"address": 0, "top": 40, "impact": 65}, {**BASE_VALUES, "trail_knee_top": None})

    entry = _entry(compare.build_comparison(user, reference), "trail_knee_top")
    assert entry["user"] is None
    assert entry["reference"] == 158.0
    assert entry["delta"] is None
    assert entry["rating"] is None
    assert entry["label"] == "Trail knee at top"


def test_spine_tilt_address_is_always_null_on_both_sides():
    """metrics.py permanently reports spine_tilt_address as null (see its
    SPEC DEVIATION comment); compare.py must not crash on that and must not
    fabricate a rating for it."""
    reference = _swing(100, {"address": 0, "top": 40, "impact": 65}, BASE_VALUES)
    user = _swing(100, {"address": 0, "top": 40, "impact": 65}, BASE_VALUES)

    entry = _entry(compare.build_comparison(user, reference), "spine_tilt_address")
    assert entry["user"] is None
    assert entry["reference"] is None
    assert entry["rating"] is None


def test_all_eight_metrics_present_in_order():
    reference = _swing(100, {"address": 0, "top": 40, "impact": 65}, BASE_VALUES)
    user = _swing(100, {"address": 0, "top": 40, "impact": 65}, BASE_VALUES)

    keys = [m["key"] for m in compare.build_comparison(user, reference)["metrics"]]
    assert keys == list(compare.METRIC_LABELS.keys())


def test_phase_to_frame_inverts_phase_of_frame():
    """With keyframes placed exactly at the phase-percent anchors on a
    0..100-frame clip, phase_of_frame is the identity map, so its inverse
    (phase_to_frame) must be too."""
    keyframes = {"address": 0, "top": 40, "impact": 65}
    swing = _swing(101, keyframes, BASE_VALUES)

    comparison = compare.build_comparison(swing, swing)
    assert comparison["phase_to_frame"]["user"] == list(range(101))
    assert comparison["phase_to_frame"]["reference"] == list(range(101))


def test_trajectories_copied_verbatim():
    reference = _swing(100, {"address": 0, "top": 40, "impact": 65}, BASE_VALUES)
    reference["trajectories"]["shoulder_turn"] = [1.0] * 101
    user = _swing(100, {"address": 0, "top": 40, "impact": 65}, BASE_VALUES)
    user["trajectories"]["shoulder_turn"] = [2.0] * 101

    trajectories = compare.build_comparison(user, reference)["trajectories"]
    assert trajectories["user"]["shoulder_turn"] == [2.0] * 101
    assert trajectories["reference"]["shoulder_turn"] == [1.0] * 101
    assert trajectories["phase_percent"] == list(range(101))
