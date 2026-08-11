import math

import numpy as np
import pytest

from no_layups.pipeline.metrics import angle_at, compute_metrics, spine_tilt, turn_angle


def test_angle_at_90_degrees():
    assert angle_at([1, 0, 0], [0, 0, 0], [0, 1, 0]) == pytest.approx(90.0, abs=1e-6)


def test_angle_at_180_degrees():
    assert angle_at([1, 0, 0], [0, 0, 0], [-1, 0, 0]) == pytest.approx(180.0, abs=1e-6)


def test_turn_angle_90_degree_rotation_about_y():
    line_addr = [1, 0, 0]
    line_top = [0, 0, -1]  # [1,0,0] rotated 90 degrees about Y
    assert turn_angle(line_addr, line_top) == pytest.approx(90.0, abs=1e-6)


def test_spine_tilt_straight_up_is_zero():
    assert spine_tilt([0, 1, 0], [0, 0, 0]) == pytest.approx(0.0, abs=1e-6)


def test_spine_tilt_30_degrees_along_x():
    trunk = [math.sin(math.radians(30)), math.cos(math.radians(30)), 0]
    assert spine_tilt(trunk, [0, 0, 0]) == pytest.approx(30.0, abs=1e-6)


def test_straight_lead_arm_is_180():
    shoulder = np.array([0.2, 1.4, 0.0])
    wrist = np.array([0.2, 0.8, 0.0])
    elbow = (shoulder + wrist) / 2
    assert angle_at(shoulder, elbow, wrist) == pytest.approx(180.0, abs=1e-6)


def _address_top_impact_frames():
    """3-frame synthetic canonical data: address (frame 0), top (frame 1,
    shoulders/hips turned 45 degrees about Y), impact (frame 2, back near
    address)."""
    address = {
        "nose": [0.0, 1.6, -0.1],
        "left_shoulder": [0.2, 1.4, 0.0],
        "right_shoulder": [-0.2, 1.4, 0.0],
        "left_elbow": [0.25, 1.1, 0.0],
        "right_elbow": [-0.25, 1.1, 0.0],
        "left_wrist": [0.2, 0.8, 0.0],
        "right_wrist": [-0.2, 0.8, 0.0],
        "left_hip": [0.1, 0.9, 0.0],
        "right_hip": [-0.1, 0.9, 0.0],
        "left_knee": [0.1, 0.5, 0.0],
        "right_knee": [-0.1, 0.5, 0.0],
        "left_ankle": [0.1, 0.1, 0.0],
        "right_ankle": [-0.1, 0.1, 0.0],
    }

    def rotate_y(p, deg):
        rad = math.radians(deg)
        x, y, z = p
        return [x * math.cos(rad) + z * math.sin(rad), y, -x * math.sin(rad) + z * math.cos(rad)]

    top = {j: rotate_y(p, 45) for j, p in address.items()}
    # straighten the lead (left, right-handed default) arm at top for a clean angle_at check
    top["left_elbow"] = [(a + b) / 2 for a, b in zip(top["left_shoulder"], top["left_wrist"])]
    impact = {j: rotate_y(p, 5) for j, p in address.items()}

    frames = {
        joint: np.array([address[joint], top[joint], impact[joint]], dtype=float)
        for joint in address
    }
    return frames


def test_compute_metrics_x_factor_and_plausible_values():
    frames = _address_top_impact_frames()
    keyframes = {"address": 0, "top": 1, "impact": 2}
    result = compute_metrics(frames, keyframes, handedness="right")

    assert result["shoulder_turn_top"]["value"] == pytest.approx(45.0, abs=1e-6)
    assert result["hip_turn_top"]["value"] == pytest.approx(45.0, abs=1e-6)
    assert result["x_factor"]["value"] == pytest.approx(0.0, abs=1e-6)
    assert result["lead_elbow_top"]["value"] == pytest.approx(180.0, abs=1e-6)
    assert result["head_sway_top"]["unit"] == "cm"


def test_spine_tilt_address_is_measured_not_nulled():
    """This metric used to be forced to null. Under the spec's Section 6.1
    frame canonical +Y WAS the address trunk vector, so Section 8.2's formula
    returned exactly 0.0 for every clip and reporting nothing was the honest
    choice. canonical.transform now anchors +Y to world up, so the metric is a
    real measurement again.

    This fixture's trunk is vertical, so 0.0 is the correct answer here rather
    than a degeneracy -- test_spine_tilt_address_tracks_actual_lean shows it
    moves when the trunk does."""
    frames = _address_top_impact_frames()
    result = compute_metrics(frames, {"address": 0, "top": 1, "impact": 2}, handedness="right")

    assert result["spine_tilt_address"] is not None
    assert result["spine_tilt_address"]["value"] == pytest.approx(0.0, abs=1e-6)
    assert result["spine_tilt_impact"] is not None


def test_spine_tilt_address_tracks_actual_lean():
    """Section 8.2: tilt = atan2(|t.x|, t.y) on the trunk vector, i.e. lateral
    lean in the frontal plane. Tilt the shoulders 20 degrees along +X over the
    hips and the metric must report 20 degrees."""
    frames = _address_top_impact_frames()
    mid_hip_y = (frames["left_hip"][0][1] + frames["right_hip"][0][1]) / 2
    mid_shoulder_y = (frames["left_shoulder"][0][1] + frames["right_shoulder"][0][1]) / 2
    offset = (mid_shoulder_y - mid_hip_y) * math.tan(math.radians(20.0))
    for joint in ("left_shoulder", "right_shoulder"):
        frames[joint][0][0] += offset

    result = compute_metrics(frames, {"address": 0, "top": 1, "impact": 2}, handedness="right")
    assert result["spine_tilt_address"]["value"] == pytest.approx(20.0, abs=1e-6)


def test_head_sway_ignores_the_depth_axis():
    """Section 8.3's Z term is monocular depth and swamps the real signal, so
    head sway is measured along canonical +X (the body-derived target line)."""
    frames = _address_top_impact_frames()
    keyframes = {"address": 0, "top": 1, "impact": 2}

    addr_x, top_y = frames["nose"][0][0], frames["nose"][1][1]

    # Head pinned on the target line, displaced only in depth -> no sway at all.
    frames["nose"][1] = [addr_x, top_y, frames["nose"][0][2] + 0.5]
    depth_only = compute_metrics(frames, keyframes, handedness="right")["head_sway_top"]["value"]
    assert depth_only == pytest.approx(0.0, abs=1e-9)

    # A real 10 cm move along the target line reads as exactly 10 cm.
    frames["nose"][1] = [addr_x + 0.10, top_y, frames["nose"][0][2] + 0.5]
    lateral = compute_metrics(frames, keyframes, handedness="right")["head_sway_top"]["value"]
    assert lateral == pytest.approx(10.0, abs=1e-6)


def test_compute_metrics_missing_joint_is_null():
    frames = _address_top_impact_frames()
    frames["left_elbow"][1] = [np.nan, np.nan, np.nan]  # missing at top
    keyframes = {"address": 0, "top": 1, "impact": 2}
    result = compute_metrics(frames, keyframes, handedness="right")

    assert result["lead_elbow_top"] is None
    assert result["shoulder_turn_top"] is not None  # unaffected metric stays populated
