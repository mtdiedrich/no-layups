import numpy as np
import pytest

from no_layups.pipeline.canonical import transform


def _synthetic_frames():
    """A single-frame address posture: level hips/shoulders, standing upright."""
    joints = {
        "nose": [[0.0, 1.6, -0.1]],
        "left_shoulder": [[0.2, 1.4, 0.0]],
        "right_shoulder": [[-0.2, 1.4, 0.0]],
        "left_elbow": [[0.25, 1.1, 0.0]],
        "right_elbow": [[-0.25, 1.1, 0.0]],
        "left_wrist": [[0.2, 0.8, 0.0]],
        "right_wrist": [[-0.2, 0.8, 0.0]],
        "left_hip": [[0.1, 0.9, 0.0]],
        "right_hip": [[-0.1, 0.9, 0.0]],
        "left_knee": [[0.1, 0.5, 0.0]],
        "right_knee": [[-0.1, 0.5, 0.0]],
        "left_ankle": [[0.1, 0.1, 0.0]],
        "right_ankle": [[-0.1, 0.1, 0.0]],
    }
    return {j: np.array(v, dtype=float) for j, v in joints.items()}


def test_address_mid_hip_is_origin():
    frames = _synthetic_frames()
    out = transform(frames, address_idx=0, handedness="right")
    mid_hip = (out["left_hip"][0] + out["right_hip"][0]) / 2
    assert np.allclose(mid_hip, [0.0, 0.0, 0.0], atol=1e-9)


def test_address_mid_shoulder_is_above_origin_on_y_axis():
    frames = _synthetic_frames()
    out = transform(frames, address_idx=0, handedness="right")
    mid_shoulder = (out["left_shoulder"][0] + out["right_shoulder"][0]) / 2
    assert mid_shoulder[0] == pytest.approx(0.0, abs=1e-9)
    assert mid_shoulder[2] == pytest.approx(0.0, abs=1e-9)
    assert mid_shoulder[1] > 0


def test_address_hip_line_is_level_and_lead_hip_is_positive_x():
    frames = _synthetic_frames()
    out = transform(frames, address_idx=0, handedness="right")
    # right-handed: lead hip = left_hip
    assert out["left_hip"][0][1] == pytest.approx(out["right_hip"][0][1], abs=1e-9)
    assert out["left_hip"][0][0] > out["right_hip"][0][0]


def test_left_handed_mirror_sign_convention():
    """SPEC DEVIATION (see canonical.py docstring): Section 7.5's handedness-aware
    formula plus the 6.2 mirror puts the anatomical LEFT hip at x > 0 after
    mirroring a left-handed swing -- the same convention an unmirrored
    right-handed swing has. Section 13.1's prose states the opposite; this
    test asserts the formula's actual, self-consistent behavior."""
    frames = _synthetic_frames()
    out = transform(frames, address_idx=0, handedness="left")
    assert out["left_hip"][0][0] > 0
    assert out["right_hip"][0][0] < 0
