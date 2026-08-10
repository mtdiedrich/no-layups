import numpy as np


def _mid(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a + b) / 2.0


def _normalize(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def transform(
    xyz_by_joint: dict[str, np.ndarray], address_idx: int, handedness: str
) -> dict[str, np.ndarray]:
    """Section 6.1 / 7.5: canonical frame anchored at the address joint positions.

    RESOLVED SPEC AMBIGUITY: +X is built from the fixed anatomical hips
    (left - right), never handedness-adjusted. This is the only reading under
    which the Section 6.2 left-handed mirror does anything meaningful: if +X
    were instead built from lead/trail (handedness-aware), lead would already
    land on +X for both handedness cases by construction, and the mirror step
    would flip it back off for lefties only -- undoing the one thing that was
    already consistent. Building +X from the fixed anatomical hips makes the
    mirror the sole handedness-dependent step, matches Section 13.1's test
    ("anatomical right hip has x > 0" for a mirrored left-handed swing), and
    is what's implemented and tested here. lead()/trail() remain correct
    everywhere else (metrics, the lead-wrist speed series) -- only this axis
    construction is anatomical-fixed rather than lead/trail-relative.
    """
    left_shoulder = xyz_by_joint["left_shoulder"][address_idx]
    right_shoulder = xyz_by_joint["right_shoulder"][address_idx]
    left_hip = xyz_by_joint["left_hip"][address_idx]
    right_hip = xyz_by_joint["right_hip"][address_idx]

    up = _normalize(_mid(left_shoulder, right_shoulder) - _mid(left_hip, right_hip))

    x_raw = left_hip - right_hip
    x_axis = _normalize(x_raw - np.dot(x_raw, up) * up)

    z_axis = np.cross(x_axis, up)
    assert abs(np.linalg.norm(z_axis) - 1.0) < 1e-6

    origin = _mid(left_hip, right_hip)
    basis = np.stack([x_axis, up, z_axis])  # rows: x, y, z basis vectors

    out: dict[str, np.ndarray] = {}
    for joint, arr in xyz_by_joint.items():
        transformed = (arr - origin) @ basis.T
        if handedness == "left":
            transformed[:, 0] *= -1
        out[joint] = transformed
    return out
