import numpy as np

from .. import config


def _mid(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a + b) / 2.0


def _normalize(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def transform(
    xyz_by_joint: dict[str, np.ndarray], address_idx: int, handedness: str
) -> dict[str, np.ndarray]:
    """Section 6.1 / 7.5: canonical frame anchored at the address joint positions.

    NOTE — SPEC DEVIATION (documented, not silent): Section 7.5's formula uses
    handedness-aware lead/trail hips for +X (matching Section 6.1's "trail hip
    toward lead hip" definition and the 6.2 lead()/trail() helpers). Combined
    with the 6.2 left-handed mirror, this makes the anatomical LEFT hip end up
    at x > 0 after mirroring a left-handed swing — the same convention a
    right-handed swing has unmirrored. Section 13.1's prose describes the
    opposite ("anatomical right hip has x > 0"); we follow the normative 7.5
    formula and flag the discrepancy for the user to confirm.
    """
    left_shoulder = xyz_by_joint["left_shoulder"][address_idx]
    right_shoulder = xyz_by_joint["right_shoulder"][address_idx]
    left_hip = xyz_by_joint["left_hip"][address_idx]
    right_hip = xyz_by_joint["right_hip"][address_idx]

    up = _normalize(_mid(left_shoulder, right_shoulder) - _mid(left_hip, right_hip))

    lead_hip = xyz_by_joint[config.lead("hip", handedness)][address_idx]
    trail_hip = xyz_by_joint[config.trail("hip", handedness)][address_idx]
    x_raw = lead_hip - trail_hip
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
