import numpy as np
import pytest

from no_layups.pipeline.canonical import transform


LEAN_DEG = 30.0


def _synthetic_frames():
    """A single-frame address posture in MediaPipe's world convention.

    Two properties matter and the previous fixture had neither:

    * Y grows DOWNWARD, as MediaPipe world landmarks do (see
      canonical.WORLD_UP). A Y-up fixture silently tests a mirrored world.
    * The golfer is bent LEAN_DEG forward at address, as a golfer is. The
      previous fixture stood perfectly upright, which made the spine and
      gravity axes identical -- so it could not distinguish them, and the
      31-degree skeleton rotation this file now guards against passed it.

    Feet are deliberately level and directly under the hips: the whole point
    of the canonical frame is that it preserves that.
    """
    lean = np.radians(LEAN_DEG)
    hip_y, hip_z = 0.0, 0.0

    def above_hips(height, forward=0.0):
        """A point `height` up the spine from the hips, leaned forward."""
        return [
            hip_z + height * np.sin(lean) + forward,
            hip_y - height * np.cos(lean),  # minus: Y grows downward
        ]

    sh_z, sh_y = above_hips(0.5)
    el_z, el_y = above_hips(0.25, forward=0.10)
    wr_z, wr_y = above_hips(0.05, forward=0.18)
    no_z, no_y = above_hips(0.70)

    joints = {
        "nose": [[0.0, no_y, no_z]],
        "left_shoulder": [[0.2, sh_y, sh_z]],
        "right_shoulder": [[-0.2, sh_y, sh_z]],
        "left_elbow": [[0.25, el_y, el_z]],
        "right_elbow": [[-0.25, el_y, el_z]],
        "left_wrist": [[0.2, wr_y, wr_z]],
        "right_wrist": [[-0.2, wr_y, wr_z]],
        "left_hip": [[0.1, hip_y, hip_z]],
        "right_hip": [[-0.1, hip_y, hip_z]],
        # legs vertical, feet level and under the hips
        "left_knee": [[0.1, hip_y + 0.45, 0.0]],
        "right_knee": [[-0.1, hip_y + 0.45, 0.0]],
        "left_ankle": [[0.1, hip_y + 0.90, 0.0]],
        "right_ankle": [[-0.1, hip_y + 0.90, 0.0]],
    }
    return {j: np.array(v, dtype=float) for j, v in joints.items()}


def test_feet_are_level_and_under_the_hips_AT_ADDRESS():
    """The regression this file exists for. Under the spec's spine-derived +Y,
    a golfer leaning LEAN_DEG forward has the whole skeleton rotated by that
    angle: the feet swing out behind the hips in +Z and stop being level.
    Measured on the reference clip before the fix, mid-ankle sat 0.394 m in +Z
    with the ankles 0.042 m apart in Y.

    Scoped to the address frame deliberately. Feet START level; they do not
    stay that way. Through the follow-through the trail heel lifts and that
    foot rolls onto its toe, so a level-ankles assertion over the whole clip
    would be asserting something false about golf."""
    frames = _synthetic_frames()
    out = transform(frames, address_idx=0, handedness="right")
    left_ankle, right_ankle = out["left_ankle"][0], out["right_ankle"][0]

    assert left_ankle[1] == pytest.approx(right_ankle[1], abs=1e-9)
    mid_ankle = (left_ankle + right_ankle) / 2
    assert mid_ankle[0] == pytest.approx(0.0, abs=1e-9)
    assert mid_ankle[2] == pytest.approx(0.0, abs=1e-9)
    assert out["left_hip"][0][1] > mid_ankle[1]  # hips above the feet


def test_pelvis_lands_over_the_feet_so_the_golfer_can_stand_up():
    """A standing golfer's centre of mass is over their base of support. Assuming
    MediaPipe's Y is gravity put the pelvis 0.093 m in FRONT of the ankle line
    on the reference clip (shoulders 0.384 m, nose 0.709 m) -- a golfer falling
    on their face. +Y comes from the legs at address instead.

    The fixture leans the TRUNK forward but keeps the legs vertical, so a
    correct vertical leaves the pelvis over the feet."""
    frames = _synthetic_frames()
    out = transform(frames, address_idx=0, handedness="right")

    mid_ankle = (out["left_ankle"][0] + out["right_ankle"][0]) / 2
    mid_hip = (out["left_hip"][0] + out["right_hip"][0]) / 2
    assert mid_hip[2] - mid_ankle[2] == pytest.approx(0.0, abs=1e-9)
    assert mid_hip[0] - mid_ankle[0] == pytest.approx(0.0, abs=1e-9)


def test_vertical_survives_a_tilted_camera():
    """The whole point of deriving +Y from the body: pitch the camera and the
    golfer must still stand upright rather than lean by the pitch angle."""
    frames = _synthetic_frames()
    tilt = np.radians(12.0)
    rot = np.array(
        [[1, 0, 0], [0, np.cos(tilt), -np.sin(tilt)], [0, np.sin(tilt), np.cos(tilt)]]
    )
    tilted = {j: (rot @ a.T).T for j, a in frames.items()}

    out = transform(tilted, address_idx=0, handedness="right")
    mid_ankle = (out["left_ankle"][0] + out["right_ankle"][0]) / 2
    mid_hip = (out["left_hip"][0] + out["right_hip"][0]) / 2
    assert mid_hip[2] - mid_ankle[2] == pytest.approx(0.0, abs=1e-9)


def test_azimuth_comes_from_the_ankles_not_the_hips():
    """+X is the target line, and the feet define it. Skew ONLY the hip line in
    depth -- the kind of error 5 cm of MediaPipe depth noise produces over a
    0.19 m hip width -- and the frame must not rotate: the feet stay square.
    Under the spec's hip-derived +X this skew rotated the entire skeleton,
    which is what put the reference clip's feet 13.9 deg out of square."""
    frames = _synthetic_frames()
    frames["left_hip"][0][2] += 0.05
    frames["right_hip"][0][2] -= 0.05

    out = transform(frames, address_idx=0, handedness="right")
    left_ankle, right_ankle = out["left_ankle"][0], out["right_ankle"][0]

    # the ankle line lies along +X: square to the camera, no depth component
    assert left_ankle[2] == pytest.approx(right_ankle[2], abs=1e-9)
    assert left_ankle[0] > right_ankle[0]


def test_stance_direction_ignores_frames_outside_the_planted_span():
    """Feet only start planted. Rotate the ankle line late in the clip, as the
    trail foot does when it pivots onto its toe through the follow-through, and
    the frame must ignore it -- on the reference clip those frames drag a
    whole-clip median from ~15 deg to a meaningless 5.1 deg."""
    frames = _synthetic_frames()
    n = 20
    frames = {j: np.repeat(a, n, axis=0) for j, a in frames.items()}
    # back half of the clip: swing the trail ankle well out of line
    frames["right_ankle"][n // 2 :, 2] += 0.30

    planted = transform(frames, address_idx=0, handedness="right", stance_span=(0, n // 2 - 1))
    la, ra = planted["left_ankle"][0], planted["right_ankle"][0]
    assert la[2] == pytest.approx(ra[2], abs=1e-9)


SLIDE = 0.10


def _two_frame_pelvis_slide():
    """Frame 0 address, frame 1 after the pelvis slides SLIDE toward the target
    -- expressed the way MediaPipe actually emits it. Because the mid-hip is
    pinned at the origin in every frame, a pelvis that slides one way is
    emitted as the LEGS AND FEET sliding the other way."""
    frames = _synthetic_frames()
    out = {}
    for joint, arr in frames.items():
        moved = arr[0].copy()
        if joint in ("left_ankle", "right_ankle", "left_knee", "right_knee"):
            moved[0] -= SLIDE
        out[joint] = np.array([arr[0], moved], dtype=float)
    return out


def test_ground_anchoring_plants_the_feet_and_moves_the_pelvis():
    """MediaPipe pins the mid-hip at the origin in every frame, so pelvis
    motion comes out as the feet skating around it -- measured on the reference
    clip, mid-hip travelled 0.001 m over the whole swing while the ankles
    ranged 0.169 m in X. Anchoring to the ground inverts that: the feet hold
    still and the pelvis moves."""
    frames = _two_frame_pelvis_slide()
    out = transform(frames, address_idx=0, handedness="right")

    mid_ankle = (out["left_ankle"] + out["right_ankle"]) / 2
    mid_hip = (out["left_hip"] + out["right_hip"]) / 2

    # the ground anchor IS the feet, so their midpoint cannot drift
    assert np.allclose(mid_ankle[0], mid_ankle[1], atol=1e-9)
    # and the pelvis now carries the motion, in the correct direction
    assert mid_hip[1][0] - mid_hip[0][0] == pytest.approx(SLIDE, abs=1e-9)


def test_vertical_anchor_follows_the_planted_foot_not_the_average():
    """Feet only start level. Lift the trail heel, as happens through the
    follow-through, and the ground must stay put -- an averaged anchor would
    rise with the lifted foot and make the whole body appear to sink."""
    frames = _synthetic_frames()
    frames = {j: np.repeat(a, 2, axis=0) for j, a in frames.items()}
    # MediaPipe Y grows downward, so lifting the heel means DECREASING y
    frames["right_ankle"][1][1] -= 0.20

    out = transform(frames, address_idx=0, handedness="right")
    planted = out["left_ankle"]
    assert planted[1][1] == pytest.approx(planted[0][1], abs=1e-9)


def test_address_trunk_leans_forward_rather_than_defining_up():
    """+Y must be gravity, not the spine. If it were the spine the trunk would
    land exactly on the up axis (z == 0) and spine_tilt_address would be
    identically 0.0 for every clip -- the degeneracy that forced metrics.py to
    null that metric out."""
    frames = _synthetic_frames()
    out = transform(frames, address_idx=0, handedness="right")
    mid_shoulder = (out["left_shoulder"][0] + out["right_shoulder"][0]) / 2
    mid_hip = (out["left_hip"][0] + out["right_hip"][0]) / 2
    trunk = mid_shoulder - mid_hip  # measured hip-to-shoulder, not from the origin

    assert trunk[1] > 0  # shoulders above the hips
    # Sign convention: z_axis = x_axis x up = (0,0,-1), so canonical +Z points
    # toward the camera and leaning forward (away from it) lands in -Z. The
    # magnitude is what matters here, not which way the camera faces --
    # Section 6.1 calls that direction an accepted ambiguity.
    assert abs(trunk[2]) == pytest.approx(0.5 * np.sin(np.radians(LEAN_DEG)), abs=1e-9)
    recovered_lean = np.degrees(np.arctan2(abs(trunk[2]), trunk[1]))
    assert recovered_lean == pytest.approx(LEAN_DEG, abs=1e-6)


def test_ground_contact_is_the_origin_not_the_hips():
    """Section 13.1 asks for address mid-hip == (0,0,0). That origin is not
    reachable: MediaPipe emits hip-centred coordinates, so the mid-hip is
    ALREADY (0,0,0) in every frame and subtracting it does nothing -- which is
    why the golfer could never translate. The origin is the ground contact
    instead; see the SPEC DEVIATION on canonical._ground_anchor."""
    frames = _synthetic_frames()
    out = transform(frames, address_idx=0, handedness="right")

    mid_ankle = (out["left_ankle"][0] + out["right_ankle"][0]) / 2
    assert np.allclose(mid_ankle, [0.0, 0.0, 0.0], atol=1e-9)

    mid_hip = (out["left_hip"][0] + out["right_hip"][0]) / 2
    assert mid_hip[0] == pytest.approx(0.0, abs=1e-9)
    assert mid_hip[2] == pytest.approx(0.0, abs=1e-9)
    assert mid_hip[1] > 0  # the golfer stands above the ground


def test_address_mid_shoulder_is_centred_laterally_and_above_the_hips():
    """Section 13.1 asks for address mid-shoulder at x~0, z~0, y>0. The x and y
    halves still hold. The z~0 half does NOT, and asserting it was the thing
    that locked in the bug: z~0 says the trunk lies exactly along +Y, which is
    true only if +Y is the spine. See
    test_address_trunk_leans_forward_rather_than_defining_up for the
    replacement."""
    frames = _synthetic_frames()
    out = transform(frames, address_idx=0, handedness="right")
    mid_shoulder = (out["left_shoulder"][0] + out["right_shoulder"][0]) / 2
    assert mid_shoulder[0] == pytest.approx(0.0, abs=1e-9)
    assert mid_shoulder[1] > 0


def test_address_hip_line_is_level_and_lead_hip_is_positive_x():
    frames = _synthetic_frames()
    out = transform(frames, address_idx=0, handedness="right")
    # right-handed: lead hip = left_hip
    assert out["left_hip"][0][1] == pytest.approx(out["right_hip"][0][1], abs=1e-9)
    assert out["left_hip"][0][0] > out["right_hip"][0][0]


def test_left_handed_mirror_sign_convention():
    """Section 13.1: after mirroring a left-handed swing, the anatomical
    right hip has x > 0 (see canonical.py docstring for why +X is built from
    the fixed anatomical hips rather than lead/trail)."""
    frames = _synthetic_frames()
    out = transform(frames, address_idx=0, handedness="left")
    assert out["right_hip"][0][0] > 0
    assert out["left_hip"][0][0] < 0
