import numpy as np
from scipy.signal import savgol_filter

from .. import config
from . import filtering

# MediaPipe world landmarks grow DOWNWARD in Y (measured: nose at y=-0.37,
# ankle at y=+0.73 at address -- see segment.wrist_height, which relies on the
# same fact). For a level camera this axis is gravity, so -Y is world up.
WORLD_UP = np.array([0.0, -1.0, 0.0])


def _mid(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a + b) / 2.0


def _normalize(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def _vertical_direction(
    xyz_by_joint: dict[str, np.ndarray], stance_span: tuple[int, int]
) -> np.ndarray:
    """Up, taken from the legs rather than assumed to be the camera's vertical.

    SPEC DEVIATION (Section 6.1, continued): WORLD_UP assumes MediaPipe's Y is
    gravity, which holds only for a level camera. It does not hold here, and
    not because of the camera: the pelvis lands in FRONT of the feet, so the
    rendered golfer is falling on their face. Measured at address, the hips sit
    0.093 m forward of the ankle line, the shoulders 0.384 m, the nose 0.709 m,
    which puts the centre of mass well outside the toes.

    A single rotation about +X that puts the pelvis back over the feet -- a
    balance constraint every standing golfer satisfies -- also drags two
    quantities it was not fitted to into range, which is what makes it
    credible rather than curve-fitting:

        spine from vertical   39.3 -> 32.3 deg   (25-35 for a driver)
        shoulder fwd of feet  0.384 -> 0.246 m   (0.25-0.35)

    That rotation is -7.0 deg on the reference clip and -6.2 deg on an
    unrelated clip shot on a different camera. Two cameras do not agree by
    accident, so this is a systematic bias in MediaPipe's leg/pelvis depth, not
    camera pitch -- which is the argument for correcting it generally.

    Using the LEG as the vertical reference, never the trunk: a golfer's trunk
    genuinely leans ~30 deg at address (that error is what the WORLD_UP change
    fixed), while the leg is near-vertical by definition of standing up. Since
    MediaPipe pins the mid-hip at the origin, "mid-hip minus mid-ankle" reduces
    to the negated ankle midpoint.

    The span must be a window around ADDRESS, not the planted span used for the
    azimuth. The pelvis genuinely moves over the feet during the backswing
    (that is the weight shift), so averaging across it blends real motion into
    the reference and under-corrects -- measured, address..top left 4.4 deg of
    the 7 deg still uncorrected.

    Residual: this forces the pelvis exactly over the ankles, where a real
    setup sits a few cm behind them. That is a 2-4 deg over-correction against
    the 7 deg it removes.

    What this does NOT fix, because no rotation can: the nose sits 0.325 m in
    front of the shoulder line and level with it, before and after (0.325 ->
    0.327). A head is ~0.15 m forward and ~0.15 m above. That is the same
    non-rigid world-landmark head that varies arm length by 92%.
    """
    lo, hi = stance_span
    left = xyz_by_joint["left_ankle"][lo : hi + 1]
    right = xyz_by_joint["right_ankle"][lo : hi + 1]

    up = -(left + right) / 2.0  # mid-hip is the origin, so this is hip - ankle
    norms = np.linalg.norm(up, axis=1)
    usable = np.isfinite(norms) & (norms > 1e-9)
    if not usable.any():
        return WORLD_UP
    return _normalize(np.median(up[usable] / norms[usable, None], axis=0))


def _stance_direction(
    xyz_by_joint: dict[str, np.ndarray], up: np.ndarray, stance_span: tuple[int, int]
) -> np.ndarray:
    """The target-line direction, taken from the ANKLES over a span where the
    feet are planted.

    SPEC DEVIATION (Section 6.1): +X comes from the ankle line, not "the hip
    line at address".

    The azimuth of +X -- the frame's rotation about vertical -- is an atan2 of
    a depth difference over a body width. Depth is MediaPipe's worst channel
    and a hip is only ~0.19 m wide, so ~5 cm of depth error swings the whole
    skeleton ~15 deg about the vertical axis. Taking it from one frame makes
    that a coin flip, which is why the reference clip rendered with its feet
    13.9 deg out of square in the face-on view.

    The ankles are the better reference for two reasons. They are planted, so
    the estimate can be averaged over hundreds of frames instead of trusted at
    one; and two independent lower-body lines agree on it while the address hip
    line does not (measured: ankles 15.9 deg, knees 15.6 deg, address hip line
    0.0 deg by construction).

    The span matters and must not be the whole clip. Feet only start planted:
    after impact the trail foot pivots up onto its toe and the ankle line
    swings about 20 deg, which drags a whole-clip median to a meaningless 5.1
    deg. Everything from address to top agrees within a degree (address alone
    13.9, address +-0.5 s 13.9, address..top 14.7), so the caller passes the
    planted span.

    A median of per-frame directions, not a mean: it shrugs off the occasional
    frame where an ankle is mistracked.

    Note this makes the feet square by construction, which is the intent -- a
    golfer sets up with the feet on the target line. The turn metrics do not
    care either way: turn_angle measures the angle BETWEEN two lines, which is
    invariant to how the frame is rotated about vertical. spine_tilt and
    head_sway_top do care, since both read the X component.
    """
    lo, hi = stance_span
    left = xyz_by_joint["left_ankle"][lo : hi + 1]
    right = xyz_by_joint["right_ankle"][lo : hi + 1]

    d = left - right
    d = d - np.outer(d @ up, up)  # project onto the ground plane
    norms = np.linalg.norm(d, axis=1)
    usable = np.isfinite(norms) & (norms > 1e-9)
    if not usable.any():
        raise ValueError("no usable ankle tracking to derive the stance direction")
    return _normalize(np.median(d[usable] / norms[usable, None], axis=0))


def transform(
    xyz_by_joint: dict[str, np.ndarray],
    address_idx: int,
    handedness: str,
    stance_span: tuple[int, int] | None = None,
    address_span: tuple[int, int] | None = None,
    fps: float | None = None,
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

    SPEC DEVIATION (Section 6.1): +Y is world up (WORLD_UP), not the spec's
    normalize(mid(shoulders) - mid(hips)) at address. That vector is the
    golfer's SPINE, and a golfer at address is bent ~30 deg forward, so
    adopting it as +Y rotates the entire skeleton by the address spine angle.
    Measured on the reference clip under the spec's definition, the feet land
    0.394 m in +Z -- 40 cm "behind" the hips, at atan(0.394/0.648) = 31 deg,
    exactly the address lean -- and the two ankles sit 0.042 m apart in Y
    despite both being flat on the ground. Section 11.2's viewer draws a
    horizontal GridHelper under that, so the rendered golfer stands bolt
    upright with the feet swung out behind and one foot through the floor.

    Switching to world up puts the feet back under the hips (mid-ankle z:
    +0.394 -> -0.085) and levels them (0.042 -> 0.008 m), on both clips tested.
    lead_elbow_top is unchanged to the decimal (139.5 deg), which is the
    control: angle_at is rotation-invariant, so an unchanged joint angle
    confirms this is a pure re-orientation and not a change to the data.

    Two things the spec's own text wanted also start working:
      * Section 8.1 says the turn metrics "project onto the ground plane (drop
        Y)". Under a spine +Y that dropped the wrong plane, tilted 31 deg off
        the ground. shoulder_turn_top moves 54.6 -> 59.0 deg on the reference
        clip, toward Section 13.2's expected 60-110.
      * spine_tilt_address stops being degenerate. Under a spine +Y the address
        trunk IS the up axis, so Section 8.2's formula returned exactly 0.0 for
        every clip ever processed, and metrics.py had to null it out. It is a
        real measurement again.

    This assumes a level camera, which Section 7.2 already assumes (it warns on
    camera motion and Section 1.3 excludes stabilization). A rolled camera
    tilts the skeleton by the roll angle; the spec's spine-relative frame was
    immune to that, which is the one thing given up here.
    """
    left_hip = xyz_by_joint["left_hip"][address_idx]
    right_hip = xyz_by_joint["right_hip"][address_idx]

    if stance_span is None:
        stance_span = (address_idx, address_idx)
    if address_span is None:
        address_span = (address_idx, address_idx)
    up = _vertical_direction(xyz_by_joint, address_span)
    x_axis = _stance_direction(xyz_by_joint, up, stance_span)

    z_axis = np.cross(x_axis, up)
    assert abs(np.linalg.norm(z_axis) - 1.0) < 1e-6

    origin = _mid(left_hip, right_hip)
    basis = np.stack([x_axis, up, z_axis])  # rows: x, y, z basis vectors

    rotated = {joint: (arr - origin) @ basis.T for joint, arr in xyz_by_joint.items()}
    anchor = _ground_anchor(rotated, fps)

    out: dict[str, np.ndarray] = {}
    for joint, arr in rotated.items():
        transformed = arr - anchor
        if handedness == "left":
            transformed[:, 0] *= -1
        out[joint] = transformed
    return out


def _fill_nan_rows(a: np.ndarray) -> np.ndarray:
    """Interpolate whole missing rows so a gap in ankle tracking cannot wipe out
    every joint in that frame via the anchor."""
    a = a.copy()
    missing = np.isnan(a).any(axis=1)
    if not missing.any():
        return a
    if missing.all():
        return np.zeros_like(a)
    idx = np.arange(len(a))
    for axis in range(a.shape[1]):
        a[missing, axis] = np.interp(idx[missing], idx[~missing], a[~missing, axis])
    return a


def _ground_anchor(rotated: dict[str, np.ndarray], fps: float | None = None) -> np.ndarray:
    """Per-frame origin: where the golfer meets the ground.

    SPEC DEVIATION (Section 6.1): the origin is the golfer's ground contact per
    frame, not "midpoint of the hips at address".

    The spec's origin cannot work, because MediaPipe world landmarks are
    already hip-centred by definition -- the mid-hip is (0,0,0) in every frame
    before we touch it. Measured on the reference clip, mid-hip travels 0.001 m
    over the whole swing. So subtracting the address mid-hip is a no-op, the
    golfer can never translate, and every bit of pelvis motion is expressed
    instead as the FEET moving around the pinned hips: the ankles range 0.169 m
    in X and 0.260 m in Z across address..impact, on a golfer whose feet are
    planted. The rendered skeleton is a man pinned at the waist skating his
    feet.

    Anchoring to the ground inverts that back: the feet hold still and the
    pelvis moves, giving 0.176 m of lateral hip slide and 0.056 m of vertical
    squat over address..impact (real golf is roughly 0.05-0.15 m and
    0.02-0.10 m respectively). Depth stays wrong at 0.218 m -- that is the
    separate non-rigidity problem in MediaPipe's world head, which no choice of
    origin repairs.

    Horizontal comes from the ankle midpoint, but vertical comes from the
    LOWER ankle rather than the midpoint, because feet only start level.
    Through the follow-through the trail heel lifts and that foot rolls onto
    its toe; a midpoint would rise with it and make the whole body appear to
    sink at the finish. The lower ankle is the one still bearing weight, so it
    stays on the ground and keeps the ground where it belongs.

    This is a per-frame translation, so it is rigid within each frame: bone
    lengths and every joint angle are untouched, and so is every metric that
    reads a single frame. It does change head_sway_top, which compares the
    nose across two frames -- that measurement becomes head movement relative
    to the ground instead of relative to the pelvis, which is what Section 8.3
    ("horizontal distance of nose between address and top") asks for anyway.
    """
    left, right = _fill_nan_rows(rotated["left_ankle"]), _fill_nan_rows(rotated["right_ankle"])
    anchor = (left + right) / 2.0
    anchor[:, 1] = np.minimum(left[:, 1], right[:, 1])

    # Smoothed harder than any joint. This is a translation applied to every
    # joint, so whatever noise it carries is added to the entire skeleton --
    # it was a source of jitter in the rendered figure, not just in the feet.
    # Where a golfer stands changes far more slowly than their limbs move, so a
    # wide window costs nothing real here. The vertical component takes a
    # minimum over the two ankles, which is a non-linear operation that kinks
    # whenever the weight-bearing foot changes; smoothing rounds that off too.
    if fps is not None:
        window = filtering._savgol_window(
            fps, len(anchor), scale=config.GROUND_ANCHOR_WINDOW_SCALE
        )
        if window >= config.SAVGOL_POLYORDER + 1:
            for axis in range(3):
                anchor[:, axis] = savgol_filter(
                    anchor[:, axis],
                    window_length=window,
                    polyorder=config.SAVGOL_POLYORDER,
                    mode="interp",
                )
    return anchor
