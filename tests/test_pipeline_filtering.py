import numpy as np
import pytest

from no_layups import config
from no_layups.pipeline.errors import PipelineError
from no_layups.pipeline.filtering import (
    _savgol_window,
    check_quality,
    metric_critical_joints,
    process,
    smooth_depth,
)
from no_layups.pipeline.pose import PoseSeries

FPS = 30.0
N = 60


def _baseline_raw() -> PoseSeries:
    vis = {}
    xyz = {}
    t = np.arange(N, dtype=float)
    for i, joint in enumerate(config.JOINTS):
        vis[joint] = np.ones(N)
        # smooth, distinct linear motion per joint so joints don't collide
        xyz[joint] = np.stack([0.01 * t + i, 0.5 * t / N + i, np.zeros(N)], axis=1)
    return PoseSeries(vis=vis, xyz=xyz, frame_count=N)


def _drop_tracking(raw: PoseSeries, joint: str, start: int, length: int) -> None:
    """Mark frames untracked the way the pipeline now recognises it: absent
    coordinates. Visibility is no longer consulted (see the SPEC DEVIATION in
    filtering.process), so zeroing it would no longer drop anything -- which
    test_low_visibility_alone_is_not_missing_data asserts directly."""
    raw.xyz[joint][start : start + length] = np.nan


def test_low_visibility_alone_is_not_missing_data():
    """The core of the SPEC DEVIATION in filtering.process: MediaPipe scoring a
    joint as occluded must not discard coordinates it did produce. Measured on
    real footage, a plainly-visible lead arm scores a median visibility of 0.02
    while being tracked correctly."""
    raw = _baseline_raw()
    raw.vis["left_wrist"][:] = 0.0
    out = process(raw, FPS)
    assert not np.isnan(out["left_wrist"]).any()
    assert check_quality(out, 0, N - 1, "right") == []


def test_short_gap_is_interpolated_and_no_nan_remains():
    raw = _baseline_raw()
    _drop_tracking(raw, "left_wrist", start=20, length=5)  # < round(0.25*30)=8
    out = process(raw, FPS)
    assert not np.isnan(out["left_wrist"]).any()


def test_long_gap_stays_missing():
    raw = _baseline_raw()
    _drop_tracking(raw, "left_wrist", start=10, length=12)  # > max_gap
    out = process(raw, FPS)
    assert np.isnan(out["left_wrist"][10:22]).any()


def test_long_gap_does_not_contaminate_neighbouring_frames():
    """Savitzky-Golay must not bleed NaN outside the gap it came from."""
    raw = _baseline_raw()
    _drop_tracking(raw, "left_wrist", start=20, length=12)
    out = process(raw, FPS)
    assert not np.isnan(out["left_wrist"][:20]).any()
    assert not np.isnan(out["left_wrist"][32:]).any()


def test_process_does_not_raise_on_poor_tracking():
    """The quality gate is the caller's job now (scoped to address..impact)."""
    raw = _baseline_raw()
    _drop_tracking(raw, "left_ankle", start=0, length=20)  # 33% of 60 frames
    process(raw, FPS)  # must not raise


def test_check_quality_raises_on_a_metric_critical_joint():
    """left_wrist is the lead wrist for a right-handed golfer, and Section 8
    reads it for lead_elbow_top -- losing it makes the swing unmeasurable."""
    raw = _baseline_raw()
    _drop_tracking(raw, "left_wrist", start=0, length=20)  # 33% of 60 frames
    out = process(raw, FPS)
    with pytest.raises(PipelineError) as exc:
        check_quality(out, 0, N - 1, "right")
    assert exc.value.code == "poor_tracking"
    assert "left_wrist" in exc.value.message


def test_check_quality_only_warns_on_a_render_only_joint():
    """right_elbow is the trail elbow for a right-handed golfer and feeds no
    metric, so losing it degrades the skeleton but not the analysis. A
    down-the-line clip always hides one arm, and Section 13.2 requires that
    view to pass end-to-end."""
    raw = _baseline_raw()
    _drop_tracking(raw, "right_elbow", start=0, length=20)  # 33% of 60 frames
    out = process(raw, FPS)
    assert check_quality(out, 0, N - 1, "right") == ["right_elbow"]


def test_metric_critical_joints_follow_handedness():
    right = set(metric_critical_joints("right"))
    left = set(metric_critical_joints("left"))
    assert "left_wrist" in right and "right_wrist" not in right
    assert "right_wrist" in left and "left_wrist" not in left
    assert {"nose", "left_shoulder", "right_shoulder", "left_hip", "right_hip"} <= right & left


def test_check_quality_ignores_dropouts_outside_the_swing_span():
    """Tracking lost only during the follow-through must not fail the clip."""
    raw = _baseline_raw()
    _drop_tracking(raw, "left_wrist", start=40, length=20)  # all after 'impact'
    out = process(raw, FPS)
    assert check_quality(out, 0, 39, "right") == []  # must not raise


def test_smooth_depth_touches_only_the_depth_axis():
    """The extra pass is depth-only: MediaPipe's depth jitter measures 3-8x its
    in-image jitter, and a window wide enough to settle Z would over-smooth the
    two accurate axes."""
    raw = _baseline_raw()
    rng = np.random.default_rng(0)
    for joint in config.JOINTS:
        raw.xyz[joint][:, 2] += rng.normal(0, 0.02, N)
    base = process(raw, FPS)
    out = smooth_depth(base, FPS)

    for joint in config.JOINTS:
        assert np.allclose(out[joint][:, 0], base[joint][:, 0], atol=1e-12)
        assert np.allclose(out[joint][:, 1], base[joint][:, 1], atol=1e-12)

    def jitter(series):
        return float(np.nanmean(np.abs(np.diff(series, n=2))))

    assert jitter(out["left_wrist"][:, 2]) < jitter(base["left_wrist"][:, 2])


def test_smooth_depth_does_not_feed_segmentation():
    """Event detection and rendering have opposite requirements, so they must
    not share a window. detect_takeaway_start looks for the frame the lead
    wrist starts moving and a wide window smears that onset -- coupling the two
    collapsed the reference clip's address from a verified 49 to 31. process()
    output, which segmentation reads, must be unaffected by the render pass."""
    raw = _baseline_raw()
    base = process(raw, FPS)
    before = {j: a.copy() for j, a in base.items()}
    smooth_depth(base, FPS)

    for joint in config.JOINTS:
        assert np.array_equal(base[joint], before[joint], equal_nan=True)


def test_smooth_depth_keeps_missing_frames_missing():
    raw = _baseline_raw()
    _drop_tracking(raw, "left_wrist", start=10, length=12)  # > max_gap, stays NaN
    out = smooth_depth(process(raw, FPS), FPS)
    assert np.isnan(out["left_wrist"][10:22]).any()


def test_savgol_window_is_odd_and_clamped():
    assert _savgol_window(fps=30.0, n_frames=600) % 2 == 1
    assert _savgol_window(fps=30.0, n_frames=600) == 11  # round(30/3)=10 -> 11
    assert _savgol_window(fps=30.0, n_frames=6) == 5  # clamped to series length
