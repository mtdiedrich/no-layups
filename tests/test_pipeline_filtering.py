import numpy as np
import pytest

from no_layups import config
from no_layups.pipeline.errors import PipelineError
from no_layups.pipeline.filtering import _savgol_window, check_quality, process
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


def _drop_visibility(raw: PoseSeries, joint: str, start: int, length: int) -> None:
    raw.vis[joint][start : start + length] = 0.0


def test_short_gap_is_interpolated_and_no_nan_remains():
    raw = _baseline_raw()
    _drop_visibility(raw, "left_wrist", start=20, length=5)  # < round(0.25*30)=8
    out = process(raw, FPS)
    assert not np.isnan(out["left_wrist"]).any()


def test_long_gap_stays_missing():
    raw = _baseline_raw()
    _drop_visibility(raw, "left_wrist", start=10, length=12)  # > max_gap
    out = process(raw, FPS)
    assert np.isnan(out["left_wrist"][10:22]).any()


def test_long_gap_does_not_contaminate_neighbouring_frames():
    """Savitzky-Golay must not bleed NaN outside the gap it came from."""
    raw = _baseline_raw()
    _drop_visibility(raw, "left_wrist", start=20, length=12)
    out = process(raw, FPS)
    assert not np.isnan(out["left_wrist"][:20]).any()
    assert not np.isnan(out["left_wrist"][32:]).any()


def test_process_does_not_raise_on_poor_tracking():
    """The quality gate is the caller's job now (scoped to address..impact)."""
    raw = _baseline_raw()
    _drop_visibility(raw, "left_ankle", start=0, length=20)  # 33% of 60 frames
    process(raw, FPS)  # must not raise


def test_check_quality_raises_past_25_percent_missing():
    raw = _baseline_raw()
    _drop_visibility(raw, "left_ankle", start=0, length=20)  # 33% of 60 frames
    out = process(raw, FPS)
    with pytest.raises(PipelineError) as exc:
        check_quality(out, 0, N - 1)
    assert exc.value.code == "poor_tracking"
    assert "left_ankle" in exc.value.message


def test_check_quality_ignores_dropouts_outside_the_swing_span():
    """Tracking lost only during the follow-through must not fail the clip."""
    raw = _baseline_raw()
    _drop_visibility(raw, "left_ankle", start=40, length=20)  # all after 'impact'
    out = process(raw, FPS)
    check_quality(out, 0, 39)  # must not raise


def test_savgol_window_is_odd_and_clamped():
    assert _savgol_window(fps=30.0, n_frames=600) % 2 == 1
    assert _savgol_window(fps=30.0, n_frames=600) == 11  # round(30/3)=10 -> 11
    assert _savgol_window(fps=30.0, n_frames=6) == 5  # clamped to series length
