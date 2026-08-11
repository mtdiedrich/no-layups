import numpy as np
import pytest

from no_layups import config
from no_layups.pipeline.errors import PipelineError
from no_layups.pipeline.segment import (
    wrist_speed,
    detect_address,
    detect_keyframes,
    detect_takeaway_start,
    takeaway_speed_threshold,
)

FPS = 30.0


def _takeaway_path(n=90, stationary=20, move_frames=15, step=0.1):
    """Stationary, then a fast move (takeaway), matching Section 7.6 steps 1-2."""
    x = np.zeros(n)
    for i in range(stationary, min(stationary + move_frames, n)):
        x[i] = x[i - 1] + step
    x[stationary + move_frames :] = x[min(stationary + move_frames, n) - 1]
    return np.stack([x, np.zeros(n), np.zeros(n)], axis=1)


def test_detect_takeaway_start_finds_first_fast_run():
    wrist = _takeaway_path(stationary=20)
    t = detect_takeaway_start(wrist, FPS)
    assert 19 <= t <= 21  # first frame of the fast run, tolerant of boundary


def test_detect_address_looks_back_from_takeaway():
    wrist = _takeaway_path(stationary=20)
    takeaway = detect_takeaway_start(wrist, FPS)
    address = detect_address(wrist, FPS)
    assert address == max(0, takeaway - round(config.ADDRESS_LOOKBACK_S * FPS))


def _build_wrist_path(fps: float):
    """Synthetic lead-wrist path shaped like a real swing:

        stationary (address) -> linear rise (backswing) -> brief hold (top)
        -> descent through the ball (impact = lowest point) -> follow-through

    Stored in MediaPipe's raw convention, where Y grows DOWNWARD, so the
    stored y is the negated height. Segment boundaries use constant velocity
    and a strict global minimum so the truth frames are unambiguous: argmax
    returns the first frame of the top hold, argmin the single lowest frame of
    the descent.
    """
    stationary = round(fps)
    rise = round(fps)
    hold = round(0.2 * fps)
    descent = round(0.5 * fps)
    follow = round(0.3 * fps)

    rise_start = stationary
    rise_end = rise_start + rise  # exclusive
    hold_end = rise_end + hold
    descent_end = hold_end + descent
    n = descent_end + follow

    height = np.zeros(n)
    height[rise_start:rise_end] = np.linspace(0.6 / rise, 0.6, rise)
    height[rise_end:hold_end] = 0.6
    height[hold_end:descent_end] = np.linspace(0.6, -0.10, descent)
    height[descent_end:] = np.linspace(-0.10, 0.2, follow)

    truth = {
        "address": max(0, rise_start - round(0.4 * fps)),
        "top": rise_end - 1,
        "impact": descent_end - 1,
    }
    wrist_xyz = np.stack([np.zeros(n), -height, np.zeros(n)], axis=1)
    return wrist_xyz, truth


@pytest.mark.parametrize("fps", [30.0, 60.0])
def test_detect_keyframes_within_3_frames_of_truth(fps):
    wrist_xyz, truth = _build_wrist_path(fps)
    detected = detect_keyframes(wrist_xyz, fps)
    for key in ("address", "top", "impact"):
        assert abs(detected[key] - truth[key]) <= 3, f"{key}: {detected[key]} vs truth {truth[key]}"
    assert detected["address"] < detected["top"] < detected["impact"]


def test_stationary_series_raises_no_full_swing():
    wrist_xyz = np.zeros((90, 3))
    with pytest.raises(PipelineError) as exc:
        detect_keyframes(wrist_xyz, 30.0)
    assert exc.value.code == "no_full_swing"


def test_address_waggle_does_not_trigger_takeaway():
    """A waggle oscillates and returns; a takeaway translates away and stays.
    The lowered playback-invariant speed trigger fires on both, so net
    displacement is what separates them. Regression test for the reference
    clip picking frame 1, ~2 s before the lead wrist actually left address."""
    fps = 30.0
    wrist_xyz, truth = _build_wrist_path(fps)
    # Superimpose an oscillating waggle across the whole address hold. Its peak
    # speed sits above the relative trigger but its net displacement is ~0.
    hold = truth["address"] + round(0.4 * fps)
    t = np.arange(hold)
    wrist_xyz[:hold, 0] += 0.01 * np.sin(2 * np.pi * t / 5.0)

    v = wrist_speed(wrist_xyz, fps)
    assert v[1:hold].max() > takeaway_speed_threshold(v), "waggle must beat the speed trigger"

    detected = detect_keyframes(wrist_xyz, fps)
    assert abs(detected["address"] - truth["address"]) <= 3
    assert abs(detected["top"] - truth["top"]) <= 3


def test_takeaway_threshold_never_exceeds_the_spec_absolute():
    """The relative term may only make takeaway detection more sensitive.
    Normal-speed footage must keep exactly the spec's 0.4 m/s behaviour."""
    fast = np.array([0.0, 5.0, 18.0, 30.0, 12.0])  # real swing, real time
    assert takeaway_speed_threshold(fast) == pytest.approx(
        config.TAKEAWAY_SPEED_THRESHOLD_MPS
    )

    slow = fast / 8.0  # same swing, 8x slow motion
    assert takeaway_speed_threshold(slow) < config.TAKEAWAY_SPEED_THRESHOLD_MPS
    assert takeaway_speed_threshold(slow) == pytest.approx(
        config.TAKEAWAY_SPEED_FRACTION_OF_PEAK * 30.0 / 8.0
    )


def test_takeaway_threshold_on_a_stationary_clip_still_blocks_detection():
    """Peak 0 gives threshold 0; the strict `>` must keep rejecting it."""
    still = np.zeros(60)
    assert takeaway_speed_threshold(still) == 0.0
    assert not (still > takeaway_speed_threshold(still)).any()


def test_slow_motion_swing_is_still_detected():
    """A swing filmed in slow motion covers the same distance in metres over
    more frames, so its peak speed is far below the spec's absolute 3.0 m/s
    impact gate. It must still segment -- see MIN_BACKSWING_RISE_M."""
    fps = 30.0
    wrist_xyz, truth = _build_wrist_path(fps)
    # Stretch the timeline 8x: same geometry, 1/8th the speed.
    n = len(wrist_xyz)
    stretched = np.stack(
        [np.interp(np.linspace(0, n - 1, n * 8), np.arange(n), wrist_xyz[:, axis]) for axis in range(3)],
        axis=1,
    )
    peak_speed = float(
        np.nanmax(np.linalg.norm(np.diff(stretched, axis=0), axis=1) * fps)
    )
    assert peak_speed < 3.0, "fixture should be below the spec's absolute impact gate"

    detected = detect_keyframes(stretched, fps)
    assert detected["address"] < detected["top"] < detected["impact"]
    assert abs(detected["top"] - truth["top"] * 8) <= 8
    assert abs(detected["impact"] - truth["impact"] * 8) <= 8


def test_finish_higher_than_the_top_does_not_steal_it():
    """A clip that runs through to a full finish ends with the hands HIGHER
    than they ever were at the top of the backswing. Regression test for the
    global-argmax top, which returned the finish and dragged impact after it:
    on the reference clip the finish peaked at +0.888 m against the backswing
    top's +0.719 m, giving top=711 and impact=725 out of 873 frames."""
    fps = 60.0
    wrist_xyz, truth = _build_wrist_path(fps)
    # Replace the short follow-through with a full finish that climbs past the top.
    n = len(wrist_xyz)
    finish_from = truth["impact"] + 2
    top_height = -wrist_xyz[truth["top"], 1]
    finish = np.linspace(-0.10, top_height + 0.25, n - finish_from)
    wrist_xyz[finish_from:, 1] = -finish

    height = -wrist_xyz[:, 1]
    assert height[finish_from:].max() > height[truth["top"]], "fixture must have a higher finish"

    detected = detect_keyframes(wrist_xyz, fps)
    assert abs(detected["top"] - truth["top"]) <= 3, f"top landed in the finish: {detected}"
    assert abs(detected["impact"] - truth["impact"]) <= 3
    assert detected["address"] < detected["top"] < detected["impact"]


def test_nan_gap_after_impact_is_ignored():
    """Tracking loss in the follow-through must not corrupt detection --
    regression test for np.argmin/argmax silently picking a NaN entry."""
    fps = 30.0
    wrist_xyz, truth = _build_wrist_path(fps)
    wrist_xyz[truth["impact"] + 2 :] = np.nan
    detected = detect_keyframes(wrist_xyz, fps)
    assert abs(detected["impact"] - truth["impact"]) <= 3
    assert abs(detected["top"] - truth["top"]) <= 3


def test_all_nan_after_top_raises_no_full_swing():
    fps = 30.0
    wrist_xyz, truth = _build_wrist_path(fps)
    wrist_xyz[truth["top"] + 3 :] = np.nan
    with pytest.raises(PipelineError) as exc:
        detect_keyframes(wrist_xyz, fps)
    assert exc.value.code == "no_full_swing"


def test_motion_without_a_backswing_raises_no_full_swing():
    """Lateral movement with no lead-wrist rise is not a golf swing."""
    fps = 30.0
    n = 90
    x = np.zeros(n)
    x[30:] = np.linspace(0.02, 1.2, n - 30)
    wrist_xyz = np.stack([x, np.zeros(n), np.zeros(n)], axis=1)
    with pytest.raises(PipelineError) as exc:
        detect_keyframes(wrist_xyz, fps)
    assert exc.value.code == "no_full_swing"
