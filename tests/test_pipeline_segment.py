import numpy as np
import pytest

from no_layups.pipeline.errors import PipelineError
from no_layups.pipeline.segment import detect_address, detect_takeaway_start

FPS = 30.0


def _wrist_path(n=90, stationary=20, move_frames=15, step=0.1):
    """Stationary, then a fast move (takeaway), matching Section 7.6's speed heuristic."""
    x = np.zeros(n)
    for i in range(stationary, min(stationary + move_frames, n)):
        x[i] = x[i - 1] + step
    x[stationary + move_frames :] = x[min(stationary + move_frames, n) - 1]
    return np.stack([x, np.zeros(n), np.zeros(n)], axis=1)


def test_detect_takeaway_start_finds_first_fast_run():
    wrist = _wrist_path(stationary=20)
    t = detect_takeaway_start(wrist, FPS)
    assert 19 <= t <= 21  # first frame of the fast run, tolerant of boundary


def test_detect_address_looks_back_from_takeaway():
    wrist = _wrist_path(stationary=20)
    takeaway = detect_takeaway_start(wrist, FPS)
    address = detect_address(wrist, FPS)
    assert address == max(0, takeaway - round(0.4 * FPS))


def test_stationary_series_raises_no_full_swing():
    wrist = np.zeros((90, 3))
    with pytest.raises(PipelineError) as exc:
        detect_takeaway_start(wrist, FPS)
    assert exc.value.code == "no_full_swing"
