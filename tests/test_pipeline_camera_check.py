import numpy as np

from no_layups.pipeline.camera_check import detect_motion

SIZE = 240
MARGIN = 60


def _base_texture():
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, (SIZE + MARGIN, SIZE + MARGIN), dtype=np.uint8)


def _sampled_sequence(n_samples: int, shift_per_sample: int):
    """Every 5th frame (matching camera_check's own stride) carries the next
    shifted crop; frames in between are irrelevant filler."""
    base = _base_texture()
    frames = []
    for i in range(n_samples):
        dx = min(i * shift_per_sample, MARGIN)
        crop = base[0:SIZE, dx : dx + SIZE]
        bgr = np.stack([crop, crop, crop], axis=-1)
        frames.append(bgr)
        for _ in range(4):
            frames.append(np.zeros((SIZE, SIZE, 3), dtype=np.uint8))
    return frames


def test_static_camera_no_warning():
    frames = _sampled_sequence(n_samples=10, shift_per_sample=0)
    assert detect_motion(frames) is False


def test_panning_camera_triggers_warning():
    frames = _sampled_sequence(n_samples=10, shift_per_sample=5)
    assert detect_motion(frames) is True


def test_too_few_frames_returns_false():
    assert detect_motion([np.zeros((SIZE, SIZE, 3), dtype=np.uint8)]) is False
