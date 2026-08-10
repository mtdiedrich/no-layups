from typing import Iterable

import cv2
import numpy as np

from .. import config


def _border_mask(shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    bx = max(1, int(w * config.CAMERA_CHECK_BORDER_FRACTION))
    by = max(1, int(h * config.CAMERA_CHECK_BORDER_FRACTION))
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[:by, :] = 255
    mask[h - by :, :] = 255
    mask[:, :bx] = 255
    mask[:, w - bx :] = 255
    return mask


def detect_motion(frame_iter: Iterable[np.ndarray]) -> bool:
    """Section 7.2: samples every 5th frame (up to 40), tracks border corners,
    and flags camera motion if median displacement exceeds the threshold for
    more than half of the sample pairs."""
    samples = []
    for i, frame in enumerate(frame_iter):
        if i % config.CAMERA_CHECK_STRIDE == 0:
            samples.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            if len(samples) >= config.CAMERA_CHECK_MAX_SAMPLES:
                break

    if len(samples) < 2:
        return False

    exceed_count = 0
    pair_count = 0
    for prev, nxt in zip(samples, samples[1:]):
        mask = _border_mask(prev.shape)
        corners = cv2.goodFeaturesToTrack(
            prev,
            maxCorners=config.CAMERA_CHECK_MAX_CORNERS,
            qualityLevel=config.CAMERA_CHECK_QUALITY_LEVEL,
            minDistance=config.CAMERA_CHECK_MIN_DISTANCE,
            mask=mask,
        )
        if corners is None or len(corners) < config.CAMERA_CHECK_MIN_CORNERS:
            continue

        next_pts, status, _err = cv2.calcOpticalFlowPyrLK(prev, nxt, corners, None)
        status = status.flatten() == 1
        good_old = corners[status].reshape(-1, 2)
        good_new = next_pts[status].reshape(-1, 2)
        if len(good_old) == 0:
            continue

        median_disp = float(np.median(np.linalg.norm(good_new - good_old, axis=1)))
        pair_count += 1
        if median_disp > config.CAMERA_CHECK_DISPLACEMENT_THRESHOLD_PX:
            exceed_count += 1

    if pair_count == 0:
        return False
    return exceed_count > pair_count / 2
