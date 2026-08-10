from dataclasses import dataclass
from typing import Iterable

import cv2
import mediapipe as mp
import numpy as np

from .. import config
from .errors import PipelineError


@dataclass
class PoseSeries:
    vis: dict[str, np.ndarray]  # joint -> (N,) visibility from pose_landmarks
    xyz: dict[str, np.ndarray]  # joint -> (N, 3) world coords, NaN where undetected
    frame_count: int


def extract(frame_iter: Iterable[np.ndarray]) -> PoseSeries:
    """Section 7.3: one MediaPipe Pose instance per call, closed when done."""
    vis_lists: dict[str, list[float]] = {j: [] for j in config.JOINTS}
    xyz_lists: dict[str, list[tuple[float, float, float]]] = {j: [] for j in config.JOINTS}
    any_detected = False
    n = 0

    pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=config.POSE_MODEL_COMPLEXITY,
        min_detection_confidence=config.POSE_MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=config.POSE_MIN_TRACKING_CONFIDENCE,
    )
    try:
        for frame in frame_iter:
            n += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)
            if results.pose_landmarks is None:
                for joint in config.JOINTS:
                    vis_lists[joint].append(0.0)
                    xyz_lists[joint].append((np.nan, np.nan, np.nan))
                continue
            any_detected = True
            landmarks_2d = results.pose_landmarks.landmark
            landmarks_3d = results.pose_world_landmarks.landmark
            for joint in config.JOINTS:
                idx = config.MEDIAPIPE_LANDMARK_INDEX[joint]
                vis_lists[joint].append(landmarks_2d[idx].visibility)
                world = landmarks_3d[idx]
                xyz_lists[joint].append((world.x, world.y, world.z))
    finally:
        pose.close()

    if not any_detected:
        raise PipelineError("no_person_detected", "no frame in the video had a detectable person")

    vis = {j: np.array(v, dtype=float) for j, v in vis_lists.items()}
    xyz = {j: np.array(v, dtype=float) for j, v in xyz_lists.items()}
    return PoseSeries(vis=vis, xyz=xyz, frame_count=n)
