"""Render a video with the tracked pose drawn on top of it.

The M3 viewer shows the abstract 3D skeleton; this shows what MediaPipe
actually saw, in image space, on the original frames. It is the tool for
answering "is the tracking wrong, or is the maths wrong?" -- the two failure
modes look identical in swing.json.

    uv run python tools/overlay_pose.py RORY.mov -o overlay.mp4
    uv run python tools/overlay_pose.py RORY.mov -o overlay.mp4 \
        --swing reference/reference.json

Pass --swing to label the detected address/top/impact frames and hold each for
a moment so they are visible at normal playback speed.

Why this re-runs MediaPipe instead of reading swing.json: swing.json stores
canonical 3D metres (Section 6.1), which cannot be projected back onto the
image -- the 2D landmarks are never persisted. Extraction settings are taken
from config so the overlay matches the pipeline's own view of the clip, and
frames come from video_io.frames() so any downscaling matches too.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from no_layups import config  # noqa: E402
from no_layups.pipeline import video_io  # noqa: E402

# Section 11.2 -- the same 12 bones the viewer draws.
BONES = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]

LEAD_COLOR = (80, 220, 80)  # BGR: lead side (anatomical left) green
TRAIL_COLOR = (80, 160, 255)  # trail side orange
AXIAL_COLOR = (230, 230, 230)  # nose / midline white
LOW_VIS_COLOR = (60, 60, 220)  # below the Section 7.4 visibility floor, red


def _joint_color(joint: str, visibility: float) -> tuple:
    if visibility < config.VISIBILITY_THRESHOLD:
        return LOW_VIS_COLOR
    if joint.startswith("left_"):
        return LEAD_COLOR
    if joint.startswith("right_"):
        return TRAIL_COLOR
    return AXIAL_COLOR


def _load_keyframes(swing_path: Path | None) -> dict:
    if swing_path is None:
        return {}
    data = json.loads(swing_path.read_text())
    keyframes = data.get("keyframes")
    if not keyframes:
        return {}
    return {int(v): k.upper() for k, v in keyframes.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", help="path to the input video")
    parser.add_argument("-o", "--output", required=True, help="path to write the annotated mp4")
    parser.add_argument("--swing", help="swing.json whose keyframes should be labelled")
    parser.add_argument(
        "--hold",
        type=int,
        default=15,
        help="extra frames to hold on each labelled keyframe (default 15, 0 to disable)",
    )
    args = parser.parse_args()

    video_path = Path(args.video)
    keyframes = _load_keyframes(Path(args.swing) if args.swing else None)

    probe = video_io.probe(video_path)
    decodable = video_io.resolve_decodable_path(video_path)

    writer = None
    pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=config.POSE_MODEL_COMPLEXITY,
        min_detection_confidence=config.POSE_MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=config.POSE_MIN_TRACKING_CONFIDENCE,
    )
    tracked = 0
    try:
        for index, frame in enumerate(video_io.frames(decodable)):
            height, width = frame.shape[:2]
            if writer is None:
                writer = cv2.VideoWriter(
                    args.output,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    probe.fps,
                    (width, height),
                )
                if not writer.isOpened():
                    print(f"error: could not open {args.output} for writing", file=sys.stderr)
                    return 1

            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            canvas = frame.copy()

            points: dict[str, tuple[int, int]] = {}
            visibility: dict[str, float] = {}
            if results.pose_landmarks is not None:
                tracked += 1
                landmarks = results.pose_landmarks.landmark
                for joint in config.JOINTS:
                    lm = landmarks[config.MEDIAPIPE_LANDMARK_INDEX[joint]]
                    points[joint] = (int(lm.x * width), int(lm.y * height))
                    visibility[joint] = float(lm.visibility)

            for a, b in BONES:
                if a in points and b in points:
                    faded = min(visibility[a], visibility[b]) < config.VISIBILITY_THRESHOLD
                    cv2.line(
                        canvas,
                        points[a],
                        points[b],
                        LOW_VIS_COLOR if faded else (200, 200, 200),
                        2,
                        cv2.LINE_AA,
                    )
            for joint, point in points.items():
                color = _joint_color(joint, visibility[joint])
                cv2.circle(canvas, point, 5, color, -1, cv2.LINE_AA)
                cv2.circle(canvas, point, 5, (20, 20, 20), 1, cv2.LINE_AA)

            label = f"frame {index}"
            if not points:
                label += "  NO POSE"
            cv2.putText(
                canvas, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA
            )
            cv2.putText(
                canvas,
                label,
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            event = keyframes.get(index)
            if event is not None:
                cv2.rectangle(canvas, (0, 0), (width - 1, height - 1), (0, 215, 255), 4)
                cv2.putText(
                    canvas,
                    event,
                    (10, height - 16),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 0),
                    5,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    canvas,
                    event,
                    (10, height - 16),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 215, 255),
                    2,
                    cv2.LINE_AA,
                )

            writer.write(canvas)
            if event is not None:
                for _ in range(max(0, args.hold)):
                    writer.write(canvas)
    finally:
        pose.close()
        if writer is not None:
            writer.release()

    if writer is None:
        print("error: no frames decoded", file=sys.stderr)
        return 1

    print(f"wrote {args.output}")
    print(f"  {tracked} frames with a detected pose")
    if keyframes:
        print("  labelled: " + ", ".join(f"{v.lower()}={k}" for k, v in sorted(keyframes.items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
