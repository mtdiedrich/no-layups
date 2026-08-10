import os
from pathlib import Path

# Section 5.1 — the 13 joints, in the order they must appear everywhere.
JOINTS = [
    "nose",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

MEDIAPIPE_LANDMARK_INDEX = {
    "nose": 0,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
}


def lead(name: str, handedness: str) -> str:
    """Section 6.2: lead side = anatomical left for right-handed golfers, right for left-handed."""
    side = "left" if handedness == "right" else "right"
    return f"{side}_{name}"


def trail(name: str, handedness: str) -> str:
    side = "right" if handedness == "right" else "left"
    return f"{side}_{name}"


# Section 4 — runtime data directories, resolved relative to CWD (repo root).
DATA_DIR = Path(os.environ.get("GOLF_DATA_DIR", "data"))
UPLOADS_DIR = DATA_DIR / "uploads"
JOBS_DIR = DATA_DIR / "jobs"
SWINGS_DIR = DATA_DIR / "swings"
REFERENCE_PATH = Path("reference") / "reference.json"

# Section 7.1 — Stage 1 validation thresholds.
MIN_DURATION_S = 2.0
MAX_DURATION_S = 15.0
MIN_WIDTH = 640
MIN_HEIGHT = 360
MIN_FPS = 24.0
MAX_FPS = 60.0
MAX_LONG_EDGE = 1280

# Section 7.2 — Stage 2 camera-motion check.
CAMERA_CHECK_STRIDE = 5
CAMERA_CHECK_MAX_SAMPLES = 40
CAMERA_CHECK_MAX_CORNERS = 200
CAMERA_CHECK_QUALITY_LEVEL = 0.01
CAMERA_CHECK_MIN_DISTANCE = 20
CAMERA_CHECK_BORDER_FRACTION = 0.15
CAMERA_CHECK_MIN_CORNERS = 20
CAMERA_CHECK_DISPLACEMENT_THRESHOLD_PX = 2.0

# Section 7.3 — Stage 3 pose extraction.
POSE_MODEL_COMPLEXITY = 2
POSE_MIN_DETECTION_CONFIDENCE = 0.5
POSE_MIN_TRACKING_CONFIDENCE = 0.5

# Section 7.4 — Stage 4 gap-fill + smoothing.
VISIBILITY_THRESHOLD = 0.4
GAP_FILL_MAX_FRACTION_OF_FPS = 0.25
SAVGOL_POLYORDER = 2
POOR_TRACKING_MAX_MISSING_FRACTION = 0.25

# Section 7.6 — Stage 6 key-event detection (address portion used by Stage 5's
# Pass A in M2; top/impact detection lands in M4).
TAKEAWAY_SEARCH_FRACTION = 0.4
TAKEAWAY_SPEED_THRESHOLD_MPS = 0.4
TAKEAWAY_CONSECUTIVE_FRAMES = 3
ADDRESS_LOOKBACK_S = 0.4
