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

# Section 7.6 — Stage 6 key-event detection.
TAKEAWAY_SEARCH_FRACTION = 0.4
TAKEAWAY_SPEED_THRESHOLD_MPS = 0.4
TAKEAWAY_CONSECUTIVE_FRAMES = 3
# SPEC DEVIATION (Section 7.6 step 1), same root cause as MIN_BACKSWING_RISE_M
# below: the spec's takeaway trigger is an absolute 0.4 m/s, but m/s is a
# function of the playback timeline, so slow-motion footage never reaches it
# and the clip dies at address detection before the swing gate is ever
# consulted. The trigger is therefore the *smaller* of the spec's absolute
# threshold and this fraction of the clip's own peak lead-wrist speed, which
# scales with the footage. Taking the min means normal-speed footage keeps
# exactly the spec's 0.4 m/s behaviour -- the relative term can only ever make
# detection more sensitive, never less.
TAKEAWAY_SPEED_FRACTION_OF_PEAK = 0.05
# Lowering the speed trigger above buys sensitivity on slow-motion footage at
# the cost of specificity: a golfer's address waggle is slow but not still, and
# on the reference clip it peaks at 0.20 m/s -- under the spec's absolute 0.4
# but well over 5% of that clip's 1.44 m/s peak. Speed alone cannot separate
# the two, so a candidate must also be confirmed by NET displacement: a waggle
# oscillates and returns (net ~ 0), a real takeaway translates away and stays.
# The lead wrist must therefore travel at least this fraction of its own total
# range over the confirmation window before a candidate is accepted.
TAKEAWAY_MIN_NET_DISPLACEMENT_FRACTION = 0.05
# The confirmation window is a fraction of clip length, not a fixed number of
# seconds. A slow-motion clip of the same swing has proportionally more frames,
# so a clip-relative window covers the same portion of the real motion at any
# playback rate, while a fixed 0.5 s window would shrink to nothing at 8x slow
# motion and reject the genuine takeaway along with the waggle.
TAKEAWAY_CONFIRM_WINDOW_FRACTION = 0.05
ADDRESS_LOOKBACK_S = 0.4
TOP_WINDOW_ADDRESS_MARGIN = 5
TOP_WINDOW_IMPACT_MARGIN = 3
MIN_ARGMAX_WINDOW_FRAMES = 4
# SPEC DEVIATION (Section 7.6 step 3): the spec gates "did a swing happen?"
# on lead-wrist speed exceeding an absolute 3.0 m/s. Speed in m/s is a
# function of the playback timeline, so slow-motion footage -- extremely
# common for swing video -- never reaches it (a real 40 m/s swing filmed at
# 240fps and played at 30fps reads as ~5 m/s; the reference clip peaks at
# 1.6 m/s). Distance in metres is invariant to playback rate, so the gate is
# instead "did the lead wrist actually rise into a backswing?".
MIN_BACKSWING_RISE_M = 0.20
# Impact is located as the FIRST return to address level after a genuine
# backswing rise, never as a global extremum over the rest of the clip. On any
# clip that runs through to a full finish, the hands finish HIGHER than they
# ever were at the top of the backswing -- measured on the reference clip, the
# finish peaks at +0.888 m against the backswing top's +0.719 m -- so a global
# argmax lands on the finish and drags impact after it. "First descent" is the
# only anchor that survives a full follow-through.
#
# The wrist counts as back down once it falls to within this fraction of
# MIN_BACKSWING_RISE_M of the address height. It is deliberately not "exactly
# address height": at impact the hands lead the ball slightly and monocular
# depth adds its own offset, so the wrist often never quite returns to where
# it started.
IMPACT_RETURN_FRACTION_OF_RISE = 0.5
# Once the wrist is back down, the low point may sit a little further on. Search
# this fraction of the clip past the crossing for it.
IMPACT_LOW_POINT_SEARCH_FRACTION = 0.1

# Section 8.4 — phase-percent anchors for trajectory normalization.
PHASE_PERCENT_ADDRESS = 0
PHASE_PERCENT_TOP = 40
PHASE_PERCENT_IMPACT = 65
PHASE_PERCENT_LAST_FRAME = 100

# Section 10 — upload limit.
MAX_UPLOAD_BYTES = 100 * 1024 * 1024
