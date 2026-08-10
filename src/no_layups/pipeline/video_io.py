import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from .. import config
from .errors import PipelineError


@dataclass
class ProbeResult:
    duration: float
    width: int
    height: int
    fps: float
    codec_name: str


def _parse_frame_rate(value: str) -> float:
    num, _, den = value.partition("/")
    den = den or "1"
    return float(num) / float(den)


def probe(video_path: Path) -> ProbeResult:
    """Section 7.1 step 1: ffprobe the container for duration/width/height/fps."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-show_entries",
        "stream=width,height,r_frame_rate,codec_name",
        "-of",
        "json",
        str(video_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise PipelineError("unsupported_format", f"could not probe video: {exc}") from exc

    data = json.loads(result.stdout)
    stream = next(
        (s for s in data.get("streams", []) if "width" in s and "height" in s), None
    )
    if stream is None or "format" not in data:
        raise PipelineError("unsupported_format", "no decodable video stream found")

    return ProbeResult(
        duration=float(data["format"]["duration"]),
        width=int(stream["width"]),
        height=int(stream["height"]),
        fps=_parse_frame_rate(stream["r_frame_rate"]),
        codec_name=stream.get("codec_name", ""),
    )


def validate(probe_result: ProbeResult) -> None:
    """Section 7.1 step 2."""
    if probe_result.duration < config.MIN_DURATION_S:
        raise PipelineError(
            "too_short",
            f"video duration {probe_result.duration:.1f}s is below the "
            f"{config.MIN_DURATION_S}s minimum",
        )
    if probe_result.duration > config.MAX_DURATION_S:
        raise PipelineError(
            "too_long",
            f"video duration {probe_result.duration:.1f}s exceeds the "
            f"{config.MAX_DURATION_S}s maximum",
        )
    if probe_result.width < config.MIN_WIDTH or probe_result.height < config.MIN_HEIGHT:
        raise PipelineError(
            "low_resolution",
            f"video resolution {probe_result.width}x{probe_result.height} is below "
            f"the {config.MIN_WIDTH}x{config.MIN_HEIGHT} minimum",
        )
    if not (config.MIN_FPS <= probe_result.fps <= config.MAX_FPS):
        raise PipelineError(
            "unsupported_format",
            f"video fps {probe_result.fps:.1f} is outside the supported "
            f"{config.MIN_FPS}-{config.MAX_FPS} range",
        )


def resolve_decodable_path(video_path: Path) -> Path:
    """Section 7.1 step 3: transcode via ffmpeg if cv2 can't read the first frame."""
    cap = cv2.VideoCapture(str(video_path))
    ok = cap.read()[0]
    cap.release()
    if ok:
        return video_path

    tmp_dir = Path(tempfile.mkdtemp(prefix="no_layups_"))
    transcoded = tmp_dir / "transcoded.mp4"
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-c:v",
        "libx264",
        "-crf",
        "20",
        "-an",
        str(transcoded),
    ]
    try:
        subprocess.run(cmd, capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise PipelineError(
            "decode_failed", f"could not decode video even after transcoding: {exc}"
        ) from exc

    cap = cv2.VideoCapture(str(transcoded))
    ok = cap.read()[0]
    cap.release()
    if not ok:
        raise PipelineError("decode_failed", "could not decode video even after transcoding")
    return transcoded


def frames(decodable_path: Path, max_long_edge: int = config.MAX_LONG_EDGE) -> Iterator[np.ndarray]:
    """Section 7.1 step 5: yields BGR frames, downscaled if the long edge exceeds max_long_edge."""
    cap = cv2.VideoCapture(str(decodable_path))
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            long_edge = max(h, w)
            if long_edge > max_long_edge:
                scale = max_long_edge / long_edge
                frame = cv2.resize(
                    frame,
                    (max(1, round(w * scale)), max(1, round(h * scale))),
                    interpolation=cv2.INTER_AREA,
                )
            yield frame
    finally:
        cap.release()
