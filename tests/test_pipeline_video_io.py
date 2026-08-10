import shutil
import subprocess
from pathlib import Path

import pytest

from no_layups.pipeline.errors import PipelineError
from no_layups.pipeline.video_io import (
    ProbeResult,
    _parse_frame_rate,
    frames,
    probe,
    resolve_decodable_path,
    validate,
)

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not on PATH",
)


def _make_clip(path: Path, width: int, height: int, fps: int, duration: float) -> Path:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=size={width}x{height}:rate={fps}:duration={duration}",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        capture_output=True,
        check=True,
    )
    return path


def test_parse_frame_rate_fraction():
    assert _parse_frame_rate("30000/1001") == pytest.approx(29.97, abs=0.01)
    assert _parse_frame_rate("30/1") == 30.0


def test_probe_reads_valid_clip(tmp_path):
    clip = _make_clip(tmp_path / "valid.mp4", 640, 360, 30, 3.0)
    result = probe(clip)
    assert result.width == 640
    assert result.height == 360
    assert result.fps == pytest.approx(30.0, abs=0.1)
    assert result.duration == pytest.approx(3.0, abs=0.2)


def test_validate_accepts_in_spec_clip():
    validate(ProbeResult(duration=5.0, width=640, height=360, fps=30.0, codec_name="h264"))


def test_validate_rejects_too_short():
    with pytest.raises(PipelineError) as exc:
        validate(ProbeResult(duration=1.0, width=640, height=360, fps=30.0, codec_name="h264"))
    assert exc.value.code == "too_short"


def test_validate_rejects_too_long():
    with pytest.raises(PipelineError) as exc:
        validate(ProbeResult(duration=20.0, width=640, height=360, fps=30.0, codec_name="h264"))
    assert exc.value.code == "too_long"


def test_validate_rejects_low_resolution():
    with pytest.raises(PipelineError) as exc:
        validate(ProbeResult(duration=5.0, width=320, height=240, fps=30.0, codec_name="h264"))
    assert exc.value.code == "low_resolution"


def test_validate_rejects_unsupported_fps():
    with pytest.raises(PipelineError) as exc:
        validate(ProbeResult(duration=5.0, width=640, height=360, fps=15.0, codec_name="h264"))
    assert exc.value.code == "unsupported_format"


def test_resolve_decodable_path_returns_same_path_when_already_decodable(tmp_path):
    clip = _make_clip(tmp_path / "valid.mp4", 640, 360, 30, 2.0)
    assert resolve_decodable_path(clip) == clip


def test_frames_downscales_above_max_long_edge(tmp_path):
    clip = _make_clip(tmp_path / "big.mp4", 1920, 1080, 30, 2.0)
    first = next(frames(clip, max_long_edge=1280))
    assert max(first.shape[:2]) == 1280


def test_frames_yields_expected_count(tmp_path):
    clip = _make_clip(tmp_path / "valid.mp4", 640, 360, 30, 2.0)
    all_frames = list(frames(clip))
    assert 55 <= len(all_frames) <= 65
