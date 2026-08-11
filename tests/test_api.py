"""Section 13.1 test_api.py. The pipeline is stubbed throughout (Section 7
preamble): these tests never run real video through MediaPipe."""

import copy
import json
import time

import pytest
from fastapi.testclient import TestClient

import no_layups.jobs as jobs_module
from no_layups import config
from no_layups.main import app
from no_layups.pipeline.errors import PipelineError

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

STUB_SWING = {
    "version": 1,
    "meta": {
        "fps": 30.0,
        "frame_count": 10,
        "source_width": 640,
        "source_height": 360,
        "handedness": "right",
        "created_at": "2026-01-01T00:00:00Z",
        "warnings": [],
        "keyframe_source": "auto",
    },
    "joints": JOINTS,
    "frames": [
        {
            "t": i / 30.0,
            "pos": {
                joint: [0.01 * i + 0.1 * j, 0.02 * i + 0.05 * j, 0.03 * i - 0.02 * j]
                for j, joint in enumerate(JOINTS)
            },
        }
        for i in range(10)
    ],
    "keyframes": {"address": 1, "top": 5, "impact": 8},
    "metrics": None,
    "trajectories": None,
}


def _stub_success(video_path, handedness, on_progress):
    on_progress("validating")
    on_progress("computing")
    swing = copy.deepcopy(STUB_SWING)
    swing["meta"]["handedness"] = handedness
    return swing


def _stub_too_long(video_path, handedness, on_progress):
    on_progress("validating")
    raise PipelineError("too_long", "video duration 20.0s exceeds the 15.0s maximum")


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "JOBS_DIR", tmp_path / "jobs")
    monkeypatch.setattr(config, "SWINGS_DIR", tmp_path / "swings")
    monkeypatch.setattr(config, "UPLOADS_DIR", tmp_path / "uploads")
    return TestClient(app)


def _wait_for_terminal(client, job_id, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        res = client.get(f"/api/jobs/{job_id}")
        assert res.status_code == 200
        job = res.json()
        if job["status"] in ("done", "error"):
            return job
        time.sleep(0.02)
    raise TimeoutError("job did not reach a terminal status in time")


def _upload(client):
    res = client.post(
        "/api/swings",
        files={"video": ("clip.mp4", b"fake video bytes", "video/mp4")},
        data={"handedness": "right"},
    )
    assert res.status_code == 202
    return res.json()["job_id"]


def _upload_and_wait_done(client):
    job_id = _upload(client)
    job = _wait_for_terminal(client, job_id)
    assert job["status"] == "done"
    return job_id


def test_upload_reaches_done_and_swing_matches_stub(client, monkeypatch):
    monkeypatch.setattr(jobs_module, "run_pipeline", _stub_success)

    job_id = _upload(client)
    job = _wait_for_terminal(client, job_id)
    assert job["swing_id"] == job_id

    swing = client.get(f"/api/swings/{job_id}").json()
    expected = copy.deepcopy(STUB_SWING)
    expected["meta"]["handedness"] = "right"
    assert swing == expected


def test_unknown_job_id_404(client):
    assert client.get("/api/jobs/does-not-exist").status_code == 404


def test_unknown_swing_id_404(client):
    assert client.get("/api/swings/does-not-exist").status_code == 404


def test_keyframes_invalid_ordering_422(client, monkeypatch):
    monkeypatch.setattr(jobs_module, "run_pipeline", _stub_success)
    job_id = _upload_and_wait_done(client)

    res = client.post(f"/api/swings/{job_id}/keyframes", json={"address": 5, "top": 5, "impact": 8})
    assert res.status_code == 422


def test_keyframes_valid_update_recomputes_metrics(client, monkeypatch):
    monkeypatch.setattr(jobs_module, "run_pipeline", _stub_success)
    job_id = _upload_and_wait_done(client)

    res = client.post(f"/api/swings/{job_id}/keyframes", json={"address": 1, "top": 4, "impact": 7})
    assert res.status_code == 200
    body = res.json()
    assert body["keyframes"] == {"address": 1, "top": 4, "impact": 7}
    assert body["meta"]["keyframe_source"] == "manual"
    assert body["metrics"] is not None
    assert body["trajectories"] is not None
    assert len(body["comparison"]["metrics"]) == 8

    # Persisted file matches the Section 5.2 shape exactly -- "comparison"
    # is a Section 5.3 concept and must not leak into swing.json.
    reloaded = client.get(f"/api/swings/{job_id}").json()
    assert reloaded["keyframes"] == {"address": 1, "top": 4, "impact": 7}
    assert reloaded["meta"]["keyframe_source"] == "manual"
    assert "comparison" not in reloaded


def test_pipeline_error_marks_job_error_with_code(client, monkeypatch):
    monkeypatch.setattr(jobs_module, "run_pipeline", _stub_too_long)

    job_id = _upload(client)
    job = _wait_for_terminal(client, job_id)

    assert job["status"] == "error"
    assert job["error"]["code"] == "too_long"
    assert job["swing_id"] is None


def test_reference_endpoint_returns_bundled_reference(client):
    res = client.get("/api/reference")
    assert res.status_code == 200
    body = res.json()
    assert body["keyframes"] is not None
    assert body["metrics"] is not None
    assert len(body["joints"]) == 13


def test_comparison_404_when_swing_has_no_keyframes_yet(client, monkeypatch):
    monkeypatch.setattr(jobs_module, "run_pipeline", _stub_success)
    job_id = _upload_and_wait_done(client)  # STUB_SWING ships with metrics: None

    assert client.get(f"/api/swings/{job_id}/comparison").status_code == 404


def test_comparison_available_after_keyframes_are_set(client, monkeypatch):
    monkeypatch.setattr(jobs_module, "run_pipeline", _stub_success)
    job_id = _upload_and_wait_done(client)
    client.post(f"/api/swings/{job_id}/keyframes", json={"address": 1, "top": 4, "impact": 7})

    res = client.get(f"/api/swings/{job_id}/comparison")
    assert res.status_code == 200
    comparison = res.json()
    assert len(comparison["metrics"]) == 8
    assert set(comparison["trajectories"]["user"].keys()) == {"shoulder_turn", "spine_tilt"}
    assert len(comparison["phase_to_frame"]["user"]) == 101
    assert len(comparison["phase_to_frame"]["reference"]) == 101
