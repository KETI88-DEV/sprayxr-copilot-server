"""Lifecycle retries and durable report execution survive client/server retries."""
from __future__ import annotations

import gzip
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from trainer.demo_api import session_analysis, v1_router
from trainer.demo_api.anomaly_stream import AnomalyBroadcaster
from trainer.demo_api.report_jobs import ReportJobs, save_artifact
from trainer.demo_api.sequence_store import SequenceStore
from trainer.demo_api.v1_store import V1Store


def make_client(state: V1Store, bus: AnomalyBroadcaster | None = None):
    router = v1_router.create_v1_router(
        store=state, sequences=SequenceStore(":memory:"), broadcaster=bus,
        transport=httpx.MockTransport(lambda request: httpx.Response(503)),
    )
    app = FastAPI()
    app.include_router(router)
    return TestClient(app), router


START = {
    "session_id": "S1", "user_id": "W1", "scenario_id": "3",
    "started_at": "2026-10-30T00:00:00Z", "target_thickness_um": 150,
    "worker": {"skill_level": "novice", "handedness": "right"},
}


def complete(client, raw=b"a,b\n1,2\n", *, score="80", image=None, image_name="../../height.png"):
    files = {"telemetry_csv": ("../../client.csv.gz", raw, "application/octet-stream")}
    if image is not None:
        files["height_map_image"] = (image_name, image, "image/png")
    return client.post("/api/v1/sessions/complete", data={
        "session_id": "S1", "user_id": "W1", "scenario_id": "3", "vr_score": score,
        "is_pass": "true", "avg_thickness_um": "150",
    }, files=files)


def test_start_retry_keeps_completion_and_anomaly_history():
    state = V1Store(":memory:")
    bus = AnomalyBroadcaster()
    client, _ = make_client(state, bus)
    assert client.post("/api/v1/sessions/start", json=START).status_code == 200
    bus.publish(session_id="S1", sequence_id="SEQ-001", anomaly_type="distance_deviation", payload={})
    end = {"session_id": "S1", "user_id": "W1", "scenario_id": "3", "ended_at": "2026-10-30T01:00:00Z"}
    assert client.post("/api/v1/sessions/end", json=end).status_code == 200
    retry = client.post("/api/v1/sessions/start", json=START).json()["data"]
    assert retry["session_status"] == "COMPLETED"
    assert retry["ended_at"] == end["ended_at"]
    assert len(bus.since("S1", last_event_id=0)) == 1


@pytest.mark.parametrize("changes", [
    {"user_id": "W2"}, {"scenario_id": "4"},
    {"started_at": "2026-10-30T00:01:00Z"}, {"target_thickness_um": 200},
])
def test_start_conflict_returns_409_without_mutating_state_or_users(changes):
    state = V1Store(":memory:")
    client, _ = make_client(state)
    client.post("/api/v1/sessions/start", json=START)
    before = state.get_session("S1")
    response = client.post("/api/v1/sessions/start", json=START | changes)
    assert response.status_code == 409
    assert set(response.json()) == {"status", "code", "message", "data"}
    assert state.get_session("S1") == before
    assert state.get_user("W2") is None


def test_end_wrong_owner_returns_409_and_retries_keep_first_end():
    state = V1Store(":memory:")
    client, _ = make_client(state)
    client.post("/api/v1/sessions/start", json=START)
    end = {"session_id": "S1", "user_id": "W1", "scenario_id": "3", "ended_at": "2026-10-30T01:00:00Z"}
    assert client.post("/api/v1/sessions/end", json=end | {"user_id": "W2"}).status_code == 409
    first = client.post("/api/v1/sessions/end", json=end).json()["data"]
    second = client.post("/api/v1/sessions/end", json=end | {"ended_at": "2026-10-30T02:00:00Z"}).json()["data"]
    assert first == second


def test_complete_retry_reuses_report_after_database_reopen(tmp_path):
    path = str(tmp_path / "state.db")
    state = V1Store(path)
    client, _ = make_client(state)
    first = complete(client).json()["data"]
    state.close()
    reopened = V1Store(path)
    client, _ = make_client(reopened)
    second = complete(client).json()["data"]
    assert second["report_id"] == first["report_id"]
    assert second["report_status"] == "complete"
    assert complete(client, score="81").json()["data"]["report_id"] != first["report_id"]


def test_startup_hook_recovers_durable_pending_input(tmp_path, monkeypatch):
    path = str(tmp_path / "state.db")
    state = V1Store(path)
    with monkeypatch.context() as patch:
        patch.setattr(ReportJobs, "run", lambda *args: None)
        client, _ = make_client(state)
        result = complete(client).json()["data"]
    report_id = result["report_id"]
    old = state.get_completion(report_id)
    assert old.report_status == "pending"
    assert (state.artifact_dir / old.payload["artifacts"]["telemetry"]).read_bytes() == b"a,b\n1,2\n"
    state.close()
    reopened = V1Store(path)
    client, router = make_client(reopened)
    assert router.recover_report_jobs() == 1
    deadline = time.monotonic() + 5
    while reopened.get_completion(report_id).report_status == "pending" and time.monotonic() < deadline:
        time.sleep(0.01)
    assert reopened.get_completion(report_id).report_status == "complete"
    assert complete(client).json()["data"]["report_id"] == report_id


def test_running_old_report_is_not_failed_by_polling(tmp_path, monkeypatch):
    state = V1Store(str(tmp_path / "state.db"))
    raw = save_artifact(state.artifact_dir, b"a,b\n")
    row = state.create_completion(session_id="S1", requested_at="2020-01-01T00:00:00Z", payload={
        "user_id": "W1", "scenario_id": "3", "artifacts": {"telemetry": raw},
    })
    client, _ = make_client(state)
    assert client.get(f"/api/v1/sessions/S1/reports/{row.report_id}").json()["data"]["report_status"] == "pending"


def test_parallel_report_retries_execute_analysis_only_once(tmp_path):
    state = V1Store(str(tmp_path / "state.db"))
    raw = save_artifact(state.artifact_dir, b"a,b\n")
    row = state.create_completion(session_id="S1", requested_at="t", payload={
        "user_id": "W1", "scenario_id": "3", "artifacts": {"telemetry": raw},
    })
    entered = threading.Event()
    release = threading.Event()
    calls = []
    def analyze(report_id, **kwargs):
        calls.append(kwargs["csv_bytes"])
        entered.set()
        assert release.wait(5)
        state.finish_completion(report_id, status="complete", analysis={})
    jobs = ReportJobs(state, analyze)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(jobs.run, row.report_id)
        assert entered.wait(5)
        second = pool.submit(jobs.run, row.report_id)
        second.result(timeout=5)
        release.set()
        first.result(timeout=5)
    assert calls == [b"a,b\n"]


def test_missing_persisted_input_fails_cleanly(tmp_path):
    state = V1Store(str(tmp_path / "state.db"))
    row = state.create_completion(session_id="S1", requested_at="t", payload={
        "user_id": "W1", "scenario_id": "3", "artifacts": {"telemetry": "0" * 64},
    })
    ReportJobs(state, lambda *args, **kwargs: pytest.fail("must not analyze missing input")).run(row.report_id)
    assert state.get_completion(row.report_id).report_status == "failed"
    client, _ = make_client(state)
    report = client.get(f"/api/v1/reports/{row.report_id}").json()["data"]["report"]
    for area in ("coating_report", "sequence_report", "pose_report"):
        assert report[area]["analysis_status"] == "unavailable"
        assert report[area]["reason_code"] == "report_input_unavailable"
        assert "입력 자료" in report[area]["guideline"]
    assert report["coating_report"]["distribution"] is None
    assert report["pose_report"]["average_similarity_score"] is None


def test_two_processes_do_not_analyze_the_same_report_twice(tmp_path):
    path = str(tmp_path / "state.db")
    state = V1Store(path)
    raw = save_artifact(state.artifact_dir, b"a,b\n")
    row = state.create_completion(session_id="S1", requested_at="t", payload={
        "user_id": "W1", "scenario_id": "3", "artifacts": {"telemetry": raw},
    })
    counter = tmp_path / "calls"
    script = """
import sys, time
from pathlib import Path
from trainer.demo_api.v1_store import V1Store
from trainer.demo_api.report_jobs import ReportJobs
state = V1Store(sys.argv[1])
def analyze(report_id, **kwargs):
    with Path(sys.argv[3]).open('a') as handle:
        handle.write('analyzed\\n')
    time.sleep(0.2)
    state.finish_completion(report_id, status='complete', analysis={})
ReportJobs(state, analyze).run(sys.argv[2])
"""
    children = [subprocess.Popen([sys.executable, "-c", script, path, row.report_id, str(counter)]) for _ in range(2)]
    try:
        assert [child.wait(timeout=10) for child in children] == [0, 0]
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
                child.wait()
    assert counter.read_text().splitlines() == ["analyzed"]
    assert state.get_completion(row.report_id).report_status == "complete"


def test_storage_fallback_does_not_acknowledge_a_durable_report(monkeypatch):
    state = V1Store(":memory:")
    state.report_storage_available = False
    client, _ = make_client(state)
    assert complete(client).status_code == 503
    assert state.pending_completions() == []


@pytest.mark.parametrize("kind", ["raw", "gzip", "png"])
def test_complete_upload_limits_reject_before_creating_report(kind, monkeypatch):
    monkeypatch.setattr(v1_router, "CSV_UPLOAD_LIMIT", 16)
    monkeypatch.setattr(v1_router, "PNG_UPLOAD_LIMIT", 16)
    monkeypatch.setattr(session_analysis, "CSV_INFLATED_LIMIT", 32)
    state = V1Store(":memory:")
    client, _ = make_client(state)
    raw = b"x" * 17 if kind == "raw" else b"a,b\n"
    if kind == "gzip":
        monkeypatch.setattr(v1_router, "CSV_UPLOAD_LIMIT", 128)
        raw = gzip.compress(b"x" * 33)
    response = complete(client, raw=raw, image=b"x" * 17 if kind == "png" else None)
    assert response.status_code == 413
    assert state.pending_completions() == []


def test_session_target_is_forwarded_to_coating_analysis(monkeypatch):
    targets = []
    def coating(**kwargs):
        targets.append(kwargs["target_thickness_um"])
        return None
    monkeypatch.setattr(v1_router, "analyze_session_coating", coating)
    state = V1Store(":memory:")
    router = v1_router.create_v1_router(store=state, sequences=SequenceStore(":memory:"), coating_engine=object())
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    client.post("/api/v1/sessions/start", json=START)
    complete(client)
    assert targets == [150.0]


def test_surface_filename_survives_initial_analysis_and_restart(tmp_path, monkeypatch):
    names = []
    def coating(**kwargs):
        names.append(kwargs["height_map_name"])
        return None
    monkeypatch.setattr(v1_router, "analyze_session_coating", coating)
    path = str(tmp_path / "state.db")
    state = V1Store(path)
    def client_with_coating(store):
        router = v1_router.create_v1_router(store=store, sequences=SequenceStore(":memory:"), coating_engine=object())
        app = FastAPI()
        app.include_router(router)
        return TestClient(app), router
    client, _ = client_with_coating(state)
    first = complete(client, image=b"same-image", image_name="../../Level_Map02.png").json()["data"]
    assert names == ["Level_Map02.png"]
    assert state.get_completion(first["report_id"]).payload["height_map_name"] == "Level_Map02.png"
    with monkeypatch.context() as patch:
        patch.setattr(ReportJobs, "run", lambda *args: None)
        pending = complete(client, image=b"same-image", image_name="../Level_Map03.png").json()["data"]
    assert pending["report_id"] != first["report_id"]
    state.close()
    reopened = V1Store(path)
    client, router = client_with_coating(reopened)
    router.recover_report_jobs()
    deadline = time.monotonic() + 5
    while reopened.get_completion(pending["report_id"]).report_status == "pending" and time.monotonic() < deadline:
        time.sleep(0.01)
    assert names == ["Level_Map02.png", "Level_Map03.png"]
    assert reopened.get_completion(pending["report_id"]).report_status == "complete"
