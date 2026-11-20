"""Exam inspection reads registered inputs; fresh extraction is bounded and separate."""
import csv
import gzip
import hashlib
import importlib
import io
import json
import sys
import time

import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from trainer.demo_api.db_admin_auth import COOKIE, DBAdminAuth, configure_credentials
from trainer.demo_api.training_data_catalog import TrainingDataCatalog


IDENT = "a" * 32
BASE = f"/admin/api/training-data/{IDENT}"
ORIGIN = {"Origin": "http://testserver"}


def build_record(tmp_path, *, process="spray", compressed=False, relative=None):
    sources, output = tmp_path / "sources", tmp_path / "datasets"
    sources.mkdir(); (output / "records").mkdir(parents=True)
    name = "record.csv.gz" if compressed else "record.csv"
    source = sources / name
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(["시간(초)", "분사", "Zone ID", "Ray Texture Map Number", "distance (input)",
                     "speed (input)", "distance (real)", "speed (real)", "cur_sequence_heatmap"])
    writer.writerows([[0, "Off", "1", 1, 0, 0, 0, 0, ""],
                      [1, "On", "1", 1, 100, 200, 110, 210, "PRIVATE_BASE64_" * 600],
                      [2, "On", "1", 1, 200, 400, 210, 410, ""]])
    content = stream.getvalue().encode("utf-8")
    source.write_bytes(gzip.compress(content) if compressed else content)
    distribution = {"count": 2, "mean": 125, "median": 125, "p25": 110, "p75": 140}
    features = {"frame_count": 3, "duration_seconds": 2, "sequence_count": 1,
                "spraying_frame_count": 2, "mapped_spraying_frame_count": 2,
                "unmapped_spraying_frames": 0, "zone_id_available": True,
                "global_metrics": {"distance_mm": distribution},
                "zones": [{"map_num": 1, "zone_id": "1", "frame_count": 2,
                           "distance_mm": distribution}], "zone_order": [{"map_num": 1, "zone_id": "1"}],
                "sequence_heatmap_count": 0, "sequence_heatmaps": [{"data": "DO_NOT_RETURN_PAYLOAD"}]}
    metadata = {"dataset_id": IDENT, "dataset_name": "Exam record", "worker_id": "worker-example",
                "worker_name": "등록 작업자", "role": "worker", "qualification": "unknown",
                "process": process, "scenario_id": "4", "session_id": "session-example",
                "content_type": "learning", "quality_status": "complete"}
    coating = {"status": "available", "semantics": "저장된 최종 상태", "maps": [{"map_num": 1,
               "status": "available", "summary": {"mean_um": 180, "target_basis": "stored_target"},
               "histogram": [{"min_um": 0, "max_um": 200, "count": 2}],
               "image_url": "javascript:untrusted", "thickness_image_url": "/stored-preview",
               "zones": [{"zone_id": "1", "summary": {"mean_um": 170, "median_um": 165}}]}]}
    artifact = {"schema_version": "record-features-1.2", "dataset_id": IDENT, "metadata": metadata,
                "features": features, "coating": coating, "quality": {"status": "complete", "issues": []},
                "provenance": {"source_csv": {"relative_path": relative or name, "filename": name,
                               "bytes": source.stat().st_size}, "heightmaps": [{"relative_path": "map.png"}]}}
    record = output / "records" / f"{IDENT}.json"
    payload = json.dumps(artifact, ensure_ascii=False).encode("utf-8"); record.write_bytes(payload)
    row = {**metadata, "json_bytes": len(payload), "artifact_sha256": hashlib.sha256(payload).hexdigest()}
    (output / "catalog.json").write_text(json.dumps({"records": [row]}), encoding="utf-8")
    return TrainingDataCatalog(output, sources), source, record


@pytest.fixture
def make_client(tmp_path):
    module = importlib.import_module("trainer.demo_api.training_data_inspection")
    created = []

    def make(*, process="spray", compressed=False, relative=None, **limits):
        folder = tmp_path / str(len(created)); folder.mkdir()
        catalog, source, record = build_record(folder, process=process, compressed=compressed, relative=relative)
        auth_path = folder / "test-auth.json"
        configure_credentials(auth_path, "exam-admin", "test-password")
        auth = DBAdminAuth(auth_path)
        store = module.TrainingDataInspection(catalog, **limits)
        app = FastAPI()

        @app.post("/test-login")
        def login(request: Request):
            token, session = auth.login(request, "exam-admin", "test-password")
            response = JSONResponse({"csrf_token": session.csrf})
            response.set_cookie(COOKIE, token, httponly=True)
            return response

        app.include_router(module.create_training_inspection_router(auth, store))
        client = TestClient(app); created.append((client, store))
        return client, store, source, record

    yield make
    for client, store in created:
        store.close(); client.close()


def login(client):
    result = client.post("/test-login", headers=ORIGIN)
    assert result.status_code == 200
    return {**ORIGIN, "X-DB-CSRF": result.json()["csrf_token"]}


def rewrite_record(record, change):
    data = json.loads(record.read_text(encoding="utf-8")); change(data)
    content = json.dumps(data, ensure_ascii=False).encode(); record.write_bytes(content)
    index_path = record.parent.parent / "catalog.json"
    index = json.loads(index_path.read_text())
    index["records"][0].update(json_bytes=len(content), artifact_sha256=hashlib.sha256(content).hexdigest())
    index_path.write_text(json.dumps(index), encoding="utf-8")


def wait_job(client, url):
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        response = client.get(url)
        assert response.status_code == 200, response.text
        state = response.json()
        if state["status"] in {"completed", "failed"}:
            return state
        time.sleep(.05)
    pytest.fail("bounded exam extraction did not finish")


def test_public_inspection_is_stored_information_with_bounded_csv_preview(make_client):
    client, _, source, record = make_client()
    before = source.read_bytes(), record.read_bytes()
    response = client.get(BASE + "/inspection")
    assert response.status_code == 200
    data = response.json()
    assert data["metadata"]["qualification"] == "unknown" and data["metadata"]["role"] == "worker"
    assert data["scope"]["result_kind"] == "stored_record_features"
    assert data["summary"]["frame_count"] == 3 and data["summary"]["spray_interval_count"] == 1
    assert next(metric for metric in data["summary"]["metrics"] if metric["key"] == "distance_mm_mean")["value"] == 125
    assert data["source_preview"]["sampled_rows"] == 3
    assert "cur_sequence_heatmap" in data["source_preview"]["omitted_columns"]
    assert "PRIVATE_BASE64" not in response.text and "DO_NOT_RETURN_PAYLOAD" not in response.text
    assert len(data["source_preview"]["rows"]) <= 8 and len(data["source_preview"]["columns"]) <= 24
    assert "javascript:" not in response.text
    assert data["links"]["motion"] == f"/motion-viewer?dataset_id={IDENT}"
    assert (source.read_bytes(), record.read_bytes()) == before


def test_zone_uses_stored_metrics_first_visit_and_stored_coating_without_pose_fabrication(make_client):
    client, _, _, _ = make_client()
    data = client.get(BASE + "/inspection").json()
    zone = data["zones"][0]
    assert zone["map_num"] == 1 and zone["zone_id"] == "1"
    assert zone["sequence"]["first_visit_index"] == 1
    assert zone["coating"]["basis"] == "stored" and zone["coating"]["summary"]["mean_um"] == 170
    assert data["summary"]["pose"]["status"] == "unavailable"
    assert data["summary"]["direction"]["status"] == "unavailable"
    assert all(metric["value"] is None and metric["status"] == "unavailable"
               for metric in zone["metrics"] if metric["group"] == "speed_real_mms")


def test_blasting_never_presents_stored_texture_as_coating_thickness(make_client):
    client, _, _, _ = make_client(process="blasting")
    data = client.get(BASE + "/inspection").json()
    assert data["coating"]["status"] == "not_applicable"
    assert data["zones"][0]["coating"]["status"] == "not_applicable"
    assert not data["zones"][0]["coating"]["metrics"]


def test_unregistered_and_unsafe_ids_never_open_arbitrary_sources(make_client):
    client, _, _, _ = make_client(relative="../../outside.csv")
    data = client.get(BASE + "/inspection").json()
    assert data["source_preview"]["status"] == "unavailable"
    assert data["reconvert"]["available"] is False
    assert client.get("/admin/api/training-data/" + "b" * 32 + "/inspection").status_code == 404
    assert client.get("/admin/api/training-data/not-an-id/inspection").status_code == 404
    headers = login(client)
    assert client.post(BASE + "/reconvert", headers=headers).status_code == 409


def test_reconversion_requires_admin_csrf_and_accepts_no_source_path_override(make_client):
    client, _, _, _ = make_client()
    assert client.post(BASE + "/reconvert").status_code == 401
    headers = login(client)
    assert client.post(BASE + "/reconvert", headers=ORIGIN).status_code == 403
    assert client.post(BASE + "/reconvert", headers={**headers, "Origin": "http://other.test"}).status_code == 403
    assert client.post(BASE + "/reconvert", headers=headers, json={"source_path": "/etc/passwd"}).status_code == 422


def test_source_size_limit_is_explicit_and_refuses_job_before_extraction(make_client):
    client, _, _, _ = make_client(max_source_bytes=64)
    data = client.get(BASE + "/inspection").json()
    assert data["reconvert"]["available"] is False and data["reconvert"]["reason"] == "source_too_large"
    response = client.post(BASE + "/reconvert", headers=login(client))
    assert response.status_code == 413 and response.json()["detail"] == "source_too_large"


def test_real_single_record_extraction_is_memory_only_and_compares_feature_values(make_client):
    client, _, source, record = make_client()
    before = source.read_bytes(), record.read_bytes()
    headers = login(client)
    started = client.post(BASE + "/reconvert", headers=headers)
    assert started.status_code == 202
    job = started.json(); url = BASE + "/reconvert/" + job["job_id"]
    assert len(job["job_id"]) == 32
    assert client.post(BASE + "/reconvert", headers=headers).status_code == 409
    done = wait_job(client, url)
    assert done["status"] == "completed", done
    assert done["result_available"] is True and done["comparison"]["basis"] == "feature_values"
    assert "global_metrics" in done["comparison"]["changed_fields"]
    response = client.get(done["download_url"])
    assert response.status_code == 200 and response.headers["cache-control"] == "no-store"
    fresh = response.json()
    assert fresh["result_kind"] == "exam_sample_fresh_features"
    assert fresh["features"]["frame_count"] == 3
    assert fresh["features"]["global_metrics"]["distance_mm"]["mean"] == 150
    assert fresh["stored_analysis"]["coating"]["status"] == "stored_only"
    assert fresh["stored_analysis"]["pose"]["status"] == "unavailable"
    assert "PRIVATE_BASE64" not in response.text
    other = "/admin/api/training-data/" + "b" * 32 + "/reconvert/" + job["job_id"]
    assert client.get(other).status_code == 404 and client.get(other + "/download").status_code == 404
    assert (source.read_bytes(), record.read_bytes()) == before


def test_job_status_download_require_auth_and_job_dataset_binding(make_client):
    client, _, _, _ = make_client()
    url = BASE + "/reconvert/" + "b" * 32
    assert client.get(url).status_code == 401
    assert client.get(url + "/download").status_code == 401
    login(client)
    assert client.get(url).status_code == 404
    assert client.get(BASE + "/reconvert/unsafe-job").status_code == 404


def test_completed_job_expires_without_changing_registered_data(make_client):
    now = [1000.0]
    client, _, _, _ = make_client(clock=lambda: now[0], job_ttl_seconds=5)
    started = client.post(BASE + "/reconvert", headers=login(client)).json()
    url = BASE + "/reconvert/" + started["job_id"]
    assert wait_job(client, url)["status"] == "completed"
    now[0] = 1006
    assert client.get(url).status_code == 404 and client.get(url + "/download").status_code == 404


def test_compressed_source_expansion_is_bounded(make_client):
    client, _, _, _ = make_client(compressed=True, max_source_bytes=1000)
    response = client.post(BASE + "/reconvert", headers=login(client))
    assert response.status_code == 202
    done = wait_job(client, BASE + "/reconvert/" + response.json()["job_id"])
    assert done["status"] == "failed" and done["reason"] == "source_expanded_too_large"


def test_result_memory_limit_fails_without_publishing_partial_features(make_client):
    client, _, _, _ = make_client(max_result_bytes=64)
    response = client.post(BASE + "/reconvert", headers=login(client))
    url = BASE + "/reconvert/" + response.json()["job_id"]
    done = wait_job(client, url)
    assert done["status"] == "failed" and done["reason"] == "result_too_large"
    assert done["result_available"] is False and client.get(url + "/download").status_code == 409


def test_worker_deadline_terminates_its_own_extraction_and_releases_slot(make_client):
    client, _, _, _ = make_client(job_timeout_seconds=.001)
    headers = login(client)
    response = client.post(BASE + "/reconvert", headers=headers)
    done = wait_job(client, BASE + "/reconvert/" + response.json()["job_id"])
    assert done["status"] == "failed" and done["reason"] == "extraction_timeout"
    assert client.post(BASE + "/reconvert", headers=headers).status_code == 202


def test_malformed_optional_stored_structures_become_unavailable_not_server_errors(make_client):
    client, _, _, record = make_client()
    def malformed(data):
        data["metadata"] = []
        data["provenance"] = {"source_csv": ["untrusted"]}
        data["coating"] = {"maps": {"untrusted": 1}}
        data["features"]["zones"] = {"untrusted": 1}
        data["features"]["zone_order"] = {"untrusted": 1}
    rewrite_record(record, malformed)
    response = client.get(BASE + "/inspection")
    assert response.status_code == 200
    assert response.json()["source_preview"]["status"] == "unavailable"
    assert response.json()["zones"] == []
    assert response.json()["summary"]["zone_count"] is None
    assert response.json()["coating"]["status"] == "unavailable"


def test_missing_global_metrics_never_get_filled_from_one_zone(make_client):
    client, _, _, record = make_client()
    rewrite_record(record, lambda data: data["features"].pop("global_metrics"))
    data = client.get(BASE + "/inspection").json()
    assert all(metric["status"] == "unavailable" and metric["value"] is None for metric in data["summary"]["metrics"])
    assert next(metric for metric in data["zones"][0]["metrics"] if metric["key"] == "distance_mm_mean")["value"] == 125


def test_changed_source_is_rejected_by_the_real_worker(make_client, monkeypatch):
    module = importlib.import_module("trainer.demo_api.training_data_inspection")
    client, _, source, _ = make_client()
    original = module.subprocess.Popen
    def change_before_launch(*args, **kwargs):
        source.write_bytes(source.read_bytes() + b"\n")
        return original(*args, **kwargs)
    monkeypatch.setattr(module.subprocess, "Popen", change_before_launch)
    response = client.post(BASE + "/reconvert", headers=login(client))
    done = wait_job(client, BASE + "/reconvert/" + response.json()["job_id"])
    assert done["status"] == "failed" and done["reason"] == "source_changed"


def test_normal_existing_record_integrity_guard_is_preserved(make_client):
    client, _, _, record = make_client()
    record.write_bytes(record.read_bytes() + b" ")
    assert client.get(BASE + "/inspection").status_code == 404
    assert client.post(BASE + "/reconvert", headers=login(client)).status_code == 404


def test_retained_job_count_is_bounded_and_evicts_oldest_result(make_client):
    client, _, _, _ = make_client()
    headers = login(client)
    urls = []
    for _ in range(3):
        response = client.post(BASE + "/reconvert", headers=headers)
        assert response.status_code == 202
        urls.append(BASE + "/reconvert/" + response.json()["job_id"])
        assert wait_job(client, urls[-1])["status"] == "completed"
    assert client.get(urls[0]).status_code == 404
    assert client.get(urls[1]).status_code == 200 and client.get(urls[2]).status_code == 200


def test_worker_crash_is_not_misreported_as_a_measured_data_or_memory_fault(make_client, monkeypatch):
    module = importlib.import_module("trainer.demo_api.training_data_inspection")
    client, _, _, _ = make_client()
    original = module.subprocess.Popen
    monkeypatch.setattr(module.subprocess, "Popen", lambda args, **kwargs:
                        original([sys.executable, "-c", "raise SystemExit(7)"], **kwargs))
    response = client.post(BASE + "/reconvert", headers=login(client))
    done = wait_job(client, BASE + "/reconvert/" + response.json()["job_id"])
    assert done["status"] == "failed" and done["reason"] == "extraction_worker_failed"
