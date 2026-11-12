import json

from fastapi.testclient import TestClient
import httpx
import pytest

from trainer.coating_explanation_service import service
from trainer.coating_explanation_service.tests.test_contract import sample, selection


@pytest.mark.parametrize("response_model,done_reason,status", [("qwen3:4b", "stop", 200), ("other", "stop", 503), ("qwen3:4b", "length", 503)])
def test_model_identity_and_completed_generation_required(monkeypatch, tmp_path, response_model, done_reason, status):
    monkeypatch.setenv("EXPLANATION_MODEL_DIGEST", "sha256:" + "a" * 64)
    monkeypatch.setenv("EXPLANATION_LOG_DIR", str(tmp_path))

    class Backend:
        def __init__(self, **kwargs): pass
        def get(self, path):
            return httpx.Response(200, request=httpx.Request("GET", "http://qwen/api/tags"), json={"models": [{"name": "qwen3:4b", "digest": "a" * 64, "details": {"quantization_level": "Q4_K_M"}}]})
        def post(self, path, json):
            return httpx.Response(200, request=httpx.Request("POST", "http://qwen/api/chat"), json={"model": response_model, "done": True, "done_reason": done_reason, "message": {"content": __import__("json").dumps(selection())}, "eval_count": 12})

    monkeypatch.setattr(service.httpx, "Client", Backend)
    with TestClient(service.create_app()) as client:
        response = client.post("/explain", json={"assessment": sample(), "kind": "zone"})
    assert response.status_code == status
    records = list(tmp_path.glob("*.json"))
    if status == 200:
        assert response.json()["status"] == "generated"
        assert json.loads(records[0].read_text())["result"]["inference"]["model"] == "qwen3:4b"
    else:
        assert not records


def test_withheld_grade_cannot_admit_injected_coaching():
    assessment = sample(None)
    catalog = service.catalogs(assessment)
    assert catalog["causes"] == catalog["actions"] == []
