"""GET /api/maintenance 와 점검 배너 주입."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from trainer.demo_api.app import create_app
from trainer.demo_api.maintenance import read_maintenance
from trainer.demo_api.portal_navigation import portal_document

NOW = datetime(2026, 11, 17, 0, 0, tzinfo=timezone.utc)
ACTIVE = {"active": True, "what": "vllm", "reason": "vLLM 교체", "until": "2026-11-17T01:30:00Z",
          "started_at": "2026-11-17T00:00:00Z", "impact": "상세 분석(27B) 일시 중단"}


def test_read_maintenance_cases(tmp_path):
    path = tmp_path / "maintenance.json"
    assert read_maintenance(path, NOW) == {"active": False}  # 파일 없음
    path.write_text(json.dumps(ACTIVE), encoding="utf-8")
    assert read_maintenance(path, NOW) == {"active": True, "what": "vllm", "reason": "vLLM 교체",
                                           "impact": "상세 분석(27B) 일시 중단", "until": "2026-11-17T01:30:00Z"}
    assert read_maintenance(path, NOW + timedelta(hours=2)) == {"active": False}  # 만료
    path.write_text('{"active": true, "until": ', encoding="utf-8")
    assert read_maintenance(path, NOW) == {"active": False}  # 깨진 JSON
    for broken in ({**ACTIVE, "until": "내일"}, {**ACTIVE, "until": "2026-11-17T10:30"}, {k: v for k, v in ACTIVE.items() if k != "until"},
                   {**ACTIVE, "active": "true"}, [ACTIVE]):
        path.write_text(json.dumps(broken), encoding="utf-8")
        assert read_maintenance(path, NOW) == {"active": False}, broken
    path.write_text(json.dumps({**ACTIVE, "impact": 3, "by": "KURE"}), encoding="utf-8")
    assert read_maintenance(path, NOW)["impact"] == "" and "by" not in read_maintenance(path, NOW)


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DEMO_LIVE_MODE", "off")
    monkeypatch.setenv("MAINTENANCE_FILE", str(tmp_path / "ops" / "maintenance.json"))
    with TestClient(create_app()) as test_client:
        yield test_client


def test_endpoint_is_hidden_200_and_follows_the_file(client, tmp_path):
    response = client.get("/api/maintenance")
    assert response.status_code == 200 and response.json() == {"active": False}
    assert response.headers["cache-control"] == "no-store"
    (tmp_path / "ops").mkdir()
    future = (datetime.now(timezone.utc) + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    (tmp_path / "ops" / "maintenance.json").write_text(json.dumps({**ACTIVE, "until": future}), encoding="utf-8")
    assert client.get("/api/maintenance").json()["active"] is True
    (tmp_path / "ops" / "maintenance.json").write_text("{", encoding="utf-8")
    assert client.get("/api/maintenance").json() == {"active": False}
    assert "/api/maintenance" not in client.get("/openapi.json").json()["paths"]
    script = client.get("/assets/maintenance_banner.js")
    assert script.status_code == 200 and "bannerText" in script.text


def test_banner_is_injected_on_docs_motion_chat_only():
    page = "<html><head></head><body><main></main></body></html>"
    for active in ("docs", "motion", "chat"):
        document = portal_document(page, active)
        assert document.count('id="copilot-maintenance"') == 1
        assert document.index('id="copilot-navigation"') < document.index('id="copilot-maintenance"') < document.index("<main>")
        assert '<script defer src="/assets/maintenance_banner.js"></script></body>' in document
    for active in ("database", "feedback"):
        document = portal_document(page, active)
        assert 'id="copilot-maintenance"' not in document and "maintenance_banner.js" not in document


def test_served_pages_carry_the_banner(client):
    for url in ("/docs", "/query-chat", "/motion-viewer"):
        response = client.get(url)
        assert response.status_code == 200, url
        assert 'id="copilot-maintenance"' in response.text and "/assets/maintenance_banner.js" in response.text, url
