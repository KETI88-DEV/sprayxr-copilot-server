from __future__ import annotations

import re
from datetime import datetime
from typing import cast

from fastapi.testclient import TestClient

from trainer.demo_api.app import create_app


def test_health_and_sequence_routes() -> None:
    client = TestClient(create_app())

    health_response = client.get("/api/health")
    assert health_response.status_code == 200
    health = cast(dict[str, object], health_response.json())
    assert health["status"] == "ok"
    assert health["servers"] == 1
    assert datetime.fromisoformat(cast(str, health["timestamp"]))

    snapshot_response = client.get("/api/sequence/SP0035-L01")
    assert snapshot_response.status_code == 200
    snapshot = cast(dict[str, object], snapshot_response.json())
    assert snapshot["level_id"] == "SP0035-L01"
    assert snapshot["next_zone_ids"] == ["A-2", "A-3"]

    invalid_response = client.get("/api/sequence/bad level")
    assert invalid_response.status_code == 400
    assert invalid_response.json() == {"detail": "invalid level_id"}


def test_chat_stream_emits_sse_events() -> None:
    client = TestClient(create_app())

    with client.stream(
        "POST",
        "/api/chat",
        json={"message": "다음에 어떤 구역을 칠해야 해?", "session_id": "s-1"},
    ) as response:
        body = "".join(response.iter_text())

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "event: sequence_update" in body
    assert '"next_zone_ids":["A-2","A-3"]' in body
    assert "event: complete" in body
    assert re.search(r"data: \{.*\}", body)
