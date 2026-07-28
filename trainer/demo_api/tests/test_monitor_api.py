from __future__ import annotations

import json

import anyio
from fastapi import FastAPI
from fastapi.testclient import TestClient

from trainer.demo_api.monitor import MonitorRecorder
from trainer.demo_api.monitor_api import create_monitor_router, monitor_sse_body


def _parse_frame(raw: bytes) -> tuple[str, dict[str, object]]:
    lines = raw.decode("utf-8").split("\n")
    payload: dict[str, object] = json.loads(lines[1].removeprefix("data: "))
    return lines[0].removeprefix("event: "), payload


def test_events_snapshot_returns_ordered_events() -> None:
    recorder = MonitorRecorder(maxlen=3)
    app = FastAPI()
    app.include_router(create_monitor_router(recorder))
    client = TestClient(app)

    _ = recorder.record("s1", "start", "input", {"a": 1})
    _ = recorder.record("s1", "start", "output", {"a": 2})

    response = client.get("/api/monitor/events", params={"cursor": 0, "limit": 10})

    assert response.status_code == 200
    body = response.json()
    assert [event["seq"] for event in body["events"]] == [1, 2]
    assert body["next_cursor"] == 2
    assert body["dropped_before"] is None


def test_events_snapshot_reports_gap_after_overflow() -> None:
    recorder = MonitorRecorder(maxlen=3)
    app = FastAPI()
    app.include_router(create_monitor_router(recorder))
    client = TestClient(app)

    for index in range(5):
        _ = recorder.record("s1", "runtime", "input", {"i": index})

    body = client.get("/api/monitor/events", params={"cursor": 0, "limit": 10}).json()

    assert body["dropped_before"] == 3
    assert [event["seq"] for event in body["events"]] == [3, 4, 5]


def test_sse_body_emits_gap_then_replay_then_live() -> None:
    async def _run() -> list[tuple[str, dict[str, object]]]:
        recorder = MonitorRecorder(maxlen=3)
        for index in range(5):
            _ = recorder.record("s1", "runtime", "input", {"i": index})

        frames: list[tuple[str, dict[str, object]]] = []
        body = monitor_sse_body(recorder, cursor=0)
        with anyio.fail_after(2.0):
            # gap marker + the 3 events still held by the bounded buffer
            for _ in range(4):
                frames.append(_parse_frame(await anext(body)))
            # recorded after the generator subscribed -> forwarded live
            _ = recorder.record("s1", "session_end", "output", {"done": True})
            frames.append(_parse_frame(await anext(body)))
        await body.aclose()
        return frames

    frames = anyio.run(_run)

    assert [name for name, _ in frames] == ["gap", "monitor", "monitor", "monitor", "monitor"]
    assert frames[0][1]["dropped_before"] == 3
    assert [payload["seq"] for _, payload in frames[1:]] == [3, 4, 5, 6]
    assert frames[4][1]["phase"] == "session_end"


def test_sse_body_without_gap_starts_at_cursor_and_unsubscribes_on_close() -> None:
    async def _run() -> tuple[list[tuple[str, dict[str, object]]], int]:
        recorder = MonitorRecorder(maxlen=10)
        for index in range(3):
            _ = recorder.record("s1", "start", "input", {"i": index})

        frames: list[tuple[str, dict[str, object]]] = []
        body = monitor_sse_body(recorder, cursor=2)
        with anyio.fail_after(2.0):
            frames.append(_parse_frame(await anext(body)))
        await body.aclose()

        # Closing the stream must leave no subscriber behind.
        _ = recorder.record("s1", "runtime", "input", {"after": True})
        return frames, len(recorder.snapshot(cursor=0, limit=10).events)

    frames, recorded = anyio.run(_run)

    assert [name for name, _ in frames] == ["monitor"]
    assert frames[0][1]["seq"] == 3
    assert recorded == 4
