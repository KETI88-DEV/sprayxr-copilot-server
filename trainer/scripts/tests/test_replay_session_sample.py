"""벤더 세션 샘플 재생 스크립트.

합성 샘플(요청 기록 JSON + 작은 CSV)과 가짜 서버로 본다 — 실제 서버·실데이터 재생은 스크립트 독스트링의 예.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from trainer.scripts import replay_session_sample as rs

_CSV = "﻿분사,Speed (Real),Distance (Real)\nOff,9999,9999\nOn,1000,600\nOn,1200,700\nOff,9999,9999\n"


def _sample(tmp_path: Path) -> rs.Sample:
    sequence = {"session_id": "S1", "user_id": "u", "scenario_id": "3", "zone_id": "9", "sequence_id": "SEQ-001",
                "sequence_setting": {"tool_type": "spray_gun"}, "sequence_info": [], "sequence_result": {}}
    requests = [
        {"endpoint": "POST /sessions/start", "body": {"session_id": "S1", "user_id": "u", "scenario_id": "3"}},
        {"endpoint": "POST /copilot/analyze-sequence", "body": sequence},
        {"endpoint": "POST /copilot/analyze-sequence", "body": dict(sequence, sequence_id="SEQ-002")},
        {"endpoint": "POST /sessions/end", "body": {"session_id": "S1", "user_id": "u", "scenario_id": "3"}},
        {"endpoint": "POST /sessions/complete", "body": {
            "session_id": "S1", "user_id": "u", "scenario_id": "3", "vr_score": "54", "is_pass": "false",
            "avg_thickness_um": "138.91", "_parts": {"telemetry_csv": "s.csv (100 bytes, 별도 첨부)"}}},
    ]
    (tmp_path / "s.csv").write_text(_CSV, encoding="utf-8")
    (tmp_path / "sample.json").write_text(json.dumps({"requests": requests}, ensure_ascii=False), encoding="utf-8")
    return rs.load_sample(tmp_path)


class _Server:
    """받은 요청을 기록한다. complete는 평균 필드를 명세서 p.12대로 **필수**로 받는 서버를 흉내 낸다
    (이 저장소의 서버는 a2fbe37부터 선택으로 받는다 — 그래도 422를 내는 서버에서의 동작을 본다)."""

    def __init__(self) -> None:
        self.seen: list[tuple[str, Any]] = []
        self.polls = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/sessions/complete"):
            body = request.content.decode("utf-8", "replace")
            self.seen.append((path, body))
            if "avg_speed_mms" not in body or "avg_distance_mm" not in body:
                return httpx.Response(422, json={"status": "ERROR"})
            return httpx.Response(200, json={"data": {"session_id": "S1", "report_id": "RPT-1", "report_status": "pending"}})
        if request.method == "GET" and "/reports/" in path:
            self.polls += 1
            status = "pending" if self.polls < 2 else "complete"
            return httpx.Response(200, json={"data": {"report_status": status, "pose_report": {"analysis_status": "available"}}})
        body = json.loads(request.content)
        self.seen.append((path, body))
        if path.endswith("/copilot/query"):
            return httpx.Response(200, json={"data": {"pose_context": {
                "analysis_status": "available", "reason_code": None, "similarity_score": 0.5}}})
        return httpx.Response(200, json={"status": "SUCCESS"})


def _run(tmp_path: Path, server: _Server, **kwargs: Any):
    return rs.replay(_sample(tmp_path), api="http://api.test", transport=httpx.MockTransport(server), poll_sec=0, **kwargs)


def test_bodies_are_sent_unchanged_in_recorded_order(tmp_path: Path) -> None:
    """본문을 고치지 않는다 — 벤더가 보낸 그대로의 처리를 보는 것이 목적이다.

    깨뜨리는 법: analyze-sequence에 `mode_id`를 덧붙이면 첫 단언이 실패한다(실제로 그 때문에 422가 났다).
    """
    server = _Server()
    sample = _sample(tmp_path)

    _run(tmp_path, server)

    sent = [(p, b) for p, b in server.seen if p.endswith("/copilot/analyze-sequence")]
    assert [b for _, b in sent] == [r["body"] for r in sample.requests[1:3]]
    assert [p for p, _ in server.seen] == [
        "/api/v1/sessions/start", "/api/v1/copilot/analyze-sequence", "/api/v1/copilot/analyze-sequence",
        "/api/v1/sessions/end", "/api/v1/sessions/complete",
    ]


def test_complete_without_averages_is_422_by_default(tmp_path: Path) -> None:
    """기본은 벤더 그대로 보낸다 — 필드를 필수로 받는 서버면 422이고 리포트는 없다."""
    rows, report = _run(tmp_path, _Server())

    assert rows[-1]["HTTP"] == 422
    assert report is None
    assert "리포트: 없음" in rs.summarize(rows, report)


def test_fill_averages_uses_spray_frames_only_and_polls_the_report(tmp_path: Path) -> None:
    """`--fill-averages`는 분사 On 프레임만 평균한다(Off의 9999가 섞이면 값이 튄다).

    깨뜨리는 법: `spraying` 필터를 빼면 평균이 5549.5가 된다.
    """
    server = _Server()
    rows, report = _run(tmp_path, server, fill_averages=True)

    complete_body = server.seen[-1][1]
    assert 'name="avg_speed_mms"\r\n\r\n1100.0' in complete_body
    assert 'name="avg_distance_mm"\r\n\r\n650.0' in complete_body
    assert rows[-1]["HTTP"] == 200 and "avg_speed_mms=1100.0" in rows[-1]["비고"]
    assert report is not None and report["report_status"] == "complete"


def test_skip_and_query(tmp_path: Path) -> None:
    """`--skip`한 시퀀스는 보내지도 묻지도 않고, 나머지는 pose_recomm 질의 결과를 남긴다."""
    server = _Server()
    rows, _ = _run(tmp_path, server, query=True, skip=["SEQ-002"])

    queries = [b for p, b in server.seen if p.endswith("/copilot/query")]
    assert [q["context"] for q in queries] == ["SEQ-001"]
    assert queries[0]["task_type"] == "pose_recomm"
    by_seq = {r["시퀀스"]: r for r in rows if r["시퀀스"]}
    assert by_seq["SEQ-001"]["질의 상태"] == "available" and by_seq["SEQ-001"]["유사도"] == 0.5
    assert by_seq["SEQ-002"]["비고"].startswith("업로드 생략") and by_seq["SEQ-002"]["HTTP"] == ""


def test_outputs_are_korean_csv_and_json(tmp_path: Path) -> None:
    rows, report = _run(tmp_path, _Server(), fill_averages=True)
    out = tmp_path / "out"

    rs.write_outputs(rows, report, out, "t")

    assert (out / "t_requests.csv").read_bytes().startswith("﻿요청".encode("utf-8"))
    assert json.loads((out / "t_report.json").read_text(encoding="utf-8"))["report_status"] == "complete"


@pytest.mark.parametrize("bad", ["GET /sessions/start", "POST sessions/start"])
def test_unknown_endpoints_are_rejected(bad: str) -> None:
    with pytest.raises(ValueError):
        rs.endpoint_path(bad)
