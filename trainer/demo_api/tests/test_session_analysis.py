"""세션 종합 자세 분석(`sessions/complete`의 `pose_report`).

실데이터(`/data/sources/new_format/sample_2026-10-08`)가 없으면 **skip하지 않고 실패한다** — 시뮬레이터가 보낸
시퀀스와 CSV 구간이 같다는 것이 이 설계의 전제이고, 그것은 실제 세션으로만 확인된다.
"""

from __future__ import annotations

import glob
import gzip
import json
import math
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from trainer.demo_api import session_analysis as sa
from trainer.demo_api import v1_router
from trainer.demo_api.sequence_store import SequenceStore
from trainer.demo_api.v1_router import create_v1_router
from trainer.demo_api.v1_store import V1Store

SAMPLE = Path("/data/sources/new_format/sample_2026-10-08")
SESSION = "3_11000346_1"
MCP = "http://mcp.test"


def _csv_bytes() -> bytes:
    return (SAMPLE / "3_11000346_1.csv.gz").read_bytes()


def _vendor_sequences() -> list[dict[str, Any]]:
    payload = json.loads(Path(glob.glob(str(SAMPLE / "*.json"))[0]).read_text(encoding="utf-8"))
    found: list[dict[str, Any]] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if "sequence_info" in node and "sequence_id" in node:
                found.append(node)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(payload)
    return found


class _FakeMcp:
    """실제 MCP 요청 모델로 본문을 검증하고, 본문 출처에 따라 다른 점수를 낸다.

    저장 본문에는 시험이 `sequence_result.marker = "stored"`를 넣는다 — 어느 출처가 채점됐는지 보려고.
    `refuse`에 든 시퀀스는 엔진이 판정을 거절한 것처럼 답한다.
    """

    def __init__(self, refuse: frozenset[str] = frozenset()) -> None:
        self.seen: dict[str, dict[str, Any]] = {}
        self.refuse = refuse

    def __call__(self, request: httpx.Request) -> httpx.Response:
        from pydantic import ValidationError

        from mcp_servers.pose_estimation import SequenceRequest

        body = json.loads(request.content)
        try:
            SequenceRequest.model_validate(body)
        except ValidationError as error:
            return httpx.Response(422, json={"detail": error.errors(include_url=False)})
        sequence_id = body["sequence_id"]
        self.seen[sequence_id] = body
        if sequence_id in self.refuse:
            result = {"analysis_status": "not_available", "reason_code": "skeleton_height_out_of_range"}
        else:
            stored = (body.get("sequence_result") or {}).get("marker") == "stored"
            result = {
                "analysis_status": "available",
                "similarity_score": 0.9 if stored else 0.5,
                "correction_mode": "single",
                "corrections": [{"joint": "RightUpperArm", "correction_text": f"{sequence_id} 오른쪽 위팔을 내리세요"}],
            }
        return httpx.Response(200, json={"status": "success", "result": result, "error": None})


def _store_vendor(store: SequenceStore, body: dict[str, Any]) -> None:
    body = dict(body, sequence_result={"marker": "stored"}, mode_id=1)
    store.record(
        session_key=SESSION, sequence_id=body["sequence_id"], body_sha256=body["sequence_id"],
        frames=len(body["sequence_info"]), received_at="2026-10-08T00:00:00Z",
        body=json.dumps(body, ensure_ascii=False),
    )


def _analyze(store: SequenceStore, mcp: _FakeMcp, csv_bytes: bytes | None) -> sa.SessionPoseAnalysis:
    return sa.analyze_session(
        session_id=SESSION, user_id="u", scenario_id="c", csv_bytes=csv_bytes,
        store=store, mcp_base_url=MCP, transport=httpx.MockTransport(mcp),
    )


def _zone_less(ids: list[str]) -> int:
    vendor = {b["sequence_id"]: b for b in _vendor_sequences()}
    return sum(1 for s in ids if str(vendor[s]["zone_id"]) == sa.NO_ZONE)


def test_csv_spray_runs_reproduce_the_simulators_sequences() -> None:
    """CSV 분사 구간 = 시뮬레이터가 보낸 시퀀스. 번호·구역·프레임 수·관절·건 위치가 같아야 CSV로 빈 시퀀스를 메울 수 있다.

    깨뜨리는 법: 구간 경계를 한 행 밀거나 다른 건 강체를 고르면 프레임 수·위치가 어긋난다.
    """
    rows = sa.read_csv(_csv_bytes())
    mine = sa.csv_sequences(rows, session_id=SESSION, user_id="u", scenario_id="c")
    vendor = _vendor_sequences()

    assert len(vendor) == 32
    assert [b["sequence_id"] for b in mine] == [b["sequence_id"] for b in vendor]
    for theirs, ours in zip(vendor, mine):
        assert ours["zone_id"] == str(theirs["zone_id"]), theirs["sequence_id"]
        assert len(ours["sequence_info"]) == len(theirs["sequence_info"]), theirs["sequence_id"]
        for frame_t, frame_o in zip(theirs["sequence_info"], ours["sequence_info"]):
            assert frame_o["index"] == frame_t["index"]
            assert math.dist(frame_o["tool_pos_rot"]["pos"], frame_t["tool_pos_rot"]["pos"]) < 1e-3
            for joint, point in frame_t["skeleton_joints"].items():
                assert math.dist(frame_o["skeleton_joints"][joint], point) < 1e-3, (theirs["sequence_id"], joint)


def test_query_less_sequences_are_scored_from_the_csv_and_stored_ones_win() -> None:
    """질의가 없었던(=올라오지 않은) 시퀀스도 CSV로 채점한다. 올라온 시퀀스는 그 본문을 쓴다.

    깨뜨리는 법: CSV를 무시하면 total이 2가 되고, 저장 본문보다 CSV를 앞세우면 SEQ-002가 0.5를 받는다.
    """
    store = SequenceStore(":memory:")
    vendor = {b["sequence_id"]: b for b in _vendor_sequences()}
    _store_vendor(store, vendor["SEQ-002"])
    _store_vendor(store, vendor["SEQ-010"])
    mcp = _FakeMcp()

    result = _analyze(store, mcp, _csv_bytes())

    no_zone = _zone_less(list(vendor))
    stored_scored = 2 - _zone_less(["SEQ-002", "SEQ-010"])
    scored = 32 - no_zone
    assert result.counts["total"] == 32
    assert result.counts["from_stored"] == 2 and result.counts["from_csv"] == 30
    assert result.counts.get("no_zone", 0) == no_zone
    assert result.counts["scored"] == scored
    assert mcp.seen["SEQ-002"]["sequence_result"] == {"marker": "stored"}
    assert "mode_id" not in mcp.seen["SEQ-002"]
    expected = (0.9 * stored_scored + 0.5 * (scored - stored_scored)) / scored
    assert result.analysis_status == "available"
    assert result.average_similarity_score == pytest.approx(expected, abs=1e-4)
    report = result.pose_report()
    assert set(report) == {"analysis_status", "algorithm_version", "guideline", "average_similarity_score"}
    # 문장은 분사마다 다르지만 부위는 같다 — 부위로 센다.
    assert f"오른쪽 위팔입니다({scored}회)" in report["guideline"]


def test_unscorable_sequences_leave_the_average_and_are_counted() -> None:
    """엔진이 판정을 거절한 시퀀스는 평균에서 빠지고 사유별로 센다 — 0으로 채우지 않는다.

    깨뜨리는 법: 거절을 0점으로 넣으면 평균이 0.5보다 낮아진다.
    """
    refused = ["SEQ-001", "SEQ-005"]
    assert _zone_less(refused) == 0  # 둘 다 엔진까지 가야 이 시험이 뜻을 가진다
    result = _analyze(SequenceStore(":memory:"), _FakeMcp(refuse=frozenset(refused)), _csv_bytes())

    assert result.average_similarity_score == pytest.approx(0.5)
    assert result.counts["not_scored"] == 2
    assert result.counts["reason:skeleton_height_out_of_range"] == 2
    assert "판정할 수 없는 분사 2회" in result.guideline


def test_a_legacy_csv_falls_back_to_stored_sequences_only() -> None:
    """관절 월드 열이 없는 예전 형식 CSV에서는 시퀀스를 만들지 않는다 — 올라온 것만 채점한다."""
    store = SequenceStore(":memory:")
    _store_vendor(store, _vendor_sequences()[2])

    result = _analyze(store, _FakeMcp(), b"index,time\n1,0\n")

    assert result.counts["total"] == 1 and result.counts["from_stored"] == 1
    assert result.average_similarity_score == pytest.approx(0.9)


def test_no_data_is_unavailable_without_a_number() -> None:
    result = _analyze(SequenceStore(":memory:"), _FakeMcp(), b"not,a,telemetry\n")

    assert result.analysis_status == "unavailable"
    assert result.average_similarity_score is None
    assert result.counts["total"] == 0


def test_a_plain_csv_reads_the_same_as_gzip() -> None:
    raw = _csv_bytes()
    assert sa.read_csv(gzip.decompress(raw)) == sa.read_csv(raw)


# --- 라우터: complete → 비동기 분석 → 리포트 ------------------------------------


def _client(mcp: _FakeMcp, store: SequenceStore) -> TestClient:
    app = FastAPI()
    app.include_router(
        create_v1_router(
            store=V1Store(":memory:"), sequences=store, mcp_base_url=MCP, transport=httpx.MockTransport(mcp),
        )
    )
    return TestClient(app)


def _complete(client: TestClient, csv_bytes: bytes) -> str:
    response = client.post(
        "/api/v1/sessions/complete",
        data={
            "session_id": SESSION, "user_id": "u", "scenario_id": "c", "vr_score": "80", "is_pass": "true",
            "avg_thickness_um": "100", "avg_speed_mms": "1300", "avg_distance_mm": "650",
        },
        files={"telemetry_csv": (f"{SESSION}.csv.gz", csv_bytes, "application/gzip")},
    )
    assert response.status_code == 200, response.text
    assert response.json()["data"]["report_status"] == "pending"
    return str(response.json()["data"]["report_id"])


def test_complete_runs_the_session_analysis_and_the_report_carries_pose() -> None:
    """complete는 세션 종합 분석이다. 응답은 pending, 분석이 끝나면 complete이고 `pose_report`에 평균이 실린다.

    깨뜨리는 법: 백그라운드 분석을 빼면 상태가 pending에 머물고 pose_report가 unavailable이다.
    """
    client = _client(_FakeMcp(), SequenceStore(":memory:"))
    report_id = _complete(client, _csv_bytes())

    status = client.get(f"/api/v1/sessions/{SESSION}/reports/{report_id}").json()["data"]
    detail = client.get(f"/api/v1/reports/{report_id}").json()["data"]

    assert status["report_status"] == "complete"
    assert detail["report"]["report_status"] == "complete"
    assert detail["report"]["pose_report"]["analysis_status"] == "available"
    assert detail["report"]["pose_report"]["average_similarity_score"] == pytest.approx(0.5)
    assert detail["report"]["pose_report"]["algorithm_version"] == sa.ALGORITHM_VERSION
    # 엔진이 없는 영역은 그대로 unavailable(p.15).
    assert detail["report"]["coating_report"]["analysis_status"] == "unavailable"
    assert detail["report"]["sequence_report"]["analysis_status"] == "unavailable"


def test_an_analysis_crash_marks_the_report_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    """분석이 예외로 끝나면 `failed` — 폴링이 pending에 영원히 갇히지 않는다."""

    def boom(**_: Any) -> sa.SessionPoseAnalysis:
        raise RuntimeError("boom")

    monkeypatch.setattr(v1_router, "analyze_session", boom)
    client = _client(_FakeMcp(), SequenceStore(":memory:"))
    report_id = _complete(client, b"index,time\n1,0\n")

    detail = client.get(f"/api/v1/reports/{report_id}").json()["data"]
    assert detail["report"]["report_status"] == "failed"
    assert detail["report"]["pose_report"]["analysis_status"] == "unavailable"
    assert detail["report"]["pose_report"]["average_similarity_score"] is None


def test_a_sequence_report_crash_also_marks_the_report_failed() -> None:
    class BrokenSequenceEngine:
        def report_bodies(self, *_: Any, **__: Any) -> dict[str, Any]:
            raise RuntimeError("sequence boom")

    app = FastAPI()
    app.include_router(create_v1_router(
        store=V1Store(":memory:"), sequences=SequenceStore(":memory:"),
        mcp_base_url=MCP, transport=httpx.MockTransport(_FakeMcp()),
        sequence_engine=BrokenSequenceEngine(),  # type: ignore[arg-type]
    ))
    client = TestClient(app)
    report_id = _complete(client, b"index,time\n1,0\n")

    detail = client.get(f"/api/v1/reports/{report_id}").json()["data"]
    assert detail["report"]["report_status"] == "failed"
