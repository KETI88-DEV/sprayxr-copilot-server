#!/usr/bin/env python3
# pyright: reportAny=false, reportUnusedCallResult=false

from __future__ import annotations

from fastapi.testclient import TestClient

from trainer.pose_lab.app import create_app
from trainer.pose_lab.service import PassRef, RawPass

PARKED = (-694.61, 701.29, 698.25)
STILL = (-2.99, 4.66, -0.27)

QUERY = {"worker": "A", "date": "20250901", "level": "10", "session": "s1", "pass_index": 0}
REFERENCE = {"worker": "B", "date": "20250902", "level": "10", "session": "s2", "pass_index": 0}


def _rows(shift: float, descending: bool = False) -> list[dict[str, str]]:
    """GPU도 실데이터도 없이 도는 결정론 입력."""
    rows: list[dict[str, str]] = []
    for i in range(30):
        step = i * 0.03
        rise = -step if descending else step
        row: dict[str, str] = {
            "인덱스": str(i),
            "시간(초)": f"{i * 0.0111:.4f}",
            "분사": "On",
            "Ray point X": "1.26",
            " Ray Point Y": str(4.2 + step),
            " Ray Point Z": str(0.7 + rise),
            "Ray Texture UV X": "1024",
            "Ray Texture UV Y": "512",
        }
        for axis, value in zip("XYZ", (0.5, 4.2 + step, 0.8 + shift)):
            row[f"카메라 Position {axis}"] = str(value)
        for axis, value in zip("XYZ", (0.85, 4.2 + step, 0.65 + shift)):
            row[f"한길숏건 Position {axis}"] = str(value)
        for axis, value in zip("XYZ", (0.67, 4.2 + step, 0.79 + shift)):
            row[f"플래쉬 Position {axis}"] = str(value)
        for axis, value in zip("XYZ", PARKED):
            row[f"한길폴건 Position {axis}"] = str(value)
            row[f"반도폴건 Position {axis}"] = str(value)
        for axis, value in zip("XYZ", STILL):
            row[f"반도숏건 Position {axis}"] = str(value)
        rows.append(row)
    return rows


class FakeSource:
    def __init__(self, mismatch: bool = False) -> None:
        self._mismatch = mismatch

    def workers(self) -> list[str]:
        return ["A", "B"]

    def load(self, ref: PassRef) -> RawPass:
        if ref.worker == "B":
            return RawPass(rows=_rows(0.0))
        # 0.3m: 오른손/머리 sigma가 CORRECTION_SIGMA_THRESHOLD(2.0)를 넘도록 벌린
        # 값이다. 그래야 각도 미표시 불변량을 실제로 통과하는 교정 문장으로 검증한다.
        return RawPass(rows=_rows(0.3, descending=self._mismatch))


def test_health_reports_ready() -> None:
    client = TestClient(create_app())
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_compare_rejects_a_malformed_body_with_422() -> None:
    client = TestClient(create_app())
    response = client.post("/api/compare", json={"query": {}, "reference": {}})
    assert response.status_code == 422


def test_compare_reports_both_axes() -> None:
    """두 축을 병기하는 것이 이 프로그램의 존재 이유다. 하나만 오면 실패다."""
    client = TestClient(create_app(source=FakeSource()))
    response = client.post("/api/compare", json={"query": QUERY, "reference": REFERENCE})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["measured"]["basis"] == "tracked"
    assert body["joint"]["basis"] == "ik_derived"
    assert isinstance(body["measured"]["score"], float)
    assert body["measured"]["improvement"]["split"] is None
    assert isinstance(body["measured"]["improvement"]["same"], float)


def test_joint_axis_never_borrows_the_measured_score() -> None:
    """복사하면 두 축이 정의상 같아져 IK 솔버 거동 판별이 불가능해진다(스펙 D1)."""
    client = TestClient(create_app(source=FakeSource()))
    body = client.post("/api/compare", json={"query": QUERY, "reference": REFERENCE}).json()
    assert body["joint"]["status"] == "not_wired"
    assert body["joint"]["score"] is None
    assert body["joint"]["improvement"] is None


def test_compare_emits_no_angle_in_corrections() -> None:
    """REQ-19 게이트 통과 전까지 각도 지시를 내지 않는다. 교정 문장이 실제로 나와야
    이 불변량이 검증된다 — 빈 리스트를 도는 assert는 아무것도 확인하지 않는다."""
    client = TestClient(create_app(source=FakeSource()))
    body = client.post("/api/compare", json={"query": QUERY, "reference": REFERENCE}).json()
    assert len(body["measured"]["corrections"]) > 0
    for line in body["measured"]["corrections"]:
        assert "도" not in line
    assert body["joint"]["corrections"] == []


def test_compare_refuses_when_motion_keys_differ() -> None:
    """의도가 다른 동작은 비교하지 않는다. 억지로 붙이면 의도 차이가 오차로 보고된다."""
    client = TestClient(create_app(source=FakeSource(mismatch=True)))
    body = client.post("/api/compare", json={"query": QUERY, "reference": REFERENCE}).json()
    assert body["status"] == "unavailable"
    assert body["query_key"] != body["reference_key"]
