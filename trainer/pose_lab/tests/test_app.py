#!/usr/bin/env python3
# pyright: reportAny=false, reportUnusedCallResult=false

from __future__ import annotations

from fastapi.testclient import TestClient

from trainer.pose_lab import service
from trainer.pose_lab.app import create_app
from trainer.pose_lab.service import PassRef, RawPass

PARKED = (-694.61, 701.29, 698.25)
STILL = (-2.99, 4.66, -0.27)

QUERY = {"worker": "A", "date": "20250901", "level": "10", "session": "s1", "pass_index": 0}
REFERENCE = {"worker": "B", "date": "20250902", "level": "10", "session": "s2", "pass_index": 0}


def _rows(
    shift: float, descending: bool = False, overlap_left_at: int | None = None
) -> list[dict[str, str]]:
    """GPU도 실데이터도 없이 도는 결정론 입력.

    overlap_left_at은 그 프레임에서 플래쉬를 카메라에 겹쳐 놓는다. 실측 |머리-왼손|
    최소가 0.000 m라(설계 §2.2) 실데이터에서 상시 발생하는 조건이고, §5.4는 그때
    "왼손 지표만 제외"하라고 못박는다.
    """
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
        # 겹친 프레임은 |머리-왼손| = 0.000 m라 왼손만 마스킹된다.
        left = (
            (0.5, 4.2 + step, 0.8 + shift)
            if i == overlap_left_at
            else (0.67, 4.2 + step, 0.79 + shift)
        )
        for axis, value in zip("XYZ", left):
            row[f"플래쉬 Position {axis}"] = str(value)
        for axis, value in zip("XYZ", PARKED):
            row[f"한길폴건 Position {axis}"] = str(value)
            row[f"반도폴건 Position {axis}"] = str(value)
        for axis, value in zip("XYZ", STILL):
            row[f"반도숏건 Position {axis}"] = str(value)
        rows.append(row)
    return rows


class FakeSource:
    def __init__(
        self,
        mismatch: bool = False,
        overlap_left_at: int | None = None,
        rayless: bool = False,
    ) -> None:
        self._mismatch = mismatch
        self._overlap_left_at = overlap_left_at
        self._rayless = rayless

    def workers(self) -> list[str]:
        return ["A", "B"]

    def load(self, ref: PassRef) -> RawPass:
        if ref.worker == "B":
            return RawPass(rows=_rows(0.0))
        # 0.3m: 오른손/머리 sigma가 CORRECTION_SIGMA_THRESHOLD(2.0)를 넘도록 벌린
        # 값이다. 그래야 각도 미표시 불변량을 실제로 통과하는 교정 문장으로 검증한다.
        rows = _rows(
            0.3,
            descending=self._mismatch,
            overlap_left_at=self._overlap_left_at,
        )
        if self._rayless:
            # Ray가 한 번도 맞지 않은 패스. 착탄점이 없으니 키를 만들 수 없다.
            rows = [
                {key: value for key, value in row.items() if not key.strip().startswith("Ray")}
                for row in rows
            ]
        return RawPass(rows=rows)


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


def test_compare_exposes_a_structured_finding_next_to_corrections() -> None:
    """VLM이 최종 문구를 지으려면 표본 수 같은 근거가 필요하다. 문장만으로는 안 된다."""
    client = TestClient(create_app(source=FakeSource()))
    body = client.post("/api/compare", json={"query": QUERY, "reference": REFERENCE}).json()
    corrections = body["measured"]["corrections"]
    finding = body["measured"]["finding"]
    assert len(corrections) == 1
    assert finding is not None
    assert finding["axis"] in ("wall", "vertical", "lateral")
    assert isinstance(finding["magnitude_cm"], float)
    assert isinstance(finding["sigma"], float)
    assert isinstance(finding["samples"], int) and finding["samples"] > 0
    # 문장과 구조체가 같은 이탈에서 나왔다는 것을 겉에서도 확인한다.
    if finding["direction"]:
        assert finding["direction"] in corrections[0]


def test_joint_axis_finding_stays_none() -> None:
    """관절 축은 배선되지 않았다. finding도 다른 literal 필드처럼 값을 빌려오지 않는다."""
    client = TestClient(create_app(source=FakeSource()))
    body = client.post("/api/compare", json={"query": QUERY, "reference": REFERENCE}).json()
    assert body["joint"]["finding"] is None


def test_compare_refuses_when_motion_keys_differ() -> None:
    """의도가 다른 동작은 비교하지 않는다. 억지로 붙이면 의도 차이가 오차로 보고된다."""
    client = TestClient(create_app(source=FakeSource(mismatch=True)))
    body = client.post("/api/compare", json={"query": QUERY, "reference": REFERENCE}).json()
    assert body["status"] == "unavailable"
    assert body["reason"] == "motion_key_mismatch"
    assert body["query_key"] is not None and body["reference_key"] is not None
    assert body["query_key"] != body["reference_key"]


def test_unmeasurable_pass_is_not_reported_as_an_intent_mismatch() -> None:
    """키가 None인 것은 "재지 못했다"이지 "의도가 다르다"가 아니다.

    착탄점이 있는 분사 프레임이 2개 미만이면 키가 없다. 그것을 동작 불일치로 적으면
    재지 못한 것을 재서 다르다고 말한 것이 된다.
    """
    client = TestClient(create_app(source=FakeSource(rayless=True)))
    body = client.post("/api/compare", json={"query": QUERY, "reference": REFERENCE}).json()
    assert body["status"] == "unavailable"
    assert body["reason"] == "insufficient_frames"
    assert body["query_key"] is None


def test_left_axis_survives_a_single_masked_frame() -> None:
    """한 프레임의 왼손이 빠졌다고 패스 전체의 왼손을 버리지 않는다(설계 §5.4).

    실측 |머리-왼손| 최소가 0.000 m라 패스 단위로 버리는 규칙은 실데이터에서 상시
    발동한다. 대신 그 점이 몇 개의 격자점 위에 서 있는지를 응답이 말해야 한다.
    """
    client = TestClient(create_app(source=FakeSource(overlap_left_at=0)))
    body = client.post("/api/compare", json={"query": QUERY, "reference": REFERENCE}).json()
    assert body["status"] == "ok"
    deltas = {delta["point"]: delta for delta in body["measured"]["deltas"]}
    assert "left" in deltas
    assert 0 < deltas["left"]["samples"] < deltas["right"]["samples"]


def test_gate_is_not_asserted_from_provisional_spreads() -> None:
    """게이트의 뜻은 "각도 지시가 이제 정당하다"다.

    상수 두 개(임시 산포 0.05 m / DEFAULT_ERROR_M)로 나눈 값은 언제나 pass가 나온다.
    설계 §10은 REQ-19 각도 유효화를 미판정으로 두고 D4는 게이트를 옮기지 말라고
    못박는다. 잰 분포가 없으면 판정을 내걸지 않는다.
    """
    client = TestClient(create_app(source=FakeSource()))
    body = client.post("/api/compare", json={"query": QUERY, "reference": REFERENCE}).json()
    assert body["measured"]["gate"] is None
    assert body["measured"]["gate_reason"] == "no_reference_distribution"
    assert body["measured"]["normalization"] == "provisional"


def test_gate_is_judged_once_measured_spreads_arrive() -> None:
    """임계 논리는 그대로다. 없는 것은 입력뿐이라 산포를 주면 판정이 나온다."""
    body = service.compare(
        FakeSource(),
        PassRef(**QUERY),
        PassRef(**REFERENCE),
        spreads={"right": 0.06, "head": 0.06, "left": 0.06},
    )
    measured = body["measured"]
    assert isinstance(measured, dict)
    assert measured["gate"] == "pass"
    assert measured["gate_reason"] is None
    assert measured["normalization"] == "measured"


def test_response_discloses_the_assumed_yaw() -> None:
    """동작 키의 ②④는 몸 기준 이름을 달고 있지만 요를 0으로 고정해 만든 값이다.

    실측 요 추출은 별개 작업이다. 그때까지는 응답이 그 가정을 말해야 한다.
    """
    client = TestClient(create_app(source=FakeSource()))
    body = client.post("/api/compare", json={"query": QUERY, "reference": REFERENCE}).json()
    assert body["yaw_assumed"] == 0.0

    mismatched = TestClient(create_app(source=FakeSource(mismatch=True)))
    unavailable = mismatched.post(
        "/api/compare", json={"query": QUERY, "reference": REFERENCE}
    ).json()
    assert unavailable["yaw_assumed"] == 0.0
