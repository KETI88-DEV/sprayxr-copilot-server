#!/usr/bin/env python3
# pyright: reportAny=false, reportUnusedCallResult=false

from __future__ import annotations

import math

from fastapi.testclient import TestClient

from trainer.pose_lab import service
from trainer.pose_lab.app import create_app
from trainer.pose_lab.service import PassRef, RawPass

PARKED = (-694.61, 701.29, 698.25)
STILL = (-2.99, 4.66, -0.27)
# 정면 방위 +60°가 나오는 HMD 회전(아래 _rows 주석 참조).
HEAD_ROTATION = (0.0, 30.0, 0.0)
EXPECTED_YAW = math.radians(60.0)

QUERY = {"worker": "A", "date": "20250901", "level": "10", "session": "s1", "pass_index": 0}
REFERENCE = {"worker": "B", "date": "20250902", "level": "10", "session": "s2", "pass_index": 0}


def _rows(
    shift: float, descending: bool = False, overlap_left_at: int | None = None
) -> list[dict[str, str]]:
    """GPU도 실데이터도 없이 도는 결정론 입력.

    overlap_left_at은 그 프레임에서 플래쉬를 카메라에 겹쳐 놓는다. 실측 |머리-왼손|
    최소가 0.000 m라(설계 §2.2) 실데이터에서 상시 발생하는 조건이고, §5.4는 그때
    "왼손 지표만 제외"하라고 못박는다.

    `shift`는 오른손/머리/왼손에만 걸리고 착탄점(Ray Point)에는 걸리지 않는 수직
    오프셋이다(유니티 Y-up이므로 월드 Y에 싣는다 — 옛 코드는 이것을 월드 Z에 실었는데,
    `motion_key.body_placement`의 수평이 (index 0, index 2)로 바뀌면서 Z에 실린
    shift가 head-hit 수평 델타에 새어 들어가 근/원 판정을 오염시켰다. Y는
    `world_axes.horizontal_of`가 아예 읽지 않는 축이라 이 값이 몸 기준 좌우·근원
    판정(motion_key)에 새지 않는다 — 옛 설계 의도(오른손/머리 sigma만 벌리고 동작
    키는 건드리지 않는다)를 그대로 지킨다.

    `step`(벽을 따라가는 진행)과 `rise`(오르내림, `descending`이 부호를 정한다)도
    마찬가지로 옛 Z-up 자리에 그대로 있었다(리뷰 A5) — `step`이 Y(이제 수직)에,
    `rise`가 Z(이제 수평)에 실려 있어 착탄점 궤적이 실제로는 거의 수직으로
    솟구치고 `descending`은 수평 성분의 부호만 바꿨다. Y·Z를 맞바꿔 `rise`를
    Y(수직 — 오르내림이 진짜 오르내림이 되게)에, `step`을 Z(수평 — 벽을 따라가는
    진행)에 싣는다. `head`/`gun`/`flash`의 Y도 더는 `step`을 따라가지 않는다 —
    `4.2 + step + shift`(사람 키가 4~5m가 되던 값, 리뷰 A5)를 `0.8/0.65/0.79 +
    shift`(사람 크기의 고정 높이)로 바꾸고, 대신 Z에 `4.2 + step`을 실어 착탄점의
    수평 진행을 따라가게 한다(머리·오른손이 스트로크를 따라간다는 원래 의도는
    유지된다).
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
            " Ray Point Y": str(0.7 + rise),
            " Ray Point Z": str(4.2 + step),
            "Ray Texture UV X": "1024",
            "Ray Texture UV Y": "512",
        }
        for axis, value in zip("XYZ", (0.5, 0.8 + shift, 4.2 + step)):
            row[f"카메라 Position {axis}"] = str(value)
        # 실측 요를 태우는 HMD 회전. `measured_pose.EULER_ORDER`(YXZ) 규약에서
        # 전방 = (sin RotY·cos RotX, −sin RotX, cos RotY·cos RotX)이고,
        # `world_axes.yaw_of`는 수평(X,Z) 성분으로 방위를 잰다(유니티 Y-up). RotX=0
        # 이면 전방=(sin RotY, 0, cos RotY)로 수평이라 yaw=atan2(cos RotY, sin RotY)
        # =90°−RotY다(항등식은 test_measured_pose.test_facing_yaw_reads_the_
        # convention_the_data_chose 참고). (RotX, RotY, RotZ) = (0, 30, 0)이면
        # 정면 방위가 정확히 +60°다. 요가 0일 때와 몸 기준 축(②④)이 둘 다 달라지는
        # 값을 골랐다.
        for axis, value in zip("XYZ", HEAD_ROTATION):
            row[f"카메라 Rotation {axis}"] = str(value)
        for axis, value in zip("XYZ", (0.85, 0.65 + shift, 4.2 + step)):
            row[f"한길숏건 Position {axis}"] = str(value)
        # 겹친 프레임은 |머리-왼손| = 0.000 m라 왼손만 마스킹된다.
        left = (
            (0.5, 0.8 + shift, 4.2 + step)
            if i == overlap_left_at
            else (0.67, 0.79 + shift, 4.2 + step)
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
        spreads=service.measured_score.uniform_spreads(0.06),
    )
    measured = body["measured"]
    assert isinstance(measured, dict)
    assert measured["gate"] == "pass"
    assert measured["gate_reason"] is None
    assert measured["normalization"] == "measured"


def test_response_reports_the_measured_yaw_it_actually_used() -> None:
    """동작 키의 ②④는 이제 실측 요 위에 선다. 응답이 그 값과 출처를 말해야 한다.

    `yaw_assumed` 이름은 유지한다 — 화면과 기존 소비자가 그 키를 읽고 있고, 이름을
    바꾸면 조용히 사라진다. 대신 값이 실제로 쓴 요가 되고 `yaw_source`가 그것이
    실측인지 대체값인지 가른다.
    """
    client = TestClient(create_app(source=FakeSource()))
    body = client.post("/api/compare", json={"query": QUERY, "reference": REFERENCE}).json()
    assert body["yaw_source"] == "measured"
    assert math.isclose(body["yaw_assumed"], EXPECTED_YAW, abs_tol=1e-9)

    mismatched = TestClient(create_app(source=FakeSource(mismatch=True)))
    unavailable = mismatched.post(
        "/api/compare", json={"query": QUERY, "reference": REFERENCE}
    ).json()
    assert unavailable["status"] == "unavailable"
    assert unavailable["yaw_source"] == "measured"
    assert math.isclose(unavailable["yaw_assumed"], EXPECTED_YAW, abs_tol=1e-9)


def test_measured_yaw_moves_the_body_relative_axes_of_the_key() -> None:
    """요가 0이면 ②④는 몸 기준 이름을 단 월드 방위각이다. 그것이 이 배선의 요점이다.

    같은 패스를 회전 컬럼 없이 읽으면(=대체값 0) 키가 달라져야 한다. 달라지지 않으면
    실측 요가 키에 도달하지 않은 것이고, 그때는 이 배선이 아무것도 하지 않는다.
    """
    measured = service.compare(FakeSource(), PassRef(**QUERY), PassRef(**REFERENCE))

    class Rotationless(FakeSource):
        def load(self, ref: PassRef) -> RawPass:
            raw = super().load(ref)
            return RawPass(
                rows=[
                    {k: v for k, v in row.items() if not k.startswith("카메라 Rotation")}
                    for row in raw.rows
                ]
            )

    assumed = service.compare(Rotationless(), PassRef(**QUERY), PassRef(**REFERENCE))
    assert assumed["yaw_source"] == "assumed"
    assert assumed["yaw_assumed"] == service.ASSUMED_YAW
    assert measured["query_key"] != assumed["query_key"]
    # 바뀌는 것은 몸 기준 축뿐이다. 기울기 대역(①)은 착탄 변위에서만 나온다.
    assert str(measured["query_key"]).split(":")[1] == str(assumed["query_key"]).split(":")[1]


def test_improvement_counts_only_the_correction_that_was_issued() -> None:
    """화면이 말한 교정과 개선율이 잰 교정이 같아야 한다.

    개선율은 goal_year2의 납품 지표이고 그 정의는 "교정 알고리즘의 출력을 적용한 뒤의
    유사도 상승"이다. 교정문은 fa9c4b0 이후 오른손 하나의 최대 이탈 성분 하나뿐인데
    시뮬레이션이 임계를 넘은 세 점을 전부 옮기면, 작업자에게 오른손만 말하고 머리·
    왼손까지 고쳤을 때의 향상을 보고하게 된다.
    """
    measured_score = service.measured_score
    body = service.compare(FakeSource(), PassRef(**QUERY), PassRef(**REFERENCE))
    measured = body["measured"]
    assert isinstance(measured, dict)

    corrections = measured["corrections"]
    assert isinstance(corrections, list)
    assert len(corrections) == 1
    assert corrections[0].startswith("오른손")

    deltas = measured["deltas"]
    assert isinstance(deltas, list)
    sigmas = {item["point"]: item["sigma"] for item in deltas}
    # 머리도 임계를 넘는다. 그래서 옛 시뮬레이션이라면 머리까지 옮겼을 상황이다.
    assert sigmas["head"] > measured_score.CORRECTION_SIGMA_THRESHOLD

    source = FakeSource()
    query_frames = service.pass_frames(source.load(PassRef(**QUERY)))
    reference_frames = service.pass_frames(source.load(PassRef(**REFERENCE)))
    query_track = service._local_tracks(query_frames)
    reference_track = service._local_tracks(reference_frames)
    before = measured_score.score_track(
        query_track,
        reference_track,
        service.PROVISIONAL_SPREADS,
        measured_score.DEFAULT_WEIGHTS,
    )
    moves = {
        delta.point: delta.local_delta
        for delta in before.deltas
        if delta.sigma > measured_score.CORRECTION_SIGMA_THRESHOLD
    }
    assert set(moves) > {"right"}, "머리·왼손도 임계를 넘어야 이 회귀가 뜻을 갖는다"
    all_points = {
        name: [
            None
            if point is None
            else tuple(point[axis] + moves[name][axis] for axis in range(3))
            for point in points
        ]
        if name in moves
        else list(points)
        for name, points in query_track.items()
    }
    wide = measured_score.score_track(
        all_points,
        reference_track,
        service.PROVISIONAL_SPREADS,
        measured_score.DEFAULT_WEIGHTS,
    )
    assert measured["score_after"] < wide.score
    assert measured["score_after"] > measured["score"]


# --- compare()가 score_track에 대표 기저를 넘기는지 --------------------------------
#
# `_local_tracks`가 만드는 local_delta는 스트로크 프레임 벡터이지 월드 벡터가 아니다.
# compare()가 score_track에 기저(basis)를 안 넘기면 σ가 WORLD_FRAME으로 잘못 나뉘고,
# 그 σ로 `right_hand_finding`이 축을 고르면 축 이름은 같아도(wall/vertical/lateral)
# 물리적으로 다른 방향을 가리키는 값을 근거로 삼게 된다. 아래 두 테스트는 착탄점
# 궤적을 대각으로 만들어(대표 기저의 tangent·up이 월드 Y·Z에서 45° 돈다) 그 어긋남을
# 실제로 관측 가능하게 만든다.


def _rotated_rows(shift_z: float = 0.0, shift_y: float = 0.0) -> list[dict[str, str]]:
    """`_rows`와 같은 대각 착탄점 궤적(Ray Y·Z가 함께 늘어난다)을 쓴다.

    그래서 대표 기저의 normal은 우연히 월드 X와 겹치지만 tangent·up은 월드 Y·Z에서
    45° 돌아간다 — WORLD_FRAME과 실제 기저가 벌어지는 조건이다. 오른손(한길숏건)
    위치만 `shift_z`/`shift_y`로 독립적으로 어긋난다. 착탄점 궤적 자체는 이 두 값의
    영향을 받지 않으므로 대표 기저는 두 패스에서 동일하다.
    """
    rows: list[dict[str, str]] = []
    for i in range(30):
        step = i * 0.03
        row: dict[str, str] = {
            "인덱스": str(i),
            "시간(초)": f"{i * 0.0111:.4f}",
            "분사": "On",
            "Ray point X": "1.26",
            " Ray Point Y": str(4.2 + step),
            " Ray Point Z": str(0.7 + step),
            "Ray Texture UV X": "1024",
            "Ray Texture UV Y": "512",
        }
        for axis, value in zip("XYZ", (0.5, 4.2 + step, 0.8)):
            row[f"카메라 Position {axis}"] = str(value)
        for axis, value in zip("XYZ", HEAD_ROTATION):
            row[f"카메라 Rotation {axis}"] = str(value)
        for axis, value in zip("XYZ", (0.85, 4.2 + step + shift_y, 0.65 + shift_z)):
            row[f"한길숏건 Position {axis}"] = str(value)
        for axis, value in zip("XYZ", (0.67, 4.2 + step, 0.79)):
            row[f"플래쉬 Position {axis}"] = str(value)
        for axis, value in zip("XYZ", PARKED):
            row[f"한길폴건 Position {axis}"] = str(value)
            row[f"반도폴건 Position {axis}"] = str(value)
        for axis, value in zip("XYZ", STILL):
            row[f"반도숏건 Position {axis}"] = str(value)
        rows.append(row)
    return rows


class _RotatedBasisSource:
    """착탄점이 대각으로 흘러 대표 기저가 월드 축에서 회전한 픽스처."""

    def workers(self) -> list[str]:
        return ["A", "B"]

    def load(self, ref: PassRef) -> RawPass:
        if ref.worker == "B":
            return RawPass(rows=_rotated_rows())
        return RawPass(rows=_rotated_rows(shift_z=-0.30, shift_y=0.30))


# wall은 넓게(이 시나리오에선 편차가 0이라 무관), vertical·lateral은 서로 다르게
# 좁혀서 WORLD_FRAME과 실제 기저의 σ가 서로 다른 축·다른 크기를 내도록 만든다
# (vertical/lateral 산포가 같으면 45° 회전 앞뒤로 벡터 노름이 보존돼 결합 σ가
# 우연히 같아진다 — 두 산포를 다르게 잡아야 그 우연을 깬다).
_ROTATED_SPREADS = {
    "right": {"wall": 0.5, "vertical": 0.15, "lateral": 0.06},
    "head": {"wall": 0.5, "vertical": 0.5, "lateral": 0.5},
    "left": {"wall": 0.5, "vertical": 0.5, "lateral": 0.5},
}


def test_sigma_is_decomposed_in_the_representative_basis_not_world_frame() -> None:
    """compare()가 보고하는 σ는 WORLD_FRAME이 아니라 실제 대표 기저로 나뉜 값이어야
    한다.

    산포가 축마다 다르고(vertical 15cm, lateral 6cm, wall 50cm) 대표 기저가 월드
    축에서 45° 돌아가 있으면, WORLD_FRAME으로 읽은 σ와 실제 기저로 읽은 σ가
    벌어진다. compare()가 score_track에 기저를 안 넘기면(수정 전 결함) WORLD_FRAME
    값이 새어 나온다.
    """
    source = _RotatedBasisSource()
    query = PassRef(**QUERY)
    reference = PassRef(**REFERENCE)

    query_frames = service.pass_frames(source.load(query))
    reference_frames = service.pass_frames(source.load(reference))
    query_track = service._local_tracks(query_frames)
    reference_track = service._local_tracks(reference_frames)
    basis = service.representative_basis(query_frames)

    # 이 기저가 정말 월드 축에서 벗어나 있는지 먼저 확인한다 — 안 그러면 아래
    # assertion이 우연히 통과할 수 있어 아무것도 검증하지 못한다.
    assert not math.isclose(basis.tangent[2], 0.0, abs_tol=1e-6), "기저가 회전하지 않았다"

    measured_score = service.measured_score
    correct = measured_score.score_track(
        query_track, reference_track, _ROTATED_SPREADS, measured_score.DEFAULT_WEIGHTS, basis
    )
    world_default = measured_score.score_track(
        query_track, reference_track, _ROTATED_SPREADS, measured_score.DEFAULT_WEIGHTS
    )
    correct_right = next(d for d in correct.deltas if d.point == "right")
    world_right = next(d for d in world_default.deltas if d.point == "right")
    # 이 시나리오가 실제로 두 σ를 갈라놓는지 먼저 확인한다(가드) — 그렇지 않으면 아래
    # assertion이 우연히 통과할 수 있다.
    assert not math.isclose(correct_right.sigma, world_right.sigma, abs_tol=1e-6)

    body = service.compare(source, query, reference, spreads=_ROTATED_SPREADS)
    assert body["status"] == "ok"
    actual_right = next(d for d in body["measured"]["deltas"] if d["point"] == "right")
    # 수정 전에는 이 값이 world_right.sigma와 같았다 — 대표 기저가 아니라 WORLD_FRAME
    # 으로 나뉜 σ가 새어 나왔기 때문이다.
    assert math.isclose(actual_right["sigma"], correct_right.sigma, abs_tol=1e-6)


def test_finding_axis_matches_the_basis_score_track_used() -> None:
    """right_hand_finding이 고르는 축은 score_track이 실제로 쓴 기저의 σ를 따라야
    한다.

    교정 후보 목록(`_correction_candidates`)은 항상 올바른 기저로 만들어지므로 축
    이름·부호·방향은 늘 맞다. 하지만 그 후보들의 순위를 매기는 σ가 WORLD_FRAME에서
    새어 나오면(수정 전 결함) 물리적으로 틀린 축이 뽑힌다 — 이 시나리오에서는 아예
    반대 축이 뽑힌다.
    """
    source = _RotatedBasisSource()
    query = PassRef(**QUERY)
    reference = PassRef(**REFERENCE)

    query_frames = service.pass_frames(source.load(query))
    reference_frames = service.pass_frames(source.load(reference))
    basis = service.representative_basis(query_frames)

    body = service.compare(source, query, reference, spreads=_ROTATED_SPREADS)
    assert body["status"] == "ok"
    finding = body["measured"]["finding"]
    assert finding is not None

    query_track = service._local_tracks(query_frames)
    reference_track = service._local_tracks(reference_frames)
    measured_score = service.measured_score
    correct = measured_score.score_track(
        query_track, reference_track, _ROTATED_SPREADS, measured_score.DEFAULT_WEIGHTS, basis
    )
    expected = measured_score.right_hand_finding(correct.deltas, basis)
    assert expected is not None

    # WORLD_FRAME으로 σ를 잘못 읽으면 이 시나리오는 반대 축을 고른다(가드) — 그렇지
    # 않으면 아래 assertion이 아무것도 검증하지 못한다.
    world_default = measured_score.score_track(
        query_track, reference_track, _ROTATED_SPREADS, measured_score.DEFAULT_WEIGHTS
    )
    wrong = measured_score.right_hand_finding(world_default.deltas, basis)
    assert wrong is not None and wrong.axis != expected.axis

    assert finding["axis"] == expected.axis
    assert math.isclose(finding["magnitude_cm"], expected.magnitude_cm, abs_tol=1e-6)


# --- /api/diagnose — 배포 경로를 그대로 돌려 보는 길 --------------------------


class FkSource(FakeSource):
    """FK가 되는 가짜 소스.

    `FakeSource`의 행에는 관절 `Position`/`Rotation` 컬럼이 없어 `bind_offsets`가
    빈 dict를 낸다 — 그 소스로는 `pass_inputs`가 늘 `None`이라 진단이 `ok`에
    **한 번도 닿지 않는다.** 실측 바인드 오프셋을 얹어 닿게 한다.
    """

    def load(self, ref: PassRef) -> RawPass:
        from trainer.scripts.tests.test_pose_fk_csv import _row as _fk_row

        raw = super().load(ref)
        return RawPass(rows=[{**_fk_row(), **row} for row in raw.rows])


def _arm_reference_file(tmp_path, monkeypatch, *, band: str = "+0.20",
                        phases: int = 4, n: int = 300) -> None:
    """랩 검사용 팔 레퍼런스. 실물 `/output` 아티팩트에 기대지 않는다 —
    `/output`은 마운트된 산출물이지 저장소 데이터가 아니다(AGENTS.md).

    `band`는 `FkSource`가 내는 reach(0.20 m)가 떨어지는 밴드다. 다른 값을 쓰면
    `resolve`가 `reference_data_missing`을 내고 검사가 조용히 공허해진다.
    """
    import json

    from trainer.scripts import arm_reference as ar

    joints = {
        name: {
            "mean": [0.1, -0.2, 0.05], "sd": [0.05, 0.05, 0.05], "n": n,
            "radius": {"mean": 0.2291, "sd": 0.05},
            "heading": {"mean": [0.436, -0.873, 0.218], "angular_sd": 0.30},
        }
        for name in ar.SCORING_JOINTS_ARM
    }
    path = tmp_path / "arm_reference.json"
    path.write_text(json.dumps({
        "version": 1,
        "scoring_set_id": ar.SCORING_SET_ID,
        "reach_band_m": ar.REACH_BAND_M,
        "buckets": {
            band: {
                "n": n, "workers": 6, "joints": joints,
                "prototype": [[0.0, 0.0, 0.4 * i] for i in range(phases)],
                "prototype_support": {"to_prototype": 0.05, "between": 0.40, "n": n},
                "phases": [{"n": 60, "joints": joints} for _ in range(phases)],
            }
        },
    }), encoding="utf-8")
    monkeypatch.setenv("ARM_REFERENCE_PATH", str(path))


def test_diagnose_says_the_reference_is_missing_rather_than_guessing(monkeypatch, tmp_path) -> None:
    """아티팩트가 없으면 **지어내지 않는다.**

    깨뜨리는 법: `_arm_buckets`가 빈 dict를 내게 하면 `resolve`가 다른 사유를 내
    여기서 실패한다.
    """
    monkeypatch.setenv("ARM_REFERENCE_PATH", str(tmp_path / "없는파일.json"))
    client = TestClient(create_app(source=FkSource()))
    body = client.get("/api/diagnose", params=QUERY).json()
    assert body["status"] == "unavailable"
    assert body["reason"] == "reference_data_missing"
    # **아티팩트가 아예 없는 것**과 **있는데 밴드가 없는 것**은 다른 사건이다.
    # 앞은 패스를 읽을 이유조차 없으므로 `band`가 없어야 한다. 뒤는 밴드를 싣는다
    # (`test_diagnose_reasons_stay_in_a_closed_set`이 그쪽을 본다).
    assert "band" not in body, "레퍼런스가 없는데 패스를 읽어 밴드를 냈다"
    assert "reach_m" not in body


def test_diagnose_does_not_take_a_reference_pass(monkeypatch, tmp_path) -> None:
    """**`/api/compare`와 논리가 다르다.** 저쪽은 동작 키로 고른 **기준 패스 하나**에
    대고 재고, 이쪽은 reach 밴드로 고른 숙련자 **버킷**에 DTW로 맞춰 잰다. 제품에는
    기준 패스 입력이 없으므로 랩도 받지 않는다.

    깨뜨리는 법: 라우트에 기준 패스 인자를 더하면 이 검사가 실패한다.
    """
    _arm_reference_file(tmp_path, monkeypatch)
    client = TestClient(create_app(source=FkSource()))
    body = client.get("/api/diagnose", params=QUERY).json()
    assert body["status"] == "ok", body
    assert "reference" not in body, "기준 패스를 응답에 실으면 논리를 섞은 것이다"
    assert "motion_key" not in body, "동작 키는 이 논리의 것이 아니다"


def test_diagnose_reports_the_same_field_names_as_the_deployed_route(
    monkeypatch, tmp_path
) -> None:
    """랩에서 본 값과 제품이 낸 값을 **눈으로 대조**하려면 이름이 같아야 한다.

    깨뜨리는 법: `similarity_score`를 `score`로 줄이면 대조가 끊겨 실패한다.
    """
    _arm_reference_file(tmp_path, monkeypatch)
    client = TestClient(create_app(source=FkSource()))
    body = client.get("/api/diagnose", params=QUERY).json()
    assert body["status"] == "ok", body
    for key in ("scoring_set_id", "scoring_mode", "band", "reach_m", "support",
                "reference_quality", "similarity_score", "expert_percentile",
                "burst", "joint_sigmas", "corrections"):
        assert key in body, f"배포 응답에 있는 {key}가 랩 응답에 없다"
    assert body["scoring_set_id"] == service.arm_reference.SCORING_SET_ID
    assert body["band"] == "+0.20", "이 소스의 reach가 떨어지는 밴드가 아니다"
    assert set(body["joint_sigmas"]) == set(service.arm_reference.SCORING_JOINTS_ARM)
    assert len(body["prototype"]) == 4
    assert len(body["impact_path"]) == 30


def test_diagnose_reasons_stay_in_a_closed_set(monkeypatch, tmp_path) -> None:
    """지어낸 문자열이 나가면 소비자가 모르는 값을 받는다.

    깨뜨리는 법: `diagnose_sequence`가 집합 밖 문자열을 내면 실패한다.
    """
    _arm_reference_file(tmp_path, monkeypatch, band="+9.00")  # reach가 안 떨어지는 밴드
    client = TestClient(create_app(source=FkSource()))
    body = client.get("/api/diagnose", params=QUERY).json()
    assert body["status"] == "unavailable"
    assert body["reason"] in service.DIAGNOSE_REASONS, body["reason"]
    # 밴드를 못 찾아도 **왜인지 보여야** 한다.
    assert "band" in body


def test_diagnose_rereads_the_artifact_on_every_request(monkeypatch, tmp_path) -> None:
    """레퍼런스를 다시 구워도 랩을 재시작하지 않아야 한다 — 랩은 그걸 보는 도구다.

    깨뜨리는 법: `_arm_buckets`를 `create_app`에서 한 번만 읽게 하면 실패한다.
    """
    _arm_reference_file(tmp_path, monkeypatch)
    client = TestClient(create_app(source=FkSource()))
    assert client.get("/api/diagnose", params=QUERY).json()["status"] == "ok"

    monkeypatch.setenv("ARM_REFERENCE_PATH", str(tmp_path / "사라짐.json"))
    after = client.get("/api/diagnose", params=QUERY).json()
    assert after["reason"] == "reference_data_missing", (
        "아티팩트를 치웠는데 랩이 옛 값을 계속 낸다"
    )


def test_diagnose_says_why_the_percentile_is_missing(monkeypatch, tmp_path) -> None:
    """백분위가 없을 때 **사유가 둘**이다. 하나로 적으면 운영자를 엉뚱한 데로 보낸다 —
    MCP에서 실제로 그랬다(레퍼런스가 붙어 있는데 `reference_data_missing`이 나갔다).

    여기 레퍼런스에는 `expert_scores_phase`가 없다. 그러면 사다리가 없는 것이지
    합산 점수여서가 아니다.

    깨뜨리는 법: 두 사유를 한 문구로 합치면 실패한다.
    """
    _arm_reference_file(tmp_path, monkeypatch)
    client = TestClient(create_app(source=FkSource()))
    body = client.get("/api/diagnose", params=QUERY).json()
    assert body["status"] == "ok", body
    assert body["display"]["headline"] == "—", "사다리가 없는데 백분위가 나왔다"
    note = body["display"]["headline_note"]
    assert "사다리가 없어" in note, note
    assert "합쳤습니다" not in note, "합산이 아닌데 합산 때문이라고 말한다"


def test_diagnose_admits_when_a_bar_is_over_but_no_cue_fires(monkeypatch, tmp_path) -> None:
    """`build_corrections`는 **생 σ로 순위를 매긴 뒤** 1등에만 관절별 문턱을 적용한다.
    문턱이 관절마다 다르므로 1등이 아닌 관절이 자기 범위를 벗어나도 교정이 안 나간다.
    실측 222패스 중 범위를 벗어난 65건에서 **18건(28%)** 이 그 경우다.

    막대는 빨간데 교정이 없는 상태를 설명 없이 두면 화면이 자기모순한다.

    깨뜨리는 법: `_bars_note`가 늘 `None`을 내면 실패한다.
    """
    _arm_reference_file(tmp_path, monkeypatch)
    client = TestClient(create_app(source=FkSource()))
    body = client.get("/api/diagnose", params=QUERY).json()
    assert body["status"] == "ok", body

    over = [bar for bar in body["bars"] if bar["over"]]
    if body["correction_text"] is None and over:
        assert body["bars_note"] is not None, "막대는 넘었는데 화면이 아무 말도 안 한다"
        assert "교정" in body["bars_note"]
    else:
        # 이 소스에서 엇갈림이 안 나면 **함수 자체**가 엇갈림을 말하는지 직접 본다 —
        # 조건부 단언만 두면 검사가 조용히 공허해진다.
        from types import SimpleNamespace

        from trainer.scripts import arm_reference as ar

        hot = ar.JointSigma(joint="RightUpperArm", sigma=9.0,
                            axis_sigmas=(9.0, 0.0, 0.0), delta_m=(0.9, 0.0, 0.0))
        note = service._bars_note(
            SimpleNamespace(sigmas=[hot], corrections=[]), {"RightUpperArm": 2.0}
        )
        assert note is not None and "교정" in note, note
        assert service._bars_note(
            SimpleNamespace(sigmas=[hot], corrections=["있음"]), {"RightUpperArm": 2.0}
        ) is None, "교정이 나갔는데도 엇갈림이라고 말한다"
