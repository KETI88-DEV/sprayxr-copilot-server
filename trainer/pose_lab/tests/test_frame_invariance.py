#!/usr/bin/env python3
# pyright: reportAny=false, reportUnusedCallResult=false
"""좌표 프레임 경계의 불변성을 고정한다(task-0-brief.md, 조정자의 중간 수정 반영).

지금 캡처 CSV는 로컬 좌표이고, `capture_frame.to_world`는 레벨 1의 검증된 메시
정렬로 캡처 좌표를 실제 월드(미터) 좌표로 옮긴다(0절). 채점 파이프라인이 회전·
평행이동·스케일에 어떻게 반응하는지는 별개로 세 갈래로 못박는다(1~3절, `to_world`
와 무관하게 캡처 프레임 자체의 성질이다).

0. **`capture_frame.to_world` 자체**: 옮긴 착탄점이 레벨 메시 표면 위에 있는가,
   검증되지 않은 레벨·정렬을 조용히 통과시키지 않는가.
1. **불변이어야 하는 것**: 한 패스의 실측 점(head/right/left/hit) 전부에 수직축
   둘레 회전 θ + 수평 평행이동을 걸어도 `service.compare()`가 내는 유사도 점수,
   발부된 교정 축의 크기(cm), 개선율은 전부 같아야 한다. `motion_key.heading_word`
   (요 기준, 회전 불변), `motion_key.body_placement`의 lateral/reach(몸 기준 델타,
   평행이동 불변), `stroke_frame.stroke_basis`(수직축 둘레 회전·평행이동 불변)가
   이미 그렇게 설계돼 있다는 것을 컨트롤러가 훑었다 — 여기서는 그 설계를 재조사하지
   않고 `compare()` 출력 수준에서 못박기만 한다.
2. **바뀌어야 하는 것**: 같은 패스에 수직 평행이동을 걸면 `motion_key`가 달라져야
   한다. `motion_key.body_placement`의 마지막 반환값(`pass_segmentation.height_band`)
   이 절대 미터 임계값을 쓰기 때문이다 — 수평은 몸 기준인데 수직만 월드 기준인
   섞인 좌표다. "안 바뀐다"고 단언하면 거짓이므로 여기서 그 반대를 확인한다.
3. **스케일**: 전체 좌표에 1.05배를 걸면 교정 크기(cm)가 그 배율을 따라 바뀐다.

`_rotate_translate_scale`이 유일한 기하 변환이고, 나머지는 CSV 행을 만드는
배관이다. 회전은 수평(X,Z) 성분에만 걸고 수직(Y) 성분은 그대로 둔다 — 캡처 CSV가
유니티 Y-up이라는 사실은 `world_axes.py`가 유일한 출처이고 여기서는 그 규약을
다시 정의하지 않는다(축 인덱스 리터럴을 새로 쓰지 않는다는 뜻은 아니다: 이 파일은
`world_axes`가 아니라 원시 CSV 컬럼을 만드는 테스트 픽스처이므로 컬럼명이 곧 축
이름이다).
"""

from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

capture_frame = importlib.import_module("capture_frame")
level_mesh = importlib.import_module("level_mesh")
motion_key = importlib.import_module("motion_key")
pass_segmentation = importlib.import_module("pass_segmentation")

from trainer.pose_lab import service
from trainer.pose_lab.service import PassRef, RawPass

Vec3 = tuple[float, float, float]

PARKED: Vec3 = (-694.61, 701.29, 698.25)
STILL: Vec3 = (-2.99, 4.66, -0.27)
# HEAD_ROTATION Y=30 -> 정면 방위 60도(측정 규약은 `measured_pose.py`가 데이터로
# 고른 것 — 이 파일에서 다시 정하지 않는다. `test_app.py`의 같은 상수·주석 참고).
BASE_YAW = math.radians(60.0)
N_FRAMES = 24
# 착탄점 수직 범위(0.72~0.92m)를 HEIGHT_LOW_MAX_M(0.6)/HEIGHT_MID_MAX_M(1.0) 사이
# 한가운데("mid")에 잡는다 — 회전·평행이동·±5% 스케일 테스트에서 동작 키가
# height_band 경계를 우연히 넘지 않게 하려는 것이다(그 경계를 넘기는 것은 오직
# "바뀌어야 하는 것" 테스트의 몫이다).
_RISE_SPAN = 0.2
_BASE_HIT_Y = 0.72
# 왕복이 아니라 단조 진행이므로 경사가 완만해(atan(0.2/0.6)=18.4' < 20') "horizontal"
# 대역에 든다 — heading_word가 정의되는 대역이면 어느 쪽이든 상관없다.
_STEP_SPAN = 0.6

QUERY = PassRef(worker="Q", date="20250901", level="10", session="s1", pass_index=0)
REFERENCE = PassRef(worker="R", date="20250902", level="10", session="s2", pass_index=0)
# 오른손/머리를 CORRECTION_SIGMA_THRESHOLD(2.0)를 넘게 벌리는 수직 오프셋. 값 자체는
# 임의이고, 요점은 "발부된 교정 축의 크기"를 실제로 낼 만큼 크다는 것뿐이다.
QUERY_OFFSET_M = 0.3
FIXED_SPREADS = service.measured_score.uniform_spreads(0.03)


def _rotate_translate_scale(
    point: Vec3, theta: float, translation: tuple[float, float], vertical_shift: float, scale: float
) -> Vec3:
    """수직축(Y) 둘레 회전 θ + 수평(X,Z) 평행이동 + 수직 평행이동 + 등방 스케일.

    회전은 원점 기준이다 — 평행이동을 별도 인자로 두므로 회전 중심을 어디로 잡든
    일반 강체 운동(회전 후 이동)을 다 표현한다. Y(수직)는 회전에 관여하지 않는다.
    """
    x, y, z = point
    x, y, z = x * scale, y * scale, z * scale
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    x_rot = x * cos_t - z * sin_t
    z_rot = x * sin_t + z * cos_t
    return (x_rot + translation[0], y * 1.0 + vertical_shift, z_rot + translation[1])


def _rows(
    role: str,
    theta: float = 0.0,
    translation: tuple[float, float] = (0.0, 0.0),
    vertical_shift: float = 0.0,
    scale: float = 1.0,
) -> list[dict[str, str]]:
    """결정론적 대각 패스 하나. `role`이 "query"면 오른손/머리/왼손을 수직으로
    `QUERY_OFFSET_M`만큼 더 벌려 교정이 실제로 발동하게 한다(착탄점은 벌리지
    않는다 — `test_app.py`의 `_rows(shift=...)`와 같은 구조).
    """
    offset = QUERY_OFFSET_M if role == "query" else 0.0
    new_yaw = BASE_YAW + theta
    roty = 90.0 - math.degrees(new_yaw)

    rows: list[dict[str, str]] = []
    for i in range(N_FRAMES):
        t = i / (N_FRAMES - 1)
        step = t * _STEP_SPAN
        rise = t * _RISE_SPAN

        hit = _rotate_translate_scale(
            (1.2, _BASE_HIT_Y + rise, 4.2 + step), theta, translation, vertical_shift, scale
        )
        head = _rotate_translate_scale(
            (0.5, 0.8 + offset, 4.2 + step), theta, translation, vertical_shift, scale
        )
        right = _rotate_translate_scale(
            (0.85, 0.65 + offset, 4.2 + step), theta, translation, vertical_shift, scale
        )
        left = _rotate_translate_scale(
            (0.67, 0.79 + offset, 4.2 + step), theta, translation, vertical_shift, scale
        )

        row: dict[str, str] = {
            "인덱스": str(i),
            "시간(초)": f"{i * 0.0111:.4f}",
            "분사": "On",
            "Ray point X": repr(hit[0]),
            " Ray Point Y": repr(hit[1]),
            " Ray Point Z": repr(hit[2]),
            "Ray Texture UV X": "1024",
            "Ray Texture UV Y": "512",
        }
        for axis, value in zip("XYZ", head):
            row[f"카메라 Position {axis}"] = repr(value)
        for axis, value in zip("XYZ", (0.0, roty, 0.0)):
            row[f"카메라 Rotation {axis}"] = repr(value)
        for axis, value in zip("XYZ", right):
            row[f"한길숏건 Position {axis}"] = repr(value)
        for axis, value in zip("XYZ", left):
            row[f"플래쉬 Position {axis}"] = repr(value)
        for axis, value in zip("XYZ", PARKED):
            row[f"한길폴건 Position {axis}"] = repr(value)
            row[f"반도폴건 Position {axis}"] = repr(value)
        for axis, value in zip("XYZ", STILL):
            row[f"반도숏건 Position {axis}"] = repr(value)
        rows.append(row)
    return rows


class _FakeSource:
    """질의(Q)·레퍼런스(R) 둘 다에 같은 기하 변환을 건다 — "월드 좌표계를 바꿔
    본다"는 것이 곧 두 패스 모두를 같은 회전·평행이동·스케일로 옮기는 것이다."""

    def __init__(
        self,
        theta: float = 0.0,
        translation: tuple[float, float] = (0.0, 0.0),
        vertical_shift: float = 0.0,
        scale: float = 1.0,
    ) -> None:
        self.theta = theta
        self.translation = translation
        self.vertical_shift = vertical_shift
        self.scale = scale

    def workers(self) -> list[str]:
        return ["Q", "R"]

    def load(self, ref: PassRef) -> RawPass:
        role = "query" if ref.worker == "Q" else "reference"
        return RawPass(
            rows=_rows(role, self.theta, self.translation, self.vertical_shift, self.scale)
        )


def _compare(
    theta: float = 0.0,
    translation: tuple[float, float] = (0.0, 0.0),
    vertical_shift: float = 0.0,
    scale: float = 1.0,
) -> dict[str, object]:
    source = _FakeSource(theta, translation, vertical_shift, scale)
    return service.compare(source, QUERY, REFERENCE, spreads=FIXED_SPREADS)


def _axes_signature(body: dict[str, object]) -> list[tuple[str, str, float]]:
    measured = body["measured"]
    assert isinstance(measured, dict)
    return [(a["point"], a["axis"], round(float(a["magnitude_cm"]), 6)) for a in measured["axes"]]


# --- 0. capture_frame 자체 -----------------------------------------------------
#
# `to_world`는 더 이상 항등이 아니다 — 레벨 1의 검증된("solved") 메시 정렬로 캡처
# 좌표를 월드(미터) 좌표로 실제로 옮긴다. 세 가지를 고정한다: (1) 옮긴 착탄점이
# 레벨 메시 표면 위에 있는가(외부 기하 증거), (2) 검증된 정렬이 없는 레벨을 조용히
# 통과시키지 않는가, (3) "declared"(검증되지 않은) 정렬도 조용히 쓰지 않는가.

# 레벨 1은 더 이상 "solved"가 아니다 — 착탄점이 정확히 평면이라 12자유도가 데이터로
# 결정되지 않는다(`level_mesh.GATES["cond_min_over_max"]`, 실측 2.06e-09). 이 파일은
# "검증된 정렬이 있는 레벨 아무거나"를 써야 하지 특정 숫자에 묶이면 안 된다.
def _first_solved_level() -> int:
    import json

    for entry in json.loads(level_mesh.ALIGNMENT_PATH.read_text(encoding="utf-8")):
        if entry["provenance"] == "solved":
            return int(entry["level"])
    raise AssertionError("검증된 정렬이 하나도 없다")


SOLVED_LEVEL = _first_solved_level()
UNVERIFIABLE_LEVEL = 1
# `test_level_mesh.py::test_committed_translation_places_real_hits_on_the_mesh`와
# 같은 허용치(1cm) — 실측 median은 그보다 여섯 자릿수 이상 작다(~2e-10 m).
_MESH_SURFACE_TOLERANCE_M = 0.01


def _mesh_in_metres(alignment) -> "level_mesh.Mesh":  # noqa: F821 — 타입은 아래서 임포트한다.
    """레벨 메시 정점을 `alignment.scale`로만 미터로 바꾼다(회전·평행이동 없음) —
    `capture_frame.to_world`가 내는 좌표계와 정확히 짝을 이루는 메시다
    (`capture_frame.py` 모듈 독스트링의 유도 참고)."""
    import json

    source = next(
        entry["source_fbx"]
        for entry in json.loads(level_mesh.ALIGNMENT_PATH.read_text(encoding="utf-8"))
        if int(entry["level"]) == SOLVED_LEVEL
    )
    mesh = level_mesh.read_fbx(level_mesh.DEFAULT_MODELINGS_ROOT / source)
    scale = alignment.scale
    return level_mesh.Mesh(
        vertices=[(v[0] * scale, v[1] * scale, v[2] * scale) for v in mesh.vertices],
        triangles=mesh.triangles,
        uvs=mesh.uvs,
        declared=mesh.declared,
        uv_indices=mesh.uv_indices,
    )


def test_to_world_lands_a_converted_hit_on_the_level_mesh_surface() -> None:
    """캡처 착탄점을 `to_world`로 옮기면, 그 점은 레벨 메시 표면(미터로 바꾼 것)
    위에 있어야 한다 — 이 저장소에서 월드 좌표 해석이 맞다는 것을 확인하는 유일한
    외부 증거(`place()`가 반대 방향에서 이미 확인한 것)를 여기서도 확인한다."""
    alignment = level_mesh.load_alignment(SOLVED_LEVEL)
    mesh_m = _mesh_in_metres(alignment)
    hits, _uvs = level_mesh._collect_level_hits(SOLVED_LEVEL, min_hits=500)  # noqa: SLF001
    assert len(hits) >= 500, "표본이 모자라면 이 검사는 아무것도 보증하지 않는다"
    sample = hits[:500]
    distances = sorted(
        level_mesh.closest_distance(mesh_m, capture_frame.to_world(hit, level=SOLVED_LEVEL))
        for hit in sample
    )
    median = distances[len(distances) // 2]
    assert median < _MESH_SURFACE_TOLERANCE_M


def test_to_world_refuses_a_level_with_no_verified_alignment() -> None:
    """`level_alignment.json`에 항목이 없는 레벨을 물으면 캡처 좌표를 조용히
    돌려주지 않고 멈춰야 한다."""
    with pytest.raises(LookupError):
        capture_frame.to_world((1.2, 0.8, 4.2), level=99)


def test_to_world_refuses_an_unverifiable_alignment() -> None:
    """레벨 1은 항목은 있으나 `provenance`가 "unverifiable"이다 — 화면이 "없다"와
    "검증할 수 없다"를 다르게 말할 수 있어야 하고, `to_world`는 둘 다 거절해야 한다."""
    assert level_mesh.load_alignment(UNVERIFIABLE_LEVEL).provenance == "unverifiable"
    with pytest.raises(ValueError):
        capture_frame.to_world((1.2, 0.8, 4.2), level=UNVERIFIABLE_LEVEL)


def test_to_world_refuses_a_declared_but_unverified_alignment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`provenance`가 "solved"가 아니면(예: "declared") 검증되지 않은 값이다 —
    레벨 1의 declared 변환은 실측과 3.68m 어긋났다(`level_mesh.py` 모듈 독스트링).
    `to_world`가 그런 항목을 조용히 쓰면 안 된다."""
    import dataclasses

    real = level_mesh.load_alignment(SOLVED_LEVEL)
    unverified = dataclasses.replace(real, provenance="declared")
    monkeypatch.setattr(level_mesh, "load_alignment", lambda level: unverified)
    with pytest.raises(ValueError):
        capture_frame.to_world((1.2, 0.8, 4.2), level=SOLVED_LEVEL)


# --- 이 파일의 변환 헬퍼가 실제로 좌표를 움직이는지(검증 항목 2에 대한 상시 가드) ---


def test_transform_helper_actually_moves_points() -> None:
    """아래 회전·평행이동 불변성 테스트가 "θ를 걸어도 그대로"라고 주장하려면, θ가
    실제로 좌표를 움직인다는 것부터 참이어야 한다. 이 assert가 없으면 나중에 누가
    `_rotate_translate_scale` 호출을 실수로 지워도(θ가 무시돼도) 불변성 테스트가
    계속 통과해 아무것도 확인하지 않는 테스트가 된다 — 검증 항목 2가 지적하는
    무력화 경로를 이 테스트가 상시로 막는다.
    """
    point: Vec3 = (1.2, 0.8, 4.2)
    rotated = _rotate_translate_scale(point, math.radians(37.0), (0.0, 0.0), 0.0, 1.0)
    translated = _rotate_translate_scale(point, 0.0, (2.3, -1.7), 0.0, 1.0)
    shifted = _rotate_translate_scale(point, 0.0, (0.0, 0.0), 0.5, 1.0)
    scaled = _rotate_translate_scale(point, 0.0, (0.0, 0.0), 0.0, 1.05)
    assert rotated != point
    assert translated != point
    assert shifted != point
    assert scaled != point
    # 항등(θ=0, 평행이동 없음, 수직 이동 없음, 배율 1)은 그대로여야 한다 — 그래야
    # 위 네 개의 "달라짐"이 변환이 걸렸기 때문이지 헬퍼 자체가 늘 좌표를 바꾸는
    # 결함 때문이 아니라는 것도 같이 보장된다.
    assert _rotate_translate_scale(point, 0.0, (0.0, 0.0), 0.0, 1.0) == point


# --- 1. 불변이어야 하는 것: 수직축 둘레 회전 + 수평 평행이동 ------------------------

_THETA_DEGREES = (0.0, 37.0, 180.0, -90.0)
_TRANSLATION = (2.3, -1.7)


@pytest.mark.parametrize("theta_deg", _THETA_DEGREES)
def test_rotation_and_horizontal_translation_are_score_invariant(theta_deg: float) -> None:
    """수직축 둘레 회전 θ + 수평 평행이동을 걸어도 유사도 점수가 같아야 한다."""
    baseline = _compare()
    theta = math.radians(theta_deg)
    transformed = _compare(theta=theta, translation=_TRANSLATION)

    assert baseline["status"] == "ok"
    assert transformed["status"] == "ok"
    base_measured = baseline["measured"]
    tr_measured = transformed["measured"]
    assert isinstance(base_measured, dict) and isinstance(tr_measured, dict)
    assert math.isclose(tr_measured["score"], base_measured["score"], rel_tol=1e-6, abs_tol=1e-9)


@pytest.mark.parametrize("theta_deg", _THETA_DEGREES)
def test_rotation_and_horizontal_translation_are_correction_invariant(theta_deg: float) -> None:
    """발부된 교정 축의 크기(cm)가 회전·평행이동에도 같아야 한다."""
    baseline = _compare()
    theta = math.radians(theta_deg)
    transformed = _compare(theta=theta, translation=_TRANSLATION)

    base_axes = _axes_signature(baseline)
    tr_axes = _axes_signature(transformed)
    assert base_axes, "적어도 하나의 교정이 실제로 발동해야 이 테스트가 뜻을 갖는다"
    assert len(base_axes) == len(tr_axes)
    for (b_point, b_axis, b_mag), (t_point, t_axis, t_mag) in zip(base_axes, tr_axes):
        assert b_point == t_point
        assert b_axis == t_axis
        assert math.isclose(b_mag, t_mag, rel_tol=1e-6, abs_tol=1e-6)


@pytest.mark.parametrize("theta_deg", _THETA_DEGREES)
def test_rotation_and_horizontal_translation_are_improvement_invariant(theta_deg: float) -> None:
    """개선율(same)도 회전·평행이동에 같아야 한다."""
    baseline = _compare()
    theta = math.radians(theta_deg)
    transformed = _compare(theta=theta, translation=_TRANSLATION)

    base_measured = baseline["measured"]
    tr_measured = transformed["measured"]
    assert isinstance(base_measured, dict) and isinstance(tr_measured, dict)
    base_same = base_measured["improvement"]["same"]
    tr_same = tr_measured["improvement"]["same"]
    assert isinstance(base_same, float) and isinstance(tr_same, float)
    assert math.isclose(tr_same, base_same, rel_tol=1e-6, abs_tol=1e-9)


@pytest.mark.parametrize("theta_deg", _THETA_DEGREES)
def test_rotation_and_horizontal_translation_keep_the_motion_key(theta_deg: float) -> None:
    """동작 키 자체도 회전·평행이동에 같아야 한다 — 다르면 그 아래 비교가 애초에
    "재지 못했다"로 갈릴 수 있으므로 이 사실이 먼저 참이어야 위 세 테스트가 뜻을
    갖는다."""
    baseline = _compare()
    theta = math.radians(theta_deg)
    transformed = _compare(theta=theta, translation=_TRANSLATION)
    assert transformed["query_key"] == baseline["query_key"]
    assert transformed["reference_key"] == baseline["reference_key"]


# --- 2. 바뀌어야 하는 것: 수직 평행이동 -----------------------------------------


def test_vertical_translation_changes_the_motion_key() -> None:
    """`motion_key.body_placement`의 height_band 성분이 절대 미터 임계값을 쓰므로,
    같은 패스를 수직으로만 옮기면(수평은 그대로) 동작 키가 달라져야 한다. "안
    바뀐다"고 단언하면 거짓이다 — 그 반대를 여기서 확인한다.
    """
    baseline = _compare()
    shifted = _compare(vertical_shift=0.5)
    assert baseline["status"] == "ok"
    assert shifted["status"] == "ok"
    assert shifted["query_key"] != baseline["query_key"]


def test_vertical_translation_changes_the_height_band_specifically() -> None:
    """동작 키가 달라지는 이유가 정확히 height_band인지 직접 확인한다 — 다른 성분
    (heading/uv/lateral/reach)은 그대로여야 한다. `world_axes.vertical_of`가
    호출 시점에 축을 읽으므로 여기서 인덱스를 새로 적지 않는다.
    """
    import world_axes  # noqa: E402 — scripts 경로는 위에서 이미 sys.path에 얹었다.

    before_hit: Vec3 = (1.2, _BASE_HIT_Y, 4.2)
    after_hit: Vec3 = (1.2, _BASE_HIT_Y + 0.5, 4.2)
    before_band = pass_segmentation.height_band(world_axes.vertical_of(before_hit))
    after_band = pass_segmentation.height_band(world_axes.vertical_of(after_hit))
    assert before_band != after_band

    head: Vec3 = (0.5, 0.8, 4.2)
    params = motion_key.KeyParams()
    before_key = motion_key.body_placement(head, BASE_YAW, before_hit, params)
    after_key = motion_key.body_placement(
        (head[0], head[1] + 0.5, head[2]), BASE_YAW, after_hit, params
    )
    assert before_key[2] != after_key[2]
    assert before_key[2] == before_band
    assert after_key[2] == after_band
    # lateral·reach(수평, 몸 기준)는 수직 평행이동에 그대로여야 한다 — 섞인 것은
    # height_band 하나뿐이라는 브리핑의 주장을 직접 확인한다.
    assert before_key[0] == after_key[0]
    assert before_key[1] == after_key[1]


# --- 3. 스케일: 전체 좌표에 배율을 걸면 교정 크기(cm)도 그만큼 바뀐다 ----------------


def test_scale_changes_correction_magnitude_proportionally() -> None:
    baseline = _compare()
    scaled = _compare(scale=1.05)
    assert baseline["status"] == "ok"
    assert scaled["status"] == "ok"
    # 스케일은 절대 위치도 바꾸므로(착탄점 높이 포함) 동작 키의 height_band 성분이
    # 우연히 넘어가지 않게 픽스처를 잡아 뒀다 — 그래서 두 키가 여전히 같아야
    # 교정 축 집합을 1:1로 비교할 수 있다.
    assert scaled["query_key"] == baseline["query_key"]

    base_axes = _axes_signature(baseline)
    scaled_axes = _axes_signature(scaled)
    assert base_axes, "적어도 하나의 교정이 실제로 발동해야 이 테스트가 뜻을 갖는다"
    assert len(base_axes) == len(scaled_axes)
    for (b_point, b_axis, b_mag), (s_point, s_axis, s_mag) in zip(base_axes, scaled_axes):
        assert b_point == s_point
        assert b_axis == s_axis
        assert b_mag != 0.0, "0을 기준으로 비율을 재면 어떤 배율도 통과한다"
        assert math.isclose(s_mag, b_mag * 1.05, rel_tol=1e-6, abs_tol=1e-6)
