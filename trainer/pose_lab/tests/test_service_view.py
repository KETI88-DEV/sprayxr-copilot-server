#!/usr/bin/env python3
# pyright: reportAny=false, reportUnusedCallResult=false
"""`compare()`의 `view` 블록(task-23-brief.md).

`corrected` 층과 `corrections`가 `measured_score.issued_axes`가 낸 것 하나에서만
파생되는지, 좌표가 (스트로크 로컬이 아니라) `pass_frames`의 월드 좌표인지, 격자점
개수가 채점이 실제로 쓰는 표본 수와 같은지, `weights`가 그리는 점을 다 덮는지,
메시가 검증된 정렬이 있는 레벨에서만 실리는지를 고정한다.

`FakeSource`/`QUERY`/`REFERENCE`는 `test_app.py`에서 그대로 가져다 쓴다(복제하지
않는다) — worker "A"(질의, shift=0.3)의 오른손·머리 sigma가
`CORRECTION_SIGMA_THRESHOLD`를 넘도록 벌어져 있어 실제로 교정이 발부된다.
"""

from __future__ import annotations

import importlib
import math
import random
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

measured_score = importlib.import_module("measured_score")
level_mesh = importlib.import_module("level_mesh")

simulator_csv = importlib.import_module("simulator_csv")
pose_fk_csv = importlib.import_module("pose_fk_csv")
rotation_math = importlib.import_module("rotation_math")
world_axes = importlib.import_module("world_axes")
measured_pose = importlib.import_module("measured_pose")

from trainer.pose_lab import service, skeleton
from trainer.pose_lab.skeleton import AnchorMount, Mounts
from trainer.pose_lab.service import PassRef, RawPass
from trainer.pose_lab.tests.test_app import FakeSource, QUERY, REFERENCE, _rows

# 뼈 길이(bind offset)는 프레임 불변이다 — `test_pose_fk_csv.OFFSETS`(실측치)와 같은
# 값을 여기 다시 적는다. 임포트로 공유하지 않는 이유는 그 파일이 `trainer/scripts/tests`
# 패키지에 있어 `trainer/pose_lab/tests`에서 상대 임포트로 닿지 않기 때문이다(두
# 트리가 각자 `sys.path` 삽입으로 평평하게 펴져 있다) — 값 자체의 출처는 그 파일이다.
_SKELETON_OFFSETS: dict[str, tuple[float, float, float]] = {
    "Pelvis": (0.0, 0.0, 0.0),
    "Spine": (-0.1053, -0.0001, 0.0),
    "Chest": (-0.1053, -0.0001, 0.0),
    "Head": (-0.1368, 0.0, 0.0),
    "LeftShoulder": (0.0, 0.0001, 0.0284),
    "LeftUpperArm": (-0.1797, 0.0, 0.0),
    "LeftForeArm": (-0.2430, 0.0, 0.0),
    "LeftHand": (-0.2430, 0.0, 0.0),
    "RightShoulder": (0.0, 0.0001, -0.0284),
    "RightUpperArm": (-0.1797, 0.0, 0.0),
    "RightForeArm": (-0.2430, 0.0, 0.0),
    "RightHand": (-0.2430, 0.0, 0.0),
    "LeftThigh": (0.0931, -0.0024, 0.0965),
    "LeftCalf": (-0.4223, 0.0, 0.0),
    "LeftFoot": (-0.4110, 0.0, 0.0),
    "LeftToe": (-0.0672, 0.1337, 0.0),
    "RightThigh": (0.0931, -0.0024, -0.0965),
    "RightCalf": (-0.4223, 0.0, 0.0),
    "RightFoot": (-0.4110, 0.0, 0.0),
    "RightToe": (-0.0672, 0.1337, 0.0),
}


def _rows_with_skeleton(shift: float) -> list[dict[str, str]]:
    """`test_app._rows`(실측 3점)에 FK 관절 컬럼을 얹는다 — 회전은 항등(0,0,0)이라
    `local`이 곧게 선 자세를 낸다."""
    rows = _rows(shift)
    for row in rows:
        for column in simulator_csv.JOINT_COLUMNS:
            key = simulator_csv.JOINT_KEYS[column]
            offset = _SKELETON_OFFSETS.get(key, (0.0, 0.0, 0.0))
            for axis, value in zip("XYZ", offset):
                row[f"{column} Position {axis}"] = str(value)
            for axis in "XYZ":
                row[f"{column} Rotation {axis}"] = "0.0"
        # 건·플래쉬 회전도 싣는다 — 배치(M6)는 실측 강체 **회전**으로 루트를 세운다.
        for group in ("한길숏건", "플래쉬"):
            for axis in "XYZ":
                row[f"{group} Rotation {axis}"] = "0.0"
    return rows


def _compare() -> dict[str, object]:
    result = service.compare(FakeSource(), PassRef(**QUERY), PassRef(**REFERENCE))
    assert result["status"] == "ok"
    return result


def test_correction_arrow_delta_matches_its_own_axis_exactly() -> None:
    """`to - from`이 그 축의 `magnitude_cm/100 * unit`과 정확히 같다(요구사항 2)."""
    view = _compare()["view"]
    corrections = view["corrections"]
    assert corrections, "이 픽스처는 교정이 발부돼야 한다(shift=0.3m)"
    for entry in corrections:
        metres = entry["magnitude_cm"] / 100.0
        expected = [entry["unit"][i] * metres for i in range(3)]
        actual = [entry["to"][i] - entry["from"][i] for i in range(3)]
        for i in range(3):
            assert actual[i] == expected[i], (entry, actual, expected)


def test_corrected_layer_shift_equals_the_independently_summed_issued_axes() -> None:
    """`corrected` 층의 이동이 `issued_axes` 합과 정확히 같다(요구사항 1, 독립 계산).

    `service._world_shifts`(구현이 실제로 쓰는 합산 경로)를 재사용하지 않고, 이
    테스트가 직접 `AxisCorrection` 목록을 훑어 점별 이동을 다시 더한다 — 구현이
    다른 값을 더하거나, 한 점의 축을 하나만 반영하거나, 부호를 잘못 더하면 이
    독립 합산과 어긋나 잡힌다.
    """
    query_frames = service.pass_frames(FakeSource().load(PassRef(**QUERY)))
    result = _compare()
    deltas = result["measured"]["deltas"]  # noqa: F841 (참고용, 아래는 basis가 필요해 재구성)
    view = result["view"]

    # basis는 응답에 실리지 않으므로(설계상 내부 개념) 같은 방식으로 다시 구한다 —
    # `compare()`가 쓰는 것과 동일한 `representative_basis(query_frames)`다.
    point_deltas = [
        measured_score.PointDelta(
            point=d["point"],
            sigma=d["sigma"],
            local_delta=tuple(d["delta"]),
            samples=d["samples"],
            axis_sigmas=dict(d["axis_sigmas"]),
        )
        for d in result["measured"]["deltas"]
    ]
    basis = service.representative_basis(query_frames)
    issued = measured_score.issued_axes(point_deltas, basis)
    assert issued, "이 픽스처는 교정이 발부돼야 한다"

    expected_shift: dict[str, list[float]] = {}
    for correction in issued:
        metres = correction.magnitude_cm / 100.0
        acc = expected_shift.setdefault(correction.point, [0.0, 0.0, 0.0])
        for i in range(3):
            acc[i] += correction.unit[i] * metres

    query_layer = view["layers"]["query"]
    corrected_layer = view["layers"]["corrected"]
    for point, shift in expected_shift.items():
        query_points = query_layer[point]
        corrected_points = corrected_layer[point]
        assert len(query_points) == len(corrected_points)
        for q, c in zip(query_points, corrected_points):
            for i in range(3):
                assert c[i] - q[i] == shift[i]

    # 발부되지 않은 점(왼손·착탄점)은 조금도 옮기지 않는다.
    for point in ("left", "hit"):
        assert query_layer[point] == corrected_layer[point]


def test_layer_coordinates_are_the_world_capture_frame_not_stroke_local() -> None:
    """`query`/`reference` 층의 좌표가 `pass_frames`의 `MeasuredFrame` 월드 좌표
    그대로다(요구사항 3) — `compare()`의 `query_track`/`_local_tracks`(스트로크 로컬)를
    쓰지 않는다. 값이 조금이라도 바뀌면(로컬로 원점 이동·기저 회전) 이 등식이 깨진다.
    """
    query_frames = service.pass_frames(FakeSource().load(PassRef(**QUERY)))
    reference_frames = service.pass_frames(FakeSource().load(PassRef(**REFERENCE)))
    view = _compare()["view"]

    query_layer = view["layers"]["query"]
    for i, frame in enumerate(query_frames):
        assert query_layer["head"][i] == list(frame.head)
        assert query_layer["right"][i] == list(frame.right)
        assert query_layer["hit"][i] == list(frame.hit)

    reference_layer = view["layers"]["reference"]
    for i, frame in enumerate(reference_frames):
        assert reference_layer["head"][i] == list(frame.head)
        assert reference_layer["right"][i] == list(frame.right)
        assert reference_layer["hit"][i] == list(frame.hit)


def test_corrected_layer_sums_every_axis_issued_for_the_same_point() -> None:
    """오른손처럼 한 점에 축이 둘(wall+vertical) 발부되면, `corrected` 층은 그 둘의
    합만큼 옮겨야 한다 — 하나만 반영하거나 마지막 것으로 덮어쓰면 안 된다.

    `_deltas()`/`test_axes_block_lists_every_issued_axis_with_its_own_sigma`가 이미
    다축 발부를 확인한 그 픽스처를 그대로 쓴다(local_delta=(0.08, 0.0, -0.06),
    wall σ=3.5·vertical σ=2.6 둘 다 임계 0.1을 넘는다) — `issued_axes`가 wall 8cm +
    vertical -6cm 두 축을 낸다.
    """
    frame_kwargs = dict(index=0, time_sec=0.0, left=None, uv=None, spraying=True)
    frames = [
        service.measured_pose.MeasuredFrame(
            head=(0.0, 1.6, 0.0), right=(0.3, 1.4, 0.0), hit=(0.0, 1.0, float(i)),
            **frame_kwargs,
        )
        for i in range(2)
    ]
    delta = measured_score.PointDelta(
        point="right",
        sigma=4.0,
        local_delta=(0.08, 0.0, -0.06),
        samples=2,
        axis_sigmas={"wall": 3.5, "vertical": 2.6, "lateral": 0.1},
    )
    view = service.view_block(
        query_frames=frames,
        reference_frames=frames,
        deltas=[delta],
        basis=measured_score.WORLD_FRAME,
        weights=measured_score.DEFAULT_WEIGHTS,
        level_text="10",
        grid_n=2,
    )
    issued = measured_score.issued_axes([delta], measured_score.WORLD_FRAME)
    assert {c.axis for c in issued} == {"wall", "vertical"}, "픽스처가 다축을 내야 한다"
    expected = [0.0, 0.0, 0.0]
    for correction in issued:
        metres = correction.magnitude_cm / 100.0
        for i in range(3):
            expected[i] += correction.unit[i] * metres

    query_right = view["layers"]["query"]["right"]
    corrected_right = view["layers"]["corrected"]["right"]
    for q, c in zip(query_right, corrected_right):
        for i in range(3):
            # 부동소수 잡음(~1e-16) 안에서 비교한다 — "정확히 같다"는 요구는 두 축의
            # 합이 반영됐는가를 뜻하지, 덧셈 결합 순서까지 bit-for-bit 같아야 한다는
            # 뜻이 아니다. 실제 결함(축 하나 누락·덮어쓰기)은 이 허용오차보다
            # 여섯 자릿수 이상 크므로 여전히 잡힌다.
            assert math.isclose(c[i] - q[i], expected[i], rel_tol=0.0, abs_tol=1e-9)


def test_layer_point_count_matches_the_local_track_resample_count() -> None:
    """층의 점 개수가 `_local_tracks`의 리샘플 수와 같다(요구사항 4)."""
    query_frames = service.pass_frames(FakeSource().load(PassRef(**QUERY)))
    reference_frames = service.pass_frames(FakeSource().load(PassRef(**REFERENCE)))
    local_query = service._local_tracks(query_frames)  # noqa: SLF001
    local_reference = service._local_tracks(reference_frames)  # noqa: SLF001

    view = _compare()["view"]
    assert len(view["layers"]["query"]["right"]) == len(local_query["right"])
    assert len(view["layers"]["reference"]["right"]) == len(local_reference["right"])
    assert view["grid_n"] == len(local_query["right"])


def test_weights_cover_every_point_the_layers_draw() -> None:
    """`weights`가 그리는 모든 점을 덮는다 — 빠진 점이 있으면 실패한다(요구사항 5)."""
    view = _compare()["view"]
    drawn_points = {name for name in view["layers"]["query"] if name != "hit"}
    assert drawn_points <= set(view["weights"]), (drawn_points, view["weights"])


def test_mesh_is_unavailable_for_a_level_with_no_entry_at_all() -> None:
    """`level_alignment.json`에 항목 자체가 없는 레벨에서 `available`이 `False`이고
    기하가 비어 있다(요구사항 6).

    옛 판본은 질의 픽스처의 레벨 10을 그 예로 썼다 — 그때는 레벨 1만 검증돼 있었기
    때문이다. **지금 레벨 10은 검증된 정렬을 갖는다**(그것이 이 작업의 목적이다).
    그래서 이 검사는 "항목 자체가 없는 레벨"로 옮긴다 — `available: False`를 내는
    세 갈래(항목 없음 / provenance != solved / FBX 파일 없음)는 각각 따로 고정한다.
    """
    mesh = service.mesh_block("99")
    assert mesh is not None
    assert mesh["available"] is False
    assert mesh["vertices"] == []
    assert mesh["triangles"] == []


def test_the_query_fixture_level_now_draws_real_geometry() -> None:
    """레벨 10은 이제 검증된 정렬을 갖는다 — 화면이 벽면을 실제로 그린다.

    이 작업 이전에는 레벨 1만 검증돼 있어 데모에서 레벨 6·10 등을 고르면 작업 공간이
    뜨지 않았다. 그 상태로 되돌아가면 이 테스트가 빨개진다."""
    view = _compare()["view"]
    mesh = view["mesh"]
    assert mesh is not None
    assert mesh["available"] is True
    assert mesh["vertices"] and mesh["triangles"]


def _first_solved_level() -> int:
    import json

    for entry in json.loads(level_mesh.ALIGNMENT_PATH.read_text(encoding="utf-8")):
        if entry["provenance"] == "solved":
            return int(entry["level"])
    raise AssertionError("검증된 정렬이 하나도 없다")


def test_mesh_is_available_for_a_verified_level_with_real_vertices() -> None:
    """검증된 정렬이 있는 레벨에서 `available`이 `True`이고 정점이 있다(요구사항 6).

    레벨 1로 고정하지 않는다 — 레벨 1은 착탄점이 정확히 평면이라 12자유도가 데이터로
    결정되지 않아 `provenance`가 "unverifiable"이다."""
    level = _first_solved_level()
    block = service.mesh_block(str(level))
    assert block is not None
    assert block["available"] is True
    assert block["vertices"], f"레벨 {level} 메시는 정점을 가져야 한다"
    assert block["triangles"], f"레벨 {level} 메시는 삼각형을 가져야 한다"
    assert level_mesh.load_alignment(level).provenance == "solved"


def test_mesh_is_unavailable_for_an_unverifiable_level() -> None:
    """레벨 1은 항목이 있으나 검증할 수 없다 — 화면이 벽을 그리면 안 된다."""
    assert level_mesh.load_alignment(1).provenance == "unverifiable"
    block = service.mesh_block("1")
    assert block is not None
    assert block["available"] is False
    assert block["vertices"] == []


def test_mesh_is_unavailable_for_a_declared_but_unverified_alignment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`provenance`가 "solved"가 아니면(예: "declared") 검증되지 않은 값이다 —
    `capture_frame.to_world`가 이미 이 규칙을 지킨다
    (`test_frame_invariance.py::test_to_world_refuses_a_declared_but_unverified_alignment`
    와 같은 패턴). `mesh_block`이 이 검사를 건너뛰면 검증되지 않은 기하를 화면에
    올리게 된다."""
    import dataclasses

    level = _first_solved_level()
    real = level_mesh.load_alignment(level)
    unverified = dataclasses.replace(real, provenance="declared")
    monkeypatch.setattr(level_mesh, "load_alignment", lambda _level: unverified)
    block = service.mesh_block(str(level))
    assert block is not None
    assert block["available"] is False
    assert block["vertices"] == []


def test_mesh_is_none_when_the_level_itself_is_unknown() -> None:
    """레벨 자체를 모르면(숫자가 아닌 레벨, 예: 실전모드 `1-A`) `mesh`가 `None`이다 —
    `available: False`(레벨은 알지만 정렬이 없음)와 다른 상태다(요구사항 6)."""
    assert service.mesh_block("1-A") is None


def test_mesh_is_unavailable_when_the_source_fbx_file_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """정렬 레코드는 있고(검증된 레벨, provenance="solved") `source_fbx` 경로도 있지만
    그 경로에 실제 파일이 없으면(`read_fbx`가 `FileNotFoundError`를 던진다)
    `mesh_block`이 그 예외를 삼키지 않고 전파하면 `/api/compare` 전체가 500으로
    죽는다(태스크 0). 장식용 벽면 하나 때문에 점수·교정까지 죽어서는 안 된다 —
    `available: False`로 조용히 떨어져야 한다."""

    def _missing(path: Path) -> object:
        raise FileNotFoundError(path)

    monkeypatch.setattr(level_mesh, "read_fbx", _missing)
    block = service.mesh_block(str(_first_solved_level()))
    assert block is not None
    assert block["available"] is False
    assert block["vertices"] == []
    assert block["triangles"] == []


def test_mesh_parse_failure_still_propagates_instead_of_being_swallowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """파일은 있는데 형식이 깨진 경우(`read_fbx`가 `ValueError`를 던지는 경우)는
    버그이지 "배포 조건"이 아니다 — 파일 없음과 같은 취급으로 조용히
    `available: False`에 떨어뜨리면 진짜 파싱 결함을 숨기게 된다. 이 예외는 계속
    크게 터뜨려야 한다."""

    def _broken(path: Path) -> object:
        raise ValueError(f"{path}: not a binary FBX file (bad magic)")

    monkeypatch.setattr(level_mesh, "read_fbx", _broken)
    with pytest.raises(ValueError):
        service.mesh_block(str(_first_solved_level()))


# ---- 11-A → 피드백: 스켈레톤 배선(M6: 실측 회전 + 중력 투영, 스냅 없음) ----------------

IDENTITY = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
_TEST_MOUNTS = service.MountsTable(
    passed=True,
    table={
        "head": AnchorMount(rotation=IDENTITY, offset=(0.0, 0.05, 0.0)),
        "left": AnchorMount(rotation=IDENTITY, offset=(0.02, 0.0, 0.0)),
        "right:한길숏건": AnchorMount(rotation=IDENTITY, offset=(-0.1, 0.0, 0.05)),
    },
    grip_limit_m=dict(service.DEFAULT_GRIP_LIMIT_M),
)
_UNVALIDATED = service.MountsTable(passed=False, table={}, grip_limit_m=dict(service.DEFAULT_GRIP_LIMIT_M))

# 합성 "참" 장착 — 일관된 실측을 만들 때 쓴다.
_KNOWN_MOUNTS = Mounts(
    head=AnchorMount(rotation=rotation_math.euler_matrix("YXZ", (10.0, -20.0, 5.0)), offset=(0.03, 0.08, -0.05)),
    right=AnchorMount(rotation=rotation_math.euler_matrix("YXZ", (-30.0, 40.0, 15.0)), offset=(-0.05, 0.03, 0.06)),
    left=AnchorMount(rotation=rotation_math.euler_matrix("YXZ", (5.0, 70.0, -10.0)), offset=(0.04, -0.02, 0.03)),
)


@pytest.fixture
def test_mounts(monkeypatch: pytest.MonkeyPatch) -> service.MountsTable:
    monkeypatch.setattr(service, "load_mounts", lambda *args, **kwargs: _TEST_MOUNTS)
    return _TEST_MOUNTS


def _set_vec(row: dict[str, str], group: str, kind: str, value) -> None:  # noqa: ANN001
    for axis, component in zip("XYZ", value):
        row[f"{group} {kind} {axis}"] = repr(float(component))


def _consistent_rows(n: int, dt: float, seed: int = 11) -> list[dict[str, str]]:
    """알려진 루트·관절 회전·장착(`_KNOWN_MOUNTS`)에서 실측 세 강체를 만들어 적은 행.

    관절 회전은 프레임마다 무작위(±10°)라 뼈 길이 보존을 모든 프레임에서 시험할 수 있다."""
    rng = random.Random(seed)
    to_up = skeleton.minimal_rotation(np.array(skeleton.RIG_UP), np.array(world_axes.UP, dtype=float))
    rows: list[dict[str, str]] = []
    for i in range(n):
        row: dict[str, str] = {
            "인덱스": str(i),
            "시간(초)": f"{i * dt:.4f}",
            "분사": "On",
            "Ray Texture UV X": "1024",
            "Ray Texture UV Y": "512",
        }
        for column in simulator_csv.JOINT_COLUMNS:
            key = simulator_csv.JOINT_KEYS[column]
            _set_vec(row, column, "Position", _SKELETON_OFFSETS.get(key, (0.0, 0.0, 0.0)))
            _set_vec(row, column, "Rotation", [rng.uniform(-10.0, 10.0) for _ in range(3)])
        offsets = pose_fk_csv.bind_offsets(row)
        positions, rotations = pose_fk_csv.forward_kinematics_with_rotations(row, offsets)
        root = np.array(rotation_math.axis_matrix("Y", 30.0 + 0.5 * i)) @ to_up
        t = np.array([0.5, 1.0, 4.2 + 0.01 * i])
        for point, group in (("head", "카메라"), ("right", "한길숏건"), ("left", "플래쉬")):
            mount = _KNOWN_MOUNTS.get(point)
            joint = skeleton.ANCHOR_JOINTS[point]
            r_meas = root @ np.array(rotations[joint]) @ np.array(mount.rotation)
            marker = root @ np.array(positions[joint]) + t + r_meas @ np.array(mount.offset)
            _set_vec(row, group, "Position", marker)
            matrix = tuple(tuple(float(x) for x in line) for line in r_meas)
            _set_vec(row, group, "Rotation", rotation_math.euler_from_matrix("YXZ", matrix, 1.0))
            if point == "right":
                hit = marker + np.array([0.6, 0.0, 0.0])
        row["Ray point X"], row[" Ray Point Y"], row[" Ray Point Z"] = (repr(float(x)) for x in hit)
        for group in ("한길폴건", "반도폴건"):
            _set_vec(row, group, "Position", (-694.61, 701.29, 698.25))
        _set_vec(row, "반도숏건", "Position", (-2.99, 4.66, -0.27))
        rows.append(row)
    return rows


class SkeletonFakeSource(FakeSource):
    """질의 원본 행에 FK 관절 컬럼을 얹는다. 레퍼런스는 그대로(`FakeSource`와 동일) —
    이 화면은 질의 층에만 스켈레톤을 얹는다(`view_block` 독스트링)."""

    def load(self, ref: PassRef):  # noqa: ANN201
        raw = super().load(ref)
        if ref.worker == QUERY["worker"]:
            return RawPass(rows=_rows_with_skeleton(0.3))
        return raw


def test_skeleton_block_returns_joints_bones_and_fit_when_fk_columns_are_present(test_mounts) -> None:  # noqa: ANN001
    block = service.skeleton_block(RawPass(rows=_rows_with_skeleton(0.3)))
    assert block.joints is not None
    assert set(simulator_csv.JOINT_KEYS.values()) <= set(block.joints)
    assert block.bones == [list(pair) for pair in simulator_csv.BONES]
    assert block.fit is not None
    assert block.fit["method"] == "fk_csv_yxz_orient_gravity"
    assert block.fit["mounts"] == "global_lowo_validated"
    assert isinstance(block.fit["residual_m"], float)
    assert block.fit_reason is None


def test_skeleton_block_reports_a_reason_without_joint_columns() -> None:
    """`FakeSource`의 원본 행에는 관절 컬럼이 없다 — 스켈레톤 없이 사유를 낸다."""
    block = service.skeleton_block(FakeSource().load(PassRef(**QUERY)))
    assert block.joints is None
    assert block.fit is None
    assert block.fit_reason == "no_joint_columns"
    assert block.fit_note == service.FIT_NOTES["no_joint_columns"]


def test_skeleton_joints_are_not_snapped_to_the_measured_markers(test_mounts) -> None:  # noqa: ANN001
    """리그 오프셋을 3배로 부풀려 FK가 실측과 어긋나도 관절을 실측 위치로 덮어쓰지 않는다 —
    어긋남은 앵커 miss로 드러난다(옛 스냅은 목·전완을 0.46~2.1배로 늘였다)."""
    rows = _rows_with_skeleton(0.3)
    for row in rows:
        for axis in "XYZ":
            for column in simulator_csv.JOINT_COLUMNS:
                key = f"{column} Position {axis}"
                if key in row:
                    row[key] = str(float(row[key]) * 3.0)
    raw = RawPass(rows=rows)
    block = service.skeleton_block(raw)
    assert block.fit is not None
    representative = service.representative_frame(raw)
    frame = representative.frame
    assert block.joints["Head"] != list(frame.head)
    assert block.joints["RightHand"] != list(frame.right)
    assert max(anchor["miss_cm"] for anchor in block.fit["anchors"]) > 1.0


def test_skeleton_bones_preserved() -> None:
    """모든 분사 프레임·BONES 쌍에서 배치된 뼈 길이 == 바인드 오프셋 노름(1e-9 m).

    `service.place_frame`이 대표 프레임 배치(`skeleton_block`)가 쓰는 바로 그 경로다."""
    rows = _consistent_rows(40, 0.02)
    raw = RawPass(rows=rows)
    offsets = pose_fk_csv.bind_offsets(rows[0])
    gun = measured_pose.active_gun(rows)
    paired = [(f, r) for f, r in measured_pose.measured_frames_with_rows(rows) if f.spraying and f.hit is not None]
    assert len(paired) == 40
    for frame, row in paired:
        sample = skeleton.frame_sample(frame, row, gun, offsets)
        placed = service.place_frame(sample, _KNOWN_MOUNTS)
        assert placed is not None
        for parent, child in simulator_csv.BONES:
            dist = math.dist(placed.joints[parent], placed.joints[child])
            assert abs(dist - math.hypot(*offsets[child])) < 1e-9, (frame.index, child)
    table = service.MountsTable(
        passed=True,
        table={"head": _KNOWN_MOUNTS.head, "left": _KNOWN_MOUNTS.left, "right:한길숏건": _KNOWN_MOUNTS.right},
        grip_limit_m=dict(service.DEFAULT_GRIP_LIMIT_M),
    )
    block = service.skeleton_block(raw, mounts_table=table)
    for parent, child in simulator_csv.BONES:
        dist = math.dist(block.joints[parent], block.joints[child])
        assert abs(dist - math.hypot(*offsets[child])) < 1e-9, child


def test_anchor_miss_is_reported_not_hidden(test_mounts) -> None:  # noqa: ANN001
    raw = RawPass(rows=_rows_with_skeleton(0.3))
    block = service.skeleton_block(raw)
    frame = service.representative_frame(raw).frame
    measured = {"head": frame.head, "right": frame.right, "left": frame.left}
    assert {a["point"] for a in block.fit["anchors"]} == {"head", "right", "left"}
    for anchor in block.fit["anchors"]:
        assert anchor["measured_marker"] == list(measured[anchor["point"]])
        dist = math.dist(anchor["predicted_marker"], anchor["measured_marker"])
        assert abs(anchor["miss_cm"] - dist * 100.0) < 1e-9
        assert anchor["miss_display"] == f"{anchor['miss_cm']:.1f}cm"


def test_pass_fit_excludes_representative_frame() -> None:
    """폴백 적합 재료에 대표 프레임 ±1.0 s가 없다 — 표본 안 잔차를 검증값처럼 보이지 않게."""
    rows = _consistent_rows(240, 0.02)
    raw = RawPass(rows=rows)
    representative = service.representative_frame(raw)
    offsets = pose_fk_csv.bind_offsets(rows[0])
    samples = service.pass_fit_samples(representative, offsets)
    assert len(samples) >= service.PASS_FIT_MIN_FRAMES
    centre = representative.frame.time_sec
    assert all(abs(sample.time_sec - centre) > service.PASS_FIT_EXCLUDE_SEC for sample in samples)
    assert representative.frame.index not in {sample.index for sample in samples}

    block = service.skeleton_block(raw, mounts_table=_UNVALIDATED)
    assert block.fit is not None
    assert block.fit["mounts"] == "pass_fit_holdout"
    assert "적합 잔차" in block.fit["residual_label"]
    assert all(anchor["miss_display"].startswith("적합 잔차 ") for anchor in block.fit["anchors"])
    # 합성 실측은 알려진 장착에서 만들었으므로 패스 적합이 그것을 되찾아 대표 프레임에서 거의 맞는다.
    assert max(anchor["miss_cm"] for anchor in block.fit["anchors"]) < 1.0


def test_pass_fit_without_sixty_frames_outside_the_window_draws_no_skeleton() -> None:
    raw = RawPass(rows=_consistent_rows(80, 0.02))  # 1.6 s — 대표 프레임 ±1.0 s를 빼면 60 미만
    block = service.skeleton_block(raw, mounts_table=_UNVALIDATED)
    assert block.joints is None
    assert block.fit_reason == "mounts_unavailable"


def test_fit_notes_disclose_estimate(test_mounts) -> None:  # noqa: ANN001
    block = service.skeleton_block(RawPass(rows=_rows_with_skeleton(0.3)))
    notes = block.fit["notes"]
    assert "아바타 추정" in notes[0] and "측정값이 아닙니다" in notes[0]
    assert "중력 방향" in notes[1]
    assert any("목·윗가슴 관절이 없고" in note for note in notes)
    assert block.fit["offset_line_label"] == "보정 오프셋(적합값)"
    assert "적합 잔차" not in block.fit["residual_label"]

    fitted = service.skeleton_block(RawPass(rows=_consistent_rows(240, 0.02)), mounts_table=_UNVALIDATED)
    assert "아바타 추정" in fitted.fit["notes"][0]
    assert "적합 잔차" in fitted.fit["residual_label"]


def test_faded_only_when_gravity_correction_exceeds_twenty_degrees(test_mounts) -> None:  # noqa: ANN001
    raw = RawPass(rows=_rows_with_skeleton(0.3))
    block = service.skeleton_block(raw)
    representative = service.representative_frame(raw)
    offsets = pose_fk_csv.bind_offsets(raw.rows[0])
    mounts = skeleton.mounts_for(_TEST_MOUNTS.table, representative.gun)
    base = skeleton.place_oriented(
        *(lambda s: (s.local_pos, s.local_rot, s.measured))(
            skeleton.frame_sample(representative.frame, representative.row, representative.gun, offsets)
        ),
        mounts,
    )
    import dataclasses

    for gravity, expected in ((20.0, False), (20.0001, True), (3.0, False)):
        placed = dataclasses.replace(base, gravity_correction_deg=gravity)
        fit = service._fit_view(placed, representative, service.MOUNTS_GLOBAL, mounts, offsets, {})  # noqa: SLF001
        assert fit["faded"] is expected, gravity
    assert block.fit["faded"] is (block.fit["gravity_correction_deg"] > 20.0)


def test_compare_wires_skeleton_into_the_query_layer_only(test_mounts) -> None:  # noqa: ANN001
    result = service.compare(SkeletonFakeSource(), PassRef(**QUERY), PassRef(**REFERENCE))
    assert result["status"] == "ok"
    view = result["view"]
    assert "joints" in view["layers"]["query"]
    assert "bones" in view["layers"]["query"]
    assert "joints" not in view["layers"]["reference"], "레퍼런스 층에는 스켈레톤을 얹지 않는다"
    assert "joints" not in view["layers"]["corrected"], "교정 후 층에는 스켈레톤을 얹지 않는다"
    assert view["fit"] is not None
    assert view["fit"]["method"] == "fk_csv_yxz_orient_gravity"
    assert "source" not in view["fit"]


def test_new_joints_are_added_to_weights_at_zero_and_covered_by_the_existing_invariant(test_mounts) -> None:  # noqa: ANN001
    """새 관절은 전부 weights에 0.0으로 들어간다(브리프) — 채점에는 안 들어가지만
    "weights가 그리는 모든 점을 덮는다"는 일반 불변량은 여전히 성립해야 한다."""
    result = service.compare(SkeletonFakeSource(), PassRef(**QUERY), PassRef(**REFERENCE))
    view = result["view"]
    joints = view["layers"]["query"]["joints"]
    assert joints, "이 픽스처는 스켈레톤이 실려야 한다"
    for name in joints:
        assert view["weights"].get(name) == 0.0, name
    drawn_points = {name for name in view["layers"]["query"] if name not in ("hit", "joints", "bones")}
    assert drawn_points <= set(view["weights"])


def test_frame_arrow_starts_at_measured_marker(test_mounts) -> None:  # noqa: ANN001
    """화살표의 대표 프레임 시작점은 그 점의 대표 프레임 실측 마커와 **정확히** 같다."""
    result = service.compare(SkeletonFakeSource(), PassRef(**QUERY), PassRef(**REFERENCE))
    view = result["view"]
    markers = {anchor["point"]: anchor["measured_marker"] for anchor in view["fit"]["anchors"]}
    drawn = [arrow for arrow in view["corrections"] if arrow["from_frame"] is not None]
    assert drawn, "스켈레톤이 있으면 대표 프레임 화살표가 있어야 한다"
    for arrow in drawn:
        assert math.dist(arrow["from_frame"], markers[arrow["point"]]) == 0.0
        metres = arrow["magnitude_cm"] / 100.0
        for i in range(3):
            assert arrow["to_frame"][i] == arrow["from_frame"][i] + arrow["unit"][i] * metres
    assert any(arrow["from_frame"] != arrow["from"] for arrow in drawn)


def test_frame_arrows_are_null_without_skeleton() -> None:
    view = _compare()["view"]
    assert view["corrections"]
    assert all(arrow["from_frame"] is None and arrow["to_frame"] is None for arrow in view["corrections"])


# 실데이터(전형기 3_1000444_1) 첫 행의 휴지 자세 관절 회전. 리그 +Z가 위로 서고 무릎이 발끝 쪽으로 약간 굽는다.
_REST_ROTATIONS: dict[str, tuple[float, float, float]] = {
    "Pelvis": (270.0, 90.0, 0.0),
    "Spine": (0.0, 0.0, 0.0),
    "Chest": (0.0, 0.0, 359.6),
    "Left Thigh": (359.9, 179.0, 358.6),
    "Left Calf": (0.0, 0.0, 14.7),
    "Left Foot": (354.9, 2.1, 348.2),
    "Left Toe": (0.0, 0.0, 270.0),
    "Right Thigh": (0.1, 181.0, 358.6),
    "Right Calf": (0.0, 0.0, 14.7),
    "Right Foot": (5.1, 357.9, 348.2),
    "Right Toe": (0.0, 0.0, 270.0),
}
_BACKWARD_KNEES = {**_REST_ROTATIONS, "Left Calf": (0.0, 0.0, -40.0), "Right Calf": (0.0, 0.0, -40.0)}


def _set_rotations(row: dict[str, str], rotations) -> None:  # noqa: ANN001
    for column, value in rotations.items():
        _set_vec(row, column, "Rotation", value)


def _fk_local(rotations) -> dict:  # noqa: ANN001
    row: dict[str, str] = {}
    for column in simulator_csv.JOINT_COLUMNS:
        _set_vec(row, column, "Position", _SKELETON_OFFSETS.get(simulator_csv.JOINT_KEYS[column], (0.0, 0.0, 0.0)))
        _set_vec(row, column, "Rotation", rotations.get(column, (0.0, 0.0, 0.0)))
    return pose_fk_csv.forward_kinematics(row, pose_fk_csv.bind_offsets(row))


def _legs(knee_x: float, left_ankle_y: float = 0.1) -> dict:
    """리그 +Z가 위인 곧은 다리. `knee_x`만큼 무릎이 앞(+X, 발끝 방향)으로 나온다."""
    pose = {"Pelvis": (0.0, 0.0, 0.05)}
    for side, y, ankle_y in (("Left", 0.1, left_ankle_y), ("Right", -0.1, -left_ankle_y)):
        pose[side + "Thigh"] = (0.0, y, 0.0)
        pose[side + "Calf"] = (knee_x, y, -0.415)
        pose[side + "Foot"] = (0.0, ankle_y, -0.83)
        pose[side + "Toe"] = (0.15, ankle_y, -0.85)
    return pose


def test_leg_check_flags_backward_knees_and_crossed_feet() -> None:
    """무릎이 발끝 반대로 꺾이거나 두 발이 교차하면 해부학 범위 밖이다. 휴지 자세(실데이터 회전)는 범위 안이다."""
    assert skeleton.leg_check(_fk_local(_REST_ROTATIONS)).plausible is True
    assert skeleton.leg_check(_fk_local(_BACKWARD_KNEES)).plausible is False

    forward = skeleton.leg_check(_legs(0.12))
    # 발끝이 약간 아래(−0.02 m)를 향해 굽는 방향과 7.6° 차이가 난다 — 같은 쪽이라는 뜻이다.
    assert forward.knee_flex_deg >= skeleton.KNEE_BENT_DEG and forward.knee_toe_deg < 15.0 and forward.plausible
    backward = skeleton.leg_check(_legs(-0.12))
    assert backward.knee_toe_deg > 170.0 and backward.plausible is False
    crossed = skeleton.leg_check(_legs(0.12, left_ankle_y=-0.1))
    assert crossed.legs_crossed is True
    assert crossed.knee_toe_deg <= skeleton.LEG_LIMITS["knee_toe_deg"] and crossed.leg_tilt_deg <= skeleton.LEG_LIMITS["leg_tilt_deg"]
    assert crossed.plausible is False
    assert skeleton.leg_check({"Pelvis": (0.0, 0.0, 0.0)}) is None


def _rows_with_leg_poses(n: int, plausible_offsets: tuple[int, ...]) -> tuple[list[dict[str, str]], int]:
    rows = _consistent_rows(n, 0.02)
    for row in rows:
        _set_rotations(row, _BACKWARD_KNEES)
    representative = service.representative_frame(RawPass(rows=rows))
    middle = len(representative.spraying) // 2
    for delta in plausible_offsets:
        index = representative.spraying[middle + delta][0].index
        _set_rotations(next(r for r in rows if int(r["인덱스"]) == index), _REST_ROTATIONS)
    return rows, middle


def test_skeleton_frame_picks_the_nearest_frame_with_anatomical_legs(test_mounts) -> None:  # noqa: ANN001
    """가운데 프레임 다리가 범위 밖이면 가운데에서 **가장 가까운** 범위 안 프레임을 고르고(+3, −7 중 +3),
    그 프레임이 스켈레톤·실측 마커·fit의 대표 프레임이 된다. 선택은 notes와 `fit.legs`로 드러난다."""
    rows, middle = _rows_with_leg_poses(41, (3, -7))
    raw = RawPass(rows=rows)
    representative = service.representative_frame(raw)
    chosen = service.skeleton_frame(representative, pose_fk_csv.bind_offsets(rows[0]))
    assert chosen is not None and chosen.legs.plausible is True
    assert chosen.offset_frames == 3
    assert chosen.representative.frame.index == representative.spraying[middle + 3][0].index

    block = service.skeleton_block(raw)
    assert block.fit["frame_index"] == chosen.representative.frame.index
    assert block.fit["legs"]["plausible"] is True and block.fit["legs"]["frame_offset"] == 3
    assert block.fit["legs_visible_default"] is True
    assert service.LEGS_PICKED_NOTE.format(k=3) in block.fit["notes"]
    frame = chosen.representative.frame
    markers = {a["point"]: a["measured_marker"] for a in block.fit["anchors"]}
    assert markers["head"] == list(frame.head) and markers["right"] == list(frame.right)


def test_fit_hides_legs_by_default_when_no_frame_is_anatomical(test_mounts) -> None:  # noqa: ANN001
    rows, _middle = _rows_with_leg_poses(41, ())
    raw = RawPass(rows=rows)
    block = service.skeleton_block(raw)
    legs = block.fit["legs"]
    assert legs["plausible"] is False and block.fit["legs_visible_default"] is False
    assert legs["joints"] == list(skeleton.LEG_JOINTS)
    assert legs["frame_offset"] == 0
    assert block.fit["frame_index"] == service.representative_frame(raw).frame.index
    assert service.LEGS_HIDDEN_NOTE in block.fit["notes"]
    assert "아바타 추정" in block.fit["notes"][0] and "중력 방향" in block.fit["notes"][1]


def test_offset_label_only_when_the_grip_limit_is_exceeded(test_mounts) -> None:  # noqa: ANN001
    raw = RawPass(rows=_rows_with_skeleton(0.3))
    representative = service.representative_frame(raw)
    offsets = pose_fk_csv.bind_offsets(raw.rows[0])
    mounts = skeleton.mounts_for(_TEST_MOUNTS.table, representative.gun)
    sample = skeleton.frame_sample(representative.frame, representative.row, representative.gun, offsets)
    placed = skeleton.place_oriented(sample.local_pos, sample.local_rot, sample.measured, mounts)
    loose = service._fit_view(placed, representative, service.MOUNTS_GLOBAL, mounts, offsets, {representative.gun: 1.0})  # noqa: SLF001
    assert all(a["offset_display"] is None for a in loose["anchors"])
    tight = service._fit_view(placed, representative, service.MOUNTS_GLOBAL, mounts, offsets, {representative.gun: 0.05})  # noqa: SLF001
    labels = {a["point"]: a["offset_display"] for a in tight["anchors"]}
    norm = math.hypot(*_TEST_MOUNTS.table["right:한길숏건"].offset)
    assert labels == {"head": None, "right": f"보정 오프셋 {norm:.2f}m(적합값)", "left": None}
    assert any("손으로 쥔 거리로 설명되지 않습니다" in note for note in tight["notes"])


def test_mesh_reason_detail_is_read_from_the_alignment_record() -> None:
    one = service.mesh_block("1")
    assert "손대칭" in one["reason_detail"]
    two = service.mesh_block("2")
    assert "모델 UV" in two["reason_detail"]
    solved = [str(level) for level in service._solved_levels()]  # noqa: SLF001
    assert solved and one["reason_hint"] == "벽면이 검증된 레벨: " + "·".join(solved)
    ok = service.mesh_block(solved[0])
    assert ok["reason_detail"] is None and ok["reason_hint"] is None


def test_view_fit_is_null_when_skeleton_is_unavailable() -> None:
    view = _compare()["view"]
    assert view["fit"] is None
    assert view["fit_reason"] == "no_joint_columns"
    assert view["fit_note"] == service.FIT_NOTES["no_joint_columns"]
    assert view["display_errors"] == []
    assert view["legend_notes"][0] == "교정 화살표는 건·HMD 실측 마커에서 나갑니다"


def test_mesh_view_carries_reason_text() -> None:
    unverified = service.mesh_block("1")
    assert unverified["reason"] == "unverified_alignment"
    assert unverified["reason_text"] == service.MESH_REASON_TEXT["unverified_alignment"]
    missing = service.mesh_block("99")
    assert missing["reason"] == "no_alignment_record"
    solved = service.mesh_block(str(_first_solved_level()))
    assert solved["available"] is True and solved["reason"] is None and solved["reason_text"] is None


def test_mesh_view_passes_handedness_evidence_through(monkeypatch: pytest.MonkeyPatch) -> None:
    level = _first_solved_level()
    evidence = {
        "gun_occlusion": {"P": 0.0, "mirror": 1.0},
        "hmd_occlusion": {"P": 0.01, "mirror": 0.9},
        "flash_occlusion_report": {"P": 0.0, "mirror": 0.5},
        "rule_doc": "docs/measurements/2026-09-15-fixed-linear-alignment-rule.md",
        "posthoc_parts": ["G_occ_gun", "uniqueness"],
    }
    real = service._alignment_entry  # noqa: SLF001

    def with_evidence(lv: int):  # noqa: ANN202
        entry = dict(real(lv) or {})
        entry.update({"handedness_source": "data:occlusion_gun+hmd", "handedness_evidence": evidence})
        return entry

    monkeypatch.setattr(service, "_alignment_entry", with_evidence)
    block = service.mesh_block(str(level))
    assert block["handedness_source"] == "data:occlusion_gun+hmd"
    assert block["handedness_evidence"] == evidence
    assert block["handedness_note"] == service.HANDEDNESS_NOTE
    assert service.mesh_block(str(level))["available"] is True


def test_deviation_cm_is_the_local_delta_norm_in_centimetres() -> None:
    """②(얼마나 다른가)의 부위별 편차 — `local_delta`의 노름을 cm로 낸다(화면이
    다시 계산하지 않는다는 이 저장소의 규칙, 브리프 ② 각주)."""
    result = _compare()
    for delta in result["measured"]["deltas"]:
        expected = math.dist(delta["delta"], (0.0, 0.0, 0.0)) * 100.0
        assert math.isclose(delta["deviation_cm"], expected, rel_tol=1e-9)
