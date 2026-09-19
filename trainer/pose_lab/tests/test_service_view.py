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

import dataclasses
import importlib
import re
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

from trainer.pose_lab import body_proportions, composite, service, skeleton
from trainer.pose_lab.skeleton import AnchorMount, Mounts
from trainer.pose_lab.service import PassRef, RawPass
from trainer.pose_lab.tests.test_app import FakeSource, QUERY, REFERENCE, _rows
from trainer.pose_lab.tests import test_scoring_isolation as isolation

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

    레벨을 상수로 고정하지 않는다 — 어느 레벨이 검증되는지는 정렬 파일이 정하고
    바뀔 수 있다. 레벨 1의 `provenance`는 꼬리표가 붙은 "solved:..." 변형이라
    `is_solved()`를 거쳐야 한다."""
    level = _first_solved_level()
    block = service.mesh_block(str(level))
    assert block is not None
    assert block["available"] is True
    assert block["vertices"], f"레벨 {level} 메시는 정점을 가져야 한다"
    assert block["triangles"], f"레벨 {level} 메시는 삼각형을 가져야 한다"
    assert level_mesh.is_solved(level_mesh.load_alignment(level).provenance)


def test_mesh_is_unavailable_for_an_unverifiable_level() -> None:
    """레벨 2는 항목이 있으나 검증할 수 없다 — 화면이 벽을 그리면 안 된다."""
    assert level_mesh.load_alignment(2).provenance == "unverifiable"
    block = service.mesh_block("2")
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
    # 몸통 줄기는 그려 넣은 `Spine1`·`Neck`이 들어간 다섯 마디로 갈아 끼운다(U4).
    assert block.bones == body_proportions.display_bones(simulator_csv.BONES, block.joints)
    assert ["Spine", "Spine1"] in block.bones and ["Chest", "Neck"] in block.bones
    assert ["Spine", "Chest"] not in block.bones
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
    # 서버가 FK에 넘기는 바로 그 리그다(`service.skeleton_block`) — 몸통 두 마디가 다시 재진다.
    offsets = body_proportions.rig_offsets(pose_fk_csv.bind_offsets(rows[0]))
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
    # 그려 넣은 두 관절은 선분 위 내분점이라 줄기 다섯 마디의 합이 세 마디의 합과 같다.
    stem = sum(math.dist(block.joints[a], block.joints[b]) for a, b in body_proportions.TORSO_STEM_BONES)
    old_stem = sum(math.hypot(*offsets[name]) for name in body_proportions.TORSO_BONES)
    assert abs(stem - old_stem) < 1e-9, (stem, old_stem)


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
    assert any("목·윗가슴 관절이 없어" in note and "BVH" in note for note in notes)
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


class BothSkeletonFakeSource(FakeSource):
    """질의·기준 **양쪽** 원본 행에 FK 관절 컬럼을 얹는다.

    사용자가 원한 그림이 "기준 스켈레톤과 질의 스켈레톤을 같이 그리고 유사도가 떨어지는 조인트일수록
    색상을 진하게"이므로(S1) 두 층이 다 서야 한다. 실측 마커를 0.3m 어긋나게 둬 관절 편차가 0이
    아니게 만든다 — 0이면 색 규칙이 실제로 도는지 알 수 없다."""

    def load(self, ref: PassRef):  # noqa: ANN201
        return RawPass(rows=_rows_with_skeleton(0.3 if ref.worker == QUERY["worker"] else 0.0))


def test_compare_wires_skeleton_into_the_query_layer_only(test_mounts) -> None:  # noqa: ANN001
    """기준 원본 행에 관절 컬럼이 없으면 기준 층에는 스켈레톤이 없다 — 지어내지 않는다."""
    result = service.compare(SkeletonFakeSource(), PassRef(**QUERY), PassRef(**REFERENCE))
    assert result["status"] == "ok"
    view = result["view"]
    assert "joints" in view["layers"]["query"]
    assert "bones" in view["layers"]["query"]
    assert "joints" not in view["layers"]["reference"], "기준 원본에 관절 컬럼이 없으면 세우지 않는다"
    assert view["reference_fit"] is None
    assert view["reference_fit_reason"] == "no_joint_columns"
    assert view["deviation"] is None, "한쪽만 서면 벌어짐을 잴 수 없다 — 0으로 칠하지 않는다"
    assert "joints" not in view["layers"]["corrected"], "교정 후 층에는 스켈레톤을 얹지 않는다"
    assert view["fit"] is not None
    assert view["fit"]["method"] == "fk_csv_yxz_orient_gravity"
    assert "source" not in view["fit"]


def test_reference_layer_carries_a_skeleton_built_by_the_same_path(test_mounts) -> None:  # noqa: ANN001
    """기준도 **질의와 같은 경로**(`skeleton_block`)로 선다(S1). 두 경로를 두면 같은 사람이
    층에 따라 다른 몸이 된다 — 여기서는 기준 층의 관절이 그 패스를 직접 세운 것과 글자까지 같다."""
    source = BothSkeletonFakeSource()
    result = service.compare(source, PassRef(**QUERY), PassRef(**REFERENCE))
    view = result["view"]
    reference = view["layers"]["reference"]
    assert "joints" in reference and "bones" in reference
    direct = service.skeleton_block(source.load(PassRef(**REFERENCE)))
    assert reference["joints"] == direct.joints
    assert reference["bones"] == direct.bones
    assert view["reference_fit"] is not None
    assert view["reference_fit"]["method"] == view["fit"]["method"]
    # 기준 몸은 질의 몸과 **다른 자리**에 선다 — 같으면 겹쳐 그릴 이유가 없다.
    assert reference["joints"] != view["layers"]["query"]["joints"]


def test_joint_deviation_is_the_world_gap_between_the_two_skeletons(test_mounts) -> None:  # noqa: ANN001
    """관절 색의 근거를 **서버가** 낸다(S1). 화면은 이 값으로 진하기만 정하고 다시 계산하지 않는다.

    재는 것은 배치가 끝난 월드 거리다 — 화면에 실제로 보이는 간격이 곧 색이 된다. 여기서는 서버
    값과 두 층의 관절 좌표로 독립적으로 다시 잰 거리가 같은지 본다(구현의 합산 경로를 재사용하지
    않는다)."""
    view = service.compare(BothSkeletonFakeSource(), PassRef(**QUERY), PassRef(**REFERENCE))["view"]
    deviation = view["deviation"]
    assert deviation is not None
    assert deviation["basis"] == "world_distance_m"
    query_joints = view["layers"]["query"]["joints"]
    reference_joints = view["layers"]["reference"]["joints"]
    assert set(deviation["joints"]) == set(query_joints) & set(reference_joints)
    assert deviation["joints"], "이 픽스처는 공통 관절이 있어야 한다"
    for name, value in deviation["joints"].items():
        expected = math.dist(query_joints[name], reference_joints[name])
        assert value == pytest.approx(expected, abs=1e-12), name
    assert max(deviation["joints"].values()) > 0.0, "0이면 색 규칙이 도는지 알 수 없다"
    assert deviation["joint_max_m"] == pytest.approx(max(deviation["joints"].values()))


def test_deviation_keeps_measured_and_estimated_apart(test_mounts) -> None:  # noqa: ANN001
    """실측 3점의 벌어짐은 실측이고 나머지는 추정이다 — 한 자루에 담으면 화면이 구분을 잃는다(S1).

    추적 장치가 붙은 강체는 머리(HMD)·건·플래쉬 셋뿐이다. 그 셋의 마커 간격만 `markers`에 들어가고,
    관절은 전부 `joints`에 들어가며 눈금도 따로 나온다."""
    source = BothSkeletonFakeSource()
    view = service.compare(source, PassRef(**QUERY), PassRef(**REFERENCE))["view"]
    deviation = view["deviation"]
    assert set(deviation["markers"]) <= set(service.DEVIATION_MEASURED_POINTS)
    assert deviation["measured_points"] == [
        name for name in service.DEVIATION_MEASURED_POINTS if name in deviation["markers"]
    ]
    assert deviation["markers"], "이 픽스처는 실측 앵커가 있어야 한다"
    # 실측 마커 간격은 두 스켈레톤의 앵커에서 독립적으로 다시 잰 값과 같다.
    query_markers = {a["point"]: a["measured_marker"] for a in view["fit"]["anchors"]}
    reference_markers = {a["point"]: a["measured_marker"] for a in view["reference_fit"]["anchors"]}
    for point, value in deviation["markers"].items():
        assert value == pytest.approx(math.dist(query_markers[point], reference_markers[point]), abs=1e-12)
    # 관절 이름이 실측 자루에 섞이지 않는다.
    assert not (set(deviation["joints"]) & set(deviation["markers"]))
    assert deviation["marker_max_m"] == pytest.approx(max(deviation["markers"].values()))
    assert "추정끼리의 차이" in deviation["estimate_note"]
    assert "실측끼리의 차이" in deviation["measured_note"]


def test_deviation_scale_comes_from_joints_the_screen_actually_draws(test_mounts) -> None:  # noqa: ANN001
    """진하기 눈금은 **그려지는 관절**에서 잡는다(S1).

    안 그리는 관절이 눈금을 정하면 보이는 관절이 색 범위의 아랫도리에만 몰린다 — 검증 라운드
    실측: L3의 argmax가 숨긴 `LeftCalf` 0.7988m라 보이는 관절의 t가 0.412에서 멈춰 설계된 색
    범위의 26%만 썼고, 범례는 화면 어디에도 없는 "79.9cm"를 적었다. 사용자가 요청한
    "유사도가 떨어지는 조인트일수록 진하게"가 가장 필요한 앉은 자세에서 가장 약해진 것이다."""
    source = BothSkeletonFakeSource()
    view = service.compare(source, PassRef(**QUERY), PassRef(**REFERENCE))["view"]
    query_skeleton = service.skeleton_block(source.load(PassRef(**QUERY)))
    reference_skeleton = service.skeleton_block(source.load(PassRef(**REFERENCE)), who="기준")

    shown = service.deviation_block(query_skeleton, reference_skeleton)
    assert shown["scale_excludes"] == []
    assert shown["joint_max_m"] == pytest.approx(max(shown["joints"].values()))

    hidden_names = tuple(skeleton.LEG_JOINTS)
    cut = service.deviation_block(query_skeleton, reference_skeleton, hidden_joints=hidden_names)
    drawn = {n: v for n, v in cut["joints"].items() if n not in hidden_names}
    assert drawn, "이 픽스처는 다리 말고도 관절이 있어야 한다"
    # 눈금은 그리는 관절의 최대다. 이름도 같이 나오고, 그 이름이 숨긴 목록에 들어 있지 않다.
    assert cut["joint_max_m"] == pytest.approx(max(drawn.values()))
    assert cut["joint_max_name"] == max(drawn, key=lambda n: drawn[n])
    assert cut["joint_max_name"] not in hidden_names
    assert sorted(cut["scale_excludes"]) == sorted(n for n in hidden_names if n in cut["joints"])
    # `joints` 자체는 줄지 않는다 — 숨긴 것은 눈금이지 값이 아니다.
    assert set(cut["joints"]) == set(shown["joints"])


def test_deviation_scale_ignores_a_hidden_joint_that_is_the_true_max(test_mounts) -> None:  # noqa: ANN001
    """검증 라운드가 지목한 구멍: `BothSkeletonFakeSource`는 강체를 통째로 옮기므로 관절마다
    벌어짐이 거의 똑같다. 그래서 `scale_source = drawn if drawn else joints`를 `scale_source = joints`로
    되돌려도 `test_deviation_scale_comes_from_joints_the_screen_actually_draws`는 안 빨개진다 —
    숨긴 관절과 보이는 관절의 값이 우연히 비슷해서 argmax가 갈리지 않기 때문이다.

    여기서는 숨긴 다리 관절 하나를 **일부러** 보이는 관절보다 훨씬 멀게 둬서, 눈금이 숨긴 값을
    주워가는지를 직접 가른다. `SkeletonResult`를 바로 만들어 FK·CSV 경로를 거치지 않는다 —
    이 검사는 `deviation_block` 하나만 본다."""
    query = service.SkeletonResult(
        joints={"Head": [0.0, 0.0, 0.0], "Pelvis": [0.0, 0.0, 0.0], "LeftCalf": [0.0, 0.0, 0.0]},
        bones=[], fit=None, fit_reason=None, fit_note=None, markers={}, representative=None,
    )
    reference = service.SkeletonResult(
        joints={"Head": [0.05, 0.0, 0.0], "Pelvis": [0.03, 0.0, 0.0], "LeftCalf": [5.0, 0.0, 0.0]},
        bones=[], fit=None, fit_reason=None, fit_note=None, markers={}, representative=None,
    )
    cut = service.deviation_block(query, reference, hidden_joints=skeleton.LEG_JOINTS)
    drawn = {n: v for n, v in cut["joints"].items() if n not in skeleton.LEG_JOINTS}
    assert drawn, "이 픽스처는 다리 말고도 관절이 있어야 한다"
    assert "LeftCalf" not in drawn
    assert cut["joints"]["LeftCalf"] == pytest.approx(5.0), "숨긴 관절이 실제로 눈에 띄게 더 멀어야 한다"
    assert cut["joint_max_m"] == pytest.approx(max(drawn.values()))
    assert cut["joint_max_name"] == max(drawn, key=lambda n: drawn[n])
    assert cut["joint_max_name"] != "LeftCalf"


def test_deviation_legend_names_a_joint_that_is_on_the_screen(test_mounts) -> None:  # noqa: ANN001
    """범례 문구를 **서버가** 짓는다. 화면이 이름과 숫자를 조립하면 눈금과 문구가 갈린다 —
    이전 판은 "가장 진한 곳 79.9cm"라고 적었는데 그 관절은 다리를 숨겨서 화면에 없었다.

    실데이터 L3·L6은 안 그린 다리 관절이 그리는 관절보다 더 멀다. 그 사실도 같은 자리에서 말한다."""
    for level, session in (("3", "1000391_1"), ("6", "1000394_1")):
        source, kwargs, query, reference = _real_pair(level, session, 0)
        view = service.compare(source, query, reference, **kwargs)["view"]
        deviation = view["deviation"]
        # 이 두 레벨은 두 층 다 IK 다리가 해부학 범위 밖이라 **기본 그림**이 자리표시 다리다(B).
        # IK 토글은 열려 있지만 기본으로 켜지지 않으므로, 눈금은 화면에 있는 관절에서만 잡는다.
        assert view["fit"]["legs_visible_default"] is False, level
        assert view["reference_fit"]["legs_visible_default"] is False, level
        assert view["fit"]["legs_within_range"] is False, level
        assert view["reference_fit"]["legs_within_range"] is False, level
        assert view["fit"]["legs_toggle_enabled"] is True, level
        assert view["reference_fit"]["legs_toggle_enabled"] is True, level
        assert deviation["scale_excludes"], level
        legend = deviation["legend"]
        first = legend[0]
        assert deviation["joint_max_label"] in first, (level, first)
        assert deviation["joint_max_display"] in first, (level, first)
        # 눈금을 정한 관절은 안 그린 관절이 아니다.
        assert deviation["joint_max_name"] not in deviation["scale_excludes"], level
        # 안 그린 관절이 더 멀면 그 사실을 적는다 — 숨겼다는 것을 숨기지 않는다.
        assert deviation["hidden_max_m"] > deviation["joint_max_m"], level
        hidden_line = next(line for line in legend if deviation["hidden_max_label"] in line)
        assert deviation["hidden_max_display"] in hidden_line, (level, hidden_line)
        # 옛 문구(그리지 않는 관절의 숫자를 "가장 진한 곳"이라고 부른 것)는 어디에도 없다.
        assert all("가장 진한 곳" not in line for line in legend), level


def test_deviation_says_how_much_of_the_gap_is_the_whole_body_shifting(test_mounts) -> None:  # noqa: ANN001
    """관절 편차는 배치가 끝난 월드 거리라, 두 패스의 벽 위 자리 차이가 **모든 관절에 공통
    평행이동으로** 섞인다. 한 관절의 색을 "이 관절만 다르다"로 읽으면 과대 해석이다 — 범례가
    그 몫이 얼마인지 말한다.

    값은 두 스켈레톤 관절 무게중심 사이 거리다. 여기서는 기준 몸을 통째로 옮겨 그 값이 정확히
    옮긴 만큼 따라 움직이는지 본다(공통 평행이동을 재는 양이 맞는지)."""
    source = BothSkeletonFakeSource()
    query = service.skeleton_block(source.load(PassRef(**QUERY)))
    reference = service.skeleton_block(source.load(PassRef(**REFERENCE)), who="기준")
    base = service.deviation_block(query, reference)
    assert base["common_shift_m"] == pytest.approx(
        math.dist(
            [sum(query.joints[n][i] for n in base["joints"]) / len(base["joints"]) for i in range(3)],
            [sum(reference.joints[n][i] for n in base["joints"]) / len(base["joints"]) for i in range(3)],
        ),
        abs=1e-12,
    )
    shifted = dataclasses.replace(
        reference, joints={n: [p[0] + 0.5, p[1], p[2]] for n, p in reference.joints.items()}
    )
    moved = service.deviation_block(query, shifted)
    # 몸을 +x로 0.5m 옮기면 공통 몫이 그 벡터만큼 달라진다(부호는 방향에 따라 ±0.5 이동).
    assert abs(moved["common_shift_m"] - base["common_shift_m"]) > 0.01
    assert service.DEVIATION_SHIFT_LEGEND.format(cm=base["common_shift_display"]) in base["legend"]


def test_leg_notes_carry_each_layers_own_verdict(test_mounts) -> None:  # noqa: ANN001
    """다리 고지는 **층마다 자기 판정**을 말한다(S4).

    질의 것만 싣던 동안에는 화면이 그림과 반대되는 말을 했다: 김경호 L1 1000389_1 #9는 기준 층
    다리 뼈 8개가 그려져 있는데 "다리는 그리지 않았습니다"라고 적혔다. 자동 기준으로 맺은 24쌍 중
    10쌍(41.7%)에서 두 층의 `leg_check` 판정이 갈린다."""
    source, kwargs, query, reference = _real_pair("1", "1000389_1", 9)
    view = service.compare(source, query, reference, **kwargs)["view"]
    # 이 쌍이 바로 검증 라운드가 잡은 방향이다: 대상은 IK가 범위 밖, 기준은 범위 안이다.
    assert view["fit"]["legs_within_range"] is False
    assert view["reference_fit"]["legs_within_range"] is True
    # 두 층 다 IK 다리가 있으므로 **토글은 열려 있다**(B). 갈리는 것은 고지의 꼬리다.
    assert view["fit"]["legs_toggle_enabled"] is True
    assert view["reference_fit"]["legs_toggle_enabled"] is True
    notes = view["leg_notes"]
    assert len(notes) == 2, notes
    # 두 줄 다 **자리표시**를 말한다(B) — 다리가 없는 층은 이제 없다. 갈리는 것은 IK 토글 꼬리다.
    assert all("자리표시입니다(측정 아님)" in line for line in notes), notes
    assert notes[0].startswith("대상 ") and notes[1].startswith("기준 ")
    # 범위 밖 줄은 `{why}`가 채워져 있어 상수와 글자가 같지 않다 — 앞뒤 토막과 숫자로 확인한다.
    head, tail = service.LEGS_IK_OUT_OF_RANGE_SUFFIX.split("{why}")
    assert head in notes[0] and notes[0].endswith(tail), notes[0]
    assert "°" in notes[0], notes[0]
    assert notes[1].endswith(service.LEGS_IK_AVAILABLE_SUFFIX), notes[1]
    # 그림과 반대되는 문장이 없다: 다리를 그리는 층의 줄에 "그리지 않았습니다"가 없다.
    assert all("그리지 않았습니다" not in line for line in notes), notes


def test_leg_notes_merge_into_one_line_when_both_layers_agree() -> None:
    """두 층이 **같은 문장**이면 "대상·기준" 한 줄로 합친다 — 같은 말을 두 번 싣지 않는다.

    실데이터로는 이 성질을 더 이상 보일 수 없다: 범위 밖 고지에 층마다 자기 숫자가 붙어 두 문장이
    갈리기 때문이다(아래 검사). 합치는 규칙 자체는 그대로이므로 여기서 규칙만 따로 고정한다.
    """
    same = {"leg_note_template": "{who} 다리는 자리표시입니다(측정 아님)."}
    merged = service.leg_notes(dict(same), dict(same))
    assert merged == ["대상·기준 다리는 자리표시입니다(측정 아님)."], merged


def test_the_floating_foot_is_the_short_rig_torso_not_short_legs() -> None:
    """서 있는 패스에서 발이 뜨는 원인을 **가른다**(A/B의 미해결 자리).

    `skeleton.py`의 자리표시 주석이 적은 두 비율(골반→머리 · 골반→발목을 같은 작업자 BVH와 견준
    값)을 **실제 배치에서 다시 재서** 그 주석과 맞춘다. 다리가 짧은 것이 아니라 몸통이 짧다는 것이
    이 두 숫자의 뜻이다. 몸통을 늘리면 실측(앵커 어긋남)이 나빠지므로 몸통은 그대로 두고, 측정이
    아닌 쪽(자리표시 다리)을 그 틈만큼 늘려 발을 바닥에 세운다 — 늘린 길이는 이 틈과 같은 숫자다.

    **`test_mounts` 픽스처를 쓰지 않는다.** 그 픽스처는 `service.load_mounts`를 합성 장착 상수로
    바꾸는데, 장착 상수가 바뀌면 골반이 다른 높이에 놓인다 — 같은 패스에서 엉덩이–다리 틈이
    합성 상수로 0.3263m, 운영 상수(`skeleton_mounts.json`, 서버가 쓰는 것)로 0.2120m다. 주석이
    적은 값은 서버가 내는 값이므로 여기서도 운영 상수로 재야 출처와 라벨이 같아진다.
    """
    source, kwargs, query, reference = _real_pair("1", "1000389_1", 0)
    view = service.compare(source, query, reference, **kwargs)["view"]
    joints = view["layers"]["query"]["joints"]
    legs = view["fit"]["legs"]

    def span(chain: tuple[str, ...]) -> float:
        return sum(math.dist(joints[a], joints[b]) for a, b in zip(chain, chain[1:]))

    torso = span(("Pelvis", "Spine", "Spine1", "Chest", "Neck", "Head"))
    leg = span(("Pelvis", "LeftThigh", "LeftCalf", "LeftFoot"))
    bvh_torso = sum(body_proportions.BVH_TORSO_M.values())
    bvh_leg = sum(body_proportions.BVH_LEG_M.values())

    comment = (Path(service.__file__).parent / "skeleton.py").read_text(encoding="utf-8")
    written = {
        key: (float(csv), float(bvh), float(ratio))
        for key, csv, bvh, ratio in re.findall(
            r"골반→(머리|발목) CSV ([\d.]+)m / BVH ([\d.]+)m = ([\d.]+)", comment
        )
    }
    assert set(written) == {"머리", "발목"}, written
    for name, measured, bvh in (("머리", torso, bvh_torso), ("발목", leg, bvh_leg)):
        csv_written, bvh_written, ratio_written = written[name]
        assert measured == pytest.approx(csv_written, abs=5e-4), (name, measured, csv_written)
        assert bvh == pytest.approx(bvh_written, abs=5e-4), (name, bvh, bvh_written)
        assert measured / bvh == pytest.approx(ratio_written, abs=5e-4), (name, measured / bvh)
    # 몸통은 짧고(<0.8) 다리는 거의 제 길이다(>0.95) — 이 순서가 뒤집히면 원인 진단이 뒤집힌다.
    assert torso / bvh_torso < 0.8 < 0.95 < leg / bvh_leg
    # 뜨는 높이 = 엉덩이 − 리그 다리. 사용자 결정에 따라 **늘리지 않고** 그만큼 띄운 채 그 틈을
    # 화면이 숫자와 점선으로 말한다(바로 앞 판은 여기서 다리를 1.25배 늘리고 배율을 적었다).
    gap = legs["hip_above_floor_m"] - legs["rig_leg_length_m"]
    assert gap == pytest.approx(0.212, abs=0.005)
    assert legs["placeholder_reaches_floor"] is False
    # 두 다리 길이가 1e-7m 자리에서 다르므로(리그 오프셋) 여기서는 그 자리까지만 견준다.
    above = legs["placeholder_feet_above_floor_m"]
    assert max(above.values()) == pytest.approx(gap, abs=1e-6)
    assert min(above.values()) > 0.0, above  # 두 발 다 떠 있다
    # 보조선은 **뜬 발마다** 하나씩이고, 선의 길이가 곧 옆에 적히는 간격이다(화면이 다시 재지 않는다).
    expected_gap = {}
    for _hip, _knee, ankle, toe in skeleton.PLACEHOLDER_LEG_CHAIN:
        expected_gap[ankle] = above[ankle]
        expected_gap[toe] = above[ankle]
    guides = legs["placeholder_floor_gaps"]
    assert len(guides) == len(above), guides
    for guide in guides:
        assert guide["gap_m"] == pytest.approx(expected_gap[guide["joint"]], abs=1e-12), guide
        assert math.dist(guide["from"], guide["to"]) == pytest.approx(guide["gap_m"], abs=1e-9), guide
        assert guide["label"] == f"간격 {guide['gap_m'] * 100.0:.0f}cm", guide
    # 고지도 같은 숫자를 말하고, 그 숫자가 점선으로 그려진다는 것까지 적는다.
    note = view["fit"]["notes"][view["fit"]["note_keys"].index("legs")]
    assert f"최대 {max(above.values()):.2f}m 떠 있고" in note, note
    assert "점선" in note, note


def test_out_of_range_ik_opens_the_toggle_and_the_note_carries_the_numbers(test_mounts) -> None:  # noqa: ANN001
    """범위 밖 IK도 **토글로 볼 수 있고**(B), 고지가 어느 검사를 얼마나 벗어났는지 숫자로 말한다.

    예전에는 이 패스에서 체크박스가 잠겨 있었다(검증 실측: 6조합 중 4칸). 잠긴 컨트롤은 고장으로
    읽히고, 사용자 정책은 "IK는 토글로 볼 수 있게"였다. 숫자는 `leg_check`가 잰 값이고 문턱은
    `skeleton.LEG_LIMITS`다 — 두 출처가 한 문장에 그대로 실린다.
    """
    source, kwargs, query, reference = _real_pair("3", "1000391_1", 0)
    view = service.compare(source, query, reference, **kwargs)["view"]
    assert view["fit"]["legs_toggle_enabled"] is True
    assert view["fit"]["legs_within_range"] is False
    notes = view["leg_notes"]
    # 층마다 자기 숫자가 붙으므로 두 줄이다(합쳐지지 않는다).
    assert len(notes) == 2, notes
    note = next(line for line in notes if line.startswith("대상 "))
    assert "무릎 굴곡 145.8° > 110°" in note, note
    assert "무릎-발끝 101.7° > 60°" in note, note
    assert f"> {skeleton.LEG_LIMITS['knee_flex_deg']:.0f}°" in note, note
    # 범위 안이던 검사는 문장에 없다(실측 발끝 벌어짐 9.9° ≤ 60°).
    assert "발끝 벌어짐" not in note, note


def test_reference_stroke_comes_from_the_reference_pass_not_the_query(test_mounts) -> None:  # noqa: ANN001
    """기준 쪽 벽면 스트로크가 실린다(S2). 사용자가 "기준 스트로크(벽면 착탄점)이 보이지 않음"이라고
    했고, 그 원인은 `view.surface_hits`가 질의 전용이었던 것이다. 두 줄은 서로 달라야 한다 —
    같으면 질의 것을 두 번 그린 것이다."""
    source, kwargs, query, reference = _real_pair("1", "1000389_1", 0)
    view = service.compare(source, query, reference, **kwargs)["view"]
    assert view["reference_surface_hits"], "기준 착탄점이 실려야 한다"
    assert view["reference_surface_hits"] != view["surface_hits"]
    direct = composite.surface_hits(service.skeleton_block(source.load(reference)).representative)
    assert view["reference_surface_hits"] == direct
    # 자르는 규칙은 하나다 — 기준 구간도 질의와 같은 함수(`composite.stroke_runs`)가 낸다.
    assert view["reference_stroke"]["point_count"] == len(view["reference_surface_hits"])
    assert view["reference_stroke"]["tolerance_mm"] == view["stroke"]["tolerance_mm"]
    flattened = [point for run in view["reference_stroke"]["runs"] for point in run]
    assert all(point in view["reference_surface_hits"] for point in flattened), "점을 만들지 않는다"


def _real_pair(level: str, session: str, pass_index: int):  # noqa: ANN202
    """실데이터 한 쌍(대상 + 자동으로 고른 전형기 기준). 실데이터가 없으면 skip하지 않고 실패한다."""
    source, kwargs = isolation._source_for({"name": "real"})  # noqa: SLF001
    query = PassRef(worker="김경호", date="20250831", level=level, session=session, pass_index=pass_index)
    candidates = service.reference_candidates(source, query)
    assert candidates["status"] == "ok" and candidates["candidates"], candidates
    first = candidates["candidates"][0]
    reference = PassRef(**{key: first[key] for key in ("worker", "date", "level", "session", "pass_index")})
    return source, kwargs, query, reference


def test_real_pass_draws_two_skeletons_close_enough_to_read_as_one_picture() -> None:
    """실데이터: 두 스켈레톤이 **겹쳐 보이는 거리**에 선다(S1).

    같은 동작 키의 두 패스라도 벽 위 자리가 다르면 월드에서 멀리 떨어져 그림이 두 사람으로
    읽힌다. 실측(김경호 20250831 대 자동 기준): 관절 편차 중앙값이 L1 0.285m · L3 0.246m ·
    L6 0.241m이고 최대가 0.60~0.80m다 — 사람 한 명 크기 안이다. 이 성질이 깨지면 색을
    아무리 잘 칠해도 사용자가 원한 그림이 나오지 않는다.
    """
    for level, session in (("1", "1000389_1"), ("3", "1000391_1"), ("6", "1000394_1")):
        source, kwargs, query, reference = _real_pair(level, session, 0)
        view = service.compare(source, query, reference, **kwargs)["view"]
        deviation = view["deviation"]
        assert deviation is not None and deviation["joints"], level
        values = sorted(deviation["joints"].values())
        median = values[len(values) // 2]
        assert 0.05 < median < 0.5, (level, median)
        assert deviation["joint_max_m"] < 1.0, (level, deviation["joint_max_m"])
        # 실측 3점도 다 잡힌다 — 추정만 있고 실측이 없으면 화면이 구분을 보여 줄 수 없다.
        assert set(deviation["markers"]) == {"head", "right", "left"}, (level, deviation["markers"])


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
    # 기본 표시는 **자리표시 다리**다(B) — IK가 범위 안이어도 기본은 자리표시이고, IK는 토글이다.
    assert block.fit["legs"]["display_default"] == "placeholder"
    assert block.fit["legs_visible_default"] is False
    assert block.fit["legs_toggle_enabled"] is True
    assert block.fit["leg_note_template"].endswith(service.LEGS_IK_AVAILABLE_SUFFIX)
    assert any(n == block.fit["leg_note_template"].format(who=block.fit["leg_note_who"]) for n in block.fit["notes"])
    frame = chosen.representative.frame
    markers = {a["point"]: a["measured_marker"] for a in block.fit["anchors"]}
    assert markers["head"] == list(frame.head) and markers["right"] == list(frame.right)


def test_out_of_range_ik_stays_off_by_default_but_the_toggle_opens(test_mounts) -> None:  # noqa: ANN001
    """IK가 범위 밖이어도 **토글은 열린다**(B) — 기본 그림만 자리표시로 남는다.

    범위 밖 다리를 저절로 그리지 않는다는 판단(S4)은 `legs_visible_default is False`가 지킨다.
    """
    rows, _middle = _rows_with_leg_poses(41, ())
    raw = RawPass(rows=rows)
    block = service.skeleton_block(raw)
    legs = block.fit["legs"]
    assert legs["plausible"] is False and block.fit["legs_within_range"] is False
    assert block.fit["legs_toggle_enabled"] is True
    assert block.fit["legs_visible_default"] is False
    assert legs["display_default"] == "placeholder"
    assert legs["joints"] == list(skeleton.LEG_JOINTS)
    assert legs["frame_offset"] == 0
    assert block.fit["frame_index"] == service.representative_frame(raw).frame.index
    head, tail = service.LEGS_IK_OUT_OF_RANGE_SUFFIX.split("{why}")
    template = block.fit["leg_note_template"]
    assert head in template and template.endswith(tail), template
    # 어느 검사를 얼마나 벗어났는지가 **숫자로** 적힌다(문턱의 출처는 skeleton.LEG_LIMITS 하나뿐).
    assert "°" in template and any(label in template for label in service.LEG_CHECK_LABELS.values()), template
    assert "자리표시입니다(측정 아님)" in template
    assert "아바타 추정" in block.fit["notes"][0] and "중력 방향" in block.fit["notes"][1]


def test_placeholder_legs_are_drawn_and_the_screen_calls_them_a_placeholder(test_mounts) -> None:  # noqa: ANN001
    """자리표시 다리가 기본 표시로 돌아왔다(B). **그리는 것과 말하는 것이 같아야 한다** —
    재지 않은 것을 잰 것처럼 보이게 하지 않는다는 규칙이라, 그릴 기하를 싣는 바로 그 응답이
    "측정 아님"도 같이 말해야 한다. notes는 위치가 아니라 `note_keys`로 짝짓는다."""
    rows, _middle = _rows_with_leg_poses(41, ())
    block = service.skeleton_block(RawPass(rows=rows))
    fit = block.fit
    legs = fit["legs"]
    assert legs["plausible"] is False
    assert legs["display_default"] == "placeholder"
    # 그릴 기하가 실린다 — 관절 이름도 뼈 짝도 서버가 준다(화면이 지어내지 않는다).
    assert legs["placeholder_joints"] and legs["placeholder_bones"]
    assert set(legs["placeholder_joints"]) <= set(skeleton.LEG_JOINTS)
    assert legs["rig_leg_length_m"] > 0.0
    assert len(fit["note_keys"]) == len(fit["notes"])
    assert fit["note_keys"][:4] == ["avatar_estimate", "gravity", "legs", "torso"]
    leg_note = fit["notes"][fit["note_keys"].index("legs")]
    # 기하를 실으면서 "측정 아님"을 말하지 않는 응답은 이 검사를 통과하지 못한다.
    assert "자리표시입니다(측정 아님)" in leg_note, leg_note
    assert set(fit["notes_collapsed"]) <= set(fit["note_keys"])
    # 무엇이 추정인지·다리가 무엇인지는 접지 않는다.
    assert "legs" not in fit["notes_collapsed"] and "avatar_estimate" not in fit["notes_collapsed"]


def test_plausible_ik_legs_stay_behind_a_toggle_not_the_default(test_mounts) -> None:  # noqa: ANN001
    """IK 다리가 해부학 범위 안이어도 **기본은 자리표시**다(사용자 정책 B). IK는 토글로 본다."""
    rows, _middle = _rows_with_leg_poses(41, (0,))
    fit = service.skeleton_block(RawPass(rows=rows)).fit
    assert fit["legs"]["plausible"] is True
    assert fit["legs"]["display_default"] == "placeholder"
    assert fit["legs_visible_default"] is False
    assert fit["legs_toggle_enabled"] is True
    leg_note = fit["notes"][fit["note_keys"].index("legs")]
    assert "자리표시입니다(측정 아님)" in leg_note
    assert leg_note.endswith(service.LEGS_IK_AVAILABLE_SUFFIX)


def test_low_confidence_marks_the_right_forearm_when_the_grip_limit_is_exceeded(test_mounts) -> None:  # noqa: ANN001
    """아바타 손과 건이 어긋난 패스는 **오른 전완·손**을 저신뢰로 내고, 그 사실을 적합값이 아니라
    캡처 파이프라인의 문제로 말한다(S5). 3D 안에 걸 숫자 라벨(`offset_display`)은 더 이상 없다(S6)."""
    raw = RawPass(rows=_rows_with_skeleton(0.3))
    representative = service.representative_frame(raw)
    offsets = pose_fk_csv.bind_offsets(raw.rows[0])
    mounts = skeleton.mounts_for(_TEST_MOUNTS.table, representative.gun)
    sample = skeleton.frame_sample(representative.frame, representative.row, representative.gun, offsets)
    placed = skeleton.place_oriented(sample.local_pos, sample.local_rot, sample.measured, mounts)
    loose = service._fit_view(placed, representative, service.MOUNTS_GLOBAL, mounts, offsets, {representative.gun: 1.0})  # noqa: SLF001
    assert loose["low_confidence"] is False
    assert loose["low_confidence_bones"] == [] and loose["low_confidence_joints"] == []
    assert loose["low_confidence_gap_m"] is None
    assert "grip_limit" not in loose["note_keys"]

    tight = service._fit_view(placed, representative, service.MOUNTS_GLOBAL, mounts, offsets, {representative.gun: 0.05})  # noqa: SLF001
    norm = math.hypot(*_TEST_MOUNTS.table["right:한길숏건"].offset)
    assert tight["low_confidence"] is True
    assert tight["low_confidence_bones"] == [["RightForeArm", "RightHand"]]
    assert tight["low_confidence_joints"] == ["RightHand"]
    assert tight["low_confidence_gap_m"] == pytest.approx(norm)
    note = tight["notes"][tight["note_keys"].index("grip_limit")]
    assert "손 IK↔컨트롤러" in note and "쥐는 거리가 아니라" in note
    # 적합 상수를 "보정"이라고 부르지 않는다.
    assert "보정" not in note
    # 3D 안에 걸 숫자 라벨은 어느 앵커에도 없다.
    assert all("offset_display" not in anchor for anchor in tight["anchors"])


def test_mesh_reason_detail_is_read_from_the_alignment_record() -> None:
    """사유는 정렬 기록에서 읽는다 — 레벨 번호를 기대값에 박지 않는다.

    레벨 1은 2026-09-16에 총구 가림이 손대칭을 갈라 `solved`가 됐다(그전에는 여기서
    레벨 1의 "손대칭" 사유를 단언했고, 그래서 이 테스트가 깨졌다). 레벨 번호로 고정하는
    대신 **합성 기록으로 분기 자체를 고정한다** — 검증된 레벨은 사유 계산에 도달조차
    하지 않으므로, 실데이터만 보는 단언은 사유를 상수로 바꿔도 통과한다(실제로 확인했다).
    """
    detail = service._mesh_reason_detail  # noqa: SLF001
    one_left = {"fixed_linear_attempt": {"candidates_passed_A": 1}}
    none_left = {"fixed_linear_attempt": {"candidates_passed_A": 0}}
    ambiguous = {"fixed_linear_attempt": {"candidates_passed_A": 2}}
    assert "손대칭" in (detail(one_left) or "")
    assert "모델 UV" in (detail(none_left) or "")
    assert detail(one_left) != detail(none_left)
    assert detail(ambiguous) is None
    assert detail({}) is None

    two = service.mesh_block("2")
    assert two["reason"] == "unverified_alignment"
    assert two["reason_detail"] == detail(service._alignment_entry(2))  # noqa: SLF001
    solved = [str(level) for level in service._solved_levels()]  # noqa: SLF001
    assert solved and two["reason_hint"] == "벽면이 검증된 레벨: " + "·".join(solved)
    ok = service.mesh_block(solved[0])
    assert ok["reason_detail"] is None and ok["reason_hint"] is None


def test_view_fit_is_null_when_skeleton_is_unavailable() -> None:
    view = _compare()["view"]
    assert view["fit"] is None
    assert view["fit_reason"] == "no_joint_columns"
    assert view["fit_note"] == service.FIT_NOTES["no_joint_columns"]
    assert view["display_errors"] == []
    assert view["legend_notes"][0] == service.LEGEND_NOTES[0]
    assert "겹쳐 그립니다" in view["legend_notes"][0]


def test_mesh_view_carries_reason_text() -> None:
    unverified = service.mesh_block("2")
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


# --- D3 자리표시 다리 바닥 맞춤 ---------------------------------------------------


def _floor_geometry(level: float, half: float = 4.0) -> composite.MeshGeometry:
    """수평 상자 하나. **가장 낮은 수평면의 윗면**이 `level`이다 — `composite.floor_vertical`이 고르는 면.
    0.02 m 위에 두 번째 수평면을 두어, 바닥이 더 높은 면이 아니라 가장 낮은 면에서 나오는지 가른다."""
    up = world_axes.UP_INDEX
    side_a, side_b = world_axes.HORIZONTAL_INDICES

    def point(a: float, b: float, v: float) -> list[float]:
        p = [0.0, 0.0, 0.0]
        p[side_a], p[side_b], p[up] = a, b, v
        return p

    vertices: list[list[float]] = []
    triangles: list[list[int]] = []
    for v in (level, level + 0.02):
        base = len(vertices)
        vertices += [
            point(-half, -half, v),
            point(half, -half, v),
            point(half, half, v),
            point(-half, half, v),
        ]
        triangles += [[base, base + 1, base + 2], [base, base + 2, base + 3]]
    return composite.MeshGeometry.from_mesh(vertices, triangles)


def _head_vertical(fit: dict) -> float:
    """대표 프레임 머리(HMD) 실측 마커의 월드 수직 높이. 바닥면을 놓을 자리를 이 값에서 잡는다 —
    자리표시 다리가 사라졌으므로 발 높이는 더 이상 기준점이 될 수 없다."""
    head = next(a for a in fit["anchors"] if a["point"] == "head")
    return world_axes.vertical_of(tuple(head["measured_marker"]))


def test_placeholder_legs_keep_rig_length_and_report_the_gap_to_the_floor(test_mounts) -> None:  # noqa: ANN001
    """자리표시 다리는 **리그 길이를 지킨다** — 닿으면 무릎으로, 못 닿으면 뜬 채로 간격을 말한다.

    옛 코드는 발을 고른 면에 닿게 하려고 다리 길이에 배율을 먹였다: 줄이는 쪽은 레벨 3에서 0.35배
    (난쟁이), 늘리는 쪽은 레벨 1에서 1.25배(없는 다리). 사용자가 그 둘을 다 물렸다 — "리그 길이
    지키고 발이 뜨게". 진단용 접촉 시트는 이미 그 규칙을 고정하고 있었고
    (`test_pose_contact_sheet.test_placeholder_legs_keep_rig_length_and_report_the_gap_to_the_floor`),
    여기서는 **화면 경로도 같은 규칙을 쓰는지**를 본다. 이 검사의 바로 앞 판은 이 자리에서
    `placeholder_stretch_factor > 1.0`(늘려서 세웠다)을 단언했다 — 그 성질이 뒤집혔다.
    """
    rows, _middle = _rows_with_leg_poses(41, ())
    raw = RawPass(rows=rows)
    loose = service.skeleton_block(raw).fit
    assert loose["legs"]["floor_vertical"] is None
    assert loose["head_above_floor_m"] is None and loose["head_above_floor_display"] is None

    floor = _head_vertical(loose) - 1.20
    fit = service.skeleton_block(raw, geometry=_floor_geometry(floor)).fit
    assert fit["legs"]["floor_vertical"] == pytest.approx(floor)
    assert set(fit["legs"]) == {
        "plausible", "joints", "checks", "limits", "frame_offset", "searched_frames",
        "display_default", "placeholder_joints", "placeholder_bones", "placeholder_reaches_floor",
        "placeholder_feet_above_floor_m", "placeholder_floor_gaps",
        "hip_above_floor_m", "rig_leg_length_m", "floor_vertical", "head_above_floor_m",
    }
    assert fit["legs"]["display_default"] == "placeholder"

    def bone_lengths(block: dict) -> dict[str, float]:
        joints = block["legs"]["placeholder_joints"]
        return {
            f"{a}>{b}": math.dist(joints[a], joints[b])
            for a, b in block["legs"]["placeholder_bones"]
            if a in joints and b in joints
        }

    # 두 갈래를 **엉덩이 높이에서** 잡는다: 무릎이 접어 닿는 면, 엉덩이가 리그 다리보다 높은 면.
    leg_m = fit["legs"]["rig_leg_length_m"]
    hip_vertical = fit["legs"]["floor_vertical"] + fit["legs"]["hip_above_floor_m"]
    bend = service.skeleton_block(raw, geometry=_floor_geometry(hip_vertical - 0.5 * leg_m)).fit
    hang = service.skeleton_block(raw, geometry=_floor_geometry(hip_vertical - 1.2 * leg_m)).fit
    far = service.skeleton_block(raw, geometry=_floor_geometry(hip_vertical - 3.0 * leg_m)).fit

    # 1) 무릎이 접어 닿는 면: 뼈 길이는 리그 그대로이고, 뜬 발이 없으니 보조선도 없다.
    assert bend["legs"]["placeholder_reaches_floor"] is True
    assert bend["legs"]["placeholder_floor_gaps"] == []
    assert all(v == pytest.approx(0.0, abs=1e-9) for v in bend["legs"]["placeholder_feet_above_floor_m"].values())
    base = bone_lengths(bend)
    assert base

    # 2) 엉덩이가 리그 다리보다 높은 면: **늘리지 않는다** — 뼈 길이가 한 마디도 안 변하고,
    #    발마다의 간격과 그 간격을 그리는 보조선이 실린다.
    assert hang["legs"]["placeholder_reaches_floor"] is False
    for name, length in bone_lengths(hang).items():
        assert length == pytest.approx(base[name], abs=1e-9), name
    above = hang["legs"]["placeholder_feet_above_floor_m"]
    assert min(above.values()) > 0.0, above
    assert max(above.values()) == pytest.approx(hang["legs"]["hip_above_floor_m"] - leg_m, abs=1e-9)
    guides = hang["legs"]["placeholder_floor_gaps"]
    assert len(guides) == len(above), guides
    for guide in guides:
        assert math.dist(guide["from"], guide["to"]) == pytest.approx(guide["gap_m"], abs=1e-9), guide
        assert guide["label"] == f"간격 {guide['gap_m'] * 100.0:.0f}cm", guide
    # 고지가 그 간격을 말하고, 늘렸다는 말은 어디에도 없다.
    note = hang["notes"][hang["note_keys"].index("legs")]
    assert f"최대 {max(above.values()):.2f}m 떠 있고" in note, note
    assert "배 늘려" not in note, note

    # 3) 훨씬 낮은 면(옛 배율 한계 ×2.0을 넘겨야 닿던 자리)도 규칙이 같다 — 간격만 커진다.
    assert far["legs"]["placeholder_reaches_floor"] is False
    for name, length in bone_lengths(far).items():
        assert length == pytest.approx(base[name], abs=1e-9), name
    assert min(far["legs"]["placeholder_feet_above_floor_m"].values()) > max(above.values())
    # 바닥면을 받아도 실측 마커·대표 프레임은 그대로다 — 바닥은 자리표시와 문구만 바꾼다.
    assert fit["frame_index"] == loose["frame_index"]
    assert [a["measured_marker"] for a in fit["anchors"]] == [a["measured_marker"] for a in loose["anchors"]]


def test_real_pass_head_diameter_comes_from_this_rigs_torso_not_a_fixed_constant() -> None:
    """실데이터 L3: `fit.head_diameter_m`이 이 리그의 골반→머리 길이 × BVH 실측 비례고, 고정 상수가 아니다.

    옛 화면 상수(지름 0.18m)는 그려지는 골반→머리 0.409m의 44%였다 — 풍선으로 보였다(F2). 서버가
    낸 지름이 그 44%가 아니라 사람 범위(17~22%)에 있는지, `body_proportions.head_diameter_m`을
    독립적으로 다시 불러 같은 값인지 본다.
    """
    pose_fk_csv = importlib.import_module("pose_fk_csv")
    source, kwargs = isolation._source_for({"name": "real"})  # noqa: SLF001
    raw = source.load(PassRef(worker="김경호", date="20250831", level="3", session="1000391_1", pass_index=0))
    fit = service.skeleton_block(raw).fit
    offsets = body_proportions.rig_offsets(pose_fk_csv.bind_offsets(raw.rows[0]))
    expected = body_proportions.head_diameter_m(offsets)
    assert fit["head_diameter_m"] == pytest.approx(expected, rel=1e-12)
    torso = body_proportions.torso_length_m(offsets)
    ratio = fit["head_diameter_m"] / torso
    assert 0.17 < ratio < 0.22, (fit["head_diameter_m"], torso, ratio)
    old_ratio = 0.18 / torso
    assert not (0.17 < old_ratio < 0.22), old_ratio  # 옛 고정 상수는 이 범위 밖이었다


def test_leg_joint_names_come_from_the_server_not_from_the_screen(test_mounts) -> None:  # noqa: ANN001
    """화면이 "다리가 지금 보이는가"를 재려면 어느 관절이 다리인지 알아야 한다 — 그 **이름**은
    서버가 준다. 화면에서 리그 이름을 지어내면 출처와 다른 라벨이 된다."""
    rows, _middle = _rows_with_leg_poses(41, ())
    fit = service.skeleton_block(RawPass(rows=rows)).fit
    assert fit["legs"]["joints"] == list(skeleton.LEG_JOINTS)
    assert all(name in fit["legs"]["joints"] for name in ("LeftFoot", "RightToe"))


# --- 고른 바닥면 위에 무엇이 서 있는지 화면이 말하는가 -------------------------------------
#
# 검증이 반증한 주장이 있다: "배율이 레벨마다 다른 근본 원인은 리그 몸통이 짧은 것". 몸통 고지는 네
# 레벨이 같고(리그 값), 갈리는 것은 골반–선택된 바닥 거리뿐이다. 그 거리가 이상한 레벨에서는 고른 면이
# 작업자가 선 지면이 아니다 — 실측으로 이 면 위 머리 실측 마커가 L1 1.508m인데 L3 0.696m · L6 0.879m다.
# 문턱을 지어내지 않고 **실측 높이를 그대로** 적어 보는 사람이 판단하게 한다.


def _head_above(fit: dict, floor: float) -> float:
    head = next(a for a in fit["anchors"] if a["point"] == "head")
    return world_axes.vertical_of(tuple(head["measured_marker"])) - floor


def test_head_height_panel_says_how_high_the_measured_head_is_above_the_chosen_plane(test_mounts) -> None:  # noqa: ANN001
    """바닥 기준 머리 높이는 이제 다리 고지가 아니라 **히어로 패널 한 자리**(`view.head_height`)에
    산다(S4). 그 자리가 그 면 위 머리 실측 높이를 적고, 바닥을 옮기면 숫자가 따라 움직인다.
    같은 값이 구조체(`head_above_floor_m`)로도 나가 화면이 다시 계산하지 않는다."""
    rows, _middle = _rows_with_leg_poses(41, ())
    raw = RawPass(rows=rows)
    loose = service.skeleton_block(raw).fit
    high = _head_vertical(loose) - 1.20
    low = high - 0.20  # 더 낮은 면을 고르면 그 위 머리 높이는 그만큼 커진다

    said = []
    for floor in (high, low):
        fit = service.skeleton_block(raw, geometry=_floor_geometry(floor)).fit
        assert fit["legs"]["floor_vertical"] == pytest.approx(floor), floor
        assert fit["head_above_floor_m"] == pytest.approx(_head_above(fit, floor), abs=1e-9)
        assert fit["head_above_floor_display"] == f"{fit['head_above_floor_m']:.2f}m"
        block = service.head_height_block(fit, None)
        # 한 층만 있으면 그 층 이름만 적는다 — 없는 몸의 높이를 지어내지 않는다.
        assert block["display"] == f"{service.LEG_NOTE_WHO['query']} {fit['head_above_floor_display']}"
        assert block["note"] == service.HEAD_HEIGHT_NOTE.format(floor=floor)
        # 다리 고지에는 이제 높이 문장이 없다 — 같은 문장이 두 층으로 두 번 실리지 않게.
        leg_note = fit["notes"][fit["note_keys"].index("legs")]
        assert "머리 실측 마커" not in leg_note, leg_note
        said.append(float(fit["head_above_floor_display"].removesuffix("m")))
    # 바닥을 0.20m 내리면 적힌 높이도 0.20m 커진다 — 상수를 찍어 놓은 문구가 아니다.
    assert said[1] - said[0] == pytest.approx(0.20, abs=0.011), said


def test_head_height_panel_names_both_bodies(test_mounts) -> None:  # noqa: ANN001
    """대상·기준 두 몸의 높이를 나란히 적는다. 한 값만 보이면 어느 몸의 높이인지 알 수 없다 —
    실측에서 두 값이 다르다(L1 1.51m 대 1.66m · L3 0.70m 대 0.95m)."""
    rows, _middle = _rows_with_leg_poses(41, ())
    raw = RawPass(rows=rows)
    fit = service.skeleton_block(raw, geometry=_floor_geometry(_head_vertical(service.skeleton_block(raw).fit) - 1.20)).fit
    other = dict(fit)
    other["head_above_floor_m"] = 0.70
    other["head_above_floor_display"] = "0.70m"
    block = service.head_height_block(fit, other)
    assert block["display"] == f"대상 {fit['head_above_floor_display']} · 기준 0.70m"
    assert block["query_display"] == fit["head_above_floor_display"]
    assert block["reference_display"] == "0.70m"


def test_head_height_panel_does_not_claim_a_plane_it_never_chose(test_mounts) -> None:  # noqa: ANN001
    """벽 메시가 없는 레벨은 면을 고른 적이 없다 — 그 위 높이를 말하면 출처 없는 숫자가 된다.
    "—" 대신 왜 없는지 적는다."""
    rows, _middle = _rows_with_leg_poses(41, ())
    fit = service.skeleton_block(RawPass(rows=rows)).fit
    note = fit["notes"][fit["note_keys"].index("legs")]
    assert "머리 실측 마커" not in note
    block = service.head_height_block(fit, fit)
    assert block["display"] is None
    assert block["note"] == service.HEAD_HEIGHT_NO_FLOOR
    assert "0.00m" not in str(block)
