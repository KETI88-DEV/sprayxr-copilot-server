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
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

measured_score = importlib.import_module("measured_score")
level_mesh = importlib.import_module("level_mesh")

simulator_csv = importlib.import_module("simulator_csv")

from trainer.pose_lab import service
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


def test_mesh_is_unavailable_for_a_level_with_no_verified_alignment() -> None:
    """검증된 정렬이 없는 레벨(질의 픽스처는 레벨 10)에서 `available`이 `False`이고
    기하가 비어 있다(요구사항 6)."""
    view = _compare()["view"]
    mesh = view["mesh"]
    assert mesh is not None
    assert mesh["available"] is False
    assert mesh["vertices"] == []
    assert mesh["triangles"] == []


def test_mesh_is_available_for_level_1_with_real_vertices() -> None:
    """레벨 1에서 `available`이 `True`이고 정점이 있다(요구사항 6)."""
    block = service.mesh_block("1")
    assert block is not None
    assert block["available"] is True
    assert block["vertices"], "레벨 1 메시는 정점을 가져야 한다"
    assert block["triangles"], "레벨 1 메시는 삼각형을 가져야 한다"
    alignment = level_mesh.load_alignment(1)
    assert alignment.provenance == "solved"


def test_mesh_is_unavailable_for_a_declared_but_unverified_alignment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`provenance`가 "solved"가 아니면(예: "declared") 검증되지 않은 값이다 —
    `capture_frame.to_world`가 이미 이 규칙을 지킨다
    (`test_frame_invariance.py::test_to_world_refuses_a_declared_but_unverified_alignment`
    와 같은 패턴). `mesh_block`이 이 검사를 건너뛰면 검증되지 않은 기하를 화면에
    올리게 된다."""
    import dataclasses

    real = level_mesh.load_alignment(1)
    unverified = dataclasses.replace(real, provenance="declared")
    monkeypatch.setattr(level_mesh, "load_alignment", lambda level: unverified)
    block = service.mesh_block("1")
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
    """정렬 레코드는 있고(레벨 1, provenance="solved") `source_fbx` 경로도 있지만
    그 경로에 실제 파일이 없으면(`read_fbx`가 `FileNotFoundError`를 던진다)
    `mesh_block`이 그 예외를 삼키지 않고 전파하면 `/api/compare` 전체가 500으로
    죽는다(태스크 0). 장식용 벽면 하나 때문에 점수·교정까지 죽어서는 안 된다 —
    `available: False`로 조용히 떨어져야 한다."""

    def _missing(path: Path) -> object:
        raise FileNotFoundError(path)

    monkeypatch.setattr(level_mesh, "read_fbx", _missing)
    block = service.mesh_block("1")
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
        service.mesh_block("1")


# ---- 11-A: 스켈레톤 배선 ----------------------------------------------------


class SkeletonFakeSource(FakeSource):
    """질의 원본 행에 FK 관절 컬럼을 얹는다. 레퍼런스는 그대로(`FakeSource`와 동일) —
    이 화면은 질의 층에만 스켈레톤을 얹는다(`view_block` 독스트링)."""

    def load(self, ref: PassRef):  # noqa: ANN201
        raw = super().load(ref)
        if ref.worker == QUERY["worker"]:
            return RawPass(rows=_rows_with_skeleton(0.3))
        return raw


def test_skeleton_block_returns_joints_bones_and_fit_when_fk_columns_are_present() -> None:
    block = service.skeleton_block(RawPass(rows=_rows_with_skeleton(0.3)))
    assert block is not None
    assert set(simulator_csv.JOINT_KEYS.values()) <= set(block["joints"])
    assert block["bones"] == [list(pair) for pair in simulator_csv.BONES]
    assert block["fit"]["source"] == "fk_csv"
    assert isinstance(block["fit"]["residual_m"], float)


def test_skeleton_block_is_none_without_joint_columns() -> None:
    """`FakeSource`의 원본 행에는 관절 컬럼이 없다 — FK를 못 얻으면 조용히 None이다."""
    assert service.skeleton_block(FakeSource().load(PassRef(**QUERY))) is None


def test_skeleton_anchors_snap_to_the_measured_position_not_the_fk_output() -> None:
    """FK가 낸 머리·양손이 실측과 어긋나도(리그 오프셋을 3배로 부풀림) 화면에 실릴
    머리·오른손·왼손은 실측 위치 그대로다(브리프: "실측 점은 실측 위치에 그대로")."""
    rows = _rows_with_skeleton(0.3)
    for row in rows:
        for axis in "XYZ":
            for column in simulator_csv.JOINT_COLUMNS:
                key = f"{column} Position {axis}"
                if key in row:
                    row[key] = str(float(row[key]) * 3.0)  # 뼈 길이를 왜곡해 어긋남을 만든다
    raw = RawPass(rows=rows)
    block = service.skeleton_block(raw)
    assert block is not None
    assert block["fit"]["residual_m"] > 1e-6, "왜곡한 뼈 길이는 실제로 어긋나야 한다"

    paired = service.measured_pose.measured_frames_with_rows(raw.rows)
    spraying = [f for f, _ in paired if f.spraying and f.hit is not None]
    frame = spraying[len(spraying) // 2]
    assert list(frame.head) == block["joints"]["Head"]
    assert list(frame.right) == block["joints"]["RightHand"]
    assert list(frame.left) == block["joints"]["LeftHand"]


def test_compare_wires_skeleton_into_the_query_layer_only() -> None:
    result = service.compare(SkeletonFakeSource(), PassRef(**QUERY), PassRef(**REFERENCE))
    assert result["status"] == "ok"
    view = result["view"]
    assert "joints" in view["layers"]["query"]
    assert "bones" in view["layers"]["query"]
    assert "joints" not in view["layers"]["reference"], "레퍼런스 층에는 스켈레톤을 얹지 않는다"
    assert "joints" not in view["layers"]["corrected"], "교정 후 층에는 스켈레톤을 얹지 않는다"
    assert view["fit"] is not None
    assert view["fit"]["source"] == "fk_csv"


def test_new_joints_are_added_to_weights_at_zero_and_covered_by_the_existing_invariant() -> None:
    """새 관절은 전부 weights에 0.0으로 들어간다(브리프) — 채점에는 안 들어가지만
    "weights가 그리는 모든 점을 덮는다"는 일반 불변량은 여전히 성립해야 한다."""
    result = service.compare(SkeletonFakeSource(), PassRef(**QUERY), PassRef(**REFERENCE))
    view = result["view"]
    joints = view["layers"]["query"]["joints"]
    assert joints, "이 픽스처는 스켈레톤이 실려야 한다"
    for name in joints:
        assert view["weights"].get(name) == 0.0, name
    # 일반 불변량: 질의 층이 그리는 최상위 점 이름(궤적) 전부 여전히 weights에 있다.
    drawn_points = {name for name in view["layers"]["query"] if name not in ("hit", "joints", "bones")}
    assert drawn_points <= set(view["weights"])


def test_view_fit_is_null_when_skeleton_is_unavailable() -> None:
    view = _compare()["view"]
    assert view["fit"] is None


def test_deviation_cm_is_the_local_delta_norm_in_centimetres() -> None:
    """②(얼마나 다른가)의 부위별 편차 — `local_delta`의 노름을 cm로 낸다(화면이
    다시 계산하지 않는다는 이 저장소의 규칙, 브리프 ② 각주)."""
    result = _compare()
    for delta in result["measured"]["deltas"]:
        expected = math.dist(delta["delta"], (0.0, 0.0, 0.0)) * 100.0
        assert math.isclose(delta["deviation_cm"], expected, rel_tol=1e-9)
