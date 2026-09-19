#!/usr/bin/env python3
# pyright: reportAny=false, reportUnusedCallResult=false
"""배치된 메시 위 기하 질의와 대표 프레임 합성 정렬(`composite.py`)."""

from __future__ import annotations

import importlib
import math
import random
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

level_mesh = importlib.import_module("level_mesh")
world_axes = importlib.import_module("world_axes")

from trainer.pose_lab import body_proportions, composite, service, skeleton  # noqa: E402
from trainer.pose_lab.tests import test_scoring_isolation as isolation  # noqa: E402


def test_nearest_matches_the_scalar_reference_on_a_real_level_mesh() -> None:
    """벡터화 최근접 거리가 `level_mesh.closest_distance`(표준 라이브러리 구현)와 같다."""
    placed = service.placed_mesh("3")
    assert placed is not None and placed.mesh is not None
    geometry = composite.MeshGeometry.from_mesh(placed.mesh.vertices, placed.mesh.triangles)
    vertices = np.asarray(placed.mesh.vertices)
    low, high = vertices.min(axis=0) - 0.3, vertices.max(axis=0) + 0.3
    rng = random.Random(3)
    points = np.array([[rng.uniform(low[i], high[i]) for i in range(3)] for _ in range(40)])
    distance, _rows, closest = geometry.nearest(points)
    for point, got, near in zip(points, distance, closest):
        expected = level_mesh.closest_distance(placed.mesh, tuple(point))
        assert abs(got - expected) < 1e-9
        assert abs(np.linalg.norm(near - point) - got) < 1e-12


def _plane() -> composite.MeshGeometry:
    vertices = [(-5.0, -5.0, 2.0), (5.0, -5.0, 2.0), (5.0, 5.0, 2.0), (-5.0, 5.0, 2.0)]
    return composite.MeshGeometry.from_mesh(vertices, [(0, 1, 2), (0, 2, 3)])


def test_first_hit_on_a_plane_and_miss_when_pointing_away() -> None:
    geometry = _plane()
    hit = geometry.first_hit(np.array([0.3, 0.1, 0.0]), np.array([0.0, 0.0, 1.0]))
    assert hit is not None and np.allclose(hit, [0.3, 0.1, 2.0])
    assert geometry.first_hit(np.array([0.3, 0.1, 0.0]), np.array([0.0, 0.0, -1.0])) is None


def _representative(gun, hit, rotation=(90.0, 0.0, 0.0)):  # noqa: ANN001, ANN202
    row = {f"한길숏건 Rotation {axis}": str(v) for axis, v in zip("XYZ", rotation)}
    frame = SimpleNamespace(hit=hit, right=gun, head=gun, left=None, index=0, time_sec=0.0)
    return SimpleNamespace(frame=frame, row=row, gun="한길숏건", spraying=[])


def test_composite_block_reports_hit_distance_side_and_spray_ray() -> None:
    geometry = _plane()
    # YXZ Rotation X=90 → 전방 +Y가 +Z로 간다(R = Rx(90)).
    representative = _representative((0.2, 0.0, 1.5), (0.2, 0.0, 2.0))
    front = composite.composite_block(representative, geometry, {"Pelvis": (0.0, -0.9, 1.4)})
    assert front["hit_to_mesh_mm"] < 1e-9
    assert front["pelvis_front_of_surface"] is True
    assert front["spray_ray_hits_mesh"] is True
    assert front["spray_ray_hit_miss_mm"] < 1e-6
    behind = composite.composite_block(representative, geometry, {"Pelvis": (0.0, -0.9, 2.6)})
    assert behind["pelvis_front_of_surface"] is False


def test_composite_block_without_mesh_keeps_only_the_hit() -> None:
    block = composite.composite_block(_representative((0.2, 0.0, 1.5), (0.2, 0.0, 2.0)), None, None)
    assert block == {
        "hit": [0.2, 0.0, 2.0],
        "hit_to_mesh_mm": None,
        "pelvis_front_of_surface": None,
        "spray_ray_hits_mesh": None,
        "spray_ray_hit_miss_mm": None,
        "spray_ray_angle_deg": 0.0,
    }


def test_composite_alignment_L3() -> None:
    """실데이터 L3(김경호 1000391_1 #0): 대표 프레임 착탄점이 배치 메시 표면에 붙고, 건 분사축
    반직선이 메시에서 착탄점 근처(50 mm 안)에 닿는다. 데이터가 없으면 실패한다."""
    case = next(c for c in isolation._cases() if c["name"] == "real_L3_kkh_1000391_1_0")  # noqa: SLF001
    body = isolation._run(case, include_view=True)  # noqa: SLF001
    block = body["view"]["composite"]
    assert block["hit_to_mesh_mm"] < 0.01, block
    assert block["spray_ray_hits_mesh"] is True, block
    assert block["spray_ray_hit_miss_mm"] < 50.0, block


def test_spray_ray_angle_is_the_angle_between_spray_axis_and_gun_to_hit() -> None:
    """거리와 무관한 척도. 분사축 +Z(Rotation X=90)에서 착탄점을 옆으로 0.1 m 옮기면 atan(0.1/0.5)."""
    geometry = _plane()
    block = composite.composite_block(_representative((0.2, 0.0, 1.5), (0.3, 0.0, 2.0)), geometry, None)
    assert abs(block["spray_ray_angle_deg"] - np.degrees(np.arctan2(0.1, 0.5))) < 1e-9
    assert abs(block["spray_ray_hit_miss_mm"] - 100.0) < 1e-6


def test_surface_hits_are_raw_hits_on_the_real_L6_mesh() -> None:
    """실데이터 L6(김경호 1000394_1 #0): `view.surface_hits`는 원본 착탄점이라 배치 메시 표면 0.01 mm 안이다.
    리샘플 격자(`layers.query.hit`)는 모서리를 가로질러 벽에서 뜬다(최대 약 63 mm) — 둘을 구분한다."""
    source, kwargs = isolation._source_for({"name": "real"})  # noqa: SLF001
    body = service.compare(
        source,
        service.PassRef(worker="김경호", date="20250831", level="6", session="1000394_1", pass_index=0),
        service.PassRef(worker="전형기", date="20250904", level="6", session="1000569_1", pass_index=0),
        **kwargs,
    )
    assert body["status"] == "ok"
    view = body["view"]
    placed = service.placed_mesh("6")
    geometry = composite.MeshGeometry.from_mesh(placed.mesh.vertices, placed.mesh.triangles)
    hits = np.asarray(view["surface_hits"], dtype=float)
    assert len(hits) >= 20
    distance, _rows, _closest = geometry.nearest(hits)
    assert float(distance.max()) * 1000.0 < 0.01, float(distance.max())
    grid = np.asarray([p for p in view["layers"]["query"]["hit"] if p is not None], dtype=float)
    grid_distance, _r, _c = geometry.nearest(grid)
    assert float(grid_distance.max()) * 1000.0 > 1.0  # 격자점은 표면 점이 아니다(이 테스트가 무엇을 가르는지)



# --- D3 바닥면 ------------------------------------------------------------------


def _slab(bottom: float, top: float, half: float = 1.0) -> composite.MeshGeometry:
    """수평 윗면·아랫면과 수직 옆면 네 장을 가진 상자. 수직축은 `world_axes.UP_INDEX`다."""
    up = world_axes.UP_INDEX
    side_a, side_b = world_axes.HORIZONTAL_INDICES

    def point(a: float, b: float, v: float) -> list[float]:
        p = [0.0, 0.0, 0.0]
        p[side_a], p[side_b], p[up] = a, b, v
        return p

    vertices: list[list[float]] = []
    triangles: list[list[int]] = []
    for level in (bottom, top):
        base = len(vertices)
        vertices += [
            point(-half, -half, level),
            point(half, -half, level),
            point(half, half, level),
            point(-half, half, level),
        ]
        triangles += [[base, base + 1, base + 2], [base, base + 2, base + 3]]
    # 옆면(수직) 한 장 — 수평면만 고르는지 가른다.
    base = len(vertices)
    vertices += [point(-half, -half, bottom), point(half, -half, bottom), point(half, -half, top)]
    triangles += [[base, base + 1, base + 2]]
    return composite.MeshGeometry.from_mesh(vertices, triangles)


def test_floor_vertical_is_the_top_of_the_lowest_near_horizontal_face() -> None:
    """바닥면 = 수평에 가까운 면 중 **가장 낮은** 면의 최고점. 위 선반이나 옆면을 고르면 안 된다."""
    geometry = _slab(3.5, 5.5)
    assert composite.floor_vertical(geometry) == 3.5
    # 수평면이 하나도 없으면(옆면만) 바닥을 지어내지 않는다.
    up = world_axes.UP_INDEX
    side_a, side_b = world_axes.HORIZONTAL_INDICES

    def point(a: float, b: float, v: float) -> list[float]:
        p = [0.0, 0.0, 0.0]
        p[side_a], p[side_b], p[up] = a, b, v
        return p

    wall = composite.MeshGeometry.from_mesh(
        [point(-1.0, 0.0, 0.0), point(1.0, 0.0, 0.0), point(1.0, 0.0, 2.0)], [[0, 1, 2]]
    )
    assert composite.floor_vertical(wall) is None
    assert composite.floor_vertical(None) is None


def test_floor_vertical_follows_up_index_not_a_hardcoded_axis(monkeypatch) -> None:  # noqa: ANN001
    """수직축은 `world_axes.UP_INDEX` 하나에서 온다 — 축을 바꾸면 바닥값이 따라 움직인다."""
    before = composite.floor_vertical(_slab(3.5, 5.5))
    monkeypatch.setattr(world_axes, "UP_INDEX", 2)
    after = composite.floor_vertical(_slab(3.5, 5.5))
    assert before == 3.5 and after == 3.5


def test_floor_vertical_on_the_three_verified_levels() -> None:
    """실데이터: 검증된 세 레벨의 바닥면이 거의 같은 높이다(시험편이 같은 바닥에 서 있다)."""
    floors = {}
    for level in ("1", "3", "6"):
        placed = service.placed_mesh(level)
        geometry = composite.MeshGeometry.from_mesh(placed.mesh.vertices, placed.mesh.triangles)
        floors[level] = composite.floor_vertical(geometry)
    assert all(value is not None for value in floors.values()), floors
    assert max(floors.values()) - min(floors.values()) < 0.1, floors


# --- 주석에 적힌 숫자가 코드가 내는 값과 같은가(출처와 다른 라벨 막기) --------------------------
#
# 이 저장소의 반복 결함은 "출처와 다른 라벨이 붙은 숫자"다. D3 바닥 규칙을 설명하는 주석은 레벨마다의
# 바닥 높이와 레벨 3의 관통 깊이를 숫자로 적는데, 그 숫자가 코드가 실제로 내는 값과 어긋난 적이 있다
# (주석 "L3 3.552m" vs 실제 3.5312m). 그래서 **주석에서 숫자를 뽑아** 다시 재서 비교한다.

_SOURCE_DIR = Path(__file__).resolve().parents[1]
FLOOR_COMMENT_TOL_M = 0.001  # 주석은 mm 단위(소수 셋째 자리)로 적는다


def comment_numbers(text: str, pattern: str) -> list[float]:
    return [float(value) for value in re.findall(pattern, text)]


def comment_level_row(source: str, label: str) -> dict[str, float]:
    """`#   <label>: L1 3.557m · L3 3.531m · L6 3.560m` 한 줄을 레벨→값으로 읽는다."""
    line = next(ln for ln in source.splitlines() if ln.lstrip("# ").startswith(f"{label}:"))
    return {level: float(value) for level, value in re.findall(r"L(\d+) (\d\.\d{3})m", line)}


def test_floor_comment_numbers_match_the_code() -> None:
    """`composite.py`의 D3 바닥 주석이 적은 레벨별 바닥 높이가 코드가 내는 값과 같다."""
    source = (_SOURCE_DIR / "composite.py").read_text(encoding="utf-8")
    quoted = comment_level_row(source, "바닥면 수직")
    assert set(quoted) == {"1", "3", "6"}, quoted
    for level, said in quoted.items():
        placed = service.placed_mesh(level)
        geometry = composite.MeshGeometry.from_mesh(placed.mesh.vertices, placed.mesh.triangles)
        measured = composite.floor_vertical(geometry)
        assert measured is not None
        assert abs(measured - said) <= FLOOR_COMMENT_TOL_M, (level, said, measured)


def test_floor_comment_can_fail_when_a_number_drifts() -> None:
    """주석 숫자를 21mm 틀리게 써 놓으면 위 검사가 실제로 잡는다(실패할 수 없는 단언 방지)."""
    placed = service.placed_mesh("3")
    geometry = composite.MeshGeometry.from_mesh(placed.mesh.vertices, placed.mesh.triangles)
    real = composite.floor_vertical(geometry)
    drifted = 3.552  # 옛 주석이 적었던 값
    assert abs(real - drifted) > FLOOR_COMMENT_TOL_M


def _standing_legs_for_level3() -> tuple[object, float]:
    """레벨 3 실데이터 한 패스의 자리표시 다리와 그 레벨이 고른 바닥면 높이.

    `service._fit_view`가 예전에 하던 것과 **같은 호출**이다(관절·리그 오프셋·착탄 방향·바닥면).
    화면은 더 이상 이 다리를 그리지 않지만 `skeleton.standing_legs`의 주석이 아직 숫자를
    주장하므로, 그 숫자를 여기서 다시 잰다.
    """
    pose_fk_csv = importlib.import_module("pose_fk_csv")
    source, _kwargs = isolation._source_for({"name": "real"})  # noqa: SLF001
    ref = service.PassRef(worker="김경호", date="20250831", level="3", session="1000391_1", pass_index=0)
    raw = source.load(ref)
    placed = service.placed_mesh("3")
    geometry = composite.MeshGeometry.from_mesh(placed.mesh.vertices, placed.mesh.triangles)
    floor = composite.floor_vertical(geometry)
    block = service.skeleton_block(raw, geometry=geometry)
    offsets = body_proportions.rig_offsets(pose_fk_csv.bind_offsets(raw.rows[0]))
    hit = block.representative.frame.hit
    pelvis = block.joints["Pelvis"]
    forward = [hit[i] - pelvis[i] for i in range(3)]
    legs = skeleton.standing_legs(block.joints, offsets, forward, floor=floor)
    assert legs is not None
    return legs, floor


def test_level3_penetration_numbers_in_the_comments_match_the_code() -> None:
    """주석이 적은 "레벨 3에서 발이 바닥면보다 0.540m 아래" 세 숫자를 실제 배치로 다시 잰다.

    비교 대상은 `composite.py`(0.540m·2.991m·3.531m)와 `skeleton.py`(0.540m)의 D3 설명문이다.
    """
    source = (_SOURCE_DIR / "composite.py").read_text(encoding="utf-8")
    gap_said = comment_numbers(source, r"바닥면보다\n# (\d\.\d{3})m 아래")[0]
    foot_said = comment_numbers(source, r"자리표시 발 수직 (\d\.\d{3})m")[0]
    floor_said = comment_numbers(source, r"바닥면 (\d\.\d{3})m")[0]
    leg_source = (_SOURCE_DIR / "skeleton.py").read_text(encoding="utf-8")
    leg_gap_said = comment_numbers(leg_source, r"발이 바닥면 (\d\.\d{3})m 아래")[0]

    # 자리표시 다리는 3D 화면에서 사라졌지만(S4) `skeleton.standing_legs`는 진단용 접촉 시트가
    # 그대로 쓴다. 그래서 그 함수를 **직접** 불러 잰다 — 주석이 숫자를 계속 주장하는 한 검사도 남는다.
    placeholder, floor = _standing_legs_for_level3()
    up = np.asarray(world_axes.UP, dtype=float)
    # 다리는 리그 제 길이로 내린다(배율 없음).
    rig_feet = [
        float(np.dot(np.asarray(placeholder.joints[ankle], dtype=float), up))
        for ankle in ("LeftFoot", "RightFoot")
    ]
    lowest = min(rig_feet)
    # 함수가 낸 "발이 바닥면보다 얼마나 아래인가"도 같은 값이어야 한다(라벨과 출처가 같은지).
    assert abs(placeholder.feet_below_floor_m - (floor - lowest)) < 1e-9, placeholder.feet_below_floor_m

    assert abs(floor - floor_said) <= FLOOR_COMMENT_TOL_M, (floor, floor_said)
    assert abs(lowest - foot_said) <= FLOOR_COMMENT_TOL_M, (lowest, foot_said)
    assert abs((floor - lowest) - gap_said) <= FLOOR_COMMENT_TOL_M, (floor - lowest, gap_said)
    assert leg_gap_said == gap_said, (leg_gap_said, gap_said)


# --- 고른 바닥면 위에 실제로 무엇이 서 있는가(D3 원인 주장 검증) --------------------------------
#
# 보고서가 한때 "배율이 레벨마다 다른 근본 원인은 리그 **몸통**이 짧은 것"이라고 적었다. 화면이 그 주장을
# 반증한다: 골반→머리 고지는 네 레벨이 글자 하나까지 같은데(리그 값) 자리표시 배율은 갈린다. 실제로 갈리는
# 것은 **골반–선택된 바닥 거리**다. 그래서 `composite.py`의 바닥 주석이 적은 세 줄(바닥면 수직 · 이 면 위
# 머리 실측 마커 · 이 면 위 골반)을 실제 배치로 다시 재고, 몸통 길이가 세 레벨에서 **같다**는 것도 같이
# 잰다 — 같으면 몸통은 배율 산포를 설명할 수 없다.

FLOOR_STANDING_CASES = {
    "1": {"worker": "김경호", "date": "20250831", "level": "1", "session": "1000389_1", "pass_index": 0},
    "3": {"worker": "김경호", "date": "20250831", "level": "3", "session": "1000391_1", "pass_index": 0},
    "6": {"worker": "김경호", "date": "20250831", "level": "6", "session": "1000394_1", "pass_index": 0},
}


def _floor_standing_view(level: str) -> dict:
    source, kwargs = isolation._source_for({"name": f"real_L{level}_floor"})  # noqa: SLF001
    query = service.PassRef(**FLOOR_STANDING_CASES[level])
    candidates = service.reference_candidates(source, query)
    assert candidates["status"] == "ok", (level, candidates)
    first = candidates["candidates"][0]
    reference = service.PassRef(
        worker=first["worker"],
        date=first["date"],
        level=first["level"],
        session=first["session"],
        pass_index=first["pass_index"],
    )
    body = service.compare(source, query, reference, include_view=True, **kwargs)
    assert body["status"] == "ok", (level, body.get("reason"))
    return body["view"]


def test_floor_choice_comment_reports_what_stands_on_it() -> None:
    """바닥 주석의 세 줄(바닥면 · 그 위 머리 실측 · 그 위 골반)을 실제 배치로 다시 잰다."""
    source = (_SOURCE_DIR / "composite.py").read_text(encoding="utf-8")
    floors = comment_level_row(source, "바닥면 수직")
    heads = comment_level_row(source, "이 면 위 머리 실측 마커(HMD)")
    pelvises = comment_level_row(source, "이 면 위 골반")
    assert set(floors) == set(heads) == set(pelvises) == {"1", "3", "6"}
    up = np.asarray(world_axes.UP, dtype=float)
    torso_notes = set()
    for level in ("1", "3", "6"):
        view = _floor_standing_view(level)
        floor = view["fit"]["legs"]["floor_vertical"]
        assert floor is not None, level
        pelvis = float(np.asarray(view["layers"]["query"]["joints"]["Pelvis"], dtype=float) @ up)
        head = next(
            float(np.asarray(a["measured_marker"], dtype=float) @ up)
            for a in view["fit"]["anchors"]
            if a["point"] == "head"
        )
        assert abs(floor - floors[level]) <= FLOOR_COMMENT_TOL_M, (level, floor, floors[level])
        assert abs((head - floor) - heads[level]) <= FLOOR_COMMENT_TOL_M, (level, head - floor, heads[level])
        assert abs((pelvis - floor) - pelvises[level]) <= FLOOR_COMMENT_TOL_M, (
            level,
            pelvis - floor,
            pelvises[level],
        )
        torso_notes.add(next(n for n in view["fit"]["notes"] if "골반→머리" in n))
    # 몸통 고지가 세 레벨에서 **글자 하나까지 같다** — 몸통 길이는 세 레벨에서 같으므로 "발이 면보다
    # 얼마나 아래인가"의 산포를 설명할 수 없다.
    assert len(torso_notes) == 1, torso_notes

    # --- 숫자에 붙은 **라벨**도 검사한다 ------------------------------------------------
    # 위 세 줄은 재현되는 값이지만, 그것을 "레벨의 성질"로 읽으면 "바닥이 틀렸다"는 결론이 나온다 —
    # 실제로 그 오독이 자리표시 다리를 없앴다. 그래서 주석이 스스로 "한 패스"라고 말하는지, 그리고
    # 레벨 중앙값을 같이 적어 두 라벨이 **다른 양**임을 보이는지까지 본다.
    assert "한 패스의 대표 프레임 한 장" in source
    median_line = next(ln for ln in source.splitlines() if "레벨 중앙값" in ln)
    medians = {level: float(value) for level, value in re.findall(r"L(\d+) (\d\.\d{3})m", median_line)}
    assert set(medians) == {"1", "3", "6"}, medians
    # 중앙값이 이 패스 값과 **다르다**. 같다면 "한 패스"라는 라벨을 붙일 이유도 없다.
    for level in ("1", "3", "6"):
        assert abs(medians[level] - heads[level]) > FLOOR_COMMENT_TOL_M, (level, medians[level], heads[level])


def test_floor_choice_comment_can_fail_when_a_number_drifts() -> None:
    """주석의 머리 높이를 L3에서 L1 값으로 바꿔 적으면 위 검사가 실제로 잡는다."""
    view = _floor_standing_view("3")
    up = np.asarray(world_axes.UP, dtype=float)
    floor = view["fit"]["legs"]["floor_vertical"]
    head = next(
        float(np.asarray(a["measured_marker"], dtype=float) @ up)
        for a in view["fit"]["anchors"]
        if a["point"] == "head"
    )
    source = (_SOURCE_DIR / "composite.py").read_text(encoding="utf-8")
    said_l1 = comment_level_row(source, "이 면 위 머리 실측 마커(HMD)")["1"]
    assert abs((head - floor) - said_l1) > FLOOR_COMMENT_TOL_M, (head - floor, said_l1)


# --- 벽 위 분사 스트로크: 모서리를 가로지르는 현에서 끊는다(U1) --------------------------------
#
# 사용자가 "벽이랑 시퀀스 접촉점이랑 정렬이 안됨"이라고 했고, 씬 그래프에서 직접 재 보니 **점은 전부
# 벽에 붙어 있었다**(L3 38점·L6 77점 모두 0.00mm). 뜨는 것은 점을 잇는 **현**이다. 아래 검사는 그
# 분기를 합성 기하로 직접 고정한다 — 실데이터만 보면 문턱을 아무 값으로 바꿔도 통과할 수 있다.


def _corner(size: float = 1.0) -> composite.MeshGeometry:
    """z=0 평면과 x=0 평면이 90°로 만나는 기하(두 사각형). 모서리는 x=0, z=0 직선."""
    vertices = [
        (0.0, -size, 0.0), (size, -size, 0.0), (size, size, 0.0), (0.0, size, 0.0),   # z=0 바닥
        (0.0, -size, size), (0.0, size, size),                                        # x=0 벽 위쪽
    ]
    triangles = [(0, 1, 2), (0, 2, 3), (0, 3, 5), (0, 5, 4)]
    return composite.MeshGeometry.from_mesh(vertices, triangles)


def test_chord_lift_is_zero_along_a_face_and_large_across_a_corner() -> None:
    """같은 면 위의 두 점을 이은 현은 벽에 붙고, 모서리를 가로지르는 현은 뜬다."""
    geometry = _corner()
    flat = composite.chord_lift_mm(geometry, (0.2, 0.0, 0.0), (0.8, 0.0, 0.0))
    across = composite.chord_lift_mm(geometry, (0.5, 0.0, 0.0), (0.0, 0.0, 0.5))
    assert flat < 1e-6, flat
    # 두 점을 잇는 현의 중점 (0.25, 0, 0.25)에서 가장 가까운 면 위 점은 (0.25, 0, 0)·(0, 0, 0.25)이라 0.25m다.
    assert across == pytest.approx(250.0, rel=1e-3), across


def test_stroke_splits_only_where_the_chord_leaves_the_wall() -> None:
    """모서리를 가로지르는 한 곳에서만 끊고, 나머지는 한 줄로 잇는다."""
    geometry = _corner()
    hits = [
        (0.9, 0.0, 0.0), (0.7, 0.0, 0.0), (0.5, 0.0, 0.0),   # 바닥 위 세 점
        (0.0, 0.0, 0.5), (0.0, 0.0, 0.7), (0.0, 0.0, 0.9),   # 벽 위 세 점
    ]
    block = composite.stroke_runs(geometry, hits)
    assert block["break_count"] == 1, block
    assert [len(run) for run in block["runs"]] == [3, 3], block["runs"]
    # 점을 옮기거나 만들지 않는다 — 구간을 이어 붙이면 입력 그대로다.
    assert [tuple(p) for run in block["runs"] for p in run] == hits
    assert block["point_count"] == 6
    assert block["max_lift_mm"] > composite.STROKE_MAX_LIFT_MM
    assert block["max_kept_lift_mm"] <= composite.STROKE_MAX_LIFT_MM


def test_stroke_keeps_one_run_when_every_chord_stays_on_the_wall() -> None:
    geometry = _corner()
    hits = [(0.9, 0.0, 0.0), (0.7, 0.0, 0.0), (0.5, 0.0, 0.0), (0.3, 0.0, 0.0)]
    block = composite.stroke_runs(geometry, hits)
    assert block["break_count"] == 0 and len(block["runs"]) == 1, block
    assert block["note"] == composite.STROKE_NOTE_CLEAN


def test_stroke_split_can_fail_when_the_tolerance_is_raised(monkeypatch) -> None:  # noqa: ANN001
    """문턱을 이탈보다 크게 올리면 **끊지 않는다** — 위 검사가 문턱 분기를 실제로 타는지 확인한다."""
    geometry = _corner()
    hits = [(0.5, 0.0, 0.0), (0.0, 0.0, 0.5), (0.0, 0.0, 0.9)]
    assert composite.stroke_runs(geometry, hits)["break_count"] == 1
    monkeypatch.setattr(composite, "STROKE_MAX_LIFT_MM", 1000.0)
    assert composite.stroke_runs(geometry, hits)["break_count"] == 0


def test_stroke_without_a_mesh_says_it_did_not_measure() -> None:
    hits = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
    block = composite.stroke_runs(None, hits)
    assert block["runs"] == [[list(p) for p in hits]]
    assert block["break_count"] == 0 and block["max_lift_mm"] is None
    assert block["note"] == composite.STROKE_NOTE_NO_MESH


def test_stroke_on_the_real_L6_mesh_breaks_at_the_two_corner_chords() -> None:
    """실데이터 L6(김경호 1000394_1 #0): 76세그먼트 중 모서리 둘에서만 끊고, 남은 현은 문턱 안이다."""
    view = _floor_standing_view("6")
    block = view["stroke"]
    assert block["point_count"] == len(view["surface_hits"])
    assert block["break_count"] == 2, block["break_count"]
    assert block["max_lift_mm"] > 80.0, block["max_lift_mm"]
    assert block["max_kept_lift_mm"] <= composite.STROKE_MAX_LIFT_MM
    # 이어 붙이면 원본 착탄점 순서 그대로다(옮기지도 만들지도 않았다).
    joined = [p for run in block["runs"] for p in run]
    assert joined == view["surface_hits"]


def test_stroke_on_the_real_L3_mesh_stays_one_run() -> None:
    """실데이터 L3(김경호 1000391_1 #0): 37세그먼트 최악 이탈이 1mm 미만이라 끊을 곳이 없다."""
    case = next(c for c in isolation._cases() if c["name"] == "real_L3_kkh_1000391_1_0")  # noqa: SLF001
    block = isolation._run(case, include_view=True)["view"]["stroke"]  # noqa: SLF001
    assert block["break_count"] == 0, block
    assert block["max_lift_mm"] < 1.0, block["max_lift_mm"]
