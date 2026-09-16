#!/usr/bin/env python3
# pyright: reportAny=false, reportUnusedCallResult=false
"""배치된 메시 위 기하 질의와 대표 프레임 합성 정렬(`composite.py`)."""

from __future__ import annotations

import importlib
import random
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

level_mesh = importlib.import_module("level_mesh")
world_axes = importlib.import_module("world_axes")

from trainer.pose_lab import composite, service  # noqa: E402
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


def test_floor_comment_numbers_match_the_code() -> None:
    """`composite.py`의 D3 바닥 주석이 적은 레벨별 바닥 높이가 코드가 내는 값과 같다."""
    source = (_SOURCE_DIR / "composite.py").read_text(encoding="utf-8")
    quoted = {level: float(value) for level, value in re.findall(r"L(\d+) (\d\.\d{3})m", source)}
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

    case = next(c for c in isolation._cases() if c["name"] == "real_L3_kkh_1000391_1_0")  # noqa: SLF001
    legs = isolation._run(case, include_view=True)["view"]["fit"]["legs"]  # noqa: SLF001
    up = np.asarray(world_axes.UP, dtype=float)
    scale = legs["placeholder_scale"]
    placeholder = legs["placeholder_joints"]
    # 리그 길이 그대로 내렸을 때(배율 1.0)의 발: 엉덩이는 그대로 두고 내린 길이를 배율로 되돌린다.
    rig_feet = []
    for hip, ankle in (("LeftThigh", "LeftFoot"), ("RightThigh", "RightFoot")):
        h = np.asarray(placeholder[hip], dtype=float)
        drop = h - np.asarray(placeholder[ankle], dtype=float)
        rig_feet.append(float(np.dot(h - drop / scale, up)))
    floor = legs["floor_vertical"]
    lowest = min(rig_feet)

    assert abs(floor - floor_said) <= FLOOR_COMMENT_TOL_M, (floor, floor_said)
    assert abs(lowest - foot_said) <= FLOOR_COMMENT_TOL_M, (lowest, foot_said)
    assert abs((floor - lowest) - gap_said) <= FLOOR_COMMENT_TOL_M, (floor - lowest, gap_said)
    assert leg_gap_said == gap_said, (leg_gap_said, gap_said)
