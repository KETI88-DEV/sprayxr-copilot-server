#!/usr/bin/env python3
# pyright: reportAny=false, reportUnusedCallResult=false
"""배치된 메시 위 기하 질의와 대표 프레임 합성 정렬(`composite.py`)."""

from __future__ import annotations

import importlib
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

level_mesh = importlib.import_module("level_mesh")

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

