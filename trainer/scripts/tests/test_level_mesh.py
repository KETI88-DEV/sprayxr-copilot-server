#!/usr/bin/env python3
# pyright: reportAny=false, reportUnusedCallResult=false

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

level_mesh = importlib.import_module("level_mesh")
world_axes = importlib.import_module("world_axes")

LEVEL1_FBX = Path("/data/sources/modelings/learning_mode_lv.1_rev.1.1.fbx")


def test_reader_parses_a_known_level_exactly() -> None:
    """레벨 1은 정점 16개, UV 32쌍이다(스파이크 실측)."""
    mesh = level_mesh.read_fbx(LEVEL1_FBX)
    assert len(mesh.vertices) == 16
    assert len(mesh.uvs) == 32
    assert mesh.triangles


def test_declared_up_axis_agrees_with_the_world_convention() -> None:
    """FBX가 선언한 위 축이 world_axes의 규약과 같다. 세 번째 독립 확인이다."""
    mesh = level_mesh.read_fbx(LEVEL1_FBX)
    assert mesh.declared["UpAxis"] == world_axes.UP_INDEX


def test_closest_distance_is_to_the_triangle_not_the_nearest_vertex() -> None:
    """면 한가운데 바로 위의 점은 어느 정점보다도 훨씬 가깝다.

    정점 최근접으로 구현하면 이 값이 4.3 근처가 나온다(가장 가까운 정점까지의 거리).
    """
    mesh = level_mesh.Mesh(
        vertices=[(0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (0.0, 0.0, 10.0)],
        triangles=[(0, 1, 2)],
        uvs=[],
        declared={},
    )
    assert level_mesh.closest_distance(mesh, (3.0, 0.5, 3.0)) == pytest.approx(0.5)


def test_alignment_records_its_provenance() -> None:
    """선언된 변환과 풀어낸 변환은 다른 주장이다. 그 구분이 사라지면 안 된다."""
    a = level_mesh.load_alignment(1)
    assert a.provenance in {"solved", "declared"}
    assert a.provenance == "solved"  # 현재는 풀어낸 것뿐이다
    assert a.declared_scale != a.scale  # 선언값과 다르다는 것이 이 필드의 존재 이유다


def test_place_moves_vertices_by_the_alignment() -> None:
    """`place`는 회전+스케일+평행이동을 정점에 적용한다 — 삼각형·UV는 그대로 옮긴다."""
    mesh = level_mesh.Mesh(
        vertices=[(1.0, 0.0, 0.0)],
        triangles=[],
        uvs=[(0.1, 0.2)],
        declared={"marker": True},
    )
    identity_alignment = level_mesh.Alignment(
        level=1,
        provenance="solved",
        rotation=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        translation=(5.0, 6.0, 7.0),
        scale=2.0,
        declared_scale=1.0,
        fit={},
    )
    placed = level_mesh.place(mesh, identity_alignment)
    assert placed.vertices[0] == pytest.approx((7.0, 6.0, 7.0))
    assert placed.uvs == mesh.uvs
    assert placed.declared == mesh.declared
