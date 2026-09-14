#!/usr/bin/env python3
# pyright: reportAny=false, reportUnusedCallResult=false

from __future__ import annotations

import collections
import dataclasses
import functools
import importlib
import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

level_mesh = importlib.import_module("level_mesh")
world_axes = importlib.import_module("world_axes")
rotation_math = importlib.import_module("rotation_math")

LEVEL1_FBX = Path("/data/sources/modelings/learning_mode_lv.1_rev.1.1.fbx")
CSV_ROOT = Path("/data/sources/professionals")


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
    """선언된 변환·풀어낸 변환·검증 불가는 서로 다른 주장이다. 그 구분이 사라지면 안 된다.

    레벨 1은 이제 "solved"가 아니다 — 착탄점이 정확히 평면이라 12자유도가 데이터로
    결정되지 않는다(G7: 조건수 2.06e-09). 커밋 325ae9a의 레벨 1 항목은 "통과 중인
    유일한 레벨"이 아니라 "틀린 채 기록된 유일한 레벨"이었다(절대 UV 잔차 수백 텍셀,
    이 파일의 회귀 대조 테스트가 그것을 고정한다).
    """
    assert level_mesh.load_alignment(1).provenance == "unverifiable"
    solved = level_mesh.load_alignment(SOLVED_LEVELS[0])
    assert solved.provenance == "solved"
    assert solved.declared_scale != solved.scale


def test_place_moves_vertices_by_the_alignment() -> None:
    """`place`는 회전+스케일+평행이동을 정점에 적용한다 — 삼각형·UV는 그대로 옮긴다.

    리뷰 B7: 항등 회전을 쓰면 회전을 전치해도(행/열을 바꿔도) 결과가 똑같아 그
    버그를 못 잡는다. 그래서 대칭이 아닌 회전(Z축 90도)을 쓴다 — `place`가
    `rotation @ v`가 아니라 `v @ rotation`(=전치를 곱한 것과 같다)을 계산하는
    버그가 있으면 이 테스트가 실패한다(직접 확인: `rotation_math.transpose`로
    바꿔 넣으면 (5,4,7)이 나와 아래 기대값과 어긋난다).
    """
    mesh = level_mesh.Mesh(
        vertices=[(1.0, 0.0, 0.0)],
        triangles=[(0, 0, 0)],
        uvs=[(0.1, 0.2)],
        declared={"marker": True},
        uv_indices=[(0, 0, 0)],
    )
    rotate_90_about_z = (0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    alignment = level_mesh.Alignment(
        level=1,
        provenance="solved",
        rotation=rotate_90_about_z,
        translation=(5.0, 6.0, 7.0),
        scale=2.0,
        declared_scale=1.0,
        fit={},
    )
    placed = level_mesh.place(mesh, alignment)
    # rotation @ (1,0,0) = (0,1,0); *2 + (5,6,7) = (5,8,7). 전치를 곱하면 (0,-1,0)이
    # 되어 (5,4,7)이 나온다 — 아래 assert가 그 차이를 실제로 가른다.
    assert placed.vertices[0] == pytest.approx((5.0, 8.0, 7.0))
    assert placed.uvs == mesh.uvs
    assert placed.declared == mesh.declared
    # 리뷰 D2에서 드러난 버그: place()가 uv_indices를 안 옮겼었다 — 옮긴 메시에
    # `triangle_uv`를 부르면 `ValueError`가 났다(직접 재현·확인). 지금은 G2가
    # 배치된 메시의 UV를 읽으므로 이 필드가 빠지면 게이트 자체가 못 돈다.
    assert placed.uv_indices == mesh.uv_indices


# =============================================================================
# 리뷰 B4: provenance는 정해진 값 밖을 생성 시점에 거절해야 한다
# =============================================================================


_ALIGNMENT_KWARGS = dict(
    level=1,
    rotation=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
    translation=(0.0, 0.0, 0.0),
    scale=1.0,
    declared_scale=1.0,
    fit={},
)


def test_alignment_accepts_the_three_valid_provenance_values() -> None:
    level_mesh.Alignment(provenance="solved", **_ALIGNMENT_KWARGS)
    level_mesh.Alignment(provenance="declared", **_ALIGNMENT_KWARGS)
    level_mesh.Alignment(provenance="unverifiable", **_ALIGNMENT_KWARGS)


def test_alignment_rejects_an_invalid_provenance() -> None:
    """B4: `Alignment(provenance="totally-made-up", ...)`가 그대로 왕복하면 안 된다.

    "맞춘 값"과 "읽은 값"의 구분이 이 필드의 존재 이유 전부다 — 레벨 1 하나만
    테스트하면 그 구분이 실제로 강제되는지(생성자 수준에서)가 아니라 이번에 우연히
    "solved"였다는 사실만 확인하게 된다. 여기서는 값 자체를 검증한다.
    """
    with pytest.raises(ValueError):
        level_mesh.Alignment(provenance="totally-made-up", **_ALIGNMENT_KWARGS)


def test_load_alignment_rejects_a_tampered_provenance(tmp_path: Path) -> None:
    """B4: `load_alignment`도 `Alignment` 생성자를 거치므로 손상된 JSON도 여기서 걸린다."""
    import json

    entry = {
        "level": 1,
        "provenance": "not-a-real-provenance",
        "solved_transform": {
            "rotation": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            "translation": [0.0, 0.0, 0.0],
            "scale": 1.0,
        },
        "declared_scale": 1.0,
        "fit": {},
        "source_fbx": "x.fbx",
    }
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps([entry]), encoding="utf-8")
    with pytest.raises(ValueError):
        level_mesh.load_alignment(1, path=tampered)


# =============================================================================
# 리뷰 B5: `UpAxis`가 없으면 기본값으로 조용히 채우지 않는다
# =============================================================================


def test_up_axis_has_no_silent_default() -> None:
    """B5: 키가 없으면 `read_fbx`가 멈춘다 — 세 번째 독립 확인이 `1 == 1`로 무너지면 안 된다.

    `read_fbx`가 실제로 이 헬퍼로 `UpAxis`를 읽는다(코드 확인). 이진 FBX를 합성해
    끼워 넣는 대신, 그 강제 지점 자체를 직접 두드린다 — 이 헬퍼가 하는 일 전부가
    "키가 없으면 멈춘다"이므로 이걸로 그 계약이 지켜지는지 정확히 잰다.
    """
    with pytest.raises(ValueError):
        level_mesh._require_scalar({}, "UpAxis", Path("dummy.fbx"), "GlobalSettings")
    assert level_mesh._require_scalar({"UpAxis": [1]}, "UpAxis", Path("dummy.fbx"), "GlobalSettings") == 1


# =============================================================================
# 리뷰 B3: UVIndex를 보존해 삼각형별 UV를 되짚을 수 있어야 한다
# =============================================================================


def test_mesh_preserves_uv_indices_per_triangle() -> None:
    """B3: `uv_indices`가 없으면 `uvs`는 어느 삼각형의 UV인지 모르는 값 목록일 뿐이다."""
    mesh = level_mesh.read_fbx(LEVEL1_FBX)
    assert len(mesh.uv_indices) == len(mesh.triangles)
    for i0, i1, i2 in mesh.uv_indices:
        assert 0 <= i0 < len(mesh.uvs)
        assert 0 <= i1 < len(mesh.uvs)
        assert 0 <= i2 < len(mesh.uvs)


def test_triangle_uv_interpolates_from_the_committed_api() -> None:
    """B3: "UV 조회를 공개 API로 못 해서 리뷰어가 재구현했다"는 지적의 재발 방지."""
    mesh = level_mesh.read_fbx(LEVEL1_FBX)
    i0, _i1, _i2 = mesh.uv_indices[0]
    corner0_uv = mesh.uvs[i0]
    # 무게중심 (1,0,0)은 삼각형의 첫 코너 자체다 — 그 코너의 UV와 정확히 같아야 한다.
    assert level_mesh.triangle_uv(mesh, 0, (1.0, 0.0, 0.0)) == pytest.approx(corner0_uv)


def test_triangle_uv_refuses_a_mesh_without_uv_indices() -> None:
    mesh = level_mesh.Mesh(vertices=[(0.0, 0.0, 0.0)], triangles=[(0, 0, 0)], uvs=[], declared={})
    with pytest.raises(ValueError):
        level_mesh.triangle_uv(mesh, 0, (1.0, 0.0, 0.0))

# =============================================================================
# 절대 UV 등식 — 옛 `EXPECTED_UV_SLOPE`(부호 있는 회귀 기울기)를 대신한다.
#
# 옛 판별자는 적합된 기울기의 부호였고, 그것은 원리적으로 180° 대칭을 못 갈랐다
# (모듈이 대수적으로 보였다: obs=k1·w+c1, pred=k2·w+c2면 slope(obs|pred)=k1/k2).
# 새 판별자는 `obs = UV_ATLAS_TEXELS · mesh_uv`이고 자유 파라미터가 0개다.
# =============================================================================

SOLVED_LEVELS = [
    int(entry["level"])
    for entry in json.loads(level_mesh.ALIGNMENT_PATH.read_text(encoding="utf-8"))
    if entry["provenance"] == "solved"
]


def _entry(level: int) -> dict:
    for entry in json.loads(level_mesh.ALIGNMENT_PATH.read_text(encoding="utf-8")):
        if int(entry["level"]) == level:
            return entry
    raise LookupError(level)


@functools.lru_cache(maxsize=None)
def _hits(level: int, min_hits: int = 3000):
    """CSV 읽기가 이 파일에서 가장 비싼 일이다 — 레벨당 한 번만 읽는다."""
    hits, uvs, sessions = level_mesh.collect_level_hits_by_session(level, min_hits=min_hits)
    return hits, uvs, sessions


@functools.lru_cache(maxsize=None)
def _mesh_and_index(source: str):
    mesh = level_mesh.read_fbx(level_mesh.DEFAULT_MODELINGS_ROOT / source)
    return mesh, level_mesh.build_uv_index(mesh)


def _fixture(level: int, n_hits: int = 120):
    mesh, index = _mesh_and_index(_entry(level)["source_fbx"])
    alignment = level_mesh.load_alignment(level)
    hits, uvs, _sessions = _hits(level)
    return mesh, index, alignment, hits[:n_hits], uvs[:n_hits]


# commit 325ae9a가 `level_alignment.json`에 유일하게 기록했던 레벨 1 항목 그대로다.
# **알려진 오답을 얼려 둔 회귀 대조다** — 이 항목은 옛 거리 게이트(max 8.46mm)를
# 통과했지만 절대 UV 등식에서는 수백 텍셀 어긋난다. 누가 나중에 G2의 문턱을
# 느슨하게 하면 이 테스트가 빨개진다.
_LEVEL1_ENTRY_AT_325AE9A = {
    "level": 1,
    "provenance": "solved",
    "solved_transform": {
        "rotation": [
            -2.7476412871167704e-10, -0.9999999999999868, -1.6235287959752675e-07,
            1.5843529468245916e-05, 1.6235287522391646e-07, -0.9999999998744781,
            0.9999999998744914, -2.7733637130933536e-10, 1.5843529468134027e-05,
        ],
        "translation": [1.4240009301575554, 4.629764529233568, 0.6590600777324329],
        "scale": 0.025398112818281186,
    },
    "declared_scale": 0.0254,
    "fit": {},
}


def test_the_level_1_entry_recorded_at_325ae9a_is_rejected_by_the_uv_gate() -> None:
    """옛 커밋의 레벨 1 항목을 그대로 먹여 G2가 거절하는 것을 단언한다.

    이 항목은 그때 유일하게 "통과"한 정렬이었고 거리로는 지금도 통과한다
    (기록된 max_mm 8.46). 절대 UV 잔차로는 수백 텍셀이다 — 즉 새 게이트는
    "옛 게이트가 잡던 것을 다 잡는다"보다 강하다: **옛 게이트가 놓친 것을 잡는다.**
    """
    mesh = level_mesh.read_fbx(LEVEL1_FBX)
    index = level_mesh.build_uv_index(mesh)
    alignment = level_mesh._alignment_from_entry(_LEVEL1_ENTRY_AT_325AE9A)
    hits, uvs, _s = _hits(1)
    measured = level_mesh.verify_alignment(mesh, alignment, hits[:120], uvs[:120], index=index)

    assert measured["uv_abs_resid_median_texel"] > 100.0, (
        f"옛 레벨 1 항목의 절대 UV 잔차 중앙값이 {measured['uv_abs_resid_median_texel']:.1f} 텍셀이다 — "
        "이 값이 작아졌다면 게이트나 규약이 조용히 바뀐 것이다"
    )
    failures = {name for name, _v, _n in level_mesh.gate_failures(measured)}
    assert "uv_abs_resid_p95_texel" in failures
    # 그리고 거리로는 통과한다는 것도 함께 고정한다 — 이것이 G2가 필요한 이유다.
    assert measured["median_mm"] < level_mesh.GATES["median_mm"]


@pytest.mark.parametrize("level", SOLVED_LEVELS)
def test_solved_entries_satisfy_the_absolute_uv_equation(level: int) -> None:
    """기록된 정렬 전부에서 `obs = 2048 · mesh_uv`가 홀드아웃 착탄점에 성립한다."""
    mesh, index, alignment, hits, uvs = _fixture(level)
    measured = level_mesh.verify_alignment(mesh, alignment, hits, uvs, index=index)
    assert measured["uv_abs_resid_p95_texel"] <= level_mesh.GATES["uv_abs_resid_p95_texel"]
    assert measured["uv_abs_resid_max_texel"] <= level_mesh.GATES["uv_abs_resid_max_texel"]
    assert measured["max_mm"] <= level_mesh.GATES["max_mm"]


# =============================================================================
# 합성 변이 배터리 — 각 변이가 **어느 게이트에** 걸리는지까지 단언한다
# =============================================================================


def _mutate(alignment, mesh, kind: str, param):
    rot = level_mesh._rotation_matrix(alignment)
    if kind in ("rot180", "mirror"):
        axis = param
        base = 1.0 if kind == "mirror" else -1.0
        m = [[0.0] * 3 for _ in range(3)]
        for i in range(3):
            m[i][i] = base
        m[axis][axis] = -base
        matrix = tuple(tuple(row) for row in m)
        centre = level_mesh._mean(mesh.vertices)
        new_rot = rotation_math.matmul(rot, matrix)
        shift = level_mesh._sub(centre, rotation_math.matvec(matrix, centre))
        rotated = rotation_math.matvec(rot, shift)
        translation = tuple(
            alignment.translation[d] + alignment.scale * rotated[d] for d in range(3)
        )
        return dataclasses.replace(
            alignment,
            rotation=tuple(new_rot[r][c] for r in range(3) for c in range(3)),
            translation=translation,
        )
    if kind == "scale":
        return dataclasses.replace(alignment, scale=alignment.scale * param)
    if kind == "nudge":
        direction, millimetres = param
        return dataclasses.replace(
            alignment,
            translation=tuple(
                alignment.translation[d] + direction[d] * millimetres / 1000.0 for d in range(3)
            ),
        )
    raise ValueError(kind)


def _dominant_normal(mesh, alignment, hits, uvs):
    placed = level_mesh._world_tris_for(
        mesh.vertices, mesh.triangles, level_mesh._rotation_matrix(alignment),
        alignment.scale, alignment.translation,
    )
    _d, _r, triangle_ids, _t = level_mesh.uv_absolute_residuals(placed, mesh, hits[:40], uvs[:40])
    common = collections.Counter(triangle_ids).most_common(1)[0][0]
    return level_mesh._tri_normal(placed[common])


_MUTATIONS = (
    ("rot180", 0), ("rot180", 1), ("rot180", 2),
    ("mirror", 0), ("mirror", 1), ("mirror", 2),
    ("scale", 1.02), ("scale", 0.81),
)


@pytest.mark.parametrize("level", [3, 8])
def test_every_synthetic_mutation_is_rejected(level: int) -> None:
    """주축 180° 회전 3개 · 주평면 반사 3개 · 스케일 ×1.02/×0.81 · 법선 이동 12/19/50 mm.

    실측(커밋 f632657, 8개 채택 레벨 × 변이 11개 = 88건): **통과한 변이는 0건**이다.
    """
    mesh, index, alignment, hits, uvs = _fixture(level, n_hits=120)
    normal = _dominant_normal(mesh, alignment, hits, uvs)
    mutations = list(_MUTATIONS) + [("nudge", (normal, mm)) for mm in (12.0, 19.0, 50.0)]
    for kind, param in mutations:
        measured = level_mesh.verify_alignment(
            mesh, _mutate(alignment, mesh, kind, param), hits, uvs, index=index
        )
        failures = level_mesh.gate_failures(measured)
        assert failures, f"level {level} {kind}/{param} passed every gate"


def test_distance_alone_would_accept_mutations_that_the_uv_gate_rejects() -> None:
    """어블레이션 — 새 판별력이 어디서 오는지 못 박는다.

    레벨 8의 방은 주축 180° 회전 셋과 주평면 반사 셋 **전부에 대해 자기 자신으로
    사상된다**. 실측(120점 홀드아웃): 여섯 변이 전부 표면 거리 중앙값 0.0002 mm /
    p95 0.0005 mm / max 0.0009 mm로 **G3를 온전히 통과**하고, G2가 514~1037 텍셀로
    거절한다. 즉 G2를 "중복"이라며 지우면 레벨 8은 180° 뒤집힌 채 기록된다.

    (정직하게 적는다: 나머지 일곱 레벨에서는 같은 변이를 거리도 잡는다 — 레벨 3은
    rot180에서 중앙값 17.7~108 mm다. 거리가 대칭을 **항상** 못 잡는다는 진술은
    측정과 어긋난다. 못 잡는 것은 자산 자신이 그 대칭을 가진 레벨에서다.)
    """
    mesh, index, alignment, hits, uvs = _fixture(8, n_hits=120)
    distance_only_passes = 0
    for kind, axis in _MUTATIONS[:6]:
        measured = level_mesh.verify_alignment(
            mesh, _mutate(alignment, mesh, kind, axis), hits, uvs, index=index
        )
        distance_gates = {"median_mm", "p95_mm", "max_mm"}
        failed = {name for name, _v, _n in level_mesh.gate_failures(measured)}
        if not (failed & distance_gates):
            distance_only_passes += 1
            assert measured["uv_abs_resid_p95_texel"] > 100.0, (
                f"{kind}/{axis}: 거리로도 UV로도 안 걸린다 — 게이트에 구멍이 있다"
            )
    assert distance_only_passes == 6, (
        f"레벨 8의 여섯 대칭 변이 중 {distance_only_passes}개만 거리 게이트를 통과했다 — "
        "이 어블레이션의 전제(이 자산은 그 대칭들에 대해 자기 자신이다)가 바뀌었다"
    )


def test_uv_gate_must_pick_the_triangle_by_3d_distance_not_by_uv_lookup() -> None:
    """순환 방지 — UV 조회로 삼각형을 고르면 `obs = 2048·mesh_uv`가 구성상 참이 된다.

    금지 이유를 주석이 아니라 실행 가능한 형태로 남긴다: 조회로 고르는 변종에서는
    180° 변이의 잔차가 **0으로 떨어져** 그대로 통과한다.
    """
    mesh, index, alignment, hits, uvs = _fixture(8, n_hits=60)
    broken = _mutate(alignment, mesh, "rot180", 0)

    # 금지된 변종: 삼각형을 관측 UV로 고른다.
    residuals = []
    for obs in uvs:
        kind, triangle, bary = level_mesh.uv_lookup(
            index, (obs[0] / level_mesh.UV_ATLAS_TEXELS, obs[1] / level_mesh.UV_ATLAS_TEXELS)
        )
        if kind != "hit":
            continue
        mesh_u, mesh_v = level_mesh.triangle_uv(mesh, triangle, bary)
        residuals.append(
            math.hypot(
                obs[0] - level_mesh.UV_ATLAS_TEXELS * mesh_u,
                obs[1] - level_mesh.UV_ATLAS_TEXELS * mesh_v,
            )
        )
    circular_max = max(residuals)

    honest = level_mesh.verify_alignment(mesh, broken, hits, uvs, index=index)
    assert circular_max < 1e-6, "조회로 고르면 잔차가 정렬과 무관하게 0이다 — 그래서 금지다"
    assert honest["uv_abs_resid_p95_texel"] > 100.0, "3D 최근접으로 고르면 같은 변이가 걸린다"


# =============================================================================
# 표본 분리 · 게이트가 실제로 무언가를 거절함 · 아틀라스 단사성
# =============================================================================


def test_fit_and_eval_samples_are_disjoint() -> None:
    """옛 코드는 같은 rng에서 두 번 독립 추출해 fit/eval이 거의 전부 겹쳤다."""
    mesh, _index = _mesh_and_index(_entry(3)["source_fbx"])
    hits, uvs, _s = _hits(3)
    _a, measured, _eh, _eu = level_mesh.solve_alignment_uv(
        mesh, hits, hit_uvs=uvs, level=3, fit_sample=1200, eval_cap=400
    )
    assert measured["fit_eval_overlap"] == 0.0


def test_the_gate_battery_actually_rejects_part_of_the_corpus() -> None:
    """10개가 전부 통과하면 성공이 아니라 배선이 잘못됐다는 증거로 다룬다(사전등록).

    실측: 레벨 1은 G7(조건수 2.06e-09)로, 레벨 2는 G1(퇴화 UV 삼각형 오대응
    135.9 mm, 조회 실패율 0.0035)로 떨어진다.
    """
    entries = json.loads(level_mesh.ALIGNMENT_PATH.read_text(encoding="utf-8"))
    rejected = [e for e in entries if e["provenance"] == "unverifiable"]
    assert rejected, "어느 레벨도 거절되지 않았다 — 게이트가 실제로 무언가를 가르는지 의심하라"
    for entry in rejected:
        assert entry["rejected_by"], "거절 사유가 숫자로 남아야 한다"
        assert "solved_transform" not in entry, "검증 못 한 변환을 적어 두면 다음 사람이 쓴다"


def test_unverifiable_alignments_carry_no_usable_transform() -> None:
    unverifiable = [
        e for e in json.loads(level_mesh.ALIGNMENT_PATH.read_text(encoding="utf-8"))
        if e["provenance"] == "unverifiable"
    ]
    assert unverifiable
    for entry in unverifiable:
        alignment = level_mesh.load_alignment(int(entry["level"]))
        assert alignment.provenance == "unverifiable"
        assert alignment.scale == 0.0  # 쓰려고 하면 즉시 무너지는 값이다


def test_degenerate_uv_triangles_are_never_lookup_candidates() -> None:
    """레벨 2의 아틀라스는 단사가 아니다 — 삼각형 14개가 3D 면적을 갖고도 UV 면적이 0이다.

    그 위에 떨어진 착탄점은 miss도 ambiguous도 아닌 **다른 삼각형의 깨끗한 hit**으로
    돌아온다. 그래서 착탄점 가중 통계로는 원리적으로 못 본다 — `audit_atlas`가
    퇴화 삼각형 위를 직접 표집해 오대응 거리를 잰다.
    """
    mesh = level_mesh.read_fbx(
        level_mesh.DEFAULT_MODELINGS_ROOT / "learning_mode_lv.2_rev.1.2.fbx"
    )
    index = level_mesh.build_uv_index(mesh)
    assert len(index.degenerate_triangles) == 14
    assert all(t not in ts for ts in index.buckets.values() for t in index.degenerate_triangles)
    audit = level_mesh.audit_atlas(mesh, index)
    assert audit["uv_degenerate_misattribution_mm"] > level_mesh.GATES["uv_degenerate_misattribution_mm"]
    assert audit["uv_degenerate_area_fraction"] == pytest.approx(0.0646, abs=0.001)


def test_barycentric_2d_refuses_a_degenerate_uv_triangle() -> None:
    """분모를 클램프하면 그 삼각형이 모든 질의점을 포함하고(모호도 1.0), nan으로
    흘려보내면 조용히 사라진다. 같은 명세에서 정반대 판정이 나온다 — 그래서 명시적
    None으로 고정한다."""
    collapsed = ((0.115001, 0.799489), (0.2178, 0.799489), (0.115001, 0.799489))
    assert level_mesh._barycentric_2d((0.15, 0.5), collapsed) is None


def test_hit_collector_spans_workers_and_sessions() -> None:
    """옛 수집기는 glob 순서로 읽다가 15,000점에서 멈춰 레벨 6·7·10이 1세션처럼 보였다."""
    _h, _uvs, sessions = _hits(6)
    assert len({level_mesh._worker_of(s) for s in sessions}) >= 4
    assert len(set(sessions)) >= 8


# =============================================================================
# 손잡이(det<0)와 그 소비자
# =============================================================================


def test_accepted_entries_share_one_handedness() -> None:
    """G5: 섞이면 하류에서 아무도 못 알아챈다. 실측상 채택 8개 전부 det<0이다."""
    entries = [
        e for e in json.loads(level_mesh.ALIGNMENT_PATH.read_text(encoding="utf-8"))
        if e["provenance"] == "solved"
    ]
    assert entries
    signs = {e["fit"]["det_sign"] < 0 for e in entries}
    assert signs == {True}, "채택 항목의 손잡이가 섞였다"
    for entry in entries:
        assert entry["handedness_source"] == "data"


def test_to_world_round_trips_through_a_reflection() -> None:
    """det<0 소비자 감사. `capture_frame.to_world`의 역행렬은 전치인데, 반사도
    직교행렬이므로 그대로 옳다 — 그 사실을 고정 테스트로 남긴다. 누가 나중에 정렬
    회전을 쿼터니언으로 바꾸면 det=-1에서 조용히 망가진다.
    """
    capture_frame = importlib.import_module("capture_frame")
    level = SOLVED_LEVELS[0]
    alignment = level_mesh.load_alignment(level)
    assert level_mesh._mat3_det(level_mesh._rotation_matrix(alignment)) < 0
    rotation = level_mesh._rotation_matrix(alignment)
    raw = (12.3, -4.5, 6.7)
    world = tuple(
        alignment.scale * rotation_math.matvec(rotation, raw)[d] + alignment.translation[d]
        for d in range(3)
    )
    back = capture_frame.to_world(world, level=level)
    assert back == pytest.approx(tuple(alignment.scale * x for x in raw), abs=1e-9)


# =============================================================================
# 커밋된 상수가 깨지면 잡는 테스트 (옛 B6 계열을 새 해에 맞게 다시 씀)
# =============================================================================


def test_committed_rotation_is_orthogonal_and_orientation_reversing() -> None:
    """회전을 항등원·0·임의 행렬로 바꿔도 통과하면 안 된다.

    옛 판본은 `det == +1`과 "정육면체 회전 24개 중 하나에 가깝다"를 단언했다. 둘 다
    레벨 1 하나에서 나온 가정이었고 **둘 다 틀렸다**:

    * `det`: 새 해는 반사를 살려 둔다(`orthogonalize`가 det 보정을 하지 않는다).
      채택된 8개 전부 det = -1이다. `_umeyama`가 `d = ±1`로 det를 항상 +1로 만든 것이
      레벨 2~10의 참 변환을 **원리적으로 표현하지 못하게** 한 원인이었다.
    * "90° 단위": 실측상 레벨 3·4·5·6·7·9·10은 부호 있는 축 치환에서 잔차 1e-14이지만
      **레벨 8은 축에서 24.172° 어긋난 yaw**다. 옛 시드 격자에 그 회전이 없었기 때문에
      옛 솔버는 레벨 8을 원리적으로 못 찾았다. 그 사실을 숫자로 고정해 둔다.
    """
    axis_aligned = 0
    for level in SOLVED_LEVELS:
        alignment = level_mesh.load_alignment(level)
        rot = level_mesh._rotation_matrix(alignment)
        product = rotation_math.matmul(rot, rotation_math.transpose(rot))
        for i in range(3):
            for j in range(3):
                assert product[i][j] == pytest.approx(1.0 if i == j else 0.0, abs=1e-9)
        assert level_mesh._mat3_det(rot) == pytest.approx(-1.0, abs=1e-6)
        nearest = min(
            (
                rotation_math.matmul(cube, mirror)
                for cube in level_mesh._cube_rotations()
                for mirror in level_mesh._mirror_matrices()
            ),
            key=lambda c: sum((rot[i][j] - c[i][j]) ** 2 for i in range(3) for j in range(3)),
        )
        residual = sum((rot[i][j] - nearest[i][j]) ** 2 for i in range(3) for j in range(3))
        if level == 8:
            trace = sum(sum(rot[i][k] * nearest[i][k] for i in range(3)) for k in range(3))
            angle = math.degrees(math.acos(max(-1.0, min(1.0, (trace - 1.0) / 2.0))))
            assert angle == pytest.approx(24.172, abs=0.02)
        else:
            assert residual < 1e-10, f"level {level} rotation residual {residual:.3e}"
            axis_aligned += 1
    assert axis_aligned == len(SOLVED_LEVELS) - 1


def test_replacing_a_committed_rotation_with_the_identity_is_rejected() -> None:
    """위 테스트가 손상을 못 잡는 경우를 막는다 — 항등 행렬도 직교이고 det는 +1이라
    `det == -1` 단언이 잡지만, 그 단언만으로는 "직교이고 det=-1인 임의 반사"를 못 잡는다.
    관측 가능한 결과로 확인한다."""
    level = SOLVED_LEVELS[0]
    mesh, index, alignment, hits, uvs = _fixture(level, n_hits=80)
    identity = dataclasses.replace(
        alignment, rotation=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    )
    measured = level_mesh.verify_alignment(mesh, identity, hits, uvs, index=index)
    assert level_mesh.gate_failures(measured)


def _least_squares(xs, ys):
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx if sxx else 0.0
    syy = sum((y - my) ** 2 for y in ys)
    r2 = (sxy * sxy) / (sxx * syy) if sxx and syy else 0.0
    return slope, r2


def test_committed_scale_matches_an_independent_texel_density_estimate() -> None:
    """B6: 맞춤과 무관한 추정치로 스케일을 다시 잰다.

    메시 쪽: 한 삼각형의 (모델 좌표 → UV) 기울기에서 텍셀/모델단위.
    CSV 쪽: 착탄점 월드 좌표 → 관측 텍셀 기울기에서 텍셀/미터.
    둘의 비가 모델단위→미터 스케일이고, 적합에 들어가지 않은 수다.
    """
    level = SOLVED_LEVELS[0]
    mesh, index, alignment, hits, uvs = _fixture(level, n_hits=2000)
    placed = level_mesh._world_tris_for(
        mesh.vertices, mesh.triangles, level_mesh._rotation_matrix(alignment),
        alignment.scale, alignment.translation,
    )
    _d, _r, triangle_ids, _t = level_mesh.uv_absolute_residuals(placed, mesh, hits[:80], uvs[:80])
    common = collections.Counter(triangle_ids).most_common(1)[0][0]
    on_face = [i for i, t in enumerate(triangle_ids) if t == common]
    assert len(on_face) >= 20

    # 그 삼각형 평면 안의 두 직교 방향으로 (월드 거리, 관측 텍셀) 기울기를 잰다.
    a, b, c = placed[common]
    e1 = level_mesh._sub(b, a)
    e1 = level_mesh._mul3(e1, 1.0 / math.sqrt(level_mesh._dot(e1, e1)))
    normal = level_mesh._tri_normal(placed[common])
    e2 = level_mesh._cross(normal, e1)
    estimates = []
    for axis_vector in (e1, e2):
        proj = [level_mesh._dot(level_mesh._sub(hits[i], a), axis_vector) for i in on_face]
        for uv_axis in (0, 1):
            obs = [uvs[i][uv_axis] for i in on_face]
            slope, r2 = _least_squares(proj, obs)
            if r2 > 0.999 and abs(slope) > 1.0:
                estimates.append(abs(slope))
    assert estimates, "이 면에서 텍셀/미터를 독립으로 잴 수 있어야 한다"
    texels_per_metre = max(estimates)
    # 메시 쪽 텍셀/모델단위: 같은 삼각형의 UV 변 길이 ÷ 모델 변 길이
    mu = level_mesh._tri_uv_corners(mesh, common)
    mv = level_mesh._tri_vertices(mesh, common)
    edge_uv = math.dist(mu[0], mu[1]) * level_mesh.UV_ATLAS_TEXELS
    edge_model = math.dist(mv[0], mv[1])
    texels_per_model_unit = edge_uv / edge_model
    assert alignment.scale == pytest.approx(texels_per_model_unit / texels_per_metre, rel=0.01)


def test_committed_translation_places_real_hits_on_the_mesh() -> None:
    """`translation`을 0으로 바꾸거나 성분을 맞바꾸면 잡힌다 — 저장된 숫자를 자기
    자신과 비교하지 않고 실제 착탄점까지의 거리로 확인한다."""
    for level in SOLVED_LEVELS:
        alignment = level_mesh.load_alignment(level)
        entry = _entry(level)
        mesh, _index = _mesh_and_index(entry["source_fbx"])
        placed = level_mesh.place(mesh, alignment)
        hits, _uvs, _s = _hits(level)
        distances = sorted(level_mesh.closest_distance(placed, h) for h in hits[:120])
        assert distances[len(distances) // 2] < 1e-4


def test_committed_fit_records_every_gate_it_was_judged_by() -> None:
    """`fit`에 게이트가 읽는 키가 전부 남아야 한다 — 없으면 다음 사람이 재검사할 수 없다."""
    for level in SOLVED_LEVELS:
        fit = _entry(level)["fit"]
        for key in level_mesh._LOWER_IS_BETTER + level_mesh._HIGHER_IS_BETTER:
            assert key in fit, f"level {level} fit is missing {key}"
        assert _entry(level)["uv_convention"] == level_mesh.UV_CONVENTION
