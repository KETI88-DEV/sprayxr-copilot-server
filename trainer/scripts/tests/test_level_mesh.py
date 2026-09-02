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
rotation_math = importlib.import_module("rotation_math")

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
    # 리뷰 D2에서 드러난 버그: place()가 uv_indices를 안 옮겼었다 — 옮긴 메시로
    # uv_pairs_on_dominant_face를 부르면 항상 빈 목록이 나왔다(직접 재현·확인).
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


def test_alignment_accepts_the_two_valid_provenance_values() -> None:
    level_mesh.Alignment(provenance="solved", **_ALIGNMENT_KWARGS)
    level_mesh.Alignment(provenance="declared", **_ALIGNMENT_KWARGS)


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


def test_uv_pairs_on_dominant_face_matches_the_committed_fit() -> None:
    """리뷰 D2: 다음 태스크의 메시 앵커가 쓸 공개 진입점.

    `_uv_agreement`(`solve_alignment`이 정렬을 고를 때 씀)와 같은 계산
    (`_uv_pairs_for_dominant_face`)을 쓴다 — 독립적으로 다시 짠 게 아니라는 것을
    커밋된 `fit` 값과 가까운 결과가 나오는지로 확인한다(표본이 달라 완전히
    같지는 않다: `solve_alignment`은 무작위 15000개 부분표본을, 여기서는 처음
    15000개를 쓴다).
    """
    a = level_mesh.load_alignment(1)
    placed = level_mesh.place(level_mesh.read_fbx(LEVEL1_FBX), a)
    hits, hit_uvs = level_mesh._collect_level_hits(1, min_hits=15000)
    pairs = level_mesh.uv_pairs_on_dominant_face(placed, list(zip(hits[:15000], hit_uvs[:15000])))

    assert len(pairs) > 14000  # 옆벽 ~5%를 뺀 나머지 대부분

    slope_u, r2_u = level_mesh.uv_regression(pairs, 0)
    slope_v, r2_v = level_mesh.uv_regression(pairs, 1)
    assert slope_u == pytest.approx(a.fit["uv_slope_u"], abs=1.0)
    assert slope_v == pytest.approx(a.fit["uv_slope_v"], abs=1.0)
    assert r2_u == pytest.approx(a.fit["uv_r2_u"], abs=1e-4)
    assert r2_v == pytest.approx(a.fit["uv_r2_v"], abs=1e-4)
    # C2에서 고정한 부호(EXPECTED_UV_SLOPE)와도 맞아야 한다 — 앵커가 이 상수로 비교한다.
    assert slope_u == pytest.approx(level_mesh.EXPECTED_UV_SLOPE, abs=5.0)
    assert slope_v == pytest.approx(level_mesh.EXPECTED_UV_SLOPE, abs=5.0)


def test_uv_regression_needs_at_least_two_pairs() -> None:
    """B2 계열: 기울기를 지어내지 않는다 — `_linreg`와 같은 규칙."""
    assert level_mesh.uv_regression([], 0) == (0.0, 0.0)
    assert level_mesh.uv_regression([((0.1, 0.2), (100.0, 200.0))], 0) == (0.0, 0.0)


# =============================================================================
# 리뷰 B6: 커밋된 상수가 깨지면 잡는 테스트
# =============================================================================


def test_committed_rotation_is_a_valid_near_permutation_rotation() -> None:
    """B6: 회전을 항등원·0·임의 행렬로 바꿔도 통과하면 안 된다.

    직접 깨서 확인했다: 회전을 항등 행렬로, 평행이동을 0으로, 스케일을 스파이크의
    옛 값(0.02568)으로, `fit`을 `{}`로, `max_mm`을 999999로 바꿔도 기존 5개 테스트는
    전부 통과했다(리뷰가 지적한 사실 그대로 재현) — 이 테스트들이 그 구멍을 막는다.

    **함정 하나를 실제로 깨서 찾았다.** 처음엔 "정육면체 회전 24개 중 가장 가까운
    것과의 잔차"만 쟀는데, **항등 행렬 자체가 그 24개 중 하나**라 이 검사가 항등
    행렬 손상을 못 잡았다(직접 확인: 잔차 0으로 통과). 그래서 방향까지 확인하는
    항을 더한다 — 착탄점이 정확히 평면인 세계축(가장 좁은 실측 범위)이 메시의 가장
    얇은 원시축에 대응해야 한다(`rot[flat_world_axis][thin_axis] ≈ ±1`). 항등
    행렬은 이 대응이 없어(그 성분이 0) 여기서 걸린다 — 재확인함.
    """
    a = level_mesh.load_alignment(1)
    r = a.rotation
    rot = ((r[0], r[1], r[2]), (r[3], r[4], r[5]), (r[6], r[7], r[8]))

    rt = rotation_math.transpose(rot)
    product = rotation_math.matmul(rot, rt)
    identity = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    for i in range(3):
        for j in range(3):
            assert product[i][j] == pytest.approx(identity[i][j], abs=1e-6)
    assert level_mesh._mat3_det(rot) == pytest.approx(1.0, abs=1e-6)

    # 씬 배치는 90도 단위다(모듈 독스트링, D1) — 가장 가까운 정육면체 회전과 거의
    # 같아야 한다. 성분별 제곱합 잔차 1e-4는 실측(약 5e-10)보다 훨씬 느슨하다.
    nearest = min(
        level_mesh._cube_rotations(),
        key=lambda c: sum((rot[i][j] - c[i][j]) ** 2 for i in range(3) for j in range(3)),
    )
    residual = sum((rot[i][j] - nearest[i][j]) ** 2 for i in range(3) for j in range(3))
    assert residual < 1e-4

    # 방향 확인: 착탄점이 평면인 세계축(실측 범위가 가장 좁은 축)이 메시의 가장 얇은
    # 원시축에 대응해야 한다. 항등 행렬은 이 대응이 없다 — 정육면체 회전군의 다른
    # 23개 원소와 항등원을 가르는 지점이 바로 여기다.
    mesh = level_mesh.read_fbx(LEVEL1_FBX)
    hits, _hit_uvs = level_mesh._collect_level_hits(1, min_hits=15000)
    hit_extent = level_mesh._extents(hits)
    flat_world_axis = min(range(3), key=lambda i: hit_extent[i])
    thin_axis = level_mesh._thin_axis_index(mesh.vertices)
    assert abs(rot[flat_world_axis][thin_axis]) == pytest.approx(1.0, abs=1e-3)


def test_committed_scale_is_within_the_texel_density_band() -> None:
    """B6: 스케일이 텍셀 밀도로 재구성한, 맞춤과 무관한 독립 추정치에서 크게 벗어나면 잡는다.

    방법(리뷰가 요구한 "띠와 그 출처"): 큰 면의 안쪽 사각형(내부 구멍 테두리) 정점만
    쓴다 — 바깥 테두리는 텍스처 블리드 여백이 섞여 원시좌표·UV의 선형 관계가 안쪽만큼
    깨끗하지 않다(실측: 안쪽은 U·V 두 축에서 텍셀/원시단위 값이 소수점 5자리까지
    일치하는데 바깥은 5~8% 어긋난다). CSV 쪽은 착탄점 UV를 월드 Y·Z에 바로 회귀해
    텍셀/미터를 잰다 — 메시나 맞춤과 무관한 순수 실측이다. (텍셀/원시단위) ÷
    (텍셀/미터) = raw→metre 스케일 추정치를 두 축에서 각각 내고, 그 평균을 기준으로
    ±0.5%를 띠로 쓴다(맞춘 스케일은 평균보다 0.19% 위 — 여유 안이다).
    """
    a = level_mesh.load_alignment(1)
    mesh = level_mesh.read_fbx(LEVEL1_FBX)
    hits, hit_uvs = level_mesh._collect_level_hits(1, min_hits=15000)

    thin_axis = level_mesh._thin_axis_index(mesh.vertices)
    in_plane = [i for i in range(3) if i != thin_axis]
    dominant_sign = a.fit["face_sign"]
    face_signs = level_mesh._face_sign_by_triangle(mesh.vertices, mesh.triangles, thin_axis)

    pairs = sorted(
        {
            (vi, ui)
            for tri, uvidx, sign in zip(mesh.triangles, mesh.uv_indices, face_signs)
            if sign == dominant_sign
            for vi, ui in zip(tri, uvidx)
        }
    )

    def radius(vertex_index: int) -> float:
        v = mesh.vertices[vertex_index]
        return v[in_plane[0]] ** 2 + v[in_plane[1]] ** 2

    radii = sorted({radius(vi) for vi, _ in pairs})
    assert len(radii) >= 2, "big face needs an inner and an outer boundary for this method"
    threshold = (radii[0] + radii[-1]) / 2.0
    inner = [(vi, ui) for vi, ui in pairs if radius(vi) < threshold]
    assert inner

    raw_a = [mesh.vertices[vi][in_plane[0]] for vi, _ in inner]
    raw_b = [mesh.vertices[vi][in_plane[1]] for vi, _ in inner]
    uv_u = [mesh.uvs[ui][0] for _, ui in inner]
    uv_v = [mesh.uvs[ui][1] for _, ui in inner]
    slope_u, _intercept_u, r2_u = level_mesh._linreg(raw_a, uv_u)
    slope_v, _intercept_v, r2_v = level_mesh._linreg(raw_b, uv_v)
    assert r2_u > 0.999 and r2_v > 0.999  # 이 방법 자체가 실제로 선형인지 확인
    tex_per_raw_u = abs(slope_u) * 2048.0
    tex_per_raw_v = abs(slope_v) * 2048.0

    world_y = [h[1] for h in hits]
    world_z = [h[2] for h in hits]
    obs_u = [uv[0] for uv in hit_uvs]
    obs_v = [uv[1] for uv in hit_uvs]
    slope_v_y, _intercept_1, r2_v_y = level_mesh._linreg(world_y, obs_v)
    slope_u_z, _intercept_2, r2_u_z = level_mesh._linreg(world_z, obs_u)
    assert r2_v_y > 0.999 and r2_u_z > 0.999
    tex_per_metre_v = abs(slope_v_y)
    tex_per_metre_u = abs(slope_u_z)

    scale_estimate_u = tex_per_raw_u / tex_per_metre_u
    scale_estimate_v = tex_per_raw_v / tex_per_metre_v
    band_center = (scale_estimate_u + scale_estimate_v) / 2.0

    assert a.scale == pytest.approx(band_center, rel=0.005)  # ±0.5%


def test_committed_uv_evidence_is_near_the_2048_atlas() -> None:
    """B6/B2/C2: `fit`의 UV 기울기가 -2048에서 멀면(크기든 부호든) 잡는다.

    `abs(slope)`가 아니라 **부호까지** 확인한다 — 리뷰 C2: 얇은 축 180도 회전으로
    관계된 두 orientation이 크기·R²는 똑같고 부호만 다르다(직접 재현: 둘 다 R²
    0.9999+, |slope|≈2048). `abs()`를 쓰면 이 대칭을 못 잡는다. 실측(모듈의
    `EXPECTED_UV_SLOPE` 참고)으로 확인한 "좋은 맞춤"의 부호는 두 축 다 음수다.
    """
    a = level_mesh.load_alignment(1)
    assert a.fit["uv_slope_u"] == pytest.approx(level_mesh.EXPECTED_UV_SLOPE, abs=5.0)
    assert a.fit["uv_slope_v"] == pytest.approx(level_mesh.EXPECTED_UV_SLOPE, abs=5.0)
    assert a.fit["uv_r2_u"] > 0.999
    assert a.fit["uv_r2_v"] > 0.999
    assert a.fit["dominant_face_fraction"] > 0.99
    assert a.fit["face_sign"] in (1.0, -1.0)


def test_committed_fit_distances_are_within_the_preregistered_tolerance() -> None:
    """B6: `max_mm`을 999999로 바꾸는 것 같은 손상은 위 UV 검사로는 안 걸린다(UV
    기울기·R²·face_sign은 `max_mm`과 무관한 필드다) — 별도로 스펙 §D5 허용치
    (p95 ≤ 10mm, max ≤ 100mm)를 확인한다. 직접 깨서 확인했다: 이 assert 없이
    `max_mm`만 999999로 바꾸면 이 파일의 나머지 테스트가 전부 통과했다.
    """
    a = level_mesh.load_alignment(1)
    assert a.fit["median_mm"] < 1.0
    assert a.fit["p95_mm"] < 10.0
    assert a.fit["max_mm"] < 100.0


def test_committed_translation_places_real_hits_on_the_mesh() -> None:
    """C1: `translation`은 이전까지 어느 테스트도 읽지 않았다.

    직접 확인했다: `translation`을 `[0,0,0]`으로 바꾸거나 `translation[1]`
    (4.63m)과 `[2]`(0.66m)을 맞바꿔도 — 둘 다 시도함 — 기존 16개 테스트가 전부
    그대로 통과했다. 저장된 숫자를 자기 자신과 비교하지 않는다 — `place()`를 거친
    뒤 **실제 착탄점**까지의 거리로, 관측 가능한 결과를 확인한다. 성분 순서에
    민감하다: `[1]`/`[2]`를 맞바꾸면 메시가 약 5.6 m 떨어진 곳으로 옮겨가 median이
    폭발한다 — 크기만 같은 손상(스케일 ×1.001 등, §D5 허용치 안)과 달리 이건
    허용치를 크게 벗어난다.
    """
    a = level_mesh.load_alignment(1)
    placed = level_mesh.place(level_mesh.read_fbx(LEVEL1_FBX), a)
    hits, _hit_uvs = level_mesh._collect_level_hits(1, min_hits=500)
    sample = hits[:500]
    distances = sorted(level_mesh.closest_distance(placed, h) for h in sample)
    median = distances[len(distances) // 2]
    assert median < 0.01  # 1cm — 실측 median은 ~0mm(§D5 p95 ≤ 10mm)에 비해 넉넉하다
