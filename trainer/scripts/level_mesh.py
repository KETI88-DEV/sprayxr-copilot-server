#!/usr/bin/env python3
"""레벨 FBX 리더와 착탄점 정렬.

**표준 라이브러리만 쓴다** — `struct` + `zlib`. 이 저장소의 트레이너 컨테이너에
mesh 라이브러리(FBX SDK, assimp, trimesh, numpy 등)가 없고, 새 의존성을 넣지 않는다.

바이너리 FBX 7400 리더: 27바이트 헤더 뒤 `(endOffset, numProps, propListLen, nameLen,
name, props...)` 레코드가 이어지고, `tell() < endOffset - 13`인 동안 재귀한 뒤
`seek(endOffset)`한다. 속성 타입 `Y C I L F D`는 스칼라, `f d l i b`는
`(len, encoding, compressedLen)` 배열로 encoding이 0이 아니면 `zlib.decompress`,
`S R`은 길이 선행 바이트열. `PolygonVertexIndex`는 다각형 끝 인덱스가 `~i`(비트
반전)로 표시된다.

**변환은 풀어낸 것이지 선언된 것이 아니다.** FBX가 선언한 노드 변환
(`PreRotation`/`Lcl Rotation`/`UnitScaleFactor`)으로 레벨을 배치하면 실제 착탄점과
미터 단위로 어긋난다 — 씬 수준 회전이 파일 어디에도 기록되지 않았기 때문이다.
`Alignment.provenance`가 "solved"(맞춘 것)·"declared"(읽은 것)·"unverifiable"
(데이터가 검증을 못 받치는 것)을 구분해 나른다.

**솔버는 관측 UV로 대응을 만든다 — 시드도, ICP도, 면 판별 휴리스틱도 없다.**
`Ray Texture UV`가 착탄점마다 이미 "메시의 어디를 맞혔는가"를 말하고 있다.
`build_uv_index`/`uv_lookup`이 그 UV를 (삼각형, 무게중심)으로 되돌리고,
`model_point_for_hit`이 모델 공간 점을 낸다. 그 대응에 `solve_affine_12`이
12자유도 아핀을 폐형식으로 맞추고 `orthogonalize`가 `R·c`로 쪼갠다.

이전 판본은 정육면체 회전 24개를 시드로 두고 점-표면 ICP를 돌린 뒤, 적합된 UV
회귀 기울기의 **부호**로 180° 대칭을 갈랐다. 그 경로가 레벨 1에서만 선 이유가
세 가지 있고 셋 다 여기서 고쳤다:

* `_umeyama`가 `d = ±1`로 det를 항상 +1로 만들어, 레벨 2~10의 참 변환
  (det = -0.0254³)을 **원리적으로 표현할 수 없었다.** `orthogonalize`는 det 보정을
  하지 않는다 — 반사가 살아남는다. 채택된 8개 레벨 전부 det < 0이다.
* 면 판별(`_thin_axis_index`/`_face_sign_by_triangle`)이 "마주보는 큰 면 두 개 +
  5% 테두리"라는 레벨 1 전용 형상(두께/최대 비 0.012인 얇은 평판)을 전제했고,
  방 모양 레벨에서는 착탄점의 72~77%를 버렸다. UV 조회에는 면 개념이 없다.
* 판별자였던 `EXPECTED_UV_SLOPE`(부호 있는 회귀 기울기)는 180° 대칭에 대해
  불변이다 — 모듈이 스스로 대수적으로 보였다. 대신 자유 파라미터가 0개인 절대
  등식 `UV_CONVENTION`(`obs = 2048 · mesh_uv`)을 쓴다.

**게이트가 이 모듈의 절반이다.** 검증 못 한 정렬을 넣는 것이 안 넣는 것보다 나쁘다
— 항목이 없으면 화면이 "없다"고 말하지만, 잘못된 항목은 권위 있어 보이는 틀린
기하를 그리고 하류에서 아무도 못 알아챈다. `GATES`가 문턱을, `verify_alignment`이
측정을, `gate_failures`가 판정을, `AlignmentRejected`가 거절을 맡는다. 문턱은 전부
참해의 실측 분포에서 잡았고 그 출처가 `GATES` 위에 적혀 있다.

**거리만으로 판정할 수 없다 — 그리고 그것은 자산마다 다르다.** 실측(채택 8개 레벨 ×
합성 변이 11개 = 88건): 통과한 변이는 0건이고, 8건은 **거리 게이트만으로는
통과한다**. 그 8건 중 6건이 레벨 8이다 — 그 방은 주축 180° 회전 셋과 주평면 반사
셋 전부에 대해 자기 자신으로 사상돼 표면 거리가 0.0002 mm로 남는다. 그 여섯 개를
갈라내는 것은 G2(절대 UV 잔차, 514~1037 텍셀) 하나뿐이다.

레벨 1의 착탄점은 정확히 평면이라(X 좌표가 7만 5천 프레임에서 비트 단위로 동일)
12자유도가 데이터로 결정되지 않는다(조건수 2.06e-09). `_svd3`의 퇴화 분기가 그
갈림길에서 U의 빠진 열을 외적으로 채워 **손잡이를 조용히 고르는** 자리이므로,
새 경로는 `orthogonalize`와 `solve_affine_12` 양쪽에서 그 분기에 닿기 전에 멈춘다.
레벨 1은 통과도 실패도 아닌 "unverifiable"로 기록된다.
"""

from __future__ import annotations

import itertools
import json
import math
import random
import struct
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Collection, Final, Mapping, Sequence

import rotation_math

Vec3 = tuple[float, float, float]
Mat3 = tuple[Vec3, Vec3, Vec3]

DATA_DIR: Final[Path] = Path(__file__).resolve().parents[1] / "data"
ALIGNMENT_PATH: Final[Path] = DATA_DIR / "level_alignment.json"

DEFAULT_MODELINGS_ROOT: Final[Path] = Path("/data/sources/modelings")
DEFAULT_PROFESSIONALS_ROOT: Final[Path] = Path("/data/sources/professionals")

# `solve_alignment`이 적합에 쓰는 표본과 최종 품질 검사에 쓰는 표본 크기. 스파이크의
# 사전등록 하한(≥10,000)과 실측치(15,000)를 그대로 쓴다.
DEFAULT_FIT_SAMPLE: Final = 1200
DEFAULT_EVAL_CAP: Final = 15000


# =============================================================================
# 바이너리 FBX 7400 리더 (struct + zlib만)
# =============================================================================


@dataclass
class _Node:
    """FBX 레코드 하나. name/props/children만 있으면 된다 — 트리 그대로 쓴다."""

    name: str
    props: list
    children: list

    def child(self, name: str) -> "_Node | None":
        for node in self.children:
            if node.name == name:
                return node
        return None

    def children_named(self, name: str) -> list["_Node"]:
        return [node for node in self.children if node.name == name]


def _read_property(f: BinaryIO) -> object:
    type_code = f.read(1)
    if not type_code:
        raise EOFError
    tc = type_code.decode("ascii")
    if tc == "Y":
        return struct.unpack("<h", f.read(2))[0]
    if tc == "C":
        return struct.unpack("<?", f.read(1))[0]
    if tc == "I":
        return struct.unpack("<i", f.read(4))[0]
    if tc == "L":
        return struct.unpack("<q", f.read(8))[0]
    if tc == "F":
        return struct.unpack("<f", f.read(4))[0]
    if tc == "D":
        return struct.unpack("<d", f.read(8))[0]
    if tc in ("f", "d", "l", "i", "b"):
        length, encoding, comp_len = struct.unpack("<III", f.read(12))
        elem_size = {"f": 4, "d": 8, "l": 8, "i": 4, "b": 1}[tc]
        fmt = {"f": "f", "d": "d", "l": "q", "i": "i", "b": "?"}[tc]
        if encoding == 0:
            data = f.read(length * elem_size)
        else:
            data = zlib.decompress(f.read(comp_len))
        return list(struct.unpack("<" + fmt * length, data))
    if tc in ("S", "R"):
        length = struct.unpack("<I", f.read(4))[0]
        data = f.read(length)
        if tc == "S":
            try:
                return data.decode("utf-8")
            except UnicodeDecodeError:
                return data
        return data
    raise ValueError(f"unknown FBX property type {tc!r} at offset {f.tell()}")


def _read_node(f: BinaryIO, is64: bool) -> "_Node | None":
    if is64:
        end_offset, num_props, prop_list_len = struct.unpack("<QQQ", f.read(24))
    else:
        end_offset, num_props, prop_list_len = struct.unpack("<III", f.read(12))
    name_len = struct.unpack("<B", f.read(1))[0]
    name = f.read(name_len).decode("ascii", errors="replace")
    if end_offset == 0 and num_props == 0 and prop_list_len == 0 and name_len == 0:
        return None  # null 레코드 — 형제 목록의 끝
    props = [_read_property(f) for _ in range(num_props)]
    children: list[_Node] = []
    sentinel_size = 25 if is64 else 13
    while f.tell() < end_offset - sentinel_size:
        child = _read_node(f, is64)
        if child is None:
            break
        children.append(child)
    f.seek(end_offset)
    return _Node(name=name, props=props, children=children)


def _parse_fbx(path: Path) -> tuple[list[_Node], int]:
    with path.open("rb") as f:
        header = f.read(27)
        if header[:21] != b"Kaydara FBX Binary  \x00":
            raise ValueError(f"{path}: not a binary FBX file (bad magic)")
        version = struct.unpack("<I", header[23:27])[0]
        is64 = version >= 7500
        roots: list[_Node] = []
        while True:
            try:
                node = _read_node(f, is64)
            except EOFError:
                break
            if node is None:
                break
            roots.append(node)
        return roots, version


def _find_all(node: _Node, name: str, out: list[_Node]) -> None:
    if node.name == name:
        out.append(node)
    for child in node.children:
        _find_all(child, name, out)


def _p70_dict(node: _Node) -> dict[str, list]:
    """`Properties70` 자식들을 이름 → 값 목록(`props[4:]`)으로."""
    p70 = node.child("Properties70")
    result: dict[str, list] = {}
    if p70 is None:
        return result
    for prop in p70.children_named("P"):
        result[prop.props[0]] = prop.props[4:]
    return result


def _triangulate(
    pvi: Sequence[int],
) -> tuple[list[tuple[int, int, int]], list[tuple[int, int, int]]]:
    """`PolygonVertexIndex`를 삼각형 부채꼴로 나눈다. (정점 인덱스 삼각형, 코너 인덱스
    삼각형)을 함께 낸다 — 후자가 `UVIndex`/`LayerElementNormal` 같은
    "폴리곤 정점당 하나"(`ByPolygonVertex`) 배열을 잘 자리다.

    각 다각형의 마지막 인덱스는 비트 반전(`~i`)으로 표시된다 — `-i - 1`로 되돌린다.
    코너 인덱스는 `PolygonVertexIndex`와 같은 길이·같은 순서를 도는 평평한 위치다
    (부호와 무관하게 항목마다 하나씩 늘어난다) — `UVIndex` 같은 배열이 그 위치로
    색인된다.
    """
    vertex_triangles: list[tuple[int, int, int]] = []
    corner_triangles: list[tuple[int, int, int]] = []
    current_v: list[int] = []
    current_c: list[int] = []
    corner = 0
    for raw in pvi:
        idx = -raw - 1 if raw < 0 else raw
        current_v.append(idx)
        current_c.append(corner)
        corner += 1
        if raw < 0:
            n = len(current_v)
            for i in range(1, n - 1):
                vertex_triangles.append((current_v[0], current_v[i], current_v[i + 1]))
                corner_triangles.append((current_c[0], current_c[i], current_c[i + 1]))
            current_v = []
            current_c = []
    return vertex_triangles, corner_triangles


# =============================================================================
# Mesh
# =============================================================================


@dataclass(frozen=True, slots=True)
class Mesh:
    """모델 공간 메시. `place`로 정렬을 적용하기 전까지는 FBX 파일 좌표 그대로다.

    `uv_indices[i]`는 `triangles[i]`의 세 코너 각각이 `uvs`(UV 풀) 어디를 가리키는지다
    — `triangle_uv`가 이걸로 무게중심 좌표에서 UV를 보간한다. 이게 없으면 `uvs`는
    떠 있는 값 목록일 뿐이고 어느 삼각형의 어느 UV인지 되짚을 수 없다(리뷰 B3) —
    UV 교차검증(B1/B2가 얼굴 판정에 쓰는 바로 그 검사)이 이 연결 없이는 불가능하다.
    비어 있으면(합성 테스트 메시 등) UV가 없다는 뜻이다.
    """

    vertices: list[Vec3]
    triangles: list[tuple[int, int, int]]
    uvs: list[tuple[float, float]]
    declared: dict[str, object]
    uv_indices: list[tuple[int, int, int]] = field(default_factory=list)


def triangle_uv(mesh: Mesh, triangle_index: int, barycentric: tuple[float, float, float]) -> tuple[float, float]:
    """`triangle_index`번 삼각형 위의 무게중심 좌표 `barycentric`에서 UV를 보간한다.

    `mesh.uv_indices`(B3)가 있어야 한다 — 없으면(합성 메시, 또는 UV 없는 파일)
    `ValueError`다.
    """
    if not mesh.uv_indices:
        raise ValueError("triangle_uv: mesh has no uv_indices (read_fbx found no LayerElementUV)")
    if not 0 <= triangle_index < len(mesh.uv_indices):
        raise ValueError(f"triangle_uv: triangle index {triangle_index} out of range")
    i0, i1, i2 = mesh.uv_indices[triangle_index]
    uv0, uv1, uv2 = mesh.uvs[i0], mesh.uvs[i1], mesh.uvs[i2]
    u, v, w = barycentric
    return (
        u * uv0[0] + v * uv1[0] + w * uv2[0],
        u * uv0[1] + v * uv1[1] + w * uv2[1],
    )


def read_fbx(path: Path) -> Mesh:
    """레벨 FBX 하나를 읽는다. 메시(Geometry) 하나, 모델 노드 하나를 가정한다.

    `declared`는 FBX가 스스로 선언한 것 그대로다 — 씬 배치를 위해 빠진 회전을
    채워 넣거나 하지 않는다. `solve_alignment`이 그 빈틈을 실측으로 채운다.
    """
    roots, _version = _parse_fbx(path)
    geometry_nodes: list[_Node] = []
    model_nodes: list[_Node] = []
    global_settings_nodes: list[_Node] = []
    for root in roots:
        _find_all(root, "Geometry", geometry_nodes)
        _find_all(root, "Model", model_nodes)
        _find_all(root, "GlobalSettings", global_settings_nodes)
    if len(geometry_nodes) != 1:
        raise ValueError(f"{path}: expected exactly one Geometry node, found {len(geometry_nodes)}")
    if len(model_nodes) != 1:
        raise ValueError(f"{path}: expected exactly one Model node, found {len(model_nodes)}")
    if len(global_settings_nodes) != 1:
        raise ValueError(
            f"{path}: expected exactly one GlobalSettings node, found {len(global_settings_nodes)}"
        )

    geometry = geometry_nodes[0]
    model = model_nodes[0]
    gs = _p70_dict(global_settings_nodes[0])
    m = _p70_dict(model)

    vertices_node = geometry.child("Vertices")
    if vertices_node is None:
        raise ValueError(f"{path}: Geometry has no Vertices")
    flat_v = vertices_node.props[0]
    vertices = [(flat_v[i], flat_v[i + 1], flat_v[i + 2]) for i in range(0, len(flat_v), 3)]

    pvi_node = geometry.child("PolygonVertexIndex")
    if pvi_node is not None:
        triangles, corner_triangles = _triangulate(pvi_node.props[0])
    else:
        triangles, corner_triangles = [], []

    uvs: list[tuple[float, float]] = []
    uv_indices: list[tuple[int, int, int]] = []
    layer_uv = geometry.child("LayerElementUV")
    if layer_uv is not None:
        # B3: `MappingInformationType`/`ReferenceInformationType`가 이 리더가 실제로
        # 검증한 값(레벨 1의 `ByPolygonVertex`/`IndexToDirect`)과 다르면, UVIndex를
        # 그대로 밀어붙이는 대신 멈춘다 — 다른 매핑에서는 `triangle_uv`의 코너→UV
        # 대응 규칙 자체가 달라진다.
        mapping_node = layer_uv.child("MappingInformationType")
        reference_node = layer_uv.child("ReferenceInformationType")
        mapping = mapping_node.props[0] if mapping_node is not None else None
        reference = reference_node.props[0] if reference_node is not None else None
        if mapping != "ByPolygonVertex" or reference != "IndexToDirect":
            raise NotImplementedError(
                f"{path}: LayerElementUV mapping/reference is {mapping!r}/{reference!r}, "
                "not ByPolygonVertex/IndexToDirect — not seen in the level fixtures this "
                "reader was built against"
            )
        uv_node = layer_uv.child("UV")
        uv_index_node = layer_uv.child("UVIndex")
        if uv_node is not None:
            flat_uv = uv_node.props[0]
            uvs = [(flat_uv[i], flat_uv[i + 1]) for i in range(0, len(flat_uv), 2)]
        if uv_index_node is not None and uvs:
            flat_index = uv_index_node.props[0]
            uv_indices = [
                (flat_index[c0], flat_index[c1], flat_index[c2])
                for c0, c1, c2 in corner_triangles
            ]

    model_name = ""
    if len(model.props) > 1 and isinstance(model.props[1], str):
        model_name = model.props[1].split("\x00")[0]

    declared: dict[str, object] = {
        # B5: `UpAxis`는 이 프로젝트의 "세 번째 독립 확인"이 읽는 값이다(모듈 리스트
        # 최상단, world_axes.UP_INDEX와 비교) — 없으면 기본값(1)이 우연히 항상 참인
        # `1 == 1` 검사가 된다. 그래서 여기서만 없으면 기본값을 쓰지 않고 멈춘다.
        # 나머지(PreRotation 등)는 FBX 스펙상 생략 = 항등이 실제 의미라 기본값이 정당하다.
        "UpAxis": _require_scalar(gs, "UpAxis", path, "GlobalSettings"),
        "UpAxisSign": int(_scalar(gs, "UpAxisSign", 1)),
        "FrontAxis": int(_scalar(gs, "FrontAxis", 2)),
        "CoordAxis": int(_scalar(gs, "CoordAxis", 0)),
        "UnitScaleFactor": float(_scalar(gs, "UnitScaleFactor", 1.0)),
        "PreRotation": _vec(m, "PreRotation"),
        "PostRotation": _vec(m, "PostRotation"),
        "LclRotation": _vec(m, "Lcl Rotation"),
        "LclTranslation": _vec(m, "Lcl Translation"),
        "LclScaling": _vec(m, "Lcl Scaling", (1.0, 1.0, 1.0)),
        "GeometricTranslation": _vec(m, "GeometricTranslation"),
        "GeometricRotation": _vec(m, "GeometricRotation"),
        "GeometricScaling": _vec(m, "GeometricScaling", (1.0, 1.0, 1.0)),
        "RotationOrder": int(_scalar(m, "RotationOrder", 0)),
        "ModelName": model_name,
    }

    return Mesh(
        vertices=vertices, triangles=triangles, uvs=uvs, declared=declared, uv_indices=uv_indices
    )


def _scalar(d: dict[str, list], key: str, default: object) -> object:
    return d[key][0] if key in d else default


def _require_scalar(d: dict[str, list], key: str, path: Path, node_name: str) -> int:
    """B5: 키가 없으면 기본값으로 조용히 채우지 않고 멈춘다.

    `read_fbx`가 `UpAxis`에 쓴다 — 세 번째 독립 확인(`world_axes.UP_INDEX`와 비교)이
    실제로 파일에서 읽은 값과 규약을 비교하는 것이지, 파이썬 기본값과 규약을
    비교하는(그래서 파일이 뭐라 하든 항상 참인) `1 == 1`이 아니게 하기 위함이다.
    """
    if key not in d:
        raise ValueError(f"{path}: {node_name}.{key} is missing — required, no silent default")
    return int(d[key][0])


def _vec(d: dict[str, list], key: str, default: Vec3 = (0.0, 0.0, 0.0)) -> Vec3:
    return (d[key][0], d[key][1], d[key][2]) if key in d else default


# =============================================================================
# 점 - 삼각형 최근접 거리 (Ericson, Real-Time Collision Detection §5.1.5)
# =============================================================================


def _sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _mul3(a: Vec3, k: float) -> Vec3:
    return (a[0] * k, a[1] * k, a[2] * k)


def _cross(a: Vec3, b: Vec3) -> Vec3:
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _closest_point_on_triangle(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> Vec3:
    ab = _sub(b, a)
    ac = _sub(c, a)
    ap = _sub(p, a)
    d1 = _dot(ab, ap)
    d2 = _dot(ac, ap)
    if d1 <= 0 and d2 <= 0:
        return a
    bp = _sub(p, b)
    d3 = _dot(ab, bp)
    d4 = _dot(ac, bp)
    if d3 >= 0 and d4 <= d3:
        return b
    vc = d1 * d4 - d3 * d2
    if vc <= 0 and d1 >= 0 and d3 <= 0:
        v = d1 / (d1 - d3)
        return _add(a, _mul3(ab, v))
    cp = _sub(p, c)
    d5 = _dot(ab, cp)
    d6 = _dot(ac, cp)
    if d6 >= 0 and d5 <= d6:
        return c
    vb = d5 * d2 - d1 * d6
    if vb <= 0 and d2 >= 0 and d6 <= 0:
        w = d2 / (d2 - d6)
        return _add(a, _mul3(ac, w))
    va = d3 * d6 - d5 * d4
    if va <= 0 and (d4 - d3) >= 0 and (d5 - d6) >= 0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return _add(b, _mul3(_sub(c, b), w))
    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom
    return _add(a, _add(_mul3(ab, v), _mul3(ac, w)))


def _world_triangles(mesh: Mesh) -> list[tuple[Vec3, Vec3, Vec3]]:
    return [(mesh.vertices[i], mesh.vertices[j], mesh.vertices[k]) for i, j, k in mesh.triangles]


def _nearest_on_mesh(
    p: Vec3, triangles: Sequence[tuple[Vec3, Vec3, Vec3]]
) -> tuple[float, Vec3 | None, int]:
    """triangles: (a,b,c) 세계좌표 삼각형 목록. (거리, 최근접점, 삼각형 인덱스)."""
    best_d2 = math.inf
    best_pt: Vec3 | None = None
    best_i = -1
    for i, (a, b, c) in enumerate(triangles):
        cp = _closest_point_on_triangle(p, a, b, c)
        diff = _sub(p, cp)
        d2 = _dot(diff, diff)
        if d2 < best_d2:
            best_d2 = d2
            best_pt = cp
            best_i = i
    return math.sqrt(best_d2) if best_d2 < math.inf else math.inf, best_pt, best_i


def closest_distance(mesh: Mesh, point: Vec3) -> float:
    """`mesh`(이미 배치된 좌표계)의 어떤 삼각형이든 `point`까지의 정확한 최근접 거리.

    정점 최근접이 아니다 — 면 한가운데 위 점은 어느 정점보다 가까울 수 있다.
    """
    world_tris = _world_triangles(mesh)
    if not world_tris:
        return math.inf
    distance, _point, _index = _nearest_on_mesh(point, world_tris)
    return distance


def _barycentric(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> tuple[float, float, float]:
    v0 = _sub(b, a)
    v1 = _sub(c, a)
    v2 = _sub(p, a)
    d00 = _dot(v0, v0)
    d01 = _dot(v0, v1)
    d11 = _dot(v1, v1)
    d20 = _dot(v2, v0)
    d21 = _dot(v2, v1)
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-15:
        return (1.0, 0.0, 0.0)
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return (u, v, w)


# =============================================================================
# 정렬: Alignment, place, load_alignment
# =============================================================================


# "solved"      게이트를 전부 통과해 실측으로 검증된 정렬이다. 화면이 그려도 된다.
# "declared"    FBX가 선언한 값을 그대로 읽은 것 — 검증되지 않았다.
# "unverifiable" 데이터가 검증을 받치지 못한다(예: 착탄점이 정확히 평면이라 12자유도가
#               결정되지 않는 레벨 1). **통과도 실패도 아닌 세 번째 상태**다 — 화면이
#               "검증된 정렬이 아직 없다"와 "이 레벨은 검증할 수 없다"를 다르게 말할 수
#               있어야 한다. 이 항목은 `solved_transform`을 싣지 않는다.
# 검증된 정렬이 **어느 경로로** 검증됐는지를 `provenance` 값 자체가 말한다. 2026-09-16
# 사전등록 §6이 요구한 형식이다: 12자유도 폐형식 경로는 그냥 "solved", 선형부 고정 +
# 총구 가림 경로는 `"solved:fixed_linear_muzzle_occlusion"`. 새 꼬리표는 여기 적어야만
# 쓸 수 있다 — 자유 문자열이면 다음 사람이 오타로 새 출처를 만든다.
SOLVED_VARIANTS: Final[frozenset[str]] = frozenset({
    "solved:fixed_linear_muzzle_occlusion",
    "solved:fixed_linear_uv_exclusion_meshonly",
})
VALID_PROVENANCE: Final[frozenset[str]] = (
    frozenset({"solved", "declared", "unverifiable"}) | SOLVED_VARIANTS
)


def is_solved(provenance: str) -> bool:
    """이 출처가 "게이트를 전부 통과했다"는 주장인가.

    `provenance == "solved"`로 직접 비교하면 꼬리표 붙은 출처를 조용히 미검증으로
    떨어뜨린다 — 소비자는 전부 이 함수를 거친다.
    """
    return provenance == "solved" or provenance in SOLVED_VARIANTS


@dataclass(frozen=True, slots=True)
class Alignment:
    """`world = scale * (rotation @ raw) + translation`.

    `provenance`가 "solved"면 `solve_alignment`이 실측 착탄점에 맞춰 푼 값이고,
    "declared"면 FBX가 선언한 값을 그대로 읽은 것이다. 두 주장은 다르다 — 레벨 1은
    선언된 변환이 3.68 m 어긋나므로 "solved"만 실제로 쓸 수 있다.

    B4: `provenance`가 이 필드의 존재 이유 전부다 — "맞춘 값"과 "읽은 값"이 뒤섞이면
    안 되므로, 생성 시점(`__post_init__`)에 `VALID_PROVENANCE` 밖의 값을 거절한다.
    `load_alignment`도 이 생성자를 거치므로 손상된 JSON도 여기서 걸린다.
    """

    level: int
    provenance: str
    rotation: tuple[float, ...]  # 9개, 행 우선(row-major) 3x3
    translation: Vec3
    scale: float
    declared_scale: float
    fit: dict[str, float]

    def __post_init__(self) -> None:
        if self.provenance not in VALID_PROVENANCE:
            raise ValueError(
                f"Alignment.provenance must be one of {sorted(VALID_PROVENANCE)}, "
                f"got {self.provenance!r}"
            )


def _rotation_matrix(alignment: Alignment) -> Mat3:
    r = alignment.rotation
    return ((r[0], r[1], r[2]), (r[3], r[4], r[5]), (r[6], r[7], r[8]))


def place(mesh: Mesh, alignment: Alignment) -> Mesh:
    """`alignment`의 회전+스케일+평행이동을 정점에 적용한 새 메시."""
    rot = _rotation_matrix(alignment)

    def transform(v: Vec3) -> Vec3:
        rv = rotation_math.matvec(rot, v)
        return (
            alignment.scale * rv[0] + alignment.translation[0],
            alignment.scale * rv[1] + alignment.translation[1],
            alignment.scale * rv[2] + alignment.translation[2],
        )

    return Mesh(
        vertices=[transform(v) for v in mesh.vertices],
        triangles=mesh.triangles,
        uvs=mesh.uvs,
        declared=mesh.declared,
        uv_indices=mesh.uv_indices,
    )


_NO_TRANSFORM: Final[dict] = {
    "rotation": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    "translation": [0.0, 0.0, 0.0],
    "scale": 0.0,
}


def _alignment_from_entry(entry: dict) -> Alignment:
    # "unverifiable" 항목은 변환을 싣지 않는다 — 검증 못 한 숫자를 적어 두면 다음
    # 사람이 그것을 쓴다. `scale=0.0`은 쓰려고 하면 즉시 무너지는 값이다.
    solved = entry.get("solved_transform", _NO_TRANSFORM)
    return Alignment(
        level=int(entry["level"]),
        provenance=entry["provenance"],
        rotation=tuple(float(x) for x in solved["rotation"]),
        translation=(
            float(solved["translation"][0]),
            float(solved["translation"][1]),
            float(solved["translation"][2]),
        ),
        scale=float(solved["scale"]),
        declared_scale=float(entry.get("declared_scale", 0.0)),
        fit=dict(entry.get("fit", {})),
    )


def load_alignment(level: int, path: Path = ALIGNMENT_PATH) -> Alignment:
    """`trainer/data/level_alignment.json`에서 레벨 하나의 정렬을 읽는다."""
    entries = json.loads(path.read_text(encoding="utf-8"))
    for entry in entries:
        if int(entry["level"]) == level:
            return _alignment_from_entry(entry)
    raise LookupError(f"no alignment recorded for level {level} in {path}")


def _entry_from_alignment(alignment: Alignment, source_fbx: str) -> dict:
    return {
        "level": alignment.level,
        "provenance": alignment.provenance,
        "solved_transform": {
            "rotation": list(alignment.rotation),
            "translation": list(alignment.translation),
            "scale": alignment.scale,
        },
        "declared_scale": alignment.declared_scale,
        "fit": alignment.fit,
        "source_fbx": source_fbx,
    }


# =============================================================================
# 선언된 변환 (FBX가 스스로 말하는 것 그대로 — solve_alignment의 초기값 후보에만 쓴다)
# =============================================================================


def _declared_rotation(declared: dict[str, object]) -> Mat3:
    """`PreRotation * Lcl Rotation * PostRotation⁻¹`. `RotationOrder`가 기본(XYZ)일 때만.

    레벨 1·10 둘 다 `RotationOrder`를 override하지 않는다(FBX SDK 기본 eEulerXYZ) —
    이 리더가 실제로 마주친 파일에서 벗어나면 조용히 잘못된 값을 내는 대신 멈춘다.
    """
    order = declared.get("RotationOrder", 0)
    if order != 0:
        raise NotImplementedError(
            f"RotationOrder {order} is not the default XYZ — not seen in the level "
            "fixtures this reader was built against"
        )
    r_pre = rotation_math.euler_matrix("XYZ", declared["PreRotation"])  # type: ignore[arg-type]
    r_lcl = rotation_math.euler_matrix("XYZ", declared["LclRotation"])  # type: ignore[arg-type]
    r_post = rotation_math.euler_matrix("XYZ", declared["PostRotation"])  # type: ignore[arg-type]
    r_post_inv = rotation_math.transpose(r_post)
    return rotation_math.matmul(rotation_math.matmul(r_pre, r_lcl), r_post_inv)


def _declared_scale(declared: dict[str, object]) -> float:
    return float(declared["UnitScaleFactor"]) / 100.0  # cm-per-unit -> m-per-unit


# =============================================================================
# Umeyama 유사변환 (stdlib 전용 3x3 SVD)
# =============================================================================


def _mat3_det(m: Mat3) -> float:
    return (
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    )


def _jacobi_eigen_sym3(
    a: Mat3, sweeps: int = 100, tol: float = 1e-14
) -> tuple[list[float], list[Vec3]]:
    """대칭 3x3 행렬의 고전 Jacobi 고유값 분해. 내림차순 (고유값, 고유벡터) 목록."""
    mat = [list(row) for row in a]
    vecs = [[1.0 if i == j else 0.0 for j in range(3)] for i in range(3)]
    for _ in range(sweeps):
        p, q, maxval = 0, 1, abs(mat[0][1])
        for i, j in ((0, 2), (1, 2)):
            if abs(mat[i][j]) > maxval:
                p, q, maxval = i, j, abs(mat[i][j])
        if maxval < tol:
            break
        app, aqq, apq = mat[p][p], mat[q][q], mat[p][q]
        if abs(apq) < 1e-300:
            continue
        phi = 0.5 * math.atan2(2 * apq, aqq - app)
        c, s = math.cos(phi), math.sin(phi)
        for k in range(3):
            akp, akq = mat[k][p], mat[k][q]
            mat[k][p] = c * akp - s * akq
            mat[k][q] = s * akp + c * akq
        for k in range(3):
            apk, aqk = mat[p][k], mat[q][k]
            mat[p][k] = c * apk - s * aqk
            mat[q][k] = s * apk + c * aqk
        for k in range(3):
            vkp, vkq = vecs[k][p], vecs[k][q]
            vecs[k][p] = c * vkp - s * vkq
            vecs[k][q] = s * vkp + c * vkq
    eigenvalues = [mat[i][i] for i in range(3)]
    eigenvectors = [(vecs[0][j], vecs[1][j], vecs[2][j]) for j in range(3)]
    order = sorted(range(3), key=lambda i: -eigenvalues[i])
    return [eigenvalues[i] for i in order], [eigenvectors[i] for i in order]


def _svd3(a: Mat3) -> tuple[Mat3, list[float], Mat3]:
    """일반 3x3 `a = U · diag(s) · Vt`. `s`는 내림차순.

    문턱은 **최대 특이값의 상대값**이다 — 절대 문턱(예: 1e-12)은 `a`가 노이즈 섞인
    유한표본 공분산 행렬일 때(참 계수가 3 미만이지만 가장 작은 특이값이 정확히 0이
    아니라 1e-6~1e-9 근처를 떠도는 경우) 불안정하다. 레벨 1의 착탄점이 정확히
    평면이라 이 경우가 실제로 걸린다(모듈 독스트링 참고).
    """
    at = rotation_math.transpose(a)
    ata = rotation_math.matmul(at, a)
    eigvals, eigvecs = _jacobi_eigen_sym3(ata)
    eigvals = [max(v, 0.0) for v in eigvals]
    s = [math.sqrt(v) for v in eigvals]
    s_thresh = max(s) * 1e-6 if max(s) > 0 else 1e-12

    v_cols = eigvecs
    u_cols: list[Vec3 | None] = []
    for i in range(3):
        v = v_cols[i]
        av = rotation_math.matvec(a, v)
        u_cols.append(_mul3(av, 1.0 / s[i]) if s[i] > s_thresh else None)

    n_missing = sum(1 for u in u_cols if u is None)
    if n_missing == 1:
        # 특이값 하나만 ~0: 대응하는 U 열은 (부호까지 제외하면) 나머지 둘의 외적으로
        # 강제된다 — 임의 선택이 아니다. "참" 특이값이 얼마나 작든 안정적이다.
        i = next(k for k, u in enumerate(u_cols) if u is None)
        others = [k for k in range(3) if k != i]
        c = _cross(u_cols[others[0]], u_cols[others[1]])  # type: ignore[arg-type]
        cn = math.sqrt(_dot(c, c))
        u_cols[i] = _mul3(c, 1.0 / cn) if cn > 1e-15 else (1.0, 0.0, 0.0)

    if any(u is None for u in u_cols):
        # 두 개 이상 비어 있는(퇴화가 더 심한) 경우: 표준축으로 정규직교 완성.
        basis = [u for u in u_cols if u is not None]
        candidates: list[Vec3] = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
        for i in range(3):
            if u_cols[i] is not None:
                continue
            for cand in candidates:
                w = cand
                for b in basis:
                    w = _sub(w, _mul3(b, _dot(w, b)))
                wn = math.sqrt(_dot(w, w))
                if wn > 1e-8:
                    w = _mul3(w, 1.0 / wn)
                    u_cols[i] = w
                    basis.append(w)
                    break

    u = tuple(tuple(u_cols[c][r] for c in range(3)) for r in range(3))  # type: ignore[misc]
    v = tuple(tuple(v_cols[c][r] for c in range(3)) for r in range(3))
    vt = rotation_math.transpose(v)  # type: ignore[arg-type]
    return u, s, vt  # type: ignore[return-value]


def _umeyama(points_p: Sequence[Vec3], points_q: Sequence[Vec3]) -> tuple[Mat3, float, Vec3]:
    """P를 Q로 옮기는 최소제곱 유사변환: `Q_i ~= c * R @ P_i + t`. 인덱스로 대응한다.

    Umeyama, 'Least-Squares Estimation of Transformation Parameters Between Two
    Point Patterns', PAMI 1991.

    **정렬 풀이 경로에서는 더 이상 쓰지 않는다.** `d = ±1`로 det를 항상 +1로 만들기
    때문에 레벨 2~10의 참 변환(det = -0.0254³)을 원리적으로 표현할 수 없다
    (모듈 독스트링). 반사가 필요한 곳은 `solve_affine_12` + `orthogonalize`를 쓴다.
    `pose_lab/skeleton.py`가 "여기서 내는 스케일 항을 쓰지 않는 이유"로 참조하므로
    함수는 남겨 둔다.
    """
    n = len(points_p)
    mu_p = tuple(sum(p[k] for p in points_p) / n for k in range(3))
    mu_q = tuple(sum(q[k] for q in points_q) / n for k in range(3))
    pc = [_sub(p, mu_p) for p in points_p]  # type: ignore[arg-type]
    qc = [_sub(q, mu_q) for q in points_q]  # type: ignore[arg-type]
    sigma_p2 = sum(_dot(p, p) for p in pc) / n

    sigma = [[0.0] * 3 for _ in range(3)]
    for p, q in zip(pc, qc):
        for r in range(3):
            for c in range(3):
                sigma[r][c] += q[r] * p[c]
    sigma_mat = tuple(tuple(x / n for x in row) for row in sigma)

    u, s, vt = _svd3(sigma_mat)  # type: ignore[arg-type]
    d = 1.0 if _mat3_det(u) * _mat3_det(vt) >= 0 else -1.0
    diag = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, d))
    rotation = rotation_math.matmul(rotation_math.matmul(u, diag), vt)

    trace_ds = s[0] + s[1] + d * s[2]
    scale = trace_ds / sigma_p2 if sigma_p2 > 1e-15 else 1.0

    r_mu_p = rotation_math.matvec(rotation, mu_p)  # type: ignore[arg-type]
    translation = tuple(mu_q[k] - scale * r_mu_p[k] for k in range(3))
    return rotation, scale, translation  # type: ignore[return-value]


def _apply_similarity(rotation: Mat3, scale: float, translation: Vec3, v: Vec3) -> Vec3:
    rv = rotation_math.matvec(rotation, v)
    return (
        scale * rv[0] + translation[0],
        scale * rv[1] + translation[1],
        scale * rv[2] + translation[2],
    )


# =============================================================================
# 후보 시드 회전: 정육면체 회전군(부호가 있는 축 치환, det=+1) 24개
# =============================================================================


def _cube_rotations() -> list[Mat3]:
    """씬 배치가 90° 단위일 것이라는 가정 아래 쓸 수 있는 24개 회전 시드.

    FBX 파일 자체는 이 씬 수준 회전을 기록하지 않는다(모듈 독스트링) — 그래서 값을
    추측하지 않고, 그럴듯한 후보를 넓게 두고 실측 착탄점으로 어느 것이 맞는지
    ICP로 고른다.
    """
    mats: list[Mat3] = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            m = [[0, 0, 0] for _ in range(3)]
            for row in range(3):
                m[row][perm[row]] = signs[row]
            det = (
                m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
                - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
            )
            if det == 1:
                mats.append(tuple(tuple(float(x) for x in row) for row in m))  # type: ignore[arg-type]
    return mats


# =============================================================================
# 점-표면 ICP: 각 착탄점을 메시 표면의 최근접점(어느 삼각형이든)에 대응시킨다
# =============================================================================


def _mean(points: Sequence[Vec3]) -> Vec3:
    n = len(points)
    return (
        sum(p[0] for p in points) / n,
        sum(p[1] for p in points) / n,
        sum(p[2] for p in points) / n,
    )


def _world_tris_for(
    vertices_raw: Sequence[Vec3],
    triangles_vidx: Sequence[tuple[int, int, int]],
    rotation: Mat3,
    scale: float,
    translation: Vec3,
) -> list[tuple[Vec3, Vec3, Vec3]]:
    return [
        (
            _apply_similarity(rotation, scale, translation, vertices_raw[i0]),
            _apply_similarity(rotation, scale, translation, vertices_raw[i1]),
            _apply_similarity(rotation, scale, translation, vertices_raw[i2]),
        )
        for i0, i1, i2 in triangles_vidx
    ]



# =============================================================================
# UV 규약 — 이름 붙인 가지 하나. 그 근거는 게이트 G0/G9가 매 빌드 다시 잰다.
# =============================================================================

UV_ATLAS_TEXELS: Final = 2048.0

# CSV의 `Ray Texture UV X/Y`(텍셀)와 메시 UV(정규화 [0,1])를 잇는 등식이다.
#
# **이것은 가지(branch)다 — 그 사실을 이름으로 남긴다.** 이전 구현은 같은 성격의
# 가지를 `EXPECTED_UV_SLOPE = -2048`로 고정했고, 그 상수를 지우면 다음 사람은
# 가지가 있었다는 사실 자체를 모르게 된다. 그래서 상수를 없애는 대신 방향을 바꿔
# 남긴다: 등식이 `obs = +2048 · mesh_uv`(옛 상수의 반대 부호)이고, 옛 상수와 달리
# 이 가지는 **매 빌드 측정으로 다시 확인된다**.
#
# 근거 1 (G0, 정렬과 무관): 관측 UV로 조회한 모델점들의 쌍거리 × 선언 스케일이
# 월드 착탄점의 쌍거리와 같은가. 적합도, 정렬도, 회전도 쓰지 않는다. 실측(커밋
# f632657, FBX 10개, 학습모드 CSV, 세션 17개, 착탄점 15,283~15,912, seed 0):
# 레벨 3~10에서 중앙값 0.00032~0.00081 mm, p95 0.00114~0.00245 mm.
# 등식이 틀리면 이 수가 0으로 남을 이유가 없다.
#
# 근거 2 (G9, 규약족 대조): `mesh_uv = K - obs/2048`(점반사)과 한 축만 뒤집는 두
# 족에 대해 K를 탐색해 다시 풀고, 그 해들이 전부 게이트에 걸리는 것을 확인한다.
# 레벨 8에는 실제로 거의 합동인 v축 반사 가지가 있다(K=(0, 0.925), G0 중앙값
# 0.00061 mm로 항등의 0.00064 mm보다 **좋다**) — 그러나 그 가지의 해는 참해와
# 90° 어긋나고(정점 이동 중앙값 1919 mm), G0의 p95가 2.05 mm(항등 0.0021 mm의
# 976배), G2가 267 텍셀, det 부호가 +1로 나머지 아홉 레벨과 반대다. 즉 **G0의
# 중앙값만으로는 이 가지를 못 가르고 p95·max와 G5가 가른다** — 그래서 G0는
# 중앙값만이 아니라 꼬리까지 게이트한다.
UV_CONVENTION: Final = "obs_texel = UV_ATLAS_TEXELS * mesh_uv"

# UV 삼각형의 면적이 이 아래면 UV 공간에 내부가 없다 — 조회의 후보가 될 수 없다.
# 그런 삼각형 위에 떨어진 착탄점은 "miss"도 "ambiguous"도 아닌 **다른 삼각형의
# 깨끗한 hit**으로 조용히 잘못 대응된다. `audit_atlas`가 그 오대응 거리를 잰다.
UV_DEGENERATE_AREA: Final = 1e-12

# UV 삼각형 내부 판정의 시임 허용치(정규화 UV 단위). 2048 텍셀 아틀라스에서 0.2텍셀.
UV_LOOKUP_TOL: Final = 1e-4

# 조회 후보가 둘 이상일 때, 그 후보들이 가리키는 **모델 공간 점**이 이 안에서
# 같으면 모호가 아니다(UV 변을 공유하는 시임의 양쪽). 모델 단위 1e-4 ≈ 0.0025 mm.
UV_SEAM_SAME_POINT: Final = 1e-4

# G2에서 "최근접과 사실상 같은 거리"로 볼 범위(월드 미터). 착탄점이 UV 시임 위에
# 떨어지면 변을 공유하는 두 삼각형 중 어느 쪽으로 읽어도 **같은 3D 점**이지만
# 보간 UV는 섬이 달라 수백 텍셀 차이난다(선행 조사 실측: 레벨 6·7·9에서 450.7 /
# 120.3 / 358.7 텍셀). 그 자리에서 잔차의 **최소**를 취한다 — 그러지 않으면 옳은
# 정렬에서도 G2의 max가 450 텍셀로 튀어 유일한 포착자가 잡음 바닥을 갖게 된다.
G2_TIE_EPS_M: Final = 2.0e-4


# =============================================================================
# 게이트 문턱 — 전부 실측 분포에서 잡았다
# =============================================================================

# 출처: 커밋 f632657, FBX 10개(`/data/sources/modelings`), 학습모드 CSV 전량에서
# 세션 17개를 가로질러 모은 착탄점(레벨당 15,283~15,912), seed 0, 적합 2,000 /
# 평가 1,500(겹치지 않는 한 순열에서 자름). 참해(레벨 3~10)의 실측 최악값:
#
#   G0 중앙값 0.00081 / p95 0.00245 / max 0.00494 mm
#   G3 중앙값 0.00030 / p95 0.00072 / max 0.00087 mm
#   G2 p95 0.00063 / max 0.00087 텍셀
#   비등방성 7.43e-07,  |c/declared - 1| 7.54e-07,  조건수 0.2264
#
# 문턱은 그 최악값의 약 20~60배 위에 둔다. **이전 제안의 문턱(G2 1텍셀 / G3 2mm /
# aniso 1e-3)은 참해보다 네 자릿수 헐거워** 그 빈 공간에 179.9998° 뒤집힌 레벨 8
# 쌍둥이(G3 중앙값 0.199 mm, aniso 3.85e-4)가 통째로 들어앉는 것이 실증됐다.
# 지금 문턱에서 그 쌍둥이는 G3에서 40배, G4에서 38배 차이로 떨어진다.
GATES: Final[dict[str, float]] = {
    # G0 — 정렬 무관 합동성(대응 모델점 쌍거리 × 선언 스케일 대 월드 쌍거리)
    "g0_median_mm": 0.01,
    "g0_p95_mm": 0.05,
    "g0_max_mm": 0.2,
    # G1 — 아틀라스 단사성. 앞의 둘은 메시만 보고(정렬·착탄점 무관), 뒤의 둘은
    # 착탄점 **전량**에서 잰다(부분표본에서 재면 판정을 시드가 정한다).
    "uv_degenerate_misattribution_mm": 1.0,
    "atlas_overlap_max_gap_mm": 1.0,
    "uv_miss_fraction": 0.002,
    "uv_ambiguous_fraction": 0.002,
    # G2 — 절대 UV 잔차(3D 최근접 삼각형으로 찾고, 시임에서는 최소를 취한다)
    "uv_abs_resid_p95_texel": 0.01,
    "uv_abs_resid_max_texel": 0.05,
    # G3 — 표면 거리
    "median_mm": 0.005,
    "p95_mm": 0.02,
    "max_mm": 0.05,
    # G4 — 등방성 + 선언 스케일. 두 개의 등방성을 따로 잰다:
    #  `fit_linear_aniso`  해를 낸 **그 적합**의 선형부 — 대응이 틀리면(예: 뒤집힌 UV
    #                      규약) 적합이 전단으로 끌려가 여기서 걸린다. 실측: 참해
    #                      ≤ 6.3e-07, 규약족 경쟁자 1.5e-04 ~ 2.6e-03.
    #  `linear_aniso`      기록된 정렬을 **다시 적합 없이** 재검사할 때, 주어진 착탄점의
    #                      대응으로 12자유도를 다시 세워 잰 값. 정렬에는 무관하다
    #                      (같은 레벨의 어떤 변이에도 같은 값이 나온다) — 그래서
    #                      이것은 "대응이 유사변환을 허용하는가"의 검사이지 정렬 검사가
    #                      아니다. 그 사실을 이름과 문턱 양쪽에 남긴다.
    "fit_linear_aniso": 1.0e-5,
    "linear_aniso": 1.0e-5,
    "scale_ratio_to_declared": 1.0e-5,
    # G7 — 기하 조건수 / 법선 다양성
    "cond_min_over_max": 0.05,
    "distinct_normals": 2,
    # G8 — 적대적 대조에서 2등이 항등보다 이만큼은 나빠야 한다(텍셀)
    "adversarial_runner_up_texel": 10.0,
}


class AlignmentRejected(Exception):
    """게이트를 통과하지 못한 레벨. 항목을 만들지 않는다 — 화면이 "없다"고 말한다.

    `failures`에 (게이트 이름, 잰 값, 문턱) 세 쌍이 들어간다. 문구만 남기면 나중에
    "확인함"이 되고, 숫자가 남으면 다음 사람이 다시 잴 수 있다.
    """

    def __init__(self, level: int, failures: Sequence[tuple[str, object, str]]) -> None:
        self.level = level
        self.failures = list(failures)
        detail = "; ".join(f"{name}={value!r} (need {need})" for name, value, need in failures)
        super().__init__(f"level {level} rejected by {len(failures)} gate(s): {detail}")


# =============================================================================
# UV 색인 — 정렬 이전에 성립한다(메시만 본다)
# =============================================================================


def _tri_uv_corners(mesh: Mesh, triangle_index: int) -> tuple[tuple[float, float], ...]:
    i0, i1, i2 = mesh.uv_indices[triangle_index]
    return (mesh.uvs[i0], mesh.uvs[i1], mesh.uvs[i2])


def _tri_vertices(mesh: Mesh, triangle_index: int) -> tuple[Vec3, Vec3, Vec3]:
    i0, i1, i2 = mesh.triangles[triangle_index]
    return (mesh.vertices[i0], mesh.vertices[i1], mesh.vertices[i2])


def _uv_area(corners: Sequence[tuple[float, float]]) -> float:
    (x1, y1), (x2, y2), (x3, y3) = corners
    return abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)) / 2.0


def _tri_area3(tri: tuple[Vec3, Vec3, Vec3]) -> float:
    a, b, c = tri
    cr = _cross(_sub(b, a), _sub(c, a))
    return math.sqrt(_dot(cr, cr)) / 2.0


def _tri_normal(tri: tuple[Vec3, Vec3, Vec3]) -> Vec3 | None:
    a, b, c = tri
    cr = _cross(_sub(b, a), _sub(c, a))
    n = math.sqrt(_dot(cr, cr))
    return None if n <= 0 else _mul3(cr, 1.0 / n)


def _barycentric_2d(
    p: tuple[float, float], corners: Sequence[tuple[float, float]]
) -> tuple[float, float, float] | None:
    (x1, y1), (x2, y2), (x3, y3) = corners
    px, py = p
    den = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
    if abs(den) < 1e-18:
        # UV 퇴화 삼각형. **분모를 클램프해서 밀어붙이지 않는다** — 클램프하면 그
        # 삼각형이 (1,0,0)으로 모든 질의점을 "포함"하게 되고(레벨 2에서 모호도가
        # 1.0000이 된다), nan으로 흘려보내면 비교가 전부 False가 되어 조용히
        # 사라진다. 두 갈래가 같은 명세에서 정반대 판정을 낸다. 그래서 여기서
        # **명시적으로 None**을 내고, 색인은 애초에 이런 삼각형을 넣지 않으며
        # (`build_uv_index`), 그 사실을 `audit_atlas`가 따로 잰다.
        return None
    a = ((y2 - y3) * (px - x3) + (x3 - x2) * (py - y3)) / den
    b = ((y3 - y1) * (px - x3) + (x1 - x3) * (py - y3)) / den
    return (a, b, 1.0 - a - b)


@dataclass(frozen=True, slots=True)
class UvIndex:
    """UV 공간 [0,1]²의 균일 격자 버킷 색인. **정렬 이전에 성립한다.**

    `degenerate_triangles`는 UV 면적이 `UV_DEGENERATE_AREA` 이하라 색인에 들어가지
    못한 삼각형들이다 — 조회는 이들을 절대 후보로 내지 않는다(고정된 정책).
    """

    mesh: Mesh
    buckets: dict[tuple[int, int], list[int]]
    resolution: int
    degenerate_triangles: tuple[int, ...]


def build_uv_index(mesh: Mesh, resolution: int = 64) -> UvIndex:
    if not mesh.uv_indices:
        raise ValueError("build_uv_index: mesh has no uv_indices (read_fbx found no LayerElementUV)")
    buckets: dict[tuple[int, int], list[int]] = {}
    degenerate: list[int] = []
    for t in range(len(mesh.triangles)):
        corners = _tri_uv_corners(mesh, t)
        if _uv_area(corners) <= UV_DEGENERATE_AREA:
            degenerate.append(t)
            continue
        us = [c[0] for c in corners]
        vs = [c[1] for c in corners]
        i0 = max(0, int(min(us) * resolution))
        i1 = min(resolution - 1, int(max(us) * resolution))
        j0 = max(0, int(min(vs) * resolution))
        j1 = min(resolution - 1, int(max(vs) * resolution))
        for i in range(i0, i1 + 1):
            for j in range(j0, j1 + 1):
                buckets.setdefault((i, j), []).append(t)
    return UvIndex(
        mesh=mesh, buckets=buckets, resolution=resolution, degenerate_triangles=tuple(degenerate)
    )


def uv_lookup(
    index: UvIndex, uv_normalized: tuple[float, float], tol: float = UV_LOOKUP_TOL
) -> tuple[str, int | None, tuple[float, float, float] | None]:
    """정규화 UV 하나를 삼각형+무게중심으로 되돌린다.

    `("hit", triangle_index, barycentric)` / `("miss", None, None)` /
    `("ambiguous", n_candidates, None)`. 후보가 둘 이상이어도 그 후보들이 가리키는
    모델 점이 `UV_SEAM_SAME_POINT` 안에서 같으면(변을 공유하는 시임) 모호가 아니다.
    """
    mesh = index.mesh
    u, v = uv_normalized
    i = min(index.resolution - 1, max(0, int(u * index.resolution)))
    j = min(index.resolution - 1, max(0, int(v * index.resolution)))
    candidates: list[tuple[int, tuple[float, float, float]]] = []
    for t in index.buckets.get((i, j), ()):
        bary = _barycentric_2d((u, v), _tri_uv_corners(mesh, t))
        if bary is None:
            continue
        if min(bary) >= -tol:
            candidates.append((t, bary))
    if not candidates:
        return ("miss", None, None)
    if len(candidates) > 1:
        points = [model_point_for_hit(mesh, t, bary) for t, bary in candidates]
        spread = max(math.dist(points[0], q) for q in points)
        if spread > UV_SEAM_SAME_POINT:
            return ("ambiguous", len(candidates), None)
    return ("hit", candidates[0][0], candidates[0][1])


def model_point_for_hit(
    mesh: Mesh, triangle_index: int, barycentric: tuple[float, float, float]
) -> Vec3:
    """`triangle_uv`와 정확히 짝을 이루는 역방향: 무게중심 → **모델 공간** 점."""
    a, b, c = _tri_vertices(mesh, triangle_index)
    u, v, w = barycentric
    return (
        u * a[0] + v * b[0] + w * c[0],
        u * a[1] + v * b[1] + w * c[1],
        u * a[2] + v * b[2] + w * c[2],
    )


# =============================================================================
# 폐형식 12자유도 아핀 + 직교화 (반사를 죽이지 않는다)
# =============================================================================


def solve_affine_12(src: Sequence[Vec3], dst: Sequence[Vec3]) -> tuple[Mat3, Vec3]:
    """`dst ≈ A · src + b`의 최소제곱 해. 4x4 Gram 정규방정식을 가우스 소거로 푼다.

    조건이 모자라면 **조용히 릿지를 넣지 않고 `ValueError`로 멈춘다** — 레벨 1의
    착탄점은 정확히 평면이라 여기서 실제로 걸린다(그리고 걸려야 한다).
    """
    gram = [[0.0] * 4 for _ in range(4)]
    rhs = [[0.0] * 3 for _ in range(4)]
    for p, q in zip(src, dst):
        x = (p[0], p[1], p[2], 1.0)
        for a in range(4):
            xa = x[a]
            for b in range(4):
                gram[a][b] += xa * x[b]
            for d in range(3):
                rhs[a][d] += xa * q[d]
    magnitude = max(abs(gram[r][c]) for r in range(4) for c in range(4))
    if magnitude <= 0.0:
        raise ValueError("solve_affine_12: empty or zero correspondence set")
    m = [gram[r][:] + rhs[r][:] for r in range(4)]
    for c in range(4):
        pivot_row = max(range(c, 4), key=lambda r: abs(m[r][c]))
        if abs(m[pivot_row][c]) < 1e-10 * magnitude:
            raise ValueError(
                "solve_affine_12: normal equations are rank deficient "
                f"(pivot {abs(m[pivot_row][c]):.3e} vs magnitude {magnitude:.3e}) — "
                "the correspondence cloud does not span three dimensions"
            )
        m[c], m[pivot_row] = m[pivot_row], m[c]
        pv = m[c][c]
        m[c] = [x / pv for x in m[c]]
        for r in range(4):
            if r == c:
                continue
            f = m[r][c]
            if f:
                m[r] = [a - f * b for a, b in zip(m[r], m[c])]
    linear: Mat3 = tuple(tuple(m[r][4 + d] for r in range(3)) for d in range(3))  # type: ignore[assignment]
    offset: Vec3 = tuple(m[3][4 + d] for d in range(3))  # type: ignore[assignment]
    return linear, offset


def orthogonalize(linear: Mat3) -> tuple[Mat3, float, float, float, list[float]]:
    """3x3 선형부를 `R · c`로 쪼갠다. **det 보정을 하지 않는다** — 반사가 살아남는다.

    반환: `(rotation, scale, aniso, det_sign, singular_values)`.

    `_svd3`의 퇴화 분기(특이값 하나가 0일 때 U의 빠진 열을 외적으로 채우는 자리)는
    **손잡이를 조용히 고른다**. 그 분기가 새 경로에서 절대 걸리지 않도록, 특이값
    비가 1e-6 아래면 여기서 멈춘다.
    """
    u, s, vt = _svd3(linear)
    if max(s) <= 0.0 or min(s) / max(s) < 1e-6:
        raise ValueError(
            f"orthogonalize: linear part is near-singular (singular values {s}) — "
            "refusing the degenerate branch of _svd3, which would pick a handedness silently"
        )
    scale = sum(s) / 3.0
    aniso = (max(s) - min(s)) / scale
    rotation = rotation_math.matmul(u, vt)
    return rotation, scale, aniso, (1.0 if _mat3_det(rotation) > 0 else -1.0), list(s)


def _percentiles(values: Sequence[float]) -> tuple[float, float, float]:
    s = sorted(values)
    n = len(s)
    if n == 0:
        return (math.inf, math.inf, math.inf)

    def at(p: float) -> float:
        return s[min(n - 1, int(round(p / 100.0 * (n - 1))))]

    return (at(50), at(95), at(100))


# =============================================================================
# G0 — 정렬 무관 합동성. `UV_CONVENTION`을 뒷받침하는 유일한 비순환 증거다.
# =============================================================================


def congruence_stats(
    model_points: Sequence[Vec3],
    world_points: Sequence[Vec3],
    declared_scale: float,
    *,
    seed: int = 0,
    n_pairs: int = 4000,
) -> dict[str, float]:
    """`|p_i - p_j| · declared_scale` 대 `|q_i - q_j|`. 적합도 정렬도 쓰지 않는다."""
    n = len(model_points)
    if n < 2:
        return {"g0_n_pairs": 0.0, "g0_median_mm": math.inf, "g0_p95_mm": math.inf, "g0_max_mm": math.inf}
    rng = random.Random(seed)
    errors: list[float] = []
    for _ in range(n_pairs):
        i = rng.randrange(n)
        j = rng.randrange(n)
        if i == j:
            continue
        model_d = math.dist(model_points[i], model_points[j]) * declared_scale
        world_d = math.dist(world_points[i], world_points[j])
        errors.append(abs(model_d - world_d) * 1000.0)
    med, p95, mx = _percentiles(errors)
    return {
        "g0_n_pairs": float(len(errors)),
        "g0_median_mm": med,
        "g0_p95_mm": p95,
        "g0_max_mm": mx,
    }


# =============================================================================
# G1 — 아틀라스 단사성 감사. 메시만 본다(정렬·착탄점과 무관).
# =============================================================================


def _atlas_overlap_cells(
    mesh: Mesh, index: UvIndex, overlap_resolution: int
) -> dict[tuple[int, int], list[tuple[int, tuple[float, float, float]]]]:
    """UV 아틀라스를 `overlap_resolution` 격자로 래스터화해 셀 중심을 덮는 비퇴화
    삼각형을 모은다. `audit_atlas`의 겹침 검사와 **같은 코드**다 — 2026-09-16 사전등록
    §2.3(E3)이 "기존 `audit_atlas`의 겹침 검사"를 쓴다고 못박았으므로, 두 곳이 갈라지지
    않도록 여기 한 곳에만 둔다."""
    cells: dict[tuple[int, int], list[tuple[int, tuple[float, float, float]]]] = {}
    indexed = set(range(len(mesh.triangles))) - set(index.degenerate_triangles)
    for t in sorted(indexed):
        corners = _tri_uv_corners(mesh, t)
        us = [q[0] for q in corners]
        vs = [q[1] for q in corners]
        i0 = max(0, int(min(us) * overlap_resolution))
        i1 = min(overlap_resolution - 1, int(max(us) * overlap_resolution))
        j0 = max(0, int(min(vs) * overlap_resolution))
        j1 = min(overlap_resolution - 1, int(max(vs) * overlap_resolution))
        for i in range(i0, i1 + 1):
            for j in range(j0, j1 + 1):
                centre = ((i + 0.5) / overlap_resolution, (j + 0.5) / overlap_resolution)
                bary = _barycentric_2d(centre, corners)
                if bary is None or min(bary) < 0.0:
                    continue
                cells.setdefault((i, j), []).append((t, bary))
    return cells


def audit_atlas(
    mesh: Mesh, index: UvIndex, *, seed: int = 0, samples_per_triangle: int = 64,
    overlap_resolution: int = 1024,
) -> dict[str, float]:
    """UV 아틀라스가 3D 표면 위에서 실제로 단사인지 잰다.

    **착탄점 가중 통계가 아니다.** `uv_miss_fraction`/`uv_ambiguous_fraction`은
    UV 내부가 없는 삼각형을 **원리적으로** 볼 수 없다 — 그런 삼각형은 조회 후보가
    될 수 없으므로, 그 위에 떨어진 점은 miss도 ambiguous도 아닌 다른 삼각형의
    깨끗한 hit으로 돌아온다. 그래서 여기서는 퇴화 삼각형 위를 직접 표집해
    **오대응 거리(mm)**를 잰다. 레벨 2가 이 검사에서 떨어진다(134 mm).
    """
    rng = random.Random(seed)
    scale = _declared_scale(mesh.declared)
    n_tris = len(mesh.triangles)
    areas = [_tri_area3(_tri_vertices(mesh, t)) for t in range(n_tris)]
    total_area = sum(areas)
    degenerate = [t for t in index.degenerate_triangles if areas[t] > 1e-9]
    degenerate_area = sum(areas[t] for t in degenerate)

    worst_gap = 0.0
    n_silent = 0
    for t in degenerate:
        a, b, c = _tri_vertices(mesh, t)
        uva, uvb, uvc = _tri_uv_corners(mesh, t)
        for _ in range(samples_per_triangle):
            r1, r2 = rng.random(), rng.random()
            if r1 + r2 > 1.0:
                r1, r2 = 1.0 - r1, 1.0 - r2
            w = (1.0 - r1 - r2, r1, r2)
            true_point = tuple(w[0] * a[d] + w[1] * b[d] + w[2] * c[d] for d in range(3))
            uv = (
                w[0] * uva[0] + w[1] * uvb[0] + w[2] * uvc[0],
                w[0] * uva[1] + w[1] * uvb[1] + w[2] * uvc[1],
            )
            kind, ti, bary = uv_lookup(index, uv)
            if kind == "hit":
                n_silent += 1
                worst_gap = max(worst_gap, math.dist(true_point, model_point_for_hit(mesh, ti, bary)))  # type: ignore[arg-type]

    # 비퇴화 삼각형끼리의 겹침을 아틀라스 래스터화로 확인한다 — 분사 위치와 무관하다.
    cells = _atlas_overlap_cells(mesh, index, overlap_resolution)
    overlapping = 0
    overlap_gap = 0.0
    for hits_in_cell in cells.values():
        if len(hits_in_cell) < 2:
            continue
        points = [model_point_for_hit(mesh, t, bary) for t, bary in hits_in_cell]
        gap = max(math.dist(points[0], q) for q in points)
        if gap > UV_SEAM_SAME_POINT:
            overlapping += 1
            overlap_gap = max(overlap_gap, gap)

    return {
        "uv_degenerate_tris": float(len(degenerate)),
        "uv_degenerate_area_fraction": (degenerate_area / total_area) if total_area else 0.0,
        "uv_degenerate_silent_samples": float(n_silent),
        "uv_degenerate_misattribution_mm": worst_gap * scale * 1000.0,
        "atlas_overlap_cell_fraction": (overlapping / len(cells)) if cells else 0.0,
        "atlas_overlap_max_gap_mm": overlap_gap * scale * 1000.0,
        "atlas_overlap_resolution": float(overlap_resolution),
    }


# =============================================================================
# G2 — 절대 UV 잔차. 삼각형은 **3D 최근접**으로 찾는다(UV 조회로 찾으면 순환이다).
# =============================================================================


def uv_absolute_residuals(
    placed_triangles: Sequence[tuple[Vec3, Vec3, Vec3]],
    mesh: Mesh,
    hits: Sequence[Vec3],
    hit_uvs: Sequence[tuple[float, float]],
    *,
    tie_eps_m: float = G2_TIE_EPS_M,
) -> tuple[list[float], list[float], list[int], int]:
    """`|obs - UV_ATLAS_TEXELS · mesh_uv|`를 평가 착탄점마다 잰다.

    삼각형은 배치된 메시에서 **3D 거리로** 고른다. UV 조회로 고르면
    `obs = 2048 · mesh_uv`가 구성상 참이 되어 실패할 수 없는 단언이 된다 —
    그 금지를 테스트가 지킨다(`test_level_mesh.py`의 순환 방지 검사).

    반환: `(거리 m, 잔차 텍셀, 최근접 삼각형 인덱스, 시임 동점 개수)`.
    """
    distances, residuals, triangle_ids, ties, _excluded = uv_absolute_residuals_excluding(
        placed_triangles, mesh, hits, hit_uvs, tie_eps_m=tie_eps_m
    )
    return distances, residuals, triangle_ids, ties


def uv_absolute_residuals_excluding(
    placed_triangles: Sequence[tuple[Vec3, Vec3, Vec3]],
    mesh: Mesh,
    hits: Sequence[Vec3],
    hit_uvs: Sequence[tuple[float, float]],
    *,
    tie_eps_m: float = G2_TIE_EPS_M,
    excluded_triangles: Collection[int] = (),
) -> tuple[list[float], list[float], list[int], int, list[bool]]:
    """`uv_absolute_residuals`와 같은 계산에 **메시 전용 제외 규칙**을 얹은 것.

    `excluded_triangles`(2026-09-16 사전등록 §2의 E1·E2·E3 합집합)는 **3D 최근접 귀속의
    후보에서 빠진다.** 최근접 탐색 자체는 메시 전량에서 한다 — 그래야 어느 착탄점이
    제외된 면 위에 떨어졌는지 셀 수 있다. 동률(시임) 후보 중 남은 것이 하나라도 있으면
    그것들로 잔차를 재고, 하나도 없으면 그 착탄점은 **제외 착탄점**이다(다섯 번째 반환값
    `True`). 제외 착탄점의 잔차도 제외 없이 계산해 그대로 실어 보낸다 — 무엇을 가렸는지
    따로 보고해야 하기 때문이다(§2.4).

    거리(G3)는 제외와 무관하게 **전량** 그대로다: 제외는 UV 잔차에만 미친다.
    """
    excluded_set = frozenset(excluded_triangles)
    distances: list[float] = []
    residuals: list[float] = []
    triangle_ids: list[int] = []
    excluded_hits: list[bool] = []
    ties = 0
    for point, obs in zip(hits, hit_uvs):
        per: list[tuple[float, int, Vec3]] = []
        best = math.inf
        for i, (a, b, c) in enumerate(placed_triangles):
            cp = _closest_point_on_triangle(point, a, b, c)
            d = math.dist(point, cp)
            per.append((d, i, cp))
            if d < best:
                best = d
        near = [row for row in per if row[0] <= best + tie_eps_m]
        if len(near) > 1:
            ties += 1
        kept = [row for row in near if row[1] not in excluded_set]
        is_excluded = not kept
        use = near if is_excluded else kept
        residual = math.inf
        for _d, i, cp in use:
            bary = _barycentric(cp, *placed_triangles[i])
            mesh_u, mesh_v = triangle_uv(mesh, i, bary)
            residual = min(
                residual,
                math.hypot(
                    obs[0] - UV_ATLAS_TEXELS * mesh_u, obs[1] - UV_ATLAS_TEXELS * mesh_v
                ),
            )
        distances.append(best)
        residuals.append(residual)
        triangle_ids.append(near[0][1])
        excluded_hits.append(is_excluded)
    return distances, residuals, triangle_ids, ties, excluded_hits


def _correspondences(
    mesh: Mesh,
    index: UvIndex,
    hits: Sequence[Vec3],
    hit_uvs: Sequence[tuple[float, float]],
    convention: tuple[str, tuple[float, float]] = ("identity", (0.0, 0.0)),
) -> tuple[list[Vec3], list[Vec3], float, float]:
    """관측 UV를 `convention`으로 읽어 (모델점, 월드점) 대응을 만든다.

    기본값은 `UV_CONVENTION`이 고른 가지다. 다른 값은 **G9(규약족 대조)만** 쓴다 —
    `verify_alignment`은 언제나 기본 가지로 재므로, 뒤집힌 규약에서 나온 해는
    G2에서 자기 규약이 아니라 **이름 붙은 규약**으로 평가받는다.
    """
    model_points: list[Vec3] = []
    world_points: list[Vec3] = []
    miss = 0
    ambiguous = 0
    family, k = convention
    for point, obs in zip(hits, hit_uvs):
        kind, ti, bary = uv_lookup(index, _apply_convention(family, k, obs))
        if kind == "miss":
            miss += 1
            continue
        if kind == "ambiguous":
            ambiguous += 1
            continue
        model_points.append(model_point_for_hit(mesh, ti, bary))  # type: ignore[arg-type]
        world_points.append(point)
    n = max(1, len(hits))
    return model_points, world_points, miss / n, ambiguous / n


def _conditioning(points: Sequence[Vec3]) -> tuple[float, list[float]]:
    n = len(points)
    if n < 3:
        return 0.0, [0.0, 0.0, 0.0]
    mu = _mean(points)
    cov = [[0.0] * 3 for _ in range(3)]
    for p in points:
        q = _sub(p, mu)
        for a in range(3):
            for b in range(3):
                cov[a][b] += q[a] * q[b]
    cov_mat: Mat3 = tuple(tuple(x / n for x in row) for row in cov)  # type: ignore[assignment]
    eigenvalues, _vectors = _jacobi_eigen_sym3(cov_mat)
    sing = [math.sqrt(max(v, 0.0)) for v in eigenvalues]
    return (sing[2] / sing[0] if sing[0] > 0 else 0.0), sing


def _normal_diversity(mesh: Mesh, triangle_ids: Sequence[int], min_share: float = 0.05) -> int:
    """평가 착탄점의 `min_share` 이상을 각각 담는, 서로 다른 면 법선의 개수."""
    counts: dict[int, int] = {}
    for t in triangle_ids:
        counts[t] = counts.get(t, 0) + 1
    clusters: list[tuple[Vec3, int]] = []
    for t, count in counts.items():
        normal = _tri_normal(_tri_vertices(mesh, t))
        if normal is None:
            continue
        for k, (existing, total) in enumerate(clusters):
            if abs(_dot(existing, normal)) > 0.99:
                clusters[k] = (existing, total + count)
                break
        else:
            clusters.append((normal, count))
    total_hits = max(1, len(triangle_ids))
    return sum(1 for _n, count in clusters if count / total_hits >= min_share)


# =============================================================================
# verify_alignment — 게이트 전부를 계산하는 독립 함수. 재적합하지 않는다.
# =============================================================================


def verify_alignment(
    mesh: Mesh,
    alignment: Alignment,
    hits: Sequence[Vec3],
    hit_uvs: Sequence[tuple[float, float]],
    *,
    index: UvIndex | None = None,
    seed: int = 0,
) -> dict[str, float]:
    """기록된 정렬 하나를 **다시 적합하지 않고** 게이트 지표로 재검사한다.

    `build_level_alignment`이 쓰는 것과 같은 함수다 — 그래서 커밋된 JSON을 나중에
    그대로 먹여 볼 수 있고, 게이트 자체가 회귀 테스트의 대상이 된다.
    """
    if index is None:
        index = build_uv_index(mesh)
    declared_scale = _declared_scale(mesh.declared)
    rotation = _rotation_matrix(alignment)
    placed = _world_tris_for(
        mesh.vertices, mesh.triangles, rotation, alignment.scale, alignment.translation
    )
    distances, residuals, triangle_ids, ties = uv_absolute_residuals(placed, mesh, hits, hit_uvs)
    d_med, d_p95, d_max = _percentiles(distances)
    r_med, r_p95, r_max = _percentiles(residuals)

    model_points, world_points, miss, ambiguous = _correspondences(mesh, index, hits, hit_uvs)
    stats = congruence_stats(model_points, world_points, declared_scale, seed=seed)

    # G4의 비등방성은 **기록된 회전**이 아니라 이 데이터가 유사변환과 얼마나 맞는지의
    # 성질이다. 기록된 회전을 직교화하면 언제나 0이 나와 실패할 수 없는 단언이 된다.
    # 그래서 주어진 착탄점으로 12자유도 아핀을 다시 세워 그 선형부의 비등방성을 잰다
    # (정렬을 다시 맞추는 것이 아니다 — 기록된 회전·평행이동·스케일은 그대로 쓴다).
    try:
        linear, _offset = solve_affine_12(model_points, world_points)
        _r, _c, aniso, _sign, _sv = orthogonalize(linear)
    except ValueError:
        aniso = math.inf
    cond, _cloud_sing = _conditioning(model_points)
    identity = rotation_math.matmul(rotation, rotation_math.transpose(rotation))
    orthogonality_error = max(
        abs(identity[r][c] - (1.0 if r == c else 0.0)) for r in range(3) for c in range(3)
    )

    return {
        "n_hits": float(len(hits)),
        "median_mm": d_med * 1000.0,
        "p95_mm": d_p95 * 1000.0,
        "max_mm": d_max * 1000.0,
        "uv_abs_resid_median_texel": r_med,
        "uv_abs_resid_p95_texel": r_p95,
        "uv_abs_resid_max_texel": r_max,
        "uv_seam_tie_fraction": ties / max(1, len(hits)),
        # **게이트되는 이름이 아니다.** 조회 실패·모호도의 문턱(0.002)은 부분표본의
        # 표집 잡음 안에 있다 — 400점 표본이면 한 점이 1/400 = 0.0025로 문턱을 넘어
        # 판정을 시드가 정한다(실측: 레벨 5). 그래서 게이트가 읽는
        # `uv_miss_fraction`/`uv_ambiguous_fraction`은 `build_level_alignment`이
        # **착탄점 전량**에서 재서 넣고, 여기서는 진단용 이름으로만 남긴다.
        "eval_uv_miss_fraction": miss,
        "eval_uv_ambiguous_fraction": ambiguous,
        "linear_aniso": aniso,
        "rotation_orthogonality_error": orthogonality_error,
        "scale_ratio_to_declared": abs(alignment.scale / declared_scale - 1.0),
        "det_sign": _mat3_det(rotation),
        "cond_min_over_max": cond,
        "distinct_normals": float(_normal_diversity(mesh, triangle_ids)),
        "surface_coverage_fraction": len(set(triangle_ids)) / max(1, len(mesh.triangles)),
        **stats,
    }


_LOWER_IS_BETTER: Final[tuple[str, ...]] = (
    "g0_median_mm",
    "g0_p95_mm",
    "g0_max_mm",
    "uv_degenerate_misattribution_mm",
    "atlas_overlap_max_gap_mm",
    "uv_miss_fraction",
    "uv_ambiguous_fraction",
    "uv_abs_resid_p95_texel",
    "uv_abs_resid_max_texel",
    "median_mm",
    "p95_mm",
    "max_mm",
    "fit_linear_aniso",
    "linear_aniso",
    "scale_ratio_to_declared",
)
_HIGHER_IS_BETTER: Final[tuple[str, ...]] = ("cond_min_over_max", "distinct_normals")


def gate_failures(measured: dict[str, float]) -> list[tuple[str, object, str]]:
    """`GATES`와 대조해 통과하지 못한 항목을 (이름, 잰 값, 필요 조건)으로 낸다."""
    failures: list[tuple[str, object, str]] = []
    for key in _LOWER_IS_BETTER:
        if key not in measured:
            continue
        value = measured[key]
        if not (value <= GATES[key]):
            failures.append((key, value, f"<= {GATES[key]}"))
    for key in _HIGHER_IS_BETTER:
        if key not in measured:
            continue
        value = measured[key]
        if not (value >= GATES[key]):
            failures.append((key, value, f">= {GATES[key]}"))
    return failures


# =============================================================================
# 솔버 — UV 색인 조회 → 12자유도 폐형식 → 직교화 → t 재적합
# =============================================================================


def solve_alignment_uv(
    mesh: Mesh,
    hits: Sequence[Vec3],
    *,
    hit_uvs: Sequence[tuple[float, float]],
    level: int = 1,
    index: UvIndex | None = None,
    fit_sample: int = 2000,
    eval_cap: int = 1500,
    trim_fraction: float = 0.15,
    seed: int = 0,
    convention: tuple[str, tuple[float, float]] = ("identity", (0.0, 0.0)),
) -> tuple[Alignment, dict[str, float], list[Vec3], list[tuple[float, float]]]:
    """관측 UV로 대응을 만들어 정렬을 푼다. 시드도, ICP도, 면 판별 휴리스틱도 없다.

    반환: `(alignment, measured, eval_hits, eval_uvs)`. `measured`는 `verify_alignment`이
    낸 게이트 지표 전부이고, `eval_*`은 **적합에 쓰이지 않은** 홀드아웃이다.
    """
    if not hits:
        raise ValueError("solve_alignment_uv: no hit points given")
    if len(hits) != len(hit_uvs):
        raise ValueError("solve_alignment_uv: hits and hit_uvs must be index-paired")
    if index is None:
        index = build_uv_index(mesh)

    # 한 순열을 **겹치지 않게** 자른다. 옛 코드는 같은 rng에서 두 번 독립 추출해
    # fit/eval이 거의 전부 겹쳤다 — 기록된 fit 통계가 홀드아웃이 아니었다는 뜻이다.
    rng = random.Random(seed)
    order = list(range(len(hits)))
    rng.shuffle(order)
    fit_idx = order[:fit_sample]
    eval_idx = order[fit_sample : fit_sample + eval_cap]
    if not eval_idx:
        raise ValueError(
            f"solve_alignment_uv: {len(hits)} hits cannot supply a disjoint holdout "
            f"after a fit sample of {fit_sample}"
        )

    fit_hits = [hits[i] for i in fit_idx]
    fit_uvs = [hit_uvs[i] for i in fit_idx]
    eval_hits = [hits[i] for i in eval_idx]
    eval_uvs = [hit_uvs[i] for i in eval_idx]

    src, dst, _miss, _ambiguous = _correspondences(mesh, index, fit_hits, fit_uvs, convention)
    if len(src) < 50:
        raise ValueError(
            f"solve_alignment_uv: only {len(src)} usable correspondences from "
            f"{len(fit_hits)} fit hits — the observed UVs do not land on this atlas"
        )

    linear, offset = solve_affine_12(src, dst)
    ranked = sorted(
        range(len(src)),
        key=lambda i: math.dist(
            tuple(sum(linear[d][k] * src[i][k] for k in range(3)) + offset[d] for d in range(3)),
            dst[i],
        ),
    )
    keep = ranked[: max(50, int(len(src) * (1.0 - trim_fraction)))]
    linear, offset = solve_affine_12([src[i] for i in keep], [dst[i] for i in keep])

    rotation, scale, fit_aniso, _det_sign, _sing = orthogonalize(linear)

    # R, c를 고정하고 t만 다시 맞춘다 — 직교화가 선형부를 바꿨으므로 offset은 낡았다.
    acc = [0.0, 0.0, 0.0]
    for i in keep:
        rv = rotation_math.matvec(rotation, src[i])
        for d in range(3):
            acc[d] += dst[i][d] - scale * rv[d]
    translation: Vec3 = tuple(x / len(keep) for x in acc)  # type: ignore[assignment]

    alignment = Alignment(
        level=level,
        provenance="solved",
        rotation=tuple(rotation[r][c] for r in range(3) for c in range(3)),
        translation=translation,
        scale=scale,
        declared_scale=_declared_scale(mesh.declared),
        fit={},
    )
    measured = verify_alignment(mesh, alignment, eval_hits, eval_uvs, index=index, seed=seed)
    # 이 해를 실제로 낸 적합의 등방성. `verify_alignment`이 내는 `linear_aniso`는
    # 기록된 규약(identity)의 대응으로 다시 세운 것이라 정렬·규약 오류에 둔감하다.
    measured["fit_linear_aniso"] = fit_aniso
    measured["fit_n"] = float(len(keep))
    measured["fit_eval_overlap"] = float(len(set(fit_idx) & set(eval_idx)))
    return alignment, measured, eval_hits, eval_uvs


# =============================================================================
# G8 — 적대적 대조. 매 빌드 돌린다.
# =============================================================================


def _mirror_matrices() -> list[Mat3]:
    out: list[Mat3] = []
    for axis in range(3):
        m = [[1.0 if i == j else 0.0 for j in range(3)] for i in range(3)]
        m[axis][axis] = -1.0
        out.append(tuple(tuple(row) for row in m))  # type: ignore[arg-type]
    return out


def adversarial_contrast(
    mesh: Mesh,
    alignment: Alignment,
    hits: Sequence[Vec3],
    hit_uvs: Sequence[tuple[float, float]],
    *,
    sample: int = 60,
) -> dict[str, float]:
    """해에 정육면체 회전 24개와 주평면 반사 3개를 **모델 중심에서** 합성해 G2를 재측정.

    항등이 유일 최소여야 하고, 2등이 `GATES["adversarial_runner_up_texel"]`를 넘어야
    한다. 통계는 **잔차 중앙값(텍셀)**이고 표본은 홀드아웃 착탄점의 앞 `sample`개다
    — 어느 통계·어느 중심·어느 표본인지 적어 둔다(앞선 조사에서 이 수가 출처와 다른
    라벨로 인용된 전례가 있다).
    """
    rotation = _rotation_matrix(alignment)
    centre = _mean(mesh.vertices)
    sample_hits = list(hits[:sample])
    sample_uvs = list(hit_uvs[:sample])

    def median_residual(rot: Mat3, trans: Vec3) -> float:
        placed = _world_tris_for(mesh.vertices, mesh.triangles, rot, alignment.scale, trans)
        _d, residuals, _t, _ties = uv_absolute_residuals(placed, mesh, sample_hits, sample_uvs)
        return _percentiles(residuals)[0]

    identity_score = median_residual(rotation, alignment.translation)
    runner_up = math.inf
    n_variants = 0
    for m in _cube_rotations() + _mirror_matrices():
        if all(
            abs(m[r][c] - (1.0 if r == c else 0.0)) < 1e-12 for r in range(3) for c in range(3)
        ):
            continue
        n_variants += 1
        rot = rotation_math.matmul(rotation, m)
        shift = _sub(centre, rotation_math.matvec(m, centre))
        r_shift = rotation_math.matvec(rotation, shift)
        trans: Vec3 = tuple(
            alignment.translation[d] + alignment.scale * r_shift[d] for d in range(3)
        )
        runner_up = min(runner_up, median_residual(rot, trans))
    return {
        "adversarial_identity_texel": identity_score,
        "adversarial_runner_up_texel": runner_up,
        "adversarial_variants": float(n_variants),
        "adversarial_sample": float(len(sample_hits)),
    }


# =============================================================================
# G9 — 규약족 대조. `UV_CONVENTION`이 고른 가지가 실제로 이기는지 매 빌드 확인한다.
# =============================================================================

UV_CONVENTION_FAMILIES: Final[tuple[str, ...]] = ("point", "u", "v")


def _apply_convention(family: str, k: tuple[float, float], obs: tuple[float, float]) -> tuple[float, float]:
    u = obs[0] / UV_ATLAS_TEXELS
    v = obs[1] / UV_ATLAS_TEXELS
    if family == "identity":
        return (u, v)
    if family == "point":
        return (k[0] - u, k[1] - v)
    if family == "u":
        return (k[0] - u, v)
    if family == "v":
        return (u, k[1] - v)
    raise ValueError(f"unknown UV convention family {family!r}")


def _convention_congruence(
    mesh: Mesh,
    index: UvIndex,
    hits: Sequence[Vec3],
    hit_uvs: Sequence[tuple[float, float]],
    family: str,
    k: tuple[float, float],
    declared_scale: float,
    *,
    n_pairs: int = 400,
) -> tuple[float, float]:
    model_points: list[Vec3] = []
    world_points: list[Vec3] = []
    for point, obs in zip(hits, hit_uvs):
        kind, ti, bary = uv_lookup(index, _apply_convention(family, k, obs))
        if kind != "hit":
            continue
        model_points.append(model_point_for_hit(mesh, ti, bary))  # type: ignore[arg-type]
        world_points.append(point)
    hit_fraction = len(model_points) / max(1, len(hits))
    if len(model_points) < 30:
        return math.inf, hit_fraction
    stats = congruence_stats(
        model_points, world_points, declared_scale, seed=1, n_pairs=n_pairs
    )
    return stats["g0_median_mm"], hit_fraction


def search_convention_candidates(
    mesh: Mesh,
    index: UvIndex,
    hits: Sequence[Vec3],
    hit_uvs: Sequence[tuple[float, float]],
    *,
    seed: int = 0,
    sample: int = 150,
    step: float = 0.02,
) -> list[tuple[str, tuple[float, float], float]]:
    """뒤집힌 UV 규약이 성립할 수 있는 `K`를 찾는다. 목적함수는 G0 합동성(정렬 무관).

    성긴 격자로 걸러 국소 정련한다. 족별 최선 하나씩만 돌려준다 — 그 해들을
    `build_level_alignment`이 실제로 풀어 게이트에 건다.
    """
    rng = random.Random(seed)
    declared_scale = _declared_scale(mesh.declared)
    picks = rng.sample(range(len(hits)), min(sample, len(hits)))
    sample_hits = [hits[i] for i in picks]
    sample_uvs = [hit_uvs[i] for i in picks]
    probe_hits, probe_uvs = sample_hits[:30], sample_uvs[:30]
    grid = [i * step for i in range(-10, int(round(2.2 / step)) + 1)]

    out: list[tuple[str, tuple[float, float], float]] = []
    for family in UV_CONVENTION_FAMILIES:
        if family == "point":
            candidates = [(a, b) for a in grid for b in grid]
        elif family == "u":
            candidates = [(a, 0.0) for a in grid]
        else:
            candidates = [(0.0, b) for b in grid]
        kept: list[tuple[float, float]] = []
        for k in candidates:
            landed = sum(
                1
                for obs in probe_uvs
                if uv_lookup(index, _apply_convention(family, k, obs))[0] == "hit"
            )
            if landed >= 0.8 * len(probe_uvs):
                kept.append(k)
        scored: list[tuple[float, tuple[float, float]]] = []
        for k in kept:
            score, fraction = _convention_congruence(
                mesh, index, sample_hits, sample_uvs, family, k, declared_scale
            )
            if fraction > 0.85:
                scored.append((score, k))
        if not scored:
            continue
        scored.sort()
        best_score, best_k = scored[0]
        current = list(best_k)
        stride = step / 2.0
        for _ in range(8):
            local: list[tuple[float, tuple[float, float]]] = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if family == "u" and dy:
                        continue
                    if family == "v" and dx:
                        continue
                    k = (current[0] + dx * stride, current[1] + dy * stride)
                    score, fraction = _convention_congruence(
                        mesh, index, sample_hits, sample_uvs, family, k, declared_scale, n_pairs=600
                    )
                    if fraction > 0.85:
                        local.append((score, k))
            if not local:
                break
            local.sort()
            if local[0][1] == tuple(current):
                stride /= 2.0
            current = list(local[0][1])
            best_score = min(best_score, local[0][0])
        out.append((family, (current[0], current[1]), best_score))
    return out


# =============================================================================
# level_alignment.json 재생성 (`python3 level_mesh.py`)
# =============================================================================

LEVEL_FBX: Final[dict[int, str]] = {
    1: "learning_mode_lv.1_rev.1.1.fbx",
    2: "learning_mode_lv.2_rev.1.2.fbx",
    3: "learning_mode_lv.3_rev.1.1.fbx",
    4: "learning_mode_lv.4_rev.3.fbx",
    5: "learning_mode_lv.5_rev.4.fbx",
    6: "learning_mode_lv.6_rev.3.13.fbx",
    7: "learning_mode_lv.7_rev.2.1.fbx",
    8: "learning_mode_lv.8_rev.1.1.fbx",
    9: "learning_mode_lv.9_rev.4.fbx",
    10: "learning_mode_lv.10_rev.1.2.fbx",
}


def collect_level_hits_by_session(
    level: int,
    professionals_root: Path = DEFAULT_PROFESSIONALS_ROOT,
    min_hits: int = DEFAULT_EVAL_CAP,
) -> tuple[list[Vec3], list[tuple[float, float]], list[str]]:
    """학습모드 `<level>_*.csv`에서 착탄점·UV·**세션 이름**을 모은다.

    **세션을 가로질러 표집한다.** 옛 구현은 정렬된 glob 순서로 파일을 읽다가
    `min_hits`에 닿으면 멈췄고, 그 결과 레벨 6·7·10은 1인 1세션이 전부인 것처럼
    보였다 — 실제로는 모든 레벨에 파일 45~49개, 세션 디렉터리 13~17개가 있다.
    그 조기 중단이 G6(교차 세션 홀드아웃)를 무력화하고 있었다. 여기서는 세션마다
    한 파일씩 돌아가며 읽고, 파일당 상한을 둬서 한 세션이 표본을 삼키지 못하게 한다.
    """
    import measured_pose
    import simulator_csv

    paths = sorted(professionals_root.glob(f"**/{level}_*.csv"))
    if not paths:
        return [], [], []
    by_session: dict[str, list[Path]] = {}
    for path in paths:
        session = "/".join(path.relative_to(professionals_root).parts[:-1])
        by_session.setdefault(session, []).append(path)
    sessions = sorted(by_session)
    per_file_cap = max(1, 3 * min_hits // len(paths))

    hits: list[Vec3] = []
    uvs: list[tuple[float, float]] = []
    session_ids: list[str] = []
    queues = {name: list(files) for name, files in by_session.items()}
    while any(queues.values()) and len(hits) < min_hits:
        for session in sessions:
            if not queues[session]:
                continue
            rows = simulator_csv.read_rows(queues[session].pop(0))
            taken = 0
            for frame in measured_pose.measured_frames(rows):
                if frame.hit is None or frame.uv is None or not frame.spraying:
                    continue
                hits.append(frame.hit)
                uvs.append(frame.uv)
                session_ids.append(session)
                taken += 1
                if taken >= per_file_cap:
                    break
            if len(hits) >= min_hits:
                break
    return hits, uvs, session_ids


def _collect_level_hits(
    level: int,
    professionals_root: Path = DEFAULT_PROFESSIONALS_ROOT,
    min_hits: int = DEFAULT_EVAL_CAP,
) -> tuple[list[Vec3], list[tuple[float, float]]]:
    """`collect_level_hits_by_session`의 (착탄점, UV) 두 값짜리 얇은 껍데기."""
    hits, uvs, _sessions = collect_level_hits_by_session(level, professionals_root, min_hits)
    return hits, uvs


def _worker_of(session: str) -> str:
    """세션 이름에서 작업자를 뽑는다 — `김경호_20250831/...` → `김경호`."""
    head = session.split("/")[0]
    return head.split("_")[0]


def build_level_alignment(
    level: int,
    fbx_path: Path,
    professionals_root: Path = DEFAULT_PROFESSIONALS_ROOT,
    min_hits: int = DEFAULT_EVAL_CAP,
    seed: int = 0,
) -> dict:
    """게이트를 전부 통과한 레벨 하나의 JSON 항목. 못 넘으면 `AlignmentRejected`.

    통과 못 한 레벨은 **항목이 생기지 않는다** — 화면이 "검증된 정렬이 없다"고
    말하고, 권위 있어 보이는 틀린 기하를 그리지 않는다.
    """
    mesh = read_fbx(fbx_path)
    index = build_uv_index(mesh)
    hits, hit_uvs, sessions = collect_level_hits_by_session(level, professionals_root, min_hits)
    if len(hits) < min_hits:
        raise ValueError(f"only collected {len(hits)} hits for level {level}, need >= {min_hits}")

    # --- G1 (정렬 이전, 메시만) + 착탄점 전량의 조회 통계 ------------------------
    atlas = audit_atlas(mesh, index, seed=seed)
    _m, _w, miss_all, ambiguous_all = _correspondences(mesh, index, hits, hit_uvs)
    atlas["uv_miss_fraction"] = miss_all
    atlas["uv_ambiguous_fraction"] = ambiguous_all
    atlas["uv_lookup_sample"] = float(len(hits))
    # G7의 조건수는 정렬 이전에 성립한다 — 대응 구름(모델 공간)의 성질이다. 레벨 1이
    # 여기서 떨어진다: 착탄점이 정확히 평면이라 12자유도가 데이터로 결정되지 않는다.
    atlas["cond_min_over_max"] = _conditioning(_m)[0]

    # --- G7의 조건수는 정렬 이전에 성립한다(대응 구름의 성질) --------------------
    pre_failures = gate_failures(atlas)
    if pre_failures:
        raise AlignmentRejected(level, pre_failures)

    try:
        alignment, measured, eval_hits, eval_uvs = solve_alignment_uv(
            mesh, hits, hit_uvs=hit_uvs, level=level, index=index, seed=seed
        )
    except ValueError as error:
        raise AlignmentRejected(level, [("solver", str(error), "a solvable 12-DoF system")]) from error

    measured.update(atlas)
    measured.update(adversarial_contrast(mesh, alignment, eval_hits, eval_uvs))
    failures = gate_failures(measured)
    if not (measured["adversarial_runner_up_texel"] > GATES["adversarial_runner_up_texel"]):
        failures.append(
            (
                "adversarial_runner_up_texel",
                measured["adversarial_runner_up_texel"],
                f"> {GATES['adversarial_runner_up_texel']}",
            )
        )
    if not (measured["adversarial_identity_texel"] < measured["adversarial_runner_up_texel"]):
        failures.append(("adversarial_identity_not_unique_minimum", measured["adversarial_identity_texel"], "< runner-up"))

    # --- G6 교차 세션 홀드아웃 (모든 레벨에 필수) --------------------------------
    measured.update(_cross_session_check(mesh, index, hits, hit_uvs, sessions, level, seed))
    if measured["cross_session_pass"] < 1.0:
        failures.append(("cross_session", measured["cross_session_worst_mm"], "G2/G3 thresholds on a held-out worker"))

    # --- G9 규약족 대조 ----------------------------------------------------------
    contrast = _convention_contrast(
        mesh, index, hits, hit_uvs, level, seed, accepted_det_sign=measured["det_sign"]
    )
    measured.update(contrast)
    if contrast["convention_alternatives_accepted"] > 0.0:
        failures.append(
            (
                "uv_convention",
                contrast["convention_alternatives_accepted"],
                "no flipped UV convention may pass the gates",
            )
        )

    if failures:
        raise AlignmentRejected(level, failures)

    alignment = Alignment(
        level=alignment.level,
        provenance="solved",
        rotation=alignment.rotation,
        translation=alignment.translation,
        scale=alignment.scale,
        declared_scale=alignment.declared_scale,
        fit={k: v for k, v in sorted(measured.items())},
    )
    entry = _entry_from_alignment(alignment, source_fbx=fbx_path.name)
    entry["uv_convention"] = UV_CONVENTION
    entry["uv_convention_source"] = "measured"
    entry["handedness_source"] = "data"
    # G6의 근거: 이 정렬이 몇 명·몇 세션을 설명하는가. 경로 전체는 수집기가 언제든
    # 다시 낸다 — 여기에는 판정에 쓰이는 요약만 남긴다.
    entry["workers"] = sorted({_worker_of(name) for name in sessions})
    entry["session_count"] = len(set(sessions))
    return entry


def _cross_session_check(
    mesh: Mesh,
    index: UvIndex,
    hits: Sequence[Vec3],
    hit_uvs: Sequence[tuple[float, float]],
    sessions: Sequence[str],
    level: int,
    seed: int,
) -> dict[str, float]:
    """한 작업자의 세션으로만 적합하고, 나머지 작업자 전원에서 평가한다.

    **"세션이 하나뿐이면 건너뛴다"는 갈래를 두지 않는다** — 실측상 모든 레벨에
    작업자 6명·세션 13개 이상이 있고, 예외 갈래는 거부할 입력이 없는 검사를 만든다.
    """
    workers: dict[str, list[int]] = {}
    for i, session in enumerate(sessions):
        workers.setdefault(_worker_of(session), []).append(i)
    if len(workers) < 2:
        return {"cross_session_pass": 0.0, "cross_session_workers": float(len(workers)), "cross_session_worst_mm": math.inf, "cross_session_worst_texel": math.inf}
    fit_worker = max(workers, key=lambda w: len(workers[w]))
    fit_rows = workers[fit_worker]
    other_rows = [i for w, rows in workers.items() if w != fit_worker for i in rows]
    fit_hits = [hits[i] for i in fit_rows]
    fit_uvs = [hit_uvs[i] for i in fit_rows]
    src, dst, _miss, _ambiguous = _correspondences(mesh, index, fit_hits, fit_uvs)
    if len(src) < 50:
        return {"cross_session_pass": 0.0, "cross_session_workers": float(len(workers)), "cross_session_worst_mm": math.inf, "cross_session_worst_texel": math.inf}
    linear, _offset = solve_affine_12(src, dst)
    rotation, scale, _aniso, _det, _sing = orthogonalize(linear)
    acc = [0.0, 0.0, 0.0]
    for p, q in zip(src, dst):
        rv = rotation_math.matvec(rotation, p)
        for d in range(3):
            acc[d] += q[d] - scale * rv[d]
    translation: Vec3 = tuple(x / len(src) for x in acc)  # type: ignore[assignment]
    held = Alignment(
        level=level,
        provenance="solved",
        rotation=tuple(rotation[r][c] for r in range(3) for c in range(3)),
        translation=translation,
        scale=scale,
        declared_scale=_declared_scale(mesh.declared),
        fit={},
    )
    rng = random.Random(seed + 991)
    picks = rng.sample(other_rows, min(600, len(other_rows)))
    out = verify_alignment(
        mesh, held, [hits[i] for i in picks], [hit_uvs[i] for i in picks], index=index, seed=seed
    )
    ok = (
        out["p95_mm"] <= GATES["p95_mm"]
        and out["max_mm"] <= GATES["max_mm"]
        and out["uv_abs_resid_p95_texel"] <= GATES["uv_abs_resid_p95_texel"]
        and out["uv_abs_resid_max_texel"] <= GATES["uv_abs_resid_max_texel"]
    )
    return {
        "cross_session_pass": 1.0 if ok else 0.0,
        "cross_session_workers": float(len(workers)),
        "cross_session_fit_worker_rows": float(len(fit_rows)),
        "cross_session_eval_rows": float(len(picks)),
        "cross_session_worst_mm": out["max_mm"],
        "cross_session_worst_texel": out["uv_abs_resid_max_texel"],
    }


def _convention_contrast(
    mesh: Mesh,
    index: UvIndex,
    hits: Sequence[Vec3],
    hit_uvs: Sequence[tuple[float, float]],
    level: int,
    seed: int,
    accepted_det_sign: float,
    log: bool = True,
) -> dict[str, float]:
    """뒤집힌 UV 규약에서 K를 찾아 **실제로 다시 풀고**, 그 해가 거절되는지 본다.

    이것이 브리핑이 요구한 반박 대응의 핵심이다. 앞선 조사가 레벨 8에서
    `mesh_uv = K - obs/2048` 가지의 해로 179.9998° 뒤집힌 정렬을 만들어 옛 게이트
    전부를 통과시켰다. 그 공격이 성립한 전제는 "솔버와 게이트가 **같은 뒤집힌 규약**을
    쓴다"는 것이었다 — 여기서는 게이트(`verify_alignment` → G2)가 언제나 이름 붙은
    `UV_CONVENTION`으로 재므로, 뒤집힌 규약의 해는 자기 규약이 아니라 기록될 규약으로
    평가받는다. 실측(레벨 8): u족 K=(0.80031, 0)의 해는 G3 중앙값 0.00284 mm로
    **거리 게이트를 통과하고** aniso·스케일도 통과한다 — 오직 G2(p95 614 텍셀)와
    G5(det=+1, 나머지 아홉 레벨과 반대)가 거절한다.
    """
    accepted = 0
    best_rival = math.inf
    tried = 0
    min_failures = math.inf
    for family, k, score in search_convention_candidates(mesh, index, hits, hit_uvs, seed=seed):
        tried += 1
        try:
            _rival, rival_measured, _eh, _eu = solve_alignment_uv(
                mesh, hits, hit_uvs=hit_uvs, level=level, index=index, seed=seed,
                fit_sample=1200, eval_cap=400, convention=(family, k),
            )
        except ValueError as error:
            if log:
                print(f"    G9 [{family} K={tuple(round(x, 5) for x in k)}] unsolvable: {error}")
            continue
        rival_failures = gate_failures(rival_measured)
        # G5: 손잡이가 채택 해와 반대인 후보는 그 자체로 거절 사유다. 이 검사에
        # 실재하는 거절 입력을 주는 것이 바로 이 규약족 대조다.
        if rival_measured["det_sign"] * accepted_det_sign < 0:
            rival_failures.append(("det_sign", rival_measured["det_sign"], f"same sign as {accepted_det_sign}"))
        min_failures = min(min_failures, len(rival_failures))
        if not rival_failures:
            accepted += 1
        best_rival = min(best_rival, rival_measured["median_mm"])
        if log:
            names = ",".join(str(name) for name, _v, _n in rival_failures) or "NOTHING"
            print(
                f"    G9 [{family} K={tuple(round(x, 5) for x in k)}] g0={score:.5f} "
                f"G3med={rival_measured['median_mm']:.5f}mm G2p95={rival_measured['uv_abs_resid_p95_texel']:.4f}tx "
                f"det={rival_measured['det_sign']:+.3f} -> rejected by {names}"
            )
    return {
        "convention_families_tried": float(tried),
        "convention_alternatives_accepted": float(accepted),
        "convention_rival_min_gate_failures": min_failures if math.isfinite(min_failures) else -1.0,
        # -1.0 = 어느 뒤집힌 규약도 풀리는 해를 내지 못했다.
        "convention_best_rival_median_mm": best_rival if math.isfinite(best_rival) else -1.0,
    }


# =============================================================================
# 선형부 고정 경로 — 부호 있는 축 순열 48개 × 평행이동 3자유도. 12자유도 경로와 G7은 그대로.
#
# 규칙 문서: docs/measurements/2026-09-15-fixed-linear-alignment-rule.md
# §A(결과 관측 후 작성): 후보·s·t 재적합·G0/G1/G2/G3/G6·G_occ_gun·유일성.
# §B(사전등록, 커밋 579e8bc가 이 코드보다 먼저): G_occ_hmd ≤ 0.25, 양성 대조 8레벨, 음성 대조.
#
# 레벨 1의 착탄점은 정확히 평면이라 12자유도가 결정되지 않는다(G7). 선형부를 고정하면
# 평행이동 3자유도는 평면으로도 결정된다 — 그러니 **이 경로에는 G7을 걸지 않는다**(걸면
# 실패할 수 없는 거절). 대신 평면 데이터의 절대 UV 등식이 원리적으로 못 가르는 후보(판 법선
# 반사, det +1)가 남는다. 그것을 가르는 것이 가림 게이트다.
# =============================================================================

FIXED_LINEAR_SCALE: Final = 0.0254
OCCLUSION_STOP_BEFORE_M: Final = 0.002
OCCLUSION_THRESHOLD: Final = 0.25
FIXED_LINEAR_RULE_DOC: Final = "docs/measurements/2026-09-15-fixed-linear-alignment-rule.md"
# 선형부 고정 경로에서 판정에 쓰는 기존 게이트 이름. G4의 aniso·G7·G8·G9는 여기 없다 —
# 이유는 규칙 문서 §A.3.
FIXED_LINEAR_GATE_KEYS: Final[tuple[str, ...]] = (
    "g0_median_mm",
    "g0_p95_mm",
    "g0_max_mm",
    "uv_degenerate_misattribution_mm",
    "atlas_overlap_max_gap_mm",
    "uv_miss_fraction",
    "uv_ambiguous_fraction",
    "uv_abs_resid_p95_texel",
    "uv_abs_resid_max_texel",
    "median_mm",
    "p95_mm",
    "max_mm",
)


def signed_permutations() -> list[Mat3]:
    """부호 있는 축 순열 48개(det +1 24개, det −1 24개). 순서는 결정적이다."""
    out: list[Mat3] = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1.0, -1.0), repeat=3):
            m = [[0.0, 0.0, 0.0] for _ in range(3)]
            for row in range(3):
                m[row][perm[row]] = signs[row]
            out.append(tuple(tuple(row) for row in m))  # type: ignore[arg-type]
    return out


def _segment_triangle_param(o: Vec3, e: Vec3, a: Vec3, b: Vec3, c: Vec3) -> float | None:
    """선분 o→e와 삼각형의 교차 매개변수 s(o + s·(e−o)). 없으면 None. Möller–Trumbore."""
    d = _sub(e, o)
    e1 = _sub(b, a)
    e2 = _sub(c, a)
    pv = _cross(d, e2)
    det = _dot(e1, pv)
    if abs(det) < 1e-15:
        return None
    inv = 1.0 / det
    tv = _sub(o, a)
    u = _dot(tv, pv) * inv
    if u < 0.0 or u > 1.0:
        return None
    qv = _cross(tv, e1)
    v = _dot(d, qv) * inv
    if v < 0.0 or u + v > 1.0:
        return None
    s = _dot(e2, qv) * inv
    if s <= 0.0 or s >= 1.0:
        return None
    return s


def occlusion_fraction(
    placed_mesh: Mesh,
    origins: Sequence[Vec3],
    hits: Sequence[Vec3],
    stop_before_m: float = OCCLUSION_STOP_BEFORE_M,
) -> float:
    """원점→착탄점 선분이 착탄 `stop_before_m` 앞에서 배치된 메시와 교차하는 프레임 비율.

    건·HMD·플래쉬가 **같은 함수**를 쓴다 — 원점만 다르다(규칙 문서 §A.2, §B.1). 착탄점
    자체의 면은 s·L = L이므로 `s·L < L − stop_before_m`에서 빠진다. 그 부등호가 뒤집히면
    착탄점이 놓인 면이 늘 자기 자신을 가려 모든 프레임이 1.0이 된다.
    """
    if len(origins) != len(hits):
        raise ValueError("occlusion_fraction: origins and hits must be index-paired")
    if not hits:
        raise ValueError("occlusion_fraction: no frames")
    triangles = _world_triangles(placed_mesh)
    blocked = 0
    for origin, hit in zip(origins, hits):
        length = math.dist(origin, hit)
        for a, b, c in triangles:
            s = _segment_triangle_param(origin, hit, a, b, c)
            if s is not None and s * length < length - stop_before_m:
                blocked += 1
                break
    return blocked / len(hits)


@dataclass(frozen=True, slots=True)
class LevelFrames:
    """선형부 고정 경로의 표본. `collect_level_hits_by_session`과 같은 순서·같은 필터에
    원점(건·HMD·플래쉬) 위치를 함께 싣는다. 플래쉬는 거리 필터 탈락 시 None이다."""

    hits: list[Vec3]
    uvs: list[tuple[float, float]]
    sessions: list[str]
    guns: list[Vec3]
    heads: list[Vec3]
    flashes: list[Vec3 | None]
    # 실제로 **열린** 파일의 경로(문자열). `collect_level_frames_by_session`은 15,000
    # 프레임에서 멈추므로 한 번도 열리지 않은 파일이 남는다 — 2026-09-16 사전등록 §3.2의
    # 새 홀드아웃 H2가 바로 그 미개봉 파일이다. 목록을 하드코딩하지 않으려면 소비
    # 알고리즘 자신이 무엇을 먹었는지 말해야 한다.
    paths: list[str] = field(default_factory=list)


def collect_level_frames_by_session(
    level: int,
    professionals_root: Path = DEFAULT_PROFESSIONALS_ROOT,
    min_hits: int = DEFAULT_EVAL_CAP,
) -> LevelFrames:
    """`collect_level_hits_by_session`과 **같은 세션 교대·파일당 상한·필터**로 모은다."""
    import measured_pose
    import simulator_csv

    frames = LevelFrames([], [], [], [], [], [])
    paths = sorted(professionals_root.glob(f"**/{level}_*.csv"))
    if not paths:
        return frames
    by_session: dict[str, list[Path]] = {}
    for path in paths:
        session = "/".join(path.relative_to(professionals_root).parts[:-1])
        by_session.setdefault(session, []).append(path)
    sessions = sorted(by_session)
    per_file_cap = max(1, 3 * min_hits // len(paths))
    queues = {name: list(files) for name, files in by_session.items()}
    while any(queues.values()) and len(frames.hits) < min_hits:
        for session in sessions:
            if not queues[session]:
                continue
            path = queues[session].pop(0)
            frames.paths.append(str(path))
            rows = simulator_csv.read_rows(path)
            taken = 0
            for frame in measured_pose.measured_frames(rows):
                if frame.hit is None or frame.uv is None or not frame.spraying:
                    continue
                frames.hits.append(frame.hit)
                frames.uvs.append(frame.uv)
                frames.sessions.append(session)
                frames.guns.append(frame.right)
                frames.heads.append(frame.head)
                frames.flashes.append(frame.left)
                taken += 1
                if taken >= per_file_cap:
                    break
            if len(frames.hits) >= min_hits:
                break
    return frames


def split_fit_holdout(n: int, fit_sample: int, eval_cap: int, seed: int = 0) -> tuple[list[int], list[int]]:
    """한 순열을 **겹치지 않게** 자른다(`solve_alignment_uv`와 같은 규칙)."""
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    return order[:fit_sample], order[fit_sample : fit_sample + eval_cap]


def solve_translation_fixed(
    rotation: Mat3,
    scale: float,
    model_points: Sequence[Vec3],
    world_points: Sequence[Vec3],
    inlier_m: float = 1e-3,
) -> tuple[Vec3, int]:
    """R·s 고정에서 t: t_i = q − s·R·p의 축별 중앙값 → 그 1 mm 안 평균. (t, 인라이어 수)."""
    if not model_points:
        raise ValueError("solve_translation_fixed: no correspondences")
    ts: list[Vec3] = []
    for p, q in zip(model_points, world_points):
        rv = rotation_math.matvec(rotation, p)
        ts.append((q[0] - scale * rv[0], q[1] - scale * rv[1], q[2] - scale * rv[2]))
    median: Vec3 = tuple(sorted(t[d] for t in ts)[len(ts) // 2] for d in range(3))  # type: ignore[assignment]
    inliers = [t for t in ts if math.dist(t, median) < inlier_m]
    if not inliers:
        return median, 0
    mean: Vec3 = tuple(sum(t[d] for t in inliers) / len(inliers) for d in range(3))  # type: ignore[assignment]
    return mean, len(inliers)


def _fixed_alignment(level: int, rotation: Mat3, translation: Vec3, provenance: str = "solved") -> Alignment:
    return Alignment(
        level=level,
        provenance=provenance,
        rotation=tuple(rotation[r][c] for r in range(3) for c in range(3)),
        translation=translation,
        scale=FIXED_LINEAR_SCALE,
        declared_scale=FIXED_LINEAR_SCALE,
        fit={},
    )


def measure_placement(
    mesh: Mesh,
    rotation: Mat3,
    scale: float,
    translation: Vec3,
    hits: Sequence[Vec3],
    uvs: Sequence[tuple[float, float]],
    *,
    excluded_triangles: Collection[int] = (),
) -> dict[str, float]:
    """G2·G3만(홀드아웃). 적합하지 않는다.

    `excluded_triangles`가 비어 있으면 예전과 **한 글자도 다르지 않은** 결과를 낸다.
    비어 있지 않으면(2026-09-16 사전등록 §2) UV 잔차 통계는 제외되지 않은 착탄점에서만
    재고, 표면 거리(G3)는 전량에서 그대로 재며, 제외된 착탄점의 잔차를
    `excluded_*` 이름으로 **따로** 싣는다 — 무엇을 가렸는지 감추지 않기 위해서다.
    """
    placed = _world_tris_for(mesh.vertices, mesh.triangles, rotation, scale, translation)
    distances, residuals, tids, ties, excluded = uv_absolute_residuals_excluding(
        placed, mesh, hits, uvs, excluded_triangles=excluded_triangles
    )
    d_med, d_p95, d_max = _percentiles(distances)
    kept = [r for r, drop in zip(residuals, excluded) if not drop]
    r_med, r_p95, r_max = _percentiles(kept)
    out = {
        "median_mm": d_med * 1000.0,
        "p95_mm": d_p95 * 1000.0,
        "max_mm": d_max * 1000.0,
        "uv_abs_resid_median_texel": r_med,
        "uv_abs_resid_p95_texel": r_p95,
        "uv_abs_resid_max_texel": r_max,
        "uv_seam_tie_fraction": ties / max(1, len(hits)),
    }
    if excluded_triangles:
        dropped = [r for r, drop in zip(residuals, excluded) if drop]
        e_med, e_p95, e_max = _percentiles(dropped)
        out.update({
            "uv_excluded_hits": float(len(dropped)),
            "uv_excluded_hit_fraction": len(dropped) / max(1, len(hits)),
            "uv_kept_hits": float(len(kept)),
            "uv_excluded_resid_median_texel": e_med if dropped else 0.0,
            "uv_excluded_resid_p95_texel": e_p95 if dropped else 0.0,
            "uv_excluded_resid_max_texel": e_max if dropped else 0.0,
            "uv_kept_distinct_normals": float(
                _normal_diversity(mesh, [t for t, drop in zip(tids, excluded) if not drop])
            ),
        })
    return out


def _placed_mesh(mesh: Mesh, rotation: Mat3, scale: float, translation: Vec3) -> Mesh:
    return place(
        mesh,
        Alignment(
            level=0, provenance="solved",
            rotation=tuple(rotation[r][c] for r in range(3) for c in range(3)),
            translation=translation, scale=scale, declared_scale=scale, fit={},
        ),
    )


def origin_occlusions(
    mesh: Mesh,
    rotation: Mat3,
    scale: float,
    translation: Vec3,
    frames: LevelFrames,
    indices: Sequence[int],
) -> dict[str, float]:
    """건·HMD·플래쉬 가림(같은 `occlusion_fraction`, 원점만 다름)."""
    placed = _placed_mesh(mesh, rotation, scale, translation)
    hits = [frames.hits[i] for i in indices]
    flash_idx = [i for i in indices if frames.flashes[i] is not None]
    out = {
        "gun_occlusion": occlusion_fraction(placed, [frames.guns[i] for i in indices], hits),
        "hmd_occlusion": occlusion_fraction(placed, [frames.heads[i] for i in indices], hits),
        "occlusion_n": float(len(indices)),
        "flash_occlusion_n": float(len(flash_idx)),
    }
    out["flash_occlusion"] = (
        occlusion_fraction(placed, [frames.flashes[i] for i in flash_idx], [frames.hits[i] for i in flash_idx])  # type: ignore[misc]
        if flash_idx
        else None
    )
    return out  # type: ignore[return-value]


def _cross_worker_fixed(
    mesh: Mesh, index: UvIndex, rotation: Mat3, frames: LevelFrames, seed: int,
    *, excluded_triangles: Collection[int] = (),
) -> dict[str, float]:
    """G6(선형부 고정판): 최다 작업자로만 t를 적합, 다른 작업자 600행에서 G2·G3 p95/max."""
    workers: dict[str, list[int]] = {}
    for i, session in enumerate(frames.sessions):
        workers.setdefault(_worker_of(session), []).append(i)
    fail = {"cross_session_pass": 0.0, "cross_session_workers": float(len(workers)),
            "cross_session_worst_mm": math.inf, "cross_session_worst_texel": math.inf}
    if len(workers) < 2:
        return fail
    fit_worker = max(workers, key=lambda w: len(workers[w]))
    fit_rows = workers[fit_worker]
    other_rows = [i for w, rows in workers.items() if w != fit_worker for i in rows]
    src, dst, _m, _a = _correspondences(
        mesh, index, [frames.hits[i] for i in fit_rows], [frames.uvs[i] for i in fit_rows]
    )
    if len(src) < 50:
        return fail
    translation, _n = solve_translation_fixed(rotation, FIXED_LINEAR_SCALE, src, dst)
    picks = random.Random(seed + 991).sample(other_rows, min(600, len(other_rows)))
    out = measure_placement(
        mesh, rotation, FIXED_LINEAR_SCALE, translation,
        [frames.hits[i] for i in picks], [frames.uvs[i] for i in picks],
        excluded_triangles=excluded_triangles,
    )
    ok = (
        out["p95_mm"] <= GATES["p95_mm"]
        and out["max_mm"] <= GATES["max_mm"]
        and out["uv_abs_resid_p95_texel"] <= GATES["uv_abs_resid_p95_texel"]
        and out["uv_abs_resid_max_texel"] <= GATES["uv_abs_resid_max_texel"]
    )
    return {
        "cross_session_pass": 1.0 if ok else 0.0,
        "cross_session_workers": float(len(workers)),
        "cross_session_fit_worker_rows": float(len(fit_rows)),
        "cross_session_eval_rows": float(len(picks)),
        "cross_session_worst_mm": out["max_mm"],
        "cross_session_worst_texel": out["uv_abs_resid_max_texel"],
    }


def _matrix_list(m: Mat3) -> list[float]:
    return [m[r][c] for r in range(3) for c in range(3)]


def solve_fixed_linear(
    level: int,
    mesh: Mesh,
    frames: LevelFrames,
    *,
    index: UvIndex | None = None,
    fit_sample: int = 2000,
    eval_cap: int = 1500,
    seed: int = 0,
    candidates: Sequence[Mat3] | None = None,
    screen: int = 100,
    cross_worker: bool = True,
    atlas: dict[str, float] | None = None,
    always_measure: Sequence[Mat3] = (),
) -> dict:
    """후보 R마다 t를 다시 풀고 §A·§B 지표를 잰다. **판정(유일성·§B 유효성)은 하지 않는다** —
    §B 유효성은 다른 레벨의 양성 대조에 달려 있어 `judge_fixed_linear`이 따로 한다.

    선별(`screen`): 홀드아웃 앞 `screen`점의 G2·G3 **max**가 이미 문턱을 넘는 후보는 전체
    홀드아웃에서도 반드시 넘는다(max는 부분집합에서 줄기만 한다). 그런 후보는 전량 평가를
    건너뛰고 `screened_out`로 기록한다 — 판정이 바뀌지 않는 가지치기다.
    """
    if index is None:
        index = build_uv_index(mesh)
    fit_idx, eval_idx = split_fit_holdout(len(frames.hits), fit_sample, eval_cap, seed)
    if len(eval_idx) < eval_cap:
        raise ValueError(f"solve_fixed_linear: only {len(frames.hits)} frames for fit {fit_sample} + holdout {eval_cap}")
    fit_hits = [frames.hits[i] for i in fit_idx]
    fit_uvs = [frames.uvs[i] for i in fit_idx]
    eval_hits = [frames.hits[i] for i in eval_idx]
    eval_uvs = [frames.uvs[i] for i in eval_idx]

    src, dst, fit_miss, fit_amb = _correspondences(mesh, index, fit_hits, fit_uvs)
    if len(src) < 50:
        raise ValueError(f"solve_fixed_linear: only {len(src)} usable correspondences")
    ev_src, ev_dst, _em, _ea = _correspondences(mesh, index, eval_hits, eval_uvs)
    shared = congruence_stats(ev_src, ev_dst, FIXED_LINEAR_SCALE, seed=seed)
    if atlas is None:
        atlas = audit_atlas(mesh, index, seed=seed)
        _m, _w, miss_all, amb_all = _correspondences(mesh, index, frames.hits, frames.uvs)
        atlas = {**atlas, "uv_miss_fraction": miss_all, "uv_ambiguous_fraction": amb_all,
                 "uv_lookup_sample": float(len(frames.hits))}
    shared.update(atlas)

    results: list[dict] = []
    for rotation in (candidates if candidates is not None else signed_permutations()):
        translation, n_inliers = solve_translation_fixed(rotation, FIXED_LINEAR_SCALE, src, dst)
        record: dict = {
            "rotation": _matrix_list(rotation),
            "det": _mat3_det(rotation),
            "translation": list(translation),
            "fit_inliers": n_inliers,
        }
        probe = measure_placement(
            mesh, rotation, FIXED_LINEAR_SCALE, translation, eval_hits[:screen], eval_uvs[:screen]
        )
        exempt = any(rotation == keep for keep in always_measure)
        if not exempt and (
            probe["max_mm"] > GATES["max_mm"] or probe["uv_abs_resid_max_texel"] > GATES["uv_abs_resid_max_texel"]
        ):
            record["screened_out"] = {"sample": float(min(screen, len(eval_hits))), **probe}
            record["gate_failures_A"] = [
                name for name in ("max_mm", "uv_abs_resid_max_texel")
                if probe[name] > GATES[name]
            ]
            record["passed_A"] = False
            record["passed_B"] = None
            results.append(record)
            continue
        measured = {**shared, **measure_placement(
            mesh, rotation, FIXED_LINEAR_SCALE, translation, eval_hits, eval_uvs
        )}
        measured.update(origin_occlusions(mesh, rotation, FIXED_LINEAR_SCALE, translation, frames, eval_idx))
        if cross_worker:
            measured.update(_cross_worker_fixed(mesh, index, rotation, frames, seed))
        failures_a = [name for name, _v, _n in gate_failures({k: measured[k] for k in FIXED_LINEAR_GATE_KEYS})]
        if cross_worker and measured["cross_session_pass"] < 1.0:
            failures_a.append("cross_session")
        if not (measured["gun_occlusion"] <= OCCLUSION_THRESHOLD):
            failures_a.append("gun_occlusion")
        record["measured"] = measured
        record["gate_failures_A"] = failures_a
        record["passed_A"] = not failures_a
        record["passed_B"] = bool(measured["hmd_occlusion"] <= OCCLUSION_THRESHOLD)
        results.append(record)

    return {
        "level": level,
        "fit_n": len(fit_idx),
        "holdout_n": len(eval_idx),
        "fit_indices": fit_idx,
        "eval_indices": eval_idx,
        "holdout_overlap": len(set(fit_idx) & set(eval_idx)),
        "fit_uv_miss_fraction": fit_miss,
        "fit_uv_ambiguous_fraction": fit_amb,
        "candidates": results,
    }


def recorded_control_occlusion(level: int, mesh: Mesh, frames: LevelFrames, *, seed: int = 0,
                               fit_sample: int = 2000, eval_cap: int = 1500) -> dict[str, float]:
    """§B 양성 대조: 기록된 변환 그대로(재적합 없음), §A.1과 같은 홀드아웃에서 가림을 잰다."""
    alignment = load_alignment(level)
    if not is_solved(alignment.provenance):
        raise ValueError(f"level {level} is not a solved control")
    _fit, eval_idx = split_fit_holdout(len(frames.hits), fit_sample, eval_cap, seed)
    return origin_occlusions(
        mesh, _rotation_matrix(alignment), alignment.scale, alignment.translation, frames, eval_idx
    )


def judge_fixed_linear(result: dict, positive_controls: dict[int, float], mirror: Mat3) -> dict:
    """규칙 문서 §B.3·§B.4 그대로. `positive_controls`는 레벨 → G_occ_hmd."""
    required = {3, 4, 5, 6, 7, 8, 9, 10}
    missing = sorted(required - set(positive_controls))
    over = {lv: v for lv, v in positive_controls.items() if not (v <= OCCLUSION_THRESHOLD)}
    mirror_list = _matrix_list(mirror)
    mirror_rec = next((c for c in result["candidates"] if c["rotation"] == mirror_list), None)
    mirror_hmd = None if mirror_rec is None or "measured" not in mirror_rec else mirror_rec["measured"]["hmd_occlusion"]
    reasons: list[str] = []
    if missing:
        reasons.append(f"positive controls missing: {missing}")
    if over:
        reasons.append(f"positive control hmd_occlusion > {OCCLUSION_THRESHOLD}: {over}")
    if mirror_hmd is None or not (mirror_hmd > OCCLUSION_THRESHOLD):
        reasons.append(f"negative control (mirror) hmd_occlusion {mirror_hmd} not > {OCCLUSION_THRESHOLD}")
    valid_b = not reasons
    passed = [c for c in result["candidates"] if c["passed_A"] and c["passed_B"]]
    passed_a = [c for c in result["candidates"] if c["passed_A"]]
    solved = valid_b and len(passed) == 1
    return {
        "section_b_valid": valid_b,
        "section_b_invalid_reasons": reasons,
        "candidates_passed_A": len(passed_a),
        "candidates_passed_A_and_B": len(passed) if valid_b else None,
        "solved": solved,
        "winner": passed[0] if solved else None,
        "mirror": mirror_rec,
    }


def _occ_pair(p: dict | None, m: dict | None, key: str) -> dict[str, float | None]:
    def get(rec: dict | None) -> float | None:
        if rec is None or "measured" not in rec:
            return None
        v = rec["measured"][key]
        return None if isinstance(v, float) and math.isnan(v) else v
    return {"P": get(p), "mirror": get(m)}


def fixed_linear_entry(
    base_entry: dict,
    result: dict,
    verdict: dict,
    positive_controls: dict[int, dict[str, float]],
    *,
    reference: Mat3,
    frames: LevelFrames,
) -> dict:
    """판정 결과로 레벨 항목을 만든다. solved면 옛 거절 기록을 `history`로 보존한다."""
    ref_list = _matrix_list(reference)
    ref_rec = next((c for c in result["candidates"] if c["rotation"] == ref_list), None)
    winner = verdict["winner"]
    evidence_p = winner if winner is not None else ref_rec
    evidence = {
        "gun_occlusion": _occ_pair(evidence_p, verdict["mirror"], "gun_occlusion"),
        "hmd_occlusion": _occ_pair(evidence_p, verdict["mirror"], "hmd_occlusion"),
        "flash_occlusion_report": _occ_pair(evidence_p, verdict["mirror"], "flash_occlusion"),
        "rule_doc": FIXED_LINEAR_RULE_DOC,
        "posthoc_parts": ["G_occ_gun", "uniqueness"],
    }

    def summary(rec: dict) -> dict:
        out = {k: rec[k] for k in ("rotation", "det", "translation", "passed_A", "passed_B", "gate_failures_A")}
        if "measured" in rec:
            out["measured"] = rec["measured"]
        else:
            out["screened_out"] = rec["screened_out"]
        return out

    attempt = {
        "linear_source": "fixed_signed_permutation",
        "rule_doc": FIXED_LINEAR_RULE_DOC,
        "scale": FIXED_LINEAR_SCALE,
        "fit_n": result["fit_n"],
        "holdout_n": result["holdout_n"],
        "holdout_overlap": result["holdout_overlap"],
        "occlusion_threshold": OCCLUSION_THRESHOLD,
        "occlusion_stop_before_m": OCCLUSION_STOP_BEFORE_M,
        "candidates_tried": len(result["candidates"]),
        "candidates_passed_A": verdict["candidates_passed_A"],
        "candidates_passed_A_and_B": verdict["candidates_passed_A_and_B"],
        "section_b_valid": verdict["section_b_valid"],
        "section_b_invalid_reasons": verdict["section_b_invalid_reasons"],
        "positive_controls": {str(k): v for k, v in sorted(positive_controls.items())},
        "passing_or_reference_candidates": [
            summary(c) for c in result["candidates"]
            if c["passed_A"] or c["rotation"] in (ref_list, _matrix_list(_plate_mirror(reference)))
        ],
        "handedness_evidence": evidence,
    }
    if not verdict["solved"]:
        entry = {k: v for k, v in base_entry.items() if k != "fixed_linear_attempt"}
        entry["fixed_linear_attempt"] = attempt
        return entry

    m = winner["measured"]
    fit = {k: m[k] for k in sorted(m)}
    fit["det_sign"] = winner["det"]
    fit["fit_eval_overlap"] = float(result["holdout_overlap"])
    fit["fit_n"] = float(result["fit_n"])
    fit["n_hits"] = float(result["holdout_n"])
    fit["scale_ratio_to_declared"] = 0.0
    history = list(base_entry.get("history", []))
    history.append({k: v for k, v in base_entry.items() if k not in ("history", "fixed_linear_attempt")})
    return {
        "level": base_entry["level"],
        "provenance": "solved",
        "solved_transform": {
            "rotation": winner["rotation"],
            "translation": winner["translation"],
            "scale": FIXED_LINEAR_SCALE,
        },
        "declared_scale": FIXED_LINEAR_SCALE,
        "fit": fit,
        "source_fbx": base_entry["source_fbx"],
        "uv_convention": UV_CONVENTION,
        "uv_convention_source": "measured",
        "linear_source": "fixed_signed_permutation",
        "handedness_source": "data:occlusion_gun+hmd",
        "handedness_evidence": evidence,
        "not_gated": {
            "reason": "12자유도 경로 전용 게이트(G4 aniso, G7, G8, G9) — 규칙 문서 §A.3",
            "scale_ratio_to_declared_is_by_construction": True,
        },
        "fixed_linear_attempt": attempt,
        "workers": sorted({_worker_of(name) for name in frames.sessions}),
        "session_count": len(set(frames.sessions)),
        "history": history,
    }


def _plate_mirror(reference: Mat3) -> Mat3:
    """월드 X(레벨 1 판 법선) 반사를 기준 선형부에 합성한 후보."""
    mx: Mat3 = ((-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    return rotation_math.matmul(mx, reference)


GLOBAL_IMPORT_LINEAR: Final[Mat3] = ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (-1.0, 0.0, 0.0))


def _control_worker(level: int) -> tuple[int, dict[str, float]]:
    mesh = read_fbx(DEFAULT_MODELINGS_ROOT / LEVEL_FBX[level])
    frames = collect_level_frames_by_session(level)
    out = recorded_control_occlusion(level, mesh, frames)
    print(f"control level {level}: {out}", flush=True)
    return level, out


def _fixed_worker(level: int) -> tuple[int, dict, list]:
    mesh = read_fbx(DEFAULT_MODELINGS_ROOT / LEVEL_FBX[level])
    frames = collect_level_frames_by_session(level)
    # 기준 선형부(P)와 판 법선 거울 후보는 선별 없이 전량 잰다 — 판정은 같고, 레벨 2 진단
    # (G2 max 초과가 앞 100점에 걸려 선별되던 P)의 수치를 기록에 남기기 위해서다.
    result = solve_fixed_linear(
        level, mesh, frames, always_measure=(GLOBAL_IMPORT_LINEAR, _plate_mirror(GLOBAL_IMPORT_LINEAR))
    )
    passing = [(c["det"], c.get("gate_failures_A"), c.get("passed_B")) for c in result["candidates"] if "measured" in c]
    print(f"fixed level {level}: fully evaluated {len(passing)}: {passing}", flush=True)
    return level, result, frames.sessions


def fixed_linear_main(out_path: Path) -> None:
    """레벨 3~10 양성 대조 → 레벨 1·2 선형부 고정 → §B 판정 → 레벨 1·2 항목만 다시 쓴다."""
    from multiprocessing import Pool

    with Pool(10) as pool:
        controls_async = pool.map_async(_control_worker, [3, 4, 5, 6, 7, 8, 9, 10])
        fixed_async = pool.map_async(_fixed_worker, [1, 2])
        controls = dict(controls_async.get())
        fixed = {lv: (res, sess) for lv, res, sess in fixed_async.get()}

    positive = {lv: v["hmd_occlusion"] for lv, v in controls.items()}
    entries = json.loads(ALIGNMENT_PATH.read_text(encoding="utf-8"))
    report: dict = {"controls": {str(k): v for k, v in sorted(controls.items())}, "levels": {}}
    mirror = _plate_mirror(GLOBAL_IMPORT_LINEAR)
    for level in (1, 2):
        result, sessions = fixed[level]
        verdict = judge_fixed_linear(result, positive, mirror)
        if level == 2:
            verdict = {**verdict, "solved": False, "winner": None}  # 사용자 결정 C: 진단만
        frames = LevelFrames([], [], list(sessions), [], [], [])
        i = next(k for k, e in enumerate(entries) if int(e["level"]) == level)
        entries[i] = fixed_linear_entry(entries[i], result, verdict, controls,
                                        reference=GLOBAL_IMPORT_LINEAR, frames=frames)
        report["levels"][str(level)] = {
            "verdict": {k: v for k, v in verdict.items() if k not in ("winner", "mirror")},
            "candidates": result["candidates"],
            "holdout_overlap": result["holdout_overlap"],
        }
        print(f"level {level}: {report['levels'][str(level)]['verdict']}", flush=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    ALIGNMENT_PATH.write_text(json.dumps(entries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


# =============================================================================
# §D 사전등록 — HMD가 착탄면의 앞면 쪽에 있는가
# 문서: docs/measurements/2026-09-15-level1-front-face-prereg.md (측정 코드 첫 실행 전 커밋)
# §B(HMD→착탄 선분 가림)는 방 모양 레벨 양성 대조로 무효가 됐다. 이 절은 그 뒤에 새로
# 사전등록한 게이트이고, 규칙 문서 §A·§B 코드는 한 줄도 바꾸지 않는다.
# =============================================================================

FRONT_FACE_PREREG_DOC: Final = "docs/measurements/2026-09-15-level1-front-face-prereg.md"
FRONT_FACE_MIN_FRAMES: Final = 1000
FRONT_FACE_DEGENERATE: Final = 1e-12


def hit_face_normals(placed_mesh: Mesh, det_sign: float, hits: Sequence[Vec3]) -> list[Vec3 | None]:
    """착탄점마다 3D 최근접 삼각형(동률은 낮은 인덱스)의 **기하 앞면 법선**.

    `n = sign(det R) · (b − a) × (c − a) / |·|`(사전등록 §1-2). 법선 벡터는 직교 선형부 L에
    대해 `L·n_model`로 옮겨 가지만 옮긴 꼭짓점의 외적은 `det(L)·L·n_model`이다 — 반사는
    감김을 뒤집으므로 부호를 되돌린다. 이 부호를 빼면 레벨 1의 P와 판 법선 거울이 같은 법선을
    받아 게이트가 둘을 가르지 못한다(`test_back_face_fraction_uses_the_geometric_front_normal`).
    """
    if det_sign not in (1.0, -1.0):
        raise ValueError("hit_face_normals: det_sign must be +1.0 or -1.0")
    triangles = _world_triangles(placed_mesh)
    if not triangles:
        raise ValueError("hit_face_normals: mesh has no triangles")
    normals: list[Vec3 | None] = []
    for hit in hits:
        _distance, _point, index = _nearest_on_mesh(hit, triangles)
        a, b, c = triangles[index]
        raw = _cross(_sub(b, a), _sub(c, a))
        norm = math.sqrt(_dot(raw, raw))
        normals.append(None if norm < FRONT_FACE_DEGENERATE else _mul3(raw, det_sign / norm))
    return normals


def back_face_fraction(
    normals: Sequence[Vec3 | None],
    origins: Sequence[Vec3],
    hits: Sequence[Vec3],
    min_frames: int = FRONT_FACE_MIN_FRAMES,
) -> dict[str, float | int | None]:
    """원점이 착탄면 앞면 법선의 뒤편(`dot(origin − hit, n) ≤ 0`)인 프레임 비율(사전등록 §1-4).

    퇴화 삼각형 프레임은 분모에서 뺀다. 남은 프레임이 `min_frames` 미만이면 `fraction`은
    None(측정 불가)이다 — 0으로 두면 양성 대조가 "통과"로 읽힌다.
    """
    if not (len(normals) == len(origins) == len(hits)):
        raise ValueError("back_face_fraction: normals, origins and hits must be index-paired")
    behind = 0
    used = 0
    degenerate = 0
    for normal, origin, hit in zip(normals, origins, hits):
        if normal is None:
            degenerate += 1
            continue
        used += 1
        if _dot(_sub(origin, hit), normal) <= 0.0:
            behind += 1
    fraction = behind / used if used > 0 and used >= min_frames else None
    return {"fraction": fraction, "behind": behind, "n": used, "excluded_degenerate": degenerate}


def face_side_report(
    mesh: Mesh,
    rotation: Mat3,
    scale: float,
    translation: Vec3,
    frames: "LevelFrames",
    indices: Sequence[int],
    min_frames: int = FRONT_FACE_MIN_FRAMES,
) -> dict[str, dict]:
    """HMD(판정)·건·플래쉬(보고) 앞면 판정. 최근접 삼각형은 착탄점마다 한 번만 찾는다."""
    placed = _placed_mesh(mesh, rotation, scale, translation)
    det_sign = 1.0 if _mat3_det(rotation) > 0 else -1.0
    hits = [frames.hits[i] for i in indices]
    normals = hit_face_normals(placed, det_sign, hits)
    flash_pos = [k for k, i in enumerate(indices) if frames.flashes[i] is not None]
    return {
        "hmd_back_face": back_face_fraction(normals, [frames.heads[i] for i in indices], hits, min_frames),
        "gun_back_face": back_face_fraction(normals, [frames.guns[i] for i in indices], hits, min_frames),
        "flash_back_face": back_face_fraction(
            [normals[k] for k in flash_pos],
            [frames.flashes[indices[k]] for k in flash_pos],  # type: ignore[misc]
            [hits[k] for k in flash_pos],
            min_frames,
        ),
    }


def recorded_front_face(level: int, mesh: Mesh, frames: "LevelFrames", *, seed: int = 0,
                        fit_sample: int = 2000, eval_cap: int = 1500) -> dict[str, dict]:
    """§D 양성 대조: 기록된 변환 그대로, 규칙 문서 §A.1과 같은 홀드아웃."""
    alignment = load_alignment(level)
    if not is_solved(alignment.provenance):
        raise ValueError(f"level {level} is not a solved control")
    _fit, eval_idx = split_fit_holdout(len(frames.hits), fit_sample, eval_cap, seed)
    return face_side_report(
        mesh, _rotation_matrix(alignment), alignment.scale, alignment.translation, frames, eval_idx
    )


def _hmd_back(record: dict | None) -> float | None:
    if record is None or "front_face" not in record:
        return None
    return record["front_face"]["hmd_back_face"]["fraction"]


def judge_front_face(result: dict, positive_controls: Mapping[int, float | None], mirror: Mat3) -> dict:
    """사전등록 §3·§4 그대로. `positive_controls`는 레벨 → G_back_hmd(None이면 측정 불가)."""
    required = {3, 4, 5, 6, 7, 8, 9, 10}
    missing = sorted(required - set(positive_controls))
    over = {lv: v for lv, v in positive_controls.items() if v is None or not (v <= OCCLUSION_THRESHOLD)}
    mirror_list = _matrix_list(mirror)
    mirror_rec = next((c for c in result["candidates"] if c["rotation"] == mirror_list), None)
    mirror_value = _hmd_back(mirror_rec)
    reasons: list[str] = []
    if missing:
        reasons.append(f"positive controls missing: {missing}")
    if over:
        reasons.append(f"positive control hmd_back_face > {OCCLUSION_THRESHOLD} or unmeasurable: {over}")
    if mirror_value is None or not (mirror_value > OCCLUSION_THRESHOLD):
        reasons.append(f"negative control (mirror) hmd_back_face {mirror_value} not > {OCCLUSION_THRESHOLD}")
    valid = not reasons
    passed_a = [c for c in result["candidates"] if c.get("passed_A")]
    passed = [c for c in passed_a if _hmd_back(c) is not None and _hmd_back(c) <= OCCLUSION_THRESHOLD]  # type: ignore[operator]
    solved = valid and len(passed) == 1
    return {
        "section_d_valid": valid,
        "section_d_invalid_reasons": reasons,
        "candidates_passed_A": len(passed_a),
        "candidates_passed_A_and_D": len(passed) if valid else None,
        "solved": solved,
        "winner": passed[0] if solved else None,
        "mirror": mirror_rec,
    }


def front_face_entry(
    base_entry: dict,
    result: dict,
    verdict: dict,
    positive_controls: Mapping[int, dict],
    *,
    reference: Mat3,
    sessions: Sequence[str],
) -> dict:
    """§D 판정으로 레벨 항목을 만든다. 옛 기록(`rejected_by`, `fixed_linear_attempt`)은 지우지 않는다."""
    ref_list = _matrix_list(reference)
    ref_rec = next((c for c in result["candidates"] if c["rotation"] == ref_list), None)
    winner = verdict["winner"]
    p_rec = winner if winner is not None else ref_rec
    mirror_rec = verdict["mirror"]

    def face_pair(key: str) -> dict[str, float | None]:
        def get(rec: dict | None) -> float | None:
            return None if rec is None or "front_face" not in rec else rec["front_face"][key]["fraction"]
        return {"P": get(p_rec), "mirror": get(mirror_rec)}

    evidence = {
        "gun_occlusion": _occ_pair(p_rec, mirror_rec, "gun_occlusion"),
        "hmd_occlusion": _occ_pair(p_rec, mirror_rec, "hmd_occlusion"),
        "flash_occlusion_report": _occ_pair(p_rec, mirror_rec, "flash_occlusion"),
        "hmd_back_face": face_pair("hmd_back_face"),
        "gun_back_face_report": face_pair("gun_back_face"),
        "flash_back_face_report": face_pair("flash_back_face"),
        "rule_doc": FIXED_LINEAR_RULE_DOC,
        "front_face_prereg": FRONT_FACE_PREREG_DOC,
        "posthoc_parts": ["G_occ_gun", "uniqueness"],
        "void_gates": ["G_occ_hmd"],
    }
    attempt = {
        "prereg": FRONT_FACE_PREREG_DOC,
        "threshold": OCCLUSION_THRESHOLD,
        "min_frames": FRONT_FACE_MIN_FRAMES,
        "holdout_n": result["holdout_n"],
        "holdout_overlap": result["holdout_overlap"],
        "positive_controls": {str(k): v for k, v in sorted(positive_controls.items())},
        "section_d_valid": verdict["section_d_valid"],
        "section_d_invalid_reasons": verdict["section_d_invalid_reasons"],
        "candidates_passed_A": verdict["candidates_passed_A"],
        "candidates_passed_A_and_D": verdict["candidates_passed_A_and_D"],
        "candidates": [
            {k: c[k] for k in ("rotation", "det", "translation", "passed_A", "gate_failures_A", "front_face")}
            for c in result["candidates"]
            if "front_face" in c
        ],
        "handedness_evidence": evidence,
    }
    if not verdict["solved"]:
        entry = {k: v for k, v in base_entry.items() if k != "front_face_attempt"}
        entry["front_face_attempt"] = attempt
        return entry

    m = winner["measured"]
    fit = {k: m[k] for k in sorted(m)}
    fit["det_sign"] = winner["det"]
    fit["fit_eval_overlap"] = float(result["holdout_overlap"])
    fit["fit_n"] = float(result["fit_n"])
    fit["n_hits"] = float(result["holdout_n"])
    fit["scale_ratio_to_declared"] = 0.0
    history = list(base_entry.get("history", []))
    history.append(
        {k: v for k, v in base_entry.items() if k not in ("history", "fixed_linear_attempt", "front_face_attempt")}
    )
    entry = {
        "level": base_entry["level"],
        "provenance": "solved",
        "solved_transform": {
            "rotation": winner["rotation"],
            "translation": winner["translation"],
            "scale": FIXED_LINEAR_SCALE,
        },
        "declared_scale": FIXED_LINEAR_SCALE,
        "fit": fit,
        "source_fbx": base_entry["source_fbx"],
        "uv_convention": UV_CONVENTION,
        "uv_convention_source": "measured",
        "linear_source": "fixed_signed_permutation",
        "handedness_source": "data:occlusion_gun+hmd_front_face",
        "handedness_evidence": evidence,
        "not_gated": {
            "reason": "12자유도 경로 전용 게이트(G4 aniso, G7, G8, G9) — 규칙 문서 §A.3",
            "scale_ratio_to_declared_is_by_construction": True,
        },
        "front_face_attempt": attempt,
        "workers": sorted({_worker_of(name) for name in sessions}),
        "session_count": len(set(sessions)),
        "history": history,
    }
    if "fixed_linear_attempt" in base_entry:
        entry["fixed_linear_attempt"] = base_entry["fixed_linear_attempt"]
    return entry


def _front_control_worker(level: int) -> tuple[int, dict]:
    mesh = read_fbx(DEFAULT_MODELINGS_ROOT / LEVEL_FBX[level])
    frames = collect_level_frames_by_session(level)
    out = recorded_front_face(level, mesh, frames)
    print(f"front-face control level {level}: {out}", flush=True)
    return level, out


def _front_fixed_worker(level: int) -> tuple[int, dict, list]:
    mesh = read_fbx(DEFAULT_MODELINGS_ROOT / LEVEL_FBX[level])
    frames = collect_level_frames_by_session(level)
    result = solve_fixed_linear(
        level, mesh, frames, always_measure=(GLOBAL_IMPORT_LINEAR, _plate_mirror(GLOBAL_IMPORT_LINEAR))
    )
    for candidate in result["candidates"]:
        if "measured" not in candidate:
            continue
        flat = candidate["rotation"]
        rotation: Mat3 = tuple(tuple(float(x) for x in flat[r * 3 : r * 3 + 3]) for r in range(3))  # type: ignore[assignment]
        translation = tuple(float(x) for x in candidate["translation"])
        candidate["front_face"] = face_side_report(
            mesh, rotation, FIXED_LINEAR_SCALE, translation, frames, result["eval_indices"]  # type: ignore[arg-type]
        )
    measured = [(c["det"], c["passed_A"], _hmd_back(c)) for c in result["candidates"] if "front_face" in c]
    print(f"front-face fixed level {level}: {measured}", flush=True)
    return level, result, frames.sessions


def front_face_main(out_path: Path) -> None:
    """§D: 레벨 3~10 양성 대조 → 레벨 1·2 후보 앞면 판정 → 판정 → 레벨 1·2 항목만 다시 쓴다.

    보고서(`out_path`)를 정렬 파일보다 먼저 쓴다 — 측정 시각(mtime)의 증거다.
    """
    from multiprocessing import Pool

    with Pool(10) as pool:
        controls_async = pool.map_async(_front_control_worker, [3, 4, 5, 6, 7, 8, 9, 10])
        fixed_async = pool.map_async(_front_fixed_worker, [1, 2])
        controls = dict(controls_async.get())
        fixed = {lv: (res, sess) for lv, res, sess in fixed_async.get()}

    positive = {lv: v["hmd_back_face"]["fraction"] for lv, v in controls.items()}
    entries = json.loads(ALIGNMENT_PATH.read_text(encoding="utf-8"))
    report: dict = {"prereg": FRONT_FACE_PREREG_DOC, "controls": {str(k): v for k, v in sorted(controls.items())}, "levels": {}}
    mirror = _plate_mirror(GLOBAL_IMPORT_LINEAR)
    new_entries = list(entries)
    for level in (1, 2):
        result, sessions = fixed[level]
        verdict = judge_front_face(result, positive, mirror)
        if level == 2:
            verdict = {**verdict, "solved": False, "winner": None}  # 사용자 결정 대기(C): 진단만
        i = next(k for k, e in enumerate(entries) if int(e["level"]) == level)
        new_entries[i] = front_face_entry(entries[i], result, verdict, controls,
                                          reference=GLOBAL_IMPORT_LINEAR, sessions=sessions)
        report["levels"][str(level)] = {
            "verdict": {k: v for k, v in verdict.items() if k not in ("winner", "mirror")},
            "candidates": [
                {k: c.get(k) for k in ("rotation", "det", "passed_A", "gate_failures_A", "front_face")}
                for c in result["candidates"] if "front_face" in c
            ],
        }
        print(f"level {level}: {report['levels'][str(level)]['verdict']}", flush=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    ALIGNMENT_PATH.write_text(json.dumps(new_entries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


# =============================================================================
# §E 사전등록(2026-09-16) — 총구 가림 게이트 + 메시 전용 UV 제외 규칙
# 문서: docs/measurements/2026-09-16-level1-2-occlusion-exclusion-prereg.md
#       (커밋 c839a37, 측정 코드 첫 실행 **전**에 커밋)
#
# 앞선 두 게이트는 양성 대조에서 무효로 끝났고 그 코드는 한 줄도 바꾸지 않는다:
#   §B HMD 선분 가림  — L4 0.4133 / L5 0.4840 > 0.25 → 무효
#   §D HMD 앞면        — L5 0.2607 > 0.25 → 무효
# 이번 게이트는 **건의 총구**를 원점으로 쓴다. 문턱 0.05는 레벨 3~10에서만 정했다
# (사전등록 §1.5 표 2: 총구 원점 참해 가림 최댓값 0.0073).
# =============================================================================

MUZZLE_PREREG_DOC: Final = "docs/measurements/2026-09-16-level1-2-occlusion-exclusion-prereg.md"

# 건의 CSV `Position`은 총구가 아니라 **강체 원점**이다. 레벨 3~10에서
# `|hit − gun|×1000 − Distance (Real)`이 레벨과 무관하게 89.13~89.24 mm의 상수라는
# 실측(사전등록 표 1)에서 고른 값이다. 소수 넷째 자리에서 끊는다.
L_MUZZLE_M: Final = 0.0892
# `|hit − gun|`이 총구보다 짧거나 2 mm 안이면 총구가 착탄점 뒤에 놓여 선분이 뒤집힌다.
# 그런 프레임은 분모에서 뺀다(레벨 3~10에서는 0개였다).
MUZZLE_SHORT_FRAME_MARGIN_M: Final = 0.002
MUZZLE_OCCLUSION_THRESHOLD: Final = 0.05

H2_PER_FILE_CAP: Final = 200
H2_FRAME_CAP: Final = 1500
H2_MIN_FRAMES: Final = 1000

# §2.5 과다 제외 안전장치
EXCLUDED_HIT_FRACTION_MAX: Final = 0.10
EXCLUSION_MIN_DISTINCT_NORMALS: Final = 3

# E2(삼각분할 모호)의 판별 상수. 둘 다 기하 동일성 판정의 수치 허용치일 뿐 문턱이 아니다 —
# 제외 문턱 자체는 `GATES["uv_abs_resid_max_texel"]`(0.05 텍셀)을 그대로 쓴다.
UV_QUAD_COPLANAR_COS: Final = 1.0 - 1e-9
UV_QUAD_SEAM_UV: Final = 1e-9

# **G7의 범위**: 조건수·법선 다양성은 12자유도 폐형식 적합이 데이터로 결정되는지의
# 검사다. 자유도 3짜리 선형부 고정 경로에 걸면 "평면 착탄점이면 무조건 거절"이 되어
# 실패할 수 없는 거절이 된다(사전등록 §1.1). 아래 이름들은 12자유도 경로 전용이고
# `fixed_linear_gate_failures`가 그 경계를 **실행 시에** 지킨다.
TWELVE_DOF_ONLY_GATES: Final[tuple[str, ...]] = (
    "cond_min_over_max",
    "distinct_normals",
    "fit_linear_aniso",
    "linear_aniso",
    "scale_ratio_to_declared",
    "adversarial_runner_up_texel",
)


def fixed_linear_gate_failures(
    measured: Mapping[str, float], keys: Sequence[str] = FIXED_LINEAR_GATE_KEYS
) -> list[tuple[str, object, str]]:
    """선형부 고정 경로의 게이트 A. **12자유도 전용 게이트가 새면 멈춘다.**

    `gate_failures`는 `measured`에 있는 이름만 본다 — 그래서 누군가 `cond_min_over_max`를
    이 경로의 measured에 얹으면 조용히 G7이 켜진다. 그 사고를 문구가 아니라 예외로 막는다.
    """
    leaked = [k for k in keys if k in TWELVE_DOF_ONLY_GATES]
    if leaked:
        raise ValueError(
            f"fixed_linear_gate_failures: 12-DoF-only gates leaked into the fixed-linear path: {leaked} "
            f"(prereg {MUZZLE_PREREG_DOC} §1.1)"
        )
    return gate_failures({k: measured[k] for k in keys if k in measured})


# -----------------------------------------------------------------------------
# G_occ_muzzle — 원점은 총구다
# -----------------------------------------------------------------------------


def muzzle_origins(
    guns: Sequence[Vec3],
    hits: Sequence[Vec3],
    *,
    length_m: float = L_MUZZLE_M,
    margin_m: float = MUZZLE_SHORT_FRAME_MARGIN_M,
) -> tuple[list[Vec3], list[Vec3], int]:
    """강체 원점 `gun`에서 착탄 방향으로 `length_m` 나아간 총구 위치.

    반환 `(origins, hits_kept, short_frames)`. `|hit − gun| ≤ length_m + margin_m`인
    프레임은 총구가 착탄점 뒤에 놓여 선분이 뒤집히므로 **분모에서 뺀다**.
    """
    if len(guns) != len(hits):
        raise ValueError("muzzle_origins: guns and hits must be index-paired")
    origins: list[Vec3] = []
    kept: list[Vec3] = []
    short = 0
    for gun, hit in zip(guns, hits):
        distance = math.dist(gun, hit)
        if distance <= length_m + margin_m:
            short += 1
            continue
        k = length_m / distance
        origins.append(
            (gun[0] + (hit[0] - gun[0]) * k, gun[1] + (hit[1] - gun[1]) * k, gun[2] + (hit[2] - gun[2]) * k)
        )
        kept.append(hit)
    return origins, kept, short


def muzzle_occlusion(
    placed_mesh: Mesh, guns: Sequence[Vec3], hits: Sequence[Vec3]
) -> dict[str, float]:
    """G_occ_muzzle = 총구→착탄 선분이 배치 메시를 뚫는 프레임 비율.

    교차 판정은 `occlusion_fraction`과 **같은 함수·같은 규칙**이고 원점만 다르다.
    """
    origins, kept, short = muzzle_origins(guns, hits)
    if not kept:
        return {"muzzle_occlusion": math.nan, "muzzle_occlusion_n": 0.0, "muzzle_short_frames": float(short)}
    return {
        "muzzle_occlusion": occlusion_fraction(placed_mesh, origins, kept),
        "muzzle_occlusion_n": float(len(kept)),
        "muzzle_short_frames": float(short),
    }


def muzzle_offset_stats(frames: LevelFrames, distances_real_mm: Sequence[float]) -> dict[str, float]:
    """`|hit − gun|×1000 − Distance(Real)`의 분포. 사전등록 표 1의 재현용."""
    deltas = [
        math.dist(gun, hit) * 1000.0 - real
        for gun, hit, real in zip(frames.guns, frames.hits, distances_real_mm)
    ]
    med, p95, mx = _percentiles(deltas)
    return {"n": float(len(deltas)), "median_mm": med, "p95_mm": p95, "max_mm": mx}


# -----------------------------------------------------------------------------
# 메시 전용 UV 제외 규칙 E1·E2·E3 — 착탄점·잔차·판정을 한 번도 읽지 않는다
# -----------------------------------------------------------------------------


def uv_degenerate_triangles(mesh: Mesh) -> frozenset[int]:
    """E1 — UV 공간 넓이 ≤ `UV_DEGENERATE_AREA`. `build_uv_index`와 같은 판정이다."""
    return frozenset(
        t for t in range(len(mesh.triangles)) if _uv_area(_tri_uv_corners(mesh, t)) <= UV_DEGENERATE_AREA
    )


def uv_quad_ambiguity_defects(mesh: Mesh) -> dict[int, float]:
    """E2 — 삼각형마다 "삼각분할을 바꾸면 기대 UV가 얼마나 달라지는가"(텍셀).

    같은 평면에서 변 하나를 공유하는 두 삼각형 t·t'가 한 사변형에서 왔는데 그 사변형의
    UV가 3D 위치의 아핀 함수가 아니면, 내부 한 점의 기대 UV는 **어느 삼각분할을 쓰느냐에
    따라 달라진다**. 그 어긋남을 t'의 바깥 꼭짓점 Q에서 잰다(사전등록 §2.2):

      Q의 3D 위치를 t의 세 꼭짓점에 대한 무게중심 좌표로 쓰고 → 그 좌표를 t의 UV 세
      꼭짓점에 그대로 적용 → 예측 UV. D = |예측 UV − t'가 Q에 준 UV| × 2048 텍셀.

    시임(A·B의 UV가 두 삼각형에서 다름)은 서로 다른 UV 섬이므로 건너뛴다.
    **메시만 본다** — 착탄점도 정렬도 잔차도 읽지 않는다.
    """
    if not mesh.uv_indices:
        return {}
    n = len(mesh.triangles)
    normals = [_tri_normal(_tri_vertices(mesh, t)) for t in range(n)]
    by_edge: dict[tuple[int, int], list[int]] = {}
    for t in range(n):
        vi = mesh.triangles[t]
        for a, b in ((0, 1), (1, 2), (2, 0)):
            by_edge.setdefault((min(vi[a], vi[b]), max(vi[a], vi[b])), []).append(t)

    def uv_of(triangle: int, vertex: int) -> tuple[float, float]:
        pos = mesh.triangles[triangle].index(vertex)
        return mesh.uvs[mesh.uv_indices[triangle][pos]]

    defects: dict[int, float] = {}
    seen: set[tuple[int, int]] = set()
    for neighbours in by_edge.values():
        for t in neighbours:
            for other in neighbours:
                if t == other or (t, other) in seen:
                    continue
                seen.add((t, other))
                shared = sorted(set(mesh.triangles[t]) & set(mesh.triangles[other]))
                if len(shared) != 2:
                    continue
                nt, no = normals[t], normals[other]
                if nt is None or no is None or abs(_dot(nt, no)) < UV_QUAD_COPLANAR_COS:
                    continue
                if any(math.dist(uv_of(t, v), uv_of(other, v)) > UV_QUAD_SEAM_UV for v in shared):
                    continue  # 시임 — 서로 다른 UV 섬이다
                rest = [v for v in mesh.triangles[other] if v not in shared]
                if len(rest) != 1:
                    continue
                q = mesh.vertices[rest[0]]
                bary = _barycentric(q, *_tri_vertices(mesh, t))
                corners = _tri_uv_corners(mesh, t)
                predicted = (
                    sum(bary[k] * corners[k][0] for k in range(3)),
                    sum(bary[k] * corners[k][1] for k in range(3)),
                )
                actual = uv_of(other, rest[0])
                d = math.hypot(predicted[0] - actual[0], predicted[1] - actual[1]) * UV_ATLAS_TEXELS
                defects[t] = max(defects.get(t, 0.0), d)
    return defects


def uv_quad_ambiguous_triangles(mesh: Mesh, *, threshold_texel: float | None = None) -> frozenset[int]:
    """E2의 제외 집합. 문턱은 `GATES["uv_abs_resid_max_texel"]`(0.05) **그대로**다 —
    "게이트가 허용하는 오차보다 큰 모호가 메시 자체에 있다"가 제외 사유이므로 문턱은
    게이트의 것과 같아야 한다(사전등록 §2.2). 새로 고른 값이 아니다."""
    if threshold_texel is None:
        threshold_texel = GATES["uv_abs_resid_max_texel"]
    return frozenset(t for t, d in uv_quad_ambiguity_defects(mesh).items() if d > threshold_texel)


def atlas_overlap_triangles(mesh: Mesh, index: UvIndex, *, overlap_resolution: int = 1024) -> frozenset[int]:
    """E3 — 기존 `audit_atlas`의 겹침 검사에 걸리는 삼각형(UV→3D가 다가인 면)."""
    out: set[int] = set()
    for hits_in_cell in _atlas_overlap_cells(mesh, index, overlap_resolution).values():
        if len(hits_in_cell) < 2:
            continue
        points = [model_point_for_hit(mesh, t, bary) for t, bary in hits_in_cell]
        if max(math.dist(points[0], q) for q in points) > UV_SEAM_SAME_POINT:
            out.update(t for t, _bary in hits_in_cell)
    return frozenset(out)


def mesh_only_uv_exclusions(mesh: Mesh, index: UvIndex | None = None) -> dict[str, object]:
    """E1·E2·E3과 그 합집합. **FBX만으로 계산된다.**"""
    if index is None:
        index = build_uv_index(mesh)
    e1 = uv_degenerate_triangles(mesh)
    e2 = uv_quad_ambiguous_triangles(mesh)
    e3 = atlas_overlap_triangles(mesh, index)
    union = e1 | e2 | e3
    n = len(mesh.triangles)
    return {
        "rule": "E1|E2|E3",
        "threshold_texel": GATES["uv_abs_resid_max_texel"],
        "n_triangles": n,
        "E1": sorted(e1),
        "E2": sorted(e2),
        "E3": sorted(e3),
        "excluded": union,
        "excluded_sorted": sorted(union),
        "n_E1": len(e1),
        "n_E2": len(e2),
        "n_E3": len(e3),
        "n_excluded": len(union),
        "excluded_triangle_fraction": len(union) / n if n else 0.0,
    }


# -----------------------------------------------------------------------------
# H2 — 한 번도 열리지 않은 파일로 만든 새 홀드아웃 (사전등록 §3.2)
# -----------------------------------------------------------------------------


def unread_level_files(
    level: int, consumed_paths: Sequence[str], professionals_root: Path = DEFAULT_PROFESSIONALS_ROOT
) -> list[Path]:
    """`collect_level_frames_by_session`이 **한 번도 열지 않은** 학습모드 파일, 경로 오름차순.

    목록을 하드코딩하지 않는다 — 소비 알고리즘 자신이 기록한 `LevelFrames.paths`를 뺀다.
    """
    consumed = set(consumed_paths)
    return [p for p in sorted(professionals_root.glob(f"**/{level}_*.csv")) if str(p) not in consumed]


def collect_frames_from_files(
    paths: Sequence[Path], *, per_file_cap: int = H2_PER_FILE_CAP, cap: int = H2_FRAME_CAP
) -> LevelFrames:
    """`collect_level_frames_by_session`과 **같은 필터**로, 주어진 파일만 경로 순서대로."""
    import measured_pose
    import simulator_csv

    frames = LevelFrames([], [], [], [], [], [], [])
    for path in paths:
        if len(frames.hits) >= cap:
            break
        frames.paths.append(str(path))
        rows = simulator_csv.read_rows(path)
        taken = 0
        for frame in measured_pose.measured_frames(rows):
            if frame.hit is None or frame.uv is None or not frame.spraying:
                continue
            frames.hits.append(frame.hit)
            frames.uvs.append(frame.uv)
            frames.sessions.append("/".join(path.parts[-3:-1]))
            frames.guns.append(frame.right)
            frames.heads.append(frame.head)
            frames.flashes.append(frame.left)
            taken += 1
            if taken >= per_file_cap or len(frames.hits) >= cap:
                break
    return frames


# -----------------------------------------------------------------------------
# 판정 — 선형부 고정 + H2 + 총구 가림 + (레벨 2) 메시 전용 UV 제외
# -----------------------------------------------------------------------------


def solve_fixed_linear_muzzle(
    level: int,
    mesh: Mesh,
    fit_frames: LevelFrames,
    holdout: LevelFrames,
    *,
    index: UvIndex | None = None,
    fit_sample: int = 2000,
    eval_cap: int = 1500,
    seed: int = 0,
    candidates: Sequence[Mat3] | None = None,
    screen: int = 100,
    excluded_triangles: Collection[int] = (),
    cross_worker: bool = True,
    always_measure: Sequence[Mat3] = (),
    unread_files: Sequence[str] = (),
) -> dict:
    """48 후보마다 t를 다시 풀고(적합 표본), **H2에서** 게이트 A와 G_occ_muzzle을 잰다.

    `solve_fixed_linear`(2026-09-15 경로)은 손대지 않았다 — 이것은 표본이 다르고
    게이트가 다른 **다른 경로**다. G7은 여기 없다(`fixed_linear_gate_failures`가 지킨다).
    """
    if index is None:
        index = build_uv_index(mesh)
    fit_idx, _unused = split_fit_holdout(len(fit_frames.hits), fit_sample, eval_cap, seed)
    if len(fit_idx) < fit_sample:
        raise ValueError(f"solve_fixed_linear_muzzle: only {len(fit_frames.hits)} fit frames for {fit_sample}")
    fit_hits = [fit_frames.hits[i] for i in fit_idx]
    fit_uvs = [fit_frames.uvs[i] for i in fit_idx]
    src, dst, fit_miss, fit_amb = _correspondences(mesh, index, fit_hits, fit_uvs)
    if len(src) < 50:
        raise ValueError(f"solve_fixed_linear_muzzle: only {len(src)} usable correspondences")

    eval_hits = holdout.hits
    eval_uvs = holdout.uvs
    eval_idx = list(range(len(eval_hits)))
    ev_src, ev_dst, miss_h2, amb_h2 = _correspondences(mesh, index, eval_hits, eval_uvs)
    shared = congruence_stats(ev_src, ev_dst, FIXED_LINEAR_SCALE, seed=seed)
    atlas = audit_atlas(mesh, index, seed=seed)
    shared.update(atlas)
    # G1의 조회 실패·모호도는 **제외 없이 H2 전량**에서 센다 — 제외된 면 위의 착탄점도
    # 조회 실패 통계에는 그대로 센다(사전등록 §2.4).
    shared["uv_miss_fraction"] = miss_h2
    shared["uv_ambiguous_fraction"] = amb_h2
    shared["uv_lookup_sample"] = float(len(eval_hits))

    results: list[dict] = []
    for rotation in (candidates if candidates is not None else signed_permutations()):
        translation, n_inliers = solve_translation_fixed(rotation, FIXED_LINEAR_SCALE, src, dst)
        record: dict = {
            "rotation": _matrix_list(rotation),
            "det": _mat3_det(rotation),
            "translation": list(translation),
            "fit_inliers": n_inliers,
        }
        probe = measure_placement(
            mesh, rotation, FIXED_LINEAR_SCALE, translation, eval_hits[:screen], eval_uvs[:screen],
            excluded_triangles=excluded_triangles,
        )
        exempt = any(rotation == keep for keep in always_measure)
        if not exempt and (
            probe["max_mm"] > GATES["max_mm"] or probe["uv_abs_resid_max_texel"] > GATES["uv_abs_resid_max_texel"]
        ):
            record["screened_out"] = {"sample": float(min(screen, len(eval_hits))), **probe}
            record["gate_failures_A"] = [
                name for name in ("max_mm", "uv_abs_resid_max_texel") if probe[name] > GATES[name]
            ]
            record["passed_A"] = False
            record["passed_occ"] = None
            results.append(record)
            continue
        measured = {**shared, **measure_placement(
            mesh, rotation, FIXED_LINEAR_SCALE, translation, eval_hits, eval_uvs,
            excluded_triangles=excluded_triangles,
        )}
        placed = _placed_mesh(mesh, rotation, FIXED_LINEAR_SCALE, translation)
        measured.update(muzzle_occlusion(placed, holdout.guns, eval_hits))
        measured.update(origin_occlusions(mesh, rotation, FIXED_LINEAR_SCALE, translation, holdout, eval_idx))
        if cross_worker:
            measured.update(_cross_worker_fixed(
                mesh, index, rotation, holdout, seed, excluded_triangles=excluded_triangles
            ))
        failures_a = [name for name, _v, _n in fixed_linear_gate_failures(measured)]
        if cross_worker and measured["cross_session_pass"] < 1.0:
            failures_a.append("cross_session")
        record["measured"] = measured
        record["gate_failures_A"] = failures_a
        record["passed_A"] = not failures_a
        occ = measured["muzzle_occlusion"]
        record["passed_occ"] = bool(occ == occ and occ <= MUZZLE_OCCLUSION_THRESHOLD)
        results.append(record)

    return {
        "level": level,
        "prereg": MUZZLE_PREREG_DOC,
        "fit_n": len(fit_idx),
        "holdout_n": len(eval_hits),
        "holdout_kind": "unread_files",
        "holdout_files": list(holdout.paths),
        "unread_files": list(unread_files),
        # `fit_idx`/`eval_idx`는 서로 다른 배열(fit_frames.hits 대 holdout.hits)의
        # 위치일 뿐이라 정수로 비교해도 의미가 없다 — 이 경로에서 적합/홀드아웃을
        # 실제로 가르는 것은 파일이다(§3.2: H2는 fit_frames가 한 번도 열지 않은
        # 파일에서만 온다). 2645행과 같은 자리에서, 같은 방식으로 실제 겹침을 센다.
        "holdout_overlap": len(set(fit_frames.paths) & set(holdout.paths)),
        "fit_files": list(fit_frames.paths),
        "fit_uv_miss_fraction": fit_miss,
        "fit_uv_ambiguous_fraction": fit_amb,
        "excluded_triangles": sorted(excluded_triangles),
        "candidates": results,
    }


def judge_muzzle(
    result: dict, mirror: Mat3, *, exclusions: dict[str, object] | None = None
) -> dict:
    """사전등록 §1.6 + §2.5 그대로. **양성 대조는 판정 조건이 아니다** — 문턱을 그 8개
    값에서 뽑았으므로 구성상 통과한다(§1.5). 판별력은 전적으로 음성 대조에 달려 있다."""
    mirror_list = _matrix_list(mirror)
    mirror_rec = next((c for c in result["candidates"] if c["rotation"] == mirror_list), None)
    mirror_occ = (
        None if mirror_rec is None or "measured" not in mirror_rec
        else mirror_rec["measured"]["muzzle_occlusion"]
    )
    reasons: list[str] = []
    if result["holdout_n"] < H2_MIN_FRAMES:
        reasons.append(f"H2 holdout has {result['holdout_n']} frames, need >= {H2_MIN_FRAMES}")
    if mirror_occ is None or not (mirror_occ > MUZZLE_OCCLUSION_THRESHOLD):
        reasons.append(
            f"negative control (plate mirror) G_occ_muzzle {mirror_occ} not > {MUZZLE_OCCLUSION_THRESHOLD}"
        )
    passed_a = [c for c in result["candidates"] if c["passed_A"]]
    passed = [c for c in passed_a if c["passed_occ"]]
    if len(passed) != 1:
        reasons.append(f"candidates passing gate A and G_occ_muzzle: {len(passed)}, need exactly 1")

    over: list[str] = []
    if exclusions is not None:
        winner = passed[0] if len(passed) == 1 else None
        probe = winner if winner is not None else next(
            (c for c in result["candidates"] if "measured" in c), None
        )
        m = probe["measured"] if probe is not None else {}
        fraction = m.get("uv_excluded_hit_fraction")
        kept = m.get("uv_kept_hits")
        normals = m.get("uv_kept_distinct_normals")
        if fraction is None or fraction > EXCLUDED_HIT_FRACTION_MAX:
            over.append(f"excluded hit fraction {fraction} > {EXCLUDED_HIT_FRACTION_MAX}")
        if kept is None or kept < H2_MIN_FRAMES:
            over.append(f"frames left after exclusion {kept} < {H2_MIN_FRAMES}")
        if normals is None or normals < EXCLUSION_MIN_DISTINCT_NORMALS:
            over.append(f"distinct normals among kept hits {normals} < {EXCLUSION_MIN_DISTINCT_NORMALS}")
        reasons.extend(over)

    return {
        "prereg": MUZZLE_PREREG_DOC,
        "negative_control_muzzle_occlusion": mirror_occ,
        "occlusion_threshold": MUZZLE_OCCLUSION_THRESHOLD,
        "candidates_passed_A": len(passed_a),
        "candidates_passed_A_and_occ": len(passed),
        "over_exclusion_reasons": over,
        "not_solved_reasons": reasons,
        "solved": not reasons,
        "winner": passed[0] if len(passed) == 1 and not reasons else None,
        "mirror": mirror_rec,
        "confirmation_not_out_of_sample": True,
    }


def muzzle_entry(
    base_entry: dict,
    result: dict,
    verdict: dict,
    *,
    reference: Mat3,
    holdout: LevelFrames,
    exclusions: dict[str, object] | None,
    provenance: str,
) -> dict:
    """판정 결과로 레벨 항목을 만든다. 통과 못 하면 `provenance`를 올리지 않는다."""
    ref_list = _matrix_list(reference)
    mirror_list = _matrix_list(_plate_mirror(reference))
    ref_rec = next((c for c in result["candidates"] if c["rotation"] == ref_list), None)
    winner = verdict["winner"]
    p_rec = winner if winner is not None else ref_rec
    evidence = {
        "muzzle_occlusion": _occ_pair(p_rec, verdict["mirror"], "muzzle_occlusion"),
        "gun_origin_occlusion_report": _occ_pair(p_rec, verdict["mirror"], "gun_occlusion"),
        "hmd_occlusion_report": _occ_pair(p_rec, verdict["mirror"], "hmd_occlusion"),
        "occlusion_origin": "muzzle",
        "muzzle_offset_m": L_MUZZLE_M,
        "occlusion_threshold": MUZZLE_OCCLUSION_THRESHOLD,
        "prereg_doc": MUZZLE_PREREG_DOC,
        # 게이트를 고른 이유 자체가 레벨 1 결과였다(강체 원점에서 P 0.000 / 거울 1.000).
        # 새 파일로 다시 재도 그 사실은 지워지지 않는다.
        "posthoc_parts": ["G_occ_muzzle", "uniqueness"],
        "void_gates": ["G_occ_hmd", "G_front_face_hmd"],
        "confirmation_not_out_of_sample": True,
    }

    def summary(rec: dict) -> dict:
        out = {k: rec[k] for k in ("rotation", "det", "translation", "passed_A", "passed_occ", "gate_failures_A")}
        out["measured" if "measured" in rec else "screened_out"] = rec.get("measured", rec.get("screened_out"))
        return out

    attempt: dict = {
        "prereg_doc": MUZZLE_PREREG_DOC,
        "linear_source": "fixed_signed_permutation",
        "scale": FIXED_LINEAR_SCALE,
        "occlusion_origin": "muzzle",
        "muzzle_offset_m": L_MUZZLE_M,
        "occlusion_threshold": MUZZLE_OCCLUSION_THRESHOLD,
        "occlusion_stop_before_m": OCCLUSION_STOP_BEFORE_M,
        "gates_not_applied": list(TWELVE_DOF_ONLY_GATES),
        "fit_n": result["fit_n"],
        "holdout": {
            "kind": "unread_files",
            # 미개봉 파일은 27개(레벨 1)지만 파일당 200프레임 상한과 총 1,500프레임
            # 상한 때문에 실제로 **열린** 것은 그중 앞쪽 몇 개뿐이다. 두 수를 따로
            # 적는다 — 하나로 뭉치면 표본이 실제보다 넓어 보인다.
            "unread_files_available": len(result.get("unread_files", ())),
            "files_opened": len(result["holdout_files"]),
            "file_list": result["holdout_files"],
            # 미개봉 27개의 전량 목록은 사전등록 문서 §3.2와 측정 보고서에 있다 —
            # 정렬 기록에는 실제로 연 것만 싣는다.
            "frames": result["holdout_n"],
            "per_file_cap": H2_PER_FILE_CAP,
            "frame_cap": H2_FRAME_CAP,
            "session_disjoint": False,
        },
        "candidates_tried": len(result["candidates"]),
        "candidates_passed_A": verdict["candidates_passed_A"],
        "candidates_passed_A_and_occ": verdict["candidates_passed_A_and_occ"],
        "negative_control_muzzle_occlusion": verdict["negative_control_muzzle_occlusion"],
        "not_solved_reasons": verdict["not_solved_reasons"],
        "confirmation_not_out_of_sample": True,
        "passing_or_reference_candidates": [
            summary(c) for c in result["candidates"]
            if c["passed_A"] or c["rotation"] in (ref_list, mirror_list)
        ],
        "handedness_evidence": evidence,
    }
    if exclusions is not None:
        m = (p_rec or {}).get("measured", {})
        attempt["uv_exclusion"] = {
            "rule": exclusions["rule"],
            "threshold_texel": exclusions["threshold_texel"],
            "n_triangles": exclusions["n_triangles"],
            "excluded_tris": exclusions["n_excluded"],
            "excluded_triangle_fraction": exclusions["excluded_triangle_fraction"],
            "by_rule": {"E1": exclusions["n_E1"], "E2": exclusions["n_E2"], "E3": exclusions["n_E3"]},
            "excluded_triangle_ids": exclusions["excluded_sorted"],
            "excluded_hit_fraction": m.get("uv_excluded_hit_fraction"),
            "excluded_hits_resid": {
                "n": m.get("uv_excluded_hits"),
                "median_texel": m.get("uv_excluded_resid_median_texel"),
                "p95_texel": m.get("uv_excluded_resid_p95_texel"),
                "max_texel": m.get("uv_excluded_resid_max_texel"),
            },
            "kept_hits": m.get("uv_kept_hits"),
            "kept_distinct_normals": m.get("uv_kept_distinct_normals"),
            "over_exclusion_reasons": verdict["over_exclusion_reasons"],
            "applies_to": ["G2", "G6 (UV part)"],
        }

    if not verdict["solved"]:
        entry = {k: v for k, v in base_entry.items() if k != "muzzle_occlusion_attempt"}
        entry["muzzle_occlusion_attempt"] = attempt
        return entry

    m = winner["measured"]
    fit = {k: m[k] for k in sorted(m)}
    fit["det_sign"] = winner["det"]
    fit["fit_eval_overlap"] = 0.0
    fit["fit_n"] = float(result["fit_n"])
    fit["n_hits"] = float(result["holdout_n"])
    fit["scale_ratio_to_declared"] = 0.0
    history = list(base_entry.get("history", []))
    history.append({
        k: v for k, v in base_entry.items()
        if k not in ("history", "fixed_linear_attempt", "front_face_attempt", "muzzle_occlusion_attempt")
    })
    entry = {
        "level": base_entry["level"],
        "provenance": provenance,
        "solved_transform": {
            "rotation": winner["rotation"],
            "translation": winner["translation"],
            "scale": FIXED_LINEAR_SCALE,
        },
        "declared_scale": FIXED_LINEAR_SCALE,
        "fit": fit,
        "source_fbx": base_entry["source_fbx"],
        "uv_convention": UV_CONVENTION,
        "uv_convention_source": "measured",
        "linear_source": "fixed_signed_permutation",
        "handedness_source": "data:occlusion_muzzle",
        "handedness_evidence": evidence,
        "prereg_doc": MUZZLE_PREREG_DOC,
        "occlusion_origin": "muzzle",
        "muzzle_offset_m": L_MUZZLE_M,
        "occlusion_threshold": MUZZLE_OCCLUSION_THRESHOLD,
        "holdout": attempt["holdout"],
        "confirmation_not_out_of_sample": True,
        "not_gated": {
            "reason": "12자유도 경로 전용 게이트(G4 aniso, G7, G8, G9) — 사전등록 §1.3",
            "gates_not_applied": list(TWELVE_DOF_ONLY_GATES),
            "scale_ratio_to_declared_is_by_construction": True,
        },
        "muzzle_occlusion_attempt": attempt,
        "workers": sorted({_worker_of(name) for name in holdout.sessions}),
        "session_count": len(set(holdout.sessions)),
        "history": history,
    }
    if exclusions is not None:
        entry["uv_exclusion"] = attempt["uv_exclusion"]
    for keep in ("fixed_linear_attempt", "front_face_attempt"):
        if keep in base_entry:
            entry[keep] = base_entry[keep]
    return entry


MUZZLE_PROVENANCE: Final[dict[int, str]] = {
    1: "solved:fixed_linear_muzzle_occlusion",
    2: "solved:fixed_linear_uv_exclusion_meshonly",
}


def recorded_muzzle_control(
    level: int, mesh: Mesh, frames: LevelFrames, *, seed: int = 0,
    fit_sample: int = 2000, eval_cap: int = 1500,
) -> dict[str, float]:
    """양성 대조(기록용): 기록된 참해 그대로, 기존 홀드아웃에서 강체 원점·총구 원점 가림.

    **판정 조건이 아니다** — 문턱 0.05를 바로 이 8개 값에서 뽑았으므로 구성상 통과한다.
    """
    alignment = load_alignment(level)
    if not is_solved(alignment.provenance):
        raise ValueError(f"level {level} is not a solved control")
    _fit, eval_idx = split_fit_holdout(len(frames.hits), fit_sample, eval_cap, seed)
    rotation = _rotation_matrix(alignment)
    placed = _placed_mesh(mesh, rotation, alignment.scale, alignment.translation)
    hits = [frames.hits[i] for i in eval_idx]
    guns = [frames.guns[i] for i in eval_idx]
    out = {"rigid_origin_occlusion": occlusion_fraction(placed, guns, hits)}
    out.update(muzzle_occlusion(placed, guns, hits))
    return out


def exclusion_table_row(level: int, mesh: Mesh, index: UvIndex, frames: LevelFrames,
                        rotation: Mat3, scale: float, translation: Vec3,
                        indices: Sequence[int]) -> dict[str, object]:
    """한 레벨의 (제외 삼각형 수·비율, 제외 착탄점 수·비율, E1/E2/E3 몫). 표 5용."""
    ex = mesh_only_uv_exclusions(mesh, index)
    hits = [frames.hits[i] for i in indices]
    uvs = [frames.uvs[i] for i in indices]
    measured = measure_placement(
        mesh, rotation, scale, translation, hits, uvs, excluded_triangles=ex["excluded"]  # type: ignore[arg-type]
    )
    return {
        "level": level,
        "n_triangles": ex["n_triangles"],
        "n_E1": ex["n_E1"],
        "n_E2": ex["n_E2"],
        "n_E3": ex["n_E3"],
        "n_excluded": ex["n_excluded"],
        "excluded_triangle_fraction": ex["excluded_triangle_fraction"],
        "n_hits": len(hits),
        "excluded_hits": measured["uv_excluded_hits"],
        "excluded_hit_fraction": measured["uv_excluded_hit_fraction"],
        "kept_distinct_normals": measured["uv_kept_distinct_normals"],
        "excluded_resid_p95_texel": measured["uv_excluded_resid_p95_texel"],
        "excluded_resid_max_texel": measured["uv_excluded_resid_max_texel"],
        "kept_resid_p95_texel": measured["uv_abs_resid_p95_texel"],
        "kept_resid_max_texel": measured["uv_abs_resid_max_texel"],
    }


def muzzle_falsifiers(
    level: int,
    mesh: Mesh,
    index: UvIndex,
    fit_frames: LevelFrames,
    holdout: LevelFrames,
    *,
    reference: Mat3 = None,  # type: ignore[assignment]
    excluded_triangles: Collection[int] = (),
    fit_sample: int = 2000,
    seed: int = 0,
) -> list[dict]:
    """사전등록 §4 반증 배터리. **게이트가 낸 메시지를 그대로 싣는다.**

    넛지와 180° 회전은 정답 후보 P를 흔든 것이다 — "게이트가 정답만 통과시킨다"가 아니라
    "정답에서 1 mm(또는 0.1 mm)만 벗어나도 떨어진다"를 보이기 위해서다.
    """
    if reference is None:
        reference = GLOBAL_IMPORT_LINEAR
    fit_idx, _u = split_fit_holdout(len(fit_frames.hits), fit_sample, 1500, seed)
    src, dst, _m, _a = _correspondences(
        mesh, index, [fit_frames.hits[i] for i in fit_idx], [fit_frames.uvs[i] for i in fit_idx]
    )
    base_t, _n = solve_translation_fixed(reference, FIXED_LINEAR_SCALE, src, dst)
    hits, uvs = holdout.hits, holdout.uvs

    def judge(name: str, rotation: Mat3, translation: Vec3) -> dict:
        measured = measure_placement(
            mesh, rotation, FIXED_LINEAR_SCALE, translation, hits, uvs,
            excluded_triangles=excluded_triangles,
        )
        failures = fixed_linear_gate_failures(
            measured, [k for k in FIXED_LINEAR_GATE_KEYS if k in measured]
        )
        placed = _placed_mesh(mesh, rotation, FIXED_LINEAR_SCALE, translation)
        occ = muzzle_occlusion(placed, holdout.guns, hits)
        rejected = bool(failures) or not (occ["muzzle_occlusion"] <= MUZZLE_OCCLUSION_THRESHOLD)
        return {
            "level": level,
            "candidate": name,
            "exclusion_applied": bool(excluded_triangles),
            "rejected": rejected,
            "gate_message": AlignmentRejected(level, failures).args[0] if failures
            else f"level {level} rejected by 0 gate(s): ",
            "muzzle_occlusion": occ["muzzle_occlusion"],
            "muzzle_occlusion_verdict": (
                f"G_occ_muzzle={occ['muzzle_occlusion']!r} (need <= {MUZZLE_OCCLUSION_THRESHOLD})"
            ),
            "measured": measured,
        }

    out = [judge("P (reference, unmodified)", reference, base_t)]
    out.append(judge("plate mirror (_plate_mirror, det +1, t refit)",
                     _plate_mirror(reference),
                     solve_translation_fixed(_plate_mirror(reference), FIXED_LINEAR_SCALE, src, dst)[0]))
    for axis, name in enumerate("XYZ"):
        for step in (0.001, 0.0001):
            delta = tuple(step if d == axis else 0.0 for d in range(3))
            moved: Vec3 = tuple(base_t[d] + delta[d] for d in range(3))  # type: ignore[assignment]
            out.append(judge(f"P + {step * 1000:g} mm on world {name}", reference, moved))
    half_turns: list[Mat3] = [
        ((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, -1.0)),
        ((-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, -1.0)),
        ((-1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),
    ]
    for axis, half in zip("XYZ", half_turns):
        rotation = rotation_math.matmul(half, reference)
        translation, _k = solve_translation_fixed(rotation, FIXED_LINEAR_SCALE, src, dst)
        out.append(judge(f"world {axis} 180 deg (t refit)", rotation, translation))
    return out


def _muzzle_control_worker(level: int) -> tuple[int, dict, dict]:
    mesh = read_fbx(DEFAULT_MODELINGS_ROOT / LEVEL_FBX[level])
    index = build_uv_index(mesh)
    frames = collect_level_frames_by_session(level)
    control = recorded_muzzle_control(level, mesh, frames)
    alignment = load_alignment(level)
    _fit, eval_idx = split_fit_holdout(len(frames.hits), 2000, 1500, 0)
    row = exclusion_table_row(
        level, mesh, index, frames, _rotation_matrix(alignment), alignment.scale,
        alignment.translation, eval_idx,
    )
    print(f"muzzle control level {level}: {control}", flush=True)
    return level, control, row


def _muzzle_fixed_worker(level: int) -> tuple[int, dict, dict, dict, list, list[str], list[str]]:
    mesh = read_fbx(DEFAULT_MODELINGS_ROOT / LEVEL_FBX[level])
    index = build_uv_index(mesh)
    fit_frames = collect_level_frames_by_session(level)
    unread = unread_level_files(level, fit_frames.paths)
    holdout = collect_frames_from_files(unread)
    exclusions = mesh_only_uv_exclusions(mesh, index) if level == 2 else None
    excluded: Collection[int] = exclusions["excluded"] if exclusions else ()  # type: ignore[assignment]
    mirror = _plate_mirror(GLOBAL_IMPORT_LINEAR)
    result = solve_fixed_linear_muzzle(
        level, mesh, fit_frames, holdout, index=index, excluded_triangles=excluded,
        always_measure=(GLOBAL_IMPORT_LINEAR, mirror),
        unread_files=[str(p) for p in unread],
    )
    verdict = judge_muzzle(result, mirror, exclusions=exclusions)
    falsifiers = muzzle_falsifiers(
        level, mesh, index, fit_frames, holdout, excluded_triangles=excluded
    )
    if level == 2:
        falsifiers += muzzle_falsifiers(
            level, mesh, index, fit_frames, holdout, excluded_triangles=()
        )
    ref_rec = next(
        (c for c in result["candidates"] if c["rotation"] == _matrix_list(GLOBAL_IMPORT_LINEAR)), None
    )
    row = exclusion_table_row(
        level, mesh, index, holdout, GLOBAL_IMPORT_LINEAR, FIXED_LINEAR_SCALE,
        tuple(ref_rec["translation"]), list(range(len(holdout.hits))),  # type: ignore[index,arg-type]
    )
    print(f"muzzle fixed level {level}: {verdict['not_solved_reasons'] or 'SOLVED'}", flush=True)
    return level, result, verdict, row, falsifiers, list(holdout.paths), list(holdout.sessions)


def muzzle_main(out_path: Path) -> None:
    """2026-09-16 사전등록의 판정. 보고서를 정렬 파일보다 **먼저** 쓴다(mtime이 증거다)."""
    from multiprocessing import Pool

    with Pool(10) as pool:
        controls_async = pool.map_async(_muzzle_control_worker, [3, 4, 5, 6, 7, 8, 9, 10])
        fixed_async = pool.map_async(_muzzle_fixed_worker, [1, 2])
        controls = {lv: c for lv, c, _row in controls_async.get()}
        rows = {lv: row for lv, _c, row in controls_async.get()}
        fixed = fixed_async.get()

    report: dict = {
        "prereg": MUZZLE_PREREG_DOC,
        "muzzle_offset_m": L_MUZZLE_M,
        "occlusion_threshold": MUZZLE_OCCLUSION_THRESHOLD,
        "positive_controls_not_a_judgement_condition": True,
        "confirmation_not_out_of_sample": True,
        "controls": {str(k): v for k, v in sorted(controls.items())},
        "levels": {},
        "falsifiers": [],
    }
    entries = json.loads(ALIGNMENT_PATH.read_text(encoding="utf-8"))
    new_entries = list(entries)
    mirror = _plate_mirror(GLOBAL_IMPORT_LINEAR)
    for level, result, verdict, row, falsifiers, paths, sessions in fixed:
        rows[level] = row
        exclusions = mesh_only_uv_exclusions(
            read_fbx(DEFAULT_MODELINGS_ROOT / LEVEL_FBX[level])
        ) if level == 2 else None
        holdout = LevelFrames([], [], list(sessions), [], [], [], list(paths))
        i = next(k for k, e in enumerate(entries) if int(e["level"]) == level)
        new_entries[i] = muzzle_entry(
            entries[i], result, verdict, reference=GLOBAL_IMPORT_LINEAR, holdout=holdout,
            exclusions=exclusions, provenance=MUZZLE_PROVENANCE[level],
        )
        report["levels"][str(level)] = {
            "verdict": {k: v for k, v in verdict.items() if k not in ("winner", "mirror")},
            "holdout_files": paths,
            "holdout_n": result["holdout_n"],
            "candidates": result["candidates"],
        }
        report["falsifiers"].extend(falsifiers)
        print(f"level {level}: solved={verdict['solved']} {verdict['not_solved_reasons']}", flush=True)
    report["exclusion_table"] = [rows[lv] for lv in sorted(rows)]
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    ALIGNMENT_PATH.write_text(json.dumps(new_entries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    entries: list[dict] = []
    rejected: list[dict] = []
    for level in sorted(LEVEL_FBX):
        fbx = DEFAULT_MODELINGS_ROOT / LEVEL_FBX[level]
        try:
            entries.append(build_level_alignment(level, fbx))
            print(f"level {level}: accepted", flush=True)
        except AlignmentRejected as error:
            rejected.append(
                {
                    "level": level,
                    "provenance": "unverifiable",
                    "source_fbx": fbx.name,
                    "rejected_by": [
                        {"gate": name, "measured": value, "required": need}
                        for name, value, need in error.failures
                    ],
                }
            )
            print(f"level {level}: REJECTED — {error}", flush=True)

    # G5: 채택된 항목의 손잡이가 파일 전체에서 같아야 한다. 섞이면 빌드 실패다.
    signs = {1.0 if e["fit"]["det_sign"] > 0 else -1.0 for e in entries}
    if len(signs) > 1:
        raise SystemExit(f"G5 handedness is mixed across accepted entries: {sorted(signs)}")

    ALIGNMENT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = entries + rejected
    payload.sort(key=lambda e: e["level"])
    ALIGNMENT_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"accepted {len(entries)} / rejected {len(rejected)}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 3 and sys.argv[1] == "fixed-linear":
        fixed_linear_main(Path(sys.argv[2]))
    elif len(sys.argv) >= 3 and sys.argv[1] == "front-face":
        front_face_main(Path(sys.argv[2]))
    elif len(sys.argv) >= 3 and sys.argv[1] == "muzzle-occlusion":
        muzzle_main(Path(sys.argv[2]))
    else:
        main()
