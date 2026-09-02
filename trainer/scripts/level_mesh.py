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
(`PreRotation`/`Lcl Rotation`/`UnitScaleFactor`)으로 레벨 1을 배치하면 실제 착탄점과
3.68 m 어긋난다 — 씬 수준 회전(~90° yaw)이 파일 어디에도 기록되지 않았기 때문이다.
`solve_alignment`이 Umeyama 유사변환(회전+평행이동+균일 스케일)으로 그 변환을 맞춘다.
`Alignment.provenance`가 "solved"(맞춘 것)와 "declared"(선언대로 읽은 것)를 구분해
나른다 — 숫자만 적어두면 다음 사람이 파일이 준 값인 줄 안다.

레벨 1의 착탄점은 정확히 평면이라(X 좌표가 7만 5천 프레임에서 비트 단위로 동일)
Umeyama의 SVD가 정확한 특이값 0을 만나 퇴화한다. `_svd3`의 문턱을 최대 특이값
상대값으로 두고, 특이값 하나가 0인 경우 그 열을 외적으로 채워 이 퇴화를 견딘다 —
문턱을 절대값으로 두면 부동소수 잡음에 따라 그 판정이 흔들리고, 회전 "행"이
`(0, 0, 0)`인 무효 행렬이 나올 수 있다(합성 퇴화 데이터로 재현·검증했다).
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
from typing import BinaryIO, Final, Sequence

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


VALID_PROVENANCE: Final[frozenset[str]] = frozenset({"solved", "declared"})


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


def _alignment_from_entry(entry: dict) -> Alignment:
    solved = entry["solved_transform"]
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
        declared_scale=float(entry["declared_scale"]),
        fit=dict(entry["fit"]),
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


def _extents(points: Sequence[Vec3]) -> Vec3:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    zs = [p[2] for p in points]
    return (max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs))


def _thin_axis_index(vertices: Sequence[Vec3]) -> int:
    """모델 공간에서 범위가 가장 짧은 축. 얇은 양면 패널의 두 큰 면이 갈리는 축이다."""
    extent = _extents(vertices)
    return min(range(3), key=lambda i: extent[i])


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


def _icp_surface(
    vertices_raw: Sequence[Vec3],
    triangles_vidx: Sequence[tuple[int, int, int]],
    hit_points: Sequence[Vec3],
    r_seed: Mat3,
    c_seed: float,
    t_seed: Vec3,
    iters: int = 15,
    tol: float = 1e-7,
) -> tuple[Mat3, float, Vec3, float, list[int]]:
    """점-이산표본이 아니라 점-연속표면 ICP다.

    각 착탄점을 어느 삼각형이든 메시 표면의 최근접점에 대응시키고, 그 점의 무게중심
    좌표로 원본 모델 공간 대응점을 만든다. 메시가 부분적으로만 덮여도(레벨 10처럼)
    덮이지 않은 영역이 맞춤을 왜곡하지 않는다 — 고정 표본점 ICP와 달리 실제로 맞은
    영역 근처의 점만 대응으로 쓰인다.

    거리(`rms`)만으로 앞/뒤 면을 가릴 수 없다는 사실이 리뷰에서 드러났다(모듈
    상단·`solve_alignment` 독스트링) — 그래서 호출자가 면·UV로 후보를 가릴 수 있게,
    수렴한 **최종** 변환으로 다시 한 번 대응을 구해 그 삼각형 인덱스 목록도 낸다
    (`final_ti`). 마지막 루프 반복이 낸 `rms`/대응은 그 반복 *전* 변환의 잔차라
    반환하는 변환과 정확히 안 맞을 수 있어, 반환 변환으로 한 번 더 계산해 맞춘다.
    """
    rotation, scale, translation = r_seed, c_seed, t_seed
    last_rms: float | None = None
    for _ in range(iters):
        world_tris = _world_tris_for(vertices_raw, triangles_vidx, rotation, scale, translation)
        raw_tris = [
            (vertices_raw[i0], vertices_raw[i1], vertices_raw[i2]) for i0, i1, i2 in triangles_vidx
        ]
        corr_raw: list[Vec3] = []
        corr_hit: list[Vec3] = []
        total_d2 = 0.0
        for p in hit_points:
            d, cp, ti = _nearest_on_mesh(p, world_tris)
            total_d2 += d * d
            aw, bw, cw = world_tris[ti]
            u, v, w = _barycentric(cp, aw, bw, cw)  # type: ignore[arg-type]
            ar, br, cr = raw_tris[ti]
            p_raw = (
                u * ar[0] + v * br[0] + w * cr[0],
                u * ar[1] + v * br[1] + w * cr[1],
                u * ar[2] + v * br[2] + w * cr[2],
            )
            corr_raw.append(p_raw)
            corr_hit.append(p)
        rms = math.sqrt(total_d2 / len(hit_points))
        rotation, scale, translation = _umeyama(corr_raw, corr_hit)
        if last_rms is not None and abs(last_rms - rms) < tol:
            last_rms = rms
            break
        last_rms = rms

    final_world_tris = _world_tris_for(vertices_raw, triangles_vidx, rotation, scale, translation)
    final_ti = [_nearest_on_mesh(p, final_world_tris)[2] for p in hit_points]
    return rotation, scale, translation, last_rms, final_ti  # type: ignore[return-value]


def _prefilter_seed_rotations(
    vertices_raw: Sequence[Vec3], declared_rotation: Mat3, hit_extent: Vec3, top_k: int
) -> list[Mat3]:
    """정육면체 회전 24개 중, 선언된 회전과 합성했을 때 실측 착탄점 범위(bounding
    box)와 가장 가까운 범위를 내는 것들. ICP를 24번 다 돌리지 않고 후보를 좁힌다."""
    scored: list[tuple[float, Mat3]] = []
    for cube_r in _cube_rotations():
        seed_r = rotation_math.matmul(cube_r, declared_rotation)
        pts = [rotation_math.matvec(seed_r, v) for v in vertices_raw]  # type: ignore[arg-type]
        ext = _extents(pts)  # type: ignore[arg-type]
        err = sum(((ext[i] - hit_extent[i]) / max(hit_extent[i], 1e-6)) ** 2 for i in range(3))
        scored.append((err, seed_r))  # type: ignore[arg-type]
    scored.sort(key=lambda item: item[0])
    return [seed_r for _, seed_r in scored[:top_k]]


# =============================================================================
# 면 판정과 UV 일치도 (리뷰 B1/B2) — 거리로는 앞/뒤 면을 가릴 수 없다
# =============================================================================
#
# 레벨 1의 메시는 얇은 두 면이 마주 보는 양면 패널이고, 착탄점은 정확히 평면이다.
# 그래서 진짜 변환과 "거울 대칭"(얇은 축과 수직인 평면 기준 180°만큼 다른, 그러나
# 여전히 고유(det=+1) 회전인) 변환 둘 다 그 평면에서 수직거리 ~0을 낸다 — median·p95
# 는 항상 ~0이고 max_mm 하나만 남는데 그마저 표본별로 흔들린다(리뷰: 7.43~35.01mm).
# 거리는 이 둘을 가를 능력이 없다. 실제로 가르는 건 (1) 대응이 한쪽 면에
# 쏠렸는가(대칭 함정에 빠지지 않았는가) — `_dominant_face`, 그리고 (2) 그 면의 UV가
# 실측 텍스처 UV와 맞는가(거울 대칭은 지오메트리는 맞아도 UV는 안 맞는다) —
# `_uv_agreement`다.


# 삼각형 법선이 얇은 축과 이보다 더 나란하면(내적 절대값 기준) "큰 면"이고, 아니면
# 테두리 옆벽이다. 옆벽 삼각형(레벨 1: 84개 폴리곤 인덱스 중 일부, 넓이 27~98 —
# 큰 면의 넓이 2015에 비해 훨씬 작다)은 두 큰 면 중 어느 쪽에도 안 넣는다.
_BIG_FACE_NORMAL_COS: Final = 0.5


def _face_sign_by_triangle(
    vertices_raw: Sequence[Vec3], triangles: Sequence[tuple[int, int, int]], thin_axis: int
) -> list[float]:
    """삼각형마다: 큰 면 중 위쪽이면 +1, 아래쪽이면 -1, 옆벽(테두리)이면 0.

    **법선 방향으로 가른다 — 얇은 축 위 중심 좌표로 가르지 않는다.** 처음엔 중심
    좌표(centroid[thin_axis]의 부호)로 갈랐는데, 옆벽 삼각형이 프레임 둘레를 도느라
    중심 좌표의 부호가 제각각이라(위쪽 절반 둘레는 +, 아래쪽 절반은 -) 두 "면" 각각에
    옆벽 조각이 반반씩 섞여 들어갔다 — UV 회귀에 무관한 UV를 실은 이상치가 끼어
    R²가 0.987 수준으로 깎였다(리뷰가 지적한, 트리밍 전 표본의 문제와 같은 종류).
    법선이 얇은 축과 거의 나란한 것만 "큰 면"으로 인정하면 옆벽이 깨끗이 빠지고
    R²가 트리밍된 표본 수준(0.9999998/1.0000000)까지 올라간다 — 실측으로 확인했다.
    """
    out: list[float] = []
    for tri in triangles:
        a, b, c = (vertices_raw[i] for i in tri)
        n = _cross(_sub(b, a), _sub(c, a))
        length = math.sqrt(_dot(n, n))
        if length < 1e-12:
            out.append(0.0)
            continue
        cos_thin = n[thin_axis] / length
        if abs(cos_thin) < _BIG_FACE_NORMAL_COS:
            out.append(0.0)  # 옆벽: 얇은 축과 거의 수직인 법선
        else:
            out.append(1.0 if cos_thin > 0 else -1.0)
    return out


def _dominant_face(
    face_sign_by_triangle: Sequence[float], triangle_indices: Sequence[int]
) -> tuple[float, float]:
    """큰 면에 대응된 것들 중 어느 면에 더 쏠렸는지: (다수 면의 부호, 그 비율).

    옆벽(부호 0)에 대응된 점은 분모에서도 뺀다 — "옆벽에 맞음"은 "틀린 큰 면에
    맞음"과 다른 사실이라 섞으면 판정이 흐려진다. 비율이 낮으면(대칭 함정에서
    절반씩 갈리면 ~0.5) 아직 한쪽 큰 면으로 수렴하지 않은 것이다 —
    `solve_alignment`이 `min_dominant_fraction`으로 이런 후보를 거른다.
    """
    pos = sum(1 for ti in triangle_indices if face_sign_by_triangle[ti] > 0)
    neg = sum(1 for ti in triangle_indices if face_sign_by_triangle[ti] < 0)
    total = pos + neg
    if total == 0:
        return 0.0, 0.0
    if pos >= neg:
        return 1.0, pos / total
    return -1.0, neg / total


def _linreg(xs: Sequence[float], ys: Sequence[float]) -> tuple[float, float, float]:
    """단순 최소제곱 직선 `(slope, intercept, R²)`. `xs`에 분산이 없으면 기울기 0."""
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    sxx = sum((x - mean_x) ** 2 for x in xs)
    syy = sum((y - mean_y) ** 2 for y in ys)
    if sxx <= 1e-15:
        return 0.0, mean_y, 0.0
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x
    if syy <= 1e-15:
        return slope, intercept, 1.0
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    return slope, intercept, 1.0 - ss_res / syy


def _uv_pairs_for_dominant_face(
    mesh: Mesh,
    points: Sequence[Vec3],
    point_uvs: Sequence[tuple[float, float]],
    triangle_indices: Sequence[int],
    world_tris: Sequence[tuple[Vec3, Vec3, Vec3]],
    dominant_sign: float,
    face_sign_by_triangle: Sequence[float],
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """큰 면(`dominant_sign`)에 대응된 점만 골라 (메시 UV, 관측 UV) 쌍을 낸다.

    작은 옆면(테두리) 삼각형에 붙은 점은 뺀다 — 서로 무관한 UV를 실어 순수 이상치로
    작용한다(리뷰 B2: 5.16%가 옆면). `_uv_agreement`(내부, `solve_alignment`의 후보
    평가에서 씀)와 공개 `uv_pairs_on_dominant_face`(리뷰 D2, 다음 태스크의 메시
    앵커가 씀) 둘 다 이 함수 하나로 계산한다 — 따로 베끼면 정렬 레코드와 앵커가
    서로 다른 걸 재는 사태가 생긴다.
    """
    pairs: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for p, uv, ti in zip(points, point_uvs, triangle_indices):
        if face_sign_by_triangle[ti] != dominant_sign:
            continue
        if ti >= len(mesh.uv_indices):
            continue
        aw, bw, cw = world_tris[ti]
        cp = _closest_point_on_triangle(p, aw, bw, cw)
        bary = _barycentric(cp, aw, bw, cw)
        pairs.append((triangle_uv(mesh, ti, bary), uv))
    return pairs


def uv_regression(
    pairs: Sequence[tuple[tuple[float, float], tuple[float, float]]], axis: int
) -> tuple[float, float]:
    """`uv_pairs_on_dominant_face`가 낸 (메시 UV, 관측 UV) 쌍에서 한 축의 (기울기, R²).

    `axis`: 0이면 U, 1이면 V. 표본이 2개 미만이면 `(0.0, 0.0)` — 기울기를 지어내지
    않는다(`_linreg`와 같은 규칙).
    """
    if len(pairs) < 2:
        return 0.0, 0.0
    mesh_vals = [pair[0][axis] for pair in pairs]
    obs_vals = [pair[1][axis] for pair in pairs]
    slope, _intercept, r2 = _linreg(mesh_vals, obs_vals)
    return slope, r2


def uv_pairs_on_dominant_face(
    placed_mesh: Mesh, hits: Sequence[tuple[Vec3, tuple[float, float]]]
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """리뷰 D2: 다음 태스크의 메시 앵커가 쓰는 공개 진입점.

    `placed_mesh`는 `place()`를 거친(이미 세계 좌표인) 메시다. `hits`는
    (착탄점, 관측 UV) 쌍의 나열이다. 각 착탄점의 최근접 삼각형이 큰 면(다수결로
    정해지는 `dominant_sign`)에 속하는 것만 골라 (메시 UV, 관측 UV) 쌍을 낸다 —
    옆벽(테두리, 약 5%)에 붙은 착탄점은 빠진다.

    `solve_alignment`이 정렬을 고를 때 쓰는 `_uv_agreement`와 **같은 계산**
    (`_uv_pairs_for_dominant_face`)을 쓴다 — 따로 다시 짜면 정렬 레코드에 적힌
    숫자와 앵커가 재는 숫자가 서로 다른 걸 재는 사태가 생긴다.
    """
    if not placed_mesh.triangles or not hits:
        return []
    thin_axis = _thin_axis_index(placed_mesh.vertices)
    face_sign_by_triangle = _face_sign_by_triangle(
        placed_mesh.vertices, placed_mesh.triangles, thin_axis
    )
    world_tris = _world_triangles(placed_mesh)
    points = [h[0] for h in hits]
    point_uvs = [h[1] for h in hits]
    triangle_indices = [_nearest_on_mesh(p, world_tris)[2] for p in points]
    dominant_sign, _dominant_fraction = _dominant_face(face_sign_by_triangle, triangle_indices)
    return _uv_pairs_for_dominant_face(
        placed_mesh, points, point_uvs, triangle_indices, world_tris, dominant_sign, face_sign_by_triangle
    )


def _uv_agreement(
    mesh: Mesh,
    points: Sequence[Vec3],
    point_uvs: Sequence[tuple[float, float]],
    triangle_indices: Sequence[int],
    world_tris: Sequence[tuple[Vec3, Vec3, Vec3]],
    dominant_sign: float,
    face_sign_by_triangle: Sequence[float],
) -> dict[str, float]:
    """`solve_alignment`의 후보 평가에서 쓴다 — 대응(`triangle_indices`/`world_tris`)이
    이미 ICP 루프에서 계산돼 있으므로 다시 최근접 탐색을 하지 않는다
    (`uv_pairs_on_dominant_face`는 이 대응이 없어 매번 새로 찾는다). 반환 딕셔너리는
    그대로 `Alignment.fit`에 들어간다.
    """
    pairs = _uv_pairs_for_dominant_face(
        mesh, points, point_uvs, triangle_indices, world_tris, dominant_sign, face_sign_by_triangle
    )
    slope_u, r2_u = uv_regression(pairs, 0)
    slope_v, r2_v = uv_regression(pairs, 1)
    return {
        "uv_n": float(len(pairs)),
        "uv_slope_u": slope_u,
        "uv_r2_u": r2_u,
        "uv_slope_v": slope_v,
        "uv_r2_v": r2_v,
    }


def _fit_stats(distances_m: Sequence[float]) -> dict[str, float]:
    s = sorted(distances_m)
    n = len(s)

    def pct(p: float) -> float:
        return s[min(n - 1, int(round(p / 100.0 * (n - 1))))]

    return {
        "n_hits": n,
        "median_mm": pct(50) * 1000.0,
        "p95_mm": pct(95) * 1000.0,
        "max_mm": pct(100) * 1000.0,
    }


DEFAULT_MIN_DOMINANT_FRACTION: Final = 0.9

# 리뷰 C2/D1: 얇은 축 둘레 180도 회전(Rx180, `EXPECTED_UV_SLOPE`가 가르는 대칭)으로
# 관계된 두 orientation이 거리로도, UV R²로도 갈리지 않는다 — 둘 다 median/p95/max
# ~0mm, UV R² 0.9999+를 낸다(직접 재현·확인). 대수적으로 확인했다: "예측 UV가 월드
# 좌표와 CSV 관측 UV와 같은 방향으로 움직이는가"는 "회귀 기울기(관측 vs 예측)가
# 양수인가"와 **완전히 같은 정보**다(obs=k1*w+c1, pred=k2*w+c2이면
# slope(obs|pred)=k1/k2이므로 부호가 같다는 것과 기울기가 양수라는 것은 동치) —
# 그래서 순수 선형 상관으로는 Rx180의 두 orientation을 원리적으로 못 가른다.
#
# **이 상수의 값은 측정이 아니라 고른 가지(branch)다 — 그 이유를 정확히 적는다.**
# 처음에 이 주석은 "실측 검증에서 좋은 맞춤의 기울기가 두 축 다 음수(-2047.85
# 근방)로 확인됐다"고 -2048을 근거로 삼았는데, 그 인용이 오독이었다(리뷰가 잡음):
# 실제 출처(`task-4-review.md` 24–26행)는 U −2047.85, **V +2047.85**로 **부호가
# 섞여** 있다. 이 저장소의 첫 커밋(`ed71ce2`)의 실제 회전을 지금 코드로 다시
# 재보면 그 섞인 값이 정확히 재현된다(U −2047.4, V +2047.8) — 그건 지금은 대체된
# 그 커밋의 진짜 기하다, 측정 오류가 아니다. 그리고 그 섞인 부호는 Rx180 하나만으로는
# 안 나온다 — Rx180은 두 기울기를 **함께** 뒤집는다. 직접 재현해 확인했다: 후보군
# 안에는 Rx180 대칭(둘 다 같은 부호로 뒤집힘, `(+,+)`↔`(-,-)`) 말고 **또 다른**
# 180도 대칭(하나만 뒤집힘, `(-,+)`↔`(+,-)`, `ed71ce2`가 여기 속한다)이 있다 —
# 판 법선이 아닌 판 안의 축 둘레 180도 회전에서 온다. 그래서 실제로는 최소
# 네 갈래(부호 조합 `++`/`--`/`-+`/`+-`)가 거리·R²로는 똑같이 좋다.
#
# 이 상수가 실제로 하는 일은: 네 갈래 중 **어느 것을 기록할지 정하는 것**이다.
# 근거는 두 가지뿐이다 — (1) 15/15 시드(본 구현 12개 + 재검토자가 늘린 3개)가
# 전부 이 갈래(두 축 다 음수)로 수렴했다, (2) 부호가 섞인 갈래는 (1,-1) 부호
# 조합이 Rx180 하나로는 설명이 안 되므로 물리적으로 있음직하지 않다(옆에서 봤을
# 때 U만 뒤집히고 V는 안 뒤집히는 단일 강체 회전은 판 법선 둘레가 아닌 축 둘레
# 회전이라 착탄 평면 자체를 깨야 한다). **이 값은 유도된 게 아니다** — 데이터만
# 으로는 두 갈래(`++` 대 `--`)를 가를 방법이 없다는 것을 이미 대수적으로 보였다.
# 외부 근거(예: Unity 씬/프리팹 파일)가 나오면 반대 갈래가 실제였다고 밝혀질 수
# 있고, 그러면 이 상수는 부호를 바꿔야 한다 — 지금은 그 경우에도 "조용히 틀린
# 채로 있는" 대신 "흔들리면 보이는" 상태를 만드는 것이 이 상수의 역할이다.
UV_ATLAS_TEXELS: Final = 2048.0
EXPECTED_UV_SLOPE: Final = -UV_ATLAS_TEXELS
MIN_UV_R2_FOR_SIGN_CHECK: Final = 0.9


def solve_alignment(
    mesh: Mesh,
    hits: Sequence[Vec3],
    *,
    hit_uvs: Sequence[tuple[float, float]],
    level: int = 1,
    fit_sample: int = DEFAULT_FIT_SAMPLE,
    eval_cap: int = DEFAULT_EVAL_CAP,
    top_k: int = 8,
    icp_iters: int = 15,
    seed: int = 0,
    min_dominant_fraction: float = DEFAULT_MIN_DOMINANT_FRACTION,
) -> Alignment:
    """실측 착탄점 `hits`(+ 짝을 이루는 `hit_uvs`)에 `mesh`를 맞추는 Umeyama
    유사변환을 푼다. 반환된 `Alignment.provenance`는 항상 `"solved"`다.

    선언된 변환은 씬 수준 회전이 빠져 있어 그대로 쓰면 미터 단위로 어긋난다
    (모듈 독스트링). 그래서 선언된 회전에 정육면체 회전 24개를 합성해 후보를 만들고,
    실측 범위와 가장 가까운 `top_k`개를 점-표면 ICP로 정련한다.

    **앞/뒤 면 대칭 함정과 그 해소 (리뷰 B1).** 레벨 1의 메시는 얇은 양면 패널이다
    — 두 큰 면이 얇은 축을 사이에 두고 마주 본다. 초기 시드의 평행이동을 전체
    정점의 평균으로 두면 정확히 두 면의 중간이 되어 대칭이 되고, ICP가 스스로 못
    깬다(각 착탄점이 두 면 아무 쪽에나 반반 대응되는 안정된 고정점). 시드마다 얇은
    축으로 평행이동을 미리 밀어(두 부호 다) 대칭을 깬다.

    **그런데 거리(RMS/median/p95/max)는 어느 면이 옳은지 가릴 능력이 없다** —
    착탄점이 정확히 평면이라, 진짜 변환과 "거울 대칭"(얇은 축과 수직인 평면 기준
    180°만큼 다른, 그러나 여전히 고유(det=+1) 회전인) 변환 둘 다 그 평면에서
    수직거리 ~0을 낸다. 실측으로 확인했다: RMS만으로 후보를 고르면 10번 중 5번이
    잘못된 면으로 수렴한다(회전·평행이동·기록되는 max_mm이 실행마다 흔들리는데,
    유일하게 안정된 건 스케일뿐이었다).

    **UV R²도 한 겹 더 있는 대칭을 못 가른다(리뷰 C2/D1).** 한쪽 면으로 수렴한
    뒤에도, 그 면 안에 최소 네 갈래(부호 조합 `++`/`--`/`-+`/`+-`)가 남는다 — 크기·
    R²는 다 똑같다(직접 재현: 넷 다 UV R² 0.9999+, 기울기 크기 ~2048). 둘은 얇은
    축(판 법선) 둘레 180° 회전(Rx180)으로 관계되어 두 기울기가 함께 뒤집히고
    (`++`↔`--`), 다른 둘은 판 안의 축 둘레 180° 회전에서 와서 하나만 뒤집힌다
    (`-+`↔`+-`) — 첫 커밋(`ed71ce2`)이 실제로 이 후자 쪽이었다(`EXPECTED_UV_SLOPE`
    독스트링 참고). `abs(slope)`나 R²는 부호를 안 보므로 넷 다 못 가른다. 그래서
    세 단계로 고른다: (1) 대응이 한쪽 면에 `min_dominant_fraction` 이상 쏠린
    후보만 남긴다(면 대칭 함정을 못 벗어난 후보를 거른다), (2) R²가 충분히 높은
    (`MIN_UV_R2_FOR_SIGN_CHECK` 이상) 후보 중에서 두 기울기 다 `EXPECTED_UV_SLOPE`
    에 가장 가까운 것을 고른다 — 이게 네 갈래 중 하나를 결정적으로 고른다(다른
    셋은 두 축 중 최소 하나가 목표에서 ~4096 떨어져 있어 절대 안 뽑힌다).
    `EXPECTED_UV_SLOPE`는 **측정값이 아니라 고정한 가지**다 — 그 상수의 독스트링에
    이유가 있다. (2)를 만족하는 후보가 없으면 R²가 가장 높은 것으로 물러선다.

    선택된 면·UV 회귀 결과는 `Alignment.fit`에 `face_sign`/`dominant_face_fraction`/
    `uv_slope_u`/`uv_r2_u`/`uv_slope_v`/`uv_r2_v`/`uv_n`으로 남는다(리뷰 B2) — 다음
    태스크의 메시 앵커 검사가 거리 대신 이 (부호 있는) UV 값으로 다시 설계된다.
    """
    if not hits:
        raise ValueError("solve_alignment: no hit points given")
    if len(hits) != len(hit_uvs):
        raise ValueError("solve_alignment: hits and hit_uvs must be the same length, index-paired")
    if not mesh.triangles:
        raise ValueError("solve_alignment: mesh has no triangles")
    if not mesh.uv_indices:
        raise ValueError(
            "solve_alignment: mesh has no uv_indices — cannot disambiguate the correct "
            "face from a distance-only fit (see B1/B2 in the module docstring)"
        )

    declared_rotation = _declared_rotation(mesh.declared)
    declared_scale = _declared_scale(mesh.declared)

    hit_list = list(hits)
    uv_list = list(hit_uvs)
    n_hits = len(hit_list)
    hit_extent = _extents(hit_list)
    thin_axis = _thin_axis_index(mesh.vertices)
    thin_extent = _extents(mesh.vertices)[thin_axis]
    # 대칭을 깨는 데 필요한 만큼만 민다 — 두께의 몇 % 정도. 너무 작으면 부동소수
    # 잡음에 묻히고, 너무 크면 시드가 원래 놓일 자리에서 멀어진다.
    nudge = max(thin_extent * 0.05, 1e-6)
    face_sign_by_triangle = _face_sign_by_triangle(mesh.vertices, mesh.triangles, thin_axis)

    rng = random.Random(seed)
    fit_idx = rng.sample(range(n_hits), min(fit_sample, n_hits))
    eval_idx = rng.sample(range(n_hits), min(eval_cap, n_hits))
    fit_target = [hit_list[i] for i in fit_idx]
    fit_target_uv = [uv_list[i] for i in fit_idx]
    eval_sample = [hit_list[i] for i in eval_idx]
    eval_sample_uv = [uv_list[i] for i in eval_idx]
    mu_t = _mean(fit_target)

    seed_rotations = _prefilter_seed_rotations(mesh.vertices, declared_rotation, hit_extent, top_k)

    candidates: list[dict[str, object]] = []
    for seed_r in seed_rotations:
        c_seed = declared_scale
        seeded_pts = [rotation_math.matvec(seed_r, v) for v in mesh.vertices]  # type: ignore[arg-type]
        mu_s = _mean(seeded_pts)  # type: ignore[arg-type]
        t_seed_centered = tuple(mu_t[k] - c_seed * mu_s[k] for k in range(3))
        nudge_raw = [0.0, 0.0, 0.0]
        nudge_raw[thin_axis] = nudge
        nudge_world = rotation_math.matvec(seed_r, tuple(nudge_raw))  # type: ignore[arg-type]
        for sign in (1.0, -1.0):
            t_seed = tuple(
                t_seed_centered[k] + sign * c_seed * nudge_world[k] for k in range(3)
            )
            r_fit, c_fit, t_fit, _rms, ti_list = _icp_surface(
                mesh.vertices, mesh.triangles, fit_target, seed_r, c_seed, t_seed, iters=icp_iters
            )
            dom_sign, dom_frac = _dominant_face(face_sign_by_triangle, ti_list)
            world_tris = _world_tris_for(mesh.vertices, mesh.triangles, r_fit, c_fit, t_fit)  # type: ignore[arg-type]
            uv_stats = _uv_agreement(
                mesh, fit_target, fit_target_uv, ti_list, world_tris, dom_sign, face_sign_by_triangle
            )
            candidates.append(
                {
                    "r": r_fit,
                    "c": c_fit,
                    "t": t_fit,
                    "dom_sign": dom_sign,
                    "dom_frac": dom_frac,
                    "uv_r2": min(uv_stats["uv_r2_u"], uv_stats["uv_r2_v"]),
                    "uv_slope_u": uv_stats["uv_slope_u"],
                    "uv_slope_v": uv_stats["uv_slope_v"],
                    # 목표 부호(EXPECTED_UV_SLOPE)까지 거리 — 얇은 축 180도 대칭의
                    # 두 orientation은 R²·크기가 같으므로 이 항이 실제로 갈라야 한다.
                    "sign_error": (uv_stats["uv_slope_u"] - EXPECTED_UV_SLOPE) ** 2
                    + (uv_stats["uv_slope_v"] - EXPECTED_UV_SLOPE) ** 2,
                }
            )

    converged = [c for c in candidates if c["dom_frac"] >= min_dominant_fraction]  # type: ignore[operator]
    pool = converged if converged else candidates

    # 1단계: R²가 충분한 후보만(면은 맞았는지 이미 걸렀으니 여기서는 선형 적합이
    # 정말 좋은지만 본다). 2단계: 그 중 목표 부호에 가장 가까운 것 — 이게 얇은 축
    # 180도 대칭(리뷰 C2)을 가른다. 부호 확인을 통과하는 후보가 없으면(R²가 다
    # 낮으면) R²로 물러선다 — 그 경우 부호 대칭 이전에 더 근본적인 문제가 있다는
    # 뜻이라 부호로 고르는 것 자체가 의미가 없다.
    sign_checkable = [c for c in pool if c["uv_r2"] >= MIN_UV_R2_FOR_SIGN_CHECK]  # type: ignore[operator]
    if sign_checkable:
        best = min(sign_checkable, key=lambda c: c["sign_error"])  # type: ignore[arg-type]
    else:
        best = max(pool, key=lambda c: c["uv_r2"])  # type: ignore[arg-type]

    r_fit: Mat3 = best["r"]  # type: ignore[assignment]
    c_fit: float = best["c"]  # type: ignore[assignment]
    t_fit: Vec3 = best["t"]  # type: ignore[assignment]
    dom_sign: float = best["dom_sign"]  # type: ignore[assignment]

    world_tris = _world_tris_for(mesh.vertices, mesh.triangles, r_fit, c_fit, t_fit)
    distances: list[float] = []
    final_ti: list[int] = []
    for p in eval_sample:
        d, _cp, ti = _nearest_on_mesh(p, world_tris)
        distances.append(d)
        final_ti.append(ti)
    _final_sign, final_dom_frac = _dominant_face(face_sign_by_triangle, final_ti)
    uv_final = _uv_agreement(
        mesh, eval_sample, eval_sample_uv, final_ti, world_tris, dom_sign, face_sign_by_triangle
    )

    fit = _fit_stats(distances)
    fit["face_sign"] = dom_sign
    fit["dominant_face_fraction"] = final_dom_frac
    fit.update(uv_final)

    return Alignment(
        level=level,
        provenance="solved",
        rotation=tuple(r_fit[r][c] for r in range(3) for c in range(3)),
        translation=t_fit,
        scale=c_fit,
        declared_scale=declared_scale,
        fit=fit,
    )


# =============================================================================
# level_alignment.json 재생성 (`python3 level_mesh.py`)
# =============================================================================


def _collect_level_hits(
    level: int, professionals_root: Path = DEFAULT_PROFESSIONALS_ROOT, min_hits: int = DEFAULT_EVAL_CAP
) -> tuple[list[Vec3], list[tuple[float, float]]]:
    """실전모드(`1-A` 등)를 뺀 `<level>_*.csv` 세션들에서 분사 중 착탄점과 그 UV를 모은다.

    CSV 리더를 새로 만들지 않는다 — `simulator_csv.read_rows` /
    `measured_pose.measured_frames`를 그대로 쓴다. UV도 함께 모은다(리뷰 B1/B2가
    면 판정·`fit`의 UV 근거에 쓴다) — 착탄점만으로는 앞/뒤 면을 가릴 수 없다.
    """
    import measured_pose
    import simulator_csv

    hits: list[Vec3] = []
    uvs: list[tuple[float, float]] = []
    for path in sorted(professionals_root.glob(f"**/{level}_*.csv")):
        rows = simulator_csv.read_rows(path)
        for frame in measured_pose.measured_frames(rows):
            if frame.hit is None or frame.uv is None or not frame.spraying:
                continue
            hits.append(frame.hit)
            uvs.append(frame.uv)
        if len(hits) >= min_hits:
            break
    return hits, uvs


def build_level_alignment(
    level: int,
    fbx_path: Path,
    professionals_root: Path = DEFAULT_PROFESSIONALS_ROOT,
    min_hits: int = DEFAULT_EVAL_CAP,
    seed: int = 0,
) -> dict:
    """`level_alignment.json`에 들어갈 항목 하나를 만든다. `solve_alignment`을 그대로 쓴다."""
    mesh = read_fbx(fbx_path)
    hits, hit_uvs = _collect_level_hits(level, professionals_root, min_hits=min_hits)
    if len(hits) < min_hits:
        raise ValueError(
            f"only collected {len(hits)} hits for level {level}, need >= {min_hits} "
            "(the §D5 pre-registered floor)"
        )
    alignment = solve_alignment(
        mesh, hits, hit_uvs=hit_uvs, level=level, eval_cap=min_hits, seed=seed
    )
    return _entry_from_alignment(alignment, source_fbx=fbx_path.name)


def main() -> None:
    entry = build_level_alignment(1, DEFAULT_MODELINGS_ROOT / "learning_mode_lv.1_rev.1.1.fbx")
    ALIGNMENT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ALIGNMENT_PATH.write_text(json.dumps([entry], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(entry, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
