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

import json
import math
import random
import struct
import zlib
from dataclasses import dataclass
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


def _triangulate(pvi: Sequence[int]) -> list[tuple[int, int, int]]:
    """`PolygonVertexIndex`를 삼각형 부채꼴로 나눈다.

    각 다각형의 마지막 인덱스는 비트 반전(`~i`)으로 표시된다 — `-i - 1`로 되돌린다.
    """
    triangles: list[tuple[int, int, int]] = []
    current: list[int] = []
    for raw in pvi:
        if raw < 0:
            current.append(-raw - 1)
            n = len(current)
            for i in range(1, n - 1):
                triangles.append((current[0], current[i], current[i + 1]))
            current = []
        else:
            current.append(raw)
    return triangles


# =============================================================================
# Mesh
# =============================================================================


@dataclass(frozen=True, slots=True)
class Mesh:
    """모델 공간 메시. `place`로 정렬을 적용하기 전까지는 FBX 파일 좌표 그대로다."""

    vertices: list[Vec3]
    triangles: list[tuple[int, int, int]]
    uvs: list[tuple[float, float]]
    declared: dict[str, object]


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

    def scalar(d: dict[str, list], key: str, default: object) -> object:
        return d[key][0] if key in d else default

    def vec(d: dict[str, list], key: str, default: Vec3 = (0.0, 0.0, 0.0)) -> Vec3:
        return (d[key][0], d[key][1], d[key][2]) if key in d else default

    vertices_node = geometry.child("Vertices")
    if vertices_node is None:
        raise ValueError(f"{path}: Geometry has no Vertices")
    flat_v = vertices_node.props[0]
    vertices = [(flat_v[i], flat_v[i + 1], flat_v[i + 2]) for i in range(0, len(flat_v), 3)]

    pvi_node = geometry.child("PolygonVertexIndex")
    triangles = _triangulate(pvi_node.props[0]) if pvi_node is not None else []

    uvs: list[tuple[float, float]] = []
    layer_uv = geometry.child("LayerElementUV")
    if layer_uv is not None:
        uv_node = layer_uv.child("UV")
        if uv_node is not None:
            flat_uv = uv_node.props[0]
            uvs = [(flat_uv[i], flat_uv[i + 1]) for i in range(0, len(flat_uv), 2)]

    model_name = ""
    if len(model.props) > 1 and isinstance(model.props[1], str):
        model_name = model.props[1].split("\x00")[0]

    declared: dict[str, object] = {
        "UpAxis": int(scalar(gs, "UpAxis", 1)),
        "UpAxisSign": int(scalar(gs, "UpAxisSign", 1)),
        "FrontAxis": int(scalar(gs, "FrontAxis", 2)),
        "CoordAxis": int(scalar(gs, "CoordAxis", 0)),
        "UnitScaleFactor": float(scalar(gs, "UnitScaleFactor", 1.0)),
        "PreRotation": vec(m, "PreRotation"),
        "PostRotation": vec(m, "PostRotation"),
        "LclRotation": vec(m, "Lcl Rotation"),
        "LclTranslation": vec(m, "Lcl Translation"),
        "LclScaling": vec(m, "Lcl Scaling", (1.0, 1.0, 1.0)),
        "GeometricTranslation": vec(m, "GeometricTranslation"),
        "GeometricRotation": vec(m, "GeometricRotation"),
        "GeometricScaling": vec(m, "GeometricScaling", (1.0, 1.0, 1.0)),
        "RotationOrder": int(scalar(m, "RotationOrder", 0)),
        "ModelName": model_name,
    }

    return Mesh(vertices=vertices, triangles=triangles, uvs=uvs, declared=declared)


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


@dataclass(frozen=True, slots=True)
class Alignment:
    """`world = scale * (rotation @ raw) + translation`.

    `provenance`가 "solved"면 `solve_alignment`이 실측 착탄점에 맞춰 푼 값이고,
    "declared"면 FBX가 선언한 값을 그대로 읽은 것이다. 두 주장은 다르다 — 레벨 1은
    선언된 변환이 3.68 m 어긋나므로 "solved"만 실제로 쓸 수 있다.
    """

    level: int
    provenance: str
    rotation: tuple[float, ...]  # 9개, 행 우선(row-major) 3x3
    translation: Vec3
    scale: float
    declared_scale: float
    fit: dict[str, float]


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
    import itertools

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


def _icp_surface(
    vertices_raw: Sequence[Vec3],
    triangles_vidx: Sequence[tuple[int, int, int]],
    hit_points: Sequence[Vec3],
    r_seed: Mat3,
    c_seed: float,
    t_seed: Vec3,
    iters: int = 15,
    tol: float = 1e-7,
) -> tuple[Mat3, float, Vec3, float]:
    """점-이산표본이 아니라 점-연속표면 ICP다.

    각 착탄점을 어느 삼각형이든 메시 표면의 최근접점에 대응시키고, 그 점의 무게중심
    좌표로 원본 모델 공간 대응점을 만든다. 메시가 부분적으로만 덮여도(레벨 10처럼)
    덮이지 않은 영역이 맞춤을 왜곡하지 않는다 — 고정 표본점 ICP와 달리 실제로 맞은
    영역 근처의 점만 대응으로 쓰인다.
    """
    rotation, scale, translation = r_seed, c_seed, t_seed
    last_rms: float | None = None
    for _ in range(iters):
        world_tris: list[tuple[Vec3, Vec3, Vec3]] = []
        raw_tris: list[tuple[Vec3, Vec3, Vec3]] = []
        for i0, i1, i2 in triangles_vidx:
            a_raw, b_raw, c_raw = vertices_raw[i0], vertices_raw[i1], vertices_raw[i2]
            raw_tris.append((a_raw, b_raw, c_raw))
            world_tris.append(
                (
                    _apply_similarity(rotation, scale, translation, a_raw),
                    _apply_similarity(rotation, scale, translation, b_raw),
                    _apply_similarity(rotation, scale, translation, c_raw),
                )
            )
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
    return rotation, scale, translation, last_rms  # type: ignore[return-value]


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


def solve_alignment(
    mesh: Mesh,
    hits: Sequence[Vec3],
    *,
    level: int = 1,
    fit_sample: int = DEFAULT_FIT_SAMPLE,
    eval_cap: int = DEFAULT_EVAL_CAP,
    top_k: int = 3,
    icp_iters: int = 15,
    seed: int = 0,
) -> Alignment:
    """실측 착탄점 `hits`에 `mesh`를 맞추는 Umeyama 유사변환을 푼다.

    선언된 변환은 씬 수준 회전이 빠져 있어 그대로 쓰면 미터 단위로 어긋난다
    (모듈 독스트링). 그래서 선언된 회전에 정육면체 회전 24개를 합성해 시드 후보를
    만들고, 실측 범위와 가장 가까운 `top_k`개만 점-표면 ICP로 정련해 최적을 고른다.
    반환된 `Alignment.provenance`는 항상 `"solved"`다.

    **앞/뒤 면 대칭 함정.** 레벨 1의 메시는 얇은 양면 패널이다(가장 얇은 축의 범위가
    다른 두 축보다 훨씬 작다) — 두 큰 면이 그 얇은 축을 사이에 두고 마주 본다. 초기
    시드의 평행이동을 전체 정점의 평균으로 두면, 그 평균이 정확히 두 면의 중간에
    놓여 두 면에서 정확히 같은 거리가 된다. 그 대칭은 ICP가 스스로 못 깬다 — 각
    착탄점이 두 면 중 아무 쪽에나(거의 반반) 대응되고, Umeyama가 그 반반 대응의
    평균을 다시 내놓으면서 같은 중간 지점에 눌러앉는 안정된(그러나 틀린) 고정점이
    된다. 실측으로 검증했다: 이 대칭을 안 깨면 최근접 삼각형 거리가 약 6 mm에서
    수렴하지 않고 멈추는데, 이는 패널 두께의 절반과 같은 크기다(두 면의 평균이므로).
    그래서 시드마다 얇은 축을 따라 평행이동을 아주 조금 밀어(두 부호 다 시도) 대칭을
    미리 깬다 — 한쪽으로 밀면 그 면에 대응이 쏠리기 시작하고, ICP의 나머지 반복이
    그 쏠림을 스스로 강화해 하나의 면으로 수렴한다. 두 부호 다 시도해 최종 RMS가 더
    낮은 쪽을 쓴다(어느 면이 진짜인지는 착탄점만으로 미리 알 수 없다).
    """
    if not hits:
        raise ValueError("solve_alignment: no hit points given")
    if not mesh.triangles:
        raise ValueError("solve_alignment: mesh has no triangles")

    declared_rotation = _declared_rotation(mesh.declared)
    declared_scale = _declared_scale(mesh.declared)

    hit_list = list(hits)
    hit_extent = _extents(hit_list)
    thin_axis = _thin_axis_index(mesh.vertices)
    thin_extent = _extents(mesh.vertices)[thin_axis]
    # 대칭을 깨는 데 필요한 만큼만 민다 — 두께의 몇 % 정도. 너무 작으면 부동소수
    # 잡음에 묻히고, 너무 크면 시드가 원래 놓일 자리에서 멀어진다.
    nudge = max(thin_extent * 0.05, 1e-6)

    rng = random.Random(seed)
    fit_target = rng.sample(hit_list, min(fit_sample, len(hit_list)))
    eval_sample = rng.sample(hit_list, min(eval_cap, len(hit_list)))
    mu_t = _mean(fit_target)

    seed_rotations = _prefilter_seed_rotations(mesh.vertices, declared_rotation, hit_extent, top_k)

    best: tuple[float, Mat3, float, Vec3] | None = None
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
            r_fit, c_fit, t_fit, rms = _icp_surface(
                mesh.vertices, mesh.triangles, fit_target, seed_r, c_seed, t_seed, iters=icp_iters
            )
            if best is None or rms < best[0]:
                best = (rms, r_fit, c_fit, t_fit)  # type: ignore[assignment]

    assert best is not None
    _rms, r_fit, c_fit, t_fit = best

    world_tris = [
        (
            _apply_similarity(r_fit, c_fit, t_fit, mesh.vertices[a]),
            _apply_similarity(r_fit, c_fit, t_fit, mesh.vertices[b]),
            _apply_similarity(r_fit, c_fit, t_fit, mesh.vertices[c]),
        )
        for a, b, c in mesh.triangles
    ]
    distances = [_nearest_on_mesh(p, world_tris)[0] for p in eval_sample]

    return Alignment(
        level=level,
        provenance="solved",
        rotation=tuple(r_fit[r][c] for r in range(3) for c in range(3)),
        translation=t_fit,
        scale=c_fit,
        declared_scale=declared_scale,
        fit=_fit_stats(distances),
    )


# =============================================================================
# level_alignment.json 재생성 (`python3 level_mesh.py`)
# =============================================================================


def _collect_level_hits(
    level: int, professionals_root: Path = DEFAULT_PROFESSIONALS_ROOT, min_hits: int = DEFAULT_EVAL_CAP
) -> list[Vec3]:
    """실전모드(`1-A` 등)를 뺀 `<level>_*.csv` 세션들에서 분사 중 착탄점을 모은다.

    CSV 리더를 새로 만들지 않는다 — `simulator_csv.read_rows` /
    `measured_pose.measured_frames`를 그대로 쓴다.
    """
    import measured_pose
    import simulator_csv

    hits: list[Vec3] = []
    for path in sorted(professionals_root.glob(f"**/{level}_*.csv")):
        rows = simulator_csv.read_rows(path)
        for frame in measured_pose.measured_frames(rows):
            if frame.hit is None or not frame.spraying:
                continue
            hits.append(frame.hit)
        if len(hits) >= min_hits:
            break
    return hits


def build_level_alignment(
    level: int,
    fbx_path: Path,
    professionals_root: Path = DEFAULT_PROFESSIONALS_ROOT,
    min_hits: int = DEFAULT_EVAL_CAP,
) -> dict:
    """`level_alignment.json`에 들어갈 항목 하나를 만든다. `solve_alignment`을 그대로 쓴다."""
    mesh = read_fbx(fbx_path)
    hits = _collect_level_hits(level, professionals_root, min_hits=min_hits)
    if len(hits) < min_hits:
        raise ValueError(
            f"only collected {len(hits)} hits for level {level}, need >= {min_hits} "
            "(the §D5 pre-registered floor)"
        )
    alignment = solve_alignment(mesh, hits, level=level, eval_cap=min_hits)
    return _entry_from_alignment(alignment, source_fbx=fbx_path.name)


def main() -> None:
    entry = build_level_alignment(1, DEFAULT_MODELINGS_ROOT / "learning_mode_lv.1_rev.1.1.fbx")
    ALIGNMENT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ALIGNMENT_PATH.write_text(json.dumps([entry], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(entry, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
