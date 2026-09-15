#!/usr/bin/env python3
"""배치된 레벨 메시 위의 기하 질의와, 대표 프레임 한 장의 합성 정렬 검사. 표시 전용.

벽(메시)·착탄점·건·스켈레톤이 한 화면에 함께 그려진다. 부품별로는 각자 검증됐어도 합성이
맞는지는 따로 재야 한다 — 착탄점이 메시 표면에 붙는지, 골반이 건과 같은 쪽(벽 앞)에
있는지, 건 분사축 반직선이 메시에서 착탄점 근처에 닿는지.

최근접 삼각형은 **3D 거리**로 찾는다. UV로 찾으면 정렬을 UV로 풀었으므로 순환이 된다.
삼각형 법선의 부호는 반사 정렬(det<0)에서 감김이 뒤집혀 믿을 수 없으므로 부호가 필요한 곳은
기준점(건) 쪽으로 다시 맞춘다.
"""

from __future__ import annotations

import importlib
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

_SCRIPTS_DIR = str(Path(__file__).resolve().parents[1] / "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

measured_pose = importlib.import_module("measured_pose")

Vec3 = tuple[float, float, float]


@dataclass(frozen=True)
class MeshGeometry:
    """배치된 메시(월드 좌표)의 삼각형 배열. 넓이 0인 삼각형은 뺀다."""

    a: np.ndarray
    b: np.ndarray
    c: np.ndarray
    normals: np.ndarray  # 단위 법선(감김 부호 그대로)
    triangle_ids: np.ndarray

    @classmethod
    def from_mesh(cls, vertices: Sequence[Vec3], triangles: Sequence[Sequence[int]]) -> "MeshGeometry":
        v = np.asarray(vertices, dtype=float)
        t = np.asarray(triangles, dtype=int).reshape(-1, 3)
        a, b, c = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]
        n = np.cross(b - a, c - a)
        length = np.linalg.norm(n, axis=1)
        keep = length > 1e-15
        return cls(
            a=a[keep],
            b=b[keep],
            c=c[keep],
            normals=n[keep] / length[keep, None],
            triangle_ids=np.flatnonzero(keep),
        )

    def nearest(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """점마다 (거리, 삼각형 행 번호, 최근접점)."""
        p = np.atleast_2d(np.asarray(points, dtype=float))
        closest = _closest_points(p[:, None, :], self.a[None], self.b[None], self.c[None])
        dist = np.linalg.norm(closest - p[:, None, :], axis=2)
        row = np.argmin(dist, axis=1)
        pick = np.arange(len(p))
        return dist[pick, row], row, closest[pick, row]

    def first_hit(self, origin: np.ndarray, direction: np.ndarray) -> np.ndarray | None:
        """반직선이 처음 만나는 점(Möller–Trumbore). 없으면 None."""
        d = direction / np.linalg.norm(direction)
        e1 = self.b - self.a
        e2 = self.c - self.a
        pvec = np.cross(d[None], e2)
        det = np.einsum("ij,ij->i", e1, pvec)
        ok = np.abs(det) > 1e-15
        inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
        tvec = origin[None] - self.a
        u = np.einsum("ij,ij->i", tvec, pvec) * inv
        qvec = np.cross(tvec, e1)
        v = np.einsum("j,ij->i", d, qvec) * inv
        t = np.einsum("ij,ij->i", e2, qvec) * inv
        hit = ok & (u >= -1e-12) & (v >= -1e-12) & (u + v <= 1.0 + 1e-12) & (t > 1e-9)
        if not hit.any():
            return None
        t_min = float(np.min(t[hit]))
        return origin + t_min * d


def _dot(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.sum(x * y, axis=-1)


def _closest_points(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """삼각형 위 최근접점(Ericson, Real-Time Collision Detection §5.1.5), 브로드캐스트."""
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = _dot(ab, ap)
    d2 = _dot(ac, ap)
    bp = p - b
    d3 = _dot(ab, bp)
    d4 = _dot(ac, bp)
    cp = p - c
    d5 = _dot(ab, cp)
    d6 = _dot(ac, cp)
    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4

    with np.errstate(divide="ignore", invalid="ignore"):
        v_ab = d1 / (d1 - d3)
        w_ac = d2 / (d2 - d6)
        w_bc = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        denom = 1.0 / (va + vb + vc)
        v_in = vb * denom
        w_in = vc * denom

    shape = np.broadcast_shapes(p.shape, a.shape)
    a_b = np.broadcast_to(a, shape)
    b_b = np.broadcast_to(b, shape)
    c_b = np.broadcast_to(c, shape)
    inside = a_b + np.broadcast_to(ab, shape) * v_in[..., None] + np.broadcast_to(ac, shape) * w_in[..., None]
    on_ab = a_b + np.broadcast_to(ab, shape) * v_ab[..., None]
    on_ac = a_b + np.broadcast_to(ac, shape) * w_ac[..., None]
    on_bc = b_b + (c_b - b_b) * w_bc[..., None]

    conditions = [
        (d1 <= 0) & (d2 <= 0),
        (d3 >= 0) & (d4 <= d3),
        (vc <= 0) & (d1 >= 0) & (d3 <= 0),
        (d6 >= 0) & (d5 <= d6),
        (vb <= 0) & (d2 >= 0) & (d6 <= 0),
        (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0),
    ]
    choices = [a_b, b_b, on_ab, c_b, on_ac, on_bc]
    out = inside.copy()
    decided = np.zeros(shape[:-1], dtype=bool)
    for cond, choice in zip(conditions, choices):
        take = cond & ~decided
        out[take] = choice[take]
        decided |= take
    return out


def oriented_normal(geometry: MeshGeometry, row: int, toward: np.ndarray, at: np.ndarray) -> np.ndarray:
    """삼각형 법선을 `toward − at` 쪽으로 부호를 맞춘다(반사 정렬에서 감김 부호를 믿지 않는다)."""
    n = geometry.normals[row]
    return n if float(np.dot(n, toward - at)) >= 0.0 else -n


def spray_direction(row: Mapping[str, str], gun: str) -> np.ndarray | None:
    try:
        x, y, z = (float(row[f"{gun} Rotation {axis}"]) for axis in "XYZ")
    except (KeyError, ValueError):
        return None
    return np.asarray(measured_pose.forward_vector((x, y, z), measured_pose.GUN_FORWARD), dtype=float)


def composite_block(
    representative: Any,
    geometry: MeshGeometry | None,
    joints: Mapping[str, Sequence[float]] | None,
) -> dict[str, object] | None:
    """대표 프레임 한 장의 합성 정렬. `representative`는 `service.Representative`.

    메시가 없으면 착탄점만 싣고 나머지는 null이다.
    """
    if representative is None:
        return None
    frame = representative.frame
    hit = None if frame.hit is None else np.asarray(frame.hit, dtype=float)
    block: dict[str, object] = {
        "hit": None if hit is None else [float(x) for x in hit],
        "hit_to_mesh_mm": None,
        "pelvis_front_of_surface": None,
        "spray_ray_hits_mesh": None,
        "spray_ray_hit_miss_mm": None,
    }
    if geometry is None or hit is None:
        return block
    gun = np.asarray(frame.right, dtype=float)
    distance, rows, _closest = geometry.nearest(hit[None])
    block["hit_to_mesh_mm"] = float(distance[0] * 1000.0)
    if joints is not None and "Pelvis" in joints:
        n = geometry.normals[rows[0]]
        pelvis = np.asarray(joints["Pelvis"], dtype=float)
        block["pelvis_front_of_surface"] = bool(
            float(np.dot(pelvis - hit, n)) * float(np.dot(gun - hit, n)) > 0.0
        )
    direction = spray_direction(representative.row, representative.gun)
    if direction is not None:
        point = geometry.first_hit(gun, direction)
        block["spray_ray_hits_mesh"] = point is not None
        block["spray_ray_hit_miss_mm"] = None if point is None else float(np.linalg.norm(point - hit) * 1000.0)
    return block


def surface_normal_mean(
    geometry: MeshGeometry, hits: Sequence[Sequence[float]], guns: Sequence[Sequence[float]]
) -> np.ndarray | None:
    """착탄점마다 최근접 삼각형 법선(건 쪽으로 부호 맞춤)의 평균 방향."""
    if not hits:
        return None
    h = np.asarray(hits, dtype=float)
    g = np.asarray(guns, dtype=float)
    _d, rows, _c = geometry.nearest(h)
    normals = geometry.normals[rows]
    signs = np.sign(np.sum(normals * (g - h), axis=1))
    signs[signs == 0] = 1.0
    mean = np.mean(normals * signs[:, None], axis=0)
    length = float(np.linalg.norm(mean))
    return None if length < 1e-12 else mean / length


def angle_between_lines_deg(u: Sequence[float], v: Sequence[float]) -> float:
    """두 직선(부호 무관) 사이 각, arccos|u·v|."""
    a = np.asarray(u, dtype=float)
    b = np.asarray(v, dtype=float)
    cosine = abs(float(np.dot(a, b)) / (float(np.linalg.norm(a)) * float(np.linalg.norm(b))))
    return math.degrees(math.acos(min(1.0, cosine)))
