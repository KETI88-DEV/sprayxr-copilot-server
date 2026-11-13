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
from typing import Any, Final, Mapping, Sequence

import numpy as np

_SCRIPTS_DIR = str(Path(__file__).resolve().parents[1] / "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

measured_pose = importlib.import_module("measured_pose")
world_axes = importlib.import_module("world_axes")

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


# --- 바닥면(자리표시 다리를 세울 높이) --------------------------------------------------
#
# 자리표시 다리(`skeleton.standing_legs`)는 측정이 아니라 표시다. 리그 제 길이로 내리면 레벨 3에서는
# 발이 바닥면보다
# 0.525m 아래로 내려간다(실측: 가장 낮은 자리표시 발 수직 3.006m,
# 바닥면 3.531m). 여기서 "바닥"을 하나로 정한다 — **수평에 가까운 면(법선이 월드 수직축과 이룬 각
# ≤ FLOOR_MAX_TILT_DEG) 중 가장 낮은 면의 최고점**. "착탄점 아래에서 가장 높은 수평면"은 L6에서
# 4.33m(골반 4.085m 보다 위)를 골라 다리를 거꾸로 세웠다 — 그래서 쓰지 않는다.
# **이 면을 고르는 규칙은 그대로 두고, 다리를 그 면에 닿게 하려고 줄이던 배율만 없앴다**(D3 재판정):
# 발이 이 면보다 아래로 내려가는 것을 허용하고, 얼마나 아래인지를 화면이 말한다.
#
# **이 면은 작업자가 선 지면이 맞다.** 예전 이 자리에는 "지면이 아닐 수 있다"고 적혀 있었고, 그
# 문장이 자리표시 다리를 없앤 근거였다. 규약 없는 실측이 그것을 반증했다.
#
# 반증: HMD(머리 강체)는 **사람 눈높이**라 좌표 규약과 무관하다. 작업자 6명 × solved 9레벨(바닥면을
# 고를 수 있는 레벨) 429개 CSV의 모든 실측 프레임에서 이 면 위 HMD 높이를 모으니, 작업자×레벨 54칸 중
# **51칸의 p95가 1.5~1.8m**였다(전체 범위 1.527~1.824m, 벗어난 3칸은 모두 한 작업자의 1.808~1.824m).
# 바닥을 잘못 골랐다면 레벨마다 다른 양만큼 틀려야 하는데, 그렇지 않다. 이 수치는 서로 다른 두
# 라운드에서 **각각 코퍼스를 다시 훑어** 같은 값이 나왔다(둘 다 위 범위, 아래 p95 대역도 자리까지 같다).
#
# 아래 세 줄은 **한 패스의 대표 프레임 한 장**(김경호 20250831 L1·L3·L6 첫 패스) 값이지 레벨의
# 성질이 아니다 — 그 구별이 이 주석이 한 번 틀렸던 자리다:
#   바닥면 수직: L1 3.557m · L3 3.531m · L6 3.560m
#   이 면 위 머리 실측 마커(HMD): L1 1.508m · L3 0.696m · L6 0.879m
#   이 면 위 골반: L1 1.114m · L3 0.349m · L6 0.523m
# 같은 면 위 **레벨 중앙값**은 L1 1.534m · L3 1.450m · L6 1.172m이다 — 집계 방식은 **작업자별
# 중앙값을 낸 뒤 그 6개의 중앙값**이지 프레임을 통째로 모아 낸 중앙값이 아니다(두 집계는 값이
# 다르다). p95는 모든 레벨에서 선 키를 회복한다(작업자×레벨 칸의 p95 범위:
# L1 1.600~1.813m · L3 1.632~1.824m · L6 1.579~1.793m).
# 이 두 줄은 코퍼스를 다시 훑어야 나오는 값이라 검사가 재계산하지 못한다 — 그래서 **어떻게 집계한
# 값인지**를 여기 적어 둔다(검사는 이 값들이 한 패스 값과 다르다는 것만 고정한다).
# 중앙값이 선 키보다 낮은 것은
# 도장 작업 중 몸을 굽히고 있는 시간이 길기 때문이다. 요컨대 L3·L6의 0.70m·0.88m는 "바닥이 틀렸다"가
# 아니라 **그 패스에서 작업자가 쭈그렸다**는 뜻이다 — 자리표시 다리는 그래서 무릎을 접어 세운다
# (`skeleton.standing_legs(reach_floor=True)`).
#
# 위 숫자는 주석이 아니라 **코드가 내는 값**이어야 한다 — `test_composite.test_floor_comment_numbers_match_the_code`와
# `test_composite.test_floor_choice_comment_reports_what_stands_on_it`이 이 주석에서 숫자를 뽑아 실제
# 메시·실제 배치로 다시 재서 비교한다(출처와 다른 라벨을 막는다).
FLOOR_MAX_TILT_DEG: Final = 30.0


def floor_vertical(geometry: MeshGeometry | None) -> float | None:
    """메시 바닥면의 월드 수직 높이. 수평에 가까운 면이 하나도 없으면 None."""
    if geometry is None or len(geometry.normals) == 0:
        return None
    up = np.asarray(world_axes.UP, dtype=float)
    horizontal = np.abs(geometry.normals @ up) >= math.cos(math.radians(FLOOR_MAX_TILT_DEG))
    if not horizontal.any():
        return None
    tops = np.max(np.stack([geometry.a @ up, geometry.b @ up, geometry.c @ up], axis=1), axis=1)
    return float(np.min(tops[horizontal]))


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
        "spray_ray_angle_deg": None,
    }
    if hit is None:
        return block
    gun = np.asarray(frame.right, dtype=float)
    direction = spray_direction(representative.row, representative.gun)
    if direction is not None and float(np.linalg.norm(hit - gun)) > 0.0:
        # 거리와 무관한 척도: 건 분사축과 (착탄점 − 건 마커) 사이 각. miss_mm는 건–벽 거리에 비례해 커지므로
        # 레벨마다 같은 문턱으로 읽을 수 없다(2라운드 검증: L10 55.9mm). 코퍼스 64,098패스 중앙값 3.89°.
        block["spray_ray_angle_deg"] = math.degrees(
            math.acos(float(np.clip(np.dot(direction, hit - gun) / (np.linalg.norm(direction) * np.linalg.norm(hit - gun)), -1.0, 1.0)))
        )
    if geometry is None:
        return block
    distance, rows, _closest = geometry.nearest(hit[None])
    block["hit_to_mesh_mm"] = float(distance[0] * 1000.0)
    if joints is not None and "Pelvis" in joints:
        n = geometry.normals[rows[0]]
        pelvis = np.asarray(joints["Pelvis"], dtype=float)
        block["pelvis_front_of_surface"] = bool(
            float(np.dot(pelvis - hit, n)) * float(np.dot(gun - hit, n)) > 0.0
        )
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


MAX_SURFACE_HITS = 200


def surface_hits(representative: Any) -> list[list[float]]:
    """대표 패스의 **원본** 분사 착탄점(균등 추출, 최대 `MAX_SURFACE_HITS`).

    `view.layers.query.hit`은 호길이 격자로 리샘플한 점이라 모서리를 가로지르면 벽에서 뜬다(L6 격자점
    최대 62.9mm). 벽을 함께 보는 화면(벽 포함 맞춤 등)은 이 원본 점을 쓴다.
    """
    if representative is None:
        return []
    hits = [frame.hit for frame, _row in representative.spraying if frame.hit is not None]
    if len(hits) > MAX_SURFACE_HITS:
        picks = np.linspace(0, len(hits) - 1, MAX_SURFACE_HITS).astype(int)
        hits = [hits[i] for i in picks]
    return [[float(x) for x in hit] for hit in hits]


# --- 벽 위 분사 스트로크(표시 전용) ---------------------------------------------------------
#
# 사용자가 화면을 보고 "벽이랑 시퀀스 접촉점이랑 정렬이 안됨"이라고 했고, 이어서 "'연결된 하나의
# 시퀀스'(=스트로크)만 남기고 부분적으로 뿌려진 건 다 없앴으면 함"이라고 했다.
#
# **데이터는 벽에 붙어 있다.** 씬 그래프에서 직접 잰 값(three.js에 올라간 벽 삼각형까지의 월드 거리):
# `surface_hits` 정점은 L3(38점)·L6(77점) 모두 min/중앙/p95/최대가 전부 0.00mm였다. 뜨는 것은 점이
# 아니라 **점을 잇는 선**이다. 폴리라인의 현이 꺾인 면을 가로지르면 공중을 질러간다:
#
#   L6 김경호 1000394_1 #0 — 77점 · 76세그먼트 · 총 길이 1.668m · 세그먼트 길이 중앙값 16.1mm
#     세그먼트 #73: 길이 343.4mm · 양 끝 면 법선 사잇각 90.0° · 현 최대 이탈 84.59mm(t=0.35)
#     세그먼트 #65: 길이  73.8mm · 법선 사잇각 66.7°      · 현 최대 이탈 21.11mm
#     나머지 74개: 전부 2.5mm 이하
#   L3 김경호 1000391_1 #0 — 38점 · 37세그먼트 · 5mm를 넘는 세그먼트 0개(최악 0.83mm, 법선 45° 구간)
#
# 그래서 뜨는 현에서 **끊는다**. 현을 표면에 투영해 붙이거나 점을 옮기지 않는다 — 그러면 없는 접촉을
# 그리게 된다. 데이터가 끊겨 있으면 끊긴 대로 보이는 편이 낫다. 화면은 몇 곳에서 끊었는지 말한다.
#
# 문턱은 위 실측에서 나온다: 모서리가 아닌 세그먼트의 최대 이탈이 2.5mm(L6)·0.83mm(L3)이므로
# 그 2배를 문턱으로 둔다. 모서리 두 곳(21.11mm·84.59mm)과는 4배 넘게 떨어져 있어 어느 쪽으로도
# 아슬아슬하지 않다.
STROKE_MAX_LIFT_MM: Final = 5.0
# 현 위를 훑는 간격과 세그먼트당 표본 상한. 2mm면 L6 최악 세그먼트(343mm)도 63표본으로 본다.
STROKE_CHORD_STEP_M: Final = 0.002
STROKE_CHORD_SAMPLES_MAX: Final = 64

STROKE_NOTE_CLEAN: Final = (
    "벽 위 분사 스트로크는 착탄점을 이은 한 줄입니다 — 전 구간이 벽에 붙어 있습니다"
)
STROKE_NOTE_SPLIT: Final = (
    "벽 위 분사 스트로크는 착탄점을 이은 한 줄입니다 — 이으면 벽에서 뜨는 {breaks}곳에서 "
    "끊어 그렸습니다, 가장 크게 뜨던 곳이 {lift:.0f}mm입니다"
)
STROKE_NOTE_NO_MESH: Final = (
    "벽 위 분사 스트로크는 착탄점을 이은 한 줄입니다 — 이 패스는 벽 메시가 없어 벽과의 거리를 "
    "재지 않았습니다"
)


def chord_lift_mm(geometry: MeshGeometry, start: Sequence[float], end: Sequence[float]) -> float:
    """두 착탄점을 이은 **현**이 벽에서 가장 크게 떨어지는 거리(mm). 양 끝은 착탄점이라 0이다."""
    p = np.asarray(start, dtype=float)
    q = np.asarray(end, dtype=float)
    seg = q - p
    length = float(np.linalg.norm(seg))
    if length <= 1e-12:
        return 0.0
    steps = int(min(STROKE_CHORD_SAMPLES_MAX, max(2, math.ceil(length / STROKE_CHORD_STEP_M))))
    ts = np.arange(1, steps, dtype=float) / steps
    if ts.size == 0:
        return 0.0
    distance, _rows, _closest = geometry.nearest(p[None] + ts[:, None] * seg[None])
    return float(np.max(distance)) * 1000.0


def stroke_runs(
    geometry: MeshGeometry | None, hits: Sequence[Sequence[float]]
) -> dict[str, object]:
    """벽에 붙은 채로 이어지는 **연결 구간들**로 자른 분사 스트로크(원본 착탄점만).

    돌려주는 `runs`의 각 원소는 끊기지 않은 폴리라인이다. 점을 옮기거나 만들지 않는다 —
    자르기만 한다. 메시가 없으면 잴 수 없으므로 자르지 않고 한 줄로 낸다.
    """
    points = [[float(x) for x in hit] for hit in hits if hit is not None and len(hit) == 3]
    block: dict[str, object] = {
        "runs": [],
        "point_count": len(points),
        "break_count": 0,
        "max_lift_mm": None,
        "max_kept_lift_mm": None,
        "tolerance_mm": STROKE_MAX_LIFT_MM,
        "note": STROKE_NOTE_NO_MESH,
    }
    if len(points) < 2:
        block["runs"] = [points] if points else []
        return block
    if geometry is None or len(geometry.a) == 0:
        block["runs"] = [points]
        return block

    lifts = [chord_lift_mm(geometry, points[i], points[i + 1]) for i in range(len(points) - 1)]
    runs: list[list[list[float]]] = []
    current = [points[0]]
    kept: list[float] = []
    for index, lift in enumerate(lifts):
        if lift > STROKE_MAX_LIFT_MM:
            runs.append(current)
            current = [points[index + 1]]
            continue
        kept.append(lift)
        current.append(points[index + 1])
    runs.append(current)
    breaks = len(runs) - 1
    block["runs"] = [run for run in runs if len(run) >= 2]
    block["break_count"] = breaks
    block["max_lift_mm"] = max(lifts)
    block["max_kept_lift_mm"] = max(kept) if kept else None
    block["note"] = (
        STROKE_NOTE_SPLIT.format(breaks=breaks, lift=max(lifts))
        if breaks > 0
        else STROKE_NOTE_CLEAN
    )
    return block
