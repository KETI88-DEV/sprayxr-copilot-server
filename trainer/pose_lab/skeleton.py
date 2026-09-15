#!/usr/bin/env python3
"""전신 FK 스켈레톤을 실측 강체 **회전**으로 세우고 중력으로 투영해 놓는다(M6).

`pose_fk_csv.forward_kinematics_with_rotations`가 내는 20관절은 루트 `Pelvis`가 원점인
리그 좌표이고, 관절마다 리그 월드 회전 `W_j`도 함께 나온다. 옛 배치는 머리·양손 **세 점
위치**에 강체 Kabsch를 풀었다. 세 점이 전부 상체라 최약축 회전으로 발이 앵커 RMS의
7.5배(p90 20배) 흔들렸고, 실측 마커를 관절 위치로 취급해 건 마커(손 관절에서 약 0.64m)
쪽으로 몸을 끌고 갔다. 여기서는 그 둘을 다 버린다.

1. 루트 회전 `A0 = Procrustes(Σ_j R_meas_j · C_jᵀ · W_jᵀ)` — 세 강체(카메라·활성 건·플래쉬)의
   실측 회전으로 푼다. `C_j`는 관절↔강체 상수 장착 회전이다.
2. 중력 투영 `A = G · A0`. `G`는 `A0 · RIG_UP`을 `world_axes.UP`으로 보내는 최소 회전이다.
   **측정이 아니라 가정의 덮어쓰기**다 — 그래서 투영량 `gravity_correction_deg`를 숨기지 않고 낸다.
3. 이동 `t = mean_j(marker_j − R_meas_j·m_j − A·FK_j)`. `m_j`는 강체 로컬 상수 장착 오프셋이다.
4. `joints = A·p + t`. **강체 변환만** 하므로 뼈 길이는 리그 그대로다. 실측 좌표로 관절을
   덮어쓰지 않는다 — 대신 앵커마다 (예측 마커 − 실측 마커) 어긋남을 낸다.

`C_j`·`m_j`는 `fit_mounts`가 여러 프레임에서 교대 최소제곱으로 적합한다(사전등록
`docs/measurements/2026-09-15-skeleton-mounts-prereg.md` §3).

채점(σ·교정·개선율)은 이 경로를 타지 않는다. 표시 전용이다.

numpy를 쓴다. 트레이너 이미지에 이미 있고(`build_index.py` 등이 쓴다), 교대 적합이 프레임
수백 개 × 앵커 셋의 3x3 SVD라 표준 라이브러리로 짜면 요청 시간이 늘어난다.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Mapping, Sequence

import numpy as np

_SCRIPTS_DIR = str(Path(__file__).resolve().parents[1] / "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import measured_pose  # noqa: E402
import pose_fk_csv  # noqa: E402
import rotation_math  # noqa: E402
import simulator_csv  # noqa: E402
import world_axes  # noqa: E402

Vec3 = tuple[float, float, float]
Matrix = tuple[Vec3, Vec3, Vec3]

# 점 이름(measured_score 규약) → FK 관절 이름. 셋 다 실측 강체와 짝이다.
ANCHOR_JOINTS: Final[dict[str, str]] = {
    "head": "Head",
    "right": "RightHand",
    "left": "LeftHand",
}
ANCHOR_ORDER: Final[tuple[str, ...]] = ("head", "right", "left")

# 리그의 위 방향. YXZ 규약 FK에서 발→머리 몸통축이 리그 +Z에서 중앙값 10.1°만 흩어졌다
# (스크래치패드 sk_p7.py, 6세션 전 프레임). 옛 ZXY에서는 52.2°였다.
RIG_UP: Final[Vec3] = (0.0, 0.0, 1.0)

MIN_ANCHORS: Final = 2

INIT_ITERATIONS: Final = 40
ALTERNATING_ITERATIONS: Final = 20


@dataclass(frozen=True, slots=True)
class AnchorMount:
    """관절↔강체 장착. `R_meas ≈ A · W_joint · rotation`, `marker = joint + R_meas · offset`."""

    rotation: Matrix
    offset: Vec3


@dataclass(frozen=True, slots=True)
class Mounts:
    head: AnchorMount | None
    right: AnchorMount | None
    left: AnchorMount | None

    def get(self, point: str) -> AnchorMount | None:
        return getattr(self, point)


@dataclass(frozen=True, slots=True)
class AnchorFit:
    point: str
    joint: str
    joint_position: Vec3
    predicted_marker: Vec3
    measured_marker: Vec3
    miss_m: float


@dataclass(frozen=True, slots=True)
class Placement:
    joints: dict[str, Vec3]
    anchors: list[AnchorFit]
    rotation_residual_deg: dict[str, float | None]
    gravity_correction_deg: float
    leg_tilt_deg: float | None
    residual_m: float
    rotation: Matrix  # A (중력 투영 후)
    rotation_before_gravity: Matrix  # A0
    translation: Vec3


@dataclass(frozen=True, slots=True)
class FrameSample:
    """적합 한 프레임의 재료. `measured[point]`는 (강체 위치, 강체 회전) 또는 None."""

    local_pos: Mapping[str, Vec3]
    local_rot: Mapping[str, Matrix]
    measured: Mapping[str, tuple[Vec3, Matrix] | None]
    group: str = ""
    index: int = 0
    time_sec: float = 0.0
    extra: dict[str, object] = field(default_factory=dict)


# --- 선형대수 -------------------------------------------------------------------


def procrustes(matrix: np.ndarray) -> np.ndarray:
    """가장 가까운 회전(det +1). `matrix`는 (3,3) 또는 (n,3,3)."""
    u, _s, vt = np.linalg.svd(matrix)
    d = np.sign(np.linalg.det(u @ vt))
    d = np.where(d == 0.0, 1.0, d)
    diag = np.zeros(u.shape)
    diag[..., 0, 0] = 1.0
    diag[..., 1, 1] = 1.0
    diag[..., 2, 2] = d
    return u @ diag @ vt


def _angle_between_deg(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return float("nan")
    cosine = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def rotation_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    """두 회전 사이 측지 각."""
    rel = a.T @ b
    cosine = float(np.clip((np.trace(rel) - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _horizontal_fallback_axis() -> np.ndarray:
    first, _ = world_axes.HORIZONTAL_INDICES
    axis = np.zeros(3)
    axis[first] = 1.0
    return axis


def minimal_rotation(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """`source` 방향을 `target` 방향으로 보내는 최소 회전(로드리게스).

    반평행이면 회전축이 정해지지 않는다. 그때는 월드 수평축(`world_axes.HORIZONTAL_INDICES`의
    첫 축) 둘레 180°로 둔다 — `target`이 UP이면 그 축은 UP에 직교한다.
    """
    a = source / np.linalg.norm(source)
    b = target / np.linalg.norm(target)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c < -1.0 + 1e-12:
        axis = _horizontal_fallback_axis()
        axis = axis - np.dot(axis, b) * b
        axis = axis / np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    k = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    return np.eye(3) + k + k @ k * (1.0 / (1.0 + c))


def gravity_projection(a0: np.ndarray) -> tuple[np.ndarray, float]:
    """`A = G·A0`와 투영 전 기울기(도)."""
    up = np.array(world_axes.UP, dtype=float)
    rig_up_world = a0 @ np.array(RIG_UP)
    correction = _angle_between_deg(rig_up_world, up)
    return minimal_rotation(rig_up_world, up) @ a0, correction


def _as_tuple(v: np.ndarray) -> Vec3:
    return (float(v[0]), float(v[1]), float(v[2]))


def _as_matrix(m: np.ndarray) -> Matrix:
    return (_as_tuple(m[0]), _as_tuple(m[1]), _as_tuple(m[2]))


# --- 배치 -----------------------------------------------------------------------


def _usable_anchors(
    local_pos: Mapping[str, Vec3],
    local_rot: Mapping[str, Matrix],
    measured: Mapping[str, tuple[Vec3, Matrix] | None],
    mounts: Mounts,
) -> list[str]:
    usable: list[str] = []
    for point in ANCHOR_ORDER:
        joint = ANCHOR_JOINTS[point]
        if joint not in local_pos or joint not in local_rot:
            continue
        if measured.get(point) is None or mounts.get(point) is None:
            continue
        usable.append(point)
    return usable


def root_rotation(
    local_rot: Mapping[str, Matrix],
    measured: Mapping[str, tuple[Vec3, Matrix] | None],
    mounts: Mounts,
    points: Sequence[str],
) -> np.ndarray:
    """`A0 = Proc(Σ_j R_meas_j · C_jᵀ · W_jᵀ)`."""
    total = np.zeros((3, 3))
    for point in points:
        mount = mounts.get(point)
        record = measured.get(point)
        assert mount is not None and record is not None
        r_meas = np.array(record[1])
        c = np.array(mount.rotation)
        w = np.array(local_rot[ANCHOR_JOINTS[point]])
        total += r_meas @ c.T @ w.T
    return procrustes(total)


def place_oriented(
    local_pos: Mapping[str, Vec3],
    local_rot: Mapping[str, Matrix],
    measured: Mapping[str, tuple[Vec3, Matrix] | None],
    mounts: Mounts,
) -> Placement | None:
    """M6 배치. 앵커(실측 위치·회전과 장착 상수가 다 있는 점)가 2개 미만이면 None."""
    points = _usable_anchors(local_pos, local_rot, measured, mounts)
    if len(points) < MIN_ANCHORS:
        return None

    a0 = root_rotation(local_rot, measured, mounts, points)
    a, gravity_deg = gravity_projection(a0)

    offsets_world: dict[str, np.ndarray] = {}
    translations = []
    for point in points:
        record = measured[point]
        mount = mounts.get(point)
        assert record is not None and mount is not None
        r_meas = np.array(record[1])
        offsets_world[point] = r_meas @ np.array(mount.offset)
        target = np.array(record[0]) - offsets_world[point]
        translations.append(target - a @ np.array(local_pos[ANCHOR_JOINTS[point]]))
    t = np.mean(np.stack(translations), axis=0)

    joints = {name: _as_tuple(a @ np.array(p) + t) for name, p in local_pos.items()}

    anchors: list[AnchorFit] = []
    rotation_residual: dict[str, float | None] = {point: None for point in ANCHOR_ORDER}
    for point in points:
        record = measured[point]
        mount = mounts.get(point)
        assert record is not None and mount is not None
        joint = ANCHOR_JOINTS[point]
        joint_position = np.array(joints[joint])
        predicted = joint_position + offsets_world[point]
        measured_marker = np.array(record[0])
        anchors.append(
            AnchorFit(
                point=point,
                joint=joint,
                joint_position=_as_tuple(joint_position),
                predicted_marker=_as_tuple(predicted),
                measured_marker=(float(record[0][0]), float(record[0][1]), float(record[0][2])),
                miss_m=float(np.linalg.norm(predicted - measured_marker)),
            )
        )
        rotation_residual[point] = rotation_angle_deg(
            a @ np.array(local_rot[joint]) @ np.array(mount.rotation), np.array(record[1])
        )

    residual = math.sqrt(sum(fit.miss_m**2 for fit in anchors) / len(anchors))

    leg_tilt: float | None = None
    if all(name in joints for name in ("Pelvis", "LeftFoot", "RightFoot")):
        feet = (np.array(joints["LeftFoot"]) + np.array(joints["RightFoot"])) / 2.0
        down = -np.array(world_axes.UP, dtype=float)
        leg_tilt = _angle_between_deg(feet - np.array(joints["Pelvis"]), down)

    return Placement(
        joints=joints,
        anchors=anchors,
        rotation_residual_deg=rotation_residual,
        gravity_correction_deg=gravity_deg,
        leg_tilt_deg=leg_tilt,
        residual_m=residual,
        rotation=_as_matrix(a),
        rotation_before_gravity=_as_matrix(a0),
        translation=_as_tuple(t),
    )


# --- 적합(사전등록 §3) --------------------------------------------------------------


def mount_key(point: str, group: str) -> str:
    """오른손은 건 그룹마다 따로 적합한다."""
    return f"right:{group}" if point == "right" else point


def _sample_keys(sample: FrameSample) -> dict[str, str]:
    return {
        point: mount_key(point, sample.group)
        for point in ANCHOR_ORDER
        if sample.measured.get(point) is not None
        and ANCHOR_JOINTS[point] in sample.local_rot
        and ANCHOR_JOINTS[point] in sample.local_pos
    }


def _init_rotations(sessions: Sequence[Sequence[FrameSample]]) -> dict[str, np.ndarray]:
    """세션마다 상수 A 하나를 두고 40회 교대 → C_{j,s}; 초기 C_j = Proc(Σ_s C_{j,s})."""
    sums: dict[str, np.ndarray] = {}
    for session in sessions:
        per_key: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
        for sample in session:
            for point, key in _sample_keys(sample).items():
                record = sample.measured[point]
                assert record is not None
                per_key.setdefault(key, []).append(
                    (np.array(record[1]), np.array(sample.local_rot[ANCHOR_JOINTS[point]]))
                )
        for key, pairs in per_key.items():
            r = np.stack([p[0] for p in pairs])
            w = np.stack([p[1] for p in pairs])
            c = np.eye(3)
            for _ in range(INIT_ITERATIONS):
                a = procrustes((r @ c.T @ np.transpose(w, (0, 2, 1))).sum(axis=0))
                c = procrustes((np.transpose(w, (0, 2, 1)) @ a.T @ r).sum(axis=0))
            sums[key] = sums.get(key, np.zeros((3, 3))) + c
    return {key: procrustes(total) for key, total in sums.items()}


def _stacked(samples: Sequence[FrameSample], keys: Mapping[str, object]) -> dict[str, dict[str, np.ndarray]]:
    """키마다 (표본 번호, R_meas, W, 마커, FK 위치)를 쌓는다 — 교대 적합을 벡터로 돌리려고."""
    acc: dict[str, dict[str, list]] = {}
    for n, sample in enumerate(samples):
        for point, key in _sample_keys(sample).items():
            if key not in keys:
                continue
            record = sample.measured[point]
            assert record is not None
            joint = ANCHOR_JOINTS[point]
            slot = acc.setdefault(key, {"idx": [], "r": [], "w": [], "marker": [], "fk": []})
            slot["idx"].append(n)
            slot["r"].append(record[1])
            slot["w"].append(sample.local_rot[joint])
            slot["marker"].append(record[0])
            slot["fk"].append(sample.local_pos[joint])
    return {key: {name: np.array(values, dtype=float) if name != "idx" else np.array(values, dtype=int) for name, values in slot.items()} for key, slot in acc.items()}


def _batched_root(
    n_samples: int, stacks: Mapping[str, Mapping[str, np.ndarray]], rotations: Mapping[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """표본마다 A0 = Proc(Σ_j R C_jᵀ Wᵀ). 앵커가 2개 미만인 표본은 mask False."""
    total = np.zeros((n_samples, 3, 3))
    count = np.zeros(n_samples, dtype=int)
    for key, stack in stacks.items():
        if key not in rotations:
            continue
        contrib = stack["r"] @ rotations[key].T @ np.transpose(stack["w"], (0, 2, 1))
        np.add.at(total, stack["idx"], contrib)
        np.add.at(count, stack["idx"], 1)
    mask = count >= MIN_ANCHORS
    a0 = np.broadcast_to(np.eye(3), (n_samples, 3, 3)).copy()
    if mask.any():
        a0[mask] = procrustes(total[mask])
    return a0, mask


def _batched_gravity(a0: np.ndarray) -> np.ndarray:
    up = np.array(world_axes.UP, dtype=float)
    return np.stack([minimal_rotation(a @ np.array(RIG_UP), up) @ a for a in a0]) if len(a0) else a0


def fit_mount_rotations(
    sessions: Sequence[Sequence[FrameSample]], iterations: int = ALTERNATING_ITERATIONS
) -> dict[str, np.ndarray]:
    """사전등록 §3-1·2. 표본마다 A0(t)를 풀고 C_j를 풀링해 푸는 교대를 `iterations`번."""
    rotations = _init_rotations(sessions)
    samples = [sample for session in sessions for sample in session]
    stacks = _stacked(samples, rotations)
    for _ in range(iterations):
        a0, mask = _batched_root(len(samples), stacks, rotations)
        updated: dict[str, np.ndarray] = {}
        for key, stack in stacks.items():
            use = mask[stack["idx"]]
            if not use.any():
                continue
            a_t = np.transpose(a0[stack["idx"][use]], (0, 2, 1))
            updated[key] = procrustes(
                (np.transpose(stack["w"][use], (0, 2, 1)) @ a_t @ stack["r"][use]).sum(axis=0)
            )
        rotations = updated
    return rotations


def fit_mount_offsets(
    samples: Sequence[FrameSample], rotations: Mapping[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """사전등록 §3-3. 차분식 `marker_a − marker_b − A(FK_a − FK_b) = R_a m_a − R_b m_b`의
    선형 최소제곱. A는 중력 투영한 루트 회전이다."""
    stacks = _stacked(samples, rotations)
    if not stacks:
        return {}
    a0, mask = _batched_root(len(samples), stacks, rotations)
    a = a0.copy()
    if mask.any():
        a[mask] = _batched_gravity(a0[mask])
    keys = sorted(stacks)
    column = {key: i for i, key in enumerate(keys)}
    # 표본 번호 → 키별 행 위치.
    where: dict[str, np.ndarray] = {}
    for key, stack in stacks.items():
        lookup = np.full(len(samples), -1, dtype=int)
        lookup[stack["idx"]] = np.arange(len(stack["idx"]))
        where[key] = lookup
    blocks: list[np.ndarray] = []
    rhs: list[np.ndarray] = []
    for i, ka in enumerate(keys):
        for kb in keys[i + 1 :]:
            both = mask & (where[ka] >= 0) & (where[kb] >= 0)
            if not both.any():
                continue
            ia = where[ka][both]
            ib = where[kb][both]
            sa, sb = stacks[ka], stacks[kb]
            lhs = (sa["marker"][ia] - sb["marker"][ib]) - np.einsum(
                "nij,nj->ni", a[both], sa["fk"][ia] - sb["fk"][ib]
            )
            row = np.zeros((int(both.sum()), 3, 3 * len(keys)))
            row[:, :, 3 * column[ka] : 3 * column[ka] + 3] += sa["r"][ia]
            row[:, :, 3 * column[kb] : 3 * column[kb] + 3] -= sb["r"][ib]
            blocks.append(row.reshape(-1, 3 * len(keys)))
            rhs.append(lhs.reshape(-1))
    if not blocks:
        return {}
    solution, *_ = np.linalg.lstsq(np.concatenate(blocks), np.concatenate(rhs), rcond=None)
    return {key: solution[3 * column[key] : 3 * column[key] + 3] for key in keys}


def _mounts_from(
    present: Mapping[str, str],
    rotations: Mapping[str, np.ndarray],
    offsets: Mapping[str, np.ndarray] | None,
) -> Mounts:
    def build(point: str) -> AnchorMount | None:
        key = present.get(point)
        if key is None or key not in rotations:
            return None
        offset = (0.0, 0.0, 0.0) if offsets is None or key not in offsets else _as_tuple(offsets[key])
        return AnchorMount(rotation=_as_matrix(rotations[key]), offset=offset)

    return Mounts(head=build("head"), right=build("right"), left=build("left"))


def fit_mounts(sessions: Sequence[Sequence[FrameSample]]) -> dict[str, AnchorMount]:
    """사전등록 §3 전체: 회전 초기값 → 회전 교대 → 오프셋 최소제곱. 키는 `mount_key`."""
    rotations = fit_mount_rotations(sessions)
    samples = [sample for session in sessions for sample in session]
    offsets = fit_mount_offsets(samples, rotations)
    return {
        key: AnchorMount(rotation=_as_matrix(rotations[key]), offset=_as_tuple(offsets[key]))
        for key in rotations
        if key in offsets
    }


def mounts_for(table: Mapping[str, AnchorMount], group: str) -> Mounts:
    return Mounts(
        head=table.get("head"),
        right=table.get(mount_key("right", group)),
        left=table.get("left"),
    )


def identity_mounts() -> Mounts:
    """장착 없음 기준선(C=I, m=0) — 사전등록 §4."""
    none = AnchorMount(rotation=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)), offset=(0.0, 0.0, 0.0))
    return Mounts(head=none, right=none, left=none)


def bone_length_errors(joints: Mapping[str, Vec3], offsets: Mapping[str, Vec3]) -> list[float]:
    """배치된 뼈 길이와 바인드 오프셋 노름의 차(m). 스냅이 끼어들면 0이 아니게 된다."""
    errors: list[float] = []
    for parent, child in simulator_csv.BONES:
        if parent in joints and child in joints and child in offsets:
            errors.append(abs(math.dist(joints[parent], joints[child]) - math.hypot(*offsets[child])))
    return errors


# --- CSV 행 → 배치 재료 --------------------------------------------------------------


def body_rotation(row: Mapping[str, str], group: str) -> Matrix | None:
    """강체 회전. 규약은 실측 강체와 같은 `measured_pose.EULER_ORDER`/`EULER_SIGN`이다."""
    try:
        angles = simulator_csv.vec3(dict(row), group, "Rotation")
    except (KeyError, ValueError):
        return None
    return rotation_math.euler_matrix(
        measured_pose.EULER_ORDER, (angles[0], angles[1], angles[2]), measured_pose.EULER_SIGN
    )


def measured_bodies(frame: object, row: Mapping[str, str], gun: str) -> dict[str, tuple[Vec3, Matrix] | None]:
    """프레임의 실측 세 강체 (위치, 회전). 위치는 `MeasuredFrame`의 것 그대로다 —
    왼손은 `measured_pose`의 거리 필터에서 떨어지면(`frame.left is None`) 앵커가 아니다."""
    head_rot = body_rotation(row, measured_pose.HEAD_GROUP)
    gun_rot = body_rotation(row, gun)
    left_rot = body_rotation(row, measured_pose.LEFT_GROUP)
    left = getattr(frame, "left", None)
    return {
        "head": None if head_rot is None else (tuple(frame.head), head_rot),  # type: ignore[attr-defined]
        "right": None if gun_rot is None else (tuple(frame.right), gun_rot),  # type: ignore[attr-defined]
        "left": None if left is None or left_rot is None else (tuple(left), left_rot),
    }


def frame_sample(
    frame: object, row: Mapping[str, str], gun: str, offsets: Mapping[str, Vec3]
) -> FrameSample | None:
    """원본 행 하나를 배치·적합 재료로. 관절 컬럼이 없으면 None."""
    if not offsets:
        return None
    positions, rotations = pose_fk_csv.forward_kinematics_with_rotations(dict(row), offsets)
    return FrameSample(
        local_pos=positions,
        local_rot=rotations,
        measured=measured_bodies(frame, row, gun),
        group=gun,
        index=int(getattr(frame, "index", 0)),
        time_sec=float(getattr(frame, "time_sec", 0.0)),
    )
