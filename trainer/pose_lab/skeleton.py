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


# --- 다리 해부학 범위(표시 규칙, 게이트 아님) ------------------------------------------
#
# 다리 관절에는 추적 장치가 없다 — HMD·건·플래쉬 세 강체에서 시뮬레이터 IK가 지은 값이다. 학습모드
# 25세션의 분사 프레임 89,460개(리그 좌표, RIG_UP 기준)에서 무릎이 발끝 반대로 꺾이거나 두 발이 교차하는
# 프레임이 흔했다. 아래 문턱은 그 분포를 **본 뒤** 정한 표시 규칙이다(스크래치패드 r2/plaus.py·plaus2.py):
# 무릎 굴곡 ≤110°(74.3% 충족), 굽은 무릎(≥20°)의 굽는 방향과 발끝 방향 사이 ≤60°(71.2%), 좌우 발끝의
# 수평 벌어짐 ≤60°(65.3%), 골반→양발 중점의 수직 기울기 ≤25°(81.9%), 다리 교차 없음(89.0%). 다섯 개를
# 모두 충족하는 프레임은 37.1%였다. 채점·교정은 이 값을 읽지 않는다 — 대표 프레임 선택과 다리 표시에만 쓴다.
LEG_LIMITS: Final[dict[str, float]] = {
    "knee_flex_deg": 110.0,
    "knee_toe_deg": 60.0,
    "toe_splay_deg": 60.0,
    "leg_tilt_deg": 25.0,
}
KNEE_BENT_DEG: Final = 20.0
LEG_JOINTS: Final[tuple[str, ...]] = (
    "LeftThigh", "LeftCalf", "LeftFoot", "LeftToe",
    "RightThigh", "RightCalf", "RightFoot", "RightToe",
)


@dataclass(frozen=True, slots=True)
class LegCheck:
    knee_flex_deg: float  # 두 다리 중 큰 값
    knee_toe_deg: float  # 굽은 무릎(≥ KNEE_BENT_DEG)만 본다. 두 다리 중 큰 값, 둘 다 곧으면 0
    toe_splay_deg: float
    legs_crossed: bool
    leg_tilt_deg: float
    plausible: bool

    def as_dict(self) -> dict[str, float | bool]:
        return {
            "knee_flex_deg": self.knee_flex_deg,
            "knee_toe_deg": self.knee_toe_deg,
            "toe_splay_deg": self.toe_splay_deg,
            "legs_crossed": self.legs_crossed,
            "leg_tilt_deg": self.leg_tilt_deg,
        }


def leg_check(local_pos: Mapping[str, Vec3]) -> LegCheck | None:
    """리그 좌표(RIG_UP이 위)에서 다리가 해부학 범위 안인지. 다리 관절이 하나라도 없으면 None.

    무릎 굽는 방향은 엉덩이→발목 선에 수직인 무릎 성분이다. 사람 무릎은 발끝 쪽으로 굽는다 —
    반대로 꺾이면(각이 크면) 범위 밖이다. 각이 정의되지 않으면(NaN) 범위 밖으로 본다.
    """
    names = ("Pelvis",) + LEG_JOINTS
    if any(name not in local_pos for name in names):
        return None
    p = {name: np.array(local_pos[name], dtype=float) for name in names}
    up = np.array(RIG_UP, dtype=float)

    def flat(v: np.ndarray) -> np.ndarray:
        return v - float(np.dot(v, up)) * up

    flexes: list[float] = []
    knee_toes: list[float] = []
    for side in ("Left", "Right"):
        hip, knee, ankle, toe = p[side + "Thigh"], p[side + "Calf"], p[side + "Foot"], p[side + "Toe"]
        flex = 180.0 - _angle_between_deg(hip - knee, ankle - knee)
        flexes.append(flex)
        if flex >= KNEE_BENT_DEG:
            line = ankle - hip
            t = float(np.dot(knee - hip, line) / np.dot(line, line))
            bend = knee - (hip + t * line)
            knee_toes.append(_angle_between_deg(bend, toe - ankle))
        else:
            knee_toes.append(0.0)
    splay = _angle_between_deg(flat(p["LeftToe"] - p["LeftFoot"]), flat(p["RightToe"] - p["RightFoot"]))
    crossed = bool(np.dot(p["LeftFoot"] - p["RightFoot"], p["LeftThigh"] - p["RightThigh"]) <= 0.0)
    feet = (p["LeftFoot"] + p["RightFoot"]) / 2.0
    tilt = _angle_between_deg(p["Pelvis"] - feet, up)
    values = {
        "knee_flex_deg": max(flexes),
        "knee_toe_deg": max(knee_toes),
        "toe_splay_deg": splay,
        "leg_tilt_deg": tilt,
    }
    within = all(math.isfinite(values[key]) and values[key] <= limit for key, limit in LEG_LIMITS.items())
    return LegCheck(
        knee_flex_deg=values["knee_flex_deg"],
        knee_toe_deg=values["knee_toe_deg"],
        toe_splay_deg=values["toe_splay_deg"],
        legs_crossed=crossed,
        leg_tilt_deg=values["leg_tilt_deg"],
        plausible=within and not crossed,
    )


# --- 자리표시 다리(표시 전용, 측정 아님) -------------------------------------------------
#
# 다리 관절에는 추적 장치가 없고 IK 다리는 해부학 범위를 벗어난 프레임이 많다(위 LEG_LIMITS 주석).
# 다리를 숨기면 상체만 떠 사람 자세로 읽히지 않았고(2라운드 검증), 범위 안 IK 다리도 사선으로 뻗어
# 원래 불만("다리가 대각선")을 되풀이했다. 그래서 기본 표시는 **곧게 선 자리표시 다리**다: 엉덩이
# 관절(골반에 붙은 FK 위치)에서 중력 아래로 리그 허벅지·종아리 길이만큼 내리고, 발끝은 대표 프레임
# 착탄 방향(수평)으로 리그 발 길이만큼 둔다. 뼈 길이는 리그 그대로다. 이 다리는 자세를 말하지 않는다 —
# 화면은 점선으로 그리고 notes가 "측정 아님"을 적는다.
PLACEHOLDER_LEG_CHAIN: Final[tuple[tuple[str, str, str, str], ...]] = (
    ("LeftThigh", "LeftCalf", "LeftFoot", "LeftToe"),
    ("RightThigh", "RightCalf", "RightFoot", "RightToe"),
)

# **줄이는 배율은 절대 쓰지 않는다.** 옛 코드는 자리표시 발을 `composite.floor_vertical`이 고른
# 면에 닿게 하려고 허벅지·종아리 길이에 한 배율(0.25~2.0)을 곱했고, 그 배율이 레벨 3에서 **0.35**,
# 레벨 6에서 0.55가 돼 화면의 작업자가 난쟁이가 됐다(옛 리그 실측). 그 판단은 그대로 유효하다 —
# 엉덩이가 바닥보다 낮게 있을 만큼 가까우면 **무릎이** 남는 길이를 접고, 길이는 건드리지 않는다.
#
# **고른 면은 지면이 맞다.** 예전 이 자리에는 "L3·L6에서 고른 면은 작업자가 선 지면이 아니다"라고
# 적혀 있었고 그 문장이 자리표시 다리를 없앤 근거였는데, 규약 없는 실측(HMD 눈높이)이 그것을
# 반증했다 — 근거는 `composite.py`의 바닥 주석에 있다. 그 문장이 읽던 L3 0.70m·L6 0.88m는
# **레벨의 성질이 아니라 한 패스 한 프레임**의 값이다(레벨 중앙값은 1.43m·1.11m, p95는 선 키를
# 회복한다). 작업자가 그 패스에서 실제로 쭈그린 것이다.
#
# 그래서 두 갈래다.
#   `reach_floor=False`(기본, 진단용 접촉 시트): 리그 제 길이로 곧게 내린다. 발이 고른 면보다
#     아래로 내려가는 것을 허용한다 — 레벨 3에서는 발이 바닥면 0.605m 아래다. 그 관계는 숨기지
#     않고 `PlaceholderLegs.feet_below_floor_m`가 그대로 싣는다.
#   `reach_floor=True`(화면): 발목을 그 면에 **세운다**. 사용자 정책이 "벽 메시가 있으면 발을
#     바닥면에 세운다"이므로 이 갈래는 뜬 발을 남기지 않는다. 어떻게 세우는지는 엉덩이 높이가 가른다.
#       엉덩이–바닥 거리가 리그 다리보다 **짧으면**: **무릎을 굽힌다**(2마디 IK). 남는 길이를
#         무릎이 흡수한다 — 쭈그린 패스에서 쭈그린 그림이 나온다. 각도라 뼈 길이를 거짓말하지 않는다.
#         실측 김경호 L3·L6 첫 패스가 여기다(엉덩이 바닥 위 0.269·0.519m).
#       엉덩이가 리그 다리보다 **높으면**: 자리표시 다리를 그 거리까지 **늘린다**(허벅지·종아리에
#         같은 배율, 1.0 미만은 절대 없다). 실측 김경호 L1 첫 패스가 여기다 — 엉덩이가 바닥 위
#         1.0453·0.9749m인데 리그 다리는 0.8333m라 배율 1.254·1.170배(+0.2120·+0.1416m)이고,
#         기준(전형기 1000564_1 #0)은 1.407·1.505배(+0.3390·+0.4211m)다. 예전에는 이 자리에서
#         늘리지 않고 발을 띄웠는데, 그 그림은 "작업자가 공중에 떠 있다"는, 역시 없는 사실을
#         주장한다. **늘린 길이는 측정이 아니므로 화면이 배율과 늘린 길이를 숫자로 적는다**
#         (`service.LEGS_PLACEHOLDER_STRETCHED_NOTE`) — 자리표시 전체가 애초에 측정이 아니다.
#       `MAX_PLACEHOLDER_STRETCH`를 넘겨야 닿는 바닥은 **고른 면이 틀렸다는 뜻**이라 세우지 않는다
#         (사람 엉덩이는 바닥 위 1.67m에 오지 않는다). 그때만 곧게 내린 채 `reaches_floor=False`다.
# 바닥면을 **고르는 규칙 자체는 그대로다**(`composite.floor_vertical`).
#
# **왜 서 있는 패스에서 발이 뜨는가 — 다리가 아니라 몸통이다.** 이전 라운드가 "리그 다리가 짧은
# 것인지, 3앵커 강체 배치가 골반을 높이 놓는 것인지 가르지 못했다"고 남긴 자리다. 그려지는 CSV
# 리그와 같은 작업자 BVH(전형기 79파일, `body_proportions`의 상수)를 재면 갈린다:
#   골반→머리 CSV 0.4091m / BVH 0.5614m = 0.729
#   골반→발목 CSV 0.9674m / BVH 0.9965m = 0.971
# 다리는 거의 제 길이이고 **몸통이 27% 짧다**. 머리·양손이 실측 마커에 붙어 있으므로 짧은 몸통은
# 골반을 그만큼 **위로** 밀고, 그 몫이 사슬 끝인 발과 바닥 사이 틈으로 나온다 — 실측(김경호 L1 첫
# 패스): 엉덩이가 바닥 위 1.0453m인데 리그 다리는 0.8333m라 왼발이 0.2120m 뜬다. 몸통이 BVH 길이
# (+0.1523m)였다면 골반이 그만큼 내려가 이 틈이 약 0.07m로 줄어든다(몸통 축이 수직에서 기울어 있어
# 수직 몫은 0.142m다). `test_service_view`가 이 숫자들을 실제 배치에서 다시 재서 이 주석과 맞춘다.
#
# 몸통을 늘리는 길은 `body_proportions`가 이미 실측으로 재 봤다: 골반→가슴은 앵커 오차가 그대로라
# 이미 늘렸고, **가슴→머리**를 BVH 비례로 늘리면 앵커 어긋남이 머리 80.6→92.8mm로 나빠진다.
# 남은 27%는 거기에 몰려 있다. 그래서 **몸통은 늘리지 않는다** — 실측을 악화시키기 때문이다.
# 대신 측정이 아닌 쪽(자리표시 다리)을 그 틈만큼 늘려 발을 바닥에 세우고, 화면이 배율과 늘린 길이를
# 숫자로 적는다. 늘린 값과 이 틈은 같은 숫자다(엉덩이 1.0453 − 리그 다리 0.8333 = 0.2120m).
#
# 발끝은 수평이라 어느 갈래에서도 길이를 받지 않는다. **이 다리는 측정이 아니다** — 화면은 점선으로
# 그리고 "자리표시(측정 아님)"라고 적는다.


# 자리표시 다리를 바닥에 세우려고 허용하는 **최대** 배율. 1.0 미만(줄이기)은 어떤 경우에도 없다.
# 사람 엉덩이는 바닥 위 1.67m(= 리그 다리 0.833m × 2.0)에 오지 않으므로, 이 값을 넘겨야 닿는 면은
# 작업자가 선 지면이 아니다 — 그때는 세우지 않고 닿지 않는다고 말한다. 실데이터에서 가장 큰 배율은
# 1.505배(김경호 L1 첫 패스의 기준 전형기 1000564_1 #0)라 이 한계는 실데이터에 걸리지 않는다.
MAX_PLACEHOLDER_STRETCH: Final = 2.0


@dataclass(frozen=True, slots=True)
class PlaceholderLegs:
    """자리표시 다리. 측정이 아니다 — 줄이는 배율은 없고, 늘린 경우에는 얼마나 늘렸는지 싣는다.

    `floor_vertical`은 벽 메시가 고른 바닥면의 월드 수직 높이(메시가 없으면 None)이고,
    `feet_below_floor_m`는 **가장 낮은 발목**이 그 면보다 얼마나 아래인가다(음수면 위).

    `reaches_floor`는 두 다리가 다 그 면에 닿았는가다 — `reach_floor=False`거나 바닥이 없으면
    None(물어본 적이 없다는 뜻이고, False와 구별한다). `hip_above_floor_m`는 **더 높은** 엉덩이가
    그 면 위 얼마인가, `rig_leg_length_m`는 리그 허벅지+종아리 길이다: 닿지 못했을 때 화면이
    "왜 못 닿았는가"를 지어내지 않고 이 두 숫자로 말한다.

    `stretch_factor`는 세우려고 다리에 먹인 **가장 큰** 배율이다 — 1.0이면 길이를 건드리지 않았다
    (무릎만 굽혔다), 1.0보다 작은 값은 나오지 않는다(난쟁이 금지). `stretch_m`은 그때 늘어난 길이
    (m)다. 물어본 적이 없으면(`reach_floor=False`거나 바닥 없음) 둘 다 None이다 — 화면은 이 두
    값으로 "늘려 그렸다"를 말하고, 스스로 배율을 다시 계산하지 않는다.
    """

    joints: dict[str, Vec3]
    floor_vertical: float | None
    feet_below_floor_m: float | None
    reaches_floor: bool | None = None
    hip_above_floor_m: float | None = None
    rig_leg_length_m: float | None = None
    stretch_factor: float | None = None
    stretch_m: float | None = None


def _knee_to_floor(
    hip: np.ndarray,
    gap: float,
    thigh: float,
    calf: float,
    down: np.ndarray,
    heading: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    """엉덩이에서 **바로 아래 바닥**까지 2마디(허벅지·종아리)로 잇는다. 닿을 수 없으면 None.

    무릎은 `heading`(수평, 발끝 방향) 쪽으로 굽는다 — 사람 무릎이 굽는 쪽이다. 뼈 길이는 그대로이고
    바뀌는 것은 무릎 각도뿐이다. 엉덩이가 다리보다 높거나(`gap > thigh + calf`) 두 마디가 접혀도
    닿지 않을 만큼 낮으면(`gap < |thigh - calf|`) 지어내지 않고 None을 돌려준다.
    """
    if not math.isfinite(gap) or gap <= 0.0:
        return None
    if gap > thigh + calf or gap < abs(thigh - calf):
        return None
    ankle = hip + down * gap
    # 엉덩이–발목 선 위에서 무릎까지의 거리(코사인 법칙), 그리고 그 선에서 옆으로 벌어지는 양.
    along = (gap * gap + thigh * thigh - calf * calf) / (2.0 * gap)
    offset_sq = thigh * thigh - along * along
    if offset_sq <= 0.0:
        return None
    knee = hip + down * along + heading * math.sqrt(offset_sq)
    return knee, ankle


def _straight_to_floor(
    hip: np.ndarray,
    gap: float,
    thigh: float,
    calf: float,
    down: np.ndarray,
) -> tuple[tuple[np.ndarray, np.ndarray], float] | None:
    """엉덩이에서 바로 아래 바닥까지 **곧게 선** 다리로 잇는다. 늘려야만 닿을 때만 쓴다.

    허벅지·종아리에 **같은 배율**을 먹이되 1.0 미만은 절대 내주지 않는다 — 줄여서 세우는 일(난쟁이)은
    이 저장소가 한 번 저지른 거짓이고, 짧은 쪽은 무릎이 맡는다(`_knee_to_floor`). `MAX_PLACEHOLDER_STRETCH`를
    넘겨야 닿는 바닥은 작업자가 선 지면이 아니라는 뜻이라 None을 돌려준다. 돌려주는 배율은 화면이
    그대로 적는 숫자다 — 화면이 다시 계산하지 않는다.
    """
    length = thigh + calf
    if not math.isfinite(gap) or length <= 0.0:
        return None
    scale = gap / length
    if scale <= 1.0 or scale > MAX_PLACEHOLDER_STRETCH:
        return None
    knee = hip + down * (thigh * scale)
    ankle = knee + down * (calf * scale)
    return (knee, ankle), scale


def standing_legs(
    joints: Mapping[str, Vec3],
    offsets: Mapping[str, Vec3],
    forward: Sequence[float] | None,
    *,
    floor: float | None = None,
    reach_floor: bool = False,
) -> PlaceholderLegs | None:
    """자리표시 다리(월드). 엉덩이 관절이나 다리 오프셋이 없으면 None.

    `forward`는 발끝 방향 후보(월드)다. 수평 성분이 거의 없으면 발끝 관절을 싣지 않고 무릎도
    굽히지 않는다 — 방향을 지어내지 않는다.

    `reach_floor=False`(기본)면 리그 제 길이로 곧게 내리고, `floor`가 오면 발목이 그 면보다 얼마나
    아래인지만 재서 싣는다(`feet_below_floor_m`). `reach_floor=True`면 발목을 그 면에 **세운다**:
    엉덩이가 리그 다리보다 낮게 있으면 **무릎을 굽히고**(`_knee_to_floor`, 길이 그대로), 높으면
    곧게 선 채로 **늘린다**(`_straight_to_floor`, 배율은 `stretch_factor`로 싣는다). 줄이는 배율은
    어느 갈래에도 없다. `MAX_PLACEHOLDER_STRETCH`를 넘겨야 닿는 바닥에서만 세우기를 포기하고 곧게
    내린 채 `reaches_floor=False`로 말한다.
    """
    down = -np.array(world_axes.UP, dtype=float)
    up = -down
    heading: np.ndarray | None = None
    if forward is not None:
        f = np.array(forward, dtype=float)
        f = f - float(np.dot(f, up)) * up
        if float(np.linalg.norm(f)) > 1e-6:
            heading = f / np.linalg.norm(f)
    chain: list[tuple[str, str, str, str]] = []
    for hip, knee, ankle, toe in PLACEHOLDER_LEG_CHAIN:
        if hip not in joints or any(name not in offsets for name in (knee, ankle, toe)):
            return None
        chain.append((hip, knee, ankle, toe))

    out: dict[str, Vec3] = {}
    ankle_heights: list[float] = []
    hip_heights: list[float] = []
    reached: list[bool] = []
    scales: list[float] = []
    added: list[float] = []
    leg_length: float | None = None
    for hip, knee, ankle, toe in chain:
        h = np.array(joints[hip], dtype=float)
        thigh = math.hypot(*offsets[knee])
        calf = math.hypot(*offsets[ankle])
        leg_length = thigh + calf
        hip_height = float(np.dot(h, up))
        hip_heights.append(hip_height)
        stood = None
        scale = 1.0
        if reach_floor and floor is not None:
            gap = hip_height - floor
            # 짧은 쪽은 무릎이(길이 그대로), 모자란 쪽은 배율이 맡는다 — 줄이는 배율은 없다.
            if heading is not None:
                stood = _knee_to_floor(h, gap, thigh, calf, down, heading)
            if stood is None:
                stretched = _straight_to_floor(h, gap, thigh, calf, down)
                if stretched is not None:
                    stood, scale = stretched
        if stood is None:
            k = h + down * thigh
            a = k + down * calf
        else:
            k, a = stood
        reached.append(stood is not None)
        scales.append(scale)
        added.append((scale - 1.0) * leg_length)
        out[hip] = _as_tuple(h)
        out[knee] = _as_tuple(k)
        out[ankle] = _as_tuple(a)
        ankle_heights.append(float(np.dot(a, up)))
        if heading is not None:
            out[toe] = _as_tuple(a + heading * math.hypot(*offsets[toe]))
    asked = reach_floor and floor is not None
    return PlaceholderLegs(
        joints=out,
        floor_vertical=floor,
        feet_below_floor_m=None if floor is None or not ankle_heights else floor - min(ankle_heights),
        reaches_floor=all(reached) if asked else None,
        hip_above_floor_m=None if floor is None or not hip_heights else max(hip_heights) - floor,
        rig_leg_length_m=leg_length,
        # 물어본 적이 없으면 None, 물어봤으면 실제로 먹인 **가장 큰** 배율(1.0 = 길이 그대로)이다.
        stretch_factor=max(scales) if asked and scales else None,
        stretch_m=max(added) if asked and added else None,
    )


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
