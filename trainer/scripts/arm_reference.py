#!/usr/bin/env python3
"""팔 자세 레퍼런스 — 숙련자의 팔이 어디 있었는지를 버킷별 분포로 굽고, 그것에 대고 잰다.

## 왜 팔만인가 — 두 리그가 어깨 아래에서만 일치한다

채점은 두 좌표계를 한 자로 재는 일이다. 레퍼런스는 CSV 코퍼스의 FK에서 오고 런타임
자세는 SIM이 보낸다. **두 리그의 뼈 길이를 대 보면 어깨 아래만 같다.**

| 뼈 | CSV-FK | SIM | 비 |
|---|---|---|---|
| LeftShoulder–RightShoulder | 0.0567 | 0.0567 | 1.000 |
| Shoulder–UpperArm | 0.1797 | 0.1797 | 1.000 |
| UpperArm–ForeArm | 0.2430 | 0.2430 | 1.000 |
| ForeArm–Hand | 0.2430 | 0.2430 | 1.000 |
| Chest–Shoulder | 0.0284 | 0.2126 | **7.50** |
| Pelvis–Spine | 0.1053 | 0.1984 | **1.88** |
| Chest–Head | 0.1368 | 0.3436 | **2.51** |

몸통·머리를 채점에 넣으면 σ가 자세 오차가 아니라 **리그 차이**를 잰다. 그래서
`Spine`·`Chest`·`Head`를 빼고 팔 6종만 남긴다. 몸통의 움직임은 채점이 아니라
`posture_sequence`의 추종 기울기가 맡는다 — 그쪽은 월드 높이 차이만 쓰므로 리그에 무관하다.

## 몸 프레임 — 길이가 아니라 각도로 세운다

FK 공간은 월드 정렬이 아니다(어깨축의 월드 수직 성분이 중앙 0.77이다). 그래서 월드
수직을 축으로 쓸 수 없고, **프레임을 리그 안에서 세운다.**

- 원점: 어깨 중점
- x̂: `LeftShoulder → RightShoulder` (두 리그가 같은 뼈)
- ŷ: `Chest → 어깨 중점` 방향
- ẑ: x̂ × ŷ

`Chest`는 **방향으로만** 쓴다. 길이는 7.50배 어긋나지만 각도는 아니다 —
어깨축과 `Chest → 어깨 중점`이 이루는 각이 두 리그 모두 **정확히 90.0°**다
(CSV-FK p10/p50/p90 = 90.0/90.0/90.0, SIM 90.0). 길이를 안 쓰므로 7.50배가 들어오지 않는다.

팔 뼈 길이가 두 리그에서 같으므로 **크기 정규화를 하지 않는다.** 거리가 곧 미터다.

## 버킷 — 양쪽에서 **같은 양**이어야 한다

숙련자의 팔 위치는 착탄을 어디에 두느냐에 달렸다. 조건 변수는
**reach = 착탄 수직 − 건 월드 수직**이다. 둘 다 월드라 차이에서 바닥 상수가 지워진다.

**머리가 아니라 건인 이유가 핵심이다.** CSV의 머리는 **카메라(HMD) 강체**이고 런타임의
머리는 **골격 `head` 관절**이다 — 다른 점이다. 그 차이로 버킷이 어긋나는 정도를 쟀더니
0.10 m 오차에서 버킷의 **44%**가 바뀌고(점수 변화 p50 0.000 · p90 0.105), 0.20 m에서는
**100%**가 바뀐다(p90 0.203). 건은 양쪽 모두 같은 물건이다 — CSV는 활성 건 강체
(`measured_pose.right`), SIM은 `tool_pos_rot.pos`.

정보를 잃지 않는다. 레벨1 221패스에서 머리 기준 reach와 건 기준 reach의 상관이
**r = 0.909**다. 건 기준 실측 분포는 p05 −0.336 · p50 −0.115 · p95 +0.084
(min −0.452 · max +0.299)로 머리 기준(−1.03~+0.43)보다 좁아, 밴드 폭도 그만큼 좁힌다.

## 점수 — 반경과 방향을 나눠 잰다

관절 σ = **반경 σ와 방향 σ의 제곱평균**, W = 관절 σ의 가중평균, **S = exp(−W²/2)**.

- 반경 σ = |어깨 중점까지의 거리 − 숙련자 평균| / max(반경 SD, 하한)
- 방향 σ = 평균 방향과의 각 / max(각 산포, `ANGULAR_SD_FLOOR`)

**축 3개로 재면 안 된다.** 손의 변동은 96%가 방향 성분이라 축별 SD가 0.2 m대로 크고,
그 큰 SD가 뻗음을 희석해 15 cm 과신전이 1σ도 안 된다 — 실측에서 손을 45 cm 옮겨도
교정이 안 나갔다. 반경(손 0.492 ± 0.071 m)과 방향은 직교하므로 나눠 재면 둘 다 산다.
반경·방향이 없는 옛 아티팩트에는 축 RMS로 내려간다. 가중치는 `measured_score.DEFAULT_WEIGHTS`를
읽어 온다 — 건을 쥔 오른쪽이 분사를 정하기 때문이고, 여기서 리터럴로 다시 적으면 두 곳이 갈린다.

RMS인 이유는 실측에 있다. 제곱합 제곱근으로 내면 축마다 1σ인 **통상적인** 숙련자도
관절 σ가 √3 = 1.73이 되어 점수가 0.22로 떨어진다. 실제로 숙련자 27패스 p50이 **0.016**이었다.

## 이 점수의 눈금

레벨1 전수 1,217패스를 **leave-one-out**으로 재면 숙련자 점수의 중앙값이 **0.809**
(p10 0.247 · p90 0.919)이고, 작업자별 중앙값이 0.683~0.839에 들어온다. 즉
**통상적인 숙련자가 0.81 근처**이고 0.2 아래는 숙련자 중에서도 드문 자세다.

절대값은 축 수·산포 하한·가중치가 스케일을 건드리므로 **`expert_percentile`이 비교
가능한 쪽**이다. 작업자별 레퍼런스도 재 봤지만 전체 p50이 0.810으로 같고 버킷만
얇아져(1208 → 876패스) 지금은 얻을 것이 없다. 측정은
`docs/measurements/2026-09-29-two-rigs-agree-only-below-the-shoulders.md`에 있다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Final, Mapping, Sequence

if __package__:
    from . import measured_score
else:
    import measured_score

Vec3 = tuple[float, float, float]

# 채점 대상. 어깨는 프레임의 원점·x축을 정의하므로 정규화 후 정보가 없다 — 뺀다.
SCORING_JOINTS_ARM: Final[tuple[str, ...]] = (
    "RightUpperArm",
    "RightForeArm",
    "RightHand",
    "LeftUpperArm",
    "LeftForeArm",
    "LeftHand",
)
# 프레임을 세우는 데 필요한 관절. 하나라도 없으면 정규화하지 않는다.
FRAME_JOINTS: Final[tuple[str, ...]] = ("LeftShoulder", "RightShoulder", "Chest")

SCORING_SET_ID: Final = "v3-arm-6"

# 이보다 적은 관절로 낸 점수는 다른 점수와 비교할 수 없다 — 분모가 관절 수다.
MIN_SCORED_JOINTS: Final = 4

# 프레임 축의 이름. 교정 문구가 이 이름으로 말한다.
AXIS_ORDER: Final[tuple[str, ...]] = ("lateral", "torso", "depth")

# reach 버킷 폭. 건 기준 실측 분포는 p05~p95가 0.42 m라 머리 기준(0.84 m)의 절반이다.
# 같은 해상도를 유지하려면 폭도 절반이어야 한다.
REACH_BAND_M: Final = 0.10

# 산포 하한·상한. `measured_score`와 같은 값을 쓴다(잡음이 작을 때 σ가 무한히 부풀지 않게).
SIGMA_FLOOR_M: Final = measured_score.SIGMA_FLOOR_M
MAX_SIGMA: Final = measured_score.MAX_SIGMA

# 레퍼런스 품질 사다리. 설계 문서 §6의 표를 그대로 옮긴다.
MIN_BUCKET_N: Final = 20
MULTI_WORKER_MIN_WORKERS: Final = 3
MULTI_WORKER_MIN_N: Final = 100

# 교정을 발부할 문턱. 이보다 작은 σ는 "숙련자 통상 범위 안"이다.
CORRECTION_SIGMA: Final = 1.5
MAX_CORRECTIONS: Final = 2


# 경계에서 나눗셈을 반올림할 자릿수. `2.8 / 0.2`가 13.999999999999998이라 그냥 내림하면
# 경계값이 **한 칸 아래 버킷**으로 떨어진다. 굽는 쪽과 재는 쪽이 같은 함수를 쓰므로
# 조용히 어긋나지는 않지만, 경계에 몰린 표본이 이웃 버킷으로 새는 것은 그대로 오류다.
_BAND_EPSILON_DIGITS: Final = 9



# --- 위상 (착탄 경로 DTW 정렬) ----------------------------------------------
#
# 중앙값 한 점으로 접으면 **스트로크 일부 구간에서만** 틀린 자세를 희석한다. 앞 절반에만
# 0.20 m 오차를 넣었을 때 중앙값 Δσ가 0.008인데 위상 채점은 0.077이다(약 10배).
# 측정은 `docs/measurements/2026-09-29-dtw-phase-alignment-prereg.md`.
#
# 위상 수. 분사 구간 실측이 p50 0.96초·115프레임(120FPS)이라 12위상이면 80 ms 간격이다.
# 더 잘게 쪼개면 위상마다 표본이 얇아져 SD가 불안정해진다.
PHASES: Final = 12

# 위상 레퍼런스를 만들 때 밴드마다 들고 있을 시퀀스 수. 전량(밴드당 최대 14,245개)을
# 메모리에 얹을 수 없고, 위상마다 수백 표본이면 SD가 안정적이라 이만큼만 쓴다.
# **점 레퍼런스는 전량으로 굽는다** — 이 상한은 위상 쪽에만 걸린다.
PHASE_SAMPLE_PER_BAND: Final = 400

# DBA 프로토타입을 고를 때 쓰는 시퀀스 수. 무게중심은 표본이 많다고 좋아지지 않고
# 비용만 제곱으로 는다.
PROTOTYPE_SAMPLE: Final = 60

# 위상 하나가 이보다 적은 표본으로 만들어졌으면 그 위상은 채점하지 않는다.
MIN_PHASE_SAMPLES: Final = 20

def reach_band(reach_m: float) -> str:
    """reach를 버킷 라벨로. 경계는 `REACH_BAND_M` 격자다."""
    index = math.floor(round(reach_m / REACH_BAND_M, _BAND_EPSILON_DIGITS))
    return f"{index * REACH_BAND_M:+.2f}"


def _sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _unit(v: Vec3) -> Vec3 | None:
    length = math.sqrt(_dot(v, v))
    return None if length < 1e-9 else (v[0] / length, v[1] / length, v[2] / length)


@dataclass(frozen=True)
class BodyFrame:
    origin: Vec3
    axes: tuple[Vec3, Vec3, Vec3]


def body_frame(joints: Mapping[str, Vec3]) -> BodyFrame | None:
    """어깨 중점을 원점으로 하는 직교 프레임. 세울 수 없으면 `None`이다.

    `Chest`는 **방향으로만** 쓴다 — 두 리그의 `Chest–Shoulder` 길이가 7.50배 어긋나지만
    어깨축과의 각은 양쪽 모두 정확히 90.0°다.
    """
    if any(name not in joints for name in FRAME_JOINTS):
        return None
    left, right, chest = (joints[name] for name in FRAME_JOINTS)
    origin = tuple((left[i] + right[i]) / 2 for i in range(3))
    lateral = _unit(_sub(right, left))
    if lateral is None:
        return None
    raw_torso = _sub(origin, chest)  # type: ignore[arg-type]
    # 이미 직교(실측 90.0°)지만 그래도 직교화한다 — 한 프레임이 어긋나도 축이 비뚤어지지 않게.
    torso = _unit(_sub(raw_torso, tuple(_dot(raw_torso, lateral) * c for c in lateral)))  # type: ignore[arg-type]
    if torso is None:
        return None
    depth = _unit(_cross(lateral, torso))
    if depth is None:
        return None
    return BodyFrame(origin=origin, axes=(lateral, torso, depth))  # type: ignore[arg-type]


def normalize(joints: Mapping[str, Vec3]) -> dict[str, Vec3] | None:
    """팔 관절을 몸 프레임 좌표로. 프레임을 못 세우면 `None`이다.

    크기 정규화를 하지 않는다 — 두 리그의 팔 뼈 길이가 같으므로 값이 그대로 미터다.
    """
    frame = body_frame(joints)
    if frame is None:
        return None
    out: dict[str, Vec3] = {}
    for name in SCORING_JOINTS_ARM:
        point = joints.get(name)
        if point is None:
            continue
        offset = _sub(point, frame.origin)
        out[name] = (
            _dot(offset, frame.axes[0]),
            _dot(offset, frame.axes[1]),
            _dot(offset, frame.axes[2]),
        )
    return out




def normalize_sequence(
    frames: Sequence[Mapping[str, Vec3]]
) -> dict[str, Vec3] | None:
    """시퀀스를 자세 하나로. **프레임마다 정규화한 뒤 중앙값을 낸다.**

    순서가 중요하다. 날것 좌표의 중앙값을 먼저 내고 한 번만 정규화하면, 프레임마다 다른
    몸 방향이 한 점에 섞인다 — FK 공간은 루트 회전이 프레임마다 달라 날것 중앙값에 뜻이 없다.
    **굽는 쪽과 재는 쪽이 이 함수를 같이 쓴다.** 두 경로가 순서를 달리 잡으면 숙련자
    자세를 숙련자 기준에 대고도 점수가 0에 붙는다(실제로 그렇게 만들어 0.008을 받았다).
    """
    collected: dict[str, list[list[float]]] = {}
    for joints in frames:
        normalized = normalize(joints)
        if normalized is None:
            continue
        for name, point in normalized.items():
            slot = collected.setdefault(name, [[], [], []])
            for index in range(3):
                slot[index].append(point[index])
    if not collected:
        return None
    import statistics as _st

    return {
        name: (_st.median(axes[0]), _st.median(axes[1]), _st.median(axes[2]))
        for name, axes in collected.items()
        if all(axes)
    }



def to_world(frame: BodyFrame, body_point: Vec3) -> Vec3:
    """몸 프레임 좌표를 **월드로 되돌린다.** `normalize`의 역이다.

    시뮬레이터는 우리 몸 프레임을 모른다 — 어깨 중점 원점에 어깨축·몸통축으로 세운
    좌표계는 이 모듈 안의 약속이다. 교정을 그 좌표로 내보내면 받는 쪽이 우리 프레임을
    다시 구현해야 하고, 그 순간 두 구현이 갈린다. **월드로 바꿔서 내보낸다.**
    """
    lateral, torso, depth = frame.axes
    return (
        frame.origin[0] + body_point[0] * lateral[0] + body_point[1] * torso[0] + body_point[2] * depth[0],
        frame.origin[1] + body_point[0] * lateral[1] + body_point[1] * torso[1] + body_point[2] * depth[1],
        frame.origin[2] + body_point[0] * lateral[2] + body_point[1] * torso[2] + body_point[2] * depth[2],
    )


def corrected_point(body_point: Vec3, target: "CorrectionTarget") -> Vec3 | None:
    """교정을 몸 프레임 좌표에 적용한다. 적용할 것이 없으면 `None`.

    `radius_m`이면 방향을 두고 길이만, `heading`이면 길이를 두고 방향만 바꾼다.
    """
    length = math.sqrt(sum(value * value for value in body_point))
    if target.radius_m is not None:
        unit = _unit(body_point)
        if unit is None:
            return None
        return (unit[0] * target.radius_m, unit[1] * target.radius_m, unit[2] * target.radius_m)
    if target.heading is not None:
        unit = _unit(target.heading)
        if unit is None:
            return None
        return (unit[0] * length, unit[1] * length, unit[2] * length)
    return None

# --- 굽기 -------------------------------------------------------------------


@dataclass
class AxisAccum:
    n: int = 0
    total: float = 0.0
    total_sq: float = 0.0

    def add(self, value: float) -> None:
        self.n += 1
        self.total += value
        self.total_sq += value * value

    def mean(self) -> float:
        return self.total / self.n if self.n else 0.0

    def sd(self) -> float:
        """표본 표준편차. 표본이 둘 미만이면 0.0이고, 그때는 σ 계산에서 하한이 대신 쓰인다."""
        if self.n < 2:
            return 0.0
        variance = (self.total_sq - self.total * self.total / self.n) / (self.n - 1)
        return math.sqrt(max(variance, 0.0))


@dataclass
class BucketAccum:
    joints: dict[str, list[AxisAccum]] = field(default_factory=dict)
    # 반경(어깨 중점까지의 거리)과 방향(단위벡터)을 따로 쌓는다. 왜인지는 `joint_sigmas` 참고.
    radius: dict[str, AxisAccum] = field(default_factory=dict)
    heading: dict[str, list[AxisAccum]] = field(default_factory=dict)
    workers: set[str] = field(default_factory=set)
    n: int = 0

    def add(self, normalized: Mapping[str, Vec3], worker: str) -> None:
        self.n += 1
        self.workers.add(worker)
        for name, point in normalized.items():
            slot = self.joints.setdefault(name, [AxisAccum(), AxisAccum(), AxisAccum()])
            for index in range(3):
                slot[index].add(point[index])
            length = math.sqrt(sum(value * value for value in point))
            self.radius.setdefault(name, AxisAccum()).add(length)
            unit = _unit(point)
            if unit is not None:
                axes = self.heading.setdefault(name, [AxisAccum(), AxisAccum(), AxisAccum()])
                for index in range(3):
                    axes[index].add(unit[index])

    def payload(self) -> dict[str, object]:
        out: dict[str, object] = {"n": self.n, "workers": len(self.workers)}
        joints: dict[str, object] = {}
        for name, axes in sorted(self.joints.items()):
            entry: dict[str, object] = {
                "mean": [axis.mean() for axis in axes],
                "sd": [axis.sd() for axis in axes],
                "n": axes[0].n,
            }
            radius = self.radius.get(name)
            if radius is not None and radius.n >= 2:
                entry["radius"] = {"mean": radius.mean(), "sd": radius.sd()}
            heading = self.heading.get(name)
            if heading is not None and heading[0].n >= 2:
                mean_unit = _unit(tuple(axis.mean() for axis in heading))  # type: ignore[arg-type]
                if mean_unit is not None:
                    # 평균 단위벡터의 길이 R에서 각 산포를 낸다(원형통계). R이 1에 가까울수록
                    # 방향이 모여 있다. σ_ang = sqrt(-2 ln R)는 작은 산포에서 표준편차와 같아진다.
                    resultant = math.sqrt(
                        sum(axis.mean() ** 2 for axis in heading)
                    )
                    resultant = min(max(resultant, 1e-9), 1.0)
                    entry["heading"] = {
                        "mean": list(mean_unit),
                        "angular_sd": math.sqrt(max(-2.0 * math.log(resultant), 0.0)),
                    }
            joints[name] = entry
        out["joints"] = joints
        return out


def reference_quality(bucket: Mapping[str, object]) -> str:
    """설계 문서 §6의 사다리. **작업자 1인은 `available`로 올리지 않는다** —
    1인 산포는 '숙련자 통상 범위'가 아니라 '그 사람의 습관'이다."""
    n = int(bucket.get("n", 0) or 0)
    workers = int(bucket.get("workers", 0) or 0)
    if n < MIN_BUCKET_N:
        return "unavailable"
    if workers >= MULTI_WORKER_MIN_WORKERS and n >= MULTI_WORKER_MIN_N:
        return "multi_worker"
    if workers <= 1:
        return "single_worker"
    return "coarse_bucket"


# --- 재기 -------------------------------------------------------------------


# 방향 산포의 하한(라디안). 0.10 rad ≈ 5.7°. 측정 잡음이 작은 버킷에서 σ가 부풀지 않게 한다.
ANGULAR_SD_FLOOR: Final = 0.10


@dataclass(frozen=True)
class JointSigma:
    joint: str
    sigma: float
    axis_sigmas: tuple[float, float, float]
    delta_m: tuple[float, float, float]
    # 반경(얼마나 뻗었나)과 방향(어디를 향했나)을 나눈 몫. 레퍼런스에 재료가 없으면 `None`.
    radial_sigma: float | None = None
    angular_sigma: float | None = None
    radius_delta_m: float | None = None
    # 숙련자 평균 방향과 실제로 이룬 각. **회전순서와 무관하다** — 두 위치벡터 사이의
    # 각이라 오일러 규약을 타지 않는다. 크기이지 방향이 아니므로 "어느 쪽으로"를 말하지 않는다.
    angle_from_expert_deg: float | None = None
    # 스트로크의 어느 구간에서 벗어났나(0.0~1.0). 위상 채점에서만 채워진다.
    phase_span: tuple[float, float] | None = None


def joint_sigmas(
    normalized: Mapping[str, Vec3], bucket: Mapping[str, object]
) -> list[JointSigma]:
    """관절별 σ. 레퍼런스에 없는 관절은 조용히 빠진다 — 호출자가 수를 세어 판정한다."""
    reference = bucket.get("joints")
    if not isinstance(reference, Mapping):
        return []
    out: list[JointSigma] = []
    for name in SCORING_JOINTS_ARM:
        point = normalized.get(name)
        entry = reference.get(name)
        if point is None or not isinstance(entry, Mapping):
            continue
        mean = entry.get("mean")
        spread = entry.get("sd")
        if not isinstance(mean, Sequence) or not isinstance(spread, Sequence):
            continue
        axis_sigmas: list[float] = []
        deltas: list[float] = []
        total = 0.0
        for index in range(3):
            delta = point[index] - float(mean[index])
            floor = max(float(spread[index]), SIGMA_FLOOR_M)
            value = min(MAX_SIGMA, abs(delta) / floor)
            axis_sigmas.append(value)
            deltas.append(delta)
            total += value * value
        # 축 3개로 재면 **뻗음이 희석된다.** 손의 산포는 96%가 방향 성분이라
        # 축별 SD가 0.2 m대로 크고, 그 큰 SD로 나누면 15 cm 과신전이 1σ도 안 된다.
        # 실측: 손을 45 cm 옮겨도 교정이 안 나갔다(0.779 → 0.769).
        #
        # 반경과 방향은 **직교한다**. 반경(어깨 중점까지 거리)은 SD 0.071 m로 좁고
        # (손 p50 0.492 m), 방향은 넓다. 나눠서 재면 "얼마나 뻗었나"를 잃지 않는다.
        radial = angular = None
        radius_delta = None
        angle_deg = None
        radius_ref = entry.get("radius")
        heading_ref = entry.get("heading")
        length = math.sqrt(sum(value * value for value in point))
        if isinstance(radius_ref, Mapping):
            floor = max(float(radius_ref.get("sd", 0.0) or 0.0), SIGMA_FLOOR_M)
            radius_delta = length - float(radius_ref.get("mean", 0.0) or 0.0)
            radial = min(MAX_SIGMA, abs(radius_delta) / floor)
        if isinstance(heading_ref, Mapping):
            mean_unit = heading_ref.get("mean")
            unit = _unit(point)
            if unit is not None and isinstance(mean_unit, Sequence) and len(mean_unit) == 3:
                cosine = max(-1.0, min(1.0, sum(unit[i] * float(mean_unit[i]) for i in range(3))))
                floor = max(float(heading_ref.get("angular_sd", 0.0) or 0.0), ANGULAR_SD_FLOOR)
                angle = math.acos(cosine)
                angular = min(MAX_SIGMA, angle / floor)
                angle_deg = math.degrees(angle)

        if radial is not None and angular is not None:
            combined = math.sqrt((radial * radial + angular * angular) / 2.0)
        else:
            # 레퍼런스가 반경·방향을 싣기 전에 구운 것이다. 축 RMS로 내려간다.
            #
            # **축 수로 나눈 제곱평균이다. 제곱합의 제곱근이 아니다.** 제곱합을 쓰면
            # 축마다 딱 1σ인 **통상적인** 숙련자도 관절 σ가 √3 = 1.73이 되고
            # `exp(−W²/2)`가 0.22로 떨어진다 — 정상값이 낮은 점수를 받는다.
            combined = math.sqrt(total / 3.0)

        out.append(
            JointSigma(
                joint=name,
                sigma=min(MAX_SIGMA, combined),
                axis_sigmas=(axis_sigmas[0], axis_sigmas[1], axis_sigmas[2]),
                delta_m=(deltas[0], deltas[1], deltas[2]),
                radial_sigma=radial,
                angular_sigma=angular,
                radius_delta_m=radius_delta,
                angle_from_expert_deg=angle_deg,
            )
        )
    return out


def _weight_of(joint: str) -> float:
    """`measured_score.DEFAULT_WEIGHTS`를 읽어 온다. 여기서 리터럴로 다시 적지 않는다."""
    side = "right" if joint.startswith("Right") else "left"
    return measured_score.DEFAULT_WEIGHTS[side]


def similarity_score(sigmas: Sequence[JointSigma]) -> float | None:
    """S = exp(−W²/2). 잴 관절이 없으면 **`None`이지 `0.0`이 아니다.**

    `0.0`은 '많이 다르다'로 읽히는데, 그것은 우리가 재지 않은 것에 대한 주장이다.
    """
    if not sigmas:
        return None
    total_weight = sum(_weight_of(item.joint) for item in sigmas)
    if total_weight <= 0.0:
        return None
    weighted = sum(_weight_of(item.joint) * item.sigma for item in sigmas) / total_weight
    return math.exp(-(weighted * weighted) / 2.0)


# --- 교정 -------------------------------------------------------------------
#
# **한 번에 한 관절만 짚는다.** 운동학습 문헌이 일관되게 말하는 바다 — 동시에 여러 단서를
# 주면 수행은 잠깐 좋아지고 학습은 나빠진다(Raisbeck & Diekfuss 2017; 안내 가설
# Salmoni 1984 · Winstein & Schmidt 1990). `docs/literature_basis.md` 참고.
MAX_CORRECTIONS: Final = 1

# σ → 심각도. 경계는 정규분포에서 각각 상위 4.6% · 0.27%에 해당한다.
SEVERITY_EDGES: Final[tuple[tuple[float, str], ...]] = ((3.0, "high"), (2.0, "medium"))


def severity_of(sigma: float) -> str:
    for edge, name in SEVERITY_EDGES:
        if sigma >= edge:
            return name
    return "low"


# 한국어 조사. 받침 유무로 고른다 — `을(를)` 같은 미처리 표기는 이 자리에서 끝낸다.
def particle(word: str, with_final: str, without_final: str) -> str:
    """`word` 뒤에 붙일 조사. 한글이 아니면 받침 없는 쪽을 쓴다."""
    if not word:
        return without_final
    last = word[-1]
    if not ("\uac00" <= last <= "\ud7a3"):
        return without_final
    return with_final if (ord(last) - 0xAC00) % 28 else without_final


@dataclass(frozen=True)
class CorrectionTarget:
    """**기계가 적용할 수 있는 명제.** 이것이 있어야 교정 정확도를 잰다.

    `correction_text`는 사람이 읽는 서술이고, 여기는 **replay가 그대로 적용할 값**이다.
    적용한 뒤 다시 채점해 `expected_gain`과 맞는지 보면 교정이 말한 대로 움직이는지
    확인된다(`verify_correction`). 그 검증이 없으면 교정은 반증 불가능한 문장일 뿐이다.

    - `radius_m`: 어깨 중점에서 이 관절까지의 **숙련자 평균 거리**. 방향은 그대로 두고
      길이만 이 값으로 맞추라는 뜻이다.
    - `heading`: 숙련자 평균 **방향**(단위벡터). 길이는 그대로 두고 방향만 맞춘다.

    둘 다 **회전순서와 무관하다** — 위치에서 바로 나오므로 오일러 규약을 타지 않는다.
    """

    joint: str
    radius_m: float | None = None
    heading: Vec3 | None = None


@dataclass(frozen=True)
class Correction:
    joint: str
    label: str
    severity: str
    sigma: float
    distance_m: float
    axis_sigmas: tuple[float, float, float]
    expected_gain: float | None
    # 반경(얼마나 뻗었나)과 방향(어디를 향했나)을 나눈 몫. 어느 쪽이 σ를 만들었는지 보여 준다.
    radial_sigma: float | None = None
    angular_sigma: float | None = None
    radius_delta_m: float | None = None
    angle_from_expert_deg: float | None = None
    phase_span: tuple[float, float] | None = None
    target: CorrectionTarget | None = None

    @property
    def dominant(self) -> str:
        """`radius`인가 `heading`인가. 둘 다 없으면 `unknown`."""
        if self.radial_sigma is None or self.angular_sigma is None:
            return "unknown"
        return "radius" if self.radial_sigma >= self.angular_sigma else "heading"

    def text(self) -> str:
        """사람이 읽을 문장. **무엇이 얼마나 달랐는지**를 말하고 지시하지 않는다.

        `"{label}을(를) 보세요"`는 두 가지로 잘못됐었다 — 정보가 없고, 몸에 주의를 돌리는
        **내적 초점** 문구라 외적 초점 문헌(Chua et al. 2021)과 정반대였다.
        """
        where = ""
        if self.phase_span is not None:
            start, end = self.phase_span
            where = f"스트로크 {start * 100:.0f}~{end * 100:.0f}% 구간에서 "
        subject = f"{self.label}{particle(self.label, '이', '가')}"
        if self.dominant == "radius" and self.radius_delta_m is not None:
            far = "멀리" if self.radius_delta_m > 0 else "가까이"
            return (
                f"{where}{subject} 숙련자 평균보다 "
                f"{abs(self.radius_delta_m) * 100:.0f} cm {far} 있었습니다."
            )
        if self.dominant == "heading" and self.angle_from_expert_deg is not None:
            return (
                f"{where}{subject} 향한 방향이 숙련자 평균과 "
                f"{self.angle_from_expert_deg:.0f}° 달랐습니다."
            )
        return f"{where}{subject} 숙련자 통상 범위를 벗어났습니다(σ {self.sigma:.1f})."


def build_corrections(
    sigmas: Sequence[JointSigma],
    labels: Mapping[str, str],
    *,
    support: int | None = None,
    reference: Mapping[str, object] | None = None,
) -> list[Correction]:
    """σ가 가장 큰 관절부터 `MAX_CORRECTIONS`개. 문턱 아래면 아무것도 내지 않는다.

    **방향도 각도도 싣지 않는다**(`correction_mode="joint_only"`). 회전순서가 미확정이고
    (실물 `rot`이 전 프레임 90° 배수), 방향 단서는 외적 초점을 내적 초점으로 바꾼다.
    관절만 짚는다.

    `expected_gain`은 **그 관절을 숙련자 평균에 맞췄을 때의 점수 상승분**이다.
    관측이 아니라 **모의값**이다 — 실제로 그렇게 움직였을 때 이만큼 오른다는 주장이 아니라,
    이 관절이 지금 점수를 얼마나 깎고 있는지를 같은 자로 잰 것이다.
    """
    _ = support
    ranked = sorted(sigmas, key=lambda item: item.sigma, reverse=True)
    base = similarity_score(sigmas)
    out: list[Correction] = []
    for item in ranked[:MAX_CORRECTIONS]:
        if item.sigma < CORRECTION_SIGMA:
            break
        gain = None
        if base is not None:
            without = [
                JointSigma(other.joint, 0.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
                if other.joint == item.joint
                else other
                for other in sigmas
            ]
            fixed = similarity_score(without)
            gain = None if fixed is None else fixed - base
        correction = Correction(
            joint=item.joint,
            label=labels.get(item.joint, item.joint),
            severity=severity_of(item.sigma),
            sigma=item.sigma,
            distance_m=math.sqrt(sum(value * value for value in item.delta_m)),
            axis_sigmas=item.axis_sigmas,
            expected_gain=gain,
            radial_sigma=item.radial_sigma,
            angular_sigma=item.angular_sigma,
            radius_delta_m=item.radius_delta_m,
            angle_from_expert_deg=item.angle_from_expert_deg,
            phase_span=item.phase_span,
        )
        out.append(replace(correction, target=_target_for(correction, reference)))
    return out


def _target_for(
    correction: Correction, reference: Mapping[str, object] | None
) -> CorrectionTarget | None:
    """교정을 **적용 가능한 값**으로 바꾼다. 재료가 없으면 `None`이다.

    σ를 만든 쪽만 건드린다 — 반경이 주범이면 길이만, 방향이 주범이면 방향만. 둘 다
    한꺼번에 맞추라고 하면 "숙련자 평균 자세를 그대로 취하라"가 되어 교정이 아니라 복제다.
    """
    if not isinstance(reference, Mapping):
        return None
    entry = reference.get(correction.joint)
    if not isinstance(entry, Mapping):
        return None
    if correction.dominant == "radius":
        radius = entry.get("radius")
        if isinstance(radius, Mapping) and radius.get("mean") is not None:
            return CorrectionTarget(correction.joint, radius_m=float(radius["mean"]))
        return None
    heading = entry.get("heading")
    if isinstance(heading, Mapping):
        mean = heading.get("mean")
        if isinstance(mean, Sequence) and len(mean) == 3:
            return CorrectionTarget(
                correction.joint,
                heading=(float(mean[0]), float(mean[1]), float(mean[2])),
            )
    return None


# --- 보정: 이 점수가 숙련자 중 어디쯤인가 --------------------------------------
#
# `exp(−W²/2)`의 **절대값은 해석하기 어렵다.** 축 수·산포 하한·가중치가 모두 스케일을
# 건드리기 때문이다. 그래서 굽는 쪽이 **숙련자 자신의 점수 분포**를 같이 싣고, 재는 쪽이
# 그 사다리에 대고 백분위를 낸다. 백분위는 스케일에 무관하다.
PERCENTILE_POINTS: Final[tuple[int, ...]] = (10, 25, 50, 75, 90)


def percentile_ladder(scores: Sequence[float]) -> dict[str, float]:
    """숙련자 점수의 사다리. 표본이 모자라면 빈 사전이다 — 없는 사다리를 지어내지 않는다."""
    ordered = sorted(scores)
    if len(ordered) < MIN_BUCKET_N:
        return {}
    ladder: dict[str, float] = {}
    for point in PERCENTILE_POINTS:
        index = min(len(ordered) - 1, int(round(point / 100.0 * (len(ordered) - 1))))
        ladder[f"p{point}"] = ordered[index]
    return ladder


def expert_percentile(
    bucket: Mapping[str, object], score: float, *, mode: str = "fold"
) -> int | None:
    """이 점수가 숙련자 분포에서 어디쯤인지. 사다리가 없거나 버킷이 바뀌었으면 `None`.

    사다리는 **잰 당시의 버킷에만** 유효하다. 평균·산포만 다시 굽고 사다리를 그대로 두면
    틀린 백분위가 조용히 나간다 — `reference_pass_count`로 대조한다.

    **`mode`가 사다리를 고른다.** 위상 점수와 접기 점수는 다른 분포다(실측 p50 0.704 vs
    0.763). 접기용 사다리에 위상 점수를 대면 전원이 낮게 나온다.
    """
    key = "expert_scores_phase" if mode == "phase" else "expert_scores"
    calibration = bucket.get(key)
    if not isinstance(calibration, Mapping):
        return None
    expected = bucket.get("n") if key == "expert_scores" else calibration.get("reference_pass_count")
    if calibration.get("reference_pass_count") != expected:
        return None
    reached: int | None = None
    for point in PERCENTILE_POINTS:
        value = calibration.get(f"p{point}")
        if isinstance(value, (int, float)) and score >= float(value):
            reached = point
    return 0 if reached is None else reached


# --- 위상별 레퍼런스 ---------------------------------------------------------


@dataclass
class PhaseAccum:
    """위상 하나의 관절 분포. 점 레퍼런스의 `BucketAccum`과 같은 재료를 위상마다 쌓는다."""

    joints: dict[str, BucketAccum] = field(default_factory=dict)

    def add(self, normalized: Mapping[str, Vec3], worker: str) -> None:
        slot = self.joints.setdefault("_", BucketAccum())
        slot.add(normalized, worker)

    def payload(self) -> dict[str, object]:
        slot = self.joints.get("_")
        if slot is None:
            return {"n": 0, "joints": {}}
        body = slot.payload()
        return {"n": body["n"], "joints": body["joints"]}


def phase_sigmas(
    phase_poses: Sequence[Mapping[str, Vec3] | None],
    phase_reference: Sequence[Mapping[str, object] | None],
    excluded: Sequence[str] = (),
) -> list[JointSigma]:
    """위상마다 σ를 내고 관절별로 **위상 RMS**를 낸다.

    축 집계와 같은 이유로 RMS다. 산술평균을 쓰면 **스트로크 일부에서만 틀린 자세가
    맑은 위상 수만큼 희석된다** — 12위상 중 5위상이 σ 1.0이면 평균은 0.42지만 RMS는
    0.64다. 어느 위상에서 틀렸든 그 관절이 틀린 것이다.

    위상 수를 분모로 쓰지 않고 **대응이 있는 위상만** 세는 이유도 같다 — 대응이 빠진
    위상을 0으로 세면 그런 시퀀스가 유리해진다.

    대응이 없는 위상(`None`)과 표본이 얕은 위상은 **건너뛴다** — 0으로 채우면
    "그 구간은 완벽했다"가 된다.
    """
    drop = set(excluded)
    collected: dict[str, list[JointSigma]] = {}
    phase_of: dict[str, list[int]] = {}
    total_phases = len(phase_poses)
    for index, (pose, entry) in enumerate(zip(phase_poses, phase_reference)):
        if pose is None or not isinstance(entry, Mapping):
            continue
        if int(entry.get("n", 0) or 0) < MIN_PHASE_SAMPLES:
            continue
        for item in joint_sigmas(pose, entry):
            if item.joint in drop:
                continue
            collected.setdefault(item.joint, []).append(item)
            phase_of.setdefault(item.joint, []).append(index)
    out: list[JointSigma] = []
    for name in SCORING_JOINTS_ARM:
        items = collected.get(name)
        if not items:
            continue
        count = len(items)
        radial = [i.radial_sigma for i in items if i.radial_sigma is not None]
        angular = [i.angular_sigma for i in items if i.angular_sigma is not None]
        deltas = [i.radius_delta_m for i in items if i.radius_delta_m is not None]
        worst = max(items, key=lambda i: i.sigma)
        # 문턱을 넘은 위상들이 스트로크의 어느 구간인가. 하나도 없으면 최악 위상 하나를 쓴다.
        indices = phase_of.get(name, [])
        flagged = [
            indices[k] for k, item in enumerate(items) if item.sigma >= CORRECTION_SIGMA
        ] or [indices[items.index(worst)]] if indices else []
        span = None
        if flagged and total_phases > 1:
            span = (
                min(flagged) / total_phases,
                (max(flagged) + 1) / total_phases,
            )
        out.append(
            JointSigma(
                joint=name,
                sigma=min(MAX_SIGMA, math.sqrt(sum(i.sigma ** 2 for i in items) / count)),
                axis_sigmas=tuple(  # type: ignore[arg-type]
                    sum(i.axis_sigmas[axis] for i in items) / count for axis in range(3)
                ),
                # 거리는 **가장 벗어난 위상**의 것을 싣는다. 평균하면 "언제 얼마나
                # 벗어났나"가 흐려지고, 교정 문구가 가리키는 크기가 실제보다 작아진다.
                delta_m=worst.delta_m,
                radial_sigma=sum(radial) / len(radial) if radial else None,
                angular_sigma=sum(angular) / len(angular) if angular else None,
                radius_delta_m=(
                    max(deltas, key=abs) if deltas else None
                ),
                angle_from_expert_deg=worst.angle_from_expert_deg,
                phase_span=span,
            )
        )
    return out
