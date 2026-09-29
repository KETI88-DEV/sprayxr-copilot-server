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

## 버킷 — 바닥을 모르고도 조건을 건다

숙련자의 팔 위치는 착탄을 어디에 두느냐에 달렸다. 조건 변수는
**reach = 착탄 수직 − 머리 월드 수직**이다. 둘 다 월드라 차이에서 바닥 상수가 지워지고,
런타임에도 `raypoint`와 `head`만으로 바로 난다. 실측 분포는 p10 −0.74 · p50 −0.25 · p90 +0.10 m다.

## 점수

축별 σ = |값 − 평균| / max(숙련자 축 표준편차, 하한), 관절 σ = **축 σ의 제곱평균(RMS)**,
W = 관절 σ의 가중평균, **S = exp(−W²/2)**. 가중치는 `measured_score.DEFAULT_WEIGHTS`를
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
from dataclasses import dataclass, field
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

# reach 버킷 폭. 실측 분포 폭(p10~p90 = 0.84 m)을 4칸 남짓으로 가른다.
REACH_BAND_M: Final = 0.20

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
    workers: set[str] = field(default_factory=set)
    n: int = 0

    def add(self, normalized: Mapping[str, Vec3], worker: str) -> None:
        self.n += 1
        self.workers.add(worker)
        for name, point in normalized.items():
            slot = self.joints.setdefault(name, [AxisAccum(), AxisAccum(), AxisAccum()])
            for index in range(3):
                slot[index].add(point[index])

    def payload(self) -> dict[str, object]:
        return {
            "n": self.n,
            "workers": len(self.workers),
            "joints": {
                name: {
                    "mean": [axis.mean() for axis in axes],
                    "sd": [axis.sd() for axis in axes],
                    "n": axes[0].n,
                }
                for name, axes in sorted(self.joints.items())
            },
        }


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


@dataclass(frozen=True)
class JointSigma:
    joint: str
    sigma: float
    axis_sigmas: tuple[float, float, float]
    delta_m: tuple[float, float, float]


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
        out.append(
            JointSigma(
                joint=name,
                # **축 수로 나눈 제곱평균(RMS)이다. 제곱합의 제곱근이 아니다.**
                # 제곱합을 쓰면 축마다 딱 1σ인 **통상적인** 숙련자도 관절 σ가 √3 = 1.73이
                # 되고 `exp(−W²/2)`가 0.22로 떨어진다 — 정상값이 낮은 점수를 받는다.
                # RMS는 "축마다 평균 몇 σ인가"라 통상적인 숙련자가 1.0 근처에 온다.
                sigma=min(MAX_SIGMA, math.sqrt(total / 3.0)),
                axis_sigmas=(axis_sigmas[0], axis_sigmas[1], axis_sigmas[2]),
                delta_m=(deltas[0], deltas[1], deltas[2]),
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


@dataclass(frozen=True)
class Correction:
    joint: str
    label: str
    severity: str
    sigma: float
    distance_m: float
    axis_sigmas: tuple[float, float, float]
    expected_gain: float | None


def build_corrections(
    sigmas: Sequence[JointSigma],
    labels: Mapping[str, str],
    *,
    support: int | None = None,
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
        out.append(
            Correction(
                joint=item.joint,
                label=labels.get(item.joint, item.joint),
                severity=severity_of(item.sigma),
                sigma=item.sigma,
                distance_m=math.sqrt(sum(value * value for value in item.delta_m)),
                axis_sigmas=item.axis_sigmas,
                expected_gain=gain,
            )
        )
    return out


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


def expert_percentile(bucket: Mapping[str, object], score: float) -> int | None:
    """이 점수가 숙련자 분포에서 어디쯤인지. 사다리가 없거나 버킷이 바뀌었으면 `None`.

    사다리는 **잰 당시의 버킷에만** 유효하다. 평균·산포만 다시 굽고 사다리를 그대로 두면
    틀린 백분위가 조용히 나간다 — `reference_pass_count`로 대조한다.
    """
    calibration = bucket.get("expert_scores")
    if not isinstance(calibration, Mapping):
        return None
    if calibration.get("reference_pass_count") != bucket.get("n"):
        return None
    reached: int | None = None
    for point in PERCENTILE_POINTS:
        value = calibration.get(f"p{point}")
        if isinstance(value, (int, float)) and score >= float(value):
            reached = point
    return 0 if reached is None else reached
