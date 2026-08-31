#!/usr/bin/env python3
"""실측 3점 축의 유사도와 교정.

근거가 측정값이라는 것이 이 축의 전부다. 관절 축은 IK 산출물을 채점하지만 여기서는
머리·양손의 실제 위치를 쓴다. 그래서 교정문이 "오른손을 5cm 띄우세요"가 되고 작업자가
그대로 따를 수 있다.

가중은 오른손 > 머리 > 왼손이다. 오른손이 건이고 도막 품질을 직접 결정한다. 왼손은
플래시라 작업 결과와의 인과가 약하다.

각도를 내지 않는다. REQ-19 correction_target_gate 통과 전까지 각도 지시는 근거보다
높은 정밀도를 표시한다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Final, Mapping, Protocol, Sequence

Vec3 = tuple[float, float, float]
# 왼손은 프레임 단위로 빠질 수 있다(설계 §5.4). 자리는 남기고 값만 None이다.
Track = Sequence[Vec3 | None]

DEFAULT_WEIGHTS: Final[dict[str, float]] = {"right": 1.0, "head": 0.4, "left": 0.15}
POINT_LABELS: Final[dict[str, str]] = {"right": "오른손", "head": "머리", "left": "왼손"}
POINT_ORDER: Final[tuple[str, ...]] = ("right", "head", "left")

# `_correction_candidates`가 내는 세 축과 이름을 맞춘다. 산포 축과 교정 축이 다르면
# σ가 교정 축을 고를 수 없다 — 산포는 재는 축, 교정은 말하는 축인데 그 둘이 같은
# 좌표계를 가리켜야 σ 하나로 두 일을 다 할 수 있다.
AXIS_ORDER: Final[tuple[str, ...]] = ("wall", "vertical", "lateral")
# 점 → 축 → 산포(m). 실측 산포가 없을 때는 `uniform_spreads`로 스칼라 하나를 편다.
AxisSpreads = Mapping[str, Mapping[str, float]]


def uniform_spreads(value: float) -> dict[str, dict[str, float]]:
    """스칼라 하나를 점·축 전부에 편다. 실측 산포가 없을 때의 잠정값이다.

    축별 산포가 모두 같으면 새 계산이 예전 스칼라 채점과 같은 σ를 낸다(직교 기저
    변환은 벡터 노름을 보존하므로) — 그래서 이 함수가 하위호환의 다리다.
    """
    return {point: {axis: value for axis in AXIS_ORDER} for point in POINT_ORDER}


# 산포 하한. 측정 잡음이 작을 때 σ가 무한히 부풀지 않게 한다.
SIGMA_FLOOR_M: Final = 0.02
CORRECTION_SIGMA_THRESHOLD: Final = 2.0
MAX_SIGMA: Final = 6.0
MIN_CORRECTION_CM: Final = 1.0
# 수평면에서 이보다 짧은 축 성분은 방향이 정해지지 않은 것으로 본다.
MIN_AXIS_LENGTH: Final = 1e-9
# 월드 수직. 설계 §2.5 — 착탄점 Z가 −0.149~1.456 m로 수직축이다.
WORLD_UP: Final[Vec3] = (0.0, 0.0, 1.0)


class AxisFrame(Protocol):
    """스트로크 기저의 세 축. `stroke_frame.Basis`가 이 모양이다.

    읽기 전용으로 선언한다. 구현이 frozen 데이터클래스라 쓰기 가능한 속성으로 두면
    구조적 부합이 깨진다.
    """

    @property
    def normal(self) -> Vec3: ...

    @property
    def tangent(self) -> Vec3: ...

    @property
    def up(self) -> Vec3: ...


@dataclass(frozen=True, slots=True)
class _WorldFrame:
    normal: Vec3 = (1.0, 0.0, 0.0)
    tangent: Vec3 = (0.0, 1.0, 0.0)
    up: Vec3 = (0.0, 0.0, 1.0)


# 기저를 주지 않은 호출자에게 쓰는 기본값. 로컬 성분을 월드 축으로 그대로 읽는다.
WORLD_FRAME: Final = _WorldFrame()


@dataclass(frozen=True, slots=True)
class PointDelta:
    """한 점의 이탈. local_delta는 질의에서 레퍼런스로 가는 스트로크 프레임 벡터다.

    samples는 이 점을 실제로 받친 격자점 수다. 왼손은 프레임 단위로 빠지므로 다른
    점보다 적은 표본 위에 서 있을 수 있고, 그 사실이 화면에 보여야 한다.
    """

    point: str
    sigma: float
    local_delta: Vec3
    samples: int = 0
    # 축별 σ. 비어 있으면(구조체를 직접 만든 옛 호출자) `_largest_correction`이
    # 예전처럼 절댓값 크기로 후보를 고른다.
    axis_sigmas: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class MeasuredResult:
    score: float
    deltas: list[PointDelta]
    samples: int


@dataclass(frozen=True, slots=True)
class RightHandFinding:
    """오른손 하나의 가장 큰 이탈. VLM이 문장을 지을 근거다.

    `corrections_text`가 내는 문장은 이 구조체에서 파생된다 — 축·방향이 문장과 어긋날
    수 없다. `samples`가 핵심이다. 표본 3개에서 나온 이탈과 24개에서 나온 이탈은 같은
    주장이 아니고, 이 구조체를 소비하는 쪽이 그 차이를 볼 수 있어야 한다.
    """

    axis: str  # "wall" | "vertical" | "lateral"
    magnitude_cm: float  # 부호 있음. 방향은 direction에 말로도 있다
    direction: str  # "띄우"/"붙이"/"올리"/"내리"/"앞으로"/"뒤로" 또는 "" (일반 좌우 이동)
    sigma: float
    samples: int


@dataclass(frozen=True, slots=True)
class AxisCorrection:
    """발부한 축 교정 하나. **이 목록에 있는 것만 시뮬레이션이 옮긴다.**

    `magnitude_cm`과 `unit`이 같은 곳(`_axis_components`)에서 나오므로 문장이 말한
    크기와 시뮬레이션이 옮기는 양이 갈라질 수 없다. `sigma`는 그 축 자신의 값이지
    점의 결합 sigma가 아니다 — 읽는 쪽이 어느 축이 얼마나 이상한지 봐야 한다.
    """

    point: str  # "right" | "head". "left"는 절대 들어가지 않는다.
    axis: str  # "wall" | "vertical" | "lateral"
    magnitude_cm: float  # 부호 있음
    direction: str  # 동사/방향어. 판정할 수 없으면 빈 문자열
    sigma: float  # 그 축의 sigma
    unit: Vec3  # 월드 단위벡터


def _mean(points: Sequence[Vec3]) -> Vec3:
    count = len(points)
    return (
        sum(point[0] for point in points) / count,
        sum(point[1] for point in points) / count,
        sum(point[2] for point in points) / count,
    )


def score_track(
    query: Mapping[str, Track],
    reference: Mapping[str, Track],
    spreads: AxisSpreads,
    weights: Mapping[str, float],
    frame: AxisFrame = WORLD_FRAME,
) -> MeasuredResult:
    """정렬 격자 위 세 점의 가중 유사도. 못 잰 점은 0점이 아니라 제외다.

    격자점 하나가 비어도(왼손 마스킹) 그 점만 빼고 나머지 격자점으로 잰다. 자리를
    없애지 않고 None으로 두는 이유는 두 패스의 같은 s가 계속 마주 보게 하기 위함이다.

    σ는 이제 점당 하나가 아니라 축(`AXIS_ORDER`)마다 하나다. 축별 산포로 나눈 뒤
    벡터 노름으로 합친다 — 직교 기저 변환은 노름을 보존하므로 `uniform_spreads`로
    편 산포에서는 예전 스칼라 σ와 정확히 같은 값이 나온다(하위호환).

    `frame`은 로컬 성분을 wall/vertical/lateral로 나누는 기저다. 기본값
    `WORLD_FRAME`은 기존 호출자(성분을 월드 좌표로 직접 주는 단위 테스트 등)가
    그대로 동작하게 한다.
    """
    deltas: list[PointDelta] = []
    samples = 0
    for name in POINT_ORDER:
        left = query.get(name) or []
        right = reference.get(name) or []
        pairs = [
            (q, r) for q, r in zip(left, right) if q is not None and r is not None
        ]
        if not pairs:
            continue
        samples = max(samples, len(pairs))
        mean_delta = _mean([(r[0] - q[0], r[1] - q[1], r[2] - q[2]) for q, r in pairs])
        axis_spread = spreads.get(name) or {}
        axis_magnitudes = _axis_magnitudes_cm(mean_delta, frame)
        axis_sigmas: dict[str, float] = {}
        total = 0.0
        for axis in AXIS_ORDER:
            floor = max(float(axis_spread.get(axis, SIGMA_FLOOR_M)), SIGMA_FLOOR_M)
            value = min(MAX_SIGMA, abs(axis_magnitudes[axis]) / 100.0 / floor)
            axis_sigmas[axis] = value
            total += value * value
        combined = min(MAX_SIGMA, math.sqrt(total))
        deltas.append(
            PointDelta(
                point=name,
                sigma=combined,
                local_delta=mean_delta,
                samples=len(pairs),
                axis_sigmas=axis_sigmas,
            )
        )

    if not deltas:
        return MeasuredResult(score=0.0, deltas=[], samples=0)

    total_weight = sum(weights.get(delta.point, 0.0) for delta in deltas)
    if total_weight <= 0.0:
        return MeasuredResult(score=0.0, deltas=deltas, samples=samples)
    weighted = (
        sum(weights.get(delta.point, 0.0) * delta.sigma for delta in deltas) / total_weight
    )
    return MeasuredResult(
        score=math.exp(-0.5 * weighted * weighted), deltas=deltas, samples=samples
    )


def _object_particle(label: str) -> str:
    """목적격 조사. 받침이 있으면 을, 없으면 를이다.

    "머리을"은 비문이다. 라벨이 늘 때마다 사람이 기억해야 하는 규칙이 아니라 마지막
    글자에서 뽑는다. 한글 음절은 0xAC00부터 종성 28개 주기로 배열돼 있어 나머지가
    0이면 받침이 없다.
    """
    last = label[-1] if label else ""
    if not ("가" <= last <= "힣"):
        return "을"
    return "을" if (ord(last) - 0xAC00) % 28 else "를"


def _dot(left: Vec3, right: Vec3) -> float:
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2]


def _flattened(vector: Vec3) -> Vec3 | None:
    """수평 성분의 단위벡터. 축이 수직이면 수평 방향이 없으므로 None이다."""
    length = math.hypot(vector[0], vector[1])
    if length < MIN_AXIS_LENGTH:
        return None
    return (vector[0] / length, vector[1] / length, 0.0)


def _world_delta(local: Vec3, frame: AxisFrame) -> Vec3:
    """로컬 성분을 월드로 되돌린다. 축 벡터가 성분의 뜻이다."""
    return (
        local[0] * frame.normal[0] + local[1] * frame.tangent[0] + local[2] * frame.up[0],
        local[0] * frame.normal[1] + local[1] * frame.tangent[1] + local[2] * frame.up[1],
        local[0] * frame.normal[2] + local[1] * frame.tangent[2] + local[2] * frame.up[2],
    )


def _axis_components(local: Vec3, frame: AxisFrame) -> dict[str, tuple[float, Vec3 | None, str]]:
    """세 축(wall/vertical/lateral) 기하를 **한 곳에서만** 계산한다.

    `_axis_magnitudes_cm`(σ가 쓴다)과 `_correction_candidates`(문장·`apply_measured`가
    쓴다)가 예전에는 이 기하를 각자 다시 계산했다 — 그래서 한쪽만 고치면 산포가 재는
    축과 교정이 말하는 축이 조용히 갈라질 수 있었다(Task 1이 고친 결함). 이제 이
    함수 하나가 축마다 `(magnitude_cm, unit, word)`를 내고, 두 공개 헬퍼는 포장만 한다.

    - `magnitude_cm`: 임계 없는 크기(부호 있음, wall/vertical). lateral 진행축이 없을
      때는 늘 ≥ 0(수평 성분의 노름이므로).
    - `unit`: 그 축의 월드 단위벡터, 또는 "이 축은 후보가 될 수 없다"를 뜻하는 `None`.
      법선에 수평 성분이 없으면 wall이 `None`이다. 진행축에 수평 성분이 없고 남은
      수평 잔차도 없으면(원점) lateral이 `None`이다. vertical은 항상 `WORLD_UP`이다.
    - `word`: 그 축·부호에 대응하는 동사/방향어. 후보에 실제로 들어갈지와 무관하게
      부호만으로 정해진다 — 임계는 호출부(`_correction_candidates`)가 따로 건다.

    로컬 성분을 축 이름으로 그대로 읽으면 안 된다. `Basis.up`은 순수 수직 스트로크에서
    수평이 되므로(그때 수직은 `tangent`다) `local[2]`를 "올리세요"로 적으면 옆으로
    틀어진 손을 위로 올리라고 말하게 된다. 그래서 월드로 되돌린 뒤 세 방향으로 다시
    나눈다.

    - 수직(`"vertical"`): 월드 Z 성분. 어느 축에 실려 있었든 여기서만 올리/내리가 나온다.
    - 벽(`"wall"`): 법선은 항상 수평이므로(`Basis` 독스트링) 수평 성분을 법선에 투영한다.
    - 진행축(`"lateral"`): 법선에 직교하는 수평 방향이다. wall 성분을 뺀 뒤의 평면으로
      잰다(뺄 것이 없으면, 즉 wall이 `None`이면 빼지 않는다) — 이 순서가 바뀌면 값이
      달라진다. 스트로크가 수평으로 진행하면 그것이 진행 방향이라 앞뒤로 적는다.
      수직 스트로크에는 수평 진행이 없어 좌우를 판정할 기준이 없다 — 작업자 요가
      실측이 아니기 때문이다(`service.ASSUMED_YAW`). 그때는 방향을 지어내지 않고
      크기만 부호 없이 내고(`hypot`), 남은 수평 잔차 방향을 단위벡터로 쓴다
      (`word`는 빈 문자열).
    """
    world = _world_delta(local, frame)
    plane: Vec3 = (world[0], world[1], 0.0)
    up_cm = world[2] * 100.0

    wall = _flattened(frame.normal)
    normal_cm = 0.0 if wall is None else _dot(plane, wall) * 100.0
    if wall is not None:
        plane = (
            plane[0] - normal_cm / 100.0 * wall[0],
            plane[1] - normal_cm / 100.0 * wall[1],
            0.0,
        )
    wall_word = "" if wall is None else ("띄우" if normal_cm > 0 else "붙이")

    vertical_word = "올리" if up_cm > 0 else "내리"

    along = _flattened(frame.tangent)
    if along is None:
        # 수평 진행이 없다. 남은 수평 성분 자체가 방향이라 크기는 늘 양수다.
        lateral_unit = _flattened(plane)
        lateral_cm = math.hypot(plane[0], plane[1]) * 100.0
        lateral_word = ""
    else:
        lateral_unit = along
        lateral_cm = _dot(plane, along) * 100.0
        lateral_word = "앞으로" if lateral_cm > 0 else "뒤로"

    return {
        "wall": (normal_cm, wall, wall_word),
        "vertical": (up_cm, WORLD_UP, vertical_word),
        "lateral": (lateral_cm, lateral_unit, lateral_word),
    }


def _axis_magnitudes_cm(local: Vec3, frame: AxisFrame) -> dict[str, float]:
    """세 축(wall/vertical/lateral) 편차를 cm로, 임계 없이 전부 낸다.

    `_correction_candidates`는 `_axis_components`가 낸 같은 값에서 `MIN_CORRECTION_CM`
    미만과 후보 불가(`unit is None`) 축을 걸러낸다 — 그건 "문장으로 낼 만큼 크다"는
    질문이다. σ는 다른 질문이다("이 축의 산포에 비춰 이상한가"). 산포가 좁으면 1cm
    아래 편차도 σ가 클 수 있으므로 임계로 값을 지우면 σ 계산 자체가 틀린다. 그래서
    필터 없는 버전을 따로 둔다.
    """
    return {axis: value for axis, (value, _unit, _word) in _axis_components(local, frame).items()}


def _correction_candidates(
    local: Vec3, frame: AxisFrame
) -> list[tuple[str, float, str, Vec3]]:
    """한 점의 이탈을 물리적으로 참인 후보 세 개(벽·수직·진행축)로 나눈다.

    기하는 `_axis_components`가 한 곳에서 계산한다. 여기서는 그 결과에서
    `MIN_CORRECTION_CM` 아래인 축과 후보가 될 수 없는 축(`unit is None`)을 걸러낸다.
    호출자가 절댓값이 가장 큰 후보 하나만 골라 문장 하나를 낸다 — 여러 후보를 동시에
    지시하지 않는다.

    후보마다 **월드 단위벡터**를 함께 낸다. 그래야 교정을 따랐을 때의 궤적을 만드는
    `apply_measured`가 문장이 고른 축 하나를 정확히 그만큼만 옮길 수 있다. 크기와
    방향이 같은 곳에서 나오므로 문장과 시뮬레이션이 어긋날 수 없다 —
    `magnitude_cm/100 × unit`이 곧 지시한 월드 이동이다.
    """
    components = _axis_components(local, frame)
    candidates: list[tuple[str, float, str, Vec3]] = []
    for axis in AXIS_ORDER:
        value, unit, word = components[axis]
        if unit is None or abs(value) < MIN_CORRECTION_CM:
            continue
        candidates.append((axis, value, word, unit))
    return candidates


def _largest_correction(
    local: Vec3, frame: AxisFrame, axis_sigmas: Mapping[str, float] | None = None
) -> tuple[str, float, str, Vec3] | None:
    """가장 이상한 후보 하나. 가장 큰 것과 가장 이상한 것은 다르다.

    벽 거리가 3cm로 일정하고 진행 방향이 20cm로 느슨한 사람에게 진행 방향 17cm는
    정상이고 벽 5cm가 이상한 것이다. 축별 산포(σ)가 있으면 σ가 가장 큰 후보를,
    없으면(구조체를 직접 만든 옛 호출자) 예전처럼 절댓값이 가장 큰 후보를 고른다.
    """
    candidates = _correction_candidates(local, frame)
    if not candidates:
        return None
    if axis_sigmas:
        return max(candidates, key=lambda candidate: axis_sigmas.get(candidate[0], 0.0))
    return max(candidates, key=lambda candidate: abs(candidate[1]))


def right_axes(
    delta: PointDelta, frame: AxisFrame = WORLD_FRAME
) -> list[AxisCorrection]:
    """오른손에서 발부할 축 전부. 점 게이트는 그대로, 축은 자기 sigma로 판정한다.

    문장 축(sigma가 가장 큰 축)은 자기 축 sigma와 무관하게 **항상** 포함한다. 이것이
    단조 보장이다 — 결합 sigma는 축별 sigma의 제곱합의 제곱근이라 "합쳐서는 임계를
    넘지만 개별 축은 아무도 못 넘는" 경우가 실제로 있고, 그때 축별 게이트만 걸면
    오늘 나가던 교정이 조용히 사라진다.

    `axis_sigmas`가 비어 있으면(구조체를 직접 만든 옛 호출자) 축별 판정을 할 수
    없으므로 예전처럼 문장 축 하나만 낸다.
    """
    if delta.sigma <= CORRECTION_SIGMA_THRESHOLD:
        return []
    chosen = _largest_correction(delta.local_delta, frame, delta.axis_sigmas)
    if chosen is None:
        return []
    axis_sigmas = delta.axis_sigmas
    picked: dict[str, tuple[str, float, str, Vec3]] = {chosen[0]: chosen}
    if axis_sigmas:
        for axis, magnitude_cm, direction, unit in _correction_candidates(
            delta.local_delta, frame
        ):
            if axis_sigmas.get(axis, 0.0) > CORRECTION_SIGMA_THRESHOLD:
                picked[axis] = (axis, magnitude_cm, direction, unit)
    return [
        AxisCorrection(
            point=delta.point,
            axis=axis,
            magnitude_cm=magnitude_cm,
            direction=direction,
            sigma=float(axis_sigmas.get(axis, delta.sigma)),
            unit=unit,
        )
        for axis, magnitude_cm, direction, unit in (
            picked[name] for name in AXIS_ORDER if name in picked
        )
    ]


# 자세 문구. 머리 위치를 cm로 지시하지 않는 이유는 그것이 자세의 **결과**이지
# 직접 조작하는 대상이 아니기 때문이다 — "머리를 3cm 내리세요"는 따를 수 없는 지시다.
# 문구는 정성적이고 정확한 cm는 `AxisCorrection.magnitude_cm`이 들고 있다.
STANCE_PHRASES: Final[dict[tuple[str, bool], str]] = {
    ("vertical", False): "자세를 한 단계 낮춰 작업하십시오.",
    ("vertical", True): "자세를 한 단계 높여 작업하십시오.",
    ("wall", True): "벽에서 한 걸음 더 떨어져 서십시오.",
    ("wall", False): "벽에 한 걸음 더 붙어 서십시오.",
}
# 머리에서 자세로 번역할 수 있는 축. lateral은 없다 — 작업자 요가 실측이 아니라
# 좌우를 판정할 기준이 없다(`service.ASSUMED_YAW`).
STANCE_AXES: Final[tuple[str, ...]] = ("wall", "vertical")


def head_stance(
    delta: PointDelta, frame: AxisFrame = WORLD_FRAME
) -> AxisCorrection | None:
    """머리의 이탈을 자세 지시 하나로 번역한다. 없으면 None.

    점 게이트를 따로 두지 않는다. 결합 sigma는 축별 sigma의 제곱합의 제곱근이므로
    어느 축이 임계를 넘으면 결합 sigma도 반드시 넘는다 — 더해도 판정이 안 바뀐다.
    """
    axis_sigmas = delta.axis_sigmas
    if not axis_sigmas:
        return None
    best: tuple[str, float, str, Vec3] | None = None
    best_sigma = CORRECTION_SIGMA_THRESHOLD
    for axis, magnitude_cm, direction, unit in _correction_candidates(
        delta.local_delta, frame
    ):
        if axis not in STANCE_AXES:
            continue
        sigma = float(axis_sigmas.get(axis, 0.0))
        if sigma > best_sigma:
            best, best_sigma = (axis, magnitude_cm, direction, unit), sigma
    if best is None:
        return None
    axis, magnitude_cm, direction, unit = best
    return AxisCorrection(
        point="head",
        axis=axis,
        magnitude_cm=magnitude_cm,
        direction=direction,
        sigma=best_sigma,
        unit=unit,
    )


def stance_text(correction: AxisCorrection) -> str:
    """자세 문장 하나. 구조체에서 지으므로 문장과 부호가 어긋날 수 없다."""
    return STANCE_PHRASES[(correction.axis, correction.magnitude_cm > 0)]


def right_hand_finding(
    deltas: Sequence[PointDelta], frame: AxisFrame = WORLD_FRAME
) -> RightHandFinding | None:
    """오른손의 가장 큰 이탈 하나. 머리·왼손은 채점에는 남지만 지목하지 않는다.

    오른손이 건을 쥐고 도막 품질을 직접 결정한다(§7). 머리는 작업자가 의도적으로
    조정하는 대상이 아니고, 왼손(플래시)은 작업 결과와의 인과가 약하다 — 가중을 낮춰
    점수에는 남기되(변경 없음), 지시로는 내지 않는다.
    """
    right = next((delta for delta in deltas if delta.point == "right"), None)
    if right is None or right.sigma <= CORRECTION_SIGMA_THRESHOLD:
        return None
    candidate = _largest_correction(right.local_delta, frame, right.axis_sigmas)
    if candidate is None:
        return None
    axis, magnitude_cm, direction, _unit = candidate
    return RightHandFinding(
        axis=axis,
        magnitude_cm=magnitude_cm,
        direction=direction,
        sigma=right.sigma,
        samples=right.samples,
    )


def _finding_phrase(finding: RightHandFinding) -> str:
    """구조체에서 문장을 짓는다. 문장과 구조체가 같은 필드에서 나오므로 어긋날 수 없다."""
    label = POINT_LABELS["right"]
    magnitude = abs(finding.magnitude_cm)
    if finding.axis == "wall":
        body = f"벽에서 {magnitude:.0f}cm 더 {finding.direction}"
    elif finding.axis == "vertical":
        body = f"{magnitude:.0f}cm {finding.direction}"
    elif finding.direction:
        body = f"진행 방향 {finding.direction} {magnitude:.0f}cm 옮기"
    else:
        body = f"벽을 따라 {magnitude:.0f}cm 옮기"
    return f"{label}{_object_particle(label)} {body}세요."


def corrections_text(
    deltas: Sequence[PointDelta], frame: AxisFrame = WORLD_FRAME
) -> list[str]:
    """오른손 하나에 대해, 가장 큰 이탈 하나만 문장 하나로 낸다. 각도가 아니라 센티미터다.

    머리·왼손은 σ가 임계를 넘어도 문장을 내지 않는다(§7 가중 근거와 별개로, 지시는
    오른손만). 사람이 한 번에 고칠 수 있는 것은 하나뿐이라 여러 이탈을 한 문장에
    쌓지 않는다.

    `frame`은 `local_delta`가 적힌 스트로크 기저다. 기저 없이 성분을 축 이름으로 읽으면
    수직 스트로크에서 좌우 이탈이 "올리세요"로 나간다(`_correction_candidates`).
    """
    finding = right_hand_finding(deltas, frame)
    if finding is None:
        return []
    return [_finding_phrase(finding)]


def issued_shift(
    deltas: Sequence[PointDelta], frame: AxisFrame = WORLD_FRAME
) -> Vec3 | None:
    """실제로 내준 교정 하나를 스트로크 프레임 성분으로 옮긴 이동벡터.

    `corrections_text`가 내는 문장과 **같은 후보**에서 나온다. 문장이 고른 축의 월드
    단위벡터에 문장이 말한 크기를 곱하고, 그것을 기저로 되돌린다. 문장이 없으면
    이동도 없다(None).
    """
    right = next((delta for delta in deltas if delta.point == "right"), None)
    if right is None or right.sigma <= CORRECTION_SIGMA_THRESHOLD:
        return None
    candidate = _largest_correction(right.local_delta, frame, right.axis_sigmas)
    if candidate is None:
        return None
    _axis, magnitude_cm, _direction, unit = candidate
    metres = magnitude_cm / 100.0
    world: Vec3 = (unit[0] * metres, unit[1] * metres, unit[2] * metres)
    # 기저가 정규직교라 각 축과의 내적이 곧 그 축의 성분이다.
    return (_dot(world, frame.normal), _dot(world, frame.tangent), _dot(world, frame.up))


def apply_measured(
    query: Mapping[str, Track],
    deltas: Sequence[PointDelta],
    frame: AxisFrame = WORLD_FRAME,
    fraction: float = 1.0,
) -> dict[str, list[Vec3 | None]]:
    """교정을 따랐을 때의 궤적. **내준 지시 그대로**만 옮긴다.

    개선율은 goal_year2의 납품 지표이고 그 정의는 "교정 알고리즘의 출력을 적용한 뒤의
    유사도 상승"이다. 그래서 시뮬레이션은 출력과 같아야 한다. `corrections_text`는
    fa9c4b0 이후 오른손 하나에 대해 가장 큰 이탈 성분 하나만 문장으로 내므로 여기서도
    오른손 하나를, 그 성분 하나만 옮긴다.

    머리·왼손은 σ가 임계를 넘어도 제자리다. 지시하지 않은 이동을 개선율에 계상하면
    작업자에게 말한 것과 보고하는 숫자가 달라진다 — 오른손만 고치라고 하고, 머리와
    왼손까지 고쳤을 때의 향상을 내걸게 된다. 문장이 좁아진 만큼 이 숫자도 작아지고
    그것이 맞다.

    못 잰 격자점은 옮기지 않는다. 자리를 지우면 s 대응이 어긋난다.
    """
    shift = issued_shift(deltas, frame)
    output: dict[str, list[Vec3 | None]] = {}
    for name, points in query.items():
        if shift is None or name != "right":
            output[name] = list(points)
            continue
        output[name] = [
            None
            if point is None
            else (
                point[0] + shift[0] * fraction,
                point[1] + shift[1] * fraction,
                point[2] + shift[2] * fraction,
            )
            for point in points
        ]
    return output
