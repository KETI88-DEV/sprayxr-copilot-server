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

# `posture_sequence`와 같은 이유다 — 불린 이름 공간에서 형제를 잡아야
# `world_axes`가 두 벌로 갈라지지 않는다.
if __package__:
    from . import world_axes
else:
    import world_axes

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
# 교정 발부 문턱. 2.0(옛값)은 200쌍/seed 20260828 코퍼스에서 발부율 47.1%,
# split median을 0.000에 박아 놓았다 — 미발부는 정의상 개선율 0이라 중앙값이
# 죽는다. Task 7·8·9(`.superpowers/sdd/2026-09-03-skeleton-view/task-{7,8,9}-report.md`)가
# 같은 코퍼스·같은 채점으로 σ를 쓸어 잰 결과:
#
#   σ      발부율    split median   ≥0.70
#   2.0    47.1%     0.000          2.6%
#   0.5    95.5%     0.661          40.7%
#   0.1    100%      0.828          76.8%
#   0.0    100%      0.835          78.1%
#
# 0.1을 골랐다(0.0이 아니다): 0.0과의 성능차가 1.3%p로 사실상 같지만, 0.1은
# "문턱"이라는 손잡이 자체를 남긴다 — 훈련생 데이터가 들어와 산포가 달라지면
# 다시 조일 자리가 필요하다.
#
# Task 9의 불일치 키 대조군(교정 후보를 고르는 레퍼런스만 질의와 무관한 전형기
# 패스로 바꾼 실험)이 이 숫자가 "교정이 아무 방향이나 향해도 이득"이 아님을
# 확인했다: 실제 방향 median 0.661 대 대조군 median −0.036(관측치의 67.4%가
# 0 이하) — 대조군 교정이 크기는 더 컸는데도(88.4cm 대 73.4cm) 결과는 더
# 나빴다. 즉 여기서 관찰된 개선은 크기가 아니라 방향(진짜 전문가 레퍼런스로
# 향했는가)에서 나온다.
#
# 범위 제한: 이 네 숫자는 전부 전문가 대 전문가다 — 코퍼스가 숙련공(전형기) 6명,
# 쌍이 전형기 질의 대 전형기 레퍼런스다. 훈련생 대 전문가는 측정된 적이 없다.
#
# MIN_CORRECTION_CM=1.0은 이 코퍼스에서 단 한 번도 안 걸린 죽은 손잡이였다
# (Task 7) — 이번 변경에서도 그대로 둔다.
CORRECTION_SIGMA_THRESHOLD: Final = 0.1
MAX_SIGMA: Final = 6.0
MIN_CORRECTION_CM: Final = 1.0
# 수평면에서 이보다 짧은 축 성분은 방향이 정해지지 않은 것으로 본다.
MIN_AXIS_LENGTH: Final = 1e-9
# 월드 수직축은 world_axes가 소유한다(캡처 CSV는 유니티 좌표 — Y가 위). 여기서 다시
# 정의하지 않는다 — 쓰는 자리마다 `world_axes.UP`/`world_axes.vertical_of` 등을 호출
# 시점에 읽는다. 모듈 상수로 한 번 복사해 두면 이후 규약이 바뀌어도 반응하지 않는다.


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


def _axis_unit(index: int) -> Vec3:
    out = [0.0, 0.0, 0.0]
    out[index] = 1.0
    return (out[0], out[1], out[2])


class _WorldFrame:
    """기저 없이 (normal, tangent, up) 슬롯 순서로 월드 축을 고정한 기본 프레임.

    로컬 index0/1/2는 `stroke_frame.Basis`와 같은 뜻(법선/진행/수직)을 유지해야
    한다 — 그래야 `PointDelta.local_delta`를 기저 없이 직접 준 호출자(단위 테스트 등)
    에서도 index2가 계속 "수직"으로 읽힌다. `up`을 `world_axes.UP`로, normal·tangent를
    나머지 두 수평축(`world_axes.HORIZONTAL_INDICES`)으로 잡는 이유가 그것이다.

    세 축 모두 **property**다 — frozen dataclass 필드였을 때는(`up: Vec3 =
    world_axes.UP`) 이 클래스가 임포트되는 시점에 딱 한 번 평가돼 그 뒤로는
    `UP_INDEX`가 바뀌어도 굳어 있었다(리뷰 A2). `WORLD_FRAME`은 모듈 하나짜리
    싱글턴이라 `frame: AxisFrame = WORLD_FRAME` 기본 인자는 이 객체 하나만
    한 번 붙잡지만, 그 객체의 `.normal`/`.tangent`/`.up`은 매번 접근할 때(즉
    호출 시점에) `world_axes`를 다시 읽으므로 객체 자체는 고정돼도 값은 고정되지
    않는다. `_axis_components`가 이 프레임으로 되돌린 월드 벡터에
    `world_axes.vertical_of` 등을 적용해 다시 판정하는 것과는 별개의 반응성이다
    — 프레임 자신의 축이 하나라도 굳어 있으면 `_world_delta`가 잘못된 방향으로
    로컬 성분을 되돌린다.
    """

    @property
    def normal(self) -> Vec3:
        first, _second = world_axes.HORIZONTAL_INDICES
        return _axis_unit(first)

    @property
    def tangent(self) -> Vec3:
        _first, second = world_axes.HORIZONTAL_INDICES
        return _axis_unit(second)

    @property
    def up(self) -> Vec3:
        return world_axes.UP


# 기저를 주지 않은 호출자에게 쓰는 기본값. 로컬 성분을 월드 축으로 그대로 읽는다.
WORLD_FRAME: Final = _WorldFrame()


@dataclass(frozen=True, slots=True)
class PointDelta:
    """한 점의 이탈.

    **`local_delta`는 월드 벡터다** — 질의에서 레퍼런스로 가는 `(r[i] - q[i])`이고
    `score_track`이 그렇게 만든다. 이름이 `local_`로 시작하지만 프레임 성분이 아니다.

    이 이름과 예전 설명("스트로크 프레임 벡터")이 실제로 사람을 오도했다. 그것을 믿고
    손으로 `local_delta=(0, 0, 0.10)`을 "수직 교정"이라 여겨 탐침을 만들면, 그 값은
    실은 "월드 Z로 0.10"이라 `apply_measured`가 월드 Z로 옮기는 것이 **옳은데**
    결함으로 읽힌다. 그렇게 읽고 `_world_delta`를 끼워 넣으면 변환이 한 번 더 얹혀
    작동하던 경로가 깨진다(검사 15개가 잡는다).

    파이프라인이 이 값을 두 번 지나며 상쇄한다: `_axis_components`가 프레임 성분으로
    보고 `_world_delta`로 월드 방향(`AxisCorrection.unit`)을 만들고, `issued_shifts`가
    그 월드 누적을 다시 프레임 기저에 투영해 되돌린다. 그래서 `apply_measured`가 더하는
    값이 결국 월드 변위와 같아진다. **이름을 바꾸려면 그 두 곳을 함께 봐야 한다** —
    한쪽만 고치면 상쇄가 깨진다.

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
    """수평 성분의 단위벡터. 축이 수직이면 수평 방향이 없으므로 None이다.

    어느 인덱스가 수평인지는 `world_axes.horizontal_of`가 호출 시점에 정한다 —
    좌표를 직접 인덱싱하면(옛 코드의 `vector[0], vector[1]`) 월드 수직축이 바뀔 때
    이 함수가 반응하지 않는다.
    """
    h0, h1 = world_axes.horizontal_of(vector)
    length = math.hypot(h0, h1)
    if length < MIN_AXIS_LENGTH:
        return None
    a, b = world_axes.HORIZONTAL_INDICES
    out = [0.0, 0.0, 0.0]
    out[a] = h0 / length
    out[b] = h1 / length
    return (out[0], out[1], out[2])


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
      수평 잔차도 없으면(원점) lateral이 `None`이다. vertical은 항상 `world_axes.UP`이다.
    - `word`: 그 축·부호에 대응하는 동사/방향어. 후보에 실제로 들어갈지와 무관하게
      부호만으로 정해진다 — 임계는 호출부(`_correction_candidates`)가 따로 건다.

    로컬 성분을 축 이름으로 그대로 읽으면 안 된다. `Basis.up`은 순수 수직 스트로크에서
    수평이 되므로(그때 수직은 `tangent`다) `local[2]`를 "올리세요"로 적으면 옆으로
    틀어진 손을 위로 올리라고 말하게 된다. 그래서 월드로 되돌린 뒤 세 방향으로 다시
    나눈다.

    - 수직(`"vertical"`): 월드 수직축 성분(`world_axes.vertical_of`). 어느 축에
      실려 있었든 여기서만 올리/내리가 나온다.
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
    plane: Vec3 = world_axes.flatten(world)
    up_cm = world_axes.vertical_of(world) * 100.0

    wall = _flattened(frame.normal)
    normal_cm = 0.0 if wall is None else _dot(plane, wall) * 100.0
    if wall is not None:
        plane = (
            plane[0] - normal_cm / 100.0 * wall[0],
            plane[1] - normal_cm / 100.0 * wall[1],
            plane[2] - normal_cm / 100.0 * wall[2],
        )
    wall_word = "" if wall is None else ("띄우" if normal_cm > 0 else "붙이")

    vertical_word = "올리" if up_cm > 0 else "내리"

    along = _flattened(frame.tangent)
    if along is None:
        # 수평 진행이 없다. 남은 수평 성분 자체가 방향이라 크기는 늘 양수다.
        lateral_unit = _flattened(plane)
        h0, h1 = world_axes.horizontal_of(plane)
        lateral_cm = math.hypot(h0, h1) * 100.0
        lateral_word = ""
    else:
        lateral_unit = along
        lateral_cm = _dot(plane, along) * 100.0
        lateral_word = "앞으로" if lateral_cm > 0 else "뒤로"

    return {
        "wall": (normal_cm, wall, wall_word),
        "vertical": (up_cm, world_axes.UP, vertical_word),
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
    # C1. 부호는 `_axis_components`의 lateral 성분 부호이고, 그 부호는 `frame.tangent`의
    # 수평 방향과의 내적이다 — 양수면 기준이 진행 방향 쪽에 있다는 뜻이다.
    # "앞으로/뒤로"를 쓰지 않는다: 작업자는 그것을 벽 쪽/벽 반대쪽으로 읽는데 그 뜻은
    # 이미 wall 축이 쓰고 있다. 문장에서 진행축임이 드러나야 한다.
    ("lateral", True): "분사 진행 방향으로 한 걸음 옮겨 서십시오.",
    ("lateral", False): "분사 진행 반대 방향으로 한 걸음 옮겨 서십시오.",
}
# 머리에서 자세로 번역할 수 있는 축.
#
# lateral이 여기 있다(C1, `docs/measurements/2026-09-23-head-lateral-stance-axis-prereg.md`).
# 예전 주석은 "lateral은 없다 — 작업자 요가 실측이 아니라 좌우를 판정할 기준이 없다
# (`service.ASSUMED_YAW`)"였는데 두 겹으로 틀렸다. (1) 요는 실측이다 — 사전등록 §0.4의
# 190패스 전부에서 `KeyInputs.yaw`가 `None`이 아니고, `service.py`의 주석도 "작업자 요는
# 이제 실측이다"라고 적고 있어 두 주석이 서로 반대였다. (2) 더 근본적으로 이 lateral은 몸의
# 좌우가 아니라 **스트로크 진행축**이라 부호와 방향어가 `frame.tangent`의 수평 성분에서
# 나온다 — 요와 무관하다. 요가 필요한 경우는 수평 진행이 전혀 없는 순수 수직 스트로크뿐이고,
# 그때는 `_axis_components`가 방향어를 비워 두므로 `head_stance`가 후보에서 뺀다.
#
# 머리 세 축의 σ 중앙값은 wall 0.515 · vertical 0.329 · lateral 0.616이었다 — 가장 큰 축이
# 유일하게 말하지 않는 축이었다.
STANCE_AXES: Final[tuple[str, ...]] = ("wall", "vertical", "lateral")


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
        if not direction:
            # 방향어가 없는 축(수평 진행이 없는 스트로크의 lateral)은 문장을 지을 수 없다.
            # `STANCE_PHRASES`는 (axis, magnitude > 0)로만 조회하므로 여기서 막지 않으면
            # 방향을 지어낸 문장이 나간다 — 사전등록 §6 F6의 안전 장치다.
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

    이 함수는 여전히 문장 하나만 낸다 — 반면 `apply_measured`/`issued_shifts`는
    발부한 축 전부(오른손 다축 + 머리 자세)를 옮긴다. 이것은 의도한 설계다:
    작업자는 산문 한 줄과 숫자 패널을 함께 받는다. 패널이 `issued_axes`/`axes_block`
    (`pose_lab/service.py`)이 내는 `axes[]`이고, 거기 있는 축은 이 문장이 말하지
    않아도 실제로 발부되고 시뮬레이션도 옮긴다. 그러니 "문장과 시뮬레이션이 다른
    것을 옮긴다"처럼 보여도 고치지 말 것 — 문장은 산문, `axes[]`는 정확한 지시다.
    """
    finding = right_hand_finding(deltas, frame)
    if finding is None:
        return []
    return [_finding_phrase(finding)]


# 작업자에게 **실제로 전달되는** 안내 수. 화면은 발부된 축을 전부 보이지만 작업자가 듣는 것은
# σ가 가장 큰 하나다(제품 MCP의 `arm_reference.MAX_CORRECTIONS`도 1). 사용자(2026-10-01):
# "실제 안내되는 게 1개라면 그 1개를 적용했을 때의 유사도 개선을 적어야 한다." 그래서 개선율은
# 이 수만큼만 옮겨 잰다 — 말하지 않은 안내를 적용한 상승을 보고하면 숫자가 부풀어 보인다.
DELIVERED_CUES: Final = 1


def issued_axes(
    deltas: Sequence[PointDelta], frame: AxisFrame = WORLD_FRAME
) -> list[AxisCorrection]:
    """실제로 발부하는 축 전부. sigma 내림차순.

    오른손은 다축, 머리는 자세 문장 하나, 왼손은 없다. 이 목록이 작업자에게 가는
    내용과 정확히 같아야 개선율이 정직하다 — `apply_measured`가 이것만 옮긴다.
    """
    issued: list[AxisCorrection] = []
    for delta in deltas:
        if delta.point == "right":
            issued.extend(right_axes(delta, frame))
        elif delta.point == "head":
            stance = head_stance(delta, frame)
            if stance is not None:
                issued.append(stance)
    return sorted(issued, key=lambda c: c.sigma, reverse=True)


def delivered_axes(
    deltas: Sequence[PointDelta], frame: AxisFrame = WORLD_FRAME, limit: int | None = DELIVERED_CUES
) -> list[AxisCorrection]:
    """작업자에게 전달되는 축 — `issued_axes`(σ 내림차순)의 앞 `limit`개. `None`이면 전부."""
    issued = issued_axes(deltas, frame)
    return issued if limit is None else issued[: max(limit, 0)]


def issued_shifts(
    deltas: Sequence[PointDelta], frame: AxisFrame = WORLD_FRAME, limit: int | None = None
) -> dict[str, Vec3]:
    """점 -> 스트로크 프레임 이동벡터. **발부한 점만 키를 갖는다.**

    `dict[str, Vec3]`라는 타입 자체는 "left" 같은 임의의 키를 막지 않는다 — 이
    딕셔너리에 발부하지 않은 점의 키가 생기지 않는 것은 이 함수의 호출부, 즉
    `issued_axes`가 `point == "right"`/`"head"`일 때만 `AxisCorrection`을 만들고
    그 외에는(왼손 포함) 아무것도 만들지 않기 때문이다. `apply_measured`는 이
    딕셔너리에서 키가 없는 점을 옮길 방법 자체가 없으므로, 지시하지 않은 이동이
    개선율에 섞이는 경로는 `issued_axes`의 이 분기가 막는다. 한 점의 여러 축은
    월드에서 더한 뒤 기저로 되돌린다.
    """
    world: dict[str, list[float]] = {}
    for correction in delivered_axes(deltas, frame, limit):
        metres = correction.magnitude_cm / 100.0
        acc = world.setdefault(correction.point, [0.0, 0.0, 0.0])
        for i in range(3):
            acc[i] += correction.unit[i] * metres
    # 기저가 정규직교라 각 축과의 내적이 곧 그 축의 성분이다.
    return {
        point: (
            _dot((v[0], v[1], v[2]), frame.normal),
            _dot((v[0], v[1], v[2]), frame.tangent),
            _dot((v[0], v[1], v[2]), frame.up),
        )
        for point, v in world.items()
    }


def apply_measured(
    query: Mapping[str, Track],
    deltas: Sequence[PointDelta],
    frame: AxisFrame = WORLD_FRAME,
    fraction: float = 1.0,
    limit: int | None = None,
) -> dict[str, list[Vec3 | None]]:
    """교정을 따랐을 때의 궤적. **내준 지시 그대로**만 옮긴다.

    개선율은 goal_year2의 납품 지표이고 그 정의는 "교정 알고리즘의 출력을 적용한 뒤의
    유사도 상승"이다. 그래서 시뮬레이션은 출력과 같아야 한다.

    `issued_axes`가 낸 축만, 그만큼만 옮긴다. 왼손은 어떤 경우에도 제자리다 —
    지시하지 않은 이동을 개선율에 계상하면 작업자에게 말한 것과 보고하는 숫자가
    달라진다. 이 함수가 `issued_shifts`의 딕셔너리를 쓰는 이유가 그것이다:
    키가 없는 점은 옮길 방법 자체가 없다.

    못 잰 격자점은 옮기지 않는다. 자리를 지우면 s 대응이 어긋난다.
    """
    # `limit`: σ 상위 몇 개를 옮기나. `None`이면 발부된 축 전부(옛 정의), `DELIVERED_CUES`면
    # 작업자에게 실제로 전달되는 것만이다.
    shifts = issued_shifts(deltas, frame, limit)
    output: dict[str, list[Vec3 | None]] = {}
    for name, points in query.items():
        shift = shifts.get(name)
        if shift is None:
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
