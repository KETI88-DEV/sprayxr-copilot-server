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
from dataclasses import dataclass
from typing import Final, Mapping, Protocol, Sequence

Vec3 = tuple[float, float, float]
# 왼손은 프레임 단위로 빠질 수 있다(설계 §5.4). 자리는 남기고 값만 None이다.
Track = Sequence[Vec3 | None]

DEFAULT_WEIGHTS: Final[dict[str, float]] = {"right": 1.0, "head": 0.4, "left": 0.15}
POINT_LABELS: Final[dict[str, str]] = {"right": "오른손", "head": "머리", "left": "왼손"}
POINT_ORDER: Final[tuple[str, ...]] = ("right", "head", "left")

# 산포 하한. 측정 잡음이 작을 때 σ가 무한히 부풀지 않게 한다.
SIGMA_FLOOR_M: Final = 0.02
CORRECTION_SIGMA_THRESHOLD: Final = 2.0
MAX_SIGMA: Final = 6.0
MIN_CORRECTION_CM: Final = 1.0
# 수평면에서 이보다 짧은 축 성분은 방향이 정해지지 않은 것으로 본다.
MIN_AXIS_LENGTH: Final = 1e-9


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
    spreads: Mapping[str, float],
    weights: Mapping[str, float],
) -> MeasuredResult:
    """정렬 격자 위 세 점의 가중 유사도. 못 잰 점은 0점이 아니라 제외다.

    격자점 하나가 비어도(왼손 마스킹) 그 점만 빼고 나머지 격자점으로 잰다. 자리를
    없애지 않고 None으로 두는 이유는 두 패스의 같은 s가 계속 마주 보게 하기 위함이다.
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
        spread = max(spreads.get(name, SIGMA_FLOOR_M), SIGMA_FLOOR_M)
        magnitude = math.sqrt(sum(axis * axis for axis in mean_delta))
        deltas.append(
            PointDelta(
                point=name,
                sigma=min(MAX_SIGMA, magnitude / spread),
                local_delta=mean_delta,
                samples=len(pairs),
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


def _correction_candidates(
    local: Vec3, frame: AxisFrame
) -> list[tuple[str, float, str]]:
    """한 점의 이탈을 물리적으로 참인 후보 세 개(벽·수직·진행축)로 나눈다.

    로컬 성분을 축 이름으로 읽으면 안 된다. `Basis.up`은 순수 수직 스트로크에서 수평이
    되므로(그때 수직은 `tangent`다) `local[2]`를 "올리세요"로 적으면 옆으로 틀어진 손을
    위로 올리라고 말하게 된다. 그래서 월드로 되돌린 뒤 세 방향으로 다시 나눈다.

    - 수직(`"vertical"`): 월드 Z 성분. 어느 축에 실려 있었든 여기서만 올리/내리가 나온다.
    - 벽(`"wall"`): 법선은 항상 수평이므로(`Basis` 독스트링) 수평 성분을 법선에 투영한다.
    - 진행축(`"lateral"`): 법선에 직교하는 수평 방향이다. 스트로크가 수평으로 진행하면
      그것이 진행 방향이라 앞뒤로 적는다. 수직 스트로크에는 수평 진행이 없어 좌우를
      판정할 기준이 없다 — 작업자 요가 실측이 아니기 때문이다(`service.ASSUMED_YAW`).
      방향을 지어내지 않고 "벽을 따라"까지만 적는다(`direction`은 빈 문자열).

    임계(`MIN_CORRECTION_CM`) 아래인 후보는 목록에 넣지 않는다. 호출자가 절댓값이 가장
    큰 후보 하나만 골라 문장 하나를 낸다 — 여러 후보를 동시에 지시하지 않는다.
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

    candidates: list[tuple[str, float, str]] = []
    if abs(normal_cm) >= MIN_CORRECTION_CM:
        verb = "띄우" if normal_cm > 0 else "붙이"
        candidates.append(("wall", normal_cm, verb))
    if abs(up_cm) >= MIN_CORRECTION_CM:
        verb = "올리" if up_cm > 0 else "내리"
        candidates.append(("vertical", up_cm, verb))

    along = _flattened(frame.tangent)
    if along is None:
        lateral_cm = math.hypot(plane[0], plane[1]) * 100.0
        if lateral_cm >= MIN_CORRECTION_CM:
            candidates.append(("lateral", lateral_cm, ""))
    else:
        lateral_cm = _dot(plane, along) * 100.0
        if abs(lateral_cm) >= MIN_CORRECTION_CM:
            direction = "앞으로" if lateral_cm > 0 else "뒤로"
            candidates.append(("lateral", lateral_cm, direction))
    return candidates


def _largest_correction(
    local: Vec3, frame: AxisFrame
) -> tuple[str, float, str] | None:
    """절댓값이 가장 큰 후보 하나. 사람은 한 번에 하나만 고칠 수 있다."""
    candidates = _correction_candidates(local, frame)
    if not candidates:
        return None
    return max(candidates, key=lambda candidate: abs(candidate[1]))


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
    candidate = _largest_correction(right.local_delta, frame)
    if candidate is None:
        return None
    axis, magnitude_cm, direction = candidate
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


def apply_measured(
    query: Mapping[str, Track],
    deltas: Sequence[PointDelta],
    fraction: float = 1.0,
) -> dict[str, list[Vec3 | None]]:
    """교정을 따랐을 때의 궤적. 지목된 점만 fraction만큼 레퍼런스 쪽으로 옮긴다.

    못 잰 격자점은 옮기지 않는다. 자리를 지우면 s 대응이 어긋난다.
    """
    moves = {
        delta.point: delta.local_delta
        for delta in deltas
        if delta.sigma > CORRECTION_SIGMA_THRESHOLD
    }
    output: dict[str, list[Vec3 | None]] = {}
    for name, points in query.items():
        shift = moves.get(name)
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
