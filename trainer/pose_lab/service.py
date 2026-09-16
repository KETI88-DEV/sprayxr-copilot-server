#!/usr/bin/env python3
"""비교 한 건을 만드는 순수 로직. HTTP를 모른다.

두 축을 병기하는 것이 이 모듈의 존재 이유다. 실측 축은 근거가 측정값이고 관절 축은
IK 산출물이다. 어느 쪽이 개선율 70%를 낼 수 있는지 지금 아무도 모르므로 하나를 골라
계약으로 굳히기 전에 둘을 나란히 놓고 실제 패스를 넘겨 본다.
"""

from __future__ import annotations

import functools
import importlib
import json
import logging
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Final, Mapping, Protocol, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

measured_pose = importlib.import_module("measured_pose")
measured_score = importlib.import_module("measured_score")
motion_key = importlib.import_module("motion_key")
pose_similarity = importlib.import_module("pose_similarity")
stroke_frame = importlib.import_module("stroke_frame")
build_spread_reference = importlib.import_module("build_spread_reference")
level_mesh = importlib.import_module("level_mesh")
pose_fk_csv = importlib.import_module("pose_fk_csv")
simulator_csv = importlib.import_module("simulator_csv")
world_axes = importlib.import_module("world_axes")

from trainer.pose_lab import body_proportions, composite, instructions, joint_angles, orientation, skeleton
from trainer.pose_lab.spread_reference import Resolved, SpreadReference

logger = logging.getLogger(__name__)

Vec3 = tuple[float, float, float]

# REQ-19 correction_target_gate. 납품 후 수치를 보고 기준을 옮기는 것을 막으려고 박아 둔다.
GATE_RATIO: Final = 0.5
# 잡으려는 최소 자세 오류. 팔꿈치 30도에 해당하는 실측 참조값이다.
DEFAULT_ERROR_M: Final = 0.144

# 레퍼런스가 버킷 평균이 아니라 **한 패스**다(`build_spread_reference.pass_axis_values`가
# 패스 하나의 점별 평균 위치를 wall/vertical/lateral로 분해한 값의 산포다). 질의와
# 레퍼런스가 같은 분포에서 독립으로 뽑힌 두 표본이므로 그 차의 분산은 2σ²다 — σ로
# 그대로 나누면 모든 z가 √2배 부풀고, 아무 문제 없는 사람도 1.41σ로 읽혀 임계 2.0에
# 육박한다. 아티팩트의 σ는 절대 위치의 산포이고 score_track이 재는 것은 차이이므로,
# 여기(배선 지점)에서 다리를 놓는다. `measured_score.py`는 건드리지 않는다 — 거기서
# 고치면 기존 테스트가 전부 흔들리고 이유가 묻힌다.
PAIR_SIGMA_FACTOR: Final = math.sqrt(2.0)

# 레퍼런스 버킷에서 잰 산포가 아니라 자리를 채워 둔 임시값이다. 이 값으로 게이트를
# 판정하면 상수 두 개에서 나온 "pass"를 화면이 판정으로 내걸게 된다. REQ-19 각도
# 유효화는 설계 §10에서 미판정이고 D4는 게이트를 옮기지 말라고 못박는다.
#
# 점당 스칼라가 아니라 점·축(wall/vertical/lateral)별 산포다. 값이 축마다 모두
# 같으므로 `measured_score.uniform_spreads`가 예전 스칼라 채점과 같은 σ를 낸다.
PROVISIONAL_SPREADS: Final[dict[str, dict[str, float]]] = measured_score.uniform_spreads(0.05)

# 작업자 요는 이제 실측이다. `measured_pose.facing_yaw`가 HMD 회전에서 뽑고, 동작 키의
# ②진행 방향과 ④몸 기준 착탄 위치가 그 위에 선다 — 이름과 실제가 처음으로 일치한다.
#
# 이 상수는 **회전 컬럼을 못 읽은 패스의 대체값**으로만 남는다. 그런 패스에서 요를
# 지어내면 몸 기준 축이 조용히 월드 방위각으로 돌아가므로, 응답이 `yaw_source`로 어느
# 쪽인지 말한다.
ASSUMED_YAW: Final = 0.0

JOINT_AXIS_NOTE: Final = (
    "관절 20개는 3점 추적에서 IK로 푼 값이다. 각도 지시는 REQ-19 통과 전까지 내지 않는다."
)

# 기준(레퍼런스) 작업자는 이제 고를 수 없다 — 전형기로 고정이다(태스크 11 브리프,
# 사용자 지시). `build_spread_reference.py`가 산포 아티팩트를 굽는 바로 그 작업자와
# 같은 사람이다. 리터럴로 고정하는 이유: 이 화면은 σ 아티팩트(`reference_spreads`)가
# 안 실려도(잠정 산포로 도는 배포에서도) 기준 후보 목록은 낼 수 있어야 하므로,
# `reference_spreads.worker`에 의존하지 않는다.
EXPERT_WORKER: Final = "전형기"


@dataclass(frozen=True, slots=True)
class PassRef:
    worker: str
    date: str
    level: str
    session: str
    pass_index: int


@dataclass(frozen=True, slots=True)
class RawPass:
    rows: list[dict[str, str]]


class PassSource(Protocol):
    def workers(self) -> list[str]: ...
    def load(self, ref: PassRef) -> RawPass: ...


def gate_verdict(spread: float, error: float) -> str | None:
    """s/e < 0.5 이면 각도를 실어도 된다. 아니면 관절 지목까지만."""
    if error <= 0.0:
        return None
    return "pass" if spread / error < GATE_RATIO else "fail"


def improvement(
    before_same: float,
    after_split: float | None,
    after_same: float,
    before_split: float | None = None,
) -> dict[str, float | None]:
    """개선율을 분리 채점과 동일 채점 둘 다 낸다.

    goal_year2의 문구는 목표 자세를 입력으로 주므로 동일 채점에 해당한다. 두 값을
    병기해야 그 값이 자기충족에 얼마나 기대는지 드러난다. 분리 채점 점수가 없으면
    (아직 계산되지 않았으면) 없는 값을 만들어내지 않고 None을 그대로 낸다.

    `same`과 `split`은 각자 자기 안에서만 일관돼야 한다 — `gap_closure`는
    `(S1-S0)/(1-S0)`인데, S0와 S1을 서로 다른 잣대(산포)로 재면 교정이 아무것도
    옮기지 않아도 잣대를 바꾼 것만으로 G≠0이 나온다. 그 방향(향상처럼 보일지 악화처럼
    보일지)은 두 잣대 중 어느 쪽이 더 넓은가에 좌우되고, 그건 버킷마다 다르므로 결과가
    잣대와 함께 흔들리는 지표가 된다. 그래서 `same`은 전체 레퍼런스(F) 하나로 S0·S1을
    다 재고, `split`은 교정을 고른 레퍼런스(A)가 아니라 **채점 레퍼런스(B) 하나로
    S0·S1을 다 잰다** — `before_split`이 그 B 위에서 낸 교정 전 점수다. 이렇게 해야
    두 숫자 사이에서 움직이는 것이 오직 "교정을 적용했는가"뿐이 된다
    (`eval_pose_correction.py::measure_correction`이 이미 쓰는 패턴: A로 교정을
    고르고 B로 앞뒤를 다 잰다).

    `gap_closure`가 내는 None도 그대로 흘린다. 이미 기준과 같으면(S0=1) 메울 격차가
    없어 개선율이 정의되지 않는데, 여기서 0.0으로 바꾸면 "메우지 못했다"로 읽힌다.
    화면은 None을 대시로 그리고 있으므로 뒤에서 값을 지어낼 이유가 없다.
    """
    same = pose_similarity.gap_closure(before_same, after_same)
    if after_split is None or before_split is None:
        return {"split": None, "same": same}
    return {"split": pose_similarity.gap_closure(before_split, after_split), "same": same}


def pass_frames(raw: RawPass, samples: int = stroke_frame.DEFAULT_SAMPLES) -> list[Any]:
    """패스 한 건의 정렬 격자. 분사 중이고 착탄점이 있는 프레임만 남는다."""
    frames = measured_pose.measured_frames(raw.rows)
    spraying = [frame for frame in frames if frame.spraying and frame.hit is not None]
    return stroke_frame.resample_frames(spraying, samples)


@dataclass(frozen=True, slots=True)
class KeyInputs:
    """동작 키를 만드는 재료. 판정 경계(KeyParams)와 분리해 둔다.

    재료는 패스에서 한 번 뽑고 경계는 요청마다 달라진다. 둘을 붙여 두면 화면이 UV
    격자를 바꿀 때마다 CSV를 다시 읽어야 한다.
    """

    displacement: Vec3
    head: Vec3
    hit: Vec3
    uv: tuple[float, float] | None
    yaw: float | None = None


def key_inputs(frames: Sequence[Any]) -> KeyInputs | None:
    if len(frames) < 2:
        return None
    first, last = frames[0], frames[-1]
    if first.hit is None or last.hit is None:
        return None
    middle = frames[len(frames) // 2]
    return KeyInputs(
        displacement=(
            last.hit[0] - first.hit[0],
            last.hit[1] - first.hit[1],
            last.hit[2] - first.hit[2],
        ),
        head=middle.head,
        hit=middle.hit or first.hit,
        uv=middle.uv,
        yaw=getattr(middle, "yaw", None),
    )


def motion_key_for(frames: Sequence[Any], params: Any) -> str | None:
    """작업자 요는 실측이다. 패스 중앙 격자점의 HMD 방위를 쓴다.

    실제로 쓴 값과 그 출처가 응답의 `yaw_assumed` · `yaw_source`로 나가고 화면이 동작
    키 옆에 적는다. 회전을 못 읽어 대체값으로 떨어진 패스에서는 몸 기준 이름을 단 축이
    다시 월드 기준이 되므로, 읽는 사람이 그 사실을 알아야 한다.
    """
    inputs = key_inputs(frames)
    if inputs is None:
        return None
    return key_from(inputs, params)


def yaw_of(inputs: KeyInputs) -> float:
    return ASSUMED_YAW if inputs.yaw is None else inputs.yaw


def key_from(inputs: KeyInputs, params: Any) -> str:
    return motion_key.motion_key(
        displacement=inputs.displacement,
        yaw=yaw_of(inputs),
        head=inputs.head,
        hit=inputs.hit,
        uv=inputs.uv,
        params=params,
    )


def representative_basis(frames: Sequence[Any]) -> Any:
    """패스 하나를 대표하는 스트로크 기저.

    격자점마다 기저가 다른데 교정문은 하나다. 어느 기저에서 적힌 성분인지 정하지 않고
    말로 옮기면 축 이름이 뜻을 대신하게 되고, 그것이 수직 스트로크에서 좌우 이탈을
    "올리세요"로 만든 결함이다. 패스는 방향 전환에서 잘린 직선 구간이므로 순 변위를
    접선으로 삼은 기저가 격자점 기저들을 대표한다.

    질의 쪽 기저를 쓴다. 교정을 따를 사람이 질의 패스의 작업자이기 때문이다.
    """
    if not frames:
        return measured_score.WORLD_FRAME
    first, last = frames[0], frames[-1]
    middle = frames[len(frames) // 2]
    if first.hit is None or last.hit is None or middle.hit is None:
        return measured_score.WORLD_FRAME
    tangent = (
        last.hit[0] - first.hit[0],
        last.hit[1] - first.hit[1],
        last.hit[2] - first.hit[2],
    )
    return stroke_frame.stroke_basis(middle.hit, tangent, middle.right)


def _local_tracks(frames: Sequence[Any]) -> dict[str, list[Vec3 | None]]:
    """격자점마다 스트로크 프레임 좌표. 왼손이 없는 격자점은 자리를 남기고 None이다.

    자리를 지우면 리스트가 짧아져 두 패스의 s 대응이 한 칸씩 밀린다. 왼손만 프레임
    단위로 빠지는 설계 §5.4의 마스킹이 다른 두 점의 대응까지 망가뜨리게 된다.
    """
    tracks: dict[str, list[Vec3 | None]] = {"right": [], "head": [], "left": []}
    previous: Vec3 = stroke_frame.FALLBACK_TANGENT
    for index, frame in enumerate(frames):
        if frame.hit is None:
            continue
        following = frames[index + 1] if index + 1 < len(frames) else None
        tangent = (0.0, 0.0, 0.0)
        if following is not None and following.hit is not None:
            tangent = (
                following.hit[0] - frame.hit[0],
                following.hit[1] - frame.hit[1],
                following.hit[2] - frame.hit[2],
            )
        # 마지막 격자점에는 다음 프레임이 없다. 월드 축으로 떨어뜨리면 그 표본의
        # 진행·수직 축만 나머지 19개와 다른 방향을 가리키고, 그 차이가 평균 이탈에
        # 섞여 들어간다. 직전 접선을 그대로 이어 쓴다.
        if tangent == (0.0, 0.0, 0.0):
            tangent = previous
        else:
            previous = tangent
        basis = stroke_frame.stroke_basis(frame.hit, tangent, frame.right)
        tracks["right"].append(stroke_frame.to_local(basis, frame.right))
        tracks["head"].append(stroke_frame.to_local(basis, frame.head))
        tracks["left"].append(
            None if frame.left is None else stroke_frame.to_local(basis, frame.left)
        )
    return tracks


def _reference_keys(inputs: KeyInputs) -> dict[str, str]:
    """질의 패스의 사다리 네 층 키. `SpreadReference.resolve`가 훑는 층 이름과 같아야
    버킷을 찾는다 — `build_spread_reference.build_from_root`가 굽는 키와 동일한 규칙
    (uv8/uv4/uv2는 `KeyParams(uv_grid=N)`, uv_off는 uv8 키의 UV 성분을 na로 치환)이다.
    """
    keys: dict[str, str] = {}
    for layer, grid in build_spread_reference.GRANULARITIES:
        if layer == "uv_off":
            continue
        keys[layer] = key_from(inputs, motion_key.KeyParams(uv_grid=grid))
    keys["uv_off"] = build_spread_reference.key_without_uv(keys["uv8"])
    return keys


def _scale_spreads(
    spreads: Mapping[str, Mapping[str, float]],
) -> dict[str, dict[str, float]]:
    """레퍼런스 σ(절대 위치의 산포)를 질의-레퍼런스 차분의 σ로 바꾼다. `PAIR_SIGMA_FACTOR`
    독스트링 참조."""
    return {
        point: {axis: value * PAIR_SIGMA_FACTOR for axis, value in axes.items()}
        for point, axes in spreads.items()
    }


def _half_sample_count(
    reference_spreads: SpreadReference, granularity: str, key: str, dates: Sequence[str]
) -> int:
    """분리 채점의 A/B 반쪽이 실제로 몇 표본으로 서 있는지.

    `SpreadReference.spreads_for`는 σ만 내고 표본 수는 버린다 — Task 4는 리뷰를 마치고
    닫혔으므로 `spread_reference.py`의 시그니처는 여기서 건드리지 않는다. 대신 같은
    원시 버킷(`_bucket`)을 직접 읽어 `resolve()`/`_merge_dates`가 쓰는 것과 같은 규칙
    (점·축 조합별로 날짜를 걸쳐 합산한 뒤 그 최댓값)으로 n을 다시 센다. `SpreadReference`
    바깥에서 `_bucket`(비공개 이름)을 부르는 것은 계약 위반이 아니라, 그 계약이 표본
    수를 내주지 않기 때문에 같은 패키지 안에서 원시 구조를 다시 읽는 것이다.
    """
    bucket = reference_spreads._bucket(granularity, key)  # noqa: SLF001
    if bucket is None:
        return 0
    by_date = bucket.get("by_date")
    if not isinstance(by_date, Mapping):
        return 0
    totals: dict[tuple[str, str], int] = {}
    for date in dates:
        day = by_date.get(date)
        if not isinstance(day, Mapping):
            continue
        for point, axes in day.items():
            for axis, raw in axes.items():
                key_pa = (point, axis)
                totals[key_pa] = totals.get(key_pa, 0) + int(raw.get("n", 0))
    return max(totals.values()) if totals else 0


def axes_block(
    issued: Sequence[Any],
    frame: measured_score.AxisFrame | None = None,
) -> list[dict[str, object]]:
    """발부한 축 전부를 sigma 내림차순으로. 작업자에게 가는 숫자 패널의 원본이다.

    `issued`는 `compare()`가 `measured_score.issued_axes`를 **한 번** 불러 얻은 목록이다 —
    이 블록과 `view.corrections`가 같은 목록을 받아야 번호(n)로 짝지을 수 있다. 옛 호출
    형태(`axes_block(deltas, frame)`)는 여기서 `issued_axes`를 한 번 불러 같은 목록을 만든다.

    이 목록은 `compare()`가 실제로 발부하는 지시이고, 그것이 곧 `apply_measured`가
    옮기는 것이다 — 그래서 화면 숫자와 `improvement.same`이 같은 이동을 잰다.

    `improvement.split`에는 그 등식이 성립하지 않는다. split은 자기충족을 피하려고
    교정 후보를 A 반쪽의 산포로 다시 고르므로(`before_a.deltas`) sigma가 달라져
    발부 축 집합이 이 목록과 갈릴 수 있다. 그것이 split의 설계지 결함이 아니다 —
    다만 이 목록을 split의 근거로 읽으면 안 된다.
    """
    if frame is not None:
        issued = measured_score.issued_axes(issued, frame)
    return [
        {
            "point": correction.point,
            "axis": correction.axis,
            "magnitude_cm": correction.magnitude_cm,
            "direction": correction.direction,
            "sigma": correction.sigma,
        }
        for correction in issued
    ]


def stance_block(
    deltas: Sequence[measured_score.PointDelta], frame: measured_score.AxisFrame
) -> dict[str, object] | None:
    """자세 문장 하나. 머리가 조용하면 None이다(0이 아니다)."""
    head = next((delta for delta in deltas if delta.point == "head"), None)
    if head is None:
        return None
    stance = measured_score.head_stance(head, frame)
    if stance is None:
        return None
    return {
        "text": measured_score.stance_text(stance),
        "axis": stance.axis,
        "magnitude_cm": stance.magnitude_cm,
        "sigma": stance.sigma,
    }


def _world_layer(frames: Sequence[Any]) -> dict[str, list[Vec3 | None]]:
    """격자점마다 월드(캡처) 좌표. `_local_tracks`와 같은 필터·순서로 프레임을 훑는다.

    `_local_tracks`는 `frame.hit is None`인 프레임을 건너뛴다(그 프레임은 스트로크
    기저를 세울 원점이 없다). 여기서도 같은 조건으로 건너뛰어야 이 층의 점 개수가
    `_local_tracks`의 리샘플 수와 같다 — 그림과 채점이 같은 표본을 보게 하는 것이
    이 함수의 존재 이유다. 좌표는 `pass_frames`가 낸 `MeasuredFrame`의 것 그대로다
    (스트로크 로컬로 바꾸지 않는다) — `measured_pose` 모듈 독스트링대로 head/right/
    left/hit 넷은 이미 같은 월드 좌표계에 있다.
    """
    layer: dict[str, list[Vec3 | None]] = {"head": [], "right": [], "left": [], "hit": []}
    for frame in frames:
        if frame.hit is None:
            continue
        layer["head"].append(frame.head)
        layer["right"].append(frame.right)
        layer["left"].append(frame.left)
        layer["hit"].append(frame.hit)
    return layer


def _world_shifts(
    corrections: Sequence[measured_score.AxisCorrection],
) -> dict[str, Vec3]:
    """점 -> 월드 이동벡터. `issued_axes`가 낸 축만, `unit·magnitude_cm`만 더한다.

    `measured_score.issued_shifts`가 스트로크 로컬에서 하는 것과 정확히 같은 합산을
    월드에서 한다 — 같은 `AxisCorrection` 목록 위의 같은 규칙이다. 별도 계산 경로를
    두지 않는다: 이것이 `corrected` 층이 개선율·문장과 같은 이동을 그리는 유일한 이유다.
    """
    acc: dict[str, list[float]] = {}
    for correction in corrections:
        metres = correction.magnitude_cm / 100.0
        vec = acc.setdefault(correction.point, [0.0, 0.0, 0.0])
        for i in range(3):
            vec[i] += correction.unit[i] * metres
    return {point: (v[0], v[1], v[2]) for point, v in acc.items()}


def _shift_point(point: Vec3 | None, shift: Vec3 | None) -> Vec3 | None:
    if point is None or shift is None:
        return point
    return (point[0] + shift[0], point[1] + shift[1], point[2] + shift[2])


def _corrected_layer(
    query_layer: Mapping[str, list[Vec3 | None]],
    corrections: Sequence[measured_score.AxisCorrection],
) -> dict[str, list[Vec3 | None]]:
    """`query` 층에 발부된 축의 월드 이동만 더한다. 발부되지 않은 점(왼손·착탄점)은
    제자리다 — `issued_axes`가 "right"/"head" 외에는 절대 내지 않기 때문이다."""
    shifts = _world_shifts(corrections)
    return {
        name: [_shift_point(point, shifts.get(name)) for point in points]
        for name, points in query_layer.items()
    }


def _mean_point(points: Sequence[Vec3 | None]) -> Vec3 | None:
    present = [point for point in points if point is not None]
    if not present:
        return None
    count = len(present)
    return (
        sum(point[0] for point in present) / count,
        sum(point[1] for point in present) / count,
        sum(point[2] for point in present) / count,
    )


def _serialize_layer(layer: Mapping[str, list[Vec3 | None]]) -> dict[str, list[list[float] | None]]:
    return {
        name: [None if point is None else list(point) for point in points]
        for name, points in layer.items()
    }


def _correction_arrows(
    corrections: Sequence[measured_score.AxisCorrection],
    query_layer: Mapping[str, list[Vec3 | None]],
    frame_markers: Mapping[str, Vec3] | None = None,
) -> list[dict[str, object]]:
    """화살표 하나마다: 시작점은 그 점의 격자 평균 위치, 끝점은 **그 축 하나만**
    옮긴 자리. 한 점에 축이 여럿 발부돼도(예: 오른손 wall+vertical) 화살표끼리는
    서로의 이동을 섞지 않는다 — `to - from`이 그 축 자신의
    `magnitude_cm/100 * unit`과 정확히 같아야 한다는 요구를 지키기 위해서다.
    합산된 이동은 `corrected` 층에서 본다.

    `n`은 `corrections`(= `measured.axes`와 같은 `issued` 목록) 안의 순번이다. 평균 위치를
    못 구한 점은 화살표가 빠지므로 배열 위치가 아니라 `n`으로 짝지어야 한다.

    `from_frame`/`to_frame`은 **대표 프레임**의 그 점 실측 마커(스켈레톤이 선 바로 그
    프레임)에서 나가는 같은 이동이다. 스켈레톤은 대표 프레임이고 `from`은 패스 평균이라,
    둘을 섞어 그리면 자세가 맞아도 화살표가 손에서 떨어져 뜬다. 스켈레톤이 없으면 None.
    """
    arrows: list[dict[str, object]] = []
    for n, correction in enumerate(corrections, start=1):
        origin = _mean_point(query_layer.get(correction.point, []))
        if origin is None:
            continue
        metres = correction.magnitude_cm / 100.0
        to_point = (
            origin[0] + correction.unit[0] * metres,
            origin[1] + correction.unit[1] * metres,
            origin[2] + correction.unit[2] * metres,
        )
        marker = None if frame_markers is None else frame_markers.get(correction.point)
        arrows.append(
            {
                "point": correction.point,
                "axis": correction.axis,
                "magnitude_cm": correction.magnitude_cm,
                "direction": correction.direction,
                "sigma": correction.sigma,
                "unit": list(correction.unit),
                "from": list(origin),
                "to": list(to_point),
                "n": n,
                "from_frame": None if marker is None else list(marker),
                "to_frame": (
                    None
                    if marker is None
                    else [
                        marker[0] + correction.unit[0] * metres,
                        marker[1] + correction.unit[1] * metres,
                        marker[2] + correction.unit[2] * metres,
                    ]
                ),
            }
        )
    return arrows


# --- 스켈레톤(표시 전용) -----------------------------------------------------------

# 장착 상수 아티팩트. `trainer/scripts/fit_skeleton_mounts.py`가 사전등록
# (`docs/measurements/2026-09-15-skeleton-mounts-prereg.md`) 기준으로 적합·교차 검증해 쓴다.
MOUNTS_PATH: Final = Path(__file__).resolve().parent / "skeleton_mounts.json"
# 폴백(사전등록 §6): 대표 프레임 ±1.0 s를 뺀 나머지 ≥ 60 프레임으로 이 패스에서 적합한다.
PASS_FIT_EXCLUDE_SEC: Final = 1.0
PASS_FIT_MIN_FRAMES: Final = 60
PASS_FIT_MAX_FRAMES: Final = 400
# 투영 전 몸 기울기가 이보다 크면 뼈를 흐리게 그린다. 진단 분포(중앙값 11.9°, p90 33.4°)를
# 본 뒤 정한 표시 문턱이지 게이트가 아니다(사전등록 §8).
FADE_GRAVITY_DEG: Final = 20.0
# 파지 상한(사전등록 §8). 넘으면 "적합값이지 쥔 거리가 아니다" 고지를 싣는다.
DEFAULT_GRIP_LIMIT_M: Final[dict[str, float]] = {
    "한길숏건": 0.25,
    "반도숏건": 0.25,
    "한길폴건": 1.2,
    "반도폴건": 1.2,
}

MOUNTS_GLOBAL: Final = "global_lowo_validated"
MOUNTS_PASS_FIT: Final = "pass_fit_holdout"
FIT_METHOD: Final = "fk_csv_yxz_orient_gravity"

AVATAR_NOTE: Final = (
    "몸 모양은 시뮬레이터 아바타 추정입니다 — 실측은 머리(HMD)·건·플래쉬 세 강체이고 "
    "관절 굽힘은 측정값이 아닙니다."
)
OFFSET_LINE_LABEL: Final = "보정 오프셋(적합값)"
RESIDUAL_LABELS: Final[dict[str, str]] = {
    MOUNTS_GLOBAL: "작업자 교차 검증을 통과한 상수로 잰 어긋남",
    MOUNTS_PASS_FIT: "이 패스로 맞춘 적합 잔차(대표 프레임은 적합에서 제외)",
}
ROTATION_RESIDUAL_LABELS: Final[dict[str, str]] = {
    MOUNTS_GLOBAL: "작업자 교차 검증을 통과한 장착 회전으로 잰 강체 회전 어긋남",
    MOUNTS_PASS_FIT: "이 패스로 맞춘 장착 회전의 적합 잔차(대표 프레임은 적합에서 제외)",
}
FIT_NOTES: Final[dict[str, str]] = {
    "no_joint_columns": "이 패스에는 관절 컬럼이 없어 스켈레톤을 그리지 않습니다.",
    "no_spraying_frame": "분사 중 착탄 프레임이 없어 스켈레톤을 그리지 않습니다.",
    "insufficient_anchors": "대표 프레임에서 위치·회전이 다 잡힌 실측 강체가 2개 미만이라 스켈레톤을 세우지 않습니다.",
    "mounts_unavailable": "검증된 장착 상수가 없고 이 패스로 맞출 프레임도 60개 미만이라 스켈레톤을 그리지 않습니다.",
    "display_error": "스켈레톤을 만드는 중 오류가 나 그리지 않았습니다.",
}
# 범례는 **실측**을 모양이 아니라 색으로 선언한다. 추적 장치가 붙은 강체는 머리(HMD)·건·플래쉬 셋뿐이고,
# 아바타 추정 기하(뼈·관절점·머리·다리)는 전부 무채색이다. 옛 문구 "구체 = 실측"은 추정 머리(반경 0.09m
# 회색 구)와 추정 관절점도 구체여서, 화면에서 가장 큰 구를 실측이라고 읽게 만들었다.
LEGEND_NOTES: Final[tuple[str, ...]] = (
    "교정 화살표는 건·HMD 실측 마커에서 나갑니다",
    "실측은 머리(HMD)·건·플래쉬 세 강체뿐 — 층 색으로 채운 구체입니다",
    "아바타 추정은 모두 회색입니다 — 뼈·관절점·윤곽선 머리·점선 다리(자리표시)",
    "● 대표 프레임 실측 · ○ 패스 평균",
)

# 층마다 궤적이 **몇 줄**인지 화면이 말한다.
#
# 사용자가 "왜인지 '기준'이 세 개가 그려져"라고 보고했다. 재현해 세어 보니 쌓임이 아니다 —
# `resetScene()`이 매 `render()`마다 층을 비우고, [다른 기준 ▸]을 세 번 넘겨도 [평가]를 세 번 다시
# 눌러도 기준 층의 씬 후손 수는 64로 같았다. 원래 **세 줄**이다: `buildLayerGroup`이 층의 점 이름
# (head·right·left)마다 폴리라인을 하나씩 만든다. 세 줄이 같은 색·라벨 없이 그려지니 "기준이 세 개"로
# 읽힌 것이다. 화면은 그 사실을 말해야 한다 — 이름과 순서는 **가중치**에서 나온다(강조도 같은 곳에서
# 나온다: `skeleton_view.emphasis`가 opacity를 weight로 정한다). 지어낸 순서를 적지 않는다.
TRACK_POINT_NAMES: Final[dict[str, str]] = {"head": "머리", "right": "오른손", "left": "왼손"}
# 동률일 때의 고정 순서. 가중치가 같으면 이 차례로 적는다(실행마다 순서가 흔들리지 않게).
TRACK_POINT_ORDER: Final[tuple[str, ...]] = ("right", "head", "left")


def track_legend_note(weights: Mapping[str, float]) -> str | None:
    """층마다 궤적이 몇 줄이고 어느 줄이 무엇인지 한 줄로. 그릴 줄이 없으면 None.

    진하기가 가중치라는 것은 화면 규칙이다(`emphasis(weight).opacity = 0.25 + 0.75·weight`).
    그래서 줄 이름을 **가중치 내림차순**으로 적으면 화면에서 진한 순서와 같은 차례가 된다.
    """
    drawn = [
        (name, float(weights[name]))
        for name in TRACK_POINT_ORDER
        if isinstance(weights.get(name), (int, float)) and float(weights[name]) > 0.0
    ]
    if not drawn:
        return None
    drawn.sort(key=lambda item: (-item[1], TRACK_POINT_ORDER.index(item[0])))
    names = " · ".join(TRACK_POINT_NAMES[name] for name, _weight in drawn)
    count = len(drawn)
    return (
        f"층마다 궤적은 {count}줄입니다 — 진한 차례로 {names}입니다. "
        f"기준 층의 {count}줄은 기준 {count}개가 아니라 기준 한 사람의 {count}점입니다"
    )


# 범례가 "실측"으로 선언한 시각 부호(색). 추정 기하는 이 색을 쓰지 않는다 — 검사가 이 짝을 고정한다.
LEGEND_MEASURED_SYMBOL: Final = "층 색으로 채운 구체"
LEGEND_ESTIMATE_SYMBOL: Final = "회색"

# 대표 스켈레톤 프레임 탐색: 분사 구간 가운데에서 가까운 순서로 이만큼만 FK를 돈다(프레임당 약 0.9 ms).
LEG_SEARCH_MAX: Final = 120
# 다리 고지. 기본 표시는 곧게 선 자리표시 다리다(`skeleton.standing_legs`) — IK 다리는 툴바로 켠다.
LEGS_PLACEHOLDER_PREFIX: Final = "다리는 추적 장치가 없어 곧게 선 자리표시(점선·측정 아님)로 그렸습니다"
LEGS_HIDDEN_NOTE: Final = (
    LEGS_PLACEHOLDER_PREFIX
    + " — 이 패스의 시뮬레이터 IK 다리는 해부학 범위(무릎이 발끝 쪽으로 굽음·두 발 교차 없음 등)를 벗어나 "
    "툴바 '다리(IK)'로만 봅니다."
)
LEGS_SHOWN_NOTE: Final = LEGS_PLACEHOLDER_PREFIX + " — 시뮬레이터 IK 다리는 툴바 '다리(IK)'로 봅니다."
# 바닥면 고지(D3). 다리는 **리그 제 길이**로 내린다 — 배율을 먹이지 않는다(`skeleton.standing_legs`).
# 화면은 고른 면과 발의 관계를 숨기지 않고 그대로 말한다.
LEGS_FLOOR_BELOW_NOTE: Final = (
    " 다리는 리그 제 길이로 골반에서 아래로 내렸습니다 — 발이 벽 메시의 바닥면(수평에 가까운 면 중 "
    "가장 낮은 면, 수직 {floor:.2f}m)보다 {below:.2f}m 아래입니다."
)
LEGS_FLOOR_ABOVE_NOTE: Final = (
    " 다리는 리그 제 길이로 골반에서 아래로 내렸습니다 — 발이 벽 메시의 바닥면(수평에 가까운 면 중 "
    "가장 낮은 면, 수직 {floor:.2f}m)보다 {above:.2f}m 위입니다."
)
LEGS_NO_FLOOR_NOTE: Final = " 다리는 리그 제 길이로 골반에서 아래로 내렸습니다 — 이 레벨은 벽 메시가 없어 바닥면을 고르지 않았습니다."
# 고른 면이 **작업자가 선 지면인지**를 보는 사람이 판단할 수 있게, 그 면 위 머리 실측 마커 높이를 같이 적는다.
# 발이 그 면보다 얼마나 아래인지가 레벨마다 갈리는 것은 리그 몸통 길이 때문이 아니다 — 골반→머리는
# 리그에서 유도한 값이라 모든 레벨이 같다. 갈리는 것은 골반–이 면 거리뿐이고, 실측으로
# L1 1.097m · L3 0.269m · L6 0.454m이다. 같은 자리에서 잰
# 머리 실측 마커는 L1 1.508m · L3 0.696m · L6 0.879m — L3·L6는 서 있는 사람의 머리 높이가 아니다.
# 문턱은 두지 않는다. 실측 높이를 그대로 적고 판단은 보는 사람에게 남긴다(`composite.py`의 바닥 주석 참고).
LEGS_FLOOR_HEAD_NOTE: Final = (
    " 이 면 위 머리 실측 마커는 {head:.2f}m입니다 — 서 있는 사람의 머리 높이가 아니면 고른 면이 "
    "작업자가 선 지면이 아니라는 뜻이고, 발이 그 면을 지나 내려간 것도 그래서입니다."
)
LEGS_PICKED_NOTE: Final = (
    LEGS_PLACEHOLDER_PREFIX
    + " — 툴바 '다리(IK)'의 IK 다리가 해부학 범위 안인 프레임(분사 구간 가운데에서 {k}프레임)을 대표로 골랐습니다."
)
# 자리표시 다리를 못 세운 경우(엉덩이 관절·다리 오프셋 없음)의 옛 고지.
LEGS_IK_ONLY_NOTE: Final = "다리는 추적 장치 없이 시뮬레이터 IK가 지은 자세라 흐리게 그렸습니다."
# fit.notes 항목 이름(`fit.note_keys`, 같은 순서). 화면은 위치가 아니라 이 이름으로 짝짓는다.
NOTE_KEYS: Final[tuple[str, ...]] = ("avatar_estimate", "gravity", "legs", "torso", "grip_limit")
# 캔버스 아래에서 접어 두는 고지. 파지 상한 고지는 3D 보정 오프셋 선 위 라벨(`offset_display`)이 같은 사실을
# 장면 안에서 말하므로 접는다 — 접힌 고지도 한 번 누르면 그 자리에서 펼쳐진다.
NOTES_COLLAPSED: Final[tuple[str, ...]] = ("grip_limit",)


@dataclass(frozen=True, slots=True)
class MountsTable:
    passed: bool
    table: dict[str, skeleton.AnchorMount]
    grip_limit_m: dict[str, float]


def _mount_from_json(entry: Mapping[str, Any]) -> skeleton.AnchorMount:
    rotation = entry["rotation"]
    offset = entry["offset"]
    return skeleton.AnchorMount(
        rotation=tuple(tuple(float(x) for x in row) for row in rotation),  # type: ignore[arg-type]
        offset=(float(offset[0]), float(offset[1]), float(offset[2])),
    )


@functools.lru_cache(maxsize=4)
def _load_mounts_cached(path_text: str, _mtime: float) -> MountsTable:
    data = json.loads(Path(path_text).read_text(encoding="utf-8"))
    passed = bool(data.get("validation", {}).get("passed", False))
    mounts = data.get("mounts", {})
    table: dict[str, skeleton.AnchorMount] = {}
    for point in ("head", "left"):
        if mounts.get(point):
            table[point] = _mount_from_json(mounts[point])
    for group, entry in (mounts.get("right") or {}).items():
        table[skeleton.mount_key("right", group)] = _mount_from_json(entry)
    limits = dict(DEFAULT_GRIP_LIMIT_M)
    limits.update({k: float(v) for k, v in (data.get("grip_limit_m") or {}).items()})
    return MountsTable(passed=passed, table=table, grip_limit_m=limits)


def load_mounts(path: Path = MOUNTS_PATH) -> MountsTable:
    """장착 상수. 파일이 없으면 검증되지 않은 것으로 본다(`passed=False`)."""
    try:
        mtime = path.stat().st_mtime
    except FileNotFoundError:
        return MountsTable(passed=False, table={}, grip_limit_m=dict(DEFAULT_GRIP_LIMIT_M))
    return _load_mounts_cached(str(path), mtime)


@dataclass(frozen=True, slots=True)
class Representative:
    """대표 프레임(분사 중·착탄점 있는 구간의 가운데)과 그 원본 행."""

    frame: Any
    row: dict[str, str]
    gun: str
    spraying: list[tuple[Any, dict[str, str]]]


def representative_frame(raw: RawPass) -> Representative | None:
    """회전 각은 원본 CSV 행에만 있고 리샘플 격자에는 없다 — 그래서 `pass_frames`가 아니라
    `measured_pose.measured_frames_with_rows`로 원본 행과 프레임을 짝지어, 분사 중이고
    착탄점이 있는 구간의 **가운데** 프레임을 고른다(`representative_basis`와 같은 규칙)."""
    if not raw.rows:
        return None
    gun = measured_pose.active_gun(raw.rows)
    if gun is None:
        return None
    paired = measured_pose.measured_frames_with_rows(raw.rows)
    spraying = [(frame, row) for frame, row in paired if frame.spraying and frame.hit is not None]
    if not spraying:
        return None
    frame, row = spraying[len(spraying) // 2]
    return Representative(frame=frame, row=row, gun=gun, spraying=spraying)


@dataclass(frozen=True, slots=True)
class SkeletonFrame:
    """스켈레톤을 세울 프레임. `offset_frames`는 분사 구간 가운데 프레임에서 떨어진 프레임 수다."""

    representative: Representative
    sample: skeleton.FrameSample
    legs: skeleton.LegCheck | None
    offset_frames: int
    searched: int


def skeleton_frame(representative: Representative, offsets: Mapping[str, Vec3]) -> SkeletonFrame | None:
    """분사 구간 **가운데에서 가까운 순서로** 최대 `LEG_SEARCH_MAX` 프레임을 보고, IK 다리가 해부학
    범위 안(`skeleton.leg_check`)인 첫 프레임을 고른다. 없으면 가운데 프레임(다리는 화면이 숨긴다).

    다리는 측정값이 아니다 — 이 선택은 실측 마커·교정 수치를 바꾸지 않고, 고른 프레임의 실측 마커가
    그대로 화살표 시작점이 된다(`from_frame`). 무엇을 골랐는지는 `fit.legs`와 notes로 드러낸다.
    """
    spraying = representative.spraying
    if not spraying or not offsets:
        return None
    middle = len(spraying) // 2
    order = sorted(range(len(spraying)), key=lambda i: (abs(i - middle), i))[:LEG_SEARCH_MAX]
    first: tuple[int, skeleton.FrameSample, skeleton.LegCheck | None] | None = None
    for count, index in enumerate(order, 1):
        frame, row = spraying[index]
        sample = skeleton.frame_sample(frame, row, representative.gun, offsets)
        if sample is None:
            continue
        legs = skeleton.leg_check(sample.local_pos)
        if first is None:
            first = (index, sample, legs)
        if legs is not None and legs.plausible:
            chosen = Representative(frame=frame, row=row, gun=representative.gun, spraying=spraying)
            return SkeletonFrame(chosen, sample, legs, abs(index - middle), count)
    if first is None:
        return None
    index, sample, legs = first
    frame, row = spraying[index]
    chosen = Representative(frame=frame, row=row, gun=representative.gun, spraying=spraying)
    return SkeletonFrame(chosen, sample, legs, abs(index - middle), len(order))


def pass_fit_samples(
    representative: Representative, offsets: Mapping[str, Vec3]
) -> list[skeleton.FrameSample]:
    """폴백 적합 재료: 대표 프레임 시각 ±`PASS_FIT_EXCLUDE_SEC`를 **뺀** 분사 프레임.

    대표 프레임을 적합에 넣으면 그 프레임의 어긋남이 표본 안 잔차가 되어, 맞춘 것이 맞은 것처럼
    보인다. 개수 문턱은 뺀 뒤의 수로 판정한다."""
    centre = float(representative.frame.time_sec)
    keep = [
        (frame, row)
        for frame, row in representative.spraying
        if abs(float(frame.time_sec) - centre) > PASS_FIT_EXCLUDE_SEC
    ]
    if len(keep) < PASS_FIT_MIN_FRAMES:
        return []
    if len(keep) > PASS_FIT_MAX_FRAMES:
        step = (len(keep) - 1) / (PASS_FIT_MAX_FRAMES - 1)
        keep = [keep[round(i * step)] for i in range(PASS_FIT_MAX_FRAMES)]
    samples = [skeleton.frame_sample(frame, row, representative.gun, offsets) for frame, row in keep]
    return [sample for sample in samples if sample is not None]


def place_frame(sample: skeleton.FrameSample, mounts: skeleton.Mounts) -> skeleton.Placement | None:
    """한 프레임 배치. 강체 변환만 — 관절을 실측 좌표로 덮어쓰지 않는다(뼈 길이 보존)."""
    return skeleton.place_oriented(sample.local_pos, sample.local_rot, sample.measured, mounts)


@dataclass(frozen=True, slots=True)
class SkeletonResult:
    joints: dict[str, list[float]] | None
    bones: list[list[str]] | None
    fit: dict[str, object] | None
    fit_reason: str | None
    fit_note: str | None
    markers: dict[str, Vec3]
    representative: Representative | None


def _skeleton_failure(reason: str, representative: Representative | None = None) -> SkeletonResult:
    return SkeletonResult(
        joints=None,
        bones=None,
        fit=None,
        fit_reason=reason,
        fit_note=FIT_NOTES[reason],
        markers={},
        representative=representative,
    )


def _head_above_m(placed: skeleton.Placement, floor: float) -> float | None:
    """고른 면 위 **머리 실측 마커(HMD)**의 높이. 없으면 None — 지어내지 않는다."""
    for fit in placed.anchors:
        if fit.point == "head":
            return world_axes.vertical_of(tuple(fit.measured_marker)) - floor
    return None


def _torso_length_m(offsets: Mapping[str, Vec3]) -> float:
    return body_proportions.torso_length_m(offsets)


def _fit_view(
    placed: skeleton.Placement,
    representative: Representative,
    kind: str,
    mounts: skeleton.Mounts,
    offsets: Mapping[str, Vec3],
    grip_limit_m: Mapping[str, float],
    *,
    chosen: SkeletonFrame | None = None,
    floor: float | None = None,
) -> dict[str, object]:
    gravity = placed.gravity_correction_deg
    faded = gravity > FADE_GRAVITY_DEG
    notes = [AVATAR_NOTE]
    note_keys = ["avatar_estimate"]
    if faded:
        notes.append(f"실측 회전과 수직 가정이 {gravity:.0f}° 어긋나 몸을 중력 방향으로 세워 그렸습니다(흐리게 표시).")
    else:
        notes.append(f"몸의 수직은 중력 방향으로 맞춘 값입니다(보정 {gravity:.0f}°).")
    note_keys.append("gravity")
    legs = None if chosen is None else chosen.legs
    hit = getattr(representative.frame, "hit", None)
    pelvis = placed.joints.get("Pelvis")
    forward = None if hit is None or pelvis is None else [hit[i] - pelvis[i] for i in range(3)]
    placeholder = (
        skeleton.standing_legs(placed.joints, offsets, forward, floor=floor) if legs is not None else None
    )
    if legs is not None:
        if placeholder is None:
            notes.append(LEGS_IK_ONLY_NOTE)
        else:
            if not legs.plausible:
                note = LEGS_HIDDEN_NOTE
            elif chosen is not None and chosen.offset_frames > 0:
                note = LEGS_PICKED_NOTE.format(k=chosen.offset_frames)
            else:
                note = LEGS_SHOWN_NOTE
            # 고른 면과 발의 관계는 같은 줄에서 말한다 — note_keys가 "legs" 하나로 남는다.
            below = placeholder.feet_below_floor_m
            if placeholder.floor_vertical is None or below is None:
                note += LEGS_NO_FLOOR_NOTE
            elif below >= 0.0:
                note += LEGS_FLOOR_BELOW_NOTE.format(floor=placeholder.floor_vertical, below=below)
            else:
                note += LEGS_FLOOR_ABOVE_NOTE.format(floor=placeholder.floor_vertical, above=-below)
            # 면을 골랐으면 그 면 위 머리 실측 높이를 같이 적는다.
            if placeholder.floor_vertical is not None:
                head_above = _head_above_m(placed, placeholder.floor_vertical)
                if head_above is not None:
                    note += LEGS_FLOOR_HEAD_NOTE.format(head=head_above)
            notes.append(note)
        note_keys.append("legs")
    notes.append(
        f"이 CSV 리그에는 목·윗가슴 관절이 없어 골반→머리를 같은 작업자 BVH의 뼈 길이 비례로 다시 재 "
        f"{_torso_length_m(offsets):.2f}m로 그리고, 빠진 목·윗가슴을 그 선분 위에 그려 넣었습니다 — "
        "실측이 아니라 추정입니다."
    )
    note_keys.append("torso")
    right_norm = None if mounts.right is None else math.hypot(*mounts.right.offset)
    limit = grip_limit_m.get(representative.gun)
    if right_norm is not None and limit is not None and right_norm > limit:
        notes.append(
            f"건 마커–손 관절 보정 거리 {right_norm:.2f}m는 손으로 쥔 거리로 설명되지 않습니다 — "
            "아바타 손과 건 강체의 어긋남을 흡수한 적합값입니다."
        )
        note_keys.append("grip_limit")
    prefix = "적합 잔차 " if kind == MOUNTS_PASS_FIT else ""
    frame = representative.frame
    return {
        "method": FIT_METHOD,
        "mounts": kind,
        # 추정 머리 구체 지름(m). 고정 상수가 아니라 이 리그의 골반→머리 길이 × BVH 실측 비례에서
        # 나온다(`body_proportions.head_diameter_m`) — 화면은 이 값을 그대로 쓰고 다시 계산하지 않는다.
        "head_diameter_m": body_proportions.head_diameter_m(offsets),
        "mounts_detail": {
            "right": None if mounts.right is None else {"group": representative.gun, "offset_norm_m": right_norm},
            "head": None if mounts.head is None else {"offset_norm_m": math.hypot(*mounts.head.offset)},
            "left": None if mounts.left is None else {"offset_norm_m": math.hypot(*mounts.left.offset)},
        },
        "frame_index": int(frame.index),
        "time_sec": float(frame.time_sec),
        "residual_m": placed.residual_m,
        "residual_label": RESIDUAL_LABELS[kind],
        "gravity_correction_deg": gravity,
        "gravity_display": f"수직 보정 {gravity:.0f}°",
        "faded": faded,
        "leg_tilt_deg": placed.leg_tilt_deg,
        "rotation_residual_deg": dict(placed.rotation_residual_deg),
        "rotation_residual_label": ROTATION_RESIDUAL_LABELS[kind],
        "offset_line_label": OFFSET_LINE_LABEL,
        "anchors": [
            {
                "point": fit.point,
                "joint": fit.joint,
                "joint_position": list(fit.joint_position),
                "predicted_marker": list(fit.predicted_marker),
                "measured_marker": list(fit.measured_marker),
                "miss_cm": fit.miss_m * 100.0,
                "miss_display": f"{prefix}{fit.miss_m * 100.0:.1f}cm",
                # 파지 상한을 넘는 보정 오프셋에만 3D 선 위 라벨을 단다 — 손과 건 마커가 떨어진 이유가
                # 그 선에서 바로 읽히게(notes의 파지 고지와 같은 조건).
                "offset_display": (
                    f"보정 오프셋 {right_norm:.2f}m(적합값)"
                    if fit.point == "right" and right_norm is not None and limit is not None and right_norm > limit
                    else None
                ),
            }
            for fit in placed.anchors
        ],
        "notes": notes,
        "note_keys": note_keys,
        "notes_collapsed": [key for key in NOTES_COLLAPSED if key in note_keys],
        "legs": None
        if legs is None or chosen is None
        else {
            "plausible": legs.plausible,
            "joints": list(skeleton.LEG_JOINTS),
            "checks": legs.as_dict(),
            "limits": dict(skeleton.LEG_LIMITS),
            "frame_offset": chosen.offset_frames,
            "searched_frames": chosen.searched,
            # 기본 표시: 곧게 선 자리표시 다리(측정 아님). 못 세우면 범위 안 IK 다리, 그것도 아니면 숨김.
            "display_default": "placeholder" if placeholder is not None else ("ik" if legs.plausible else "hidden"),
            "placeholder_joints": None
            if placeholder is None
            else {k: list(v) for k, v in placeholder.joints.items()},
            "placeholder_bones": None
            if placeholder is None
            else [["Pelvis", hip] for hip, *_ in skeleton.PLACEHOLDER_LEG_CHAIN]
            + [
                [a, b]
                for chain in skeleton.PLACEHOLDER_LEG_CHAIN
                for a, b in zip(chain, chain[1:])
                if a in placeholder.joints and b in placeholder.joints
            ],
            # 바닥 맞춤(D3). 값이 있어도 자리표시는 측정이 아니다.
            # `placeholder_feet`는 바닥에 세운 **발 관절의 이름**이다 — 화면이 "발이 지금 보이는가"를
            # 재려면 어느 관절이 발인지 알아야 하는데, 리그 이름을 화면에서 지어내면 출처와 다른 라벨이 된다.
            "placeholder_feet": None
            if placeholder is None
            else [ankle for _hip, _knee, ankle, _toe in skeleton.PLACEHOLDER_LEG_CHAIN if ankle in placeholder.joints],
            "floor_vertical": None if placeholder is None else placeholder.floor_vertical,
            # 자리표시 발이 고른 바닥면보다 얼마나 아래인가(음수면 위). 다리 길이는 리그 그대로다 —
            # 배율은 더 이상 없다.
            "feet_below_floor_m": None if placeholder is None else placeholder.feet_below_floor_m,
        },
        # IK 다리를 기본으로 켜는가. 자리표시 다리가 있으면 IK 다리는 툴바로만 켠다.
        "legs_visible_default": bool(legs is not None and legs.plausible and placeholder is None),
    }


def skeleton_block(
    raw: RawPass,
    representative: Representative | None = None,
    mounts_table: MountsTable | None = None,
    geometry: composite.MeshGeometry | None = None,
) -> SkeletonResult:
    """대표 프레임 하나에 전신 FK 스켈레톤을 세운다(`skeleton.place_oriented`, M6).

    옛 경로는 머리·양손 세 점 위치에 Kabsch를 풀고 그 뒤 머리·양손을 실측으로 **덮어써**
    목·전완이 0.46~2.1배로 늘고 줄었다. 여기서는 실측 강체 회전으로 루트를 세우고 중력으로
    투영한 뒤 강체 변환만 한다 — 뼈 길이는 리그 그대로이고, 대신 앵커마다 어긋남을 싣는다.

    장착 상수는 작업자 교차 검증을 통과한 전역 값(`skeleton_mounts.json`, `passed=true`이고
    이 패스의 건 그룹 키가 있을 때)을 쓰고, 아니면 이 패스에서 대표 프레임 ±1.0 s를 뺀
    프레임으로 맞춘다(`pass_fit_samples`). 그마저 모자라면 스켈레톤을 싣지 않는다.
    """
    # 몸통 세 마디를 BVH 실측 비례 길이로 다시 재고(`body_proportions.rig_offsets`), 아래에서
    # 빠진 `Spine1`·`Neck`을 그려 넣는다. FK보다 **먼저** 바꿔야 관절 위치가 같이 따라온다 —
    # 배치(`place_oriented`)는 강체 변환뿐이라 여기서 바꾼 뼈 길이를 그대로 유지한다.
    offsets = body_proportions.rig_offsets(pose_fk_csv.bind_offsets(raw.rows[0])) if raw.rows else {}
    representative = representative if representative is not None else representative_frame(raw)
    if not offsets:
        return _skeleton_failure("no_joint_columns", representative)
    if representative is None:
        return _skeleton_failure("no_spraying_frame")
    chosen = skeleton_frame(representative, offsets)
    if chosen is None:
        return _skeleton_failure("no_joint_columns", representative)
    representative = chosen.representative
    sample = chosen.sample

    table = mounts_table if mounts_table is not None else load_mounts()
    global_mounts = skeleton.mounts_for(table.table, representative.gun)
    if table.passed and global_mounts.right is not None and global_mounts.head is not None:
        kind = MOUNTS_GLOBAL
        mounts = global_mounts
    else:
        samples = pass_fit_samples(representative, offsets)
        if not samples:
            return _skeleton_failure("mounts_unavailable", representative)
        kind = MOUNTS_PASS_FIT
        mounts = skeleton.mounts_for(skeleton.fit_mounts([samples]), representative.gun)

    placed = place_frame(sample, mounts)
    if placed is None:
        return _skeleton_failure("insufficient_anchors", representative)

    # 그려 넣는 두 관절은 **선분 위 내분점**이라 배치 뒤에 계산해도 같다(방향을 지어내지 않는다).
    joints = dict(placed.joints)
    joints.update(body_proportions.inserted_joints(joints))
    return SkeletonResult(
        joints={name: list(position) for name, position in joints.items()},
        bones=body_proportions.display_bones(simulator_csv.BONES, joints),
        fit=_fit_view(
            placed,
            representative,
            kind,
            mounts,
            offsets,
            table.grip_limit_m,
            chosen=chosen,
            floor=composite.floor_vertical(geometry),
        ),
        fit_reason=None,
        fit_note=None,
        markers={fit.point: fit.measured_marker for fit in placed.anchors},
        representative=representative,
    )


def _mesh_source_path(level: int) -> Path | None:
    """정렬 레코드(`level_alignment.json`)에 적힌 원본 FBX 경로. 항목 자체가 없으면
    None이다 — 하드코딩한 파일명 규칙을 두지 않고 레코드가 적어 둔 `source_fbx`를
    그대로 읽는다(`level_mesh._entry_from_alignment`가 적는 바로 그 필드)."""
    try:
        entries = json.loads(level_mesh.ALIGNMENT_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    for entry in entries:
        if int(entry.get("level", -1)) == level:
            source = entry.get("source_fbx")
            return None if not source else level_mesh.DEFAULT_MODELINGS_ROOT / str(source)
    return None


_NO_MESH: Final[dict[str, object]] = {"vertices": [], "triangles": []}

MESH_REASON_TEXT: Final[dict[str, str]] = {
    "unverified_alignment": "이 레벨은 아직 검증된 벽면 정렬이 없어 벽면을 그리지 않습니다.",
    "no_alignment_record": "이 레벨의 벽면 정렬 기록이 없어 벽면을 그리지 않습니다.",
    "fbx_missing": "벽면 모델 파일을 찾지 못해 벽면을 그리지 않습니다.",
    "display_error": "벽면을 만드는 중 오류가 나 그리지 않았습니다.",
}
HANDEDNESS_NOTE: Final = (
    "벽 켜기·착탄 거리로는 손대칭을 판별하지 못합니다 — 판정 근거는 건·HMD 가림 비율입니다."
)


def _alignment_entry(level: int) -> dict[str, Any] | None:
    try:
        entries = json.loads(level_mesh.ALIGNMENT_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    for entry in entries:
        if int(entry.get("level", -1)) == level:
            return entry
    return None


@dataclass(frozen=True, slots=True)
class MeshResult:
    level: int
    mesh: Any | None  # level_mesh.Mesh (배치 후)
    reason: str | None
    entry: dict[str, Any] | None


def placed_mesh(level_text: str) -> MeshResult | None:
    """검증된("solved") 정렬이 있는 레벨만 배치한 메시를 낸다(설계 §5).

    None("이 패스의 레벨 자체를 모른다" — 예: 실전모드 `1-A`)과 `mesh is None`("레벨은 알지만
    그릴 기하가 없다")을 구분하고, 후자는 사유를 함께 낸다.
    """
    if not level_text.isdigit():
        return None
    level = int(level_text)
    try:
        alignment = level_mesh.load_alignment(level)
    except LookupError:
        return MeshResult(level=level, mesh=None, reason="no_alignment_record", entry=None)
    entry = _alignment_entry(level)
    if not level_mesh.is_solved(alignment.provenance):
        return MeshResult(level=level, mesh=None, reason="unverified_alignment", entry=entry)
    source = _mesh_source_path(level)
    if source is None:
        return MeshResult(level=level, mesh=None, reason="fbx_missing", entry=entry)
    try:
        mesh = level_mesh.read_fbx(source)
    except FileNotFoundError:
        # 정렬 레코드는 있고 경로도 적혀 있지만 그 경로에 실제 FBX 파일이 없다
        # (배포 환경에서 데이터 마운트가 일시적으로 빠진 경우 등). 장식용 벽면
        # 하나 때문에 점수·교정까지 담긴 /api/compare 전체를 500으로 죽이지
        # 않는다 — `available: False`로 조용히 떨어뜨린다. 파일은 있는데 형식이
        # 깨진 경우(ValueError 등, `read_fbx`가 그대로 던진다)는 버그이므로 계속
        # 크게 터뜨린다 — 이 except는 "파일 없음" 하나만 잡는다.
        return MeshResult(level=level, mesh=None, reason="fbx_missing", entry=entry)
    return MeshResult(level=level, mesh=level_mesh.place(mesh, alignment), reason=None, entry=entry)


def _solved_levels() -> list[int]:
    try:
        entries = json.loads(level_mesh.ALIGNMENT_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return []
    return sorted(int(e["level"]) for e in entries if level_mesh.is_solved(str(e.get("provenance", ""))))


def _mesh_reason_detail(entry: Mapping[str, Any]) -> str | None:
    """정렬 기록이 **왜** 미검증인지 기록에서 읽는다(레벨 번호로 분기하지 않는다)."""
    attempt = entry.get("front_face_attempt") or entry.get("fixed_linear_attempt")
    if not isinstance(attempt, Mapping):
        return None
    passed = attempt.get("candidates_passed_A")
    if passed == 1:
        return (
            "위치·표면 검증을 통과한 정렬은 하나로 좁혔지만, 벽의 앞뒤 뒤집힘(손대칭)을 사전등록한 검사로 "
            "확정하지 못했습니다."
        )
    if passed == 0:
        return "위치·표면 검증(모델 UV 잔차)을 통과한 정렬 후보가 없습니다."
    return None


def _evidence_display(evidence: Mapping[str, Any] | None) -> str | None:
    if not isinstance(evidence, Mapping):
        return None
    parts = []
    for key, label in (("gun_occlusion", "건 가림"), ("hmd_back_face", "HMD 뒤편 비율")):
        pair = evidence.get(key)
        if isinstance(pair, Mapping) and isinstance(pair.get("P"), (int, float)) and isinstance(pair.get("mirror"), (int, float)):
            parts.append(f"{label} 채택 {pair['P']:.3f} / 거울 {pair['mirror']:.3f}")
    return " · ".join(parts) if parts else None


def mesh_view(result: MeshResult | None) -> dict[str, object] | None:
    if result is None:
        return None
    entry = result.entry or {}
    evidence = entry.get("handedness_evidence")
    view: dict[str, object] = {"level": result.level, "available": result.mesh is not None}
    if result.mesh is None:
        view.update(_NO_MESH)
    else:
        view["vertices"] = [list(vertex) for vertex in result.mesh.vertices]
        view["triangles"] = [list(triangle) for triangle in result.mesh.triangles]
    view.update(
        {
            "reason": result.reason,
            "reason_text": None if result.reason is None else MESH_REASON_TEXT[result.reason],
            "reason_detail": _mesh_reason_detail(entry) if result.reason == "unverified_alignment" else None,
            "reason_hint": (
                "벽면이 검증된 레벨: " + "·".join(str(level) for level in _solved_levels())
                if result.reason == "unverified_alignment" and _solved_levels()
                else None
            ),
            "handedness_source": entry.get("handedness_source"),
            "handedness_evidence": evidence,
            "handedness_note": HANDEDNESS_NOTE if evidence else None,
            "handedness_evidence_display": _evidence_display(evidence) if result.mesh is not None else None,
        }
    )
    return view


def mesh_block(level_text: str) -> dict[str, object] | None:
    """검증된("solved") 정렬이 있는 레벨만 기하를 싣는다(설계 §5).

    `mesh`가 `None`인 것("이 패스의 레벨 자체를 모른다" — 예: 실전모드 `1-A`처럼
    숫자가 아닌 레벨)과 `available: False`인 것("레벨은 알지만 검증된 정렬이 없다")을
    구분한다 — 소비자가 두 상태를 다르게 다뤄야 한다(설계 §5). 파싱 결함(`read_fbx`의
    ValueError)은 여기서 삼키지 않는다 — 격리는 `compare` 층에서만, 사유를 따로 붙여 한다.
    """
    return mesh_view(placed_mesh(level_text))


def _mesh_error_view(level_text: str) -> dict[str, object] | None:
    if not level_text.isdigit():
        return None
    return {
        "level": int(level_text),
        "available": False,
        **_NO_MESH,
        "reason": "display_error",
        "reason_text": MESH_REASON_TEXT["display_error"],
        "reason_detail": None,
        "reason_hint": None,
        "handedness_source": None,
        "handedness_evidence": None,
        "handedness_note": None,
        "handedness_evidence_display": None,
    }


_UNSET: Final = object()


def view_block(
    query_frames: Sequence[Any],
    reference_frames: Sequence[Any],
    deltas: Sequence[measured_score.PointDelta] | None = None,
    basis: measured_score.AxisFrame | None = None,
    weights: Mapping[str, float] = measured_score.DEFAULT_WEIGHTS,
    level_text: str = "",
    grid_n: int = stroke_frame.DEFAULT_SAMPLES,
    query_raw: RawPass | None = None,
    *,
    corrections: Sequence[measured_score.AxisCorrection] | None = None,
    skeleton_result: SkeletonResult | None = None,
    mesh: object = _UNSET,
) -> dict[str, object]:
    """화면이 그릴 층·교정·메시·가중치. 좌표는 전부 캡처(월드) 프레임이다.

    `corrected` 층과 `corrections`는 **한 목록**(`corrections`, 없으면 여기서 한 번
    `measured_score.issued_axes`)에서 파생된다 — 두 갈래로 나눠 계산하면 그림이 개선율·문장과
    다른 이동을 보일 수 있다(요구사항 1). `compare()`는 `measured.axes`와 같은 `issued`를 넘긴다.

    전신 스켈레톤은 **질의 층에만** 얹는다. "지금 자세가 기준과 얼마나 다른가"가 이 화면의
    답이므로 검증 대상은 질의다. `query_raw`도 `skeleton_result`도 없으면(테스트처럼 원본 행
    없이 층만 조립하는 호출) 스켈레톤 없이 점·화살표만 낸다.
    """
    if corrections is None:
        if deltas is None or basis is None:
            raise ValueError("view_block: corrections 또는 (deltas, basis)가 필요하다")
        corrections = measured_score.issued_axes(deltas, basis)
    query_layer = _world_layer(query_frames)
    reference_layer = _world_layer(reference_frames)
    corrected_layer = _corrected_layer(query_layer, corrections)

    query_layer_view: dict[str, object] = dict(_serialize_layer(query_layer))
    weights_view = dict(weights)
    if skeleton_result is None and query_raw is not None:
        skeleton_result = skeleton_block(query_raw)
    fit: dict[str, object] | None = None
    fit_reason: str | None = None
    fit_note: str | None = None
    markers: Mapping[str, Vec3] | None = None
    if skeleton_result is not None:
        fit_reason = skeleton_result.fit_reason
        fit_note = skeleton_result.fit_note
        if skeleton_result.joints is not None:
            query_layer_view["joints"] = skeleton_result.joints
            query_layer_view["bones"] = skeleton_result.bones
            fit = skeleton_result.fit
            markers = skeleton_result.markers
            # 새 관절은 전부 weights에 넣고 값은 0.0이다(채점에 안 들어간다) — 브리프.
            # 강조(emphasis)는 view.weights[name]에서만 파생되므로, 여기 없으면 화면이
            # 이 관절을 "그리는 점인데 weights가 모른다"는 결함 상태로 받는다.
            for name in skeleton_result.joints:
                weights_view.setdefault(name, 0.0)

    return {
        "grid_n": grid_n,
        "layers": {
            "query": query_layer_view,
            "reference": _serialize_layer(reference_layer),
            "corrected": _serialize_layer(corrected_layer),
        },
        "corrections": _correction_arrows(corrections, query_layer, markers),
        "mesh": mesh_block(level_text) if mesh is _UNSET else mesh,
        "weights": weights_view,
        "fit": fit,
        "fit_reason": fit_reason,
        "fit_note": fit_note,
        "legend_notes": [
            note
            for note in (*LEGEND_NOTES, track_legend_note(weights_view))
            if isinstance(note, str)
        ],
        "display_errors": [],
        "composite": None,
        "surface_hits": [],
        "stroke": composite.stroke_runs(None, []),
    }


# --- 표시 블록 격리 ----------------------------------------------------------------

_TEXT_FIELDS: Final[tuple[str, ...]] = ("n", "text", "text_reason", "text_mode", "figure", "sigma_display", "trace")


def pass_surface_angle_deg(
    raw: RawPass, geometry: composite.MeshGeometry | None, basis: measured_score.AxisFrame
) -> float | None:
    """스트로크 기저 법선과 패스 착탄점들의 최근접 삼각형 법선(건 쪽 부호) 평균 사이 각."""
    if geometry is None:
        return None
    paired = [
        (frame, row)
        for frame, row in measured_pose.measured_frames_with_rows(raw.rows)
        if frame.spraying and frame.hit is not None
    ]
    if not paired:
        return None
    if len(paired) > orientation.MAX_FRAMES:
        step = (len(paired) - 1) / (orientation.MAX_FRAMES - 1)
        paired = [paired[round(i * step)] for i in range(orientation.MAX_FRAMES)]
    mean = composite.surface_normal_mean(
        geometry, [frame.hit for frame, _row in paired], [frame.right for frame, _row in paired]
    )
    if mean is None:
        return None
    return composite.angle_between_lines_deg(basis.normal, mean)


def instruction_entries(
    issued: Sequence[measured_score.AxisCorrection], surface_angle_deg: float | None
) -> list[dict[str, object]]:
    """`measured.axes`와 같은 순서의 표시 필드. 문장이 표에 없으면 text=None, text_reason='no_phrase'."""
    entries: list[dict[str, object]] = []
    for n, correction in enumerate(issued, start=1):
        entry: dict[str, object] = {"n": n, **instructions.axis_display(correction, surface_angle_deg)}
        try:
            text, mode = instructions.axis_text(correction, surface_angle_deg)
            entry.update({"text": text, "text_reason": None, "text_mode": mode})
        except ValueError:
            entry.update({"text": None, "text_reason": "no_phrase", "text_mode": "directive"})
        entries.append(entry)
    return entries


def measured_display(measured: Mapping[str, Any]) -> dict[str, object]:
    improvement_values = measured.get("improvement") or {}
    return {
        "improvement_labels": dict(instructions.IMPROVEMENT_LABELS),
        "display": {
            "score": instructions.score_display(measured.get("score")),
            "score_after": instructions.score_display(measured.get("score_after")),
            "improvement_same": instructions.score_display(improvement_values.get("same")),
            "improvement_split": instructions.score_display(improvement_values.get("split")),
        },
        "deltas_display": [
            {
                "point": delta["point"],
                "deviation_display": instructions.deviation_display(delta["deviation_cm"]),
                "sigma_display": instructions.sigma_display(delta["sigma"]),
            }
            for delta in measured.get("deltas", [])
        ],
    }


def _attach_view(
    body: dict[str, Any],
    *,
    query: PassRef,
    query_raw: RawPass,
    reference_raw: RawPass,
    query_frames: Sequence[Any],
    reference_frames: Sequence[Any],
    issued: Sequence[measured_score.AxisCorrection],
    basis: measured_score.AxisFrame,
    weights: Mapping[str, float],
    samples: int,
) -> None:
    """채점이 끝난 응답에 표시 블록을 붙인다. 블록마다 예외를 격리한다.

    표시 코드의 결함이 점수·교정·개선율 응답을 인질로 잡으면 안 된다. 블록이 던지면 로그를
    남기고 그 블록만 null(메시는 사유 'display_error')로 두고 `view.display_errors`에 적는다.
    `measured`의 기존 필드는 이 함수가 건드리지 않는다 — 새 표시 필드만 더한다.
    """
    errors: list[dict[str, str]] = []

    def guarded(block: str, fn: Callable[..., Any], *args: Any) -> tuple[bool, Any]:
        try:
            return True, fn(*args)
        except Exception as exc:  # noqa: BLE001 — 표시 블록 격리가 이 함수의 목적이다.
            logger.exception("display block %s failed", block)
            errors.append({"block": block, "error_type": type(exc).__name__})
            return False, None

    measured = body["measured"]

    ok, mesh_result = guarded("mesh", placed_mesh, query.level)
    mesh = mesh_view(mesh_result) if ok else _mesh_error_view(query.level)
    geometry: composite.MeshGeometry | None = None
    if ok and mesh_result is not None and mesh_result.mesh is not None:
        ok, geometry = guarded(
            "mesh", composite.MeshGeometry.from_mesh, mesh_result.mesh.vertices, mesh_result.mesh.triangles
        )
        if not ok:
            mesh = _mesh_error_view(query.level)
            geometry = None

    rep_ok, representative = guarded("skeleton", representative_frame, query_raw)
    skeleton_ok = False
    skeleton_result: SkeletonResult | None = None
    if rep_ok:
        skeleton_ok, skeleton_result = guarded(
            "skeleton", skeleton_block, query_raw, representative, None, geometry
        )
    if not skeleton_ok or skeleton_result is None:
        skeleton_result = _skeleton_failure("display_error", representative if rep_ok else None)

    view = view_block(
        query_frames,
        reference_frames,
        weights=weights,
        level_text=query.level,
        grid_n=samples,
        corrections=issued,
        skeleton_result=skeleton_result,
        mesh=mesh,
    )

    ok, surface_angle = guarded("instructions", pass_surface_angle_deg, query_raw, geometry, basis)
    ok_entries, entries = guarded("instructions", instruction_entries, issued, surface_angle if ok else None)
    if ok_entries:
        for axis_entry, extra in zip(measured["axes"], entries):
            axis_entry.update(extra)
        by_n = {entry["n"]: entry for entry in entries}
        for arrow in view["corrections"]:
            extra = by_n[arrow["n"]]
            arrow.update({name: extra[name] for name in _TEXT_FIELDS})

    ok, shown = guarded("instructions", measured_display, measured)
    if ok:
        measured.update(shown)

    ok, orient = guarded("orientation", orientation.orientation_block, query_raw.rows, reference_raw.rows, geometry)
    measured["orientation"] = orient if ok else None
    ok, joint_axes = guarded("orientation", orientation.joint_axes_block, query_raw.rows, reference_raw.rows)
    measured["joint_axes"] = joint_axes if ok else None
    # 관절 각도는 **추정**이다(D4) — 표시 전용 블록이고 채점 경로를 타지 않는다.
    ok, angles = guarded("joint_angles", joint_angles.joint_angles_block, query_raw.rows, reference_raw.rows)
    measured["joint_angles"] = angles if ok else None

    joints = skeleton_result.joints if skeleton_result is not None else None
    ok, composite_view = guarded(
        "composite", composite.composite_block, skeleton_result.representative, geometry, joints
    )
    view["composite"] = composite_view if ok else None
    ok, hits = guarded("composite", composite.surface_hits, skeleton_result.representative)
    view["surface_hits"] = hits if ok else []
    # 스트로크는 **끊어진 구간 목록**으로 낸다 — 착탄점을 이은 현이 모서리를 가로질러 벽에서 뜨는
    # 곳에서 자른다(`composite.stroke_runs`). 화면은 이 구간만 그리고 자기가 다시 자르지 않는다.
    ok, stroke = guarded("composite", composite.stroke_runs, geometry, view["surface_hits"])
    view["stroke"] = stroke if ok else composite.stroke_runs(None, [])
    view["display_errors"] = errors
    body["view"] = view


def compare(
    source: PassSource,
    query: PassRef,
    reference: PassRef,
    params: Any | None = None,
    samples: int = stroke_frame.DEFAULT_SAMPLES,
    spreads: measured_score.AxisSpreads | None = None,
    weights: dict[str, float] | None = None,
    reference_spreads: SpreadReference | None = None,
    *,
    include_view: bool = True,
) -> dict[str, object]:
    """비교 1건. 동작 키가 다르면 지표를 만들지 않는다.

    `include_view=False`면 채점 필드만 낸다(`view`는 None, `measured`에 표시 전용 필드 없음).
    개선율 측정 도구(`improvement_study.py`)는 이 모드로만 부른다 — 표시 코드의 비용이나
    예외가 개선율 재측정을 인질로 잡지 않게 한다. 이 모드의 `measured`는 b5f7341과 JSON
    문자열까지 같다(`test_scoring_isolation.py`).
    """
    params = params or motion_key.KeyParams()
    weights = weights or measured_score.DEFAULT_WEIGHTS

    # 원본 행을 따로 들고 있는다 — 스켈레톤 FK(`skeleton_block`)가 회전 각을 원본
    # 행에서만 읽을 수 있고, `pass_frames`가 낸 리샘플 격자로는 얻을 수 없다.
    query_raw = source.load(query)
    query_frames = pass_frames(query_raw, samples)
    reference_raw = source.load(reference)
    reference_frames = pass_frames(reference_raw, samples)
    query_inputs = key_inputs(query_frames)
    reference_inputs = key_inputs(reference_frames)
    query_key = None if query_inputs is None else key_from(query_inputs, params)
    reference_key = None if reference_inputs is None else key_from(reference_inputs, params)
    # 몸 기준 축(②④)이 선 요다. 실측이면 그 값을, 회전을 못 읽었으면 대체값을 낸다.
    yaw_used = ASSUMED_YAW if query_inputs is None else yaw_of(query_inputs)
    yaw_source = (
        "measured" if query_inputs is not None and query_inputs.yaw is not None else "assumed"
    )

    # 키가 None인 것과 두 키가 다른 것은 다른 사건이다. 키는 착탄점이 있는 분사
    # 프레임이 2개 미만일 때 None이 된다 — 분사 컬럼을 못 읽었거나, 분산 기준으로
    # 활성 건을 못 고르거나, Ray가 한 번도 맞지 않은 패스다. 그것을 "의도가 다르다"로
    # 적으면 재지 못한 것을 재서 다르다고 말한 것이 된다.
    if query_key is None or reference_key is None:
        return {
            "status": "unavailable",
            "reason": "insufficient_frames",
            "query_key": query_key,
            "reference_key": reference_key,
            "yaw_assumed": yaw_used,
            "yaw_source": yaw_source,
        }
    if query_key != reference_key:
        return {
            "status": "unavailable",
            "reason": "motion_key_mismatch",
            "query_key": query_key,
            "reference_key": reference_key,
            "yaw_assumed": yaw_used,
            "yaw_source": yaw_source,
        }

    query_track = _local_tracks(query_frames)
    reference_track = _local_tracks(reference_frames)
    # 문장·구조체·시뮬레이션·σ가 모두 이 하나의 기저 위에서 나온다. `local_delta`는
    # 스트로크 프레임 벡터이지 월드 벡터가 아니다 — score_track에 기저를 안 넘기면
    # WORLD_FRAME으로 읽어 wall/vertical/lateral로 잘못 나눈 σ가 나오고, 그 σ가
    # 나중에 (basis로 읽는) 교정 후보 선택과 축 이름은 같지만 물리적으로 다른 방향을
    # 가리키게 된다 — 축별 산포가 균일하지 않으면 그 어긋남이 σ와 교정 선택을 조용히
    # 오염시킨다. 기저를 먼저 구해 모든 소비자(score_track ×2, apply_measured,
    # corrections_text, right_hand_finding)에 하나만 넘긴다.
    basis = representative_basis(query_frames)

    # 산포 출처는 세 갈래다. (1) 호출자가 이미 계산된 산포를 직접 넘기면(`spreads`)
    # 그것이 우선이고 그대로 쓴다 — 기존 계약, 스케일을 건드리지 않는다. (2) 아니면
    # `reference_spreads` 아티팩트를 사다리(우8→우4→우2→우off)로 훑어 답을 낸 층을
    # 쓴다 — 이때는 `PAIR_SIGMA_FACTOR`로 스케일한다(레퍼런스가 한 패스이지 버킷
    # 평균이 아니므로). (3) 둘 다 없으면 잠정 상수로 떨어진다.
    #
    # 질의가 전형기 자신이면(같은 worker) 그 날짜를 레퍼런스에서 뺀다 — 자기가 자기
    # 평균에 가까운 것은 당연하다(설계 D3).
    if query_inputs is None:
        # query_key가 not None으로 이미 확인됐으므로(위 두 조기 반환 중 하나로
        # 빠지지 않았다면) 여기 오지 않는다 — `key_from`이 `query_inputs`가 있어야만
        # `query_key`를 만들기 때문이다. `assert`는 `python -O`에서 벗겨져 나가므로
        # 실제 제어 흐름으로 막는다.
        raise RuntimeError("unreachable: query_key가 not None이면 query_inputs도 not None이다")
    measured_spreads = spreads is not None
    resolved: Resolved | None = None
    exclude_dates: list[str] = []
    if spreads is not None:
        spreads_used = spreads
    elif reference_spreads is not None:
        ladder_keys = _reference_keys(query_inputs)
        exclude_dates = (
            [query.date] if query.worker == reference_spreads.worker else []
        )
        resolved = reference_spreads.resolve(ladder_keys, exclude_dates=exclude_dates)
        if resolved is not None:
            measured_spreads = True
            spreads_used = _scale_spreads(resolved.spreads)
        else:
            spreads_used = PROVISIONAL_SPREADS
    else:
        spreads_used = PROVISIONAL_SPREADS

    before = measured_score.score_track(query_track, reference_track, spreads_used, weights, basis)
    # 개선율은 "교정 알고리즘의 출력을 적용한 뒤의 상승"이다(goal_year2). 그래서
    # 시뮬레이션에 같은 기저를 넘겨 **발부한 것만** 적용한다 — 오른손의 임계를 넘은
    # 축 전부와, 자세 문장이 나갔다면 머리의 그 축. 왼손은 어떤 경우에도 제자리다
    # (`measured_score.issued_axes`가 애초에 내지 않는다). 점수는 세 점을 다 보지만
    # 지시하지 않은 점을 옮기면 작업자에게 말한 것과 보고하는 숫자가 달라진다.
    fixed = measured_score.apply_measured(query_track, before.deltas, basis)
    after = measured_score.score_track(fixed, reference_track, spreads_used, weights, basis)

    # 분리 채점: 교정을 고르는 레퍼런스(A)와 채점하는 레퍼런스(B)가 겹치지 않아야
    # 개선율이 자기충족이 아니다(설계 D4). 날짜를 반으로 갈라 A로 교정 후보(deltas)를
    # 뽑는다. 채점(전·후 둘 다)은 B 하나로만 한다 — S0를 F로, S1을 B로 재면 잣대를
    # 바꾼 것만으로 gap_closure가 흔들린다(`improvement` 독스트링). B로 두 번 재면
    # 두 점수 사이에서 움직이는 것이 교정 적용 여부뿐이 된다.
    #
    # `normalization`이 "measured"인 것과 split을 낼 수 있는 것은 다른 질문이다 —
    # 호출자가 `spreads=`를 직접 넘긴 경로는 measured지만 날짜별 구조가 아예 없다.
    # 그래서 이 기본 사유는 gate_reason이 쓰는 "no_reference_distribution"(=측정이
    # 아니다)과 다른 이름을 쓴다 — 그 문구를 여기서도 쓰면 normalization="measured"인
    # 응답이 "측정이 아니다"라고 말하는 모순이 생긴다.
    before_split_score: float | None = None
    after_split_score: float | None = None
    split_reason: str | None = "no_split_reference"
    split_dates_a: list[str] = []
    split_dates_b: list[str] = []
    split_n_a: int | None = None
    split_n_b: int | None = None
    if resolved is not None:
        halves = reference_spreads.split_dates(resolved.granularity, resolved.key)
        if halves is None:
            split_reason = "insufficient_dates"
        else:
            # `split_dates`/`spreads_for`는 Task 4의 확정된 계약이라 질의 날짜를
            # 모른다 — 그래서 `resolve()`에 준 것과 같은 `exclude_dates`를 여기서도
            # 적용한다. 안 그러면 질의가 전형기 자신일 때 그 날짜가 A/B 양쪽에 새고,
            # 응답의 `reference_dates`(F, 이미 제외됨)와 실제로 split을 만든 날짜가
            # 서로 다른 걸 가리키게 된다 — "출처가 아닌 값에 붙은 이름표" 결함이다.
            split_dates_a = [date for date in halves[0] if date not in exclude_dates]
            split_dates_b = [date for date in halves[1] if date not in exclude_dates]
            split_n_a = _half_sample_count(
                reference_spreads, resolved.granularity, resolved.key, split_dates_a
            )
            split_n_b = _half_sample_count(
                reference_spreads, resolved.granularity, resolved.key, split_dates_b
            )
            if split_n_a < 2 or split_n_b < 2:
                # `N_MIN`(20)은 resolve()가 쓰는, 이전 연구에서 사전등록한 문턱이지
                # 반쪽의 문턱이 아니다. 지금 데이터를 보고 나서 반쪽 전용 문턱을 새로
                # 만들면 이 계획 전체가 막으려는 사후 튜닝이 되고, split을 가장 두꺼운
                # 버킷에서만 나오게 편향시켜 Task 7의 숫자를 위로 왜곡한다. 유일한
                # 문턱은 수학적 필요조건이다 — 표본이 2개 미만이면 분산 자체가
                # 정의되지 않는다(`spread_stats.summarize`). 얇은 반쪽은 버리지 않고
                # `split_n_a`/`split_n_b`로 그대로 보고해, Task 7이 두께로 계층화할
                # 수 있게 한다.
                split_reason = "insufficient_samples"
            else:
                raw_a = reference_spreads.spreads_for(
                    resolved.granularity, resolved.key, split_dates_a
                )
                raw_b = reference_spreads.spreads_for(
                    resolved.granularity, resolved.key, split_dates_b
                )
                if raw_a is None or raw_b is None:
                    split_reason = "insufficient_dates"
                else:
                    spreads_b = _scale_spreads(raw_b)
                    # A는 교정 후보(deltas)를 고르는 데만 쓴다 — 채점에는 쓰지 않는다.
                    before_a = measured_score.score_track(
                        query_track, reference_track, _scale_spreads(raw_a), weights, basis
                    )
                    fixed_a = measured_score.apply_measured(query_track, before_a.deltas, basis)
                    # B가 교정 전·후를 둘 다 잰다(`eval_pose_correction.measure_correction`과
                    # 같은 패턴: A로 고르고 B로 앞뒤를 다 잰다).
                    before_split = measured_score.score_track(
                        query_track, reference_track, spreads_b, weights, basis
                    )
                    after_split = measured_score.score_track(
                        fixed_a, reference_track, spreads_b, weights, basis
                    )
                    before_split_score = before_split.score
                    after_split_score = after_split.score
                    split_reason = None

    # 발부 목록은 **한 번만** 구한다. `measured.axes`와 `view.corrections`가 같은 목록을
    # 받아야 번호(n)가 짝짓는 키가 된다. `apply_measured`는 measured_score 안에서 같은
    # 규칙(`issued_shifts` → `issued_axes`)으로 따로 부르지만 입력이 같아 같은 목록이다
    # (`test_issued_axes_single_list_matches_apply_measured`).
    issued = measured_score.issued_axes(before.deltas, basis)
    body: dict[str, Any] = {
        "status": "ok",
        "query_key": query_key,
        "reference_key": reference_key,
        # 몸 기준 축(②④)이 선 요. `yaw_source`가 실측인지 대체값인지 말한다.
        "yaw_assumed": yaw_used,
        "yaw_source": yaw_source,
        "samples": before.samples,
        # 스켈레톤 뷰(층·교정 화살표·메시·가중치)는 채점 필드를 다 만든 뒤
        # `include_view`일 때만 붙인다(아래 `_attach_view`). `corrected`/`corrections`는
        # `axes`와 같은 `issued` 목록에서 나온다(요구사항 1).
        "view": None,
        "measured": {
            "basis": "tracked",
            # σ와 점수 exp(-σ²/2)는 이 산포로 나눈 값이다. 잰 분포가 아니면 읽는
            # 사람이 그 사실을 알아야 한다. resolve()가 None을 낸 경우(사다리의 어느
            # 층도 n_min을 못 채움)에도 여기서 잠정으로 떨어진다 — 잠정 상수 위에
            # "measured" 딱지를 붙이는 것이 이 배선에서 낼 수 있는 최악의 결과다.
            "normalization": "measured" if measured_spreads else "provisional",
            # `resolved`가 실제로 답을 낸 층이지 질의가 요청한 층이 아니다. 없으면
            # (레퍼런스가 없거나 사다리 전부 실패) 어느 층에서 나온 숫자도 아니므로
            # None이다 — 기본값이나 요청값을 대신 채우지 않는다.
            "granularity": resolved.granularity if resolved is not None else None,
            "reference_n": resolved.n if resolved is not None else None,
            "reference_dates": resolved.dates if resolved is not None else [],
            "score": before.score,
            "score_after": after.score,
            "deltas": [
                {
                    "point": delta.point,
                    "sigma": delta.sigma,
                    "delta": list(delta.local_delta),
                    # 대응 격자점 사이 거리(cm) — "지금 자세가 기준과 얼마나 다른가"의
                    # 답(브리프 ②)이다. `local_delta`는 스트로크 로컬 벡터지만 회전은
                    # 노름을 보존하므로 그 크기는 월드 프레임에서 잰 거리와 같다. 화면이
                    # 다시 계산하지 않도록 여기서 낸다(이 저장소의 규칙 — 브리프 ②
                    # 각주: "서버에서 내주는 쪽이 낫다").
                    "deviation_cm": math.dist(delta.local_delta, (0.0, 0.0, 0.0)) * 100.0,
                    # 왼손은 프레임 단위로 빠지므로 점마다 받친 격자점 수가 다르다.
                    "samples": delta.samples,
                    # 축별(wall/vertical/lateral) σ. 화면 막대 옆에 붙일 근거다 —
                    # `delta.sigma`는 이 셋을 직교로 합친(제곱합 후 MAX_SIGMA에서
                    # 클립한) 값이라 축 하나가 왜 이상한지는 이 필드 없이는 안 보인다.
                    "axis_sigmas": dict(delta.axis_sigmas),
                }
                for delta in before.deltas
            ],
            # 기저를 함께 넘긴다. 성분을 축 이름으로 읽으면 수직 스트로크에서 좌우
            # 이탈이 "올리세요"로 나간다(`stroke_frame.Basis` 독스트링).
            "corrections": measured_score.corrections_text(before.deltas, basis),
            # 자세 문장 하나(머리)와 발부한 축 전부(오른손+머리).
            # `axes`는 **작업자가 실제로 받는 지시**이고, 그것이 곧 위 `fixed`가 옮긴
            # 것이다 — 그래서 화면 숫자와 `improvement.same`이 같은 이동을 잰다.
            # `improvement.split`은 다르다: 자기충족을 피하려고 교정 후보를 A 반쪽의
            # 산포로 **다시 고르므로**(`before_a.deltas`) σ가 달라져 발부 축 집합이
            # 이 목록과 갈릴 수 있다. 그것이 split의 설계이지 결함이 아니다 —
            # 다만 "화면 = split"이라고 읽으면 안 된다.
            "stance": stance_block(before.deltas, basis),
            "axes": axes_block(issued),
            # 최종 작업자 문구는 다운스트림 VLM이 짓는다. 문장만 주면 VLM이 근거 없이
            # 다시 지어내야 한다 — 표본 수는 문장에 없다. 구조체를 병기해 축·부호·σ·
            # 표본 n을 그대로 넘긴다. 오른손이 임계 아래거나 못 잰 경우 None이다.
            "finding": (
                None
                if (finding := measured_score.right_hand_finding(before.deltas, basis))
                is None
                else asdict(finding)
            ),
            # 진짜 분리 채점 점수는 겹치지 않는 날짜에서 뽑은 held-out 레퍼런스가
            # 있어야 나온다(위의 `after_split_score`). 없으면(레퍼런스가 없거나 날짜가
            # 모자라면) split을 None으로 낸다 — after.score를 split 자리에 다시 넣으면
            # 두 숫자가 항상 같아져 자기충족 정도를 가리는 바로 그 결함이 된다.
            # `after`는 위에서 낸 문장 하나를 그대로 적용한 점수다. 지시하지 않은
            # 이동은 들어 있지 않다.
            "improvement": improvement(
                before.score, after_split_score, after.score, before_split_score
            ),
            # split이 None이면 왜인지 말한다 — 레퍼런스 객체가 아예 없거나(D3/D4 이전과
            # 같은 잠정/직접-spreads 상태, "no_split_reference"), 있어도 날짜가 모자라
            # A/B로 못 가르거나("insufficient_dates"), 갈랐는데 어느 한쪽이 표본 2개
            # 미만이거나("insufficient_samples")다. split 값이 있으면(None이 아니면)
            # 사유도 None이다 — 지어낼 이유가 없다.
            "split_reason": split_reason,
            # A/B가 실제로 어떤 날짜·표본 수로 섰는지. `reference_dates`(F)와 다른
            # 것을 나타낸다는 점이 중요하다 — 여기 실리지 않으면 `reference_dates`가
            # split의 출처인 것처럼 잘못 읽힌다. n<2로 계산을 안 한 반쪽도 실제로 걸러진
            # 날짜/표본 수를 그대로 낸다(위 F2 사유 참고) — 만들어내지도, 숨기지도 않는다.
            "split_dates_a": split_dates_a,
            "split_dates_b": split_dates_b,
            "split_n_a": split_n_a,
            "split_n_b": split_n_b,
            # 게이트의 뜻은 "각도 지시가 이제 정당하다"다. 잰 분포가 없으면 판정을
            # 만들지 않는다. 임계 논리(`gate_verdict`)는 그대로 두고 입력이 없다는
            # 사실만 낸다.
            # 산포가 점·축별이 된 뒤에도 게이트 임계 논리는 스칼라 하나를 받는다.
            # 가장 넓은(가장 관대한) 축·점의 산포로 판정한다 — 예전에 점당 스칼라의
            # 최댓값을 쓰던 것과 같은 "최악의 경우로 판정" 규칙을 축 차원까지 넓힌다.
            # `spreads_used`는 이미 (레퍼런스에서 왔다면) √2로 스케일된 값이다 — 실제
            # score_track이 나눈 산포와 게이트가 판정하는 산포가 같아야 한다.
            "gate": (
                gate_verdict(
                    max(value for axes in spreads_used.values() for value in axes.values()),
                    DEFAULT_ERROR_M,
                )
                if measured_spreads
                else None
            ),
            "gate_reason": None if measured_spreads else "no_reference_distribution",
        },
        "joint": {
            "basis": "ik_derived",
            "status": "not_wired",
            "note": JOINT_AXIS_NOTE,
            "score": None,
            "score_after": None,
            "deltas": [],
            "corrections": [],
            "stance": None,
            "axes": [],
            "finding": None,
            "improvement": None,
            "gate": None,
            "gate_reason": "not_wired",
        },
    }
    if not include_view:
        return body
    _attach_view(
        body,
        query=query,
        query_raw=query_raw,
        reference_raw=reference_raw,
        query_frames=query_frames,
        reference_frames=reference_frames,
        issued=issued,
        basis=basis,
        weights=weights,
        samples=samples,
    )
    return body


def reference_candidates(
    source: Any,
    query: PassRef,
    params: Any | None = None,
    samples: int = stroke_frame.DEFAULT_SAMPLES,
    limit: int = 50,
) -> dict[str, object]:
    """대상 패스의 기준 후보 — 사용자 지시(브리프 §"기준은 고를 수 없다"): 전형기
    고정 · 대상과 같은 레벨 · 동작 키가 일치 · 대상 자신은 제외.

    화면은 이 함수가 돌려준 `candidates` 안에서만 고른다 — 다른 작업자·다른 레벨을
    기준으로 보낼 길이 애초에 없다. `source`는 `CsvPassSource`(또는 같은 표면을 가진
    페이크) 전체를 받는다 — `PassSource` 프로토콜(`workers`/`load`)보다 넓다
    (`passes`/`key_of`도 쓴다); `app.py`의 `/api/passes`가 이미 같은 폭으로
    `source`를 직접 쓰는 것과 같은 이유다.
    """
    params = params or motion_key.KeyParams()
    if EXPERT_WORKER not in source.workers():
        return {"status": "unavailable", "reason": "no_expert_worker"}
    try:
        raw_query = source.load(query)
    except LookupError:
        return {"status": "unavailable", "reason": "pass_not_found"}

    query_frames_ = pass_frames(raw_query, samples)
    query_inputs = key_inputs(query_frames_)
    if query_inputs is None:
        return {"status": "unavailable", "reason": "insufficient_frames"}
    query_key = key_from(query_inputs, params)

    found, sessions_total = source.passes(EXPERT_WORKER, level=query.level, limit=limit)
    candidates: list[dict[str, object]] = []
    for item, entry in found:
        ref = PassRef(
            worker=item.worker,
            date=item.date,
            level=item.level,
            session=item.session,
            pass_index=entry.pass_index,
        )
        if ref == query:
            # 자기 자신과 비교하면 편차가 0이 되어 화면이 거짓을 말한다(브리프).
            continue
        motion = source.key_of(entry, params)
        if motion != query_key:
            continue
        candidates.append(
            {
                "worker": ref.worker,
                "date": ref.date,
                "level": ref.level,
                "session": ref.session,
                "pass_index": ref.pass_index,
                "frames": entry.frames,
                "length_mm": entry.length_m * 1000.0,
                "motion_key": motion,
            }
        )

    return {
        "status": "ok",
        "expert_worker": EXPERT_WORKER,
        "query_key": query_key,
        "level": query.level,
        "sessions_scanned": min(sessions_total, limit),
        "sessions_total": sessions_total,
        # 훑은 세션을 한도로 잘랐다는 사실을 화면이 알아야 한다 — 숨기면 "이 동작
        # 키의 전형기 패스가 없다"와 "한도 안에서 못 찾았다"가 구별되지 않는다.
        "truncated": sessions_total > limit,
        "candidates": candidates,
    }
