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
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Final, Iterable, Mapping, Protocol, Sequence

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
# 배포된 채점 경로. **랩이 제품과 다른 계산을 하면 랩으로 제품을 못 고친다.**
arm_reference = importlib.import_module("arm_reference")
pose_stages = importlib.import_module("pose_stages")

from trainer.pose_lab import body_proportions, composite, instructions, joint_angles, orientation, right_arm, skeleton
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


def spray_frames(raw: RawPass) -> list[Any]:
    """패스 한 건의 원 프레임(리샘플 전). 분사 중이고 착탄점이 있는 프레임만 남는다.

    동작 키의 UV 칸은 이 원 프레임에서 읽는다(`motion_key.midpoint_uv`) — 리샘플 격자는
    모든 점에 첫 프레임의 UV를 복사하기 때문이다.
    """
    frames = measured_pose.measured_frames(raw.rows)
    return [frame for frame in frames if frame.spraying and frame.hit is not None]


def pass_frames(raw: RawPass, samples: int = stroke_frame.DEFAULT_SAMPLES) -> list[Any]:
    """패스 한 건의 정렬 격자. 분사 중이고 착탄점이 있는 프레임만 남는다."""
    return stroke_frame.resample_frames(spray_frames(raw), samples)


def pass_key_inputs(
    raw: RawPass, samples: int = stroke_frame.DEFAULT_SAMPLES
) -> tuple[list[Any], "KeyInputs | None"]:
    """(리샘플 격자, 키 재료). 색인(`csv_source`)·비교(`compare`)·기준 후보·산포 빌더가
    같은 규칙으로 키를 만든다."""
    raw_frames = spray_frames(raw)
    frames = stroke_frame.resample_frames(raw_frames, samples)
    return frames, key_inputs(frames, raw_frames=raw_frames)


@dataclass(frozen=True, slots=True)
class KeyInputs:
    """동작 키를 만드는 재료. 판정 경계(KeyParams)와 분리해 둔다.

    재료는 패스에서 한 번 뽑고 경계는 요청마다 달라진다. 둘을 붙여 두면 화면이 UV
    격자를 바꿀 때마다 CSV를 다시 읽어야 한다.
    """

    displacement: Vec3
    head: Vec3
    hit: Vec3
    # 원 프레임 호장 중점의 UV(`motion_key.midpoint_uv`, m3). 리샘플 격자의 UV(=첫 프레임 UV)가 아니다.
    uv: tuple[float, float] | None
    yaw: float | None = None


def key_inputs(frames: Sequence[Any], *, raw_frames: Sequence[Any]) -> KeyInputs | None:
    """키 재료. `frames`는 리샘플 격자, `raw_frames`는 그 격자를 만든 원 프레임이다.

    `raw_frames`를 키워드 필수로 둔다 — 기본값으로 빠지면 UV가 조용히 격자의 UV, 곧
    첫 프레임 UV가 된다(m1의 결함 D1, 사전등록 2026-09-22 §3.1).
    """
    if len(frames) < 2:
        return None
    first, last = frames[0], frames[-1]
    if first.hit is None or last.hit is None:
        return None
    middle = frames[len(frames) // 2]
    raw_hits = [frame for frame in raw_frames if frame.hit is not None]
    return KeyInputs(
        displacement=(
            last.hit[0] - first.hit[0],
            last.hit[1] - first.hit[1],
            last.hit[2] - first.hit[2],
        ),
        head=middle.head,
        hit=middle.hit or first.hit,
        uv=motion_key.midpoint_uv([frame.hit for frame in raw_hits], [frame.uv for frame in raw_hits]),
        yaw=getattr(middle, "yaw", None),
    )


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


# --- 두 스켈레톤이 벌어진 거리(표시 전용) -------------------------------------------
#
# 사용자가 원한 그림: "기준 스켈레톤과 질의 스켈레톤을 같이 그리고 유사도가 떨어지는 조인트일수록
# 색상을 진하게". 그래서 **서버가** 관절마다 두 스켈레톤이 벌어진 거리를 재고 화면은 그 값으로
# 진하기만 정한다(화면은 다시 계산하지 않는다 — 이 저장소의 규칙).
#
# 재는 것은 배치가 끝난 **월드 거리**다. 화면에 실제로 보이는 간격이 곧 색이 된다 — 다른 기저로
# 재면 눈에 보이는 간격과 색이 갈린다. 실측(김경호 20250831 대 자동 기준): 관절 편차 중앙값이
# L1 0.285m · L3 0.246m · L6 0.241m이고 최대가 0.60~0.80m다. 두 자세가 겹쳐 보이는 크기다.
#
# **정직성**: 관절 편차는 추정끼리의 차이다. 추적 장치가 붙은 강체는 머리(HMD)·건·플래쉬 셋뿐이고,
# 그 셋의 마커 간격만 실측끼리의 차이다. 그래서 두 값을 한 자루에 담지 않고 `joints`(추정)와
# `markers`(실측)로 나눠 싣고, 눈금(`*_max_m`)도 따로 낸다 — 화면이 섞어 칠하지 못하게.
DEVIATION_BASIS: Final = "world_distance_m"
DEVIATION_ESTIMATE_NOTE: Final = (
    "관절이 벌어진 거리는 추정끼리의 차이입니다 — 두 아바타가 IK로 지은 자세를 뺀 값이지 잰 값이 아닙니다"
)
DEVIATION_MEASURED_NOTE: Final = (
    "머리(HMD)·건·플래쉬 마커가 벌어진 거리만 실측끼리의 차이입니다"
)
# 실측 강체에 붙은 앵커 이름(`skeleton.place_oriented`가 내는 `AnchorFit.point`).
DEVIATION_MEASURED_POINTS: Final[tuple[str, ...]] = ("head", "right", "left")

# 진하기 눈금 범례(S1). 셋 다 **서버가 잰 값**으로 채운다 — 화면이 이름과 숫자를 조립하면
# 눈금과 문구가 갈린다(검증 라운드: 범례가 "가장 진한 곳 79.9cm"라고 적었는데 화면 어디에도
# 그 관절이 없었다).
# 첫 줄(범례에 뜨는 줄)은 뜻만 말한다 — 길이·관절 이름은 둘째 줄로 가서 접힌 상세에 든다(사용자: "불필요한
# 텍스트는 넣지 말 것. UI는 간단한 편이 좋음").
DEVIATION_MEANING_LEGEND: Final = "관절 진하기 = 기준과 벌어진 거리"
DEVIATION_SCALE_LEGEND: Final = "가장 진한 {joint} {cm}cm가 진하기 눈금 끝입니다"
DEVIATION_HIDDEN_LEGEND: Final = "안 그린 다리 관절은 눈금에서 뺐습니다 — 그중 {joint}{particle} {cm}cm로 더 멉니다"
DEVIATION_SHIFT_LEGEND: Final = "이 거리에는 두 몸이 통째로 옮겨 앉은 몫 {cm}cm가 모든 관절에 공통으로 섞여 있습니다"

# 관절 이름을 사람이 읽는 말로. 범례가 "LeftCalf"라고 적으면 화면에서 그 관절을 못 찾는다.
JOINT_LABELS: Final[dict[str, str]] = {
    "Pelvis": "골반", "Spine": "허리", "Spine1": "등", "Chest": "가슴", "Neck": "목", "Head": "머리",
    "LeftShoulder": "왼 어깨", "LeftUpperArm": "왼 위팔", "LeftForeArm": "왼 전완", "LeftHand": "왼손",
    "RightShoulder": "오른 어깨", "RightUpperArm": "오른 위팔", "RightForeArm": "오른 전완", "RightHand": "오른손",
    "LeftThigh": "왼 허벅지", "LeftCalf": "왼 종아리", "LeftFoot": "왼발", "LeftToe": "왼 발끝",
    "RightThigh": "오른 허벅지", "RightCalf": "오른 종아리", "RightFoot": "오른발", "RightToe": "오른 발끝",
}


def subject_particle(word: str) -> str:
    """한국어 주격 조사 이/가. 끝 글자에 받침이 있으면 "이", 없으면 "가"다.

    범례가 "왼 종아리이"라고 적으면 서버가 지은 문장이 사람 말이 아니게 된다.
    """
    if not word:
        return "가"
    last = word[-1]
    if "\uac00" <= last <= "\ud7a3":
        return "가" if (ord(last) - 0xAC00) % 28 == 0 else "이"
    return "이"


def joint_label(name: str) -> str:
    """관절의 사람이 읽는 이름. 모르는 이름은 리그 이름 그대로 낸다 — 지어내지 않는다."""
    return JOINT_LABELS.get(name, name)



def _pair_distances(a: Mapping[str, Sequence[float]], b: Mapping[str, Sequence[float]]) -> dict[str, float]:
    """두 이름표 집합에 **둘 다 있는** 이름만 월드 거리를 잰다. 한쪽에만 있으면 짝이 없으므로 내지 않는다."""
    return {name: math.dist(tuple(a[name]), tuple(b[name])) for name in a if name in b}


def deviation_block(
    query: SkeletonResult | None,
    reference: SkeletonResult | None,
    *,
    hidden_joints: Iterable[str] = (),
) -> dict[str, object] | None:
    """질의·기준 스켈레톤이 관절마다 얼마나 벌어졌는가(m). 한쪽이라도 없으면 None.

    `None`("두 스켈레톤이 다 서지 않아 잴 수 없다")과 빈 dict("쟀는데 공통 관절이 없다")은 다른
    사건이다 — 화면이 "벌어짐 0"으로 칠하지 않도록 구분해 낸다.
    """
    if query is None or reference is None:
        return None
    if query.joints is None or reference.joints is None:
        return None
    joints = _pair_distances(query.joints, reference.joints)
    markers = _pair_distances(query.markers, reference.markers)
    # 눈금에서 빼는 관절 = **화면에 그리지 않는 관절**. 안 그리는 관절이 눈금을 정하면 보이는
    # 관절이 전부 눈금의 아랫도리에만 몰린다 — 검증 라운드 실측: L3의 argmax가 숨긴 LeftCalf
    # 0.7988m라 보이는 관절의 t가 0.412에서 멈춰 설계된 색 범위의 26%만 썼고, 범례는 화면
    # 어디에도 없는 "79.9cm"를 적었다. 사용자가 요청한 "유사도가 떨어지는 조인트일수록 진하게"가
    # 가장 필요한 앉은 자세 레벨에서 가장 약해진 것이다.
    hidden = {name for name in hidden_joints if name in joints}
    drawn = {name: value for name, value in joints.items() if name not in hidden}
    scale_source = drawn if drawn else joints
    joint_max_name = max(scale_source, key=lambda name: scale_source[name]) if scale_source else None
    joint_max_m = None if joint_max_name is None else scale_source[joint_max_name]
    hidden_pairs = {name: joints[name] for name in hidden}
    hidden_max_name = max(hidden_pairs, key=lambda name: hidden_pairs[name]) if hidden_pairs else None
    hidden_max_m = None if hidden_max_name is None else hidden_pairs[hidden_max_name]
    shift = _centroid_gap_m(query.joints, reference.joints)
    return {
        "basis": DEVIATION_BASIS,
        "joints": joints,
        "markers": markers,
        # 색 눈금. 추정과 실측을 같은 눈금으로 칠하면 둘의 구분이 색에서 사라진다.
        # 그리는 관절에서만 잡는다 — 눈금을 정한 관절 이름도 같이 낸다(범례가 이름으로 말한다).
        "joint_max_m": joint_max_m,
        "joint_max_name": joint_max_name,
        "joint_max_label": None if joint_max_name is None else joint_label(joint_max_name),
        "marker_max_m": max(markers.values()) if markers else None,
        "joint_max_display": (
            None if joint_max_m is None else instructions.deviation_display(joint_max_m * 100.0)
        ),
        # 눈금에서 뺀 관절과 그중 가장 먼 것. 숨겼다는 사실을 숨기지 않는다.
        "scale_excludes": sorted(hidden),
        "hidden_max_name": hidden_max_name,
        "hidden_max_label": None if hidden_max_name is None else joint_label(hidden_max_name),
        "hidden_max_m": hidden_max_m,
        "hidden_max_display": (
            None if hidden_max_m is None else instructions.deviation_display(hidden_max_m * 100.0)
        ),
        # 몸 전체가 통째로 옮겨 앉은 몫(두 스켈레톤 관절 무게중심 사이 거리). 이 값이 모든 관절
        # 편차에 공통으로 섞여 있다 — 한 관절의 색을 "이 관절만 다르다"로 읽으면 과대 해석이다.
        "common_shift_m": shift,
        "common_shift_display": (
            None if shift is None else instructions.deviation_display(shift * 100.0)
        ),
        "measured_points": [name for name in DEVIATION_MEASURED_POINTS if name in markers],
        "estimate_note": DEVIATION_ESTIMATE_NOTE,
        "measured_note": DEVIATION_MEASURED_NOTE,
        # 범례 문구는 서버가 짓는다 — 화면이 숫자와 이름을 조립하면 눈금과 문구가 갈린다.
        "legend": _deviation_legend(
            joint_max_label=None if joint_max_name is None else joint_label(joint_max_name),
            joint_max_display=None if joint_max_m is None else instructions.deviation_display(joint_max_m * 100.0),
            hidden_max_label=None if hidden_max_name is None else joint_label(hidden_max_name),
            hidden_max_display=None if hidden_max_m is None else instructions.deviation_display(hidden_max_m * 100.0),
            hidden_beats_drawn=(
                hidden_max_m is not None and joint_max_m is not None and hidden_max_m > joint_max_m
            ),
            shift_display=None if shift is None else instructions.deviation_display(shift * 100.0),
        ),
    }


def _centroid_gap_m(
    query: Mapping[str, Sequence[float]], reference: Mapping[str, Sequence[float]]
) -> float | None:
    """두 스켈레톤의 **짝지어진 관절 무게중심** 사이 거리(m).

    같은 동작 키 칸 안이라도 두 패스의 벽 위 자리가 조금 다르면 몸 전체가 통째로 옮겨 앉는다.
    그 평행이동은 모든 관절 편차에 공통으로 더해지므로, 한 관절의 색을 "이 관절만 다르다"로
    읽으면 과대 해석이다. 화면이 그 사실을 말하려면 그 몫이 얼마인지 잴 값이 있어야 한다.
    """
    names = [name for name in query if name in reference]
    if not names:
        return None
    qc = [sum(float(query[name][axis]) for name in names) / len(names) for axis in range(3)]
    rc = [sum(float(reference[name][axis]) for name in names) / len(names) for axis in range(3)]
    return math.dist(qc, rc)


def _deviation_legend(
    *,
    joint_max_label: str | None,
    joint_max_display: str | None,
    hidden_max_label: str | None,
    hidden_max_display: str | None,
    hidden_beats_drawn: bool,
    shift_display: str | None,
) -> list[str]:
    """진하기 눈금을 설명하는 범례 줄. 눈금이 없으면 아무 말도 하지 않는다."""
    lines: list[str] = []
    if joint_max_display is None or joint_max_label is None:
        return lines
    lines.append(DEVIATION_MEANING_LEGEND)
    lines.append(DEVIATION_SCALE_LEGEND.format(joint=joint_max_label, cm=joint_max_display))
    if hidden_beats_drawn and hidden_max_label is not None and hidden_max_display is not None:
        lines.append(
            DEVIATION_HIDDEN_LEGEND.format(
                joint=hidden_max_label, particle=subject_particle(hidden_max_label), cm=hidden_max_display
            )
        )
    if shift_display is not None:
        lines.append(DEVIATION_SHIFT_LEGEND.format(cm=shift_display))
    return lines



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
# 범례는 **한 줄**이다(사용자: "UI는 간단한 편이 좋음"). 무엇이 실측이고 무엇이 추정인지는 글자가 아니라
# **모양**이 말하고, 범례는 그 모양의 뜻만 짧게 적는다 — 채운 구 = 실측 장치(HMD·건·플래쉬), 속 빈 표식 =
# 아바타 추정 관절, 점선 = 재지 않은 부분(자리표시 다리 · 오른손이 손잡이에 못 닿은 거리), 줄무늬 = 건.
# 예전 범례는 다섯 문장이었고 방법 설명("몸(추정 관절)에서 장치(실측)까지 벌어진 간격…")까지 실었다 —
# 사용자가 "분사구 0.3m(추정)과 같은 불필요한 텍스트는 넣지 말 것"이라고 했다. 가는 선(궤적)의 이름은
# **장치** 이름이다: 궤적은 관절이 아니라 HMD·건·플래쉬가 지나간 길이다.
LEGEND_MEASURED_WORD: Final = "채운 구 = 실측 장치"
# 몸(반투명 살)도 추정이다 — 뼈대 위에 입힌 부피라 실측처럼 보일 수 있어 범례가 이름을 댄다.
# 구분자(` · `)를 쓰지 않는다 — 범례 조각 수가 곧 부호 수다.
LEGEND_ESTIMATE_WORD: Final = "속 빈 표식 = 추정 관절(반투명 몸 포함)"
LEGEND_DASHED_WORD: Final = "점선 = 재지 않은 부분"
LEGEND_GUN_WORD: Final = "줄무늬 = 건"
LEGEND_SEPARATOR: Final = " · "

# 장치 모양(사용자 선택 "장치를 모양으로 그리기"). 채운 구체(실측 장치)마다 짧은 이름 하나만 단다 — 거리도
# 방법도 적지 않는다(사용자: "불필요한 텍스트는 넣지 말 것"). 이름표는 **대상 층에만** 달고(기준 층은 색이
# 말한다), 화면이 그 규칙을 지킨다.
#
# 막대: 건은 **손잡이 → 트래커**(분사구) 굵은 줄무늬 막대다 — 오른팔을 건 손잡이로 세운 뒤로(`right_arm`)
# 막대의 손 쪽 끝은 손잡이이고, 그 길이는 `right_arm.GRIP_BEHIND_TRACKER_M`이다. HMD·플래쉬는 추정 관절 →
# 장치 **점선**이다(몸과 장치 사이 빈자리) — 줄무늬는 건 하나뿐이라 범례 "줄무늬 = 건"이 참이다.
DEVICE_NAMES: Final[dict[str, str]] = {"head": "HMD", "right": "건", "left": "플래쉬"}
DEVICE_SHAPES: Final[dict[str, str]] = {"head": "link", "right": "gun", "left": "link"}


def device_block(point: str, start: Sequence[float], measured_marker: Sequence[float]) -> dict[str, object] | None:
    """앵커 하나의 장치 모양 블록. 막대는 `from`(건 = 손잡이, 나머지 = 추정 관절)에서 실측 구까지다.

    이름표 글자는 장치 이름뿐이다 — 거리·"(추정)"·방법 문구를 달지 않는다.
    """
    name = DEVICE_NAMES.get(point)
    if name is None:
        return None
    return {
        "name": name,
        "shape": DEVICE_SHAPES[point],
        "label": name,
        "from": [float(v) for v in start],
        "to": [float(v) for v in measured_marker],
    }


def _legs_shown(fit: Mapping[str, object] | None) -> bool:
    """이 층이 다리를 그리는가. 서버 판정(`legs_visible_default`)이 유일한 출처다."""
    return bool(fit is not None and fit.get("legs_visible_default") is True)


def hidden_joints(
    fit: Mapping[str, object] | None, reference_fit: Mapping[str, object] | None
) -> tuple[str, ...]:
    """화면에 **안 그려지는** 관절 이름. 두 층 다 다리를 숨길 때만 다리 관절이 여기 든다.

    한 층이라도 그리면 그 관절은 화면에 있으므로 눈금에서 빼지 않는다 — 안 보이는 관절이 눈금을
    정하면 보이는 관절이 색 범위의 아랫도리에만 몰린다(검증 라운드 실측: L3 26%).
    """
    if _legs_shown(fit) or _legs_shown(reference_fit):
        return ()
    return tuple(skeleton.LEG_JOINTS)


def leg_notes(
    fit: Mapping[str, object] | None, reference_fit: Mapping[str, object] | None
) -> list[str]:
    """다리 고지 — **층마다 한 줄**, 두 층이 같은 문장이면 "대상·기준" 한 줄로 합친다.

    질의 고지만 싣던 동안에는 화면이 그림과 반대되는 말을 했다: 김경호 L1 1000389_1 #9는 기준 층
    다리 뼈 8개가 그려져 있는데 "다리는 그리지 않았습니다"라고 적혔고, #0은 기준 다리가 없는데
    "흐리게 그렸습니다"라고만 적혔다. 자동 기준으로 맺은 24쌍 중 10쌍(41.7%)에서 두 층의 판정이
    갈린다 — 한 층 고지로는 맞을 수가 없다.
    """
    templates = [
        (key, fit_block.get("leg_note_template"))
        for key, fit_block in (("query", fit), ("reference", reference_fit))
        if isinstance(fit_block, Mapping)
    ]
    present = [(key, text) for key, text in templates if isinstance(text, str)]
    if not present:
        return []
    if len(present) == 2 and present[0][1] == present[1][1]:
        return [present[0][1].format(who=LEG_NOTE_WHO["both"])]
    return [text.format(who=LEG_NOTE_WHO[key]) for key, text in present]


def head_height_block(
    fit: Mapping[str, object] | None, reference_fit: Mapping[str, object] | None
) -> dict[str, object]:
    """바닥 기준 머리 높이를 **두 몸 다**. 한 값만 보이면 어느 몸의 높이인지 알 수 없다.

    실측: L1 대상 1.51m · 기준 1.66m, L3 대상 0.70m · 기준 0.95m — 두 자세의 높이가 다르다.
    벽 메시가 없어 바닥면을 못 고른 레벨은 "—" 대신 왜 없는지 적는다.
    """
    values: dict[str, object] = {}
    parts: list[str] = []
    floor: float | None = None
    for key, fit_block in (("query", fit), ("reference", reference_fit)):
        metres = fit_block.get("head_above_floor_m") if isinstance(fit_block, Mapping) else None
        display = fit_block.get("head_above_floor_display") if isinstance(fit_block, Mapping) else None
        values[f"{key}_m"] = metres
        values[f"{key}_display"] = display
        if isinstance(display, str):
            parts.append(f"{LEG_NOTE_WHO[key]} {display}")
        if floor is None and isinstance(fit_block, Mapping):
            candidate = fit_block.get("floor_vertical")
            if isinstance(candidate, (int, float)):
                floor = float(candidate)
    values["display"] = " · ".join(parts) if parts else None
    values["note"] = HEAD_HEIGHT_NOTE.format(floor=floor) if parts and floor is not None else HEAD_HEIGHT_NO_FLOOR
    return values


# 층마다 궤적이 **몇 줄**인지 화면이 말한다.
#
# 사용자가 "왜인지 '기준'이 세 개가 그려져"라고 보고했다. 재현해 세어 보니 쌓임이 아니다 —
# `resetScene()`이 매 `render()`마다 층을 비우고, [다른 기준 ▸]을 세 번 넘겨도 [평가]를 세 번 다시
# 눌러도 기준 층의 씬 후손 수는 64로 같았다. 원래 **세 줄**이다: `buildLayerGroup`이 층의 점 이름
# (head·right·left)마다 폴리라인을 하나씩 만든다. 세 줄이 같은 색·라벨 없이 그려지니 "기준이 세 개"로
# 읽힌 것이다. 화면은 그 사실을 말해야 한다.
#
# 줄 이름의 출처는 이제 **그 층에 실제로 점이 실린 이름**이다. 예전에는 가중치(`view.weights`)에서
# 뽑고 "진한 차례로"라고 적었는데, 그 진하기 규칙 자체가 이번에 사라진다(궤적은 차분한 한 색이고
# 진해지는 것은 관절뿐이다 — `deviation_block`). 없어진 규칙을 범례가 계속 주장하면 화면이 거짓을
# 말한다. 순서는 `TRACK_POINT_ORDER` 고정이다(실행마다 흔들리지 않게).
# 줄 이름은 **장치** 이름이다 — 궤적은 관절이 아니라 HMD·건·플래쉬가 지나간 길이다(범례의 "가는 선 =
# 장치가 지나간 길"과 같은 말). 예전에는 "오른손 · 머리 · 왼손"이라 적어 같은 범례 안에서 서로 어긋났다.
TRACK_POINT_NAMES: Final[dict[str, str]] = DEVICE_NAMES
TRACK_POINT_ORDER: Final[tuple[str, ...]] = ("right", "head", "left")


def track_legend_note(layer: Mapping[str, Sequence[Any]]) -> str | None:
    """범례의 궤적 조각 — 그 층에 실제로 그려지는 줄의 **장치** 이름. 그릴 줄이 없으면 None.

    입력은 **그려질 층 자체**다(`_world_layer`/`_serialize_layer`의 산출물). 화면이 점이 하나라도
    있는 이름마다 폴리라인을 하나씩 만드므로, 여기서 같은 조건으로 세면 범례 숫자와 그려진 줄 수가
    한 출처에서 나온다.
    """
    drawn = [
        name
        for name in TRACK_POINT_ORDER
        if any(point is not None for point in (layer.get(name) or ()))
    ]
    if not drawn:
        return None
    names = "·".join(TRACK_POINT_NAMES[name] for name in drawn)
    return f"가는 선 = {names} 궤적"


# 범례가 "실측"으로 선언한 시각 부호(모양). 추정 기하는 이 부호를 쓰지 않는다 — 검사가 이 짝을 고정한다.
LEGEND_MEASURED_SYMBOL: Final = "채운 구"
LEGEND_ESTIMATE_SYMBOL: Final = "속 빈 표식"
# 범례 한 줄의 고정 조각. 궤적 조각(`track_legend_note`)만 그 층에 실제로 그려지는 줄에 따라 달라진다.
LEGEND_NOTES: Final[tuple[str, ...]] = (
    LEGEND_MEASURED_WORD,
    LEGEND_ESTIMATE_WORD,
    LEGEND_DASHED_WORD,
    LEGEND_GUN_WORD,
)


def legend_line(layer: Mapping[str, Sequence[Any]]) -> str:
    """범례 **한 줄**. 채운 구(실측) · 가는 선(궤적) · 속 빈 표식(추정) · 점선 · 줄무늬."""
    parts = [LEGEND_MEASURED_WORD, track_legend_note(layer), *LEGEND_NOTES[1:]]
    return LEGEND_SEPARATOR.join(part for part in parts if isinstance(part, str))


# 대표 스켈레톤 프레임 탐색: 분사 구간 가운데에서 가까운 순서로 이만큼만 FK를 돈다(프레임당 약 0.9 ms).
LEG_SEARCH_MAX: Final = 120

# 다리 고지.
#
# **곧게 선** 자리표시 다리는 기본이 아니다. 곧게 세우면 화면이 없는 자세를 주장한다: 위 세 패스의
# 대표 프레임에서 고른 바닥면 위 머리 실측 높이가 L1 1.51m(서 있음) · L3 0.70m · L6 0.88m이고,
# **이 패스들에서** 작업자는 앉거나 깊게 쭈그리고 있다(머리·왼손 정렬 오차도 평소 50~90mm에서
# 300mm로 치솟는다). 사용자도 "앉아서 칠하는 자세도 많은데(특히 좁은 공간의 경우) 다리를 전부
# 펴도록 만들어 놔서 어색함"이라고 했다.
#
# 다만 그것은 **이 패스들의 사실이지 레벨의 사실이 아니다** — 같은 면 위 레벨 중앙값은 L1 1.534m ·
# L3 1.450m · L6 1.172m이고 p95는 모든 레벨에서 선 키를 회복한다(`composite.py`의 바닥 주석).
# 고른 면은 지면이 맞다.
#
# 그래서 기본은 **자리표시 다리**다(B): 발을 그 바닥면에 세우고(`standing_legs`의 `reach_floor=True`),
# 점선으로 그리고, 고지가 "측정 아님"을 먼저 말한다. 엉덩이가 리그 다리보다 낮으면 **무릎을 접어**
# 세우고(길이 그대로), 높으면 자리표시 다리를 그만큼 **늘려** 세운 뒤 배율과 늘린 길이를 적는다 —
# 발을 띄워 두는 그림은 "작업자가 공중에 떠 있다"는 역시 없는 사실을 주장하기 때문이다.
# 고지는 **어느 몸의 다리인지**를 말한다({who} = 대상 / 기준 / 대상·기준). 두 층의 `leg_check`
# 판정은 자주 갈린다 — 자동 기준으로 맺은 24쌍 중 10쌍(41.7%)에서 갈렸다. 질의 고지만 싣던
# 동안에는 화면이 그림과 반대되는 말을 했다(김경호 L1 #9: 기준 다리 뼈 8개가 그려져 있는데
# "다리는 그리지 않았습니다"라고 적혔다).
# IK 다리는 이제 **토글**이다(B: "IK 다리는 토글로 볼 수 있게"). 기본 그림이 자리표시이므로 IK
# 고지는 따로 선 문장이 아니라 위 자리표시 고지에 붙는 꼬리다.
#
# **토글은 IK 다리가 있으면 열린다.** 예전에는 `leg_check.plausible`이 거짓이면 잠갔고, 그래서
# 김경호 L3·L6처럼 쭈그린 패스에서 체크박스가 죽은 채로 남았다(검증 실측 6조합 중 4칸). 범위 밖
# 다리를 **기본 그림으로** 내세우지 않는다는 판단은 그대로다 — 기본은 여전히 자리표시이고, 범위 밖
# IK는 사람이 켰을 때만, 한 단계 더 흐리게 나온다. 그리고 어느 검사를 얼마나 벗어났는지를 숫자로
# 적는다(`leg_range_reason`): 잠긴 컨트롤은 고장으로 읽히고, 범위 밖이라는 사실은 숨기는 것보다
# 재서 말하는 편이 이 저장소 규칙에 맞는다.
LEGS_IK_AVAILABLE_SUFFIX: Final = " '다리(IK)'를 켜면 시뮬레이터가 지은 다리를 봅니다."
LEGS_IK_OUT_OF_RANGE_SUFFIX: Final = (
    " '다리(IK)'를 켜면 시뮬레이터가 지은 다리를 보지만, 이 패스의 IK 다리는 해부학 범위 밖입니다"
    "({why}) — 그래서 켜도 한 단계 더 흐리게 그립니다."
)
# 범위 밖 사유에 쓰는 검사 이름. 문턱은 `skeleton.LEG_LIMITS`가 하나의 출처다.
LEG_CHECK_LABELS: Final[dict[str, str]] = {
    "knee_flex_deg": "무릎 굴곡",
    "knee_toe_deg": "무릎-발끝",
    "toe_splay_deg": "발끝 벌어짐",
    "leg_tilt_deg": "다리 기울기",
}


def leg_range_reason(legs: skeleton.LegCheck) -> str:
    """IK 다리가 해부학 범위를 **어느 검사에서 얼마나** 벗어났는가.

    화면이 숫자를 다시 만들지 않도록 서버가 문장을 짓는다. 값도 문턱도 한 출처(`leg_check`·
    `LEG_LIMITS`)에서 나오므로, 문턱을 바꾸면 이 문장이 같이 바뀐다.
    """
    parts: list[str] = []
    for key, value in legs.as_dict().items():
        limit = skeleton.LEG_LIMITS.get(key)
        if limit is None or not isinstance(value, float):
            continue
        if not math.isfinite(value):
            parts.append(f"{LEG_CHECK_LABELS[key]} 값 없음")
        elif value > limit:
            parts.append(f"{LEG_CHECK_LABELS[key]} {value:.1f}° > {limit:.0f}°")
    if legs.legs_crossed:
        parts.append("두 다리가 교차")
    return " · ".join(parts) if parts else "범위 밖 검사 없음"
# 다리 관절이 아예 없는 리그(다리 컬럼 없음)의 고지.
LEGS_NO_JOINTS_NOTE: Final = "{who} 패스에는 다리 관절이 없어 다리를 그리지 않았습니다."

# 자리표시 다리 고지(B). 기본 표시가 다시 자리표시 다리가 됐으므로, 화면은 **그것이 자리표시라는
# 것**을 먼저 말해야 한다 — 재지 않은 것을 잰 것처럼 보이게 하지 않는다는 이 저장소의 규칙이다.
#
# **무엇을 했고 무엇을 하지 않았는지**를 말한다. 사용자 결정은 "리그 길이 지키고 발이 뜨게"다:
#   무릎을 접어 세운 경우(엉덩이가 리그 다리보다 낮다): 발이 그 면에 선다. L3·L6이 여기다.
#   못 닿는 경우(엉덩이가 더 높다): **늘리지 않는다**. 리그 길이 그대로 곧게 내리고 발이 뜬다 —
#     실측 김경호 L1 첫 패스는 엉덩이가 바닥면 위 1.045m인데 리그 다리가 0.833m라 0.21m 뜬다.
#     예전에는 여기서 다리를 1.25배 늘리고 그 배율을 고지했는데, 늘린 다리는 "이 사람의 다리가
#     이만큼이다"라는 없는 사실을 주장한다. 뜬 높이는 다리가 아니라 **틈**이므로 문구가 그것을
#     틈이라고 적고, 같은 숫자를 발끝→바닥 점선 보조선이 화면에서 보인다(`placeholder_floor_gaps`).
#   발이 그 면보다 아래인 경우: 무릎을 다 접어도 닿을 수 없을 만큼 엉덩이가 낮다. 깊이를 적는다.
# 숫자 자리({floor}·{leg}·{hip}·{gap}·{drop})는 서버가 채워 넣고 `{who}`만 층 이름으로 남긴다 —
# 화면은 이 문자열을 그대로 쓰고 다시 계산하지 않는다.
LEGS_PLACEHOLDER_ON_FLOOR_NOTE: Final = (
    "{who} 다리는 자리표시입니다(측정 아님) — 추적 장치가 없어 발을 바닥면(수직 {floor}m)에 세우고 "
    "남는 길이는 무릎으로 접었습니다. 뼈 길이는 리그 그대로입니다."
)
LEGS_PLACEHOLDER_OFF_FLOOR_NOTE: Final = (
    "{who} 다리는 자리표시입니다(측정 아님) — 엉덩이가 바닥면(수직 {floor}m) 위 {hip}m인데 리그 "
    "다리는 {leg}m라, 길이를 늘리지 않고 리그 그대로 곧게 내렸습니다. 발이 바닥면에서 최대 {gap}m "
    "떠 있고, 뜬 발마다 발끝에서 바닥까지 점선으로 그 간격을 표시했습니다."
)
# 무릎을 다 접어도 닿지 않는 경우 — 엉덩이가 그 면에 너무 가깝거나 아래에 있다.
LEGS_PLACEHOLDER_UNDER_FLOOR_NOTE: Final = (
    "{who} 다리는 자리표시입니다(측정 아님) — 리그 길이 그대로 곧게 내려 발이 바닥면(수직 {floor}m)"
    "보다 {drop}m 아래입니다."
)
# 바닥면을 못 고른 레벨(벽 메시 없음)의 자리표시 고지. 바닥이 없으면 세울 자리도 없다.
LEGS_PLACEHOLDER_NO_FLOOR_NOTE: Final = (
    "{who} 다리는 자리표시입니다(측정 아님) — 추적 장치가 없고, 이 레벨은 벽 메시가 없어 발을 세울 "
    "바닥면을 고르지 못했습니다."
)
LEG_NOTE_WHO: Final[dict[str, str]] = {"query": "대상", "reference": "기준", "both": "대상·기준"}

# 바닥 기준 머리 높이. 이 한 줄이 "지금 서 있나 앉아 있나"를 말한다 — 문턱은 두지 않고 실측 높이를
# 그대로 적는다(`composite.py`의 바닥 주석 참고). 실측: L1 1.508m · L3 0.696m · L6 0.879m.
#
# 이 문장은 이제 다리 고지에 붙지 않고 **히어로 패널 한 자리**(`head_height`)에만 산다. 다리 고지가
# 두 층으로 늘어난 뒤에도 같은 문장이 두 번 실리지 않고, 사용자 불만("텍스트가 너무 많아서")을
# 키우지 않으면서 두 몸의 높이를 나란히 보여 준다.
HEAD_HEIGHT_NOTE: Final = (
    "바닥면(수직 {floor:.2f}m) 위 머리 실측 마커입니다 — 서 있는 높이가 아니면 앉거나 쭈그린 자세입니다."
)
HEAD_HEIGHT_NO_FLOOR: Final = "이 레벨은 벽 메시가 없어 바닥면을 고르지 못했습니다."

# 오른팔 고지(S5). 아바타 IK 손과 트래킹된 건의 간격이 작업자·레벨 불문 중앙값 571mm다(적합 상수 0.567m와
# 일치). 두 차례 사전등록 측정이 그 원인을 **데이터**로 확정했다 — 벤더 오른팔 솔버가 트래커 거리와 무관하게
# 팔꿈치를 ~91% 편 채 고정한다(`right_arm` 머리말). 그래서 화면은 CSV 오른 전완·손을 그리지 않고 오른어깨에서
# 건 손잡이까지 팔을 다시 세운다(표시 전용). 이 고지는 그 사실과 원본의 한계를 **접힌 상세에서** 한 번 말한다 —
# 3D 안에는 방법·길이 문구를 걸지 않는다(사용자: "불필요한 텍스트는 넣지 말 것"). 적합값을 "보정"이라고
# 부르지 않는다.
LOW_CONFIDENCE_NOTE: Final = (
    "오른팔은 원본의 팔 관절 대신 오른어깨에서 건 손잡이까지 다시 세웠습니다 — 원본 데이터에서 아바타 손 IK와 "
    "트래킹된 건이 어긋나 있고(전 작업자·레벨로 맞춘 대표값 {gap:.2f}m), 캡처 파이프라인의 손 IK↔컨트롤러 "
    "매핑 문제입니다."
)

# fit.notes 항목 이름(`fit.note_keys`, 같은 순서). 화면은 위치가 아니라 이 이름으로 짝짓는다.
NOTE_KEYS: Final[tuple[str, ...]] = ("avatar_estimate", "gravity", "legs", "torso", "grip_limit")
# 캔버스 아래에서 접어 두는 고지 — **전부**다. 사용자가 "텍스트가 너무 많아서 무슨 말을 하고 싶은지
# 모르겠음", 그리고 "UI는 간단한 편이 좋음"이라고 했다. 무엇이 추정인지는 모양과 범례 한 줄이 말하고,
# 문장은 한 번 눌러 펼치는 상세에 둔다.
NOTES_COLLAPSED: Final[tuple[str, ...]] = NOTE_KEYS


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


def arm_for(
    placed: skeleton.Placement, offsets: Mapping[str, Vec3], representative: Representative
) -> right_arm.RightArm | None:
    """이 배치의 표시용 오른팔 — 오른어깨에서 이 프레임 **건 손잡이**까지(`right_arm` 머리말: 벤더 오른팔
    솔버가 팔꿈치를 편 채 고정한다). 어깨는 배치된 몸의 것이고 뼈 길이는 리그 그대로다. 채점은 이 값을 읽지
    않는다 — 실측 장치만 쓴다."""
    return right_arm.right_arm(
        placed.joints, offsets, representative.row, representative.gun, representative.frame.right
    )


def displayed_joints(placed: skeleton.Placement, arm: right_arm.RightArm | None) -> dict[str, Vec3]:
    """화면에 그리는 관절: 배치된 몸 + 건 손잡이로 세운 오른 팔꿈치·손 + 그려 넣은 목·윗가슴."""
    joints = dict(placed.joints)
    if arm is not None:
        joints[right_arm.ELBOW] = arm.elbow
        joints[right_arm.HAND] = arm.hand
    # 그려 넣는 두 관절은 **선분 위 내분점**이라 배치 뒤에 계산해도 같다(방향을 지어내지 않는다).
    joints.update(body_proportions.inserted_joints(joints))
    return joints


# 몸을 세우는(회전·평행이동을 푸는) 앵커: **머리(HMD)와 왼손(플래쉬)**. 오른손 앵커는 CSV FK 오른손 관절에
# 손→트래커 ~0.57m 장착 오프셋을 더한 것인데, 둘 다 쓰지 말라고 확정된 재료다(`right_arm` 머리말: 벤더
# 오른팔 솔버가 팔꿈치를 편 채 고정하고, 그 오프셋은 분사구 길이가 아니다). 그 앵커로 몸을 끌면 오른어깨가
# 건 손잡이 쪽으로 딸려 가 팔이 접힌다 — 실측(김경호·전형기·손영석 × L1/3/6, 624프레임): 세 앵커로 풀면
# 머리·왼손 어긋남 중앙값 8.6·11.5cm, 오른어깨→손잡이 0.15m 미만 22.3%; 두 앵커로 풀면 3.9·3.9cm, 19.1%.
# 두 앵커 중 하나가 없으면 쓸 수 있는 앵커 전부로 푼다(`skeleton.place_oriented`). 채점은 이 배치를 읽지 않는다.
BODY_FIT_POINTS: Final[tuple[str, ...]] = ("head", "left")


def place_frame(sample: skeleton.FrameSample, mounts: skeleton.Mounts) -> skeleton.Placement | None:
    """한 프레임 배치. 강체 변환만 — 관절을 실측 좌표로 덮어쓰지 않는다(뼈 길이 보존). 몸은
    `BODY_FIT_POINTS`로 세운다."""
    return skeleton.place_oriented(
        sample.local_pos, sample.local_rot, sample.measured, mounts, fit_points=BODY_FIT_POINTS
    )


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


# 점선 보조선을 그리는 **가장 작은 틈**(m). 부동소수 잡음(1e-17)은 틈이 아니다 — 길이 0인 선과
# "간격 0cm"라는 글자를 화면에 올리지 않는다. 실데이터의 틈은 0.14~0.42m라 이 문턱에 걸리지 않는다.
FLOOR_GAP_MIN_M: Final = 1e-9


def _floor_gaps(placeholder: skeleton.PlaceholderLegs | None) -> dict[str, float]:
    """발목 이름 → 바닥면 **위로 뜬** 간격(m). 닿은 발도, 면보다 아래로 내려간 발도 여기 없다.

    문구와 보조선이 같은 이 한 곳에서 나온다 — 두 곳에서 따로 고르면 화면이 말과 그림으로 갈린다.
    """
    heights = None if placeholder is None else placeholder.feet_above_floor_m
    if not heights:
        return {}
    return {name: value for name, value in heights.items() if value > FLOOR_GAP_MIN_M}


def _floor_gap_guides(placeholder: skeleton.PlaceholderLegs | None) -> list[dict[str, object]]:
    """뜬 발마다 **발끝(없으면 발목)→바로 아래 바닥면**의 두 끝점과 그 옆에 적을 간격 문자열.

    화면이 3D 안에서 숫자를 만들지 않도록 서버가 끝점도 글자도 다 준다. 끝 점은 그 관절에서 월드
    수직 성분만 바닥면 높이로 바꾼 것이라 선이 정확히 수직이고, **선의 길이가 곧 적힌 간격**이다.
    발끝은 발목과 같은 높이(수평)라 어느 쪽을 잡아도 길이가 같다.

    글자는 "간격"이라고 말한다 — 이 선은 잰 값이 아니라 자리표시 발과 바닥 사이의 틈이다.
    """
    floor = None if placeholder is None else placeholder.floor_vertical
    gaps = _floor_gaps(placeholder)
    if placeholder is None or floor is None or not gaps:
        return []
    guides: list[dict[str, object]] = []
    for _hip, _knee, ankle, toe in skeleton.PLACEHOLDER_LEG_CHAIN:
        gap = gaps.get(ankle)
        if gap is None:
            continue
        # 어느 관절이 발끝인지도 **서버(리그 사슬)**가 정한다 — 화면이 이름으로 짐작하지 않는다.
        name = toe if toe in placeholder.joints else ankle
        start = placeholder.joints.get(name)
        if start is None:
            continue
        end = list(start)
        end[world_axes.UP_INDEX] = floor
        guides.append(
            {
                "joint": name,
                "from": list(start),
                "to": end,
                "gap_m": gap,
                "label": f"간격 {gap * 100.0:.0f}cm",
            }
        )
    return guides


def _placeholder_legs(
    placed: skeleton.Placement,
    representative: Representative,
    offsets: Mapping[str, Vec3],
    floor: float | None,
) -> skeleton.PlaceholderLegs | None:
    """이 배치의 자리표시 다리(B). 바닥면이 있고 엉덩이가 리그 다리보다 낮으면 무릎이 남는 길이를
    접어 발을 그 면에 세운다. 엉덩이가 더 높으면 **늘리지 않고** 리그 길이 그대로 내려 발이 뜬다 —
    뜬 간격은 `feet_above_floor_m`이 발마다 싣고, 화면이 점선 보조선으로 그 틈을 보인다.

    발끝 방향은 대표 프레임의 **골반→착탄점** 수평 성분이다(진단용 접촉 시트가 쓰던 것과 같은 방향).
    골반이나 착탄점이 없으면 방향을 지어내지 않고 `None`을 넘겨 발끝 관절을 싣지 않는다.
    """
    pelvis = placed.joints.get("Pelvis")
    hit = representative.frame.hit
    forward = None if pelvis is None or hit is None else [hit[i] - pelvis[i] for i in range(3)]
    return skeleton.standing_legs(placed.joints, offsets, forward, floor=floor, reach_floor=True)


def _torso_length_m(offsets: Mapping[str, Vec3]) -> float:
    return body_proportions.torso_length_m(offsets)


def _body_fit_points(placed: skeleton.Placement) -> set[str]:
    """몸을 세운 앵커. `BODY_FIT_POINTS` 중 이 배치에 있는 것이 `skeleton.MIN_ANCHORS`개 미만이면
    배치가 앵커 전부로 풀렸다(`skeleton.place_oriented`)."""
    present = {fit.point for fit in placed.anchors}
    chosen = present & set(BODY_FIT_POINTS)
    return chosen if len(chosen) >= skeleton.MIN_ANCHORS else present


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
    who: str = LEG_NOTE_WHO["query"],
    arm: right_arm.RightArm | None = None,
) -> dict[str, object]:
    gravity = placed.gravity_correction_deg
    faded = gravity > FADE_GRAVITY_DEG
    notes = [AVATAR_NOTE]
    note_keys = ["avatar_estimate"]
    if faded:
        notes.append(f"실측 회전과 수직 가정이 {gravity:.0f}° 어긋나 몸을 중력 방향으로 세워 그렸습니다(흐리게 표시).")
    else:
        notes.append(f"몸의 수직은 중력 방향으로 맞춘 값입니다(맞춘 각 {gravity:.0f}°).")
    note_keys.append("gravity")

    # --- 다리(B) -----------------------------------------------------------------
    # 기본 표시는 **자리표시 다리**다(사용자 정책): 점선으로 그리고, 벽 메시가 있으면 발을 바닥면에
    # 세우며(무릎이 남는 길이를 접는다), 시뮬레이터 IK 다리는 토글로 본다. 자리표시는 측정이 아니므로
    # 고지가 **먼저 그 사실을 말한다** — 닿지 못한 경우에도 숨기지 않고 얼마나 모자라는지 적는다.
    legs = None if chosen is None else chosen.legs
    head_above = None if floor is None else _head_above_m(placed, floor)
    placeholder = _placeholder_legs(placed, representative, offsets, floor)
    # 고지는 **이 층의 판정**을 말한다({who} = 대상/기준). 두 층이 같은 문장이면 뷰가 한 줄로 합친다.
    if legs is None or placeholder is None:
        leg_template = LEGS_NO_JOINTS_NOTE
    elif placeholder.floor_vertical is None:
        leg_template = LEGS_PLACEHOLDER_NO_FLOOR_NOTE
    elif placeholder.reaches_floor is not True and _floor_gaps(placeholder):
        # 늘리지 않아 발이 떴다. 적는 숫자는 **발에서 직접 잰 틈**이고, 같은 값이 보조선으로도 간다.
        leg_template = (
            LEGS_PLACEHOLDER_OFF_FLOOR_NOTE.replace("{floor}", f"{placeholder.floor_vertical:.2f}")
            .replace("{hip}", f"{placeholder.hip_above_floor_m or 0.0:.2f}")
            .replace("{leg}", f"{placeholder.rig_leg_length_m or 0.0:.2f}")
            .replace("{gap}", f"{max(_floor_gaps(placeholder).values()):.2f}")
        )
    elif placeholder.reaches_floor is not True:
        # 뜬 발이 하나도 없는데 닿지도 못했다 — 발이 그 면보다 아래다.
        leg_template = LEGS_PLACEHOLDER_UNDER_FLOOR_NOTE.replace(
            "{floor}", f"{placeholder.floor_vertical:.2f}"
        ).replace("{drop}", f"{placeholder.feet_below_floor_m or 0.0:.2f}")
    else:
        leg_template = LEGS_PLACEHOLDER_ON_FLOOR_NOTE.replace("{floor}", f"{placeholder.floor_vertical:.2f}")
    # IK 토글이 무엇을 보여 주는지 같은 줄에서 말한다 — 범위 밖이면 그 사실과 숫자를 붙인다.
    if legs is not None and placeholder is not None:
        leg_template += (
            LEGS_IK_AVAILABLE_SUFFIX
            if legs.plausible
            # `{who}`는 뷰가 층 이름으로 채운다 — `{why}`만 여기서 미리 메운다(format이 모르는 칸).
            else LEGS_IK_OUT_OF_RANGE_SUFFIX.replace("{why}", leg_range_reason(legs))
        )
    notes.append(leg_template.format(who=who))
    note_keys.append("legs")

    notes.append(
        f"이 CSV 리그에는 목·윗가슴 관절이 없어 골반→머리를 같은 작업자 BVH의 뼈 길이 비례로 다시 재 "
        f"{_torso_length_m(offsets):.2f}m로 그리고, 빠진 목·윗가슴을 그 선분 위에 그려 넣었습니다 — "
        "실측이 아니라 추정입니다."
    )
    note_keys.append("torso")

    # --- 오른팔 고지(S5) — 원본 손 IK↔건 어긋남이 파지 상한을 넘으면 접힌 상세에 한 번 적는다 --------
    right_norm = None if mounts.right is None else math.hypot(*mounts.right.offset)
    limit = grip_limit_m.get(representative.gun)
    low_confidence = right_norm is not None and limit is not None and right_norm > limit
    if low_confidence:
        notes.append(LOW_CONFIDENCE_NOTE.format(gap=right_norm))
        note_keys.append("grip_limit")

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
        "gravity_display": f"수직 맞춤 {gravity:.0f}°",
        "faded": faded,
        "leg_tilt_deg": placed.leg_tilt_deg,
        "rotation_residual_deg": dict(placed.rotation_residual_deg),
        "rotation_residual_label": ROTATION_RESIDUAL_LABELS[kind],
        # 다리 고지의 **틀**({who} 미치환). 두 층이 같은 문장이면 뷰가 한 줄로 합친다.
        "leg_note_template": leg_template,
        "leg_note_who": who,
        # 바닥 기준 머리 높이(S4). 앉은 자세인지가 이 한 숫자에서 읽히므로 문구 말고 값으로도 낸다 —
        # 화면은 이 표시 문자열을 눈에 띄는 자리에 그대로 옮긴다(다시 계산하지 않는다).
        "floor_vertical": floor,
        "head_above_floor_m": head_above,
        "head_above_floor_display": None if head_above is None else f"{head_above:.2f}m",
        # 원본 손 IK↔건 어긋남(S5)이 파지 상한을 넘었는가. 화면은 이제 그 CSV 오른 전완·손을 그리지 않는다 —
        # 오른팔은 건 손잡이로 다시 세운다(`right_arm`). 그래서 흐리게 칠할 뼈 목록은 더 없다.
        "low_confidence": low_confidence,
        "low_confidence_gap_m": right_norm if low_confidence else None,
        # 표시용 오른팔(오른어깨 → 건 손잡이 두 마디 IK). 채점은 이 값을 읽지 않는다. 닿지 못했으면
        # `reach_gap`(손끝 → 손잡이)을 화면이 점선으로만 잇는다 — 숫자·문구는 없다.
        "right_arm": None if arm is None else arm.as_view(),
        "anchors": [
            {
                "point": fit.point,
                "joint": fit.joint,
                "joint_position": list(fit.joint_position),
                "predicted_marker": list(fit.predicted_marker),
                "measured_marker": list(fit.measured_marker),
                "miss_cm": fit.miss_m * 100.0,
                # 이 앵커로 몸을 세웠는가(`BODY_FIT_POINTS`). 세우지 않은 앵커(오른손: CSV FK 손 + 장착
                # 오프셋)의 어긋남은 그리지 않는 팔에 대한 숫자라 왼쪽 패널에 적지 않는다(`miss_display` 없음).
                "body_fit": fit.point in _body_fit_points(placed),
                # 숫자는 왼쪽 패널에만 뜬다(S6) — 3D 안에는 이 문자열을 걸지 않는다.
                "miss_display": f"{'적합 잔차 ' if kind == MOUNTS_PASS_FIT else ''}{fit.miss_m * 100.0:.1f}cm"
                if fit.point in _body_fit_points(placed)
                else None,
                # 장치 모양. 건 막대의 손 쪽 끝은 **손잡이**다(`right_arm`) — 추정 관절도 예측 마커도 아니다.
                "device": device_block(
                    fit.point,
                    arm.grip if fit.point == "right" and arm is not None else fit.joint_position,
                    fit.measured_marker,
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
            # 기본 표시는 **자리표시 다리**다(B). IK는 토글로만 보고, 자리표시를 세울 수 없을 때만 숨긴다.
            "display_default": "placeholder" if placeholder is not None else "hidden",
            # 자리표시 기하(월드). 화면은 이 점을 **점선**으로 잇고 "측정 아님"이라고 적는다.
            # 닿았는지와 그 이유가 되는 두 길이를 같이 싣는다 — 화면이 숫자를 다시 만들지 않는다.
            "placeholder_joints": None
            if placeholder is None
            else {name: list(position) for name, position in placeholder.joints.items()},
            # 자리표시 뼈의 짝도 **서버가 준다** — 화면이 리그 이름으로 연결 규칙을 지어내지 않는다.
            "placeholder_bones": None
            if placeholder is None
            else [[a, b] for chain in skeleton.PLACEHOLDER_LEG_CHAIN for a, b in zip(chain, chain[1:])],
            "placeholder_reaches_floor": None if placeholder is None else placeholder.reaches_floor,
            # 발목마다 바닥면 위 높이(m, 음수면 아래). 닿은 발은 0이다 — 화면·검사가 이 값을 그대로
            # 읽는다(다시 재지 않는다).
            "placeholder_feet_above_floor_m": None
            if placeholder is None or placeholder.feet_above_floor_m is None
            else dict(placeholder.feet_above_floor_m),
            # 뜬 발마다 **발끝→바닥** 점선 보조선의 두 끝점과 그 옆에 적을 간격 문자열. 화면은 이
            # 목록을 그대로 그린다 — 길이를 다시 재지도, cm로 다시 반올림하지도 않는다. 이것은
            # 측정이 아니라 **틈 표시**다: 선은 점선이고 글자가 "간격"이라고 말한다.
            "placeholder_floor_gaps": _floor_gap_guides(placeholder),
            "hip_above_floor_m": None if placeholder is None else placeholder.hip_above_floor_m,
            "rig_leg_length_m": None if placeholder is None else placeholder.rig_leg_length_m,
            "floor_vertical": floor,
            "head_above_floor_m": head_above,
        },
        # IK 다리를 기본으로 켜는가 — **아니다**(B). 기본은 자리표시이고 IK는 토글이다.
        # 이름이 곧 뜻이어야 하므로 "다리가 보이는가"와 섞지 않는다: 다리는 자리표시로 늘 보인다.
        "legs_visible_default": False,
        # 그 토글을 쓸 수 있는가 — **IK 다리가 있으면 쓸 수 있다**(B: "IK는 토글로 볼 수 있게").
        # 범위 밖인지는 아래 `legs_within_range`가 따로 말한다: 화면은 범위 밖 IK를 한 단계 더
        # 흐리게 그리고, 고지가 어느 검사를 얼마나 벗어났는지 숫자로 적는다. 기본 그림은 그대로
        # 자리표시다 — 범위 밖 다리가 **저절로** 나타나는 화면은 없다.
        "legs_toggle_enabled": legs is not None,
        "legs_within_range": bool(legs is not None and legs.plausible),
    }


def skeleton_block(
    raw: RawPass,
    representative: Representative | None = None,
    mounts_table: MountsTable | None = None,
    geometry: composite.MeshGeometry | None = None,
    *,
    who: str = LEG_NOTE_WHO["query"],
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

    arm = arm_for(placed, offsets, representative)
    joints = displayed_joints(placed, arm)
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
            who=who,
            arm=arm,
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


# 바닥 격자(사용자: "3D viewer grid가 있으면 함"). 크기와 눈 간격(m). 격자는 **공간감**만 준다 — 벽 메시·궤적·
# 스트로크와 헷갈리지 않게 화면이 옅게 그리고, 카메라 맞춤(`framingPoints`)에는 넣지 않는다.
GRID_SIZE_M: Final = 4.0
GRID_CELL_M: Final = 0.5


def grid_block(
    query_joints: Mapping[str, Sequence[float]] | None,
    reference_joints: Mapping[str, Sequence[float]] | None,
    fit: Mapping[str, Any] | None,
    reference_fit: Mapping[str, Any] | None,
) -> dict[str, object] | None:
    """바닥 격자를 깔 높이와 자리. 그릴 몸이 없으면 None.

    높이: 벽 메시가 있으면 서버가 이미 고른 **바닥면 수직**(`fit.floor_vertical`)이다. 없으면 **그려진 발**
    — 기본 표시인 자리표시 다리 관절(없으면 스켈레톤 관절) — 의 가장 낮은 수직값이다. 자리는 두 몸 골반의
    수평 평균이다(골반이 없으면 관절 전체의 평균).
    """
    fits = [f for f in (fit, reference_fit) if f]
    floors = [f.get("floor_vertical") for f in fits if isinstance(f.get("floor_vertical"), (int, float))]
    bodies = [j for j in (query_joints, reference_joints) if j]
    if not bodies:
        return None
    if floors:
        vertical = float(floors[0])
        source = "floor"
    else:
        feet: list[Sequence[float]] = []
        for f in fits:
            placeholder = (f.get("legs") or {}).get("placeholder_joints") or {}
            feet.extend(placeholder.values())
        if not feet:
            feet = [p for joints in bodies for p in joints.values()]
        vertical = min(world_axes.vertical_of(tuple(p)) for p in feet)  # type: ignore[arg-type]
        source = "feet"
    anchors = [joints["Pelvis"] for joints in bodies if "Pelvis" in joints] or [
        p for joints in bodies for p in joints.values()
    ]
    centre = [sum(float(p[i]) for p in anchors) / len(anchors) for i in range(3)]
    centre[world_axes.UP_INDEX] = vertical
    return {
        "vertical": vertical,
        "source": source,
        "center": centre,
        "up_index": world_axes.UP_INDEX,
        "size_m": GRID_SIZE_M,
        "cell_m": GRID_CELL_M,
    }


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
    reference_skeleton: SkeletonResult | None = None,
    mesh: object = _UNSET,
) -> dict[str, object]:
    """화면이 그릴 층·교정·메시·가중치. 좌표는 전부 캡처(월드) 프레임이다.

    `corrected` 층과 `corrections`는 **한 목록**(`corrections`, 없으면 여기서 한 번
    `measured_score.issued_axes`)에서 파생된다 — 두 갈래로 나눠 계산하면 그림이 개선율·문장과
    다른 이동을 보일 수 있다(요구사항 1). `compare()`는 `measured.axes`와 같은 `issued`를 넘긴다.

    전신 스켈레톤은 **두 층에 다** 얹는다(S1). 사용자가 원한 그림이 "기준 스켈레톤과 질의
    스켈레톤을 같이 그리고 유사도가 떨어지는 조인트일수록 색상을 진하게"이기 때문이다. 기준도
    질의와 **같은 경로**(`skeleton_block`)로 세운다 — 두 경로를 두면 같은 사람이 층에 따라 다른
    몸이 된다. `query_raw`도 `skeleton_result`도 없으면(테스트처럼 원본 행 없이 층만 조립하는
    호출) 스켈레톤 없이 점·화살표만 낸다.
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

    reference_layer_view: dict[str, object] = dict(_serialize_layer(reference_layer))
    reference_fit: dict[str, object] | None = None
    reference_fit_reason: str | None = None
    reference_fit_note: str | None = None
    if reference_skeleton is not None:
        reference_fit_reason = reference_skeleton.fit_reason
        reference_fit_note = reference_skeleton.fit_note
        if reference_skeleton.joints is not None:
            reference_layer_view["joints"] = reference_skeleton.joints
            reference_layer_view["bones"] = reference_skeleton.bones
            reference_fit = reference_skeleton.fit

    return {
        "grid_n": grid_n,
        "layers": {
            "query": query_layer_view,
            "reference": reference_layer_view,
            "corrected": _serialize_layer(corrected_layer),
        },
        "corrections": _correction_arrows(corrections, query_layer, markers),
        "mesh": mesh_block(level_text) if mesh is _UNSET else mesh,
        "weights": weights_view,
        "fit": fit,
        "fit_reason": fit_reason,
        "fit_note": fit_note,
        # 기준 쪽 적합. 질의와 **같은 모양**이라 화면이 같은 규칙으로 그린다(흐린 다리·저신뢰 구간).
        "reference_fit": reference_fit,
        "reference_fit_reason": reference_fit_reason,
        "reference_fit_note": reference_fit_note,
        # 관절마다 두 스켈레톤이 벌어진 거리. 화면은 이 값으로 진하기만 정한다(S1).
        # 눈금은 **그리는 관절**에서만 잡는다 — 두 층 다 다리를 숨기면 다리 관절을 눈금에서 뺀다.
        "deviation": deviation_block(
            skeleton_result, reference_skeleton, hidden_joints=hidden_joints(fit, reference_fit)
        ),
        # 다리 고지(S4). **두 층 몫을 다 낸다** — 질의 것만 싣던 동안에는 기준 다리가 그려져 있는데
        # "다리는 그리지 않았습니다"라고 적히는 일이 24쌍 중 10쌍에서 났다. 같은 문장이면 한 줄로 합친다.
        "leg_notes": leg_notes(fit, reference_fit),
        # 바닥 기준 머리 높이(S4). 두 몸의 높이를 나란히 한 줄로 — 한 값만 보이면 어느 몸인지 모른다.
        "head_height": head_height_block(fit, reference_fit),
        # 바닥 격자 — 바닥면이 있으면 그 높이, 없으면 그려진 발의 가장 낮은 점.
        "grid": grid_block(
            query_layer_view.get("joints"),  # type: ignore[arg-type]
            reference_layer_view.get("joints"),  # type: ignore[arg-type]
            fit,
            reference_fit,
        ),
        # 살(반투명 부피)의 부위·두께. 머리 지름과 같은 분모로 이 리그 크기에 맞춘다 —
        # 화면이 부위 이름으로 규칙을 지어내지 않게 서버가 준다.
        "flesh": body_proportions.flesh_block(
            fit.get("head_diameter_m") if isinstance(fit, Mapping) else None
        ),
        # 범례는 한 줄이다(사용자: "UI는 간단한 편이 좋음").
        "legend_notes": [legend_line(query_layer)],
        "display_errors": [],
        "composite": None,
        "surface_hits": [],
        "stroke": composite.stroke_runs(None, []),
        # 기준 쪽 벽면 스트로크(S2). 사용자가 "기준 스트로크(벽면 착탄점)이 보이지 않음"이라고 했다.
        "reference_surface_hits": [],
        "reference_stroke": composite.stroke_runs(None, []),
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
            "score_delta": instructions.score_delta_display(
                measured.get("score"), measured.get("score_after")
            ),
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

    def skeleton_for(raw: RawPass, who: str) -> SkeletonResult:
        """질의·기준을 **같은 경로**로 세운다(S1). 어느 쪽이 실패해도 표시 블록 격리는 그대로다.

        `who`만 층마다 다르다 — 다리 고지가 **자기 층의 판정**을 말하게 하는 이름이다(S4).
        """
        rep_ok, representative = guarded("skeleton", representative_frame, raw)
        built_ok = False
        built: SkeletonResult | None = None
        if rep_ok:
            build = functools.partial(skeleton_block, who=who)
            built_ok, built = guarded("skeleton", build, raw, representative, None, geometry)
        if not built_ok or built is None:
            return _skeleton_failure("display_error", representative if rep_ok else None)
        return built

    skeleton_result = skeleton_for(query_raw, LEG_NOTE_WHO["query"])
    reference_skeleton = skeleton_for(reference_raw, LEG_NOTE_WHO["reference"])

    view = view_block(
        query_frames,
        reference_frames,
        weights=weights,
        level_text=query.level,
        grid_n=samples,
        corrections=issued,
        skeleton_result=skeleton_result,
        reference_skeleton=reference_skeleton,
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
    # 기준 쪽도 **같은 함수로** 낸다(S2). 없는 접촉을 만들지 않는다 — 자르는 규칙이 하나다.
    ok, ref_hits = guarded("composite", composite.surface_hits, reference_skeleton.representative)
    view["reference_surface_hits"] = ref_hits if ok else []
    ok, ref_stroke = guarded("composite", composite.stroke_runs, geometry, view["reference_surface_hits"])
    view["reference_stroke"] = ref_stroke if ok else composite.stroke_runs(None, [])
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
    query_frames, query_inputs = pass_key_inputs(query_raw, samples)
    reference_raw = source.load(reference)
    reference_frames, reference_inputs = pass_key_inputs(reference_raw, samples)
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
    # 동작 키 일치는 **완화 사다리**를 탄다(S8) — 기준을 고르는 쪽(`reference_candidates`)과
    # **같은 사다리, 같은 순서**다. 두 쪽이 갈리면 화면이 한 자리에서 서로 반대되는 말을 한다:
    # 기준 줄은 "동작 키 완화: 벽면 4×4 칸까지 일치"를 띄우는데 [평가]는 "동작 키가 달라 비교하지
    # 않습니다."만 남기고 3D를 아예 열지 않았다(검증 라운드가 L1 #8·L2 #0·L2 #6·L6 #11·L3 #10
    # 다섯 건 전부에서 재현했다). 완화로 고른 기준이 비교에서 죽으면 S8이 푼 것이 아무것도 없다.
    #
    # 어느 단에서 만났는지는 **반드시** 응답에 싣는다(`key_match`). 엄밀 일치와 느슨한 일치를
    # 구별하지 못하면 화면이 "같은 동작"이라고 거짓을 말한다.
    key_match = motion_key_match(query_inputs, reference_inputs, params)
    if key_match["granularity"] is None:
        return {
            "status": "unavailable",
            "reason": "motion_key_mismatch",
            "query_key": query_key,
            "reference_key": reference_key,
            "key_match": key_match,
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
        # 어느 단에서 두 키가 만났는가(S8). 화면은 이 값으로 엄밀·완화를 구별해 말한다.
        "key_match": key_match,
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


# --- 기준 고르기의 완화 사다리(S8) ---------------------------------------------------
#
# 채점 쪽(`compare`)은 산포를 찾을 때 uv8→uv4→uv2→uv 무시 사다리를 훑는데(`_reference_keys`,
# `SpreadReference.resolve`) 기준 **패스**를 고르는 쪽에는 사다리가 없어 `key_from(...)` 완전
# 일치만 봤다. 그래서 "유사한 기준이 없다"가 자주 떴다. 실측(김경호 20250831, 레벨별 앞 12패스
# = 48패스): 엄밀 일치로 후보가 0인 패스가 11건인데, 사다리를 내리면 그중 7건이 살아난다
# (L1 #8 → uv2 6건 · L2 #0/#1 → uv4 30건 · L2 #6 → uv4 1건 · L2 #9 → uv4 1건 ·
#  L3 #10 → uv_off 122건 · L6 #11 → uv4 3건). 남는 4건은 어느 층에서도 전형기 패스가 없다.
#
# **어느 단계에서 맞았는지 화면이 반드시 말한다.** 엄밀 일치와 느슨한 일치를 구별하지 못하면
# 화면이 "같은 동작"이라고 거짓을 말하게 된다.
REFERENCE_LADDER_LABELS: Final[dict[str, str]] = {
    "uv_off": "벽면 어느 칸인지를 빼고 일치",
}


def reference_ladder_label(granularity: str) -> str:
    """사다리 한 단의 사람이 읽는 이름. uv 격자 층은 칸 수를 그대로 적는다."""
    known = REFERENCE_LADDER_LABELS.get(granularity)
    if known is not None:
        return known
    grid = granularity[2:]
    return f"벽면 {grid}×{grid} 칸까지 일치"


def reference_ladder(params: Any) -> list[tuple[str, Any]]:
    """(층 이름, 그 층의 KeyParams). 첫 단은 **호출자가 준 params 그대로**다.

    uv 격자를 호출자 값에서부터 내려가며 느슨하게 한다. `uv_off`는 uv 성분을 na로 바꾼 키라
    params가 아니라 키 변환으로 만든다(`build_spread_reference.key_without_uv`) — 산포 사다리가
    쓰는 바로 그 변환이다. 호출자가 uv_grid를 4로 낮춰 뒀으면 8단은 아예 없다(엄밀보다 느슨한
    단을 "엄밀"이라고 부르지 않는다).
    """
    strict = int(getattr(params, "uv_grid", 8))
    grids = [strict] + [grid for _layer, grid in build_spread_reference.GRANULARITIES if 0 < grid < strict]
    rungs: list[tuple[str, Any]] = []
    for grid in grids:
        rungs.append((f"uv{grid}", params if grid == strict else replace(params, uv_grid=grid)))
    return rungs


def _rung_key(inputs: KeyInputs, granularity: str, rung_params: Any, loosest: Any) -> str:
    """한 단의 동작 키. `uv_off` 단은 params가 아니라 키 변환으로 만든다(산포 사다리와 같은 변환)."""
    if granularity == "uv_off":
        return build_spread_reference.key_without_uv(key_from(inputs, loosest))
    return key_from(inputs, rung_params)


def motion_key_match(
    query_inputs: KeyInputs | None, reference_inputs: KeyInputs | None, params: Any
) -> dict[str, object]:
    """두 패스의 동작 키가 **어느 단에서** 만나는가. 기준을 고르는 쪽과 같은 사다리다.

    `reference_candidates`가 후보를 낸 단과 `compare`가 비교를 허락하는 단이 같아야 한다 —
    갈리면 화면이 "동작 키 완화: 벽면 4×4 칸까지 일치"를 띄우면서 동시에 "동작 키가 달라
    비교하지 않습니다."를 적는다(검증 라운드가 재현한 바로 그 상태).

    `granularity`가 None이면 어느 단에서도 만나지 않았다는 뜻이다 — 기본값이나 요청한 단을
    대신 채우지 않는다. 훑은 단은 전부 `ladder`에 적어 엄밀 일치가 틀렸다는 사실을 숨기지 않는다.
    """
    rungs = reference_ladder(params)
    strict = rungs[0][0]
    loosest = rungs[-1][1]
    ladder: list[dict[str, object]] = []
    granularity: str | None = None
    matched_key: str | None = None
    for name, rung_params in [*rungs, ("uv_off", None)]:
        if query_inputs is None or reference_inputs is None:
            break
        q = _rung_key(query_inputs, name, rung_params, loosest)
        r = _rung_key(reference_inputs, name, rung_params, loosest)
        same = q == r
        ladder.append({"granularity": name, "query_key": q, "reference_key": r, "matched": same})
        if same and granularity is None:
            granularity = name
            matched_key = q
    return {
        "granularity": granularity,
        "label": None if granularity is None else reference_ladder_label(granularity),
        "key": matched_key,
        "strict_granularity": strict,
        "strict_label": reference_ladder_label(strict),
        "relaxed": granularity is not None and granularity != strict,
        "ladder": ladder,
    }


def hit_distance_m(query_hit: Vec3 | None, entry: Any) -> float | None:
    """질의와 후보의 **착탄 중점**(`KeyInputs.hit`) 사이 3차원 유클리드 거리, m. 못 구하면 None.

    `hit`은 리샘플 격자의 가운데 점(`frames[len//2]`, 20점이면 정규 호장 s ≈ 0.526)의 착탄점이고
    없으면 첫 프레임의 것이다(`key_inputs`). 색인(`csv_source.PassIndexEntry.key_inputs`)이 이미
    들고 있어 후보마다 CSV를 다시 읽지 않는다.

    좌표는 CSV 월드 그대로다(유니티 Y-up). 레벨 좌표로 투영하지 않고, 어떤 축도 가중하지 않고,
    바닥 높이를 빼지 않는다 — 후보는 이미 질의와 같은 레벨이라 월드 거리가 같은 벽 위 두 점 사이
    거리다. 세 축을 대칭으로 쓰므로 축 이름에 의존하지 않는다(사전등록 2026-09-23 §1.1).
    """
    hit = getattr(getattr(entry, "key_inputs", None), "hit", None)
    if query_hit is None or hit is None or len(hit) != 3:
        return None
    distance = math.dist(query_hit, hit)
    return distance if math.isfinite(distance) else None


def _by_hit_distance(row: Mapping[str, Any]) -> tuple[tuple[int, float], str, str, int]:
    """후보 정렬 키 — `(거리, date, session, pass_index)` 오름차순(사전등록 2026-09-23 §1.1).

    동률이면 `(date, session, pass_index)`로 갈린다. 곧 HEAD가 목록을 세우던 바로 그 순서라
    정렬은 결정론이고 난수를 쓰지 않는다.

    거리를 못 구한 후보(`None`)는 **버리지 않고 맨 뒤**에 둔다. 버리면 후보 집합이 바뀌어
    화면의 "N개 중 k번째"가 틀리고, 정렬이 고르는 일에 끼어든 것이 된다.
    """
    distance = row["hit_distance_m"]
    return (
        (1, 0.0) if distance is None else (0, float(distance)),
        str(row["date"]),
        str(row["session"]),
        int(row["pass_index"]),
    )


def reference_candidates(
    source: Any,
    query: PassRef,
    params: Any | None = None,
    samples: int = stroke_frame.DEFAULT_SAMPLES,
    limit: int = 50,
) -> dict[str, object]:
    """대상 패스의 기준 후보 — 사용자 지시(브리프 §"기준은 고를 수 없다"): 전형기
    고정 · 대상과 같은 레벨 · 동작 키가 일치 · 대상 자신은 제외.

    동작 키 일치는 **완화 사다리**를 탄다(S8, 위 주석). 후보를 하나라도 낸 **첫 단**에서 멈추고,
    그 단의 이름(`match_granularity`)과 사람이 읽는 문구(`match_label`)를 함께 낸다. 훑은 단은
    전부 `ladder`에 몇 건이었는지 적는다 — 엄밀 일치가 0이었다는 사실을 숨기지 않는다.

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

    _query_frames, query_inputs = pass_key_inputs(raw_query, samples)
    if query_inputs is None:
        return {"status": "unavailable", "reason": "insufficient_frames"}
    query_key = key_from(query_inputs, params)

    found, sessions_total = source.passes(EXPERT_WORKER, level=query.level, limit=limit)
    # 자기 자신은 어느 단에서도 후보가 아니다 — 자기와 비교하면 편차가 0이 되어 화면이 거짓을 말한다.
    entries = [
        (item, entry)
        for item, entry in found
        if PassRef(
            worker=item.worker, date=item.date, level=item.level, session=item.session,
            pass_index=entry.pass_index,
        )
        != query
    ]

    # 후보마다 착탄 중점 거리를 한 번만 잰다 — 사다리의 단마다 다시 재지 않는다.
    distances = [hit_distance_m(query_inputs.hit, entry) for _item, entry in entries]

    rungs = reference_ladder(params)
    loosest = rungs[-1][1]
    ladder: list[dict[str, object]] = []
    chosen: tuple[str, str, list[dict[str, object]]] | None = None
    for granularity, rung_params in [*rungs, ("uv_off", None)]:
        drop_uv = rung_params is None
        key_params = loosest if drop_uv else rung_params
        # 단의 키는 `compare`의 게이트(`motion_key_match`)와 **같은 함수**로 만든다 —
        # 두 곳에서 따로 만들면 고른 단과 허락하는 단이 조용히 갈린다.
        rung_key = _rung_key(query_inputs, granularity, rung_params, loosest)
        matched: list[dict[str, object]] = []
        for (item, entry), distance in zip(entries, distances):
            motion = source.key_of(entry, key_params)
            if drop_uv and motion is not None:
                motion = build_spread_reference.key_without_uv(motion)
            if motion != rung_key:
                continue
            matched.append(
                {
                    "worker": item.worker,
                    "date": item.date,
                    "level": item.level,
                    "session": item.session,
                    "pass_index": entry.pass_index,
                    "frames": entry.frames,
                    "length_mm": entry.length_m * 1000.0,
                    "motion_key": motion,
                    "match_granularity": granularity,
                    "hit_distance_m": distance,
                }
            )
        # 가장 가까운 후보가 먼저 선다(사전등록 2026-09-23 §1.2). 이 한 줄이 구현의 전부이고,
        # 화면(`/api/reference` → `static/app.js`의 `candidates[0]`·"다른 기준 ▸" 순환)과
        # 연구(`improvement_study.select_pairs_screen`, `choose="first"`)가 이것 하나를 나눠 쓴다.
        # 정렬은 고른 뒤에 하는 일이라 멤버십·사다리 단·`ladder` 건수를 바꾸지 않는다.
        matched.sort(key=_by_hit_distance)
        ladder.append({"granularity": granularity, "key": rung_key, "count": len(matched)})
        if matched and chosen is None:
            chosen = (granularity, rung_key, matched)

    granularity = None if chosen is None else chosen[0]
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
        # 후보를 낸 단. 없으면 None이다 — 기본값이나 요청한 단을 대신 채우지 않는다.
        "match_granularity": granularity,
        "match_label": None if granularity is None else reference_ladder_label(granularity),
        "strict_granularity": rungs[0][0],
        "relaxed": granularity is not None and granularity != rungs[0][0],
        "ladder": ladder,
        "candidates": [] if chosen is None else chosen[2],
    }


# --- 배포 경로를 그대로 돌려 보기 --------------------------------------------
#
# 위의 `compare`는 **패스 대 패스**다 — 동작 키로 고른 기준 패스 하나에 대고 잰다.
# 배포된 채점은 그렇지 않다: reach 밴드로 숙련자 **버킷**(패스 수천 개의 분포)을 고르고,
# 착탄 경로를 프로토타입에 DTW로 맞춘 뒤 위상마다 σ를 낸다. 기준 패스가 없다.
#
# 두 논리를 한 화면에서 견주려면 **배포 경로를 그대로 부르는 길**이 있어야 한다.
# 여기서 새로 계산하면 랩이 제품과 다른 답을 내고, 그러면 랩으로 제품을 못 고친다.


# 이 경로가 낼 수 있는 사유의 **닫힌 집합**. 단계의 것에 랩 계층 사유 셋을 더한다.
DIAGNOSE_REASONS: Final[frozenset[str]] = pose_stages.STAGE_REASONS | {
    "no_source", "pass_not_found", "insufficient_frames",
}


# 화면은 **서버의 표시 문자열만** 쓴다. JS가 다시 포맷하면 Python과 반올림 동률 처리가
# 달라 같은 값이 두 숫자로 보인다(`test_static_sources_do_not_reformat_numbers`).
QUALITY_TEXT: Final[dict[str, str]] = {
    "multi_worker": "숙련자 여러 명",
    "single_worker": "숙련자 한 명",
    "coarse_bucket": "이웃 밴드까지 묶음",
    "unavailable": "쓸 수 없음",
}
MODE_TEXT: Final[dict[str, str]] = {
    "phase": "위상 정렬", "fold": "시퀀스 접기", "burst": "스트로크 합산",
}


def _diagnose_display(
    scored: Any, resolved: Any, shape: Any
) -> dict[str, object]:
    """화면에 그대로 찍을 문자열들. 숫자는 여기서 **한 번만** 반올림한다."""
    percentile = scored.percentile
    return {
        "headline": "—" if percentile is None else f"{percentile}",
        "headline_unit": "" if percentile is None else "백분위",
        "score": "—" if scored.score is None else f"{scored.score:.2f}",
        "band": resolved.band or "—",
        "support": f"{resolved.support:,}" if resolved.support else "—",
        "quality": QUALITY_TEXT.get(resolved.quality or "", "—"),
        "mode": MODE_TEXT.get(scored.mode, scored.mode),
        "shape": f"스트로크 {shape.strokes} · 굽이 {shape.straightness:.2f}",
        # 백분위가 없을 때 **왜 없는지**를 화면이 말해야 한다. 빈칸은 "나쁘다"로 읽힌다.
        # 사유가 둘인데 하나로 적으면 운영자를 엉뚱한 데로 보낸다 — MCP에서 실제로
        # 그랬다(`reference_data_missing`이 레퍼런스가 붙어 있는데도 나갔다).
        "headline_note": (
            None if percentile is not None
            else "스트로크마다 잰 값을 합쳤습니다. 숙련자 사다리는 스트로크 하나에서"
                 " 구웠으므로 합산 점수는 그 위에 올리지 않습니다."
            if scored.mode == "burst"
            else "이 밴드에 숙련자 점수 사다리가 없어 위치를 낼 수 없습니다."
        ),
    }


def _diagnose_bars(
    sigmas: Sequence[Any], thresholds: Mapping[str, float]
) -> list[dict[str, object]]:
    """관절별 σ 막대. **글자 없이** 문턱 대비 높이로만 말한다.

    높이도 서버가 문자열로 낸다 — 화면이 숫자를 만들지 않게.
    """
    out: list[dict[str, object]] = []
    for item in sigmas:
        limit = thresholds.get(item.joint) or arm_reference.CORRECTION_SIGMA
        ratio = item.sigma / limit if limit > 0 else 0.0
        out.append({
            "joint": item.joint,
            "label": joint_label(item.joint),
            # 문턱을 100%로 둔다. 넘치면 잘려 보이지 않도록 상한은 화면 CSS가 맡는다.
            "height": f"{min(ratio, 2.0) * 50.0:.1f}%",
            "over": ratio >= 1.0,
        })
    return out


def _bars_note(scored: Any, thresholds: Mapping[str, float]) -> str | None:
    """막대와 교정이 **엇갈릴 때** 화면이 그 사실을 말하게 한다.

    `arm_reference.build_corrections`는 **생 σ로 순위를 매긴 뒤** 그 1등에만 관절별
    문턱을 적용한다. 문턱이 관절마다 다르므로(p97.5), 1등이 아닌 관절이 자기 범위를
    벗어나도 교정이 나가지 않는다. 실측 222패스에서 범위를 벗어난 65건 중 **18건
    (28%)** 이 그 경우다 — 생 σ 최대가 LeftHand(0.74×문턱)인데 실제 초과는
    RightUpperArm(1.00×문턱)이었다.

    **화면에서 감추지 않는다.** 막대는 빨간데 교정은 없는 상태를 설명 없이 두면
    화면이 자기모순한다. 순위 규칙 자체는 사전등록으로 정해진 것이라 여기서 바꾸지
    않는다(`docs/measurements/2026-09-30-per-joint-threshold-prereg.md`).
    """
    over = [item for item in scored.sigmas
            if item.sigma >= (thresholds.get(item.joint) or arm_reference.CORRECTION_SIGMA)]
    if not over or scored.corrections:
        return None
    labels = [joint_label(item.joint) for item in over]
    names = " · ".join(labels)
    # 조사는 마지막 이름의 받침을 따른다. "이(가)"를 그대로 내보내면 화면이 초고로 읽힌다.
    return (f"{names}{subject_particle(labels[-1])} 숙련자 통상 범위를 벗어났지만,"
            " 교정은 편차가 가장 큰 관절 하나에만 나갑니다.")


def diagnose_sequence(
    source: Any, query: PassRef, buckets: Mapping[str, Any] | None
) -> dict[str, object]:
    """패스 하나를 **배포된 채점 경로**(`pose_stages.score_burst`)에 그대로 넣는다.

    MCP `/pose_estimation/sequence`가 런타임 프레임에 대고 하는 일과 같은 계산이다.
    입력만 CSV 코퍼스에서 온다.
    """
    if buckets is None:
        return {"status": "unavailable", "reason": "reference_data_missing"}
    try:
        raw = source.load(query)
    except LookupError:
        return {"status": "unavailable", "reason": "pass_not_found"}

    prepared = pose_fk_csv.pass_inputs(raw.rows)
    if prepared is None:
        return {"status": "unavailable", "reason": "insufficient_frames"}
    frames, impacts, tools = prepared

    burst = pose_stages.score_burst(
        frames, impacts, list(range(len(impacts))), tools, buckets,
        labels=pose_similarity.JOINT_LABELS,
    )
    scored = burst.result
    first = burst.strokes[0].resolve if burst.strokes else pose_stages.resolve(
        pose_stages.reach_of(impacts, tools), buckets
    )
    if not scored.ok:
        return {
            "status": "unavailable",
            "reason": scored.reason or first.reason or "quality_insufficient",
            "band": first.band,
            "reach_m": first.reach_m,
            "support": first.support,
            "burst": burst.to_dict(),
        }

    chosen = burst.strokes[burst.worst_stroke or 0]
    bucket = buckets[chosen.resolve.band] if chosen.resolve.band in buckets else {}
    phase_poses, reference = pose_stages._phase_view(
        frames, chosen.align, bucket.get("phases") or []
    )
    ladders = bucket.get("joint_sigma_ladder")
    thresholds = {
        item.joint: arm_reference.sigma_threshold(
            item.joint, ladders if isinstance(ladders, Mapping) else None
        )
        for item in scored.sigmas
    }
    return {
        "status": "ok",
        # 배포 응답과 **같은 이름**을 쓴다. 랩에서 본 값과 제품이 낸 값을 눈으로 대조한다.
        "scoring_set_id": arm_reference.SCORING_SET_ID,
        "scoring_mode": scored.mode,
        "band": chosen.resolve.band,
        "reach_m": chosen.resolve.reach_m,
        "support": chosen.resolve.support,
        "reference_quality": chosen.resolve.quality,
        "similarity_score": scored.score,
        # 합친 점수에는 백분위가 없다 — 사다리가 패스 하나에서 구워졌다.
        "expert_percentile": scored.percentile,
        "burst": burst.to_dict(),
        "joint_sigmas": {item.joint: item.sigma for item in scored.sigmas},
        # 문턱은 관절마다 다르다(p97.5). 같이 내지 않으면 화면이 막대를 그릴 기준이 없다.
        "joint_thresholds": thresholds,
        "corrections": [
            pose_stages._correction_dict(item, scored.plans[index] if index < len(scored.plans) else ())
            for index, item in enumerate(scored.corrections)
        ],
        # 그릴 재료. 착탄 경로와 프로토타입은 **같은 평면**에 놓고 봐야 정렬이 보인다.
        "impact_path": [list(point) for point in impacts],
        "prototype": [list(point) for point in chosen.resolve.prototype],
        "phase_of": [list(pair) for pair in chosen.align.phase_of],
        "phase_n": [
            int(entry.get("n", 0) or 0) if isinstance(entry, Mapping) else 0
            for entry in reference
        ],
        "measured_layer": _serialize_layer(_world_layer_of(frames)),
        "expert_layer": _expert_layer(phase_poses, reference),
        "display": _diagnose_display(scored, chosen.resolve, burst.shape),
        "bars": _diagnose_bars(scored.sigmas, thresholds),
        "bars_note": _bars_note(scored, thresholds),
        "correction_text": (
            scored.corrections[0].text() if scored.corrections else None
        ),
    }


def _world_layer_of(frames: Sequence[Mapping[str, Vec3]]) -> dict[str, list[Vec3 | None]]:
    """프레임별 월드 관절을 화면 레이어 모양으로. 채점 관절만 낸다."""
    names = list(arm_reference.SCORING_JOINTS_ARM) + list(arm_reference.FRAME_JOINTS)
    return {name: [frame.get(name) for frame in frames] for name in names}


def _expert_layer(
    phase_poses: Sequence[Mapping[str, Vec3] | None], reference: Sequence[Any]
) -> dict[str, list[list[float] | None]]:
    """위상마다 숙련자 평균 자세(**몸 기준 정규 좌표**).

    측정 자세는 월드인데 이쪽은 정규 좌표다 — 같은 축에 겹쳐 그리면 안 된다.
    이름을 다르게 두어 화면이 섞지 못하게 한다.
    """
    out: dict[str, list[list[float] | None]] = {}
    for name in arm_reference.SCORING_JOINTS_ARM:
        row: list[list[float] | None] = []
        for entry in reference:
            point = None
            if isinstance(entry, Mapping):
                joints = entry.get("joints")
                found = joints.get(name) if isinstance(joints, Mapping) else None
                mean = found.get("mean") if isinstance(found, Mapping) else None
                if isinstance(mean, Sequence) and len(mean) == 3:
                    point = [float(mean[0]), float(mean[1]), float(mean[2])]
            row.append(point)
        out[name] = row
    return out
