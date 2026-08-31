#!/usr/bin/env python3
"""비교 한 건을 만드는 순수 로직. HTTP를 모른다.

두 축을 병기하는 것이 이 모듈의 존재 이유다. 실측 축은 근거가 측정값이고 관절 축은
IK 산출물이다. 어느 쪽이 개선율 70%를 낼 수 있는지 지금 아무도 모르므로 하나를 골라
계약으로 굳히기 전에 둘을 나란히 놓고 실제 패스를 넘겨 본다.
"""

from __future__ import annotations

import importlib
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final, Mapping, Protocol, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

measured_pose = importlib.import_module("measured_pose")
measured_score = importlib.import_module("measured_score")
motion_key = importlib.import_module("motion_key")
pose_similarity = importlib.import_module("pose_similarity")
stroke_frame = importlib.import_module("stroke_frame")
build_spread_reference = importlib.import_module("build_spread_reference")

from trainer.pose_lab.spread_reference import Resolved, SpreadReference

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
    deltas: Sequence[measured_score.PointDelta], frame: measured_score.AxisFrame
) -> list[dict[str, object]]:
    """발부한 축 전부를 sigma 내림차순으로. 작업자에게 가는 숫자 패널의 원본이다.

    이 목록은 `compare()`가 실제로 발부하는 지시이고, 그것이 곧 `apply_measured`가
    옮기는 것이다 — 그래서 화면 숫자와 `improvement.same`이 같은 이동을 잰다.

    `improvement.split`에는 그 등식이 성립하지 않는다. split은 자기충족을 피하려고
    교정 후보를 A 반쪽의 산포로 다시 고르므로(`before_a.deltas`) sigma가 달라져
    발부 축 집합이 이 목록과 갈릴 수 있다. 그것이 split의 설계지 결함이 아니다 —
    다만 이 목록을 split의 근거로 읽으면 안 된다.
    """
    return [
        {
            "point": correction.point,
            "axis": correction.axis,
            "magnitude_cm": correction.magnitude_cm,
            "direction": correction.direction,
            "sigma": correction.sigma,
        }
        for correction in measured_score.issued_axes(deltas, frame)
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


def compare(
    source: PassSource,
    query: PassRef,
    reference: PassRef,
    params: Any | None = None,
    samples: int = stroke_frame.DEFAULT_SAMPLES,
    spreads: measured_score.AxisSpreads | None = None,
    weights: dict[str, float] | None = None,
    reference_spreads: SpreadReference | None = None,
) -> dict[str, object]:
    """비교 1건. 동작 키가 다르면 지표를 만들지 않는다."""
    params = params or motion_key.KeyParams()
    weights = weights or measured_score.DEFAULT_WEIGHTS

    query_frames = pass_frames(source.load(query), samples)
    reference_frames = pass_frames(source.load(reference), samples)
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

    return {
        "status": "ok",
        "query_key": query_key,
        "reference_key": reference_key,
        # 몸 기준 축(②④)이 선 요. `yaw_source`가 실측인지 대체값인지 말한다.
        "yaw_assumed": yaw_used,
        "yaw_source": yaw_source,
        "samples": before.samples,
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
            "axes": axes_block(before.deltas, basis),
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
