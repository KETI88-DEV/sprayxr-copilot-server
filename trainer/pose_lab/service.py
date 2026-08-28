#!/usr/bin/env python3
"""비교 한 건을 만드는 순수 로직. HTTP를 모른다.

두 축을 병기하는 것이 이 모듈의 존재 이유다. 실측 축은 근거가 측정값이고 관절 축은
IK 산출물이다. 어느 쪽이 개선율 70%를 낼 수 있는지 지금 아무도 모르므로 하나를 골라
계약으로 굳히기 전에 둘을 나란히 놓고 실제 패스를 넘겨 본다.
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final, Protocol, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

measured_pose = importlib.import_module("measured_pose")
measured_score = importlib.import_module("measured_score")
motion_key = importlib.import_module("motion_key")
pose_similarity = importlib.import_module("pose_similarity")
stroke_frame = importlib.import_module("stroke_frame")

Vec3 = tuple[float, float, float]

# REQ-19 correction_target_gate. 납품 후 수치를 보고 기준을 옮기는 것을 막으려고 박아 둔다.
GATE_RATIO: Final = 0.5
# 잡으려는 최소 자세 오류. 팔꿈치 30도에 해당하는 실측 참조값이다.
DEFAULT_ERROR_M: Final = 0.144

# 레퍼런스 버킷에서 잰 산포가 아니라 자리를 채워 둔 임시값이다. 이 값으로 게이트를
# 판정하면 상수 두 개에서 나온 "pass"를 화면이 판정으로 내걸게 된다. REQ-19 각도
# 유효화는 설계 §10에서 미판정이고 D4는 게이트를 옮기지 말라고 못박는다.
PROVISIONAL_SPREADS: Final[dict[str, float]] = {"right": 0.05, "head": 0.05, "left": 0.05}

# 작업자 요를 아직 실측에서 뽑지 않는다. 그래서 동작 키의 ②진행 방향과 ④몸 기준 착탄
# 위치는 몸 기준 이름을 달고 있지만 실제로는 월드 기준 방위각이다. 실측 요 추출은 별개
# 작업이므로 여기서는 값을 응답에 실어 그 사실을 드러낸다.
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
    before: float, after_split: float | None, after_same: float
) -> dict[str, float | None]:
    """개선율을 분리 채점과 동일 채점 둘 다 낸다.

    goal_year2의 문구는 목표 자세를 입력으로 주므로 동일 채점에 해당한다. 두 값을
    병기해야 그 값이 자기충족에 얼마나 기대는지 드러난다. 분리 채점 점수가 없으면
    (아직 계산되지 않았으면) 없는 값을 만들어내지 않고 None을 그대로 낸다.

    `gap_closure`가 내는 None도 그대로 흘린다. 이미 기준과 같으면(S0=1) 메울 격차가
    없어 개선율이 정의되지 않는데, 여기서 0.0으로 바꾸면 "메우지 못했다"로 읽힌다.
    화면은 None을 대시로 그리고 있으므로 뒤에서 값을 지어낼 이유가 없다.
    """
    same = pose_similarity.gap_closure(before, after_same)
    if after_split is None:
        return {"split": None, "same": same}
    return {"split": pose_similarity.gap_closure(before, after_split), "same": same}


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
    )


def motion_key_for(frames: Sequence[Any], params: Any) -> str | None:
    """작업자 요는 아직 실측에서 뽑지 않는다. 지금은 0으로 두고 화면에서 판정한다.

    이 가정은 응답의 `yaw_assumed`로 나가고 화면이 동작 키 옆에 적는다. 몸 기준
    이름을 단 축이 사실은 월드 기준이라는 것을 읽는 사람이 알아야 한다.
    """
    inputs = key_inputs(frames)
    if inputs is None:
        return None
    return key_from(inputs, params)


def key_from(inputs: KeyInputs, params: Any) -> str:
    return motion_key.motion_key(
        displacement=inputs.displacement,
        yaw=ASSUMED_YAW,
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


def compare(
    source: PassSource,
    query: PassRef,
    reference: PassRef,
    params: Any | None = None,
    samples: int = stroke_frame.DEFAULT_SAMPLES,
    spreads: dict[str, float] | None = None,
    weights: dict[str, float] | None = None,
) -> dict[str, object]:
    """비교 1건. 동작 키가 다르면 지표를 만들지 않는다."""
    params = params or motion_key.KeyParams()
    # 호출자가 산포를 넘겼을 때만 그것이 잰 분포다. 임시값으로는 게이트를 판정하지
    # 않고 σ 정규화도 잠정으로 표시한다.
    measured_spreads = spreads is not None
    spreads = spreads or PROVISIONAL_SPREADS
    weights = weights or measured_score.DEFAULT_WEIGHTS

    query_frames = pass_frames(source.load(query), samples)
    reference_frames = pass_frames(source.load(reference), samples)
    query_key = motion_key_for(query_frames, params)
    reference_key = motion_key_for(reference_frames, params)

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
            "yaw_assumed": ASSUMED_YAW,
        }
    if query_key != reference_key:
        return {
            "status": "unavailable",
            "reason": "motion_key_mismatch",
            "query_key": query_key,
            "reference_key": reference_key,
            "yaw_assumed": ASSUMED_YAW,
        }

    query_track = _local_tracks(query_frames)
    reference_track = _local_tracks(reference_frames)
    before = measured_score.score_track(query_track, reference_track, spreads, weights)
    fixed = measured_score.apply_measured(query_track, before.deltas)
    after = measured_score.score_track(fixed, reference_track, spreads, weights)

    return {
        "status": "ok",
        "query_key": query_key,
        "reference_key": reference_key,
        # 몸 기준 축(②④)은 이 요 가정 위에 서 있다. 실측 요가 아니다.
        "yaw_assumed": ASSUMED_YAW,
        "samples": before.samples,
        "measured": {
            "basis": "tracked",
            # σ와 점수 exp(-σ²/2)는 이 산포로 나눈 값이다. 잰 분포가 아니면 읽는
            # 사람이 그 사실을 알아야 한다.
            "normalization": "measured" if measured_spreads else "provisional",
            "score": before.score,
            "score_after": after.score,
            "deltas": [
                {
                    "point": delta.point,
                    "sigma": delta.sigma,
                    "delta": list(delta.local_delta),
                    # 왼손은 프레임 단위로 빠지므로 점마다 받친 격자점 수가 다르다.
                    "samples": delta.samples,
                }
                for delta in before.deltas
            ],
            # 기저를 함께 넘긴다. 성분을 축 이름으로 읽으면 수직 스트로크에서 좌우
            # 이탈이 "올리세요"로 나간다(`stroke_frame.Basis` 독스트링).
            "corrections": measured_score.corrections_text(
                before.deltas, representative_basis(query_frames)
            ),
            # 최종 작업자 문구는 다운스트림 VLM이 짓는다. 문장만 주면 VLM이 근거 없이
            # 다시 지어내야 한다 — 표본 수는 문장에 없다. 구조체를 병기해 축·부호·σ·
            # 표본 n을 그대로 넘긴다. 오른손이 임계 아래거나 못 잰 경우 None이다.
            "finding": (
                None
                if (
                    finding := measured_score.right_hand_finding(
                        before.deltas, representative_basis(query_frames)
                    )
                )
                is None
                else asdict(finding)
            ),
            # 진짜 분리 채점 점수는 겹치지 않는 날짜에서 뽑은 held-out 레퍼런스가
            # 있어야 나온다. 실제 PassSource가 배선되기 전까지는 split을 None으로
            # 낸다 — after.score를 split 자리에 다시 넣으면 두 숫자가 항상 같아져
            # 자기충족 정도를 가리는 바로 그 결함이 된다.
            "improvement": improvement(before.score, None, after.score),
            # 게이트의 뜻은 "각도 지시가 이제 정당하다"다. 잰 분포가 없으면 판정을
            # 만들지 않는다. 임계 논리(`gate_verdict`)는 그대로 두고 입력이 없다는
            # 사실만 낸다.
            "gate": (
                gate_verdict(max(spreads.values()), DEFAULT_ERROR_M)
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
            "finding": None,
            "improvement": None,
            "gate": None,
            "gate_reason": "not_wired",
        },
    }
