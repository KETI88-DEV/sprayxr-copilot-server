#!/usr/bin/env python3
"""비교 한 건을 만드는 순수 로직. HTTP를 모른다.

두 축을 병기하는 것이 이 모듈의 존재 이유다. 실측 축은 근거가 측정값이고 관절 축은
IK 산출물이다. 어느 쪽이 개선율 70%를 낼 수 있는지 지금 아무도 모르므로 하나를 골라
계약으로 굳히기 전에 둘을 나란히 놓고 실제 패스를 넘겨 본다.
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
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
    """
    same = pose_similarity.gap_closure(before, after_same)
    if after_split is None:
        return {"split": None, "same": same or 0.0}
    split = pose_similarity.gap_closure(before, after_split)
    return {"split": split or 0.0, "same": same or 0.0}


def _pass_frames(raw: RawPass, samples: int) -> list[Any]:
    frames = measured_pose.measured_frames(raw.rows)
    spraying = [frame for frame in frames if frame.spraying and frame.hit is not None]
    return stroke_frame.resample_frames(spraying, samples)


def _key_for(frames: Sequence[Any], params: Any) -> str | None:
    """작업자 요는 아직 실측에서 뽑지 않는다. 지금은 0으로 두고 화면에서 판정한다."""
    if len(frames) < 2:
        return None
    first, last = frames[0], frames[-1]
    if first.hit is None or last.hit is None:
        return None
    displacement = (
        last.hit[0] - first.hit[0],
        last.hit[1] - first.hit[1],
        last.hit[2] - first.hit[2],
    )
    middle = frames[len(frames) // 2]
    return motion_key.motion_key(
        displacement=displacement,
        yaw=0.0,
        head=middle.head,
        hit=middle.hit or first.hit,
        uv=middle.uv,
        params=params,
    )


def _local_tracks(frames: Sequence[Any]) -> dict[str, list[Vec3 | None]]:
    """격자점마다 스트로크 프레임 좌표. 왼손이 없는 격자점은 자리를 남기고 None이다.

    자리를 지우면 리스트가 짧아져 두 패스의 s 대응이 한 칸씩 밀린다. 왼손만 프레임
    단위로 빠지는 설계 §5.4의 마스킹이 다른 두 점의 대응까지 망가뜨리게 된다.
    """
    tracks: dict[str, list[Vec3 | None]] = {"right": [], "head": [], "left": []}
    for index, frame in enumerate(frames):
        if frame.hit is None:
            continue
        following = frames[min(index + 1, len(frames) - 1)]
        if following.hit is None or following is frame:
            tangent = (0.0, 1.0, 0.0)
        else:
            tangent = (
                following.hit[0] - frame.hit[0],
                following.hit[1] - frame.hit[1],
                following.hit[2] - frame.hit[2],
            )
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
    spreads = spreads or {"right": 0.05, "head": 0.05, "left": 0.05}
    weights = weights or measured_score.DEFAULT_WEIGHTS

    query_frames = _pass_frames(source.load(query), samples)
    reference_frames = _pass_frames(source.load(reference), samples)
    query_key = _key_for(query_frames, params)
    reference_key = _key_for(reference_frames, params)

    if query_key is None or reference_key is None or query_key != reference_key:
        return {
            "status": "unavailable",
            "reason": "motion_key_mismatch",
            "query_key": query_key,
            "reference_key": reference_key,
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
        "samples": before.samples,
        "measured": {
            "basis": "tracked",
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
            "corrections": measured_score.corrections_text(before.deltas),
            # 진짜 분리 채점 점수는 겹치지 않는 날짜에서 뽑은 held-out 레퍼런스가
            # 있어야 나온다. 실제 PassSource가 배선되기 전까지는 split을 None으로
            # 낸다 — after.score를 split 자리에 다시 넣으면 두 숫자가 항상 같아져
            # 자기충족 정도를 가리는 바로 그 결함이 된다.
            "improvement": improvement(before.score, None, after.score),
            "gate": gate_verdict(max(spreads.values()), DEFAULT_ERROR_M),
        },
        "joint": {
            "basis": "ik_derived",
            "status": "not_wired",
            "note": JOINT_AXIS_NOTE,
            "score": None,
            "score_after": None,
            "deltas": [],
            "corrections": [],
            "improvement": None,
            "gate": None,
        },
    }
