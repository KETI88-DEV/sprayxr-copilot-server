from __future__ import annotations

import json
import os
import statistics
from datetime import datetime
from dataclasses import dataclass, replace
from functools import lru_cache
from math import atan2, degrees, sqrt
from pathlib import Path
from typing import Any, Final, Literal, Mapping, Optional, Sequence, assert_never

from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict, Field, model_validator

from mcp_servers.common import MCPResponse, success_response
from trainer.scripts import (
    arm_reference,
    joint_names,
    pose_similarity,
    pose_stages,
    posture_sequence,
    skeleton_gate,
    world_axes,
)


router = APIRouter(prefix="/pose_estimation", tags=["pose_estimation"])

POSE_REFERENCE_ENV = "POSE_REFERENCE_PATH"
DEFAULT_POSE_REFERENCE = "/output/expert_reference/expert_pose_reference.json"

_BLOCK_TO_EXPERT_POSE: dict[str, str] = {
    "BLK-A": "EP-001",
    "BLK-B": "EP-002",
    "BLK-C": "EP-003",
}
_DEFAULT_EXPERT_POSE = "EP-001"

_JOINT_LABELS: dict[str, str] = {
    "wrist_r": "오른쪽 손목",
    "wrist_l": "왼쪽 손목",
    "elbow_r": "오른쪽 팔꿈치",
    "elbow_l": "왼쪽 팔꿈치",
    "shoulder_r": "오른쪽 어깨",
    "shoulder_l": "왼쪽 어깨",
}


class Joint(BaseModel):
    x: float
    y: float
    z: float
    roll: float | None = None
    pitch: float | None = None
    yaw: float | None = None


class Pose(BaseModel):
    joints: dict[str, Joint]


class PoseRequest(BaseModel):
    mode: Literal["compare", "recommend"]
    current_pose: Optional[Pose] = None
    expert_pose_id: str | None = None
    block_id: str | None = None
    level: str | None = None
    bucket: str | None = None

    @model_validator(mode="after")
    def mode_has_required_context(self) -> "PoseRequest":
        match self.mode:
            case "compare":
                if self.current_pose is None:
                    raise ValueError("compare mode requires current_pose")
            case "recommend":
                if self.block_id is None:
                    raise ValueError("recommend mode requires block_id")
            case unreachable:
                assert_never(unreachable)
        return self


def _get_expert_pose(expert_pose_id: str | None, block_id: str | None) -> tuple[str, Pose]:
    pose_id = expert_pose_id or _BLOCK_TO_EXPERT_POSE.get(block_id or "", _DEFAULT_EXPERT_POSE)
    reference = Pose(joints={
        "wrist_r": Joint(x=0.1, y=-0.04, z=0.44, roll=2.0, pitch=12.0, yaw=5.0),
        "wrist_l": Joint(x=-0.1, y=-0.04, z=0.44, roll=2.0, pitch=12.0, yaw=5.0),
        "elbow_r": Joint(x=0.25, y=-0.1, z=0.35, roll=0.0, pitch=8.0, yaw=0.0),
        "elbow_l": Joint(x=-0.25, y=-0.1, z=0.35, roll=0.0, pitch=8.0, yaw=0.0),
        "shoulder_r": Joint(x=0.35, y=0.0, z=0.2, roll=0.0, pitch=5.0, yaw=0.0),
        "shoulder_l": Joint(x=-0.35, y=0.0, z=0.2, roll=0.0, pitch=5.0, yaw=0.0),
    })
    return pose_id, reference


def _compute_corrections(current: Pose, reference: Pose) -> tuple[float, list[dict]]:
    corrections = []
    total_distance = 0.0
    common = sorted(set(current.joints) & set(reference.joints))

    for name in common:
        cur = current.joints[name]
        ref = reference.joints[name]
        dx, dy, dz = cur.x - ref.x, cur.y - ref.y, cur.z - ref.z
        distance = sqrt(dx*dx + dy*dy + dz*dz)
        total_distance += distance

        angle_corrections = {}
        for axis in ("roll", "pitch", "yaw"):
            cur_val = getattr(cur, axis)
            ref_val = getattr(ref, axis)
            if cur_val is not None and ref_val is not None:
                diff = cur_val - ref_val
                if abs(diff) > 3.0:
                    angle_corrections[axis] = round(diff, 1)

        label = _JOINT_LABELS.get(name, name)
        correction_texts = []
        for axis, diff in angle_corrections.items():
            direction = "과다" if diff > 0 else "부족"
            correction_texts.append(f"{axis} {abs(diff):.0f}° {direction}")

        corrections.append({
            "joint": name,
            "label": label,
            "distance": round(distance, 4),
            "angle_corrections": angle_corrections,
            "correction_text": f"{label}: {', '.join(correction_texts)}" if correction_texts else f"{label}: 양호",
        })

    similarity = max(0.0, min(1.0, 1.0 - (total_distance / max(len(common), 1) / 0.5)))
    return round(similarity, 3), corrections


@lru_cache(maxsize=4)
def _read_reference(path: str, revision: float) -> dict[str, Any]:
    _ = revision
    return json.loads(Path(path).read_text(encoding="utf-8"))


PERCENTILE_POINTS: Final = (10, 25, 50, 75, 90)


def _reference_bucket(
    level: str | None, bucket: str
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """레퍼런스 아티팩트에서 (레벨 엔트리, 버킷)을 찾는다. 없으면 추정하지 않고 None을 낸다."""
    path = Path(os.environ.get(POSE_REFERENCE_ENV, DEFAULT_POSE_REFERENCE))
    if not path.is_file():
        return None
    payload = _read_reference(str(path), path.stat().st_mtime)
    entry = payload.get(level or "")
    if not isinstance(entry, dict) or entry.get("status") != "ok":
        return None
    buckets = entry.get("buckets")
    if not isinstance(buckets, dict):
        return None
    found = buckets.get(bucket)
    return (entry, found) if isinstance(found, dict) else None


def _expert_percentile(entry: Mapping[str, Any], score: float) -> int | None:
    """숙련자 held-out 분포에서 이 점수가 어디쯤인지. 교차적합 보정이 없으면 None이다."""
    calibration = entry.get("expert_similarity")
    if not isinstance(calibration, Mapping):
        return None
    # 사다리는 측정 당시 레퍼런스에만 유효하다. 아티팩트만 재생성하면 틀린 백분위가 조용히 나간다.
    if calibration.get("reference_pose_count") != entry.get("pose_count"):
        return None
    reached: int | None = None
    for point in PERCENTILE_POINTS:
        value = calibration.get(f"p{point}")
        if isinstance(value, (int, float)) and score >= float(value):
            reached = point
    return 0 if reached is None else reached


def _unavailable(reason: str, message: str) -> MCPResponse:
    return success_response(result={
        "mode": "compare",
        "analysis_status": "not_available",
        "reason_code": reason,
        "summary": message,
    })


def _bucket_compare(payload: PoseRequest, bucket: str) -> MCPResponse:
    pose = payload.current_pose
    if pose is None or pose_similarity.ROOT_JOINT not in pose.joints:
        return _unavailable(
            "source_data_missing",
            f"자세에 기준 관절({pose_similarity.ROOT_JOINT})이 없어 비교할 수 없습니다.",
        )

    found = _reference_bucket(payload.level, bucket)
    if found is None:
        return _unavailable(
            "reference_data_missing",
            "이 시퀀스에 적용할 기준 자세가 없어 자세 판정을 제공할 수 없습니다.",
        )
    entry, reference = found

    normalized = pose_similarity.normalize_pose(
        {name: (joint.x, joint.y, joint.z) for name, joint in pose.joints.items()}
    )
    sigmas = pose_similarity.joint_sigmas(normalized, reference)
    if not sigmas:
        # similarity_score는 빈 σ에 0.0을 낸다. '많이 다르다'가 아니라 '못 잰다'는 뜻이다.
        return _unavailable(
            "quality_insufficient",
            "기준 자세에 채점 가능한 관절이 없어 자세 판정을 제공할 수 없습니다.",
        )

    similarity = pose_similarity.similarity_score(sigmas)
    corrections = pose_similarity.build_corrections(normalized, reference)

    detail = (
        f"교정 필요 관절 {len(corrections)}개: "
        + ", ".join(item.correction_text for item in corrections)
        if corrections
        else "모든 관절 양호."
    )
    return success_response(result={
        "mode": "compare",
        "analysis_status": "available",
        "reason_code": None,
        "expert_pose_id": bucket,
        "similarity_score": round(similarity, 3),
        "expert_percentile": _expert_percentile(entry, similarity),
        "support": reference.get("n"),
        "reference_quality": entry.get("reference_quality"),
        "corrections": [
            {
                "joint": item.joint,
                "label": item.label,
                "distance": round(item.distance, 4),
                "angle_corrections": item.angle_corrections,
                "correction_text": item.correction_text,
            }
            for item in corrections
        ],
        "summary": f"숙련자 자세({bucket}) 대비 유사도 {similarity:.0%}. {detail}",
    })


@router.post("/compare", response_model=MCPResponse)
def compare_pose(payload: PoseRequest) -> MCPResponse:
    if payload.mode == "compare" and payload.bucket:
        return _bucket_compare(payload, payload.bucket)

    if payload.mode == "recommend":
        pose_id = _BLOCK_TO_EXPERT_POSE.get(payload.block_id or "", _DEFAULT_EXPERT_POSE)
        return success_response(result={
            "mode": "recommend",
            "block_id": payload.block_id,
            "recommended_expert_pose_id": pose_id,
            "message": f"블록 {payload.block_id}에서 숙련자 자세 {pose_id}을 사용하세요.",
        })

    if not payload.current_pose:
        return success_response(result={
            "mode": "compare",
            "error": "compare 모드에서 current_pose가 필요합니다.",
        })

    pose_id, reference = _get_expert_pose(payload.expert_pose_id, payload.block_id)
    similarity, corrections = _compute_corrections(payload.current_pose, reference)

    needs_correction = [c for c in corrections if c["angle_corrections"]]

    return success_response(result={
        "mode": "compare",
        "analysis_status": "available",
        "reason_code": None,
        "expert_pose_id": pose_id,
        "similarity_score": similarity,
        "corrections": corrections,
        "summary": (
            f"숙련자 자세({pose_id}) 대비 유사도 {similarity:.0%}. "
            + (f"교정 필요 관절 {len(needs_correction)}개: "
               + ", ".join(c["correction_text"] for c in needs_correction)
               if needs_correction
               else "모든 관절 양호.")
        ),
    })


# --- 시퀀스 경로 -------------------------------------------------------------
#
# 위의 `/compare`는 **자세 한 점**을 받는다. 런타임이 보내는 것은 시퀀스 하나다
# (`sequence_info[]`, 샘플에서 7~11프레임 · 120FPS). 둘은 입력의 단위가 달라서
# 같은 경로에 욱여넣을 수 없다. `/compare`는 건드리지 않고 새 경로를 붙인다.
#
# **검증 강도를 층마다 다르게 둔다.** 최상위는 `forbid`다 — 키 8개가 계약으로 고정돼
# 있고, 모르는 키가 조용히 지나가면 우리가 안 읽은 신호를 보낸 쪽은 읽혔다고 믿는다.
# 프레임 이하는 `allow`이되 **모르는 키를 세어 응답에 싣는다** — 여기까지 `forbid`로
# 잠그면 실물 샘플 시퀀스가 100% 422가 된다(`raytexmapnum` 류 벤더 확장이 계속 는다).
# 조용히 버리지 않고 보고하는 것이 절충의 핵심이다.

# 프레임에서 우리가 읽는 키. 이 밖의 것은 `unknown_fields`로 보고된다.
_KNOWN_FRAME_FIELDS: Final[frozenset[str]] = frozenset(
    {"index", "time", "tool_pos_rot", "skeleton_joints", "gun_metrics"}
)

# 이 경로가 낼 수 있는 사유의 **닫힌 집합**. 게이트의 것에 배선 계층 사유 둘을 더한다.
# 반환 직전에 대조한다 — 지어낸 문자열이 나가면 소비자가 모르는 값을 받는다.
SEQUENCE_REASON_CODES: Final[frozenset[str]] = skeleton_gate.REASON_CODES | {
    "reference_data_missing",
    "reference_single_worker",
    "quality_insufficient",
}

# 세션 상태를 이 모듈이 들고 있다. 메인 에이전트는 시퀀스를 보내기만 하고
# 누적은 여기서 한다 — 추종 기울기는 시퀀스 하나로는 나오지 않기 때문이다.
_SESSIONS: Final = posture_sequence.SessionStore()


class SequenceFrame(BaseModel):
    """시퀀스의 프레임 하나. 모르는 키를 받되 **세어서 보고한다**."""

    model_config = ConfigDict(extra="allow")

    index: int | None = None
    # 실물은 ISO-8601 문자열(`2026-09-21T00:07:58.293Z`)로 보낸다. `float`으로만 선언하면
    # 정상 시퀀스가 전부 422가 된다 — 실물을 넣어 보고 알았다. 숫자로 오는 계약도 있어
    # 둘 다 받고, 초 단위로 바꾸는 일은 `_frame_seconds`가 한다.
    time: str | float | None = None
    tool_pos_rot: dict[str, Any] | None = None
    skeleton_joints: dict[str, Any] = Field(default_factory=dict)
    gun_metrics: dict[str, Any] | None = None


class SequenceRequest(BaseModel):
    """`POST /copilot/analyze-sequence`의 본문과 같은 모양이다.

    최상위만 `forbid`다. 실물 샘플의 최상위 키 8개를 모두 선언했으므로
    정상 본문은 통과하고, 계약에 없던 키가 붙으면 **크게 실패한다**.
    """

    model_config = ConfigDict(extra="forbid")

    session_id: str
    sequence_id: str
    user_id: str | None = None
    scenario_id: str | None = None
    zone_id: str | None = None
    sequence_setting: dict[str, Any] | None = None
    sequence_info: list[SequenceFrame] = Field(default_factory=list)
    sequence_result: dict[str, Any] | None = None
    # 아래 둘은 벤더가 보내지 않는다. 레벨 메시를 아는 호출자가 채우면 자세 대역을
    # 판정하고, 없으면 `posture_band_ok`가 `null`이다 — 모르는 것과 통과는 다르다.
    level: str | None = None
    floor_height_m: float | None = None
    # ①의 출력을 넘겨받는 손잡이. 주면 버킷 찾기를 건너뛴다 — 단계를 따로 부를 때 쓴다.
    band: str | None = None


ARM_REFERENCE_ENV: Final = "ARM_REFERENCE_PATH"
DEFAULT_ARM_REFERENCE: Final = "/output/expert_reference/expert_arm_reference.json"


@lru_cache(maxsize=2)
def _read_arm_reference(path: str, revision: float) -> dict[str, Any]:
    _ = revision
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _arm_buckets() -> dict[str, Any] | None:
    """아티팩트의 버킷 사전. 없으면 `None` — 추정하지 않는다."""
    path = Path(os.environ.get(ARM_REFERENCE_ENV, DEFAULT_ARM_REFERENCE))
    if not path.is_file():
        return None
    payload = _read_arm_reference(str(path), path.stat().st_mtime)
    buckets = payload.get("buckets")
    return buckets if isinstance(buckets, dict) else None


def _arm_bucket(reach_m: float | None) -> tuple[str, dict[str, Any]] | None:
    """이 reach에 해당하는 숙련자 버킷. 아티팩트가 없거나 버킷이 없으면 **추정하지 않는다**."""
    if reach_m is None:
        return None
    path = Path(os.environ.get(ARM_REFERENCE_ENV, DEFAULT_ARM_REFERENCE))
    if not path.is_file():
        return None
    payload = _read_arm_reference(str(path), path.stat().st_mtime)
    buckets = payload.get("buckets")
    if not isinstance(buckets, dict):
        return None
    band = arm_reference.reach_band(reach_m)
    found = buckets.get(band)
    return (band, found) if isinstance(found, dict) else None


def _frame_seconds(value: str | float | None) -> float | None:
    """프레임 시각을 초로. 못 읽으면 `None`이다 — 게이트의 R7이 그때 간격 검사를 건너뛴다.

    시각 하나를 못 읽는다고 시퀀스를 죽이지 않는다. 샘플은 ISO-8601에 `Z`가 붙는데
    파이썬 3.11부터는 `fromisoformat`이 그대로 받는다.
    """
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value).timestamp()
    except ValueError:
        return None


def _gate_frames(frames: list[SequenceFrame]) -> tuple[list[skeleton_gate.GateFrame], list[str]]:
    """실물 프레임을 게이트가 보는 모양으로. 모르는 키는 버리지 않고 모아서 돌려준다."""
    out: list[skeleton_gate.GateFrame] = []
    unknown: set[str] = set()
    for frame in frames:
        unknown.update(frame.model_extra or {})
        unknown.update(joint_names.unknown_names(frame.skeleton_joints))
        tool = frame.tool_pos_rot or {}
        out.append(
            skeleton_gate.GateFrame(
                joints=joint_names.positions(frame.skeleton_joints),
                tool_pos=joint_names.parse_position(tool.get("pos")),
                time_s=_frame_seconds(frame.time),
            )
        )
    return out, sorted(unknown)


def _impact_points(frames: list[SequenceFrame]) -> list[tuple[float, float, float]]:
    """착탄점들. `raypoint`가 `null`인 프레임은 **빠진다** — 0으로 채우면 바닥을 칠한 게 된다.

    실물 SEQ-001이 그 경우로, 전 프레임이 `null`이라 착탄 높이가 없다.
    """
    found = []
    for frame in frames:
        point = joint_names.parse_position((frame.gun_metrics or {}).get("raypoint"))
        if point is not None:
            found.append(point)
    return found


@dataclass(frozen=True)
class _Prepared:
    """단계들이 공통으로 쓰는 재료. 한 번만 만들고 넘긴다."""

    frames: list[skeleton_gate.GateFrame]
    unknown: list[str]
    gate: skeleton_gate.GateResult
    impacts: list[tuple[float, float, float]]
    reach: float | None
    buckets: dict[str, Any] | None


def _prepare(payload: "SequenceRequest", *, touch_session: bool) -> _Prepared:
    """게이트까지 돌린다. `touch_session`이 False면 세션 상태를 건드리지 않는다 —
    단계 경로를 여러 번 불러도 자세 시퀀스가 부풀지 않아야 한다."""
    frames, unknown = _gate_frames(payload.sequence_info)
    session = _SESSIONS.get(payload.session_id)
    frozen_before = frozenset(session.frozen_joints) if session else frozenset()
    gate = skeleton_gate.evaluate(frames, frozen_before=frozen_before)
    impacts = _impact_points(payload.sequence_info)
    tools = [frame.tool_pos for frame in frames if frame.tool_pos is not None]
    reach = pose_stages.reach_of(impacts, tools)
    _ = touch_session
    return _Prepared(frames, unknown, gate, impacts, reach, _arm_buckets())


def _resolved_of(prepared: _Prepared, band: str | None) -> pose_stages.ResolveResult:
    """`band`가 주어지면 ①을 건너뛰고 그 버킷을 쓴다 — 앞 단계 출력을 넘겨받는 경로다.

    **실측 reach는 그대로 싣는다.** 밴드의 공칭값으로 덮어쓰면 응답의 `reach_m`이
    "이 시퀀스에서 잰 값"이 아니라 "우리가 고른 칸의 이름"이 된다. 둘이 어긋나면
    (밴드를 잘못 넘겼다는 뜻) 호출자가 그것을 봐야 한다.
    """
    if band is None:
        return pose_stages.resolve(prepared.reach, prepared.buckets)
    if not isinstance(prepared.buckets, dict) or band not in prepared.buckets:
        return pose_stages.ResolveResult(
            ok=False, reason="reference_data_missing", band=band, reach_m=prepared.reach
        )
    forced = pose_stages.resolve(_band_reach(band), prepared.buckets)
    return replace(forced, reach_m=prepared.reach)


def _band_reach(band: str) -> float:
    """밴드 이름 → 그 칸 안의 한 값. 버킷을 고르는 데만 쓰고 응답에 싣지 않는다."""
    try:
        return float(band) + arm_reference.REACH_BAND_M / 2
    except ValueError:
        return 0.0


def _world_frames(prepared: _Prepared) -> list[dict[str, tuple[float, float, float]]]:
    """프레임마다 **월드 관절**. 정규화는 `pose_stages`가 한다.

    월드를 넘기는 이유: 교정 계획을 월드로 되돌리려면 그 프레임의 몸 프레임이 필요한데,
    정규화된 좌표만 넘기면 그 정보가 사라진다.
    """
    return [dict(frame.joints) for frame in prepared.frames]


@router.post("/reference", response_model=MCPResponse)
def resolve_reference(payload: "SequenceRequest") -> MCPResponse:
    """① 이 시퀀스가 어느 숙련자 버킷인가. 정렬·채점은 하지 않는다."""
    prepared = _prepare(payload, touch_session=False)
    resolved = pose_stages.resolve(prepared.reach, prepared.buckets)
    return success_response(result={
        "mode": "reference",
        "sequence_id": payload.sequence_id,
        "input_quality": _quality_block(prepared),
        **resolved.to_dict(),
    })


@router.post("/align", response_model=MCPResponse)
def align_sequence(payload: "SequenceRequest") -> MCPResponse:
    """② 착탄 경로를 버킷 프로토타입에 DTW로 맞춘다. 점수는 내지 않는다."""
    prepared = _prepare(payload, touch_session=False)
    resolved = _resolved_of(prepared, payload.band)
    aligned = pose_stages.align(prepared.impacts, resolved)
    return success_response(result={
        "mode": "align",
        "sequence_id": payload.sequence_id,
        "resolve": resolved.to_dict(),
        **aligned.to_dict(),
    })


@router.post("/score", response_model=MCPResponse)
def score_sequence(payload: "SequenceRequest") -> MCPResponse:
    """③ 위상 자세를 위상 레퍼런스에 대고 잰다. 앞 두 단계 출력도 같이 싣는다."""
    prepared = _prepare(payload, touch_session=False)
    resolved = _resolved_of(prepared, payload.band)
    aligned = pose_stages.align(prepared.impacts, resolved)
    bucket: dict[str, Any] = {}
    if resolved.band and isinstance(prepared.buckets, dict):
        found = prepared.buckets.get(resolved.band)
        if isinstance(found, dict):
            bucket = found
    scored = pose_stages.score(
        _world_frames(prepared),
        aligned,
        bucket,
        excluded=prepared.gate.excluded_joints,
        labels=pose_similarity.JOINT_LABELS,
    )
    return success_response(result={
        "mode": "score",
        "sequence_id": payload.sequence_id,
        "scoring_set_id": arm_reference.SCORING_SET_ID,
        "correction_mode": "joint_only",
        "resolve": resolved.to_dict(),
        "align": aligned.to_dict(),
        **scored.to_dict(),
    })


@router.post("/verify", response_model=MCPResponse)
def verify_sequence(payload: "SequenceRequest") -> MCPResponse:
    """교정을 **실제로 적용해** 예고한 상승분과 맞는지 잰다 — 교정 정확도의 내적 일관성.

    효능(작업자가 실제로 나아지는가)이 아니다. 그건 런타임 데이터가 있어야 한다.
    """
    prepared = _prepare(payload, touch_session=False)
    resolved = _resolved_of(prepared, payload.band)
    aligned = pose_stages.align(prepared.impacts, resolved)
    bucket: dict[str, Any] = {}
    if resolved.band and isinstance(prepared.buckets, dict):
        found = prepared.buckets.get(resolved.band)
        if isinstance(found, dict):
            bucket = found
    verified = pose_stages.verify_correction(
        _world_frames(prepared), aligned, bucket, excluded=prepared.gate.excluded_joints
    )
    return success_response(result={
        "mode": "verify",
        "sequence_id": payload.sequence_id,
        "resolve": resolved.to_dict(),
        "align": aligned.to_dict(),
        **verified.to_dict(),
    })


def _quality_block(prepared: _Prepared) -> dict[str, Any]:
    gate = prepared.gate
    return {
        "status": gate.status,
        "reason_codes": list(gate.reason_codes),
        "info_codes": list(gate.info_codes),
        "frozen_joints": list(gate.frozen_joints),
        "frames": len(prepared.frames),
        "unknown_fields": prepared.unknown,
        "measurements": {k: round(v, 4) for k, v in gate.measurements.items()},
    }


@router.post("/sequence", response_model=MCPResponse)
def analyze_sequence(payload: SequenceRequest) -> MCPResponse:
    """시퀀스 하나를 받아 자세를 판정하고, 세션에 쌓아 자세 시퀀스를 돌려준다.

    **지금은 어떤 입력에도 `similarity_score`가 `null`이다.** 레퍼런스 아티팩트가 축
    규약 수정으로 버킷 64.33%가 바뀔 예정이라 지금 기준에 대고 낸 점수는 곧 무효다.
    `0.0`을 내면 '많이 다르다'로 읽히는데 그건 우리가 재지 않은 것에 대한 주장이다.
    **키 집합은 레퍼런스가 준비된 세계와 같고 값만 다르다** — 소비자는 다시 고치지 않는다.
    """
    frames, unknown = _gate_frames(payload.sequence_info)
    session = _SESSIONS.get_or_create(payload.session_id)

    # `frozen_before`는 **앞선** 시퀀스의 것이어야 한다(R4c). 그래서 쌓기 전에 잰다.
    gate = skeleton_gate.evaluate(frames, frozen_before=frozenset(session.frozen_joints))

    prepared_buckets = _arm_buckets()
    impacts = _impact_points(payload.sequence_info)
    reduced = posture_sequence.summarize(
        [frame.joints["Head"] for frame in frames if "Head" in frame.joints],
        impacts,
    )

    # reach = 착탄 수직 − **건** 월드 수직. 머리를 쓰지 않는 이유는 `arm_reference`
    # 독스트링에 있다 — CSV의 머리는 HMD 강체, 런타임의 머리는 골격 관절이라 같은 점이
    # 아니고, 0.10 m 어긋나면 버킷의 44%가 바뀐다. 건은 양쪽 모두 같은 물건이다.
    tool_heights = [
        world_axes.vertical_of(frame.tool_pos) for frame in frames if frame.tool_pos is not None
    ]
    impact_height = None if reduced is None else reduced[1]
    reach = (
        None
        if impact_height is None or not tool_heights
        else impact_height - statistics.median(tool_heights)
    )
    found = _arm_bucket(reach)
    normalized = arm_reference.normalize_sequence([frame.joints for frame in frames])

    sigmas: list[arm_reference.JointSigma] = []
    quality = "unavailable"
    band: str | None = None
    support: int | None = None
    bucket: dict[str, Any] = {}
    if found is not None and normalized is not None:
        band, bucket = found
        quality = arm_reference.reference_quality(bucket)
        support = int(bucket.get("n", 0) or 0)
        if quality != "unavailable":
            sigmas = [
                item
                for item in arm_reference.joint_sigmas(normalized, bucket)
                if item.joint not in gate.excluded_joints
            ]
    # 위상 채점을 먼저 시도한다. 착탄이 움직이고 위상 레퍼런스가 있으면 그쪽이 낫다 —
    # 국소 오차를 접기보다 3.7배 잡는다. **안 되면 접기로 내려간다** — 실물 SEQ-002처럼
    # 건을 고정한 시퀀스는 정렬할 모양이 없어서 위상 채점 자체가 불가능하다.
    scoring_mode = "fold"
    phase_result: pose_stages.ScoreResult | None = None
    if quality != "unavailable" and band is not None:
        resolved = pose_stages.resolve(reach, prepared_buckets)
        aligned = pose_stages.align(impacts, resolved)
        attempt = pose_stages.score(
            [dict(frame.joints) for frame in frames],
            aligned,
            bucket,
            excluded=gate.excluded_joints,
            labels=pose_similarity.JOINT_LABELS,
        )
        if attempt.ok:
            phase_result, scoring_mode = attempt, "phase"
            sigmas = list(attempt.sigmas)

    used = tuple(item.joint for item in sigmas)
    if reduced is not None and gate.status != "reject":
        session.add(
            posture_sequence.PostureSample(
                sequence_id=payload.sequence_id,
                head_height_m=reduced[0],
                impact_height_m=reduced[1],
                frames=len(frames),
                gate_status=gate.status,
                excluded_joints=gate.excluded_joints,
            ),
            frozen=gate.frozen_joints,
        )

    report = posture_sequence.session_report(session, payload.floor_height_m)

    # 백분위는 **한 번만** 낸다. 필드와 문장이 따로 계산하면 같은 응답이 두 숫자를 말한다 —
    # 실제로 그랬다(필드 0, 문장 "10백분위"). 문장이 모드를 안 넘겨 접기 사다리를 썼다.
    if phase_result is not None:
        score = phase_result.score
        corrections = list(phase_result.corrections)
        plans = list(phase_result.plans)
    else:
        score = arm_reference.similarity_score(sigmas)
        corrections = arm_reference.build_corrections(
            sigmas, pose_similarity.JOINT_LABELS, support=support, reference=bucket.get("joints")
        )
        plans = [() for _ in corrections]
    percentile = (
        None if score is None else arm_reference.expert_percentile(bucket, score, mode=scoring_mode)
    )

    reason: str | None
    if gate.status == "reject":
        status, reason = "not_available", gate.reason_codes[0]
    elif quality == "unavailable" or score is None:
        status, reason = "not_available", "reference_data_missing"
    elif len(used) < arm_reference.MIN_SCORED_JOINTS:
        status, reason = "not_available", "quality_insufficient"
    elif quality in ("single_worker", "coarse_bucket"):
        # 작업자 1인 산포는 '숙련자 통상 범위'가 아니라 '그 사람의 습관'이다.
        status, reason = "degraded", "reference_single_worker"
    elif gate.status == "degrade":
        # 레퍼런스는 멀쩡하고 **입력**이 일부 빠졌다. 두 사실을 같은 사유로 적으면
        # 소비자가 무엇을 고쳐야 하는지 못 가린다.
        status, reason = "degraded", gate.reason_codes[0]
    else:
        status, reason = "available", None
    assert reason is None or reason in SEQUENCE_REASON_CODES, reason

    return success_response(result={
        "mode": "sequence",
        "sequence_id": payload.sequence_id,
        "analysis_status": status,
        "reason_code": reason,
        "scoring_set_id": arm_reference.SCORING_SET_ID,
        "scored_joints": list(used),
        "excluded_joints": list(gate.excluded_joints),
        "reference_quality": quality,
        "reference_band": band,
        "reach_m": None if reach is None else round(reach, 4),
        "support": support,
        "similarity_score": None if score is None else round(score, 4),
        # 점수의 절대값은 축 수·산포 하한·가중치가 스케일을 건드려 해석하기 어렵다.
        # **백분위가 해석 가능한 쪽이다** — 숙련자 분포에서 어디쯤인지.
        "expert_percentile": percentile,
        "correction_mode": "joint_only",
        # 어느 채점을 썼나. `phase`면 DTW 정렬로 위상마다 쟀고, `fold`면 시퀀스를
        # 한 점으로 접어 쟀다. 두 점수는 **다른 분포**라 나란히 놓으면 안 된다.
        "scoring_mode": scoring_mode,
        "corrections": [
            pose_stages._correction_dict(item, plans[index] if index < len(plans) else ())
            for index, item in enumerate(corrections)
        ],
        "joint_sigmas": {item.joint: round(item.sigma, 3) for item in sigmas},
        "input_quality": {
            "status": gate.status,
            "reason_codes": list(gate.reason_codes),
            "info_codes": list(gate.info_codes),
            "frozen_joints": list(gate.frozen_joints),
            "frames": len(frames),
            "unknown_fields": unknown,
            "measurements": {k: round(v, 4) for k, v in gate.measurements.items()},
        },
        "posture_sequence": {
            "session_id": payload.session_id,
            "posture_point_recorded": reduced is not None and gate.status != "reject",
            "head_height_m": None if reduced is None else round(reduced[0], 4),
            "impact_height_m": None if reduced is None or reduced[1] is None else round(reduced[1], 4),
            "sequences": report["sequences"],
            "tracking_slope": report["tracking_slope"],
            "tracking_r": report["tracking_r"],
            "tracking_n": report["tracking_n"],
            "posture_band_ok": report["posture_band_ok"],
            "frozen_joints": report["frozen_joints"],
            "evicted_sessions": _SESSIONS.evicted,
        },
        "summary": _sequence_summary(status, reason, report, score, percentile, corrections),
    })


def _sequence_summary(
    status: str,
    reason: str | None,
    report: Mapping[str, object],
    score: float | None,
    percentile: int | None,
    corrections: Sequence[arm_reference.Correction],
) -> str:
    if score is None:
        return f"자세 판정을 제공할 수 없습니다({reason}). 점수는 0점이 아니라 미측정입니다."
    head = f"숙련자 팔 자세 대비 유사도 {score:.0%}"
    if isinstance(percentile, int):
        head += f" — 숙련자 하위 {percentile}%" if percentile == 0 else f" — 숙련자 {percentile}백분위"
    if status == "degraded":
        head += "(참고용입니다)"
    if corrections:
        # **교정의 문장을 그대로 인용한다.** 요약이 거리를 따로 적으면 두 숫자가 갈린다 —
        # 실제로 교정은 반경 편차 31 cm를, 요약은 3D 편차 57 cm를 말하고 있었다.
        head += f". {corrections[0].text()[:-1]}"
    else:
        head += ". 여섯 관절 모두 숙련자 통상 범위 안입니다"
    slope = report.get("tracking_slope")
    if isinstance(slope, float):
        head += f". 착탄을 몸이 따라간 기울기 {slope:+.3f}(시퀀스 {report['tracking_n']}개)"
    return head + "."
