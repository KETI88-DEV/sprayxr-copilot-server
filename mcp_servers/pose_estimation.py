from __future__ import annotations

import json
import os
from functools import lru_cache
from math import atan2, degrees, sqrt
from pathlib import Path
from typing import Any, Literal, Optional, assert_never

from fastapi import APIRouter
from pydantic import BaseModel, model_validator

from mcp_servers.common import MCPResponse, success_response
from trainer.scripts import pose_similarity


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


def _reference_bucket(level: str | None, bucket: str) -> dict[str, Any] | None:
    """레퍼런스 아티팩트에서 버킷 하나를 찾는다. 없으면 추정하지 않고 None을 낸다."""
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
    return found if isinstance(found, dict) else None


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

    reference = _reference_bucket(payload.level, bucket)
    if reference is None:
        return _unavailable(
            "reference_data_missing",
            "이 시퀀스에 적용할 기준 자세가 없어 자세 판정을 제공할 수 없습니다.",
        )

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
