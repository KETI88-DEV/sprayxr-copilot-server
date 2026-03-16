from __future__ import annotations

from math import atan2, degrees, sqrt
from typing import Literal, Optional

from fastapi import APIRouter
from pydantic import BaseModel

from mcp_servers.common import MCPResponse, success_response


router = APIRouter(prefix="/pose_estimation", tags=["pose_estimation"])

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


@router.post("/compare", response_model=MCPResponse)
def compare_pose(payload: PoseRequest) -> MCPResponse:
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
