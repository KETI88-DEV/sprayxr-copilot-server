from __future__ import annotations

from math import sqrt

from fastapi import APIRouter
from pydantic import BaseModel

from mcp_servers.common import MCPResponse, success_response


router = APIRouter(prefix="/pose_estimation", tags=["pose_estimation"])


class Joint(BaseModel):
    x: float
    y: float
    z: float
    roll: float | None = None
    pitch: float | None = None
    yaw: float | None = None


class Pose(BaseModel):
    joints: dict[str, Joint]


class ImuData(BaseModel):
    accel: list[float] | None = None
    gyro: list[float] | None = None


class PoseCompareRequest(BaseModel):
    current_pose: Pose
    reference_pose: Pose
    imu_data: ImuData | None = None


@router.post("/compare", response_model=MCPResponse)
def compare_pose(payload: PoseCompareRequest) -> MCPResponse:
    common_joints = sorted(set(payload.current_pose.joints).intersection(payload.reference_pose.joints))
    deviations: dict[str, dict[str, float]] = {}
    total_distance = 0.0

    for name in common_joints:
        current = payload.current_pose.joints[name]
        reference = payload.reference_pose.joints[name]
        dx = current.x - reference.x
        dy = current.y - reference.y
        dz = current.z - reference.z
        distance = sqrt((dx * dx) + (dy * dy) + (dz * dz))
        total_distance += distance
        deviations[name] = {
            "dx": dx,
            "dy": dy,
            "dz": dz,
            "distance": distance,
        }

    mean_distance = total_distance / len(common_joints) if common_joints else 0.0
    similarity_score = max(0.0, min(1.0, 1.0 - (mean_distance / 100.0)))

    result = {
        "joint_count": len(common_joints),
        "deviations": deviations,
        "mean_deviation": mean_distance,
        "similarity_score": similarity_score,
    }
    if payload.imu_data is not None:
        result["imu_data"] = payload.imu_data.model_dump(exclude_none=True)
    return success_response(result=result)
