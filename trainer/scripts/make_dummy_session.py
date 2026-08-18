#!/usr/bin/env python3
"""BVH 자세로 납품 형식(관절 월드 좌표 CSV) 더미 세션을 만든다.

실납품은 관절 좌표를 월드로 변환한 CSV로 온다. 그러면 관절이 건·카메라와 같은 행에 있어
정합 문제가 생기지 않는다. 그 형식을 지금 확보한 BVH로 미리 만들어 파이프라인을 돌린다.

착탄점은 대용이다. BVH에는 표면이 없어 건을 쥔 손 위치를 Ray point로 적는다. 스트로크
분할과 버킷 산출에는 충분하지만 실제 도막 위치가 아니다.
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
from pathlib import Path
from typing import Final, Mapping, Sequence

import bvh_skeleton
from simulator_csv import JOINT_COLUMNS, JOINT_KEYS, PARENT_OF, descendants

logger = logging.getLogger(__name__)

Vec3 = tuple[float, float, float]

EXPORT_JOINTS: Final = tuple(JOINT_KEYS[column] for column in JOINT_COLUMNS)
GUN_COLUMN: Final = "한길숏건"
CAMERA_COLUMN: Final = "카메라"
GUN_JOINT: Final = "RightHand"
CAMERA_JOINT: Final = "Head"
RAY_COLUMNS: Final = ("Ray point X", "Ray Point Y", "Ray Point Z")
DEFAULT_SEQUENCE_FRAMES: Final = 600
SPRAY_GAP_FRAMES: Final = 3


def mutate(
    poses: Sequence[Mapping[str, Vec3]], joint: str, offset: Vec3
) -> list[dict[str, Vec3]]:
    """관절 하나를 알려진 만큼 밀어 둔 사본. 비교 결과가 이 값을 되찾아야 한다."""
    shifted: list[dict[str, Vec3]] = []
    for pose in poses:
        copy = dict(pose)
        current = copy.get(joint)
        if current is not None:
            copy[joint] = (
                current[0] + offset[0],
                current[1] + offset[1],
                current[2] + offset[2],
            )
        shifted.append(copy)
    return shifted


def _rotate_about(point: Vec3, pivot: Vec3, axis: Vec3, angle: float) -> Vec3:
    """로드리게스 회전. 피벗 기준 강체 회전이라 사슬 안의 뼈 길이가 모두 보존된다."""
    vector = (point[0] - pivot[0], point[1] - pivot[1], point[2] - pivot[2])
    cosine = math.cos(angle)
    sine = math.sin(angle)
    dot = axis[0] * vector[0] + axis[1] * vector[1] + axis[2] * vector[2]
    cross = (
        axis[1] * vector[2] - axis[2] * vector[1],
        axis[2] * vector[0] - axis[0] * vector[2],
        axis[0] * vector[1] - axis[1] * vector[0],
    )
    return (
        pivot[0] + vector[0] * cosine + cross[0] * sine + axis[0] * dot * (1.0 - cosine),
        pivot[1] + vector[1] * cosine + cross[1] * sine + axis[1] * dot * (1.0 - cosine),
        pivot[2] + vector[2] * cosine + cross[2] * sine + axis[2] * dot * (1.0 - cosine),
    )


def mutate_limb(
    poses: Sequence[Mapping[str, Vec3]], joint: str, degrees: float
) -> list[dict[str, Vec3]]:
    """부모 관절을 축으로 팔다리를 들어올린다. 자식 관절이 함께 돌아 뼈 길이가 보존된다.

    관절 하나만 평행이동하면 붙어 있는 두 뼈가 늘어나 사람이 취할 수 없는 자세가 된다.
    회전축은 뼈와 수직축에 모두 수직인 방향이라 "들어올린다/내린다"에 해당한다.
    """
    parent = PARENT_OF.get(joint)
    chain = {joint} | descendants(joint)
    angle = math.radians(degrees)

    shifted: list[dict[str, Vec3]] = []
    for pose in poses:
        copy = dict(pose)
        pivot = pose.get(parent) if parent is not None else None
        head = pose.get(joint)
        if pivot is None or head is None:
            shifted.append(copy)
            continue
        bone = (head[0] - pivot[0], head[1] - pivot[1], head[2] - pivot[2])
        axis = (bone[1], -bone[0], 0.0)
        length = math.sqrt(axis[0] ** 2 + axis[1] ** 2)
        if length == 0.0:
            shifted.append(copy)
            continue
        unit = (axis[0] / length, axis[1] / length, 0.0)
        for name in chain:
            point = pose.get(name)
            if point is not None:
                copy[name] = _rotate_about(point, pivot, unit, angle)
        shifted.append(copy)
    return shifted


def _header() -> list[str]:
    columns = ["인덱스", "시간(초)", "분사", *RAY_COLUMNS]
    for prefix in (GUN_COLUMN, CAMERA_COLUMN):
        columns += [f"{prefix} Position {axis}" for axis in "XYZ"]
    for column in JOINT_COLUMNS:
        columns += [f"{column} Position {axis}" for axis in "XYZ"]
    return columns


def write_session(
    path: str | Path,
    poses: Sequence[Mapping[str, Vec3]],
    frame_time: float,
    sequence_frames: int = DEFAULT_SEQUENCE_FRAMES,
) -> Path:
    """행 하나가 자세 하나. 관절과 건·카메라가 같은 행에 있으므로 정합이 항등이다.

    BVH에는 분사 신호가 없으므로 sequence_frames 주기로 짧은 Off 구간을 넣어 버스트를
    나눈다. 같은 스트로크를 다른 시퀀스에서 비교하려면 시퀀스가 여러 개여야 한다.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    cycle = sequence_frames + SPRAY_GAP_FRAMES
    with target.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(_header())
        for index, pose in enumerate(poses):
            gun = pose.get(GUN_JOINT, (0.0, 0.0, 0.0))
            camera = pose.get(CAMERA_JOINT, (0.0, 0.0, 0.0))
            spray = "On" if index % cycle < sequence_frames else "Off"
            row: list[object] = [index, round(index * frame_time, 5), spray, *gun, *gun, *camera]
            for column in JOINT_COLUMNS:
                row += list(pose.get(JOINT_KEYS[column], (0.0, 0.0, 0.0)))
            writer.writerow(row)
    return target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a world-coordinate dummy session from BVH.")
    parser.add_argument("--bvh", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--sequence-frames",
        type=int,
        default=DEFAULT_SEQUENCE_FRAMES,
        help="한 시퀀스(버스트)에 담을 프레임 수.",
    )
    parser.add_argument("--mutate-joint", default=None, help="편차를 주입할 관절 이름.")
    parser.add_argument("--mutate-offset", default=None, help="평행이동 오프셋 x,y,z (m). 뼈 길이가 깨지므로 회복 검산용으로만 쓴다.")
    parser.add_argument("--mutate-degrees", type=float, default=None, help="부모 기준 회전 각도. 뼈 길이를 보존한다.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")
    if not args.bvh.is_file():
        logger.error("bvh not found: %s", args.bvh)
        return 1

    clip = bvh_skeleton.parse_bvh(args.bvh)
    poses = [
        bvh_skeleton.forward_kinematics(clip, index, joints=EXPORT_JOINTS)
        for index in range(len(clip.frames))
    ]
    if args.mutate_joint and args.mutate_degrees is not None:
        poses = mutate_limb(poses, args.mutate_joint, args.mutate_degrees)
        logger.info("rotated %s by %.1f deg", args.mutate_joint, args.mutate_degrees)
    elif args.mutate_joint and args.mutate_offset:
        parts = [float(value) for value in args.mutate_offset.split(",")]
        poses = mutate(poses, args.mutate_joint, (parts[0], parts[1], parts[2]))
        logger.warning("translated %s by %s - bone lengths are broken on purpose", args.mutate_joint, parts)

    written = write_session(args.out, poses, clip.frame_time, args.sequence_frames)
    logger.info("wrote %s (%d rows, %d joints)", written, len(poses), len(EXPORT_JOINTS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
