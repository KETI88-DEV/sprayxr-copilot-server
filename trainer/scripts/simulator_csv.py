#!/usr/bin/env python3
"""The raw simulator CSV dialect: one exported row becomes one session frame.

Column names are Korean, the header carries a BOM, and the vendor ships
" Ray Point Y" / " Ray Point Z" with a leading space - callers must strip the
header before lookup.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Final, Sequence

OBJECT_KEYS: Final[dict[str, str]] = {
    "한길숏건": "gun_hangil_short",
    "한길폴건": "gun_hangil_pole",
    "반도숏건": "gun_bando_short",
    "반도폴건": "gun_bando_pole",
    "플래쉬": "flashlight",
    "카메라": "camera",
}
JOINT_COLUMNS: Final[tuple[str, ...]] = (
    "Pelvis", "Spine", "Chest",
    "Left Shoulder", "Left Upper Arm", "Left Fore Arm", "Left Hand",
    "Right Shoulder", "Right Upper Arm", "Right Fore Arm", "Right Hand",
    "Left Thigh", "Left Calf", "Left Foot", "Left Toe",
    "Right Thigh", "Right Calf", "Right Foot", "Right Toe",
    "Head",
)
JOINT_KEYS: Final[dict[str, str]] = {name: name.replace(" ", "") for name in JOINT_COLUMNS}
# 관절 계층. 뼈 길이를 보존해야 하는 쪽(주입·렌더)이 모두 이 표를 쓴다.
BONES: Final[tuple[tuple[str, str], ...]] = (
    ("Pelvis", "Spine"),
    ("Spine", "Chest"),
    ("Chest", "Head"),
    ("Chest", "LeftShoulder"),
    ("LeftShoulder", "LeftUpperArm"),
    ("LeftUpperArm", "LeftForeArm"),
    ("LeftForeArm", "LeftHand"),
    ("Chest", "RightShoulder"),
    ("RightShoulder", "RightUpperArm"),
    ("RightUpperArm", "RightForeArm"),
    ("RightForeArm", "RightHand"),
    ("Pelvis", "LeftThigh"),
    ("LeftThigh", "LeftCalf"),
    ("LeftCalf", "LeftFoot"),
    ("LeftFoot", "LeftToe"),
    ("Pelvis", "RightThigh"),
    ("RightThigh", "RightCalf"),
    ("RightCalf", "RightFoot"),
    ("RightFoot", "RightToe"),
)
PARENT_OF: Final[dict[str, str]] = {child: parent for parent, child in BONES}


def descendants(joint: str) -> set[str]:
    """joint 아래로 매달린 모든 관절. 회전 시 함께 움직여야 뼈 길이가 보존된다."""
    found: set[str] = set()
    frontier = [joint]
    while frontier:
        current = frontier.pop()
        for parent, child in BONES:
            if parent == current and child not in found:
                found.add(child)
                frontier.append(child)
    return found


ROOT_COLUMN: Final = "Pelvis"
LIMB_COLUMNS: Final[tuple[str, ...]] = JOINT_COLUMNS[1:]

# Field order of the ``thick@`` payload, fixed by the vendor Q&A in
# data/sources/professionals/readme.md. Only maker_name is non-numeric.
MODEL_INPUT_FIELDS: Final[tuple[str, ...]] = (
    "viscosity", "nozzle", "gun_speed", "set_pressure", "gun_distance", "push_time",
    "svr", "thinner_dilution_law", "temperature", "maker_name", "h1", "h2", "h3",
)
MODEL_INPUT_TEXT_FIELD: Final = "maker_name"

GUN_METRIC_COLUMNS: Final[dict[str, str]] = {
    "Distance (Input)": "distance_input_mm",
    "Speed (Input)": "speed_input_mms",
    "Distance (Real)": "distance_real_mm",
    "Speed (Real)": "speed_real_mms",
}
RAY_POINT_COLUMNS: Final[tuple[str, str, str]] = ("Ray point X", "Ray Point Y", "Ray Point Z")


MIN_JOINT_MOTION_M: Final = 0.005


def vec3(row: dict[str, str], group: str, kind: str) -> list[float]:
    return [float(row[f"{group} {kind} {axis}"]) for axis in "XYZ"]


def joint_world_positions(row: dict[str, str]) -> dict[str, tuple[float, float, float]]:
    """관절 Position 컬럼을 정규화된 이름으로 읽는다. 값이 월드 좌표인지는 보증하지 않는다."""
    positions: dict[str, tuple[float, float, float]] = {}
    for column in JOINT_COLUMNS:
        x, y, z = vec3(row, column, "Position")
        positions[JOINT_KEYS[column]] = (x, y, z)
    return positions


def has_world_joint_motion(
    rows: Sequence[dict[str, str]], joints: Sequence[str]
) -> bool:
    """요청한 관절이 실제로 움직이는지. 바인드 포즈 오프셋이면 전부 상수라 거짓이다.

    현재 납품본은 Pelvis를 뺀 관절 Position이 전부 상수다. 이것을 월드 좌표로 오인하면
    모든 표본이 같은 자세가 되어 유사도가 무의미해진다.
    """
    if len(rows) < 2:
        return False
    wanted = set(joints)
    columns = [column for column in JOINT_COLUMNS if JOINT_KEYS[column] in wanted]
    if len(columns) != len(wanted):
        return False
    for column in columns:
        samples = [vec3(row, column, "Position") for row in rows]
        spread = max(
            max(point[axis] for point in samples) - min(point[axis] for point in samples)
            for axis in range(3)
        )
        if spread < MIN_JOINT_MOTION_M:
            return False
    return True


def _model_input(raw: str) -> dict[str, Any] | None:
    parts = raw.split("@")
    if len(parts) != len(MODEL_INPUT_FIELDS) + 1:
        return None
    values: dict[str, Any] = {"tag": parts[0]}
    for name, value in zip(MODEL_INPUT_FIELDS, parts[1:]):
        values[name] = value if name == MODEL_INPUT_TEXT_FIELD else float(value)
    return values


def _surface_hit(row: dict[str, str]) -> dict[str, Any] | None:
    if not row[RAY_POINT_COLUMNS[0]].strip():
        return None
    map_number = row["Ray Texture Map Number"].strip()
    return {
        "hit_pos_m": [float(row[column]) for column in RAY_POINT_COLUMNS],
        "region_id": f"MAP{map_number}",
        "map_number": map_number,
        "uv_texel": [float(row["Ray Texture UV X"]), float(row["Ray Texture UV Y"])],
    }


def frame_from_row(row: dict[str, str]) -> dict[str, Any]:
    return {
        "i": int(row["인덱스"]),
        "t_sec": float(row["시간(초)"]),
        "spray_on": row["분사"].strip() == "On",
        "record_flag": row["Record"].strip() == "On",
        "objects": {
            key: {
                "pos_m": vec3(row, column, "Position"),
                "rot_euler_deg": vec3(row, column, "Rotation"),
            }
            for column, key in OBJECT_KEYS.items()
        },
        "skeleton": {
            "root_pos_m": vec3(row, ROOT_COLUMN, "Position"),
            "root_rot_euler_deg": vec3(row, ROOT_COLUMN, "Rotation"),
            "joints_rot_euler_deg": {
                JOINT_KEYS[column]: vec3(row, column, "Rotation") for column in LIMB_COLUMNS
            },
        },
        "gun_metrics": {name: float(row[column]) for column, name in GUN_METRIC_COLUMNS.items()},
        "surface_hit": _surface_hit(row),
        "model_input": _model_input(row["Model Input Data"]),
    }


def read_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = [column.strip() for column in next(reader)]
        return [dict(zip(header, row)) for row in reader if row]


def read_settings(csv_path: Path) -> dict[str, str]:
    settings_path = csv_path.with_suffix(".txt")
    if not settings_path.is_file():
        return {}
    entries = (line.partition("=") for line in settings_path.read_text(encoding="utf-8").splitlines())
    return {key.strip(): value.strip() for key, _, value in entries if key.strip()}
