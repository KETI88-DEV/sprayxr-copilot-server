"""Read-only browser viewer for expert skeleton motion references."""

from __future__ import annotations

import copy
import csv
import gzip
import json
import math
import re
import statistics
from functools import lru_cache
from pathlib import Path
from typing import Any, TextIO

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse
from trainer.demo_api import coating_assessment
from trainer.demo_api.portal_navigation import portal_document
from trainer.zone_reference import ZoneReferenceCatalog

from trainer.demo_api.motion_viewer_pose import _default_mesh_provider, build_pose_view
from trainer.demo_api.pose_query import score_body_raw
from trainer.demo_api.sequence_store import SequenceStore
from trainer.demo_api.sequence_analysis import SequenceAnalysisEngine, expert_reference_flows, score_visit_flow
from trainer.demo_api.v1_store import V1Store
from trainer.expert_zone_mapping import canonical_expert_zone_id

_JOINT_PARENTS: tuple[tuple[str, str | None], ...] = (
    ("Pelvis", None),
    ("Spine", "Pelvis"),
    ("Chest", "Spine"),
    ("Head", "Chest"),
    ("Left Shoulder", "Chest"),
    ("Left Upper Arm", "Left Shoulder"),
    ("Left Fore Arm", "Left Upper Arm"),
    ("Left Hand", "Left Fore Arm"),
    ("Right Shoulder", "Chest"),
    ("Right Upper Arm", "Right Shoulder"),
    ("Right Fore Arm", "Right Upper Arm"),
    ("Right Hand", "Right Fore Arm"),
    ("Left Thigh", "Pelvis"),
    ("Left Calf", "Left Thigh"),
    ("Left Foot", "Left Calf"),
    ("Left Toe", "Left Foot"),
    ("Right Thigh", "Pelvis"),
    ("Right Calf", "Right Thigh"),
    ("Right Foot", "Right Calf"),
    ("Right Toe", "Right Foot"),
)

_BONES = [[parent, joint] for joint, parent in _JOINT_PARENTS if parent is not None]
_PROCESS_LABELS = {"spray": "스프레이", "blasting": "블라스팅"}
_EXPERT_NAME = "전형기"
_EXPERT_WORKER_ID = "jeon_hyeonggi"
_DIRECTION_LABELS = {
    "horizontal": "수평 중심",
    "vertical": "수직 중심",
    "diagonal": "대각선 중심",
    "curved_irregular": "곡선·불규칙",
    "unavailable": "분석 자료 없음",
}


def _number(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _normalized_row(row: dict[str, str]) -> dict[str, str]:
    return {str(key).strip().casefold(): value for key, value in row.items()}


def _vector(row: dict[str, str], prefix: str) -> list[float] | None:
    normalized = _normalized_row(row)
    values = [_number(normalized.get(f"{prefix} position {axis}".casefold())) for axis in "XYZ"]
    if any(value is None for value in values):
        return None
    return [round(float(value), 6) for value in values]


def _ray_point(row: dict[str, str]) -> list[float] | None:
    normalized = _normalized_row(row)
    values = [_number(normalized.get(f"ray point {axis}")) for axis in "xyz"]
    if any(value is None for value in values):
        return None
    return [round(float(value), 6) for value in values]


def _map_zone_uv(row: dict[str, str]) -> tuple[str, str, float, float] | None:
    normalized = _normalized_row(row)
    map_value = _number(normalized.get("ray texture map number"))
    zone_value = str(normalized.get("zone id", "")).strip()
    u = _number(normalized.get("ray texture uv x"))
    v = _number(normalized.get("ray texture uv y"))
    if map_value is None or not zone_value or zone_value == "-1" or u is None or v is None:
        return None
    map_num = str(int(map_value)) if map_value.is_integer() else str(map_value)
    try:
        zone_number = float(zone_value)
        zone_id = str(int(zone_number)) if zone_number.is_integer() else zone_value
    except ValueError:
        zone_id = zone_value
    return map_num, zone_id, u, v


def _spraying(value: Any) -> bool:
    return str(value).strip().casefold() in {"on", "true", "1", "yes"}


def _multiply_quaternion(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return (
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    )


def _axis_quaternion(axis: str, radians: float) -> tuple[float, float, float, float]:
    sine = math.sin(radians / 2.0)
    cosine = math.cos(radians / 2.0)
    if axis == "x":
        return sine, 0.0, 0.0, cosine
    if axis == "y":
        return 0.0, sine, 0.0, cosine
    return 0.0, 0.0, sine, cosine


def _unity_euler_quaternion(x: float, y: float, z: float) -> tuple[float, float, float, float]:
    """Match Unity Quaternion.Euler's Z-X-Y application order."""

    qx = _axis_quaternion("x", math.radians(x))
    qy = _axis_quaternion("y", math.radians(y))
    qz = _axis_quaternion("z", math.radians(z))
    return _multiply_quaternion(_multiply_quaternion(qy, qx), qz)


def _rotate(
    quaternion: tuple[float, float, float, float],
    vector: tuple[float, float, float],
) -> tuple[float, float, float]:
    qx, qy, qz, qw = quaternion
    vx, vy, vz = vector
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    return (
        vx + qw * tx + qy * tz - qz * ty,
        vy + qw * ty + qz * tx - qx * tz,
        vz + qw * tz + qx * ty - qy * tx,
    )


def _rotation_forward(
    row: dict[str, str],
    prefix: str,
    local_forward: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> list[float] | None:
    normalized = _normalized_row(row)
    values = [_number(normalized.get(f"{prefix} rotation {axis}".casefold())) for axis in "XYZ"]
    if any(value is None for value in values):
        return None
    quaternion = _unity_euler_quaternion(*values)  # type: ignore[arg-type]
    return [round(value, 6) for value in _rotate(quaternion, local_forward)]


def _camera_forward(row: dict[str, str]) -> list[float] | None:
    return _rotation_forward(row, "카메라")


def _joint_angle(
    first: list[float], vertex: list[float], third: list[float]
) -> float | None:
    left = [first[index] - vertex[index] for index in range(3)]
    right = [third[index] - vertex[index] for index in range(3)]
    left_length = math.sqrt(sum(value * value for value in left))
    right_length = math.sqrt(sum(value * value for value in right))
    if left_length <= 1e-9 or right_length <= 1e-9:
        return None
    cosine = sum(a * b for a, b in zip(left, right)) / (left_length * right_length)
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def _joint_posture_state(joints: dict[str, list[float]]) -> dict[str, Any]:
    """Classify display posture from the CSV skeleton, independent of trackers.

    The skeleton joints are simulator IK output rather than tracked devices, so
    the result remains an explicit estimate.  Only ``torso_lean_deg`` feeds the
    viewer's *work* similarity (the "몸통 기울기" item); it never feeds the expert
    posture similarity, which is the arm-joint engine's alone.  Joint angles are
    invariant to the CSV rig's changing global axis.
    """

    knees: dict[str, float] = {}
    for side in ("Left", "Right"):
        names = (f"{side} Thigh", f"{side} Calf", f"{side} Foot")
        if all(name in joints for name in names):
            angle = _joint_angle(*(joints[name] for name in names))
            if angle is not None and math.isfinite(angle):
                knees[side] = angle
    required = ("Pelvis", "Head", "Left Foot", "Right Foot")
    if len(knees) != 2 or not all(name in joints for name in required):
        return {
            "key": "unavailable",
            "label": "자세 판정 보류",
            "estimated": True,
            "source": "csv_skeleton_joints",
        }

    left_knee = knees["Left"]
    right_knee = knees["Right"]
    knee_angle = (left_knee + right_knee) / 2.0
    knee_disparity = abs(left_knee - right_knee)
    pelvis = joints["Pelvis"]
    head = joints["Head"]
    feet_midpoint = [
        (joints["Left Foot"][index] + joints["Right Foot"][index]) / 2.0
        for index in range(3)
    ]
    torso_lean = _joint_angle(head, pelvis, feet_midpoint)
    torso_lean = 0.0 if torso_lean is None else 180.0 - torso_lean

    if knee_angle < 75.0:
        key, label = "deep_bend", "깊게 굽힌 자세"
    elif knee_angle < 115.0:
        key, label = "lowered", "낮춘 자세"
    elif torso_lean > 35.0:
        key, label = "torso_lean", "상체를 기울인 자세"
    else:
        key, label = "standing", "서 있는 자세"
    return {
        "key": key,
        "label": label,
        "estimated": True,
        "source": "csv_skeleton_joints",
        "left_knee_angle_deg": round(left_knee, 1),
        "right_knee_angle_deg": round(right_knee, 1),
        "knee_angle_deg": round(knee_angle, 1),
        "confidence": "low" if knee_disparity > 45.0 else "normal",
        "torso_lean_deg": round(torso_lean, 1),
    }


def _smooth_posture_states(frames: list[dict[str, Any]], *, radius: int = 2) -> None:
    """Suppress isolated IK-joint posture flicker without changing motion data."""

    states = [dict(frame.get("posture") or {}) for frame in frames]
    for index, frame in enumerate(frames):
        start = max(0, index - radius)
        stop = min(len(states), index + radius + 1)
        window = [state for state in states[start:stop] if state.get("key") != "unavailable"]
        if not window:
            frame["posture"] = {**states[index], "smoothed": True}
            continue
        counts: dict[str, int] = {}
        for state in window:
            key = str(state.get("key"))
            counts[key] = counts.get(key, 0) + 1
        current_key = str(states[index].get("key"))
        winner = max(counts, key=lambda key: (counts[key], key == current_key))
        selected = min(
            (
                (abs((start + offset) - index), state)
                for offset, state in enumerate(window)
                if state.get("key") == winner
            ),
            key=lambda item: item[0],
        )[1]
        frame["posture"] = {**selected, "smoothed": True}


def _world_joints(row: dict[str, str]) -> dict[str, list[float]]:
    positions: dict[str, tuple[float, float, float]] = {}
    rotations: dict[str, tuple[float, float, float, float]] = {}
    for joint, parent in _JOINT_PARENTS:
        position_values = tuple(_number(row.get(f"{joint} Position {axis}")) for axis in "XYZ")
        rotation_values = tuple(_number(row.get(f"{joint} Rotation {axis}")) for axis in "XYZ")
        if any(value is None for value in (*position_values, *rotation_values)):
            continue
        local_position = position_values  # type: ignore[assignment]
        local_rotation = _unity_euler_quaternion(*rotation_values)  # type: ignore[arg-type]
        if parent is None:
            world_position = local_position
            world_rotation = local_rotation
        else:
            if parent not in positions:
                continue
            rotated = _rotate(rotations[parent], local_position)
            parent_position = positions[parent]
            world_position = tuple(parent_position[index] + rotated[index] for index in range(3))
            world_rotation = _multiply_quaternion(rotations[parent], local_rotation)
        positions[joint] = world_position  # type: ignore[assignment]
        rotations[joint] = world_rotation
    return {joint: [round(value, 6) for value in position] for joint, position in positions.items()}


def _open_csv(path: Path) -> TextIO:
    if path.name.lower().endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8-sig", newline="")
    return path.open("r", encoding="utf-8-sig", newline="")


@lru_cache(maxsize=64)
def _sample_csv(
    path_text: str, stamp: tuple[int, int], frame_count: int
) -> tuple[dict[str, Any], ...]:
    del stamp
    path = Path(path_text)
    with _open_csv(path) as handle:
        row_count = max(0, sum(1 for _ in handle) - 1)
    if row_count == 0:
        return ()
    count = min(frame_count, row_count)
    targets = {
        round(index * (row_count - 1) / max(1, count - 1))
        for index in range(count)
    }
    frames: list[dict[str, Any]] = []
    tool_prefix = "블라스트 건" if "blasting" in {part.casefold() for part in path.parts} else "한길숏건"
    with _open_csv(path) as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            if index not in targets:
                continue
            joints = _world_joints(row)
            tracked = {
                "head": _vector(row, "카메라"),
                "right_hand": _vector(row, tool_prefix),
                "left_hand": _vector(row, "플래쉬"),
            }
            tracked = {key: value for key, value in tracked.items() if value is not None}
            ray_point = _ray_point(row)
            if not joints and not tracked and ray_point is None:
                continue
            raw_time = _number(row.get("시간(초)"))
            frames.append({
                "t": raw_time if raw_time is not None else float(index),
                "csv_time_seconds": raw_time,
                "joints": joints,
                "tracked": tracked,
                "camera_forward": _camera_forward(row),
                "tool_forward": _rotation_forward(row, tool_prefix, (0.0, 1.0, 0.0)),
                "posture": _joint_posture_state(joints),
                "ray_point": ray_point,
                "spraying": _spraying(row.get("분사")),
                "speed_mms": _number(row.get("Speed (Real)")),
                "distance_mm": (
                    _number(row.get("Distance (Input)"))
                    if _number(row.get("Distance (Input)")) is not None
                    else _number(row.get("Distance (Real)"))
                ),
                "zone_id": str(row.get("Zone ID", "")).strip() or None,
                "map_num": (_map_zone_uv(row) or (None,))[0],
            })
    if frames:
        _smooth_posture_states(frames)
        origin = float(frames[0]["t"])
        for frame in frames:
            frame["t"] = round(float(frame["t"]) - origin, 6)
    return tuple(frames)


def _read_profile(root: Path, process: str, mode_id: int) -> dict[str, Any]:
    if process not in _PROCESS_LABELS or not 1 <= mode_id <= 10:
        raise HTTPException(status_code=404, detail="motion profile not found")
    path = root / "generated" / "profiles" / process / "learning" / f"mode_{mode_id:02d}.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError) as error:
        raise HTTPException(status_code=404, detail="motion profile not found") from error
    if not isinstance(payload.get("sessions"), list):
        raise HTTPException(status_code=422, detail="motion profile is invalid")
    return payload


def _worker_sources(root: Path) -> list[dict[str, Any]]:
    sources = [{
        "worker_id": _EXPERT_WORKER_ID,
        "display_name": _EXPERT_NAME,
        "role": "expert",
        "root": root,
    }]
    workers_root = root / "workers"
    if not workers_root.is_dir():
        return sources
    for directory in sorted(path for path in workers_root.iterdir() if path.is_dir()):
        if not (directory / "generated" / "profiles").is_dir():
            continue
        metadata: dict[str, Any] = {}
        try:
            candidate = json.loads((directory / "worker.json").read_text(encoding="utf-8"))
            if isinstance(candidate, dict):
                metadata = candidate
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            pass
        sources.append({
            "worker_id": directory.name,
            "display_name": str(metadata.get("display_name") or directory.name),
            "role": str(metadata.get("role") or "trainee"),
            "root": directory,
        })
    return sources


def _worker_source(root: Path, worker_id: str) -> dict[str, Any]:
    source = next(
        (item for item in _worker_sources(root) if item["worker_id"] == worker_id),
        None,
    )
    if source is None:
        raise HTTPException(status_code=404, detail="worker data not found")
    return source


def _representative_session(profile: dict[str, Any]) -> str:
    sessions = [item for item in profile["sessions"] if isinstance(item, dict)]
    if not sessions:
        raise HTTPException(status_code=404, detail="expert session not found")
    orders = profile.get("aggregate", {}).get("macro_orders", [])
    dominant_order = orders[0].get("order") if orders and isinstance(orders[0], dict) else None
    aggregate_zones = profile.get("aggregate", {}).get("zones", {})

    def score(session: dict[str, Any]) -> tuple[float, str]:
        analysis = session.get("analysis", {})
        penalty = 0.0 if dominant_order is None or analysis.get("macro_order") == dominant_order else 1000.0
        session_zones = analysis.get("zones", {})
        for key, reference in aggregate_zones.items():
            observed = session_zones.get(key, {}) if isinstance(session_zones, dict) else {}
            for metric in ("speed_mms", "distance_mm"):
                median = _number(reference.get(metric, {}).get("median"))
                value = _number(observed.get(metric)) if isinstance(observed, dict) else None
                if median is not None and value is not None:
                    penalty += abs(value - median) / max(abs(median), 1.0)
        return penalty, str(session.get("session_id", ""))

    return str(min(sessions, key=score)["session_id"])


def _session_path(
    root: Path, process: str, mode_id: int, profile: dict[str, Any], session_id: str
) -> Path:
    session = next(
        (item for item in profile["sessions"] if str(item.get("session_id")) == session_id),
        None,
    )
    if session is None:
        raise HTTPException(status_code=404, detail="expert session not found")
    source = session.get("sources", {}).get("csv_path")
    if not isinstance(source, str) or not source:
        raise HTTPException(status_code=422, detail="expert session source is invalid")
    mode_root = (root / "raw" / process / "learning" / f"mode_{mode_id:02d}").resolve()
    candidate = (mode_root / source).resolve()
    if not candidate.is_relative_to(mode_root) or not candidate.is_file():
        raise HTTPException(status_code=404, detail="expert session file not found")
    return candidate


def _record_height_maps(
    root: Path, process: str, mode_id: int, profile: dict[str, Any], session_id: str
) -> dict[int, Path]:
    """기록의 최종 높이맵 {면: 경로}. `Map{면}_{WorkID}_{회차}_{수행번호}_Height.png`만 쓴다."""
    parts = session_id.split("_", 2)
    heightmaps = _session_record(profile, session_id).get("heightmaps")
    if len(parts) != 3 or not isinstance(heightmaps, list):
        return {}
    name = re.compile(rf"map(\d+)_{re.escape(parts[1])}_{re.escape(parts[2])}_(\d+)_height\.png", re.I)
    mode_root = (root / "raw" / process / "learning" / f"mode_{mode_id:02d}").resolve()
    latest: dict[int, tuple[int, Path]] = {}
    for item in heightmaps:
        source = item.get("source_path") if isinstance(item, dict) else None
        match = name.fullmatch(Path(source).name) if isinstance(source, str) and source else None
        if match is None:
            continue
        try:
            candidate = (mode_root / source).resolve()
            if not candidate.is_relative_to(mode_root) or not candidate.is_file():
                continue
        except (OSError, ValueError):
            continue
        face, run = int(match[1]), int(match[2])
        # 수행번호는 정수로 견준다. 이름순이면 _9_가 _10_ 뒤로 간다.
        if face not in latest or run > latest[face][0]:
            latest[face] = (run, candidate)
    return {face: path for face, (_, path) in latest.items()}


def _target_unavailable(reason_code: str) -> dict[str, Any]:
    return {"analysis_status": "not_available", "reason_code": reason_code}


def _target_coating(
    process: str,
    mode_id: int,
    height_maps: dict[int, Path],
    zone_keys: Any,
    session: dict[str, Any],
) -> dict[str, Any]:
    """기록 최종 높이맵의 목표 대비 도막 평가(PR #8 기준). 못 만들면 사유 코드가 든 not_available."""
    if process != "spray":
        return _target_unavailable("coating_not_applicable")
    model_key = ZoneReferenceCatalog.default().resolve_model(str(mode_id))
    if model_key is None:
        return _target_unavailable("zone_reference_unavailable")
    stamps = []
    for face, path in sorted(height_maps.items()):
        try:
            stat = path.stat()
        except OSError:
            continue
        stamps.append((face, str(path), stat.st_mtime_ns, stat.st_size))
    faces = {stamp[0] for stamp in stamps}
    zones = set()
    for key in zone_keys:
        map_text, _, zone_id = str(key).partition(":")
        # 높이맵이 없는 면의 Zone은 평가하지 않는다. 0 표면으로 재면 "칠한 곳 없음"이라는 틀린 진술이 된다.
        if map_text.isdigit() and int(map_text) in faces and zone_id:
            zones.add((int(map_text), zone_id,
                       _number(_zone_analysis_value(session, str(key), "speed_mms")),
                       _number(_zone_analysis_value(session, str(key), "distance_mm"))))
    if not zones:
        return _target_unavailable("height_map_unavailable")
    ordered = tuple(sorted(zones, key=lambda item: (item[0], len(item[1]), item[1])))
    return copy.deepcopy(_target_coating_cached(model_key, None, tuple(stamps), ordered))


@lru_cache(maxsize=8)
def _target_coating_cached(
    model_key: str,
    target_um: float | None,
    stamps: tuple[tuple[int, str, int, int], ...],
    zones: tuple[tuple[int, str, float | None, float | None], ...],
) -> dict[str, Any]:
    # 표면(2048² float64 = 32 MB)은 캐시하지 않고, 면 하나씩 읽어 평가한 뒤 놓는다.
    # shortcut: 열쇠가 mtime·크기라 같은 mtime·크기로 내용만 바뀐 PNG는 옛 결과를 낸다, 그런 갱신이 생기면 sha256을 열쇠에 넣는다.
    catalog = ZoneReferenceCatalog.default()
    rows, assessments, read_any = [], [], False
    for face, path_text, _, _ in stamps:
        try:
            surface = coating_assessment.painted_surface_from_png(Path(path_text).read_bytes())
        except (OSError, ValueError):
            continue
        read_any = True
        for map_num, zone_id, speed, distance in zones:
            if map_num != face:
                continue
            try:
                image = catalog.zone_mask(model_key, map_num, zone_id)  # 라벨맵 원래 크기. 리샘플링하지 않는다.
            except (OSError, ValueError):
                continue
            if image is None:
                continue
            mask = np.asarray(image) > 0
            if mask.shape != surface.shape:
                # PR #8 규칙: 크기가 다른 높이맵·마스크는 맞추지 않고 거부한다.
                return _target_unavailable("height_map_size_mismatch")
            assessment = coating_assessment.assess_zone(
                surface, mask, target_um,
                speed_mm_s=() if speed is None else (speed,), distance_mm=() if distance is None else (distance,))
            if assessment is not None:
                assessments.append(assessment)
                rows.append(coating_assessment.zone_row(map_num, zone_id, assessment))
        del surface
    if not read_any:
        return _target_unavailable("height_map_unavailable")
    if not rows:
        return _target_unavailable("zone_reference_unavailable")
    target, basis = coating_assessment.resolve_target(target_um)
    summary = coating_assessment.session_summary(assessments)
    return {
        "analysis_status": "available", "reason_code": None,
        "target_thickness_um": target, "target_basis": basis,
        "basis_label": coating_assessment.basis_label(target, basis),
        "criterion_version": coating_assessment.ASSESSMENT_VERSION,
        "normal_band_um": coating_assessment.normal_band_um(target),
        "band_edges_um": coating_assessment.band_edges_um(target),
        "band_labels": list(coating_assessment.BAND_LABELS),
        "grade_labels": dict(coating_assessment.GRADE_SHORT_LABELS),
        "zones": rows,
        "summary": {
            "zone_count": summary["zone_count"], "grade_counts": summary["grade_counts"],
            "painted_mean_um": None if summary["painted_mean_um"] is None else round(summary["painted_mean_um"], 1),
            "unpainted_fraction": None if summary["unpainted_fraction"] is None else round(summary["unpainted_fraction"], 4),
            "six_band_counts": summary["six_band_counts"],
        },
    }


def _frames(path: Path, frame_count: int) -> list[dict[str, Any]]:
    stat = path.stat()
    return [dict(frame) for frame in _sample_csv(str(path), (stat.st_mtime_ns, stat.st_size), frame_count)]


def _canonicalize_zone_items(
    items: list[dict[str, Any]], *, process: str, mode_id: int,
    legacy_expert_zone_ids: bool = True,
) -> list[dict[str, Any]]:
    for item in items:
        if "zone_id" in item:
            item["zone_id"] = canonical_expert_zone_id(
                process, "learning" if legacy_expert_zone_ids else "source", mode_id, item.get("zone_id")
            )
    return items


@lru_cache(maxsize=64)
def _paint_cells(
    path_text: str, stamp: tuple[int, int], cell_size: float
) -> tuple[tuple[Any, ...], ...]:
    del stamp
    cells: dict[tuple[int, int, int], dict[str, Any]] = {}
    previous: list[float] | None = None
    previous_zone: str | None = None
    with _open_csv(Path(path_text)) as handle:
        for row in csv.DictReader(handle):
            point = _ray_point(row)
            if not _spraying(row.get("분사")) or point is None:
                previous = None
                previous_zone = None
                continue
            zone = str(row.get("Zone ID", "")).strip() or None
            key = tuple(round(value / cell_size) for value in point)
            cell = cells.setdefault(key, {
                "sum": [0.0, 0.0, 0.0], "count": 0, "direction": [0.0, 0.0, 0.0],
                "moves": 0, "horizontal": 0, "vertical": 0, "diagonal": 0,
            })
            for axis in range(3):
                cell["sum"][axis] += point[axis]
            cell["count"] += 1
            if previous is not None and zone == previous_zone:
                delta = [point[axis] - previous[axis] for axis in range(3)]
                distance = math.sqrt(sum(value * value for value in delta))
                if 0.0005 <= distance <= 0.25:
                    direction = [value / distance for value in delta]
                    dominant = max(range(3), key=lambda axis: abs(direction[axis]))
                    if direction[dominant] < 0:
                        direction = [-value for value in direction]
                    for axis in range(3):
                        cell["direction"][axis] += direction[axis]
                    horizontal = math.sqrt(delta[0] ** 2 + delta[2] ** 2)
                    vertical = abs(delta[1])
                    ratio = vertical / max(distance, 1e-9)
                    category = "vertical" if ratio >= 0.7 else "horizontal" if ratio <= 0.3 else "diagonal"
                    cell[category] += 1
                    cell["moves"] += 1
            previous, previous_zone = point, zone
    return tuple(
        (
            key,
            tuple(round(value, 6) for value in value["sum"]), value["count"],
            tuple(round(item, 6) for item in value["direction"]), value["moves"],
            value["horizontal"], value["vertical"], value["diagonal"],
        )
        for key, value in cells.items()
    )


def _reference_paint_statistics(paths: list[Path], *, cell_size: float = 0.01) -> dict[str, Any]:
    combined: dict[tuple[int, int, int], dict[str, Any]] = {}
    totals = {"horizontal": 0, "vertical": 0, "diagonal": 0}
    raw_point_count = 0
    for path in paths:
        stat = path.stat()
        for key, point_sum, count, direction_sum, moves, horizontal, vertical, diagonal in _paint_cells(
            str(path), (stat.st_mtime_ns, stat.st_size), cell_size
        ):
            cell = combined.setdefault(key, {"sum": [0.0] * 3, "count": 0, "direction": [0.0] * 3, "moves": 0})
            for axis in range(3):
                cell["sum"][axis] += point_sum[axis]
                cell["direction"][axis] += direction_sum[axis]
            cell["count"] += count
            cell["moves"] += moves
            raw_point_count += count
            totals["horizontal"] += horizontal
            totals["vertical"] += vertical
            totals["diagonal"] += diagonal
    entries = sorted(combined.items())
    if len(entries) > 4000:
        dense = sorted(entries, key=lambda pair: pair[1]["count"], reverse=True)[:2000]
        dense_keys = {key for key, _ in dense}
        remaining = [pair for pair in entries if pair[0] not in dense_keys]
        step = max(1, math.ceil(len(remaining) / 2000))
        visible = [item for _, item in dense] + [item for _, item in remaining[::step]][:2000]
    else:
        visible = [item for _, item in entries]
    max_count = max((item["count"] for item in visible), default=1)
    points = [{
        "point": [round(value / item["count"], 6) for value in item["sum"]],
        "frequency": round(item["count"] / max_count, 4),
    } for item in visible]
    direction_candidates = sorted(
        (item for item in combined.values() if item["moves"] >= 1),
        key=lambda item: item["moves"], reverse=True,
    )[:160]
    max_moves = max((item["moves"] for item in direction_candidates), default=1)
    field = []
    for item in direction_candidates:
        magnitude = math.sqrt(sum(value * value for value in item["direction"]))
        if magnitude <= 1e-9:
            continue
        field.append({
            "point": [round(value / item["count"], 6) for value in item["sum"]],
            "direction": [round(value / magnitude, 6) for value in item["direction"]],
            "frequency": round(item["moves"] / max_moves, 4),
        })
    move_total = sum(totals.values())
    distribution = {
        key: round(value / move_total, 4) if move_total else 0.0
        for key, value in totals.items()
    }
    return {
        "reference_surface": {
            "points": points, "raw_point_count": raw_point_count,
            "cell_size_m": cell_size,
        },
        "direction_field": field,
        "direction_distribution": distribution,
    }


def _sample_polyline(points: list[list[float]], limit: int = 36) -> list[list[float]]:
    if len(points) <= limit:
        return [[round(value, 6) for value in point] for point in points]
    indexes = sorted({round(index * (len(points) - 1) / (limit - 1)) for index in range(limit)})
    return [[round(value, 6) for value in points[index]] for index in indexes]


@lru_cache(maxsize=64)
def _zone_strokes(path_text: str, stamp: tuple[int, int]) -> tuple[dict[str, Any], ...]:
    del stamp
    strokes: list[dict[str, Any]] = []
    points: list[list[float]] = []
    identity: tuple[str, str] | None = None
    zone_orders: dict[tuple[str, str], int] = {}

    def flush() -> None:
        nonlocal points, identity
        if identity is None or len(points) < 2:
            points = []
            identity = None
            return
        path_length = sum(math.dist(left, right) for left, right in zip(points, points[1:]))
        if path_length >= 0.03:
            zone_orders[identity] = zone_orders.get(identity, 0) + 1
            centroid = [statistics.fmean(point[axis] for point in points) for axis in range(3)]
            vector = [points[-1][axis] - points[0][axis] for axis in range(3)]
            magnitude = math.sqrt(sum(value * value for value in vector))
            direction = [value / magnitude for value in vector] if magnitude > 1e-9 else [1.0, 0.0, 0.0]
            strokes.append({
                "map_num": identity[0], "zone_id": identity[1],
                "order": zone_orders[identity], "points": _sample_polyline(points),
                "centroid": centroid, "direction": direction, "path_length": path_length,
            })
        points = []
        identity = None

    with _open_csv(Path(path_text)) as handle:
        for row in csv.DictReader(handle):
            point = _ray_point(row)
            map_zone = _map_zone_uv(row)
            current = (map_zone[0], map_zone[1]) if map_zone else None
            if not _spraying(row.get("분사")) or point is None or current is None:
                flush()
                continue
            if identity is not None and current != identity:
                flush()
            identity = current
            if not points or math.dist(points[-1], point) >= 0.005:
                points.append(point)
            else:
                points[-1] = point
    flush()
    return tuple(strokes)


def _zone_flow_guides(paths: list[Path], *, max_strokes: int = 8) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    sessions_by_zone: dict[tuple[str, str], set[int]] = {}
    for session_index, path in enumerate(paths):
        stat = path.stat()
        for stroke in _zone_strokes(str(path), (stat.st_mtime_ns, stat.st_size)):
            key = (stroke["map_num"], stroke["zone_id"])
            item = dict(stroke)
            item["session_index"] = session_index
            grouped.setdefault(key, []).append(item)
            sessions_by_zone.setdefault(key, set()).add(session_index)

    guides: list[dict[str, Any]] = []
    for key, strokes in grouped.items():
        extent = math.sqrt(sum(
            (max(stroke["centroid"][axis] for stroke in strokes) - min(stroke["centroid"][axis] for stroke in strokes)) ** 2
            for axis in range(3)
        ))
        threshold = max(0.12, extent * 0.12)
        clusters: list[dict[str, Any]] = []
        for stroke in sorted(strokes, key=lambda item: item["path_length"], reverse=True):
            matches = []
            for cluster in clusters:
                distance = math.dist(stroke["centroid"], cluster["centroid"])
                alignment = abs(sum(stroke["direction"][axis] * cluster["direction"][axis] for axis in range(3)))
                if distance <= threshold and alignment >= 0.55:
                    matches.append((distance + (1 - alignment) * threshold, cluster))
            if matches:
                cluster = min(matches, key=lambda item: item[0])[1]
                direction = list(stroke["direction"])
                if sum(direction[axis] * cluster["direction"][axis] for axis in range(3)) < 0:
                    direction = [-value for value in direction]
                cluster["items"].append(stroke)
                size = len(cluster["items"])
                cluster["centroid"] = [
                    ((size - 1) * cluster["centroid"][axis] + stroke["centroid"][axis]) / size
                    for axis in range(3)
                ]
                cluster["direction"] = [
                    ((size - 1) * cluster["direction"][axis] + direction[axis]) / size
                    for axis in range(3)
                ]
            else:
                clusters.append({"items": [stroke], "centroid": list(stroke["centroid"]), "direction": list(stroke["direction"])})

        session_total = len(sessions_by_zone[key])
        minimum_sessions = 1 if session_total == 1 else max(2, math.ceil(session_total * 0.34))
        recurring = [
            cluster for cluster in clusters
            if len({item["session_index"] for item in cluster["items"]}) >= minimum_sessions
        ]
        if not recurring:
            recurring = sorted(
                clusters,
                key=lambda cluster: len({item["session_index"] for item in cluster["items"]}),
                reverse=True,
            )[:1]
        recurring.sort(
            key=lambda cluster: (
                -len({item["session_index"] for item in cluster["items"]}),
                -sum(item["path_length"] for item in cluster["items"]),
            )
        )
        chosen = recurring[:max_strokes]
        chosen.sort(key=lambda cluster: statistics.median(item["order"] for item in cluster["items"]))
        rendered = []
        for order, cluster in enumerate(chosen, 1):
            items = cluster["items"]
            representative = min(
                items,
                key=lambda candidate: sum(
                    math.dist(candidate["centroid"], other["centroid"])
                    + abs(candidate["path_length"] - other["path_length"]) * 0.2
                    for other in items
                ),
            )
            session_count = len({item["session_index"] for item in items})
            rendered.append({
                "order": order,
                "points": representative["points"],
                "session_count": session_count,
                "confidence": round(session_count / max(1, session_total), 4),
                "median_sequence_order": round(statistics.median(item["order"] for item in items), 2),
            })
        guides.append({
            "map_num": key[0], "zone_id": key[1],
            "session_count": session_total, "stroke_count": len(rendered), "strokes": rendered,
        })
    return sorted(
        guides,
        key=lambda item: (
            item["map_num"],
            0 if item["zone_id"].replace('.', '', 1).isdigit() else 1,
            float(item["zone_id"]) if item["zone_id"].replace('.', '', 1).isdigit() else item["zone_id"],
        ),
    )


def _triangle_area(a: list[float], b: list[float], c: list[float]) -> float:
    ab = [b[axis] - a[axis] for axis in range(3)]
    ac = [c[axis] - a[axis] for axis in range(3)]
    cross = [
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
    ]
    return math.sqrt(sum(value * value for value in cross)) / 2


def _build_zone_surface(
    map_num: str, zone_id: str, samples: dict[tuple[int, int], dict[str, Any]]
) -> dict[str, Any]:
    """Build a UV-local mesh while preserving each sample's real 3-D position."""
    keys = sorted(samples)
    index_by_key = {key: index for index, key in enumerate(keys)}
    vertices = [
        [round(value / samples[key]["count"], 6) for value in samples[key]["sum"]]
        for key in keys
    ]
    candidate_triangles: list[list[int]] = []
    for u, v in keys:
        a = index_by_key[(u, v)]
        b = index_by_key.get((u + 1, v))
        c = index_by_key.get((u, v + 1))
        d = index_by_key.get((u + 1, v + 1))
        if b is not None and c is not None:
            candidate_triangles.append([a, b, c])
        if b is not None and c is not None and d is not None:
            candidate_triangles.append([b, d, c])

    edge_lengths = [
        math.dist(vertices[left], vertices[right])
        for left, right in {
            tuple(sorted((triangle[0], triangle[1])))
            for triangle in candidate_triangles
        } | {
            tuple(sorted((triangle[1], triangle[2])))
            for triangle in candidate_triangles
        } | {
            tuple(sorted((triangle[2], triangle[0])))
            for triangle in candidate_triangles
        }
    ]
    typical_edge = statistics.median(edge_lengths) if edge_lengths else 0.0
    max_edge = max(0.04, typical_edge * 2.5)
    triangles = [
        triangle for triangle in candidate_triangles
        if _triangle_area(*(vertices[index] for index in triangle)) > 1e-8
        and all(
            math.dist(vertices[triangle[start]], vertices[triangle[(start + 1) % 3]]) <= max_edge
            for start in range(3)
        )
    ]
    edge_counts: dict[tuple[int, int], int] = {}
    for triangle in triangles:
        for start in range(3):
            edge = tuple(sorted((triangle[start], triangle[(start + 1) % 3])))
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
    boundary_edges = [list(edge) for edge, count in sorted(edge_counts.items()) if count == 1]
    centroid = [
        round(statistics.fmean(point[axis] for point in vertices), 6)
        for axis in range(3)
    ] if vertices else None
    return {
        "map_num": map_num,
        "zone_id": zone_id,
        "vertices": vertices,
        "triangles": triangles,
        "boundary_edges": boundary_edges,
        "centroid": centroid,
        "observed_point_count": sum(sample["count"] for sample in samples.values()),
        "reference_uv_cell_count": len(samples),
    }


@lru_cache(maxsize=64)
def _zone_cells(
    path_text: str, stamp: tuple[int, int], uv_cell_size: float
) -> tuple[tuple[Any, ...], ...]:
    del stamp
    cells: dict[tuple[str, str, int, int], dict[str, Any]] = {}
    with _open_csv(Path(path_text)) as handle:
        for row in csv.DictReader(handle):
            point = _ray_point(row)
            identity = _map_zone_uv(row)
            if not _spraying(row.get("분사")) or point is None or identity is None:
                continue
            map_num, zone_id, u, v = identity
            key = (map_num, zone_id, round(u / uv_cell_size), round(v / uv_cell_size))
            cell = cells.setdefault(key, {"sum": [0.0, 0.0, 0.0], "count": 0})
            for axis in range(3):
                cell["sum"][axis] += point[axis]
            cell["count"] += 1
    return tuple(
        (key, tuple(round(value, 6) for value in cell["sum"]), cell["count"])
        for key, cell in cells.items()
    )


def _zone_surface_statistics(paths: list[Path], *, uv_cell_size: float = 32.0) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[tuple[int, int], dict[str, Any]]] = {}
    for path in paths:
        stat = path.stat()
        for key, point_sum, count in _zone_cells(
            str(path), (stat.st_mtime_ns, stat.st_size), uv_cell_size
        ):
            map_num, zone_id, u_cell, v_cell = key
            cell = grouped.setdefault((map_num, zone_id), {}).setdefault(
                (u_cell, v_cell), {"sum": [0.0, 0.0, 0.0], "count": 0}
            )
            for axis in range(3):
                cell["sum"][axis] += point_sum[axis]
            cell["count"] += count
    return [
        _build_zone_surface(map_num, zone_id, samples)
        for (map_num, zone_id), samples in sorted(grouped.items())
    ]


def _metric_value(value: Any) -> float | None:
    if isinstance(value, dict):
        value = value.get("median")
    return _number(value)


def _zone_playback_statistics(
    profile: dict[str, Any],
    selected_session_id: str,
    trace: list[dict[str, Any]],
    selected_direction_summary: dict[str, Any],
    expert_direction_distribution: dict[str, float],
    *,
    reference_profile: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    session = next(
        (item for item in profile.get("sessions", []) if str(item.get("session_id")) == selected_session_id),
        {},
    )
    analysis = session.get("analysis", {}) if isinstance(session, dict) else {}
    raw_order = analysis.get("macro_order", []) if isinstance(analysis, dict) else []
    if not raw_order:
        seen: set[tuple[str, str]] = set()
        raw_order = []
        for item in trace:
            identity = (str(item.get("map_num") or ""), str(item.get("zone_id") or ""))
            if not all(identity) or identity in seen:
                continue
            seen.add(identity)
            raw_order.append({"map_num": identity[0], "zone_id": identity[1]})
    zone_order = [
        {
            "map_num": str(item.get("map_num")),
            "zone_id": str(item.get("zone_id")),
            "order": index + 1,
        }
        for index, item in enumerate(raw_order)
        if item.get("map_num") is not None and item.get("zone_id") is not None
    ]

    observed: dict[str, dict[str, Any]] = {}
    previous: dict[str, Any] | None = None
    for item in trace:
        map_num, zone_id = item.get("map_num"), item.get("zone_id")
        if map_num is None or zone_id is None:
            previous = item
            continue
        key = f"{map_num}:{zone_id}"
        current = observed.setdefault(
            key,
            {"path_length": 0.0, "duration": 0.0, "points": 0, "segments": set()},
        )
        current["points"] += 1
        current["segments"].add(item.get("segment"))
        if (
            previous is not None
            and previous.get("segment") == item.get("segment")
            and previous.get("map_num") == map_num
            and previous.get("zone_id") == zone_id
        ):
            current["path_length"] += math.dist(previous["point"], item["point"])
            current["duration"] += max(0.0, float(item["t"]) - float(previous["t"]))
        previous = item

    reference_profile = reference_profile if reference_profile is not None else profile
    aggregate_zones = reference_profile.get("aggregate", {}).get("zones", {})
    selected_zones = analysis.get("zones", {}) if isinstance(analysis, dict) else {}
    all_keys = set(aggregate_zones) | set(selected_zones) | set(observed)
    statistics_output: dict[str, dict[str, Any]] = {}
    for key in sorted(all_keys):
        aggregate = aggregate_zones.get(key, {})
        selected = selected_zones.get(key, {})
        measured = observed.get(key, {})
        map_num, _, zone_id = key.partition(":")
        session_count = aggregate.get("session_count")
        if session_count is None:
            session_count = sum(
                key in (item.get("analysis", {}).get("zones", {}))
                for item in reference_profile.get("sessions", [])
            )
        direction_counts = {"horizontal": 0, "vertical": 0, "diagonal": 0}
        for item in reference_profile.get("sessions", []):
            direction = (
                item.get("analysis", {}).get("zones", {}).get(key, {}).get("expected_direction")
            )
            if direction in direction_counts:
                direction_counts[direction] += 1
        direction_total = sum(direction_counts.values())
        direction_distribution = {
            direction: round(count / direction_total, 4) if direction_total else 0.0
            for direction, count in direction_counts.items()
        }
        statistics_output[key] = {
            "map_num": str(aggregate.get("map_num", selected.get("map_num", map_num))),
            "zone_id": str(aggregate.get("zone_id", selected.get("zone_id", zone_id))),
            "expert_session_count": int(session_count or 0),
            "expected_direction": aggregate.get("expected_direction"),
            "selected_major_direction": selected.get("expected_direction"),
            "expert_direction_distribution": direction_distribution,
            "median_speed_mms": _metric_value(aggregate.get("speed_mms")),
            "median_distance_mm": _metric_value(aggregate.get("distance_mm")),
            "selected_speed_mms": _metric_value(selected.get("speed_mms")),
            "selected_distance_mm": _metric_value(selected.get("distance_mm")),
            "observed_path_length_m": round(float(measured.get("path_length", 0.0)), 3),
            "observed_duration_seconds": round(float(measured.get("duration", 0.0)), 2),
            "observed_point_count": int(measured.get("points", 0)),
            "observed_sequence_count": len(measured.get("segments", set())),
        }
    statistics_output["all"] = {
        "map_num": None,
        "zone_id": None,
        "zone_count": len(all_keys),
        "expert_session_count": len(reference_profile.get("sessions", [])),
        "selected_major_direction": selected_direction_summary.get("classification"),
        "expert_direction_distribution": expert_direction_distribution,
        "observed_path_length_m": round(
            sum(float(item.get("path_length", 0.0)) for item in observed.values()), 3
        ),
        "observed_duration_seconds": round(
            sum(float(item.get("duration", 0.0)) for item in observed.values()), 2
        ),
        "observed_point_count": sum(int(item.get("points", 0)) for item in observed.values()),
        "observed_sequence_count": sum(len(item.get("segments", set())) for item in observed.values()),
    }
    return zone_order, statistics_output


def _coating_reference(profile: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return existing cells plus a shared physical scale for the loaded profile.

    The profile builder already decodes RG16 into median_um. Do not decode or
    normalize those values again; legacy values without calibration stay relative.
    """
    output: dict[str, dict[str, Any]] = {}
    aggregate = profile.get("aggregate")
    zones = aggregate.get("zones", {}) if isinstance(aggregate, dict) else {}
    quality = profile.get("quality")
    scale_status = quality.get("heightmap_scale_status") if isinstance(quality, dict) else None
    if not isinstance(zones, dict):
        return output
    for key, zone in zones.items():
        if not isinstance(zone, dict):
            continue
        coating = zone.get("coating")
        if not isinstance(coating, dict) or coating.get("analysis_status") != "available":
            continue
        raw_cells = coating.get("cells")
        if not isinstance(raw_cells, dict):
            continue
        cells: list[list[float | int]] = []
        for cell_key, sample in raw_cells.items():
            if not isinstance(cell_key, str) or not isinstance(sample, dict):
                continue
            try:
                x_text, y_text = cell_key.split(":", 1)
                x, y = int(x_text), int(y_text)
            except (TypeError, ValueError):
                continue
            height = _number(sample.get("median_um"))
            if height is None or height <= 0:
                continue
            cells.append([x, y, round(height, 4)])
        if not cells:
            continue
        semantics = coating.get("display_semantics")
        calibrated = semantics == "physical_thickness_um" or (
            semantics is None
            and scale_status == "rg16_scale_20000"
        )
        cells.sort(key=lambda item: (item[1], item[0]))
        xs = [int(item[0]) for item in cells]
        ys = [int(item[1]) for item in cells]
        thicknesses = [float(item[2]) for item in cells]
        fallback_map, _, fallback_zone = str(key).partition(":")
        output[str(key)] = {
            "map_num": int(zone.get("map_num", fallback_map)),
            "zone_id": str(zone.get("zone_id", fallback_zone)),
            "source": "expert_final_heightmap",
            "mask_source": coating.get("mask_source", "expert_raytexuv_fallback"),
            "display_semantics": semantics or (
                "physical_thickness_um" if calibrated else "relative_height"
            ),
            "unit": "um" if calibrated else None,
            "cell_unit": "um" if calibrated else None,
            "physical_calibration": "available" if calibrated else "unavailable",
            "color_scale_um": None,
            "analysis_status": "available",
            "atlas_width": int(coating.get("atlas_width", 2048)),
            "atlas_height": int(coating.get("atlas_height", 2048)),
            "grid_size": int(coating.get("grid_size", 64)),
            "session_count": int(coating.get("session_count", 0)),
            "bounds": {
                "min_x": min(xs), "max_x": max(xs),
                "min_y": min(ys), "max_y": max(ys),
            },
            "thickness_um": {
                "minimum": round(min(thicknesses), 2),
                "median": round(statistics.median(thicknesses), 2),
                "maximum": round(max(thicknesses), 2),
            } if calibrated else None,
            "cells": cells,
        }
    physical_values = [
        float(cell[2])
        for reference in output.values() if reference["cell_unit"] == "um"
        for cell in reference["cells"]
    ]
    if physical_values and max(physical_values) > 0:
        # Zero has a stable physical meaning. The maximum is data-derived, not
        # a target thickness; all calibrated zones therefore use identical colors.
        scale = {
            "minimum": 0.0, "maximum": max(physical_values), "unit": "um",
            "scope": "loaded_profile", "basis": "maximum_cell_thickness",
            "clamp": True, "clipped_cell_count": 0,
        }
        for reference in output.values():
            if reference["cell_unit"] == "um":
                reference["color_scale_um"] = scale
    return output


def _spread(
    selected_frames: list[dict[str, Any]], all_frames: list[list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for frame_index in range(len(selected_frames)):
        progress = frame_index / max(1, len(selected_frames) - 1)
        aligned = [frames[round(progress * (len(frames) - 1))] for frames in all_frames if frames]
        joint_output: dict[str, Any] = {}
        for joint, _ in _JOINT_PARENTS:
            points = [frame["joints"].get(joint) for frame in aligned]
            points = [point for point in points if point is not None]
            if not points:
                continue
            mean = [statistics.fmean(point[axis] for point in points) for axis in range(3)]
            distances = sorted(math.dist(point, mean) for point in points)
            percentile_index = round(0.75 * (len(distances) - 1))
            joint_output[joint] = {
                "mean": [round(value, 6) for value in mean],
                "spread": round(distances[percentile_index], 6),
            }
        tracked_output: dict[str, Any] = {}
        for marker in ("head", "right_hand", "left_hand"):
            points = [frame.get("tracked", {}).get(marker) for frame in aligned]
            points = [point for point in points if point is not None]
            if not points:
                continue
            mean = [statistics.fmean(point[axis] for point in points) for axis in range(3)]
            distances = sorted(math.dist(point, mean) for point in points)
            tracked_output[marker] = {
                "mean": [round(value, 6) for value in mean],
                "spread": round(distances[round(0.75 * (len(distances) - 1))], 6),
            }
        ray_points = [frame.get("ray_point") for frame in aligned if frame.get("spraying")]
        ray_points = [point for point in ray_points if point is not None]
        ray_mean = None
        ray_spread = None
        if ray_points:
            ray_mean = [statistics.fmean(point[axis] for point in ray_points) for axis in range(3)]
            ray_distances = sorted(math.dist(point, ray_mean) for point in ray_points)
            ray_mean = [round(value, 6) for value in ray_mean]
            ray_spread = round(ray_distances[round(0.75 * (len(ray_distances) - 1))], 6)
        output.append({
            "mean": {joint: value["mean"] for joint, value in joint_output.items()},
            "spread": {joint: value["spread"] for joint, value in joint_output.items()},
            "tracked_mean": {marker: value["mean"] for marker, value in tracked_output.items()},
            "tracked_spread": {marker: value["spread"] for marker, value in tracked_output.items()},
            "ray_mean": ray_mean,
            "ray_spread": ray_spread,
        })
    return output


def _normalized_joint_pose(frame: dict[str, Any]) -> dict[str, list[float]]:
    joints = frame.get("joints") or {}
    pelvis = joints.get("Pelvis")
    head = joints.get("Head")
    if not pelvis or not head:
        return {}
    scale = math.dist(pelvis, head)
    if scale <= 1e-6:
        return {}
    return {
        name: [(float(point[axis]) - float(pelvis[axis])) / scale for axis in range(3)]
        for name, point in joints.items()
        if isinstance(point, list) and len(point) == 3
    }


def _expert_pose_evaluation(
    selected_frames: list[dict[str, Any]],
    reference_frames: list[dict[str, Any]],
    *,
    selected_session_id: str,
    reference_session_id: str,
) -> dict[str, Any]:
    """Compare an expert replay with the representative normalized joint motion."""

    distances: list[float] = []
    for index, selected_frame in enumerate(selected_frames):
        progress = index / max(1, len(selected_frames) - 1)
        reference_frame = reference_frames[round(progress * (len(reference_frames) - 1))]
        selected_pose = _normalized_joint_pose(selected_frame)
        reference_pose = _normalized_joint_pose(reference_frame)
        for joint in selected_pose.keys() & reference_pose.keys():
            distances.append(math.dist(selected_pose[joint], reference_pose[joint]))
    mean_distance = statistics.fmean(distances) if distances else 1.0
    score = round(math.exp(-2.5 * mean_distance), 3)
    if selected_session_id == reference_session_id:
        score = 1.0
    if score >= 0.9:
        message = "전문가 대표 동작과 일치합니다. 현재 자세를 유지하세요."
        predicted = score
    else:
        message = "대표 전문가 동작과 비교해 전신 관절의 위치 차이를 줄여 보세요."
        predicted = round(min(1.0, score + max(0.03, (1.0 - score) * 0.35)), 3)
    return {
        "status": "available",
        "message": message,
        "similarity_score": score,
        "predicted_score": predicted,
        "reference_session_id": reference_session_id,
        "comparison_basis": "pelvis_centered_height_normalized_skeleton",
        "corrections": [{"message": message}],
    }


_ZONE_METRIC_WEIGHTS = {
    "tool_angle": 0.20,
    "posture": 0.25,
    "paint_pattern": 0.20,
    "coating_thickness": 0.20,
    "distance": 0.15,
}


def _median_numbers(values: list[Any]) -> float | None:
    parsed = [number for value in values if (number := _number(value)) is not None]
    return statistics.median(parsed) if parsed else None


def _vector_angle_degrees(left: list[float], right: list[float]) -> float | None:
    left_size = math.sqrt(sum(value * value for value in left))
    right_size = math.sqrt(sum(value * value for value in right))
    if left_size <= 1e-9 or right_size <= 1e-9:
        return None
    cosine = sum(a * b for a, b in zip(left, right)) / (left_size * right_size)
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def _zone_frame_profiles(frames: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    previous: dict[str, list[float]] = {}
    for frame in frames:
        if not frame.get("spraying") or not frame.get("zone_id") or frame.get("map_num") is None:
            continue
        key = f"{frame['map_num']}:{frame['zone_id']}"
        group = grouped.setdefault(
            key,
            {
                "tool_angles": [], "postures": [], "posture_labels": [], "distances": [],
                "directions": {"horizontal": 0, "vertical": 0, "diagonal": 0},
            },
        )
        tool = frame.get("tool_forward")
        hand = frame.get("tracked", {}).get("right_hand")
        hit = frame.get("ray_point")
        if tool and hand and hit:
            angle = _vector_angle_degrees(tool, [hit[axis] - hand[axis] for axis in range(3)])
            if angle is not None:
                group["tool_angles"].append(angle)
        posture = _number(frame.get("posture", {}).get("torso_lean_deg"))
        if posture is not None:
            group["postures"].append(posture)
        posture_label = str(frame.get("posture", {}).get("label") or "").strip()
        if posture_label and posture_label != "자세 판정 보류":
            group["posture_labels"].append(posture_label)
        distance = _number(frame.get("distance_mm"))
        if distance is not None:
            group["distances"].append(distance)
        if hit and key in previous:
            delta = [hit[axis] - previous[key][axis] for axis in range(3)]
            size = math.sqrt(sum(value * value for value in delta))
            if size > 1e-6:
                vertical_ratio = abs(delta[1]) / size
                direction = (
                    "vertical" if vertical_ratio >= 0.7
                    else "horizontal" if vertical_ratio <= 0.3
                    else "diagonal"
                )
                group["directions"][direction] += 1
        if hit:
            previous[key] = hit

    output: dict[str, dict[str, Any]] = {}
    for key, values in grouped.items():
        directions = values["directions"]
        direction = max(directions, key=directions.get) if sum(directions.values()) else None
        posture_labels = values["posture_labels"]
        posture_label = (
            max(set(posture_labels), key=lambda label: (posture_labels.count(label), label))
            if posture_labels else None
        )
        output[key] = {
            "tool_angle_deg": _median_numbers(values["tool_angles"]),
            "posture_deg": _median_numbers(values["postures"]),
            "posture_label": posture_label,
            "distance_mm": _median_numbers(values["distances"]),
            "paint_pattern": direction,
        }
    return output


def _session_record(profile: dict[str, Any], session_id: str) -> dict[str, Any]:
    return next(
        (item for item in profile.get("sessions", []) if str(item.get("session_id")) == session_id),
        {},
    )


def _zone_analysis_value(session: dict[str, Any], key: str, metric: str) -> Any:
    zone = session.get("analysis", {}).get("zones", {}).get(key, {})
    value = zone.get(metric) if isinstance(zone, dict) else None
    if isinstance(value, dict):
        return value.get("median")
    return value


def _session_coating_median(session: dict[str, Any], key: str) -> float | None:
    coating = session.get("analysis", {}).get("coating_cells", {}).get(key, {})
    cells = coating.get("cells", {}) if isinstance(coating, dict) else {}
    values: list[Any] = []
    for sample in cells.values() if isinstance(cells, dict) else []:
        values.append(sample.get("median_um") if isinstance(sample, dict) else sample)
    return _median_numbers(values)


def _numeric_similarity(actual: float, references: list[float], *, minimum_tolerance: float) -> tuple[int, float]:
    center = statistics.median(references)
    deviations = [abs(value - center) for value in references]
    tolerance = max(minimum_tolerance, statistics.median(deviations) * 3 if deviations else 0.0)
    normalized = abs(actual - center) / max(tolerance, 1e-9)
    score = 20.0 + 80.0 * math.exp(-0.2 * normalized * normalized)
    return round(max(0.0, min(100.0, score))), center


def _metric_item(
    key: str,
    label: str,
    actual: float | None,
    references: list[float],
    *,
    unit: str,
    tolerance: float,
) -> dict[str, Any]:
    if actual is None or not references:
        return {
            "key": key, "label": label, "actual": actual, "reference": None,
            "unit": unit, "similarity_percent": None, "available": False,
        }
    score, center = _numeric_similarity(actual, references, minimum_tolerance=tolerance)
    return {
        "key": key, "label": label,
        "actual": round(actual, 1), "reference": round(center, 1),
        "unit": unit, "similarity_percent": score, "available": True,
    }


def _pattern_metric(actual: str | None, references: list[str]) -> dict[str, Any]:
    labels = {
        "horizontal": "수평형", "vertical": "수직형", "diagonal": "대각선형",
        "curved_irregular": "비정형", None: "평가 불가",
    }
    if actual is None or not references:
        return {
            "key": "paint_pattern", "label": "칠하는 방식",
            "actual": None, "reference": None, "actual_label": labels.get(actual, str(actual)),
            "reference_label": "평가 불가", "unit": "", "similarity_percent": None,
            "available": False,
        }
    counts = {value: references.count(value) for value in set(references)}
    reference = max(counts, key=lambda value: (counts[value], value))
    if actual in counts:
        score = round(85 + 15 * counts[actual] / max(counts.values()))
    else:
        score = 60
    return {
        "key": "paint_pattern", "label": "칠하는 방식",
        "actual": actual, "reference": reference,
        "actual_label": labels.get(actual, str(actual)),
        "reference_label": labels.get(reference, str(reference)),
        "unit": "", "similarity_percent": score, "available": True,
    }


def _macro_order(session: dict[str, Any]) -> list[str]:
    return [
        f"{item.get('map_num')}:{item.get('zone_id')}"
        for item in session.get("analysis", {}).get("macro_order", [])
        if item.get("map_num") is not None and item.get("zone_id") is not None
    ]


def _lcs_ratio(left: list[str], right: list[str]) -> float:
    if not left or not right:
        return 0.0
    row = [0] * (len(right) + 1)
    for left_item in left:
        previous = 0
        for index, right_item in enumerate(right, 1):
            saved = row[index]
            row[index] = previous + 1 if left_item == right_item else max(row[index], row[index - 1])
            previous = saved
    return 2.0 * row[-1] / (len(left) + len(right))


def _order_similarity(selected: list[str], references: list[list[str]]) -> int:
    usable = [order for order in references if order]
    if not selected or not usable:
        return 60
    shape_score = 60.0 + 40.0 * max(_lcs_ratio(selected, order) for order in usable)
    support: list[float] = []
    for index, zone in enumerate(selected):
        values = [order[index] for order in usable if index < len(order)]
        if not values:
            continue
        counts = {value: values.count(value) for value in set(values)}
        support.append(counts.get(zone, 0) / max(counts.values()))
    statistical_score = 60.0 + 40.0 * (statistics.fmean(support) if support else 0.0)
    return round(shape_score * 0.7 + statistical_score * 0.3)


def _weighted_score(metrics: list[dict[str, Any]], replacements: set[str] | None = None) -> int | None:
    replacements = replacements or set()
    weighted = 0.0
    total = 0.0
    for metric in metrics:
        score = metric.get("similarity_percent")
        weight = _ZONE_METRIC_WEIGHTS.get(str(metric.get("key")), 0.0)
        if score is None or weight <= 0:
            continue
        weighted += (100 if metric.get("key") in replacements else float(score)) * weight
        total += weight
    return round(weighted / total) if total else None


_POSTURE_ADVICE_LIMIT_DEG = 45.0


def _correction_message(metric: dict[str, Any]) -> str | None:
    actual = _number(metric.get("actual"))
    reference = _number(metric.get("reference"))
    if metric.get("key") == "paint_pattern":
        if metric.get("actual") == metric.get("reference"):
            return None
        return f"칠하는 방향을 전문가의 {metric.get('reference_label')} 패턴에 맞추세요."
    if actual is None or reference is None:
        return None
    difference = abs(actual - reference)
    if metric.get("key") == "tool_angle":
        action = "줄이세요" if actual > reference else "늘리세요"
        return f"도구 각도를 약 {difference:.1f}° {action}."
    if metric.get("key") == "posture":
        # 몸통은 추적 장비가 아니라 시뮬레이터 IK의 추정이다. 자세 구분이 다르면 구분으로 알리고, 큰 각도 차이는 수치로 안내하지 않는다
        # (2026-11-11 Spark: "몸통 기울기를 약 125.7° 더 세우세요").
        own, expert = metric.get("actual_label"), metric.get("reference_label")
        if own and expert and own != expert and "평가 불가" not in (own, expert):
            return f"전문가는 이 Zone에서 주로 {expert}였습니다(이 기록은 {own})."
        if difference > _POSTURE_ADVICE_LIMIT_DEG:
            return None
        action = "세우세요" if actual > reference else "기울이세요"
        return f"몸통 기울기를 약 {difference:.1f}° 더 {action}."
    if metric.get("key") == "distance":
        action = "가까이 이동하세요" if actual > reference else "멀리 이동하세요"
        return f"도구를 피도면에서 약 {difference / 10:.1f}cm {action}."
    return None


def _multi_session_zone_evaluation(
    selected_frames: list[dict[str, Any]],
    selected_profile: dict[str, Any],
    selected_session_id: str,
    reference_records: list[dict[str, Any]],
) -> dict[str, Any]:
    selected_session = _session_record(selected_profile, selected_session_id)
    selected_frame_zones = _zone_frame_profiles(selected_frames)
    reference_frame_zones = [
        _zone_frame_profiles(record["frames"]) for record in reference_records
    ]
    selected_order = _macro_order(selected_session)
    reference_orders = [_macro_order(record["session"]) for record in reference_records]
    order_score = _order_similarity(selected_order, reference_orders)
    zone_keys = set(selected_frame_zones) | set(selected_session.get("analysis", {}).get("zones", {}))
    zone_output: dict[str, dict[str, Any]] = {}
    for zone_key in sorted(zone_keys):
        selected_frame = selected_frame_zones.get(zone_key, {})
        reference_frames = [profile.get(zone_key, {}) for profile in reference_frame_zones]
        selected_pattern = (
            _zone_analysis_value(selected_session, zone_key, "expected_direction")
            or selected_frame.get("paint_pattern")
        )
        reference_patterns = [
            _zone_analysis_value(record["session"], zone_key, "expected_direction")
            or reference_frames[index].get("paint_pattern")
            for index, record in enumerate(reference_records)
        ]
        reference_patterns = [str(value) for value in reference_patterns if value]
        distance_actual = _number(selected_frame.get("distance_mm"))
        if distance_actual is None:
            distance_actual = _number(_zone_analysis_value(selected_session, zone_key, "distance_mm"))
        distance_references = []
        for index, record in enumerate(reference_records):
            reference_distance = _number(reference_frames[index].get("distance_mm"))
            if reference_distance is None:
                reference_distance = _number(
                    _zone_analysis_value(record["session"], zone_key, "distance_mm")
                )
            distance_references.append(reference_distance)
        distance_references = [value for value in distance_references if value is not None]
        coating_actual = _session_coating_median(selected_session, zone_key)
        coating_references = [
            value for record in reference_records
            if (value := _session_coating_median(record["session"], zone_key)) is not None
        ]
        posture_metric = _metric_item(
            "posture", "몸통 기울기", _number(selected_frame.get("posture_deg")),
            [value for frame in reference_frames if (value := _number(frame.get("posture_deg"))) is not None],
            unit="deg", tolerance=15.0,
        )
        reference_posture_labels = [
            str(frame.get("posture_label")) for frame in reference_frames if frame.get("posture_label")
        ]
        posture_metric["actual_label"] = selected_frame.get("posture_label") or "평가 불가"
        posture_metric["reference_label"] = (
            max(
                set(reference_posture_labels),
                key=lambda label: (reference_posture_labels.count(label), label),
            )
            if reference_posture_labels else "평가 불가"
        )
        metrics = [
            _metric_item(
                "tool_angle", "도구 각도", _number(selected_frame.get("tool_angle_deg")),
                [value for frame in reference_frames if (value := _number(frame.get("tool_angle_deg"))) is not None],
                unit="deg", tolerance=12.0,
            ),
            posture_metric,
            _pattern_metric(str(selected_pattern) if selected_pattern else None, reference_patterns),
            _metric_item(
                "coating_thickness", "도막 두께", coating_actual, coating_references,
                unit="um", tolerance=30.0,
            ),
            _metric_item(
                "distance", "이격거리", distance_actual, distance_references,
                unit="mm", tolerance=max(30.0, statistics.median(distance_references) * 0.12) if distance_references else 30.0,
            ),
        ]
        zone_score = _weighted_score(metrics)
        correctable = sorted(
            (
                metric for metric in metrics
                if metric.get("key") != "coating_thickness"
                and metric.get("similarity_percent") is not None
                and int(metric["similarity_percent"]) < 95
            ),
            key=lambda metric: metric["similarity_percent"],
        )[:3]
        correction_messages = [
            message for metric in correctable
            if (message := _correction_message(metric)) is not None
        ]
        predicted = _weighted_score(metrics, {str(metric["key"]) for metric in correctable})
        map_num, _, zone_id = zone_key.partition(":")
        zone_output[zone_key] = {
            "map_num": map_num, "zone_id": zone_id,
            "similarity_percent": zone_score,
            "predicted_similarity_percent": predicted,
            "reference_session_count": len(reference_records),
            "metrics": metrics,
            "corrections": correction_messages,
        }
    zone_scores = [
        int(zone["similarity_percent"])
        for zone in zone_output.values() if zone.get("similarity_percent") is not None
    ]
    zone_average = round(statistics.fmean(zone_scores)) if zone_scores else 60
    overall = round(zone_average * 0.8 + order_score * 0.2)
    all_corrections = [
        {"message": message}
        for zone in sorted(zone_output.values(), key=lambda item: item.get("similarity_percent") or 0)
        for message in zone.get("corrections", [])
    ][:3]
    if not all_corrections:
        all_corrections = [{"message": "전문가 정상 범위입니다. 현재 동작을 유지하세요."}]
    predicted_values = [
        int(zone["predicted_similarity_percent"])
        for zone in zone_output.values() if zone.get("predicted_similarity_percent") is not None
    ]
    predicted_zone_average = round(statistics.fmean(predicted_values)) if predicted_values else zone_average
    predicted_overall = max(overall, round(predicted_zone_average * 0.8 + order_score * 0.2))
    return {
        "status": "available",
        "message": all_corrections[0]["message"],
        "similarity_percent": overall,
        "similarity_score": round(overall / 100, 2),
        "zone_similarity_percent": zone_average,
        "order_similarity_percent": order_score,
        "predicted_similarity_percent": predicted_overall,
        "predicted_score": round(predicted_overall / 100, 2),
        "reference_session_id": "multi_session_expert_profile",
        "reference_session_count": len(reference_records),
        "comparison_basis": "multi_session_zone_normalized",
        "zones": zone_output,
        "corrections": all_corrections,
    }


def _paint_trace(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    points = [
        (frame_index, frame)
        for frame_index, frame in enumerate(frames)
        if frame.get("spraying") and frame.get("ray_point") is not None
    ]
    return [
        {
            "frame_index": frame_index,
            "t": frame["t"],
            "point": frame["ray_point"],
            "zone_id": frame.get("zone_id"),
            "progress": round(index / max(1, len(points) - 1), 6),
        }
        for index, (frame_index, frame) in enumerate(points)
    ]


@lru_cache(maxsize=64)
def _dense_paint_trace_cached(
    path_text: str,
    stamp: tuple[int, int],
    target_hz: float,
    max_points: int,
) -> tuple[tuple[Any, ...], ...]:
    del stamp
    interval = 1.0 / max(1.0, target_hz)
    output: list[tuple[Any, ...]] = []
    origin_time: float | None = None
    last_kept_time: float | None = None
    previous_identity: tuple[str | None, str | None] | None = None
    pending: tuple[Any, ...] | None = None
    segment = -1

    def flush_pending() -> None:
        nonlocal pending
        if pending is not None:
            output.append(pending)
            pending = None

    with _open_csv(Path(path_text)) as handle:
        for frame_index, row in enumerate(csv.DictReader(handle)):
            raw_time = _number(row.get("시간(초)"))
            if origin_time is None:
                origin_time = raw_time if raw_time is not None else frame_index / 120.0
            current_time = (raw_time if raw_time is not None else frame_index / 120.0) - origin_time
            point = _ray_point(row)
            if not _spraying(row.get("분사")) or point is None:
                flush_pending()
                previous_identity = None
                last_kept_time = None
                continue
            map_zone = _map_zone_uv(row)
            normalized = _normalized_row(row)
            map_num = map_zone[0] if map_zone else None
            zone_id = map_zone[1] if map_zone else str(normalized.get("zone id", "")).strip() or None
            u = map_zone[2] if map_zone else None
            v = map_zone[3] if map_zone else None
            identity = (map_num, zone_id)
            if previous_identity != identity:
                flush_pending()
                segment += 1
                last_kept_time = None
            item = (
                frame_index,
                round(max(0.0, current_time), 6),
                tuple(point),
                map_num,
                zone_id,
                u,
                v,
                segment,
            )
            if last_kept_time is None or current_time - last_kept_time >= interval - 1e-9:
                output.append(item)
                last_kept_time = current_time
                pending = None
            else:
                pending = item
            previous_identity = identity
    flush_pending()
    if len(output) > max_points:
        indexes = {
            round(index * (len(output) - 1) / max(1, max_points - 1))
            for index in range(max_points)
        }
        output = [item for index, item in enumerate(output) if index in indexes]
    return tuple(output)


def _dense_paint_trace(
    path: Path, *, target_hz: float = 30.0, max_points: int = 12_000
) -> list[dict[str, Any]]:
    stat = path.stat()
    raw = _dense_paint_trace_cached(
        str(path), (stat.st_mtime_ns, stat.st_size), target_hz, max_points
    )
    return [
        {
            "frame_index": frame_index,
            "t": time,
            "point": list(point),
            "map_num": map_num,
            "zone_id": zone_id,
            "uv": [u, v] if u is not None and v is not None else None,
            "segment": segment,
            "progress": round(index / max(1, len(raw) - 1), 6),
        }
        for index, (frame_index, time, point, map_num, zone_id, u, v, segment) in enumerate(raw)
    ]


def _direction_summary(trace: list[dict[str, Any]]) -> dict[str, Any]:
    points = [item["point"] for item in trace]
    if len(points) < 2:
        return {"classification": "unavailable", "label": _DIRECTION_LABELS["unavailable"], "arrow_count": 0}
    pairs = [
        (left["point"], right["point"])
        for left, right in zip(trace, trace[1:])
        if left.get("segment", 0) == right.get("segment", 0)
    ]
    if not pairs:
        return {"classification": "unavailable", "label": _DIRECTION_LABELS["unavailable"], "arrow_count": 0}
    deltas = [[b[axis] - a[axis] for axis in range(3)] for a, b in pairs]
    movement = [sum(abs(delta[axis]) for delta in deltas) for axis in range(3)]
    total = sum(movement)
    path_length = sum(math.dist(a, b) for a, b in pairs)
    vertical_ratio = movement[1] / total if total else 0.0
    if total == 0:
        classification = "unavailable"
    elif vertical_ratio >= 0.7:
        classification = "vertical"
    elif vertical_ratio <= 0.3:
        classification = "horizontal"
    else:
        classification = "diagonal"
    vector = [points[-1][axis] - points[0][axis] for axis in range(3)]
    magnitude = math.sqrt(sum(value * value for value in vector))
    normalized = [round(value / magnitude, 6) for value in vector] if magnitude else [0.0, 0.0, 0.0]
    return {
        "classification": classification,
        "label": _DIRECTION_LABELS[classification],
        "arrow_count": min(12, max(1, len(points) // 12)),
        "direction_vector": normalized,
        "path_length_m": round(path_length, 3),
    }


_VIEWER_HTML = r"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>작업 자세 비교 평가</title><link rel="stylesheet" href="/motion-viewer/assets/motion_viewer_pose.css"><style>
:root{--bg:#07101f;--stage-top:#2e3646;--stage:#1c2230;--stage-edge:rgba(176,196,226,.28);--panel:#101b2e;--panel2:#16243a;--ink:#eef5ff;--muted:#91a0b8;--line:#293b56;--cyan:#22d3ee;--green:#34d399}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:Inter,Pretendard,"Noto Sans KR",system-ui,sans-serif}
.shell{height:100dvh;display:grid;grid-template-rows:auto minmax(0,1fr);overflow:hidden}.top{display:flex;justify-content:space-between;align-items:center;padding:16px 22px;border-bottom:1px solid var(--line);background:#0a1424}
.title{font-weight:800;font-size:18px}.title small{display:block;color:var(--muted);font-size:11px;font-weight:500;margin-top:4px}.top a{color:#dce8ff;text-decoration:none;font-size:13px;margin-left:14px}
.workspace{display:grid;grid-template-columns:275px minmax(420px,1fr) 360px;min-height:0;overflow:hidden}.side{min-height:0;padding:19px;border-right:1px solid var(--line);background:#0c1728;overflow:auto}.right{min-width:0;border-right:0;border-left:1px solid var(--line)}
label{display:block;color:var(--muted);font-size:12px;margin:16px 0 7px}.select,button,input[type=range]{width:100%}.select,button{border:1px solid var(--line);border-radius:8px;background:var(--panel2);color:var(--ink);padding:10px 11px}
button{cursor:pointer;font-weight:700}.primary{background:#3157d5;border-color:#3157d5}.controls{display:grid;grid-template-columns:1fr 1fr;gap:8px}.stage{position:relative;height:100%;min-height:0;display:grid;grid-template-rows:minmax(120px,1fr) minmax(132px,auto);overflow:hidden;background:#07101f}.stage-viewport{position:relative;min-height:0;overflow:hidden;background:linear-gradient(180deg,var(--stage-top),var(--stage))}.stage-viewport>canvas{width:100%;height:100%;display:block;min-height:0;cursor:grab;touch-action:none}.playback-dock{position:relative;z-index:14;min-height:132px;padding:11px 16px 14px;border-top:1px solid var(--line);background:#0a1424}.playback-status{display:flex;justify-content:space-between;gap:12px;margin-bottom:7px;color:#cfe1f5;font-size:12px}.playback-live{display:flex;align-items:center;gap:8px;min-width:0}.playback-live strong{min-width:0}.playback-zone-badge{display:inline-flex;align-items:center;min-height:22px;padding:3px 9px;border:1px solid currentColor;border-radius:999px;background:#172033;color:#cfe1f5;font-size:10px;font-weight:900;white-space:nowrap;box-shadow:0 0 10px rgba(0,0,0,.25)}.playback-zone-badge[hidden]{display:none}.zone-timeline{position:relative;height:22px;margin-bottom:5px;border:1px solid #344862;border-radius:6px;overflow:hidden;background:#172033}.zone-timeline-segment{position:absolute;inset-block:0;border-right:1px solid rgba(7,16,31,.5);display:grid;place-items:center;color:#07101f;font-size:9px;font-weight:900;overflow:hidden}.playback-dock input[type=range]{margin:0}.playback-controls{display:grid;grid-template-columns:105px 105px minmax(170px,1fr);gap:8px;align-items:center;margin-top:8px}.speed-control{display:grid;grid-template-columns:auto minmax(90px,150px);gap:8px;align-items:center;color:var(--muted);font-size:11px}.speed-control label{margin:0;color:var(--muted)}.speed-control select{width:100%;font-weight:800;color:#e5f1ff;background:#14243a;border-color:#3b5575;color-scheme:dark}.speed-control select option{color:#e5f1ff;background:#14243a}
.overlay{position:absolute;left:18px;top:16px;max-width:340px;background:rgba(7,16,31,.78);border:1px solid rgba(89,116,150,.45);border-radius:10px;padding:10px 12px;font-size:12px;color:var(--muted);pointer-events:none}.overlay strong{display:block;color:var(--ink);font-size:14px;margin-bottom:4px}.viewer-legend{display:grid;grid-template-columns:1fr;gap:3px;margin-top:9px;padding-top:8px;border-top:1px solid rgba(89,116,150,.35);font-size:10px;line-height:1.45}.viewer-legend div{white-space:nowrap}.viewer-legend .dot{width:7px;height:7px;margin-right:6px}
.realtime-hud{margin-top:9px;padding:9px 10px 8px;background:rgba(7,16,31,.42);border-top:1px solid rgba(89,116,150,.35);border-radius:7px}.realtime-hud-head{display:flex;align-items:center;justify-content:space-between;gap:10px;margin-bottom:7px}.realtime-hud-head strong{font-size:12px;margin:0}.realtime-hud-activity{font-size:10px;font-weight:800;color:#94a3b8}.realtime-hud-grid{display:grid;gap:5px}.realtime-hud-row{display:grid;grid-template-columns:62px minmax(74px,1fr) auto;align-items:center;gap:8px;font-size:10px}.realtime-hud-label{color:#94a3b8}.realtime-hud-value{color:#e5f1ff;font-variant-numeric:tabular-nums;text-align:right}.realtime-hud-state{min-width:34px;padding:1px 6px;border:1px solid currentColor;border-radius:999px;font-size:9px;font-weight:800;text-align:center}.realtime-hud-state.ok{color:#22c55e}.realtime-hud-state.warn{color:#facc15}.realtime-hud-state.alert{color:#fb7185}.realtime-hud-state.idle{color:#64748b}
/* 재생 중 값이 바뀌어도 칸 폭이 그대로이도록 칸마다 가장 긴 값(data-max)을 높이 0 줄로 깔아 둔다. 같은 글꼴로 재므로 글꼴이 늦게 실려도 맞다 */
.realtime-hud [data-max]{display:inline-flex;flex-direction:column;align-items:flex-end}.realtime-hud [data-max]::after{content:attr(data-max);height:0;overflow:hidden;visibility:hidden}.realtime-hud-head strong[data-max]{align-items:flex-start}.realtime-hud-state[data-max]{align-items:center}
.loading{position:absolute;inset:0;display:grid;place-content:center;justify-items:center;gap:13px;background:rgba(5,12,24,.72);backdrop-filter:blur(3px);z-index:5;color:#dce8ff;font-size:13px;font-weight:700}.loading[hidden]{display:none}.spinner{width:34px;height:34px;border:3px solid rgba(125,211,252,.2);border-top-color:#22d3ee;border-radius:50%;animation:spin .8s linear infinite}@keyframes spin{to{transform:rotate(360deg)}}
.card{background:var(--panel);border:1px solid var(--line);border-radius:11px;padding:14px;margin-bottom:11px}.card span{display:block;color:var(--muted);font-size:11px}.card strong{display:block;font-size:20px;margin-top:5px}.legend{font-size:12px;color:var(--muted);line-height:2}.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:7px}.help{font-size:12px;color:var(--muted);line-height:1.65;margin-top:18px}.status{min-height:18px;color:var(--cyan);font-size:12px;margin-top:12px}
.zone-order{display:flex;flex-wrap:wrap;gap:7px;margin-top:10px}.zone-chip{width:auto;padding:6px 10px;border:2px solid var(--zone-color);border-radius:999px;font-size:11px;color:var(--zone-color);background:#111f35}.zone-chip.active{box-shadow:0 0 0 2px #f5f8ff,0 0 14px var(--zone-color)}.zone-chip.selected{background:var(--zone-color);color:#07101f}.zone-detail-title{font-size:15px!important;color:var(--ink)!important;font-weight:800;margin-bottom:9px}.zone-metric{display:flex!important;justify-content:space-between;gap:10px;padding:7px 0;border-top:1px solid rgba(89,116,150,.2);font-size:11px!important}.zone-metric b{color:#e8f1ff;font-weight:700;text-align:right}.zone-note{margin-top:9px;color:#7890ad;font-size:10px;line-height:1.5}
.skeleton-card{padding:0;overflow:hidden}.skeleton-card summary{cursor:pointer;padding:14px;font-weight:800;font-size:13px;list-style:none}.skeleton-card summary::-webkit-details-marker{display:none}.skeleton-card summary:after{content:'▾';float:right;color:var(--cyan)}.skeleton-card:not([open]) summary:after{content:'▸'}.skeleton-screen{position:relative;height:300px;border-top:1px solid var(--stage-edge);border-bottom:1px solid var(--stage-edge);background:var(--stage)}#skeletonCanvas{display:block;width:100%;height:300px}.skeleton-posture,.skeleton-angle{position:absolute;top:9px;padding:4px 7px;border-radius:999px;background:rgba(7,16,31,.78);color:#a9c2df;font-size:10px}.skeleton-posture{left:10px;color:#dff7ff;font-weight:800}.skeleton-angle{right:10px}.skeleton-controls{display:grid;grid-template-columns:1fr 1fr 1fr;gap:7px;padding:10px}.skeleton-turn{font-size:20px;padding:7px;text-shadow:0 2px 5px rgba(34,211,238,.45)}.skeleton-reset{font-size:11px}.skeleton-note{padding:0 12px 12px;color:#7890ad;font-size:10px;line-height:1.5}
.toggle{display:flex;align-items:center;gap:8px;margin-top:14px;color:var(--muted);font-size:12px}.toggle input{width:auto}.badge{display:inline-flex;padding:5px 9px;border:1px solid #3157d5;border-radius:999px;color:#b9caff;font-size:11px;margin-top:12px}
.range-scale{position:relative;height:18px;color:#7186a3;font-size:10px}.range-scale span{position:absolute;top:1px;transform:translateX(-50%)}.range-scale span:first-child{left:0;transform:none}.range-scale .normal{left:27.27%;color:#dce8ff;font-weight:800}.range-scale span:last-child{right:0;transform:none}
@media(max-width:1200px){.shell{height:auto;min-height:100dvh;overflow:visible}.workspace{grid-template-columns:240px minmax(0,1fr);overflow:visible}.stage{height:calc(100dvh - 67px);min-height:360px}.right{display:block;grid-column:1/-1;border-left:0;border-top:1px solid var(--line)}}@media(max-width:720px){.top{height:86px;min-height:86px;padding:12px 16px;overflow:hidden}.workspace{display:flex;flex-direction:column}.stage{order:-1;height:calc(100dvh - 86px);min-height:360px}.side{border:0}.right{display:block}.playback-controls{grid-template-columns:1fr 1fr}.speed-control{grid-column:1/-1}}
</style><style>
.shell{grid-template-rows:auto auto minmax(0,1fr)}
.selection-panel{padding:10px 18px 12px;border-bottom:1px solid var(--line);background:#0c1728}
.selection-panel h2{margin:0 0 7px;color:#91a0b8;font-size:11px}
.selection-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}
.selection-grid label{margin:0 0 5px}.selection-grid .select{min-width:0}
.workspace{grid-template-columns:368px minmax(420px,1fr) 300px}
.evaluation-side{min-height:0;padding:14px;border-right:1px solid var(--line);background:#0c1728;overflow-y:auto;overflow-x:hidden}
.evaluation-side .pose-summary{display:grid;grid-template-columns:1fr;gap:9px}
.evaluation-side .pose-summary .card{margin:0}
.evaluation-side #poseScore{font-size:46px;line-height:1.05;letter-spacing:-1px}
.viewer-tools{margin-top:12px;padding-top:10px;border-top:1px solid var(--line)}
.viewer-tools label{margin:8px 0 6px}
.right{padding:12px;display:grid;grid-template-rows:auto minmax(360px,1fr);gap:10px;overflow:auto}
.right .card{margin:0}.skeleton-card{height:auto;display:block}
.skeleton-screen{height:var(--skeleton-view-height,310px);min-height:var(--skeleton-view-height,310px);flex:none}#skeletonCanvas{height:100%}
.coating-card{height:100%;display:flex;flex-direction:column;padding:0;overflow:hidden}
.coating-header{display:flex;justify-content:space-between;align-items:center;padding:11px 12px;border-bottom:1px solid var(--line)}
.coating-header,.coating-switch,.coating-legend,.coating-note{flex:none}
.coating-header strong{font-size:13px}.coating-header span{font-size:10px;color:#91a0b8}
.coating-switch{display:flex;gap:5px;padding:8px 10px}.coating-switch button{padding:6px 8px;font-size:10px}.coating-switch button.active{background:#3157d5;border-color:#3157d5}.coating-switch button:disabled{opacity:.45;cursor:default}
.coating-screen{position:relative;min-height:100px;flex:1;background:#07101f}#coatingCanvas{display:block;width:100%;height:100%}
.coating-status{flex:none;margin:0 10px 8px;padding:6px 8px;border:1px solid rgba(90,126,171,.24);border-radius:7px;background:#0a1527;color:#dff7ff;font-size:10px;line-height:1.4}
.coating-legend{display:flex;justify-content:space-between;white-space:pre-wrap;padding:7px 11px;background:linear-gradient(90deg,#1946a0,#1cb2c1,#31cb7a,#ffda5c,#ff6848);color:#fff;font-size:9px;text-shadow:0 1px 3px #000}
.coating-note{padding:7px 10px;color:#7890ad;font-size:9px;line-height:1.45}
.hidden-data-hooks{display:none}
@media(max-width:1200px){.shell{grid-template-rows:auto auto auto}.workspace{grid-template-columns:320px minmax(0,1fr)}.right{grid-column:1/-1;grid-template-columns:1fr 1fr;grid-template-rows:auto;border-top:1px solid var(--line)}}
@media(max-width:760px){.selection-grid{grid-template-columns:repeat(2,minmax(0,1fr))}.workspace{display:flex;flex-direction:column}.evaluation-side{order:1}.stage{order:0}.right{order:2;display:grid;grid-template-columns:1fr;grid-template-rows:auto auto}.skeleton-screen{height:300px;min-height:300px}.coating-card{min-height:400px}.selection-panel{padding:9px 12px}}
</style></head><body><div class="shell">
<header class="top"><div class="title">작업 자세 비교 평가<small>도장 경로와 작업 자세를 전문가 기준으로 비교합니다.</small></div></header>
<template id="poseUiTemplate">
<div data-pose-slot="gizmo" class="camera-gizmo" aria-label="3D 카메라 기즈모"><button id="cameraHome" aria-label="전체 보기" title="전체 보기"><svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="3"/><path d="M3 7V5a2 2 0 0 1 2-2h2"/><path d="M17 3h2a2 2 0 0 1 2 2v2"/><path d="M21 17v2a2 2 0 0 1-2 2h-2"/><path d="M7 21H5a2 2 0 0 1-2-2v-2"/></svg></button><button id="cameraAxisX" title="X축 시점">X</button><button id="cameraAxisY" title="Y축 시점">Y</button><button id="cameraAxisZ" title="Z축 시점">Z</button></div>
<div data-pose-slot="summary" id="poseSummary" class="pose-summary"><div class="card pose-overall"><span>전문가 작업 유사도</span><strong id="poseScore">—%</strong><small id="poseScoreNote">Zone 순서와 Zone별 작업 유사도를 전문가 반복 수행 기준으로 비교합니다.</small></div><div id="armPoseCard" class="card"><div class="pose-zone-header"><span>전문가 자세 유사도</span><strong id="armPoseScore">—%</strong></div><small id="armPoseNote">분사마다 팔 관절 자세를 숙련자 기준과 비교합니다.</small><div id="armPoseCurrent" class="pose-arm-current" hidden><div><span id="armPoseSequence"></span><b id="armPoseSequenceScore"></b></div><p id="armPoseCorrection"></p></div></div><div id="poseZoneCard" class="card pose-zone-card"><div class="pose-zone-header"><span id="poseZoneTitle">현재 Zone 작업 유사도</span><strong id="poseZoneScore">—%</strong></div><div class="pose-meta"><span>현재 자세 상태 · <b id="posePosture">기록 준비 중</b></span><span id="poseZoneSummary">Zone 대기 중</span><span id="poseEstimateNote">CSV 조인트 기반 자세 추정</span></div><div id="poseZoneMetrics" class="pose-zone-metrics"><small>재생을 시작하면 현재 Zone의 측정값과 전문가 기준을 표시합니다.</small></div></div><div class="card pose-action"><span id="poseActionTitle">현재 Zone 작업 개선 안내</span><ol id="poseActionList"><li id="poseAction">기록을 선택하세요.</li></ol></div><div class="card pose-predicted"><span>안내 적용 시 예상</span><strong id="posePredicted">—</strong><small>실제 재측정 결과가 아닌 가상 좌표 보정 결과입니다.</small></div></div>
</template>
<section class="selection-panel"><h2>재생 데이터 선택</h2><div class="selection-grid">
<div><label for="worker">작업자</label><select id="worker" class="select"></select></div>
<div><label for="process">훈련 종류</label><select id="process" class="select"></select></div>
<div><label for="mode">학습모드</label><select id="mode" class="select"></select></div>
<div><label for="session">재생할 기록</label><select id="session" class="select"></select></div>
</div></section>
<main class="workspace"><aside class="evaluation-side"><div class="viewer-tools">
<label for="pointSize">흰색 포인트 크기 · <span id="pointSizeValue">1.5 px</span></label><input id="pointSize" type="range" min="0.5" max="5" step="0.25" value="1.5">
<label class="toggle"><input id="zoneSurface" type="checkbox" checked>Zone 영역 표시</label>
<label class="toggle"><input id="objectMesh" type="checkbox" checked>대상 구조물 표시</label>
<div id="status" class="status">데이터를 불러오는 중입니다.</div>
</div></aside>
<section class="stage"><div class="stage-viewport"><canvas id="canvas"></canvas><div class="overlay"><strong id="overlayTitle">학습모드</strong><span id="overlayFrame">프레임 준비 중</span><div id="viewerLegend" class="viewer-legend"><div><i class="dot" style="background:#f5f8ff"></i>전문가 전체 Ray Point 점군</div><div><i class="dot" style="background:#a78bfa"></i>Map·Zone별 관측 영역</div><div><i class="dot" style="background:var(--green)"></i>Zone별 전문가 표준 도장 패턴</div><div><i class="dot" style="background:#ff8a3d"></i>선택 기록의 페인트 누적 궤적</div><div><i class="dot" style="background:var(--cyan)"></i>머리·양손 실측 위치와 추정 상체</div></div><div id="realtimeHud" class="realtime-hud"><div class="realtime-hud-head"><strong id="realtimeHudZone" data-max="Zone 대기">Zone 대기</strong><span id="realtimeHudActivity" class="realtime-hud-activity" data-max="이동 중">이동 중</span></div><div class="realtime-hud-grid"><div class="realtime-hud-row"><span class="realtime-hud-label">속도</span><span id="realtimeHudSpeed" class="realtime-hud-value" data-max="9.99 m/s">—</span><span id="realtimeHudSpeedState" class="realtime-hud-state idle" data-max="진행 중">대기</span></div><div class="realtime-hud-row"><span class="realtime-hud-label">이격거리</span><span id="realtimeHudDistance" class="realtime-hud-value" data-max="9999 mm">—</span><span id="realtimeHudDistanceState" class="realtime-hud-state idle" data-max="진행 중">대기</span></div><div class="realtime-hud-row"><span class="realtime-hud-label">분사축 편차</span><span id="realtimeHudAxis" class="realtime-hud-value" data-max="99.9°">—</span><span id="realtimeHudAxisState" class="realtime-hud-state idle" data-max="진행 중">대기</span></div><div class="realtime-hud-row"><span class="realtime-hud-label">도장 방향</span><span id="realtimeHudDirection" class="realtime-hud-value" data-max="분석 중">—</span><span id="realtimeHudDirectionState" class="realtime-hud-state idle" data-max="진행 중">대기</span></div><div class="realtime-hud-row"><span class="realtime-hud-label">도포 범위</span><span id="realtimeHudCoverage" class="realtime-hud-value" data-max="100%">—</span><span id="realtimeHudCoverageState" class="realtime-hud-state idle" data-max="진행 중">대기</span></div></div></div></div><div id="loading" class="loading"><div class="spinner"></div><span>전문가 기록을 불러오는 중입니다.</span></div></div><div class="playback-dock"><div class="playback-status"><div class="playback-live"><strong id="playbackZone">현재 Zone 대기 중</strong><span id="playbackZoneBadge" class="playback-zone-badge" hidden>Zone 대기</span></div><span id="playbackTime">0.00 / 0.00초</span></div><div id="zoneTimeline" class="zone-timeline" aria-label="Zone 진행 구간"></div><input id="timeline" type="range" min="0" max="0" value="0" aria-label="진행 위치"><div class="playback-controls"><button id="play" class="primary">재생</button><button id="restart">처음으로</button><div class="speed-control"><label for="speed">재생 속도</label><select id="speed" aria-label="재생 속도"><option value="0.25">0.25배</option><option value="0.5">0.5배</option><option value="1" selected>1.0배</option><option value="1.5">1.5배</option><option value="2">2.0배</option><option value="3">3.0배</option></select></div></div></div></section>
<aside class="side right"><div class="hidden-data-hooks"><strong id="currentZone">대기 중</strong><div id="zoneOrder"></div><div id="zoneDetail"></div><strong id="sessionCount">-</strong><strong id="duration">-</strong></div>
<details class="card skeleton-card" open><summary>CSV 조인트 기반 전신 스켈레톤</summary><div class="skeleton-screen"><canvas id="skeletonCanvas" aria-label="CSV 조인트 기반 전신 스켈레톤"></canvas><span id="skeletonPosture" class="skeleton-posture">자세 판정 대기</span><span id="skeletonAngle" class="skeleton-angle">정면 · 0°</span></div><div class="skeleton-controls"><button id="skeletonTurnLeft" class="skeleton-turn" aria-label="스켈레톤 왼쪽 회전" title="왼쪽으로 15도 회전">↶</button><button id="skeletonReset" class="skeleton-reset">정면 보기</button><button id="skeletonTurnRight" class="skeleton-turn" aria-label="스켈레톤 오른쪽 회전" title="오른쪽으로 15도 회전">↷</button></div><div class="skeleton-note">CSV의 Pelvis·몸통·팔·다리 조인트를 그대로 연결해 표시합니다. 오른쪽 전완과 손은 건 위치 기록이 있으면 건 손잡이 쪽으로 다시 세워 그립니다. 시뮬레이터 IK 관절 데이터의 특성상 일부 자세가 부자연스럽게 보일 수 있습니다.</div></details>
<section class="card coating-card"><div class="coating-header"><strong>전문가 최종 도막 기준</strong><span id="coatingZone">Zone 대기 중</span></div><div class="coating-switch" role="group" aria-label="도막 보기와 색상 척도"><button id="coatingView2d" class="active" type="button" title="2D 분포 보기">2D</button><button id="coatingView3d" type="button" title="3D 높이 강조 보기">3D</button><button id="coatingScaleRelative" class="active" type="button" aria-pressed="true">상대 분포</button><button id="coatingScaleAbsolute" type="button" aria-pressed="false" disabled>절대 두께 (µm)</button></div><div id="coatingStatus" class="coating-status">도막 기준 준비 중</div><div class="coating-screen"><canvas id="coatingCanvas" aria-label="전문가 최종 높이맵 기반 현재 Zone 도막 분포"></canvas></div><div id="coatingLegend" class="coating-legend" aria-label="도막 색상 범례">0%　25%　50%　75%　100%</div><div id="coatingScaleNote" class="coating-note">Zone별 상대 분포 · 물리 두께 보정 확인 중</div><div id="coatingGeometryNote" class="coating-note">전문가 최종 도막 기준이며 훈련자의 진행 도막이 아닙니다.</div></section></aside>
</main></div><script>
const $=id=>document.getElementById(id); const canvas=$('canvas'),ctx=canvas.getContext('2d'),skeletonCanvas=$('skeletonCanvas'),skeletonCtx=skeletonCanvas.getContext('2d');
const ZONE_COLORS=[null,'#FF0000','#00FF00','#0000FF','#00FFFF','#FF00FF','#FFFF00','#FF8000','#FF0080','#00FF80','#80FF00','#0080FF','#8000FF','#FF8080','#80FF80','#8080FF','#800000','#008000','#000080','#80FFFF','#FF80FF','#FFFF80','#008080','#800080','#808000','#FFBF80','#FF80BF','#BFFF80','#80FFBF','#BF80FF','#80BFFF','#FF40BF','#FFBF40','#40FFBF','#BFFF40','#40BFFF','#BF40FF','#40BFBF','#BF40BF','#BFBF40'];
let catalog=null,data=null,index=0,playheadSeconds=0,playing=false,last=0,yaw=-0.55,pitch=0.1,zoom=175,panX=0,panY=0,viewCenter=null,drag=null,loadGeneration=0,skeletonYaw=0;
async function json(url){const r=await fetch(url);if(!r.ok)throw new Error((await r.json()).detail||'데이터를 읽을 수 없습니다.');return r.json()}
function option(value,text){const o=document.createElement('option');o.value=value;o.textContent=text;return o}
function currentWorker(){return catalog.workers.find(worker=>worker.worker_id===$('worker').value)}
function currentProcess(){return currentWorker().processes.find(p=>p.process===$('process').value)}
function currentMode(){return currentProcess().modes.find(m=>String(m.mode_id)===$('mode').value)}
function fillProcesses(){const worker=currentWorker();$('process').replaceChildren(...worker.processes.map(p=>option(p.process,p.label)));fillModes()}
function fillModes(){const p=currentProcess();$('mode').replaceChildren(...p.modes.map(m=>option(m.mode_id,`학습모드 ${m.mode_id} · ${m.session_count}회`)));const four=p.modes.find(m=>m.mode_id===4);if(four)$('mode').value='4';fillSessions()}
function fillSessions(){const worker=currentWorker(),m=currentMode();$('session').replaceChildren(option('representative',`${worker.display_name} · 전체 기록 종합 기준`),...m.sessions.map(s=>option(s,`${worker.display_name} · ${s}`)));loadMotion()}
function zoneKey(mapNum,zoneId){return mapNum!=null&&Number(zoneId)>0?`${mapNum}:${zoneId}`:null}
function metricRow(label,value){const row=document.createElement('span');row.className='zone-metric';const name=document.createElement('span'),metric=document.createElement('b');name.textContent=label;metric.textContent=value;row.append(name,metric);return row}
function directionName(value){return({horizontal:'수평',vertical:'수직',diagonal:'대각',curved_irregular:'곡선·불규칙'})[value]||'자료 없음'}
function expertDirectionMix(stats){const mix=stats.expert_direction_distribution||{};return`수평 ${((mix.horizontal||0)*100).toFixed(0)}% · 수직 ${((mix.vertical||0)*100).toFixed(0)}% · 대각 ${((mix.diagonal||0)*100).toFixed(0)}%`}
function zoneColor(zoneId){const number=Number(zoneId);return Number.isInteger(number)&&ZONE_COLORS[number]?ZONE_COLORS[number]:'#94A3B8'}
function zoneLabel(zoneId){const number=Number(zoneId);return Number.isInteger(number)&&number>=1&&number<=39?String(number).padStart(2,'0'):String(zoneId)}
function colorWithAlpha(hex,alpha){const value=hex.replace('#','');const number=Number.parseInt(value,16);return`rgba(${number>>16},${number>>8&255},${number&255},${alpha})`}
function zoneVisualColor(key){if(key==='all')return'#67e8f9';return zoneColor(key.split(':')[1])}
let displayedZoneKey=null,selectedDetailKey='all',zoneDetailMode='auto';
let referencePreview=false;
function showZoneDetail(key,manual=false){const detail=$('zoneDetail'),stats=(data&&data.zone_statistics||{})[key],isAll=key==='all',color=zoneVisualColor(key);if(manual)zoneDetailMode='manual';selectedDetailKey=key;detail.style.borderColor=color;detail.replaceChildren();for(const chip of $('zoneOrder').querySelectorAll('.zone-chip'))chip.classList.toggle('selected',chip.dataset.zoneKey===key);const title=document.createElement('span');title.className='zone-detail-title';title.textContent=isAll?'전체 Zone 통계':stats?`Zone ${zoneLabel(stats.zone_id)} 기준 통계`:'Zone 통계';detail.append(title);if(!stats){const empty=document.createElement('span');empty.textContent='해당 Zone의 통계가 없습니다.';detail.append(empty);return}const speed=stats.median_speed_mms!=null?`${(stats.median_speed_mms/1000).toFixed(2)} m/s`:'자료 없음',distance=stats.median_distance_mm!=null?`${Math.round(stats.median_distance_mm)} mm`:'자료 없음';if(isAll)detail.append(metricRow('포함 Zone',`${stats.zone_count}개`));else detail.append(metricRow('전문가 중앙 속도',speed),metricRow('전문가 중앙 이격거리',distance),metricRow('전문가 기준 방향',directionName(stats.expected_direction)));detail.append(metricRow('선택 기록의 주요 도장 방향',directionName(stats.selected_major_direction)),metricRow('전문가 방향 비중',expertDirectionMix(stats)),metricRow('선택 기록 분사 시간',`${stats.observed_duration_seconds.toFixed(1)}초`),metricRow('선택 기록 도포 경로',`${stats.observed_path_length_m.toFixed(2)} m`),metricRow('선택 기록 시퀀스',`${stats.observed_sequence_count}회`),metricRow('전문가 관측 수행',`${stats.expert_session_count}회`));const note=document.createElement('div');note.className='zone-note';note.textContent=isAll?'모든 Zone의 선택 기록 합계와 전문가 전체 방향 비중입니다.':'속도·이격거리는 전문가 반복 수행의 중앙값이며, 시간·경로·시퀀스는 현재 선택 기록 기준입니다.';detail.append(note)}
function showReferencePreview(key){playing=false;last=0;index=0;playheadSeconds=0;referencePreview=true;$('timeline').value=0;$('play').textContent='재생';showZoneDetail(key,true);$('currentZone').textContent=key==='all'?'전체 전문가 표준 패턴':`Zone ${zoneLabel(key.split(':')[1])} 전문가 표준 패턴`;draw()}
function startPlayback(){const fromPreview=referencePreview;referencePreview=false;selectedDetailKey='all';zoneDetailMode='auto';displayedZoneKey=null;if(fromPreview){index=0;playheadSeconds=0;$('timeline').value=0}showZoneDetail('all');playing=true;last=0;$('play').textContent='일시정지';draw()}
function renderZoneOrder(){const host=$('zoneOrder');host.replaceChildren();const allButton=document.createElement('button');allButton.className='zone-chip';allButton.dataset.zoneKey='all';allButton.textContent='전체';allButton.style.setProperty('--zone-color',zoneVisualColor('all'));allButton.onclick=()=>showReferencePreview('all');host.append(allButton);for(const item of data.zone_order||[]){const button=document.createElement('button'),key=zoneKey(item.map_num,item.zone_id);button.className='zone-chip';button.dataset.zoneKey=key;button.textContent=`${item.order}. Zone ${zoneLabel(item.zone_id)}`;button.style.setProperty('--zone-color',zoneVisualColor(key));button.onclick=()=>showReferencePreview(key);host.append(button)}showZoneDetail('all')}
function updateZonePanel(frame){const key=zoneKey(frame.map_num,frame.zone_id);$('currentZone').textContent=key?`Zone ${zoneLabel(frame.zone_id)}${frame.spraying?' · 분사 중':' · 조준/이동'}`:'이동/대기';for(const chip of $('zoneOrder').querySelectorAll('.zone-chip'))chip.classList.toggle('active',chip.dataset.zoneKey===key);if(zoneDetailMode==='auto'&&key&&key!==displayedZoneKey){showZoneDetail(key);displayedZoneKey=key}}
function setLoading(active){$('loading').hidden=!active;canvas.setAttribute('aria-busy',active?'true':'false');for(const id of ['process','mode','session'])$(id).disabled=active}
function averageCameraForward(payload){const vectors=(payload&&payload.frames||[]).map(frame=>frame.camera_forward).filter(vector=>Array.isArray(vector)&&vector.length===3);if(!vectors.length)return null;return unit(vectors.reduce((sum,vector)=>add(sum,vector),[0,0,0]),[0,0,1])}
function resetViewToCamera(payload=data){const first=payload&&payload.frames&&payload.frames[0],points=payload&&payload.reference_surface&&payload.reference_surface.points||[],fallback=points.length?points.reduce((sum,item)=>add(sum,item.point),[0,0,0]).map(value=>value/points.length):[0,0,0],forward=averageCameraForward(payload);viewCenter=first&&((first.tracked||{}).head||(first.joints||{}).Head||(first.joints||{}).Pelvis)||fallback;panX=0;panY=0;if(forward){yaw=Math.atan2(forward[0],forward[2]);pitch=Math.asin(Math.max(-.95,Math.min(.95,forward[1])))}else{yaw=-.55;pitch=.1}zoom=175}
async function loadMotion(){const generation=++loadGeneration;playing=false;last=0;displayedZoneKey=null;selectedDetailKey='all';zoneDetailMode='auto';referencePreview=false;$('play').textContent='재생';$('status').textContent='모션과 통계 범위를 계산하는 중입니다.';setLoading(true);try{const q=new URLSearchParams({worker_id:$('worker').value,process:$('process').value,mode_id:$('mode').value,session_id:$('session').value});const incoming=await json('/motion-viewer/api/motion?'+q);if(generation!==loadGeneration)return;data=incoming;index=0;playheadSeconds=0;resetViewToCamera(incoming);$('timeline').max=Math.max(0,data.frames.length-1);$('timeline').value=0;$('sessionCount').textContent=`${data.sampled_session_count}회`;$('duration').textContent=`${data.duration_seconds.toFixed(1)}초`;$('overlayTitle').textContent=`${data.worker.display_name} · ${data.process_label} · 학습모드 ${data.mode_id}`;renderZoneOrder();dispatchEvent(new CustomEvent('motion-viewer-data',{detail:incoming}));$('status').textContent=`${data.selected_session_id} 준비 완료`;draw()}catch(e){if(generation!==loadGeneration)return;data=null;$('status').textContent=e.message;draw()}finally{if(generation===loadGeneration)setLoading(false)}}
function resize(){const dpr=devicePixelRatio||1,rect=canvas.getBoundingClientRect(),skeletonRect=skeletonCanvas.getBoundingClientRect();canvas.width=Math.round(rect.width*dpr);canvas.height=Math.round(rect.height*dpr);ctx.setTransform(dpr,0,0,dpr,0,0);if(skeletonRect.width){skeletonCanvas.width=Math.round(skeletonRect.width*dpr);skeletonCanvas.height=Math.round(skeletonRect.height*dpr);skeletonCtx.setTransform(dpr,0,0,dpr,0,0)}draw()}
function transform(point,center){let x=point[0]-center[0],y=point[1]-center[1],z=point[2]-center[2];let cy=Math.cos(yaw),sy=Math.sin(yaw),x1=x*cy-z*sy,z1=x*sy+z*cy;let cp=Math.cos(pitch),sp=Math.sin(pitch),y1=y*cp-z1*sp,z2=y*sp+z1*cp;return [x1,y1,z2]}
function project(point,center,w,h){const [x,y,z]=transform(point,center),pers=4.5/(4.5+z);return [w/2+panX+x*zoom*pers,h/2+panY-y*zoom*pers,pers]}
function line(a,b,color,width,center,w,h){const p=project(a,center,w,h),q=project(b,center,w,h);ctx.strokeStyle=color;ctx.lineWidth=width;ctx.lineCap='round';ctx.beginPath();ctx.moveTo(p[0],p[1]);ctx.lineTo(q[0],q[1]);ctx.stroke()}
function point(p,r,color,center,w,h){const q=project(p,center,w,h);ctx.fillStyle=color;ctx.beginPath();ctx.arc(q[0],q[1],r*q[2],0,Math.PI*2);ctx.fill()}
function arrow(a,b,color,center,w,h){const p=project(a,center,w,h),q=project(b,center,w,h),angle=Math.atan2(q[1]-p[1],q[0]-p[0]);ctx.strokeStyle=color;ctx.fillStyle=color;ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(p[0],p[1]);ctx.lineTo(q[0],q[1]);ctx.stroke();ctx.beginPath();ctx.moveTo(q[0],q[1]);ctx.lineTo(q[0]-10*Math.cos(angle-.48),q[1]-10*Math.sin(angle-.48));ctx.lineTo(q[0]-10*Math.cos(angle+.48),q[1]-10*Math.sin(angle+.48));ctx.closePath();ctx.fill()}
function referencePointCloud(surface,center,w,h){const baseSize=Number($('pointSize').value);for(const sample of surface.points){const q=project(sample.point,center,w,h),alpha=.16+sample.frequency*.48,size=baseSize*(.8+sample.frequency*.5);ctx.fillStyle=`rgba(245,248,255,${alpha})`;ctx.fillRect(q[0]-size/2,q[1]-size/2,size,size)}}
function drawZoneSurfaces(surfaces,center,w,h){const projected=[];for(const surface of surfaces){const color=zoneColor(surface.zone_id);for(const triangle of surface.triangles){const points=triangle.map(i=>project(surface.vertices[i],center,w,h));projected.push({points,color,depth:points.reduce((sum,p)=>sum+p[2],0)/3})}}projected.sort((a,b)=>b.depth-a.depth);for(const item of projected){ctx.fillStyle=colorWithAlpha(item.color,.16);ctx.strokeStyle=colorWithAlpha(item.color,.24);ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(item.points[0][0],item.points[0][1]);for(const p of item.points.slice(1))ctx.lineTo(p[0],p[1]);ctx.closePath();ctx.fill();ctx.stroke()}for(const surface of surfaces){const color=zoneColor(surface.zone_id);ctx.strokeStyle=colorWithAlpha(color,.82);ctx.lineWidth=2;for(const edge of surface.boundary_edges){const a=project(surface.vertices[edge[0]],center,w,h),b=project(surface.vertices[edge[1]],center,w,h);ctx.beginPath();ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);ctx.stroke()}if(surface.centroid&&surface.triangles.length){const p=project(surface.centroid,center,w,h),label=`Zone ${zoneLabel(surface.zone_id)}`;ctx.font='700 11px system-ui';const width=ctx.measureText(label).width+12;ctx.fillStyle='rgba(7,16,31,.78)';ctx.fillRect(p[0]-width/2,p[1]-10,width,18);ctx.fillStyle=color;ctx.textAlign='center';ctx.fillText(label,p[0],p[1]+3);ctx.textAlign='start'}}}
// 대상 구조물(정렬이 검증된 FBX). 실험대 벽처럼 반투명 회색 면에 바깥 테두리와 25° 넘게 꺾인 모서리만 긋는다(삼각형 망은 긋지 않는다). 정점은 Zone 면과 같은 유니티 월드라 project() 그대로 쓴다 — 실험대처럼 Z 를 뒤집으면 거울상이 된다.
let outlineSource=null,outlineEdges=[];
function meshOutline(mesh){if(outlineSource===mesh)return outlineEdges;outlineSource=mesh;const key=p=>p.map(value=>Math.round(value*1e4)).join(','),edges=new Map(),crease=Math.cos(25*Math.PI/180);for(const triangle of mesh.triangles){const [a,b,c]=triangle.map(i=>mesh.vertices[i]),normal=unit(cross(sub(b,a),sub(c,a)),[0,0,0]);for(const [p,q] of [[a,b],[b,c],[c,a]]){const kp=key(p),kq=key(q);if(kp===kq)continue;const id=kp<kq?kp+'|'+kq:kq+'|'+kp,item=edges.get(id);if(item)item.normals.push(normal);else edges.set(id,{points:[p,q],normals:[normal]})}}outlineEdges=[...edges.values()].filter(edge=>edge.normals.length!==2||dot(edge.normals[0],edge.normals[1])<crease).map(edge=>edge.points);return outlineEdges}
// 면은 삼각형마다 화면 감김을 한 방향으로 맞춰 한 경로로 칠한다(nonzero) — 겹친 앞·뒤 면이 구멍이나 이음선 없이 한 번만 칠해진다.
function drawObjectMesh(mesh,status,center,w,h){if(status!=='verified'||!mesh||!mesh.available||!(mesh.triangles||[]).length)return;ctx.beginPath();for(const triangle of mesh.triangles){const p=triangle.map(i=>project(mesh.vertices[i],center,w,h)),area=(p[1][0]-p[0][0])*(p[2][1]-p[0][1])-(p[2][0]-p[0][0])*(p[1][1]-p[0][1]),corners=area<0?[p[0],p[2],p[1]]:p;ctx.moveTo(corners[0][0],corners[0][1]);ctx.lineTo(corners[1][0],corners[1][1]);ctx.lineTo(corners[2][0],corners[2][1]);ctx.closePath()}ctx.fillStyle='rgba(154,165,177,.16)';ctx.fill();ctx.strokeStyle='rgba(206,213,222,.5)';ctx.lineWidth=1.25;ctx.lineCap='round';ctx.beginPath();for(const [a,b] of meshOutline(mesh)){const p=project(a,center,w,h),q=project(b,center,w,h);ctx.moveTo(p[0],p[1]);ctx.lineTo(q[0],q[1])}ctx.stroke()}
function drawZoneFlowGuides(guides,center,w,h){for(const guide of guides||[]){const key=zoneKey(guide.map_num,guide.zone_id),focused=selectedDetailKey!=='all'&&selectedDetailKey===key,alpha=selectedDetailKey==='all'?.38:focused?.94:.055;for(const stroke of guide.strokes||[]){if(!stroke.points||stroke.points.length<2)continue;const projected=stroke.points.map(point=>project(point,center,w,h));ctx.strokeStyle=`rgba(52,211,153,${alpha})`;ctx.lineWidth=3.5+stroke.confidence*5;ctx.lineCap='round';ctx.lineJoin='round';ctx.shadowColor=`rgba(52,211,153,${Math.min(.7,alpha)})`;ctx.shadowBlur=focused?10:5;ctx.beginPath();ctx.moveTo(projected[0][0],projected[0][1]);for(const point of projected.slice(1))ctx.lineTo(point[0],point[1]);ctx.stroke();ctx.shadowBlur=0;const start=projected[0];ctx.fillStyle=`rgba(167,243,208,${Math.min(1,alpha+.18)})`;ctx.beginPath();ctx.arc(start[0],start[1],focused?5:3,0,Math.PI*2);ctx.fill();if(focused){ctx.font='800 10px system-ui';ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillStyle='#052e2b';ctx.beginPath();ctx.arc(start[0],start[1],8,0,Math.PI*2);ctx.fillStyle='#6ee7b7';ctx.fill();ctx.fillStyle='#052e2b';ctx.fillText(String(stroke.order),start[0],start[1]+.5)}}}ctx.textAlign='start';ctx.textBaseline='alphabetic'}
function add(a,b){return a.map((value,axis)=>value+b[axis])}function sub(a,b){return a.map((value,axis)=>value-b[axis])}function mul(a,value){return a.map(axis=>axis*value)}function dot(a,b){return a.reduce((sum,value,axis)=>sum+value*b[axis],0)}function cross(a,b){return[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]}function length(a){return Math.sqrt(dot(a,a))}function unit(a,fallback=[1,0,0]){const size=length(a);return size>.00001?mul(a,1/size):fallback}
function mixPoint(a,b,ratio){if(!a)return b||null;if(!b)return a;return a.map((value,axis)=>value+(b[axis]-value)*ratio)}
// 패널 0°는 정면에서 본 그림이다. 얼굴 없는 막대 인형은 마주 본 모습으로 읽히므로 몸 오른쪽을 화면 왼쪽(x=−across)에, 몸 앞(forward)을 보는 사람 쪽에 둔다.
function uprightSkeleton(joints,angleDegrees=skeletonYaw){const origin=joints.Pelvis||joints.Spine,head=joints.Head;if(!origin||!head)return{};const up=unit(sub(head,origin),[0,1,0]),left=joints['Left Shoulder']||joints['Left Upper Arm']||joints['Left Hand'],right=joints['Right Shoulder']||joints['Right Upper Arm']||joints['Right Hand'];let across=left&&right?sub(right,left):[1,0,0];across=unit(sub(across,mul(up,dot(across,up))),[1,0,0]);const forward=unit(cross(across,up),[0,0,1]),angle=angleDegrees*Math.PI/180,cos=Math.cos(angle),sin=Math.sin(angle),oriented={};for(const [name,point] of Object.entries(joints)){const relative=sub(point,origin),x=-dot(relative,across),y=dot(relative,up),z=dot(relative,forward);oriented[name]=[x*cos-z*sin,y,x*sin+z*cos]}return oriented}
// 몸 비율 표(m)와 좌우 색. 패널과 큰 장면 상체가 함께 쓴다. 출처: 2026-10 리그 월드 좌표에서 잰 뼈 길이(Chest→어깨 선 0.2106, 예전 리그는 0.0001이라 패널이 채운다). 팔·다리·Spine→Chest·어깨 선→Head 는 두 리그가 1e-4m 안에서 같다. gripBack 은 건에서 분사 축 뒤 손잡이까지(추정값).
const BODY={headAboveShoulders:.1368,upperChest:.2106,shoulderWidth:.416,upperArm:.243,foreArm:.243,gripBack:.3},LIMB_COLORS={arm:'#22d3ee',leg:'#9aa3ad',core:'#e6edf7'};
function gunGrip(gun,tool){return sub(gun,mul(unit(tool),BODY.gripBack))}
// 두 마디 IK: 뿌리에서 목표 쪽으로 상완·전완을 놓고 팔꿈치를 pole 쪽으로 굽힌다. 닿지 않으면 목표 쪽으로 곧게 편다.
function twoBoneIK(root,target,upper,fore,pole){const reach=sub(target,root),aim=unit(reach,[0,-1,0]),span=Math.max(.00001,Math.min(upper+fore,Math.max(length(reach),Math.abs(upper-fore)))),along=(upper*upper-fore*fore+span*span)/(2*span),bend=unit(sub(pole,mul(aim,dot(pole,aim))),[1,0,0]);return{elbow:add(add(root,mul(aim,along)),mul(bend,Math.sqrt(Math.max(0,upper*upper-along*along)))),hand:add(root,mul(aim,span))}}
// 예전 형식(2025) CSV 리그에는 Chest 와 어깨 선 사이 마디가 없어 어깨·머리가 그만큼 낮다(몸통이 짧게 보인다). 패널 그림에서만 그 마디를 끼워 어깨 두 갈래·머리를 척추(Chest→Head) 방향으로 함께 올린다. 뼈 방향·다른 뼈 길이·frame.joints 는 그대로다. 2026-10 리그는 Chest→Head 가 BODY.upperChest 보다 길어 그대로 둔다.
function withUpperChest(joints){const chest=joints.Chest,head=joints.Head;if(!chest||!head||length(sub(head,chest))>=BODY.upperChest)return joints;const lift=mul(unit(sub(head,chest)),BODY.upperChest),raised={...joints};for(const name of Object.keys(joints))if(/^(Head|(Left|Right) (Shoulder|Upper Arm|Fore Arm|Hand))$/.test(name))raised[name]=add(joints[name],lift);return raised}
// 패널 그림에서만 오른쪽 전완·손을 실측 건 손잡이(gunGrip)로 다시 세운다. 관절은 리그 좌표라 손잡이를 HMD 수평 틀로 풀어 Head 관절에 더한다. frame.joints 와 채점은 그대로이고 건 자료가 없으면 CSV 관절 그대로다.
function panelSkeleton(frame,angleDegrees=skeletonYaw){const source=withUpperChest(frame&&frame.joints||{}),joints=uprightSkeleton(source,angleDegrees),tracked=frame&&frame.tracked||{},gun=tracked.right_hand,hmd=tracked.head,tool=frame&&frame.tool_forward,camera=frame&&frame.camera_forward,body=uprightSkeleton(source,0),shoulder=body['Right Upper Arm'],elbow=body['Right Fore Arm'],hand=body['Right Hand'];if(!gun||!hmd||!tool||!camera||!shoulder||!elbow||!hand||length(tool)<.00001||Math.hypot(camera[0],camera[2])<.00001)return{joints,estimated:[]};const ahead=unit([camera[0],0,camera[2]]),side=cross([0,1,0],ahead),grip=sub(gunGrip(gun,tool),hmd),arm=twoBoneIK(shoulder,add(body.Head,[-dot(grip,side),grip[1],dot(grip,ahead)]),length(sub(elbow,shoulder)),length(sub(hand,elbow)),[-.5,-1,0]),angle=angleDegrees*Math.PI/180,cos=Math.cos(angle),sin=Math.sin(angle),turn=p=>[p[0]*cos-p[2]*sin,p[1],p[0]*sin+p[2]*cos];return{joints:{...joints,'Right Fore Arm':turn(arm.elbow),'Right Hand':turn(arm.hand)},estimated:['Right Fore Arm','Right Hand']}}
// 패널 투영: 몸 기준 틀(x 화면 오른쪽, y 위, z 보는 사람 쪽)을 위에서 약 17° 내려다본다. 화면 가로 중심은 골반, 바닥은 가장 낮은 발. 교정 고리(v2)도 이 함수로 자리를 구한다.
function skeletonProjection(joints,w,h){const names=Object.keys(joints);if(!names.length||!w||!h)return null;const tilt=.3,c=Math.cos(tilt),s=Math.sin(tilt),view=p=>[p[0],p[1]*c-p[2]*s,p[1]*s+p[2]*c],feet=names.filter(name=>/Foot|Toe/.test(name)),floor=Math.min(...(feet.length?feet:names).map(name=>joints[name][1])),seen=names.map(name=>view(joints[name])),half=Math.max(.8,...seen.map(p=>Math.abs(p[0]))),top=Math.max(...seen.map(p=>p[1])),bottom=Math.min(floor*c-1.13*s,...seen.map(p=>p[1])),span=Math.max(.25,top-bottom,2*half),scale=Math.min(w*.46/half,h*.84/(top-bottom));return{floor,project:p=>{const [x,y,d]=view(p),k=1+Math.max(-.25,Math.min(.25,d/span*.22));return[w/2+x*scale*k,h*.93-(y-bottom)*scale*k,k,d]}}}
function drawSkeletonView(frame){const rect=skeletonCanvas.getBoundingClientRect(),w=rect.width,h=rect.height;if(!w||!h)return;$('skeletonPosture').textContent=frame&&frame.posture&&frame.posture.label||'자세 판정 보류';skeletonCtx.clearRect(0,0,w,h);const {joints}=panelSkeleton(frame),view=skeletonProjection(joints,w,h);if(!view){skeletonCtx.fillStyle='#91a0b8';skeletonCtx.font='12px system-ui';skeletonCtx.textAlign='center';skeletonCtx.fillText('스켈레톤 데이터 없음',w/2,h/2);skeletonCtx.textAlign='start';return}const angle=skeletonYaw*Math.PI/180,cos=Math.cos(angle),sin=Math.sin(angle),onFloor=(x,z)=>view.project([x*cos-z*sin,view.floor,x*sin+z*cos]),polyline=(points,style,width)=>{skeletonCtx.strokeStyle=style;skeletonCtx.lineWidth=width;skeletonCtx.beginPath();points.forEach((p,i)=>i?skeletonCtx.lineTo(p[0],p[1]):skeletonCtx.moveTo(p[0],p[1]));skeletonCtx.stroke()};skeletonCtx.lineCap='round';skeletonCtx.lineJoin='round';const ticks=[-.8,-.4,0,.4,.8];for(const z of ticks)polyline([onFloor(-.8,z),onFloor(.8,z)],'rgba(190,206,232,.5)',1);for(const x of ticks)polyline([onFloor(x,-.8),onFloor(x,.8)],'rgba(190,206,232,.5)',1);polyline([onFloor(-.12,.6),onFloor(0,.76),onFloor(.12,.6)],'rgba(236,242,250,.78)',2);const side=name=>/Thigh|Calf|Foot|Toe/.test(name)?LIMB_COLORS.leg:/Shoulder|Arm|Hand/.test(name)?LIMB_COLORS.arm:LIMB_COLORS.core,items=[];for(const [a,b] of data.bones){if(!joints[a]||!joints[b])continue;const p=view.project(joints[a]),q=view.project(joints[b]),leg=/Thigh|Calf|Foot|Toe/.test(b);items.push({depth:(p[3]+q[3])/2,bone:[p,q],color:side(b),width:(leg?2.5:3.5)*(p[2]+q[2])/2,alpha:leg?.85:1})}for(const [name,p] of Object.entries(joints)){const q=view.project(p),isHead=name==='Head',isHand=name.endsWith(' Hand');items.push({depth:q[3]+.02,joint:q,color:isHead?'#f5f8ff':side(name),radius:(isHead?9:isHand?5:3)*q[2],alpha:/Thigh|Calf|Foot|Toe/.test(name)?.85:1})}const depths=items.map(item=>item.depth),near=Math.max(...depths),far=Math.min(...depths),range=Math.max(.001,near-far);items.sort((a,b)=>a.depth-b.depth);for(const item of items){skeletonCtx.globalAlpha=item.alpha*(.7+.3*(item.depth-far)/range);if(item.bone)polyline(item.bone,item.color,item.width);else{skeletonCtx.fillStyle=item.color;skeletonCtx.beginPath();skeletonCtx.arc(item.joint[0],item.joint[1],item.radius,0,Math.PI*2);skeletonCtx.fill()}}skeletonCtx.globalAlpha=1}
function setSkeletonYaw(value){skeletonYaw=((value+180)%360+360)%360-180;$('skeletonAngle').textContent=skeletonYaw===0?'정면 · 0°':`${skeletonYaw<0?'좌':'우'} · ${Math.abs(skeletonYaw)}°`;if(data&&data.frames.length)draw()}
function interpolatedFrame(seconds){let current=Math.max(0,Math.min(index,data.frames.length-1));while(current+1<data.frames.length&&data.frames[current+1].t<=seconds)current++;while(current>0&&data.frames[current].t>seconds)current--;const a=data.frames[current],b=data.frames[Math.min(current+1,data.frames.length-1)],span=Math.max(.000001,b.t-a.t),ratio=Math.max(0,Math.min(1,(seconds-a.t)/span));const tracked={},joints={};for(const key of new Set([...Object.keys(a.tracked||{}),...Object.keys(b.tracked||{})]))tracked[key]=mixPoint((a.tracked||{})[key],(b.tracked||{})[key],ratio);for(const key of new Set([...Object.keys(a.joints||{}),...Object.keys(b.joints||{})]))joints[key]=mixPoint((a.joints||{})[key],(b.joints||{})[key],ratio);const continuousPaint=a.spraying&&b.spraying&&a.ray_point&&b.ray_point&&a.zone_id===b.zone_id&&a.map_num===b.map_num;return{index:current,frame:{...a,t:seconds,tracked,joints,ray_point:continuousPaint?mixPoint(a.ray_point,b.ray_point,ratio):a.ray_point}}}
// 큰 장면 바닥: 실측 머리 높이에서 CSV 관절의 눈높이(머리−가장 낮은 발)를 뺀 높이(하위 5%)에, 머리·참조면·궤적이 차지한 가로 범위보다 1m 넓게 0.5m 격자를 깐다. 관절이 없으면 장면에서 가장 낮은 점.
let floorSource=null,floorGrid=null;
function sceneFloor(){if(floorSource===data)return floorGrid;floorSource=data;floorGrid=null;const stands=[],low=[Infinity,Infinity,Infinity],high=[-Infinity,-Infinity,-Infinity],take=p=>{if(!p)return;for(let i=0;i<3;i++){low[i]=Math.min(low[i],p[i]);high[i]=Math.max(high[i],p[i])}};for(const frame of data.frames){const head=(frame.tracked||{}).head,joints=frame.joints||{},feet=Object.keys(joints).filter(name=>/Foot|Toe/.test(name));take(head);if(head&&joints.Head&&feet.length)stands.push(head[1]-joints.Head[1]+Math.min(...feet.map(name=>joints[name][1])))}for(const item of data.reference_surface&&data.reference_surface.points||[])take(item.point);for(const sample of data.paint_trace||[])take(sample.point);if(!(low[0]<=high[0]))return null;stands.sort((a,b)=>a-b);const step=Math.max(.5,Math.ceil(Math.max(high[0]-low[0],high[2]-low[2])/20)/2);floorGrid={y:stands.length?stands[Math.floor(stands.length*.05)]:low[1],step,lo:[Math.floor(low[0]/step-2)*step,Math.floor(low[2]/step-2)*step],hi:[Math.ceil(high[0]/step+2)*step,Math.ceil(high[2]/step+2)*step]};return floorGrid}
function drawFloorGrid(center,w,h){const grid=sceneFloor();if(!grid)return;const segment=(a,b,whole)=>{const p=project(a,center,w,h),q=project(b,center,w,h);if(!(p[2]>0&&q[2]>0&&p[2]<6&&q[2]<6))return;ctx.strokeStyle=whole?'rgba(190,206,232,.5)':'rgba(190,206,232,.18)';ctx.beginPath();ctx.moveTo(p[0],p[1]);ctx.lineTo(q[0],q[1]);ctx.stroke()};ctx.lineWidth=1;for(let x=grid.lo[0];x<=grid.hi[0]+1e-6;x+=grid.step)segment([x,grid.y,grid.lo[1]],[x,grid.y,grid.hi[1]],Math.abs(x-Math.round(x))<1e-6);for(let z=grid.lo[1];z<=grid.hi[1]+1e-6;z+=grid.step)segment([grid.lo[0],grid.y,z],[grid.hi[0],grid.y,z],Math.abs(z-Math.round(z))<1e-6)}
function draw(){const rect=canvas.getBoundingClientRect(),w=rect.width,h=rect.height;ctx.clearRect(0,0,w,h);
if(!data||!data.frames.length)return;const rendered=interpolatedFrame(playheadSeconds),frame=rendered.frame,sourceIndex=rendered.index,stats=data.statistics[sourceIndex],center=viewCenter||((frame.tracked||{}).head||(frame.joints||{}).Pelvis||[0,0,0]);drawFloorGrid(center,w,h);if($('objectMesh').checked)drawObjectMesh(data.object_mesh,data.object_mesh_status,center,w,h);
if($('zoneSurface').checked)drawZoneSurfaces(data.zone_surfaces||[],center,w,h);referencePointCloud(data.reference_surface,center,w,h);drawZoneFlowGuides(data.zone_flow_guides,center,w,h);
if(!referencePreview){let previous=null,previousSegment=null,paintCount=0;for(const sample of data.paint_trace){if(sample.t>playheadSeconds)break;if(previousSegment!==sample.segment)previous=null;const color=`hsla(${24+sample.progress*28},100%,62%,${.35+sample.progress*.5})`;point(sample.point,2.8,color,center,w,h);if(previous){line(previous,sample.point,color,5,center,w,h);if(paintCount%45===0)arrow(previous,sample.point,'rgba(255,232,190,.9)',center,w,h)}previous=sample.point;previousSegment=sample.segment;paintCount++}if(previous&&frame.spraying&&frame.ray_point){line(previous,frame.ray_point,'rgba(255,184,92,.78)',5,center,w,h);point(frame.ray_point,3.5,'rgba(255,205,125,.9)',center,w,h)}
}
drawSkeletonView(frame);
if(!referencePreview)updateZonePanel(frame);
$('overlayFrame').textContent=`${sourceIndex+1} / ${data.frames.length} · ${frame.t.toFixed(2)}초`;
if(window.motionViewerPoseOverlay)window.motionViewerPoseOverlay.draw({frame,sourceIndex,playheadSeconds,durationSeconds:data.duration_seconds,center,w,h});
dispatchEvent(new CustomEvent('motion-viewer-frame',{detail:{frame,sourceIndex,playheadSeconds}}));
}
function tick(now){if(playing&&data&&data.frames.length){if(!last)last=now;const elapsed=Math.min(.1,(now-last)/1000)*Number($('speed').value);last=now;playheadSeconds+=elapsed;if(playheadSeconds>=data.duration_seconds){playheadSeconds=0;index=0}else index=interpolatedFrame(playheadSeconds).index;$('timeline').value=index;draw()}requestAnimationFrame(tick)}
$('worker').onchange=fillProcesses;$('process').onchange=fillModes;$('mode').onchange=fillSessions;$('session').onchange=loadMotion;$('timeline').oninput=e=>{referencePreview=false;index=Number(e.target.value);playheadSeconds=data.frames[index].t;last=0;draw()};$('pointSize').oninput=e=>{$('pointSizeValue').textContent=`${Number(e.target.value).toFixed(2).replace(/0+$/,'').replace(/\.$/,'')} px`;draw()};$('zoneSurface').onchange=draw;$('objectMesh').onchange=draw;$('skeletonTurnLeft').onclick=()=>setSkeletonYaw(skeletonYaw-15);$('skeletonTurnRight').onclick=()=>setSkeletonYaw(skeletonYaw+15);$('skeletonReset').onclick=()=>setSkeletonYaw(0);document.querySelector('.skeleton-card').ontoggle=()=>{if(document.querySelector('.skeleton-card').open)resize()};$('play').onclick=()=>{if(playing){playing=false;last=0;$('play').textContent='재생'}else startPlayback()};$('restart').onclick=()=>{playing=false;referencePreview=false;index=0;playheadSeconds=0;last=0;$('play').textContent='재생';$('timeline').value=0;draw()};
canvas.onpointerdown=e=>{if(e.button!==0&&e.button!==1)return;e.preventDefault();canvas.setPointerCapture(e.pointerId);drag=e.button===1?{mode:'pan',x:e.clientX,y:e.clientY,panX,panY}:{mode:'rotate',x:e.clientX,y:e.clientY,yaw,pitch};canvas.style.cursor=drag.mode==='pan'?'move':'grabbing'};canvas.onpointermove=e=>{if(!drag)return;if(drag.mode==='pan'){panX=drag.panX+e.clientX-drag.x;panY=drag.panY+e.clientY-drag.y}else{yaw=drag.yaw+(e.clientX-drag.x)*.01;pitch=Math.max(-1.2,Math.min(1.2,drag.pitch+(e.clientY-drag.y)*.01))}draw()};canvas.onpointerup=e=>{if(canvas.hasPointerCapture(e.pointerId))canvas.releasePointerCapture(e.pointerId);drag=null;canvas.style.cursor='grab'};canvas.onpointercancel=()=>{drag=null;canvas.style.cursor='grab'};canvas.onauxclick=e=>e.preventDefault();canvas.onwheel=e=>{e.preventDefault();zoom=Math.max(70,Math.min(420,zoom*(e.deltaY>0?.9:1.1)));draw()};
addEventListener('resize',resize);(async()=>{try{catalog=await json('/motion-viewer/api/catalog');$('worker').replaceChildren(...catalog.workers.map(worker=>option(worker.worker_id,`${worker.display_name}${worker.role==='expert'?' · 전문가 기준':''}`)));fillProcesses();resize();requestAnimationFrame(tick)}catch(e){$('status').textContent=e.message}})();
</script><script defer src="/motion-viewer/assets/motion_viewer_camera.js"></script><script defer src="/motion-viewer/assets/motion_viewer_pose.js"></script><script defer src="/motion-viewer/assets/motion_viewer_coating.js"></script><script defer src="/motion-viewer/assets/motion_viewer_chat.js"></script></body></html>"""


def viewer_v2_document() -> str:
    """v2 화면 틀에 기존 재생 스크립트를 그대로 끼운다. 두 화면이 같은 요소 ID와 같은 자료 경로를 쓴다."""
    shell = (Path(__file__).with_name("static") / "motion_viewer_v2.html").read_text(encoding="utf-8")
    start = _VIEWER_HTML.index("<script>") + len("<script>")
    return shell.replace("/*CORE_SCRIPT*/", _VIEWER_HTML[start:_VIEWER_HTML.index("</script>", start)], 1)


def create_motion_viewer_router(
    reference_root: Path,
    *,
    frame_count: int = 240,
    sequence_store: SequenceStore | None = None,
    mcp_base_url: str = "http://localhost:8100",
    transport: Any = None,
    pose_scorer: Any = score_body_raw,
    mesh_provider: Any = _default_mesh_provider,
    reports: V1Store | None = None,
) -> APIRouter:
    root = Path(reference_root)
    router = APIRouter()

    @router.get("/motion-viewer", include_in_schema=False, response_class=HTMLResponse)
    def motion_viewer() -> HTMLResponse:
        return HTMLResponse(portal_document(_VIEWER_HTML,"motion"), headers={"Cache-Control":"no-cache"})

    static_root = Path(__file__).with_name("static")

    @router.get("/motion-viewer/v2", include_in_schema=False, response_class=HTMLResponse)
    def motion_viewer_v2() -> HTMLResponse:
        return HTMLResponse(portal_document(viewer_v2_document(), "motion"), headers={"Cache-Control": "no-cache"})

    @router.get("/motion-viewer/assets/motion_viewer_v2.{kind}", include_in_schema=False)
    def v2_asset(kind: str) -> FileResponse:
        if kind not in {"css", "js"}:
            raise HTTPException(status_code=404, detail="Asset not found")
        return FileResponse(static_root / f"motion_viewer_v2.{kind}", media_type="text/css" if kind == "css" else "text/javascript",
                            headers={"Cache-Control": "no-cache"})

    @router.get("/motion-viewer/assets/motion_viewer_pose.js", include_in_schema=False)
    def pose_script() -> FileResponse:
        return FileResponse(
            static_root / "motion_viewer_pose.js", media_type="text/javascript",
            headers={"Cache-Control": "no-cache"},
        )

    @router.get("/motion-viewer/assets/motion_viewer_pose.css", include_in_schema=False)
    def pose_style() -> FileResponse:
        return FileResponse(
            static_root / "motion_viewer_pose.css", media_type="text/css",
            headers={"Cache-Control": "no-cache"},
        )

    @router.get("/motion-viewer/assets/motion_viewer_camera.js", include_in_schema=False)
    def camera_script() -> FileResponse:
        return FileResponse(
            static_root / "motion_viewer_camera.js", media_type="text/javascript",
            headers={"Cache-Control": "no-cache"},
        )

    @router.get("/motion-viewer/assets/motion_viewer_coating.js", include_in_schema=False)
    def coating_script() -> FileResponse:
        return FileResponse(
            static_root / "motion_viewer_coating.js", media_type="text/javascript",
            headers={"Cache-Control": "no-cache"},
        )

    @router.get("/motion-viewer/assets/motion_viewer_chat.js", include_in_schema=False)
    def chat_script() -> FileResponse:
        return FileResponse(
            static_root / "motion_viewer_chat.js", media_type="text/javascript",
            headers={"Cache-Control": "no-cache"},
        )

    @router.get("/motion-viewer/assets/flags/{country_code}.svg", include_in_schema=False)
    def answer_language_flag(country_code: str) -> FileResponse:
        if country_code not in {"kr", "us", "vn", "id", "th", "np", "mm", "uz", "lk", "ru", "cn", "kh"}:
            raise HTTPException(status_code=404, detail="Flag not found")
        return FileResponse(
            static_root / "flags" / f"{country_code}.svg", media_type="image/svg+xml",
            headers={"Cache-Control": "public, max-age=604800"},
        )

    @router.get("/motion-viewer/api/completed", include_in_schema=False)
    def completed_sessions() -> dict[str, Any]:
        """채팅이 고를 수 있는 세션 — complete 리포트가 있는 것만(질의가 그 리포트로 답한다)."""
        return {"sessions": reports.complete_sessions() if reports is not None else []}

    @router.get("/motion-viewer/api/catalog", include_in_schema=False)
    def motion_catalog() -> dict[str, Any]:
        workers: list[dict[str, Any]] = []
        for worker in _worker_sources(root):
            processes: list[dict[str, Any]] = []
            worker_root = Path(worker["root"])
            for process, label in _PROCESS_LABELS.items():
                modes: list[dict[str, Any]] = []
                for mode_id in range(1, 11):
                    try:
                        profile = _read_profile(worker_root, process, mode_id)
                        representative = _representative_session(profile)
                    except HTTPException:
                        continue
                    sessions = sorted(
                        str(item.get("session_id"))
                        for item in profile["sessions"]
                        if item.get("session_id")
                    )
                    modes.append({
                        "mode_id": mode_id,
                        "session_count": len(sessions),
                        "representative_session_id": representative,
                        "sessions": sessions,
                    })
                if modes:
                    processes.append({"process": process, "label": label, "modes": modes})
            if processes:
                workers.append({
                    "worker_id": worker["worker_id"],
                    "display_name": worker["display_name"],
                    "role": worker["role"],
                    "processes": processes,
                })
        legacy_processes = workers[0]["processes"] if workers else []
        return {"workers": workers, "processes": legacy_processes}

    @router.get("/motion-viewer/api/evaluations", include_in_schema=False)
    def evaluation_catalog() -> dict[str, Any]:
        sessions = (
            sequence_store.recent_sequence_catalog()
            if sequence_store is not None
            else []
        )
        return {"sessions": sessions}

    @router.get("/motion-viewer/api/evaluation", include_in_schema=False)
    def evaluation_data(
        session_id: str = Query(min_length=1, max_length=160),
        sequence_id: str = Query(min_length=1, max_length=160),
    ) -> dict[str, Any]:
        if sequence_store is None:
            raise HTTPException(status_code=503, detail="sequence store is unavailable")
        body = sequence_store.load_body(
            session_key=session_id, sequence_id=sequence_id
        )
        if body is None:
            raise HTTPException(status_code=404, detail="stored sequence not found")
        raw = pose_scorer(
            body,
            sequence_id=sequence_id,
            mcp_base_url=mcp_base_url,
            transport=transport,
        )
        return build_pose_view(
            body,
            raw,
            mesh_provider=mesh_provider,
            max_frames=frame_count,
        )

    @router.get("/motion-viewer/api/arm-pose", include_in_schema=False)
    def arm_pose_data(
        worker_id: str = Query(default=_EXPERT_WORKER_ID, pattern="^[A-Za-z0-9_-]+$"),
        process: str = Query(pattern="^(spray|blasting)$"),
        mode_id: int = Query(ge=1, le=10),
        session_id: str = Query(min_length=1, max_length=100),
    ) -> dict[str, Any]:
        """재생 기록의 자세 유사도(분사마다 팔 관절 엔진). 채점에 수 초 걸려 모션과 따로 받는다."""
        from trainer.demo_api.viewer_arm_pose import build_arm_pose_evaluation
        from trainer.demo_api.viewer_context import sign_context
        worker_root = Path(_worker_source(root, worker_id)["root"])
        profile = _read_profile(worker_root, process, mode_id)
        selected = _representative_session(profile) if session_id == "representative" else session_id
        evaluation = build_arm_pose_evaluation(
            _session_path(worker_root, process, mode_id, profile, selected), record_id=selected,
            process=process, mode_id=mode_id, mcp_base_url=mcp_base_url, transport=transport,
        )
        return {"selected_session_id": selected, "arm_pose_evaluation": evaluation,
                "signature": sign_context("arm_pose", evaluation)}

    @router.get("/motion-viewer/api/motion", include_in_schema=False)
    def motion_data(
        worker_id: str = Query(default=_EXPERT_WORKER_ID, pattern="^[A-Za-z0-9_-]+$"),
        process: str = Query(pattern="^(spray|blasting)$"),
        mode_id: int = Query(ge=1, le=10),
        session_id: str = Query(min_length=1, max_length=100),
    ) -> dict[str, Any]:
        worker = _worker_source(root, worker_id)
        worker_root = Path(worker["root"])
        profile = _read_profile(worker_root, process, mode_id)
        zone_id_policy = profile.get("identity", {}).get("zone_id_policy")
        legacy_zone_ids = (
            zone_id_policy == "legacy_expert"
            if zone_id_policy in {"source", "legacy_expert"}
            else worker_root == root
        )
        representative = _representative_session(profile)
        selected = representative if session_id == "representative" else session_id
        selected_path = _session_path(worker_root, process, mode_id, profile, selected)
        selected_frames = _canonicalize_zone_items(
            _frames(selected_path, frame_count), process=process, mode_id=mode_id,
            legacy_expert_zone_ids=legacy_zone_ids,
        )
        if not selected_frames:
            raise HTTPException(status_code=422, detail="expert session has no usable skeleton frames")
        sampled_sessions: list[list[dict[str, Any]]] = []
        sampled_session_records: list[dict[str, Any]] = []
        session_paths: list[Path] = []
        representative_frames: list[dict[str, Any]] = []
        reference_profile = profile
        expert_session_paths: list[Path] = []
        for session in profile["sessions"]:
            current_id = str(session.get("session_id", ""))
            if not current_id:
                continue
            try:
                current_path = _session_path(worker_root, process, mode_id, profile, current_id)
                sampled = _canonicalize_zone_items(
                    _frames(current_path, frame_count), process=process, mode_id=mode_id,
                    legacy_expert_zone_ids=legacy_zone_ids,
                )
            except HTTPException:
                continue
            if sampled:
                sampled_sessions.append(sampled)
                sampled_session_records.append({
                    "session_id": current_id,
                    "session": session,
                    "frames": sampled,
                })
                session_paths.append(current_path)
                if current_id == representative:
                    representative_frames = sampled
        if worker["role"] == "expert":
            evaluation_references = [
                record for record in sampled_session_records
                if record["session_id"] != selected
            ]
        else:
            expert_profile = _read_profile(root, process, mode_id)
            reference_profile = expert_profile
            evaluation_references = []
            for expert_session in expert_profile["sessions"]:
                expert_id = str(expert_session.get("session_id", ""))
                if not expert_id:
                    continue
                try:
                    expert_path = _session_path(root, process, mode_id, expert_profile, expert_id)
                    expert_frames = _canonicalize_zone_items(
                        _frames(expert_path, frame_count), process=process, mode_id=mode_id,
                        legacy_expert_zone_ids=expert_profile.get("identity", {}).get("zone_id_policy") != "source",
                    )
                except HTTPException:
                    continue
                if expert_frames:
                    expert_session_paths.append(expert_path)
                    evaluation_references.append({
                        "session_id": expert_id,
                        "session": expert_session,
                        "frames": expert_frames,
                    })
        duration = float(selected_frames[-1]["t"]) if selected_frames else 0.0
        paint_trace = _canonicalize_zone_items(
            _dense_paint_trace(selected_path), process=process, mode_id=mode_id,
            legacy_expert_zone_ids=legacy_zone_ids,
        )
        reference_paths = session_paths if worker["role"] == "expert" else expert_session_paths
        reference_legacy_zone_ids = (
            legacy_zone_ids if worker["role"] == "expert"
            else reference_profile.get("identity", {}).get("zone_id_policy") != "source"
        )
        paint_statistics = _reference_paint_statistics(reference_paths)
        zone_flow_guides = _canonicalize_zone_items(
            _zone_flow_guides(reference_paths), process=process, mode_id=mode_id,
            legacy_expert_zone_ids=reference_legacy_zone_ids,
        )
        zone_surfaces = _canonicalize_zone_items(
            _zone_surface_statistics(reference_paths), process=process, mode_id=mode_id,
            legacy_expert_zone_ids=reference_legacy_zone_ids,
        )
        direction_summary = _direction_summary(paint_trace)
        zone_order, zone_statistics = _zone_playback_statistics(
            profile,
            selected,
            paint_trace,
            direction_summary,
            paint_statistics["direction_distribution"],
            reference_profile=reference_profile,
        )
        selected_frame_zones = _zone_frame_profiles(selected_frames)
        reference_frames = sampled_sessions if worker["role"] == "expert" else [record["frames"] for record in evaluation_references]
        sampled_frame_zones = [_zone_frame_profiles(frames) for frames in reference_frames]
        for zone_key, statistics_item in zone_statistics.items():
            if zone_key == "all":
                continue
            input_distances = [
                distance
                for frame_profile in sampled_frame_zones
                if (distance := _number(frame_profile.get(zone_key, {}).get("distance_mm")))
                is not None
            ]
            if input_distances:
                statistics_item["median_distance_mm"] = round(
                    statistics.median(input_distances), 3
                )
            selected_input_distance = _number(
                selected_frame_zones.get(zone_key, {}).get("distance_mm")
            )
            if selected_input_distance is not None:
                statistics_item["selected_distance_mm"] = round(
                    selected_input_distance, 3
                )
        reference_flows = expert_reference_flows(reference_profile)
        selected_flow = [str(item["zone_id"]) for item in zone_order]
        order_score = score_visit_flow(selected_flow, reference_flows)
        reference_order_evaluation = {
            "recommended_flow": SequenceAnalysisEngine._flow(reference_profile),
            "selected_flow": selected_flow,
            "reference_session_count": len(reference_flows),
            **order_score,
        }
        try:
            supplied_mesh = mesh_provider(str(mode_id)) or {}
        except (FileNotFoundError, LookupError, ValueError):
            supplied_mesh = {"available": False, "reason": "display_error"}
        mesh_available = bool(
            supplied_mesh.get("available")
            and supplied_mesh.get("vertices")
            and supplied_mesh.get("triangles")
        )
        object_mesh = {
            **dict(supplied_mesh),
            "available": mesh_available,
            "vertices": supplied_mesh.get("vertices", []) if mesh_available else [],
            "triangles": supplied_mesh.get("triangles", []) if mesh_available else [],
            "fallback": None if mesh_available else "ray_point_surface",
        }
        payload = {
            "expert_name": _EXPERT_NAME,
            "worker": {
                "worker_id": worker["worker_id"],
                "display_name": worker["display_name"],
                "role": worker["role"],
            },
            "pose_evaluation": _multi_session_zone_evaluation(
                selected_frames,
                profile,
                selected,
                evaluation_references,
            ),
            "process": process,
            "process_label": _PROCESS_LABELS[process],
            "mode_id": mode_id,
            "selected_session_id": selected,
            "representative_session_id": representative,
            "sampled_session_count": len(sampled_sessions),
            "duration_seconds": round(duration, 3),
            "bones": _BONES,
            "frames": selected_frames,
            "statistics": _spread(selected_frames, sampled_sessions),
            "paint_trace": paint_trace,
            "zone_flow_guides": zone_flow_guides,
            "direction_summary": direction_summary,
            "zone_surfaces": zone_surfaces,
            "zone_order": zone_order,
            "reference_order_evaluation": reference_order_evaluation,
            "zone_statistics": zone_statistics,
            "coating_reference": _coating_reference(reference_profile),
            "object_mesh": object_mesh,
            "object_mesh_status": "verified" if mesh_available else "ray_point_fallback",
            **paint_statistics,
        }
        from trainer.demo_api.coating_query import viewer_coating_context
        payload["coating_evaluation"] = viewer_coating_context(payload, target=_target_coating(
            process, mode_id, _record_height_maps(worker_root, process, mode_id, profile, selected),
            [*payload["pose_evaluation"].get("zones", {}), *payload["coating_reference"]],
            _session_record(profile, selected)))
        from trainer.demo_api.dialogue import viewer_dialogue_context
        payload["dialogue_evaluation"] = viewer_dialogue_context(payload)
        from trainer.demo_api.playback_direction import build_playback_snapshot
        snapshot_key = f"{frame_count}:{selected_path}:{selected_path.stat().st_mtime_ns}:{selected_path.stat().st_size}:{process}:{mode_id}:{legacy_zone_ids}"
        with _open_csv(selected_path) as clock_handle:
            csv_origin = _number(next(csv.DictReader(clock_handle), {}).get("시간(초)"))
        clock_valid = csv_origin is not None and bool(selected_frames) and all(_number(frame.get("csv_time_seconds")) is not None for frame in selected_frames)
        offset = float(selected_frames[0]["csv_time_seconds"]) - csv_origin if clock_valid else 0.0
        payload["frame_direction_context"], payload["frame_direction_signature"] = build_playback_snapshot(
            paint_trace, record_id=selected, duration_seconds=duration, cache_key=snapshot_key,
            time_offset_seconds=offset, clock_valid=clock_valid)

        from trainer.demo_api.viewer_context import viewer_signatures
        payload["query_context_signatures"] = viewer_signatures(payload)
        return payload

    return router
