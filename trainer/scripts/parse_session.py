#!/usr/bin/env python3
# pyright: reportAny=false, reportExplicitAny=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportUnusedCallResult=false

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_TRACKED_OBJECT_KEYS = [
    "gun_hangil_short",
    "gun_hangil_pole",
    "gun_bando_short",
    "gun_bando_pole",
    "flashlight",
    "camera",
]
DEFAULT_JOINT_NAMES = [
    "Pelvis",
    "Spine",
    "Chest",
    "LeftShoulder",
    "LeftUpperArm",
    "LeftForeArm",
    "LeftHand",
    "RightShoulder",
    "RightUpperArm",
    "RightForeArm",
    "RightHand",
    "LeftThigh",
    "LeftCalf",
    "LeftFoot",
    "LeftToe",
    "RightThigh",
    "RightCalf",
    "RightFoot",
    "RightToe",
    "Head",
]
PARKED_COORD_EPSILON = 1e-3
ACTIVE_GUN_VARIANCE_EPSILON = 1e-6


class SessionValidationError(ValueError):
    pass


@dataclass
class SessionIndex:
    schema_version: str
    source: dict[str, Any]
    conventions: dict[str, Any]
    capture: dict[str, Any]
    worker: dict[str, Any]
    skeleton_def: dict[str, Any]
    sequences: list[dict[str, Any]]
    totals: dict[str, Any]


@dataclass
class SequenceMetadata:
    seq_id: int
    source_data_id: int | None
    definition: str
    frame_range: dict[str, int]
    spray_range: dict[str, Any]
    t_start_sec: float
    t_end_sec: float
    note: str | None


@dataclass
class SequenceDocument:
    schema_version: str
    session_ref: dict[str, Any]
    sequence: SequenceMetadata
    stats: dict[str, Any]
    frames: list[dict[str, Any]]
    settings_override: dict[str, Any] | None


@dataclass
class FrameValidity:
    frame_index: int
    t_sec: float
    is_valid: bool
    issues: list[str]


@dataclass
class DegradedField:
    name: str
    reason: str


@dataclass
class ActiveGunInference:
    key: str
    confidence: float
    position_variance: float
    candidates: dict[str, float]


@dataclass
class ParseResult:
    sequence: SequenceDocument
    session_index: SessionIndex | None
    frame_count: int
    inferred_active_gun: ActiveGunInference
    frame_validity: list[FrameValidity]
    degraded_fields: list[DegradedField]
    surface_hit_region_bursts: list[dict[str, Any]]
    warnings: list[str]
    summary: dict[str, Any]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_session_index(payload: dict[str, Any]) -> SessionIndex:
    return SessionIndex(
        schema_version=str(payload["schema_version"]),
        source=dict(payload["source"]),
        conventions=dict(payload["conventions"]),
        capture=dict(payload["capture"]),
        worker=dict(payload["worker"]),
        skeleton_def=dict(payload["skeleton_def"]),
        sequences=list(payload.get("sequences", [])),
        totals=dict(payload.get("totals", {})),
    )


def _parse_sequence_document(payload: dict[str, Any]) -> SequenceDocument:
    sequence_payload = payload["sequence"]
    return SequenceDocument(
        schema_version=str(payload["schema_version"]),
        session_ref=dict(payload.get("session_ref", {})),
        sequence=SequenceMetadata(
            seq_id=int(sequence_payload["seq_id"]),
            source_data_id=sequence_payload.get("source_data_id"),
            definition=str(sequence_payload["definition"]),
            frame_range=dict(sequence_payload["frame_range"]),
            spray_range=dict(sequence_payload["spray_range"]),
            t_start_sec=float(sequence_payload["t_start_sec"]),
            t_end_sec=float(sequence_payload["t_end_sec"]),
            note=sequence_payload.get("note"),
        ),
        stats=dict(payload.get("stats", {})),
        frames=list(payload.get("frames", [])),
        settings_override=payload.get("settings_override"),
    )


def _resolve_session_index_path(sequence_path: Path, sequence_payload: dict[str, Any], session_index_path: Path | None) -> Path | None:
    if session_index_path is not None:
        return session_index_path
    session_ref = sequence_payload.get("session_ref") or {}
    session_file = session_ref.get("session_file")
    if not session_file:
        return None
    resolved = sequence_path.parent / str(session_file)
    return resolved if resolved.is_file() else None


def _require_numeric(value: Any, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or math.isnan(float(value)):
        raise SessionValidationError(f"{label} must be numeric")
    return float(value)


def _require_vec3(value: Any, label: str) -> list[float]:
    if not isinstance(value, list) or len(value) != 3:
        raise SessionValidationError(f"{label} must be a length-3 list")
    return [_require_numeric(item, f"{label}[{idx}]") for idx, item in enumerate(value)]


def _dt_summary(frames: list[dict[str, Any]]) -> tuple[dict[str, float], list[str]]:
    if len(frames) < 2:
        return {"mean": 0.0, "min": 0.0, "max": 0.0}, []
    deltas = [frames[idx]["t_sec"] - frames[idx - 1]["t_sec"] for idx in range(1, len(frames))]
    dt_mean = mean(deltas)
    dt_min = min(deltas)
    dt_max = max(deltas)
    warnings: list[str] = []
    if dt_mean > 0:
        spread_ratio = (dt_max - dt_min) / dt_mean
        if spread_ratio > 0.25:
            warnings.append(
                f"Frame dt spread is wide: mean={dt_mean:.6f}s min={dt_min:.6f}s max={dt_max:.6f}s"
            )
    return {"mean": dt_mean, "min": dt_min, "max": dt_max}, warnings


def _expected_joint_names(session_index: SessionIndex | None) -> list[str]:
    if session_index is None:
        return DEFAULT_JOINT_NAMES[1:]
    joints = list(session_index.skeleton_def.get("joints") or DEFAULT_JOINT_NAMES)
    if len(joints) != 20:
        raise SessionValidationError(f"Expected 20 joints in skeleton_def, found {len(joints)}")
    return joints[1:]


def _expected_object_keys(session_index: SessionIndex | None) -> list[str]:
    if session_index is None:
        return DEFAULT_TRACKED_OBJECT_KEYS
    objects = [item.get("key") for item in session_index.capture.get("tracked_objects", [])]
    if len(objects) != 6:
        raise SessionValidationError(f"Expected 6 tracked objects in session index, found {len(objects)}")
    return [str(item) for item in objects]


def _validate_frame(frame: dict[str, Any], expected_joints: list[str], expected_objects: list[str]) -> FrameValidity:
    issues: list[str] = []
    frame_index = int(frame["i"])
    t_sec = _require_numeric(frame["t_sec"], f"frame[{frame_index}].t_sec")

    skeleton = frame.get("skeleton")
    if not isinstance(skeleton, dict):
        raise SessionValidationError(f"frame[{frame_index}].skeleton missing")
    _require_vec3(skeleton.get("root_pos_m"), f"frame[{frame_index}].skeleton.root_pos_m")
    _require_vec3(skeleton.get("root_rot_euler_deg"), f"frame[{frame_index}].skeleton.root_rot_euler_deg")

    joints = skeleton.get("joints_rot_euler_deg")
    if not isinstance(joints, dict):
        raise SessionValidationError(f"frame[{frame_index}].skeleton.joints_rot_euler_deg missing")
    joint_names = set(joints)
    expected_joint_set = set(expected_joints)
    if joint_names != expected_joint_set:
        missing = sorted(expected_joint_set - joint_names)
        extra = sorted(joint_names - expected_joint_set)
        details: list[str] = []
        if missing:
            details.append(f"missing joints={missing}")
        if extra:
            details.append(f"extra joints={extra}")
        raise SessionValidationError(f"frame[{frame_index}] invalid joint set: {'; '.join(details)}")
    for joint_name in expected_joints:
        _require_vec3(joints[joint_name], f"frame[{frame_index}].skeleton.joints_rot_euler_deg.{joint_name}")

    objects = frame.get("objects")
    if not isinstance(objects, dict):
        raise SessionValidationError(f"frame[{frame_index}].objects missing")
    object_names = set(objects)
    expected_object_set = set(expected_objects)
    if object_names != expected_object_set:
        missing = sorted(expected_object_set - object_names)
        extra = sorted(object_names - expected_object_set)
        details = []
        if missing:
            details.append(f"missing objects={missing}")
        if extra:
            details.append(f"extra objects={extra}")
        raise SessionValidationError(f"frame[{frame_index}] invalid tracked objects: {'; '.join(details)}")
    for object_name in expected_objects:
        obj = objects[object_name]
        if not isinstance(obj, dict):
            raise SessionValidationError(f"frame[{frame_index}].objects.{object_name} missing")
        _require_vec3(obj.get("pos_m"), f"frame[{frame_index}].objects.{object_name}.pos_m")
        _require_vec3(obj.get("rot_euler_deg"), f"frame[{frame_index}].objects.{object_name}.rot_euler_deg")

    gun_metrics = frame.get("gun_metrics")
    if not isinstance(gun_metrics, dict):
        raise SessionValidationError(f"frame[{frame_index}].gun_metrics missing")
    for metric_name in ["distance_input_mm", "speed_input_mms", "distance_real_mm", "speed_real_mms"]:
        _require_numeric(gun_metrics.get(metric_name), f"frame[{frame_index}].gun_metrics.{metric_name}")

    return FrameValidity(frame_index=frame_index, t_sec=t_sec, is_valid=not issues, issues=issues)


def _validate_frames(sequence: SequenceDocument, expected_joints: list[str], expected_objects: list[str]) -> tuple[list[FrameValidity], list[str], dict[str, float]]:
    frames = sequence.frames
    if not frames:
        raise SessionValidationError("frames must be non-empty")

    validity: list[FrameValidity] = []
    warnings: list[str] = []
    previous_i: int | None = None
    previous_t: float | None = None
    for frame in frames:
        current_i = _require_int(frame.get("i"), "frame.i")
        current_t = _require_numeric(frame.get("t_sec"), f"frame[{current_i}].t_sec")
        if previous_i is not None and current_i <= previous_i:
            raise SessionValidationError("frame indices must be strictly increasing")
        if previous_t is not None and current_t <= previous_t:
            raise SessionValidationError("frame timestamps must be strictly increasing")
        validity.append(_validate_frame(frame, expected_joints, expected_objects))
        previous_i = current_i
        previous_t = current_t

    dt_stats, dt_warnings = _dt_summary(frames)
    warnings.extend(dt_warnings)
    return validity, warnings, dt_stats


def _require_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise SessionValidationError(f"{label} must be an integer")
    return value


def _validate_ranges(sequence: SequenceDocument, warnings: list[str]) -> None:
    frame_range = sequence.sequence.frame_range
    spray_range = sequence.sequence.spray_range
    start_i = int(frame_range["start_i"])
    end_i = int(frame_range["end_i"])
    spray_start = int(spray_range["start_i"])
    spray_end = int(spray_range["end_i"])
    if start_i > end_i:
        raise SessionValidationError("frame_range start_i must be <= end_i")
    if spray_start < start_i or spray_end > end_i or spray_start > spray_end:
        raise SessionValidationError("spray_range must be within frame_range")

    actual_frame_count = len(sequence.frames)
    expected_full_count = end_i - start_i + 1
    actual_spray_count = sum(bool(frame.get("spray_on")) for frame in sequence.frames)
    stats_spray_count = sequence.stats.get("spray_frame_count")
    if stats_spray_count is not None:
        stats_spray_count = int(stats_spray_count)
        if actual_frame_count == expected_full_count and actual_spray_count != stats_spray_count:
            raise SessionValidationError(
                f"spray_on frame count {actual_spray_count} does not match stats.spray_frame_count {stats_spray_count}"
            )
        if actual_frame_count != expected_full_count:
            warnings.append(
                f"Sequence sample is truncated ({actual_frame_count}/{expected_full_count} frames); spray_frame_count sanity check skipped"
            )


def _position_variance(frames: list[dict[str, Any]], object_key: str) -> float:
    coords = list(zip(*[frame["objects"][object_key]["pos_m"] for frame in frames]))
    coordinate_variances = [mean([(value - mean(axis_values)) ** 2 for value in axis_values]) for axis_values in coords]
    return float(sum(coordinate_variances))


def _is_parked(frames: list[dict[str, Any]], object_key: str) -> bool:
    first = frames[0]["objects"][object_key]["pos_m"]
    return all(
        all(abs(current[idx] - first[idx]) <= PARKED_COORD_EPSILON for idx in range(3))
        for current in [frame["objects"][object_key]["pos_m"] for frame in frames]
    )


def infer_active_gun(sequence: SequenceDocument, session_index: SessionIndex | None) -> ActiveGunInference:
    candidate_keys = [key for key in _expected_object_keys(session_index) if key.startswith("gun_")]
    candidate_variances = {key: _position_variance(sequence.frames, key) for key in candidate_keys}
    movable_candidates = {
        key: variance
        for key, variance in candidate_variances.items()
        if variance > ACTIVE_GUN_VARIANCE_EPSILON and not _is_parked(sequence.frames, key)
    }
    if not movable_candidates:
        fallback_key = max(candidate_variances, key=lambda candidate: candidate_variances[candidate])
        fallback_variance = candidate_variances[fallback_key]
        return ActiveGunInference(
            key=fallback_key,
            confidence=0.0,
            position_variance=fallback_variance,
            candidates=candidate_variances,
        )
    best_key = max(movable_candidates, key=lambda candidate: movable_candidates[candidate])
    ranked = sorted(movable_candidates.values(), reverse=True)
    best_variance = ranked[0]
    runner_up = ranked[1] if len(ranked) > 1 else 0.0
    confidence = 1.0 if best_variance <= 0 else min(1.0, (best_variance - runner_up) / best_variance + 0.5)
    return ActiveGunInference(
        key=best_key,
        confidence=round(confidence, 4),
        position_variance=best_variance,
        candidates=candidate_variances,
    )


def _collect_degraded_fields(sequence: SequenceDocument, session_index: SessionIndex | None) -> list[DegradedField]:
    degraded: list[DegradedField] = []
    if all(frame.get("surface_hit") is None for frame in sequence.frames):
        degraded.append(DegradedField(name="surface_hit", reason="All frames have surface_hit=null; 0.4.0 region aggregation unavailable"))
    if session_index is not None:
        worker = session_index.worker
        conventions = session_index.conventions
        rotation_value = conventions.get("rotation")
        rotation = rotation_value if isinstance(rotation_value, dict) else {}
        if worker.get("skill_level") is None:
            degraded.append(DegradedField(name="skill_level", reason="worker.skill_level is null in session index"))
        if worker.get("handedness") is None:
            degraded.append(DegradedField(name="handedness", reason="worker.handedness is null in session index"))
        if rotation.get("order") is None:
            degraded.append(DegradedField(name="rotation.order", reason="conventions.rotation.order is null"))
        if conventions.get("coordinate_frame") is None:
            degraded.append(DegradedField(name="coordinate_frame", reason="conventions.coordinate_frame is null"))
    return degraded


def aggregate_surface_hit_regions(sequence: SequenceDocument) -> list[dict[str, Any]]:
    frames = sequence.frames
    if all(frame.get("surface_hit") is None for frame in frames):
        return []
    bursts: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for frame in frames:
        if frame.get("spray_on"):
            current.append(frame)
        elif current:
            bursts.append(current)
            current = []
    if current:
        bursts.append(current)

    aggregated: list[dict[str, Any]] = []
    for burst_index, burst_frames in enumerate(bursts, start=1):
        total_hit_frames = 0
        region_counts: dict[str, int] = {}
        for frame in burst_frames:
            surface_hit = frame.get("surface_hit")
            if not isinstance(surface_hit, dict):
                continue
            region_id = surface_hit.get("region_id")
            if region_id is None:
                continue
            total_hit_frames += 1
            region_counts[str(region_id)] = region_counts.get(str(region_id), 0) + 1
        if total_hit_frames == 0:
            continue
        aggregated.append(
            {
                "burst_index": burst_index,
                "frame_count": len(burst_frames),
                "hit_frame_count": total_hit_frames,
                "region_ratios": {
                    region: count / total_hit_frames for region, count in sorted(region_counts.items())
                },
            }
        )
    return aggregated


def parse_session_file(input_path: str | Path, session_index_path: str | Path | None = None) -> ParseResult:
    sequence_path = Path(input_path)
    if not sequence_path.is_file():
        raise SessionValidationError(f"Input file not found: {sequence_path}")
    sequence_payload = _load_json(sequence_path)
    sequence = _parse_sequence_document(sequence_payload)

    resolved_session_index_path = _resolve_session_index_path(
        sequence_path,
        sequence_payload,
        Path(session_index_path) if session_index_path is not None else None,
    )
    session_index = None
    if resolved_session_index_path is not None:
        session_index = _parse_session_index(_load_json(resolved_session_index_path))

    expected_joints = _expected_joint_names(session_index)
    expected_objects = _expected_object_keys(session_index)
    frame_validity, warnings, dt_stats = _validate_frames(sequence, expected_joints, expected_objects)
    _validate_ranges(sequence, warnings)
    degraded_fields = _collect_degraded_fields(sequence, session_index)
    active_gun = infer_active_gun(sequence, session_index)
    surface_hit_region_bursts = aggregate_surface_hit_regions(sequence)

    summary = {
        "schema_version": sequence.schema_version,
        "frame_range": dict(sequence.sequence.frame_range),
        "parsed_frame_count": len(sequence.frames),
        "skeleton_joint_count": len(expected_joints) + 1,
        "skeleton_joint_model": "root+19",
        "tracked_object_count": len(expected_objects),
        "dt": dt_stats,
        "spray_on_frame_count": sum(bool(frame.get("spray_on")) for frame in sequence.frames),
        "surface_hit_burst_count": len(surface_hit_region_bursts),
    }

    return ParseResult(
        sequence=sequence,
        session_index=session_index,
        frame_count=len(sequence.frames),
        inferred_active_gun=active_gun,
        frame_validity=frame_validity,
        degraded_fields=degraded_fields,
        surface_hit_region_bursts=surface_hit_region_bursts,
        warnings=warnings,
        summary=summary,
    )


def _format_validate_report(result: ParseResult) -> str:
    degraded_names = ", ".join(field.name for field in result.degraded_fields) or "none"
    lines = [
        f"Parsed {result.frame_count} frames from seq_id={result.sequence.sequence.seq_id}",
        f"Skeleton: {result.summary['skeleton_joint_count']} joints ({result.summary['skeleton_joint_model']})",
        f"Tracked objects per frame: {result.summary['tracked_object_count']}",
        (
            "Frame dt mean/min/max: "
            f"{result.summary['dt']['mean']:.6f}/{result.summary['dt']['min']:.6f}/{result.summary['dt']['max']:.6f} sec"
        ),
        (
            "Inferred active gun: "
            f"{result.inferred_active_gun.key} (confidence={result.inferred_active_gun.confidence:.2f}, "
            f"variance={result.inferred_active_gun.position_variance:.6f})"
        ),
        f"Degraded fields: {degraded_names}",
    ]
    for warning in result.warnings:
        lines.append(f"Warning: {warning}")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse and validate ship-painting session sequence exports.")
    parser.add_argument("--input", type=Path, required=True, help="Path to a sequence JSON export.")
    parser.add_argument("--session-index", type=Path, default=None, help="Optional path to the session index JSON.")
    parser.add_argument("--output", type=Path, default=None, help="Optional output path for JSON parse summary.")
    parser.add_argument("--validate", action="store_true", help="Print a concise validation report.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")
    try:
        result = parse_session_file(args.input, args.session_index)
    except SessionValidationError as exc:
        logger.error("Validation failed: %s", exc)
        return 1
    except Exception:
        logger.exception("Failed to parse session file")
        return 1

    if args.validate:
        print(_format_validate_report(result))
    else:
        payload = {
            "frame_count": result.frame_count,
            "inferred_active_gun": asdict(result.inferred_active_gun),
            "degraded_fields": [asdict(field) for field in result.degraded_fields],
            "warnings": result.warnings,
            "summary": result.summary,
            "surface_hit_region_bursts": result.surface_hit_region_bursts,
        }
        text = json.dumps(payload, ensure_ascii=False, indent=2)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text, encoding="utf-8")
        else:
            print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
