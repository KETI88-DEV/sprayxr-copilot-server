#!/usr/bin/env python3
# pyright: reportAny=false, reportExplicitAny=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportUnusedCallResult=false

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from trainer.scripts.parse_session import ParseResult, parse_session_file

logger = logging.getLogger(__name__)

MIN_SESSION_COUNT = 2
MIN_SPRAY_ON_FRAMES = 4


@dataclass
class BurstRecord:
    session_path: Path
    region_id: str
    handedness: str
    frames: list[dict[str, Any]]


@dataclass
class LoadedSession:
    path: Path
    result: ParseResult
    skill_level: str | None
    handedness: str | None
    proxy_score: float | None
    spray_on_frame_count: int
    has_region_info: bool
    is_truncated: bool


@dataclass
class BuildArtifactsResult:
    pose_payload: dict[str, Any]
    flow_payload: dict[str, Any]
    reasons: list[str]


def _expand_session_paths(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        candidate_matches = sorted(Path().glob(pattern)) if any(token in pattern for token in "*?[]") else [Path(pattern)]
        for match in candidate_matches:
            resolved = match.resolve()
            if resolved.is_file() and resolved not in seen:
                seen.add(resolved)
                paths.append(resolved)
    return paths


def _spray_on_frames(result: ParseResult) -> list[dict[str, Any]]:
    return [frame for frame in result.sequence.frames if bool(frame.get("spray_on"))]


def _frame_region_id(frame: dict[str, Any]) -> str | None:
    surface_hit = frame.get("surface_hit")
    if isinstance(surface_hit, dict) and surface_hit.get("region_id") is not None:
        return str(surface_hit["region_id"])
    target_region = frame.get("target_region")
    if target_region is not None:
        return str(target_region)
    target_region_summary = frame.get("target_region_summary")
    if isinstance(target_region_summary, dict) and target_region_summary.get("region_id") is not None:
        return str(target_region_summary["region_id"])
    return None


def _proxy_score(result: ParseResult) -> float | None:
    spray_frames = _spray_on_frames(result)
    if len(spray_frames) < 2:
        return None
    distances = [float(frame["gun_metrics"]["distance_real_mm"]) for frame in spray_frames]
    speeds = [float(frame["gun_metrics"]["speed_real_mms"]) for frame in spray_frames]
    distance_mean = mean(distances)
    speed_mean = mean(speeds)
    distance_variance = mean([(value - distance_mean) ** 2 for value in distances])
    speed_variance = mean([(value - speed_mean) ** 2 for value in speeds])
    return -(distance_variance + speed_variance)


def _load_session(path: Path) -> LoadedSession:
    result = parse_session_file(path)
    skill_level = None
    handedness = None
    if result.session_index is not None:
        worker = result.session_index.worker
        raw_skill_level = worker.get("skill_level")
        skill_level = str(raw_skill_level) if raw_skill_level is not None else None
        raw_handedness = worker.get("handedness")
        handedness = str(raw_handedness) if raw_handedness is not None else None
    spray_frames = _spray_on_frames(result)
    return LoadedSession(
        path=path,
        result=result,
        skill_level=skill_level,
        handedness=handedness,
        proxy_score=_proxy_score(result),
        spray_on_frame_count=len(spray_frames),
        has_region_info=any(_frame_region_id(frame) is not None for frame in spray_frames),
        is_truncated=any("truncated" in warning.lower() for warning in result.warnings),
    )


def _insufficient_payload(level: str, artifact: str, session_count: int, reasons: list[str]) -> dict[str, Any]:
    return {
        "level": level,
        "artifact": artifact,
        "status": "insufficient_data",
        "session_count": session_count,
        "reasons": reasons,
    }


def _collect_reasons(level: str, selected_sessions: list[LoadedSession], source_session_count: int) -> list[str]:
    reasons: list[str] = []
    if source_session_count < MIN_SESSION_COUNT:
        reasons.append(
            f"Only {source_session_count} session available for level {level}; single session cannot establish a reference distribution."
        )
    if not any(session.has_region_info for session in selected_sessions):
        reasons.append("No region grouping available: surface_hit is null and target_region is missing for spray frames.")
    total_spray_on_frames = sum(session.spray_on_frame_count for session in selected_sessions)
    if total_spray_on_frames < MIN_SPRAY_ON_FRAMES:
        reasons.append(
            f"Too few spray_on frames across selected sessions: found {total_spray_on_frames}, need at least {MIN_SPRAY_ON_FRAMES}."
        )
    if any(session.is_truncated for session in selected_sessions):
        reasons.append("At least one selected session is a truncated sample, so burst coverage is incomplete.")
    if selected_sessions and all(session.skill_level is None for session in selected_sessions) and source_session_count < MIN_SESSION_COUNT:
        reasons.append("All skill labels are null and proxy ranking cannot rank a single session.")
    return reasons


def _require_proxy_score(session: LoadedSession) -> float:
    if session.proxy_score is None:
        raise ValueError("proxy_score must be available for ranked sessions")
    return session.proxy_score


def _select_sessions(sessions: list[LoadedSession]) -> tuple[list[LoadedSession], str, list[str]]:
    expert_sessions = [session for session in sessions if session.skill_level == "expert"]
    if expert_sessions:
        return expert_sessions, "expert", []
    ranked_candidates = [session for session in sessions if session.proxy_score is not None]
    if len(ranked_candidates) < MIN_SESSION_COUNT:
        return sessions, "proxy", ["All skill labels are null and proxy mode requires at least 2 rankable sessions."]
    ranked = sorted(ranked_candidates, key=_require_proxy_score, reverse=True)
    return ranked[: len(ranked)], "proxy", []


def _joint_names_for_handedness(handedness: str) -> list[str]:
    prefix = "Right" if handedness == "right" else "Left"
    return ["Spine", "Chest", f"{prefix}Shoulder", f"{prefix}UpperArm", f"{prefix}ForeArm", f"{prefix}Hand"]


def _extract_bursts(session: LoadedSession) -> list[BurstRecord]:
    bursts: list[BurstRecord] = []
    current: list[dict[str, Any]] = []
    for frame in session.result.sequence.frames:
        if bool(frame.get("spray_on")):
            current.append(frame)
            continue
        if current:
            burst = _build_burst_record(session, current)
            if burst is not None:
                bursts.append(burst)
            current = []
    if current:
        burst = _build_burst_record(session, current)
        if burst is not None:
            bursts.append(burst)
    return bursts


def _build_burst_record(session: LoadedSession, frames: list[dict[str, Any]]) -> BurstRecord | None:
    region_ids = [_frame_region_id(frame) for frame in frames]
    valid_region_ids = [region_id for region_id in region_ids if region_id is not None]
    if not valid_region_ids or session.handedness is None:
        return None
    region_id = Counter(valid_region_ids).most_common(1)[0][0]
    region_frames = [frame for frame in frames if _frame_region_id(frame) == region_id]
    if not region_frames:
        return None
    return BurstRecord(
        session_path=session.path,
        region_id=region_id,
        handedness=session.handedness,
        frames=region_frames,
    )


def _resample_vectors(vectors: list[list[float]], target_length: int) -> list[list[float]]:
    if not vectors:
        return []
    if len(vectors) == target_length:
        return [list(vector) for vector in vectors]
    if len(vectors) == 1:
        return [list(vectors[0]) for _ in range(target_length)]
    resampled: list[list[float]] = []
    scale = (len(vectors) - 1) / (target_length - 1) if target_length > 1 else 0.0
    for target_index in range(target_length):
        position = target_index * scale
        last_index = len(vectors) - 1
        lower_index = min(int(math.floor(position)), last_index)
        upper_index = min(int(math.ceil(position)), last_index)
        if lower_index == upper_index:
            resampled.append(list(vectors[lower_index]))
            continue
        weight = position - lower_index
        lower_vector = vectors[lower_index]
        upper_vector = vectors[upper_index]
        resampled.append(
            [
                lower_vector[axis] + (upper_vector[axis] - lower_vector[axis]) * weight
                for axis in range(3)
            ]
        )
    return resampled


def _summarize_distribution(samples: list[list[float]]) -> dict[str, list[float]]:
    means = [mean([sample[axis] for sample in samples]) for axis in range(3)]
    stds = []
    for axis in range(3):
        axis_values = [sample[axis] for sample in samples]
        axis_mean = mean(axis_values)
        stds.append(math.sqrt(mean([(value - axis_mean) ** 2 for value in axis_values])))
    return {"mean": means, "std": stds}


def _build_pose_payload(level: str, sessions: list[LoadedSession], reference_quality: str) -> dict[str, Any]:
    bursts: list[BurstRecord] = []
    for session in sessions:
        bursts.extend(_extract_bursts(session))

    regions: dict[str, list[BurstRecord]] = {}
    for burst in bursts:
        regions.setdefault(burst.region_id, []).append(burst)

    region_payloads: dict[str, Any] = {}
    for region_id, region_bursts in sorted(regions.items()):
        normalized_frame_count = max(len(burst.frames) for burst in region_bursts)
        sample_count = len(region_bursts)
        joint_payloads: dict[str, list[dict[str, list[float]]]] = {}
        handedness = region_bursts[0].handedness
        joint_names = _joint_names_for_handedness(handedness)
        for joint_name in joint_names:
            per_burst_series = [
                _resample_vectors(
                    [frame["skeleton"]["joints_rot_euler_deg"][joint_name] for frame in burst.frames],
                    normalized_frame_count,
                )
                for burst in region_bursts
            ]
            frame_distributions = []
            for frame_index in range(normalized_frame_count):
                frame_samples = [series[frame_index] for series in per_burst_series]
                frame_distributions.append(_summarize_distribution(frame_samples))
            joint_payloads[joint_name] = frame_distributions
        region_payloads[region_id] = {
            "sample_count": sample_count,
            "normalized_frame_count": normalized_frame_count,
            "joint_rotation_distributions": joint_payloads,
        }

    return {
        "level": level,
        "status": "ok",
        "reference_quality": reference_quality,
        "session_count": len(sessions),
        "regions": region_payloads,
    }


def _build_flow_payload(level: str, sessions: list[LoadedSession], reference_quality: str) -> dict[str, Any]:
    edge_counter: Counter[tuple[str, str]] = Counter()
    node_counter: Counter[str] = Counter()
    for session in sessions:
        burst_regions = [burst.region_id for burst in _extract_bursts(session)]
        compact_regions: list[str] = []
        for region_id in burst_regions:
            if not compact_regions or compact_regions[-1] != region_id:
                compact_regions.append(region_id)
        for region_id in compact_regions:
            node_counter[region_id] += 1
        for source, target in zip(compact_regions, compact_regions[1:]):
            edge_counter[(source, target)] += 1
    return {
        "level": level,
        "status": "ok",
        "reference_quality": reference_quality,
        "session_count": len(sessions),
        "nodes": [
            {"region_id": region_id, "frequency": frequency}
            for region_id, frequency in sorted(node_counter.items())
        ],
        "edges": [
            {"source": source, "target": target, "frequency": frequency}
            for (source, target), frequency in sorted(edge_counter.items())
        ],
    }


def build_reference_artifacts(session_paths: Sequence[str | Path], level: str, out_dir: str | Path) -> BuildArtifactsResult:
    paths = [Path(path) for path in session_paths]
    loaded_sessions = [_load_session(path) for path in paths]
    selected_sessions, reference_quality, selection_reasons = _select_sessions(loaded_sessions)
    reasons = selection_reasons + _collect_reasons(level, selected_sessions, len(paths))
    if reasons:
        pose_payload = _insufficient_payload(level, "pose", len(paths), reasons)
        flow_payload = _insufficient_payload(level, "flow", len(paths), reasons)
    else:
        pose_payload = _build_pose_payload(level, selected_sessions, reference_quality)
        flow_payload = _build_flow_payload(level, selected_sessions, reference_quality)

    output_path = Path(out_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "expert_pose_reference.json").write_text(
        json.dumps(pose_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_path / "expert_flow.json").write_text(
        json.dumps(flow_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return BuildArtifactsResult(pose_payload=pose_payload, flow_payload=flow_payload, reasons=reasons)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build per-level expert reference artifacts from parsed session exports.")
    parser.add_argument("--sessions", nargs="+", required=True, help="Session file paths or glob patterns.")
    parser.add_argument("--level", required=True, help="Target level identifier.")
    parser.add_argument("--out", type=Path, required=True, help="Output directory for reference artifacts.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")
    session_paths = _expand_session_paths(args.sessions)
    if not session_paths:
        logger.error("No session files matched --sessions inputs")
        return 1
    try:
        result = build_reference_artifacts(session_paths, level=str(args.level), out_dir=args.out)
    except Exception:
        logger.exception("Failed to build expert reference artifacts")
        return 1

    if result.reasons:
        print("insufficient_data")
        for reason in result.reasons:
            print(f"- {reason}")
    else:
        print(f"Built expert references for level {args.level} ({result.pose_payload['reference_quality']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
