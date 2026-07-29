#!/usr/bin/env python3
"""Assemble session/sequence JSON documents from raw simulator CSV frames.

The 0.3.0 JSON export is a lossy derivative of the CSV: it drops the ``Ray``
columns and emits ``surface_hit: null``, which starves every region-aware
consumer. This converter reads the CSV directly so ``surface_hit`` survives.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Final, NamedTuple

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from trainer.scripts.simulator_csv import (
    JOINT_COLUMNS,
    JOINT_KEYS,
    LIMB_COLUMNS,
    OBJECT_KEYS,
    frame_from_row,
    read_rows,
    read_settings,
    vec3,
)

logger = logging.getLogger(__name__)

SCHEMA_VERSION: Final = "0.4.0"


@dataclass(frozen=True, slots=True)
class WorkerProfile:
    skill_level: str | None
    handedness: str | None


@dataclass(frozen=True, slots=True)
class ConvertedSession:
    stem: str
    session_index: dict[str, Any]
    sequences: list[dict[str, Any]]


@dataclass(frozen=True, slots=True)
class _SessionContext:
    csv_path: Path
    worker: WorkerProfile
    bind_pose_row: dict[str, str]


class _Span(NamedTuple):
    sequence_start: int
    burst_start: int
    burst_end: int
    sequence_end: int


def _burst_spans(frames: list[dict[str, Any]]) -> list[_Span]:
    """Partition the recording so that every frame belongs to exactly one sequence.

    A sequence carries the approach frames preceding its burst, matching the
    vendor's ``frame_range`` (approach included) vs ``spray_range`` (trigger on),
    and the final sequence runs to the end of the recording.
    """
    spans: list[_Span] = []
    sequence_start = 0
    burst_start: int | None = None
    for position, frame in enumerate(frames):
        if frame["spray_on"]:
            if burst_start is None:
                burst_start = position
        elif burst_start is not None:
            spans.append(_Span(sequence_start, burst_start, position - 1, position - 1))
            sequence_start = position
            burst_start = None
    if burst_start is not None:
        spans.append(_Span(sequence_start, burst_start, len(frames) - 1, len(frames) - 1))
    if spans:
        spans[-1] = spans[-1]._replace(sequence_end=len(frames) - 1)
    return spans


def _sequence_document(frames: list[dict[str, Any]], seq_id: int, span: _Span) -> dict[str, Any]:
    burst_start, burst_end = span.burst_start, span.burst_end
    window = frames[span.sequence_start : span.sequence_end + 1]
    return {
        "schema_version": SCHEMA_VERSION,
        "sequence": {
            "seq_id": seq_id,
            "source_data_id": None,
            "definition": "spray_trigger_on_burst",
            "frame_range": {"start_i": window[0]["i"], "end_i": window[-1]["i"]},
            "spray_range": {
                "start_i": frames[burst_start]["i"],
                "end_i": frames[burst_end]["i"],
                "t_start_sec": frames[burst_start]["t_sec"],
                "t_end_sec": frames[burst_end]["t_sec"],
                "duration_sec": frames[burst_end]["t_sec"] - frames[burst_start]["t_sec"],
            },
            "t_start_sec": window[0]["t_sec"],
            "t_end_sec": window[-1]["t_sec"],
            "note": None,
        },
        "stats": {
            "spray_frame_count": sum(1 for frame in window if frame["spray_on"]),
            "texture_map_ids": sorted({
                frame["surface_hit"]["map_number"]
                for frame in window
                if frame["surface_hit"] is not None
            }),
        },
        "frames": window,
        "settings_override": None,
    }


def _sampling(frames: list[dict[str, Any]]) -> dict[str, Any]:
    if len(frames) < 2:
        return {"fixed_rate": False, "dt_mean_sec": 0.0, "dt_min_sec": 0.0, "dt_max_sec": 0.0}
    deltas = [frames[i]["t_sec"] - frames[i - 1]["t_sec"] for i in range(1, len(frames))]
    return {
        "fixed_rate": False,
        "dt_mean_sec": mean(deltas),
        "dt_min_sec": min(deltas),
        "dt_max_sec": max(deltas),
    }


def _session_index(
    frames: list[dict[str, Any]], sequences: list[dict[str, Any]], context: _SessionContext
) -> dict[str, Any]:
    level, _, remainder = context.csv_path.stem.partition("_")
    session_id, _, data_id = remainder.partition("_")
    return {
        "schema_version": SCHEMA_VERSION,
        "source": {
            "level": level,
            "session_id": session_id,
            "data_files": [{"data_id": data_id, "file": context.csv_path.name}],
            "exporter": {"sw_name": None, "sw_version": None},
        },
        "conventions": {
            "position_unit": "m",
            "rotation": {"type": "euler_deg", "order": None, "engine": "Unity"},
            "distance_unit": "mm",
            "speed_unit": "mm/s",
            "time_origin": "recording_start_per_data_file",
            "coordinate_frame": None,
            "skeleton_frame": None,
        },
        "capture": {
            "sampling": _sampling(frames),
            "tracked_objects": [{"key": key} for key in OBJECT_KEYS.values()],
            "active_gun": None,
        },
        "settings": read_settings(context.csv_path),
        "worker": {
            "worker_id": None,
            "skill_level": context.worker.skill_level,
            "height_mm": None,
            "handedness": context.worker.handedness,
        },
        "skeleton_def": {
            "joint_count": len(JOINT_COLUMNS),
            "joints": [JOINT_KEYS[column] for column in JOINT_COLUMNS],
            "hierarchy": None,
            "bind_pose_local_pos_m": {
                JOINT_KEYS[column]: vec3(context.bind_pose_row, column, "Position")
                for column in LIMB_COLUMNS
            },
        },
        "sequences": [
            {
                "seq_id": item["sequence"]["seq_id"],
                "frame_count": len(item["frames"]),
                "spray_frame_count": item["stats"]["spray_frame_count"],
                "texture_map_ids": item["stats"]["texture_map_ids"],
            }
            for item in sequences
        ],
        "totals": {
            "sequence_count": len(sequences),
            "frame_count": len(frames),
            "duration_sec": frames[-1]["t_sec"] - frames[0]["t_sec"],
            "total_spray_time_sec": sum(
                item["sequence"]["spray_range"]["duration_sec"] for item in sequences
            ),
        },
    }


def convert_csv(csv_path: str | Path, worker: WorkerProfile) -> ConvertedSession:
    path = Path(csv_path)
    rows = read_rows(path)
    frames = [frame_from_row(row) for row in rows]
    stem = path.stem.rsplit("_", 1)[0]
    sequences = [
        _sequence_document(frames, seq_id, span)
        for seq_id, span in enumerate(_burst_spans(frames), start=1)
    ]
    for sequence in sequences:
        sequence["session_ref"] = {"session_file": f"{stem}_session.json"}
    context = _SessionContext(csv_path=path, worker=worker, bind_pose_row=rows[0])
    return ConvertedSession(
        stem=stem,
        session_index=_session_index(frames, sequences, context),
        sequences=sequences,
    )


def write_session(converted: ConvertedSession, out_dir: str | Path) -> list[Path]:
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{converted.stem}_session.json").write_text(
        json.dumps(converted.session_index, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    paths: list[Path] = []
    for sequence in converted.sequences:
        path = directory / f"{converted.stem}_seq{sequence['sequence']['seq_id']:03d}.json"
        path.write_text(json.dumps(sequence, ensure_ascii=False, indent=2), encoding="utf-8")
        paths.append(path)
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a raw simulator CSV export into session/sequence JSON.")
    parser.add_argument("--csv", type=Path, required=True, help="Raw simulator CSV path.")
    parser.add_argument("--out", type=Path, required=True, help="Output directory.")
    parser.add_argument("--skill-level", default=None, choices=["expert", "intermediate", "novice"])
    parser.add_argument("--handedness", default=None, choices=["left", "right"])
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")
    if not args.csv.is_file():
        logger.error("CSV not found: %s", args.csv)
        return 1
    converted = convert_csv(args.csv, WorkerProfile(args.skill_level, args.handedness))
    paths = write_session(converted, args.out)
    logger.info("Wrote %d sequence(s) for %s", len(paths), converted.stem)
    return 0


if __name__ == "__main__":
    sys.exit(main())
