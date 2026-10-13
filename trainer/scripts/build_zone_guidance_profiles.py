#!/usr/bin/env python3
"""Build deterministic speed/distance Zone profiles from organized expert CSVs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


def _number(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) and parsed >= 0 else None


def _quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _distribution(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "median": statistics.median(values),
        "q1": _quantile(values, 0.25),
        "q3": _quantile(values, 0.75),
    }


def _stroke_direction(points: list[tuple[float, float]]) -> str | None:
    if len(points) < 2:
        return None
    x_travel = sum(abs(b[0] - a[0]) for a, b in zip(points, points[1:]))
    y_travel = sum(abs(b[1] - a[1]) for a, b in zip(points, points[1:]))
    if x_travel + y_travel < 5.0:
        return "stationary"
    displacement = math.dist(points[0], points[-1])
    path_length = sum(math.dist(a, b) for a, b in zip(points, points[1:]))
    if x_travel >= y_travel * 1.5:
        return "horizontal"
    if y_travel >= x_travel * 1.5:
        return "vertical"
    return "diagonal" if path_length and displacement / path_length >= 0.5 else "curved_irregular"


def _column(fieldnames: list[str], name: str) -> str | None:
    normalized = {field.strip().lower(): field for field in fieldnames}
    return normalized.get(name.strip().lower())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_session(csv_path: Path, mode_root: Path) -> dict[str, Any] | None:
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = list(reader.fieldnames or [])
        spray_col = _column(fields, "분사")
        speed_col = _column(fields, "Speed (Real)") or _column(fields, "Speed (Input)")
        distance_col = _column(fields, "Distance (Real)") or _column(fields, "Distance (Input)")
        zone_col = _column(fields, "Zone ID")
        map_col = _column(fields, "Ray Texture Map Number")
        uv_x_col = _column(fields, "Ray Texture UV X")
        uv_y_col = _column(fields, "Ray Texture UV Y")
        if not all((spray_col, speed_col, distance_col, zone_col)):
            return None

        zones: dict[str, dict[str, Any]] = {}
        order: list[dict[str, Any]] = []
        seen: set[str] = set()
        active_key: str | None = None
        active_points: list[tuple[float, float]] = []
        directions: dict[str, list[str]] = {}

        def finish_stroke() -> None:
            nonlocal active_key, active_points
            direction = _stroke_direction(active_points)
            if active_key is not None and direction is not None:
                directions.setdefault(active_key, []).append(direction)
            active_key = None
            active_points = []

        for row in reader:
            spray = str(row.get(spray_col, "")).strip().lower()
            if spray not in {"on", "1", "true"}:
                finish_stroke()
                continue
            zone_id = str(row.get(zone_col, "")).strip()
            if not zone_id or zone_id in {"-1", "0", "none", "nan"}:
                continue
            map_value = _number(row.get(map_col)) if map_col is not None else None
            map_num = int(map_value) if map_value is not None and map_value >= 1 else 1
            key = f"{map_num}:{zone_id}"
            if active_key != key:
                finish_stroke()
                active_key = key
            x = _number(row.get(uv_x_col)) if uv_x_col is not None else None
            y = _number(row.get(uv_y_col)) if uv_y_col is not None else None
            if x is not None and y is not None:
                active_points.append((x, y))
            speed = _number(row.get(speed_col))
            distance = _number(row.get(distance_col))
            zone = zones.setdefault(
                key,
                {"map_num": map_num, "zone_id": zone_id, "speed": [], "distance": []},
            )
            if speed is not None:
                zone["speed"].append(speed)
            if distance is not None:
                zone["distance"].append(distance)
            if key not in seen:
                seen.add(key)
                order.append({"map_num": map_num, "zone_id": zone_id})
        finish_stroke()

    usable = {
        key: {
            "map_num": value["map_num"],
            "zone_id": value["zone_id"],
            "speed_mms": statistics.median(value["speed"]) if value["speed"] else None,
            "distance_mm": statistics.median(value["distance"]) if value["distance"] else None,
            "expected_direction": Counter(directions.get(key, [])).most_common(1)[0][0] if directions.get(key) else None,
        }
        for key, value in zones.items()
        if value["speed"] or value["distance"]
    }
    if not usable:
        return None
    session_dir = csv_path.parent
    heightmap_paths = sorted(set(session_dir.glob("*_Height.png")) | set(session_dir.glob("Map*.png")))
    heightmaps = [
        {
            "source_path": path.relative_to(mode_root).as_posix(),
            "sha256": _sha256(path),
            "analysis_status": "not_evaluated",
        }
        for path in heightmap_paths
    ]
    return {
        "session_id": csv_path.stem,
        "sources": {
            "csv_path": csv_path.relative_to(mode_root).as_posix(),
            "csv_sha256": _sha256(csv_path),
        },
        "heightmaps": heightmaps,
        "analysis": {"macro_order": order, "zones": usable},
    }


def _profile(process: str, mode_id: int, mode_root: Path, sessions: list[dict[str, Any]]) -> dict[str, Any]:
    zone_samples: dict[str, dict[str, Any]] = {}
    orders: Counter[str] = Counter()
    order_values: dict[str, list[dict[str, Any]]] = {}
    transitions: Counter[tuple[str, str]] = Counter()
    for session in sessions:
        order = session["analysis"]["macro_order"]
        serialized = json.dumps(order, ensure_ascii=False, sort_keys=True)
        orders[serialized] += 1
        order_values[serialized] = order
        keys = [f"{item['map_num']}:{item['zone_id']}" for item in order]
        transitions.update(zip(keys, keys[1:]))
        for key, zone in session["analysis"]["zones"].items():
            aggregate = zone_samples.setdefault(
                key,
                {"map_num": zone["map_num"], "zone_id": zone["zone_id"], "speed": [], "distance": [], "directions": []},
            )
            if zone["speed_mms"] is not None:
                aggregate["speed"].append(zone["speed_mms"])
            if zone["distance_mm"] is not None:
                aggregate["distance"].append(zone["distance_mm"])
            if zone.get("expected_direction") is not None:
                aggregate["directions"].append(zone["expected_direction"])

    zones = {
        key: {
            "map_num": value["map_num"],
            "zone_id": value["zone_id"],
            "session_count": max(len(value["speed"]), len(value["distance"])),
            "speed_mms": _distribution(value["speed"]) if value["speed"] else {},
            "distance_mm": _distribution(value["distance"]) if value["distance"] else {},
            "expected_direction": Counter(value["directions"]).most_common(1)[0][0] if value["directions"] else None,
        }
        for key, value in sorted(zone_samples.items())
    }
    total = len(sessions)
    macro_orders = [
        {"order": order_values[key], "session_count": count, "fraction": count / total}
        for key, count in orders.most_common()
    ]
    transition_items = []
    for (source, target), count in sorted(transitions.items()):
        source_map, source_zone = source.split(":", 1)
        target_map, target_zone = target.split(":", 1)
        transition_items.append(
            {
                "from": {"map_num": int(source_map), "zone_id": source_zone},
                "to": {"map_num": int(target_map), "zone_id": target_zone},
                "session_count": count,
                "fraction": count / total,
            }
        )
    return {
        "schema_version": 1,
        "identity": {
            "content_name": mode_root.name,
            "content_type": "learning",
            "mode_id": mode_id,
            "process": process,
        },
        "quality": {
            "status": "multi_session" if total >= 2 else "single_session",
            "session_count": total,
            "heightmap_scale_status": "not_evaluated",
        },
        "aggregate": {"zones": zones, "macro_orders": macro_orders, "transitions": transition_items},
        "sessions": sessions,
    }


def build_profiles(reference_root: Path, *, process: str) -> list[Path]:
    root = Path(reference_root)
    raw = root / "raw" / process / "learning"
    output_root = root / "generated" / "profiles" / process / "learning"
    output_root.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for mode_root in sorted(raw.glob("mode_*")):
        try:
            mode_id = int(mode_root.name.rsplit("_", 1)[-1])
        except ValueError:
            continue
        sessions = [
            session
            for path in sorted(mode_root.rglob("*.csv"))
            if (session := _load_session(path, mode_root))
        ]
        if not sessions:
            continue
        target = output_root / f"mode_{mode_id:02d}.json"
        temporary = target.with_suffix(f".json.tmp-{os.getpid()}")
        temporary.write_text(
            json.dumps(
                _profile(process, mode_id, mode_root, sessions),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        temporary.replace(target)
        outputs.append(target)

    audit_root = root / "generated" / "audits"
    audit_root.mkdir(parents=True, exist_ok=True)
    audit = {
        "schema_version": 1,
        "input": {
            "process": process,
            "content_type": "learning",
            "raw_root": f"raw/{process}/learning",
        },
        "outputs": [path.relative_to(root / "generated").as_posix() for path in outputs],
    }
    (audit_root / f"{process}_learning.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--process", choices=("spray", "blasting"), required=True)
    args = parser.parse_args()
    outputs = build_profiles(args.reference_root, process=args.process)
    print(json.dumps({"process": args.process, "profiles": len(outputs)}, ensure_ascii=False))
    return 0 if outputs else 1


if __name__ == "__main__":
    raise SystemExit(main())
