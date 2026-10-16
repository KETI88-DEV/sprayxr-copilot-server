#!/usr/bin/env python3
"""Build deterministic speed/distance Zone profiles from organized expert CSVs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from trainer.expert_zone_mapping import canonical_expert_zone_id
from trainer.zone_reference import ZoneReferenceCatalog
from PIL import Image, UnidentifiedImageError

_HEIGHTMAP_NUMBER = re.compile(r"Map(\d+)", re.IGNORECASE)
# 전문가 7명·479파일로 정한 레벨별 권장 순서와 구역 전이(`trainer.scripts.map_expert_zones`, 2026-10-16 결정).
# Spark 의 raw 전문가 자료는 반장 한 사람의 녹화라 "가장 많이 나온 순서"를 거기서 낼 수 없다.
EXPERT_ZONE_ORDERS = Path(__file__).resolve().parents[1] / "demo_api" / "assets" / "zone_reference" / "expert_zone_orders.json"
_COATING_SCALE_UM = 20000.0


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


def _heightmap_number(path: Path) -> int:
    match = _HEIGHTMAP_NUMBER.search(path.name)
    return int(match.group(1)) if match is not None else 1


def _uses_blue_validity(image: Image.Image) -> bool:
    """Whether this file follows the current TSB B8 validity convention."""
    return image.convert("RGB").getchannel("B").getbbox() is not None


def _cell_mean_um(
    image: Image.Image,
    gx: int,
    gy: int,
    grid_size: int,
    *,
    zone_mask: Image.Image | None = None,
    blue_is_validity: bool = True,
) -> float:
    """Return painted mean thickness; UV Y is bottom-up, PNG rows are top-down."""
    width, height = image.size
    left = math.floor(gx * width / grid_size)
    right = math.ceil((gx + 1) * width / grid_size)
    top = math.floor((grid_size - gy - 1) * height / grid_size)
    bottom = math.ceil((grid_size - gy) * height / grid_size)
    pixels = image.crop((left, top, right, bottom)).convert("RGB")
    mask_values = None
    if zone_mask is not None:
        cropped_mask = zone_mask.crop((left, top, right, bottom))
        mask_values = list(
            cropped_mask.get_flattened_data()
            if hasattr(cropped_mask, "get_flattened_data")
            else cropped_mask.getdata()
        )
    pixel_values = (
        pixels.get_flattened_data()
        if hasattr(pixels, "get_flattened_data")
        else pixels.getdata()
    )
    values = []
    for index, (red, green, blue) in enumerate(pixel_values):
        painted = blue > 0 if blue_is_validity else red > 0 or green > 0
        if not painted or (mask_values is not None and mask_values[index] <= 0):
            continue
        values.append((red / 255.0 + green / 65025.0) * _COATING_SCALE_UM)
    return statistics.fmean(values) if values else 0.0


def _session_coating_cells(
    zones: dict[str, dict[str, Any]],
    heightmap_paths: list[Path],
    *,
    grid_size: int,
    mode_id: int,
    zone_catalog: ZoneReferenceCatalog | None,
) -> dict[str, dict[str, Any]]:
    images: dict[int, Image.Image] = {}
    blue_validity: dict[int, bool] = {}
    try:
        for path in heightmap_paths:
            map_num = _heightmap_number(path)
            images[map_num] = Image.open(path).convert("RGB")
            blue_validity[map_num] = _uses_blue_validity(images[map_num])
        output: dict[str, dict[str, Any]] = {}
        for key, zone in zones.items():
            map_num = int(zone["map_num"])
            image = images.get(map_num)
            if image is None:
                continue
            width, height = image.size
            zone_mask = None
            model_key = zone_catalog.resolve_model(str(mode_id)) if zone_catalog else None
            if model_key:
                zone_mask = zone_catalog.zone_mask(
                    model_key, map_num, str(zone["zone_id"]), size=(width, height)
                )
            if zone_mask is not None and zone_mask.getbbox() is not None:
                left, top, right, bottom = zone_mask.getbbox()
                gx_start = max(0, math.floor(left * grid_size / width))
                gx_stop = min(grid_size, math.ceil(right * grid_size / width))
                gy_start = max(0, math.floor((height - bottom) * grid_size / height))
                gy_stop = min(grid_size, math.ceil((height - top) * grid_size / height))
                candidates = {
                    (gx, gy)
                    for gx in range(gx_start, gx_stop)
                    for gy in range(gy_start, gy_stop)
                }
                mask_source = "tsb_color_label_map"
            else:
                visited: set[tuple[int, int]] = set()
                for x, y in zone["uv_points"]:
                    if 0 <= x < width and 0 <= y < height:
                        visited.add(
                            (
                                min(grid_size - 1, int(x / width * grid_size)),
                                min(grid_size - 1, int(y / height * grid_size)),
                            )
                        )
                candidates = {
                    (nx, ny)
                    for gx, gy in visited
                    for nx in range(max(0, gx - 1), min(grid_size, gx + 2))
                    for ny in range(max(0, gy - 1), min(grid_size, gy + 2))
                }
                mask_source = "expert_raytexuv_fallback"
            cells = {}
            for gx, gy in sorted(candidates):
                mean_um = _cell_mean_um(
                    image,
                    gx,
                    gy,
                    grid_size,
                    zone_mask=zone_mask,
                    blue_is_validity=blue_validity.get(map_num, True),
                )
                if mean_um > 0:
                    cells[f"{gx}:{gy}"] = mean_um
            if cells:
                output[key] = {
                    "atlas_width": width,
                    "atlas_height": height,
                    "grid_size": grid_size,
                    "mask_source": mask_source,
                    "display_semantics": "physical_thickness_um",
                    "cells": cells,
                }
        return output
    except (OSError, UnidentifiedImageError, ValueError):
        return {}
    finally:
        for image in images.values():
            image.close()


def _load_session(
    csv_path: Path,
    mode_root: Path,
    *,
    process: str,
    mode_id: int,
    coating_grid_size: int,
    zone_catalog: ZoneReferenceCatalog | None,
) -> dict[str, Any] | None:
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
            zone_id = canonical_expert_zone_id(
                process, "learning", mode_id, row.get(zone_col, "")
            )
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
                {
                    "map_num": map_num,
                    "zone_id": zone_id,
                    "speed": [],
                    "distance": [],
                    "uv_points": [],
                },
            )
            if speed is not None:
                zone["speed"].append(speed)
            if distance is not None:
                zone["distance"].append(distance)
            uv_x = _number(row.get(uv_x_col)) if uv_x_col is not None else None
            uv_y = _number(row.get(uv_y_col)) if uv_y_col is not None else None
            if uv_x is not None and uv_y is not None:
                zone["uv_points"].append((uv_x, uv_y))
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
    coating_cells = (
        _session_coating_cells(
            zones,
            heightmap_paths,
            grid_size=coating_grid_size,
            mode_id=mode_id,
            zone_catalog=zone_catalog,
        )
        if process == "spray"
        else {}
    )
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
        "analysis": {
            "macro_order": order,
            "zones": usable,
            "coating_cells": coating_cells,
        },
    }


def _expert_orders(path: Path | None) -> dict[str, Any]:
    """레벨(scenario_id) → 권장 순서·전이. 파일이 없거나 깨졌으면 빈 dict(그때는 raw 세션에서 낸다)."""
    if path is None:
        return {}
    try:
        scenarios = json.loads(Path(path).read_text(encoding="utf-8"))["scenarios"]
        return scenarios if isinstance(scenarios, dict) else {}
    except (OSError, KeyError, TypeError, ValueError):
        return {}


def _apply_expert_orders(
    chosen: dict[str, Any], macro_orders: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """권장 순서를 맨 앞에 두고(엔진은 `macro_orders[0]`을 쓴다) 전이를 기준 자료의 것으로 바꾼다."""
    files = max(int(chosen["file_count"]), 1)
    order = {
        "order": [{"map_num": 1, "zone_id": str(zone)} for zone in chosen["recommended_flow"]],
        "session_count": int(chosen["flow_file_count"]),
        "fraction": int(chosen["flow_file_count"]) / files,
        "source": "expert_zone_orders",
        "decision": chosen.get("decision"),
    }
    transitions = [
        {
            "from": {"map_num": 1, "zone_id": str(item["from"])},
            "to": {"map_num": 1, "zone_id": str(item["to"])},
            "session_count": int(item["count"]),
            "fraction": int(item["count"]) / files,
            "source": "expert_zone_orders",
        }
        for item in chosen["transitions"]
    ]
    return [order, *macro_orders], transitions


def _profile(
    process: str,
    mode_id: int,
    mode_root: Path,
    sessions: list[dict[str, Any]],
    expert_orders: dict[str, Any] | None = None,
) -> dict[str, Any]:
    zone_samples: dict[str, dict[str, Any]] = {}
    orders: Counter[str] = Counter()
    order_values: dict[str, list[dict[str, Any]]] = {}
    transitions: Counter[tuple[str, str]] = Counter()
    coating_samples: dict[str, dict[str, Any]] = {}
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
        for key, coating in session["analysis"].get("coating_cells", {}).items():
            aggregate = coating_samples.setdefault(
                key,
                {
                    "atlas_width": coating["atlas_width"],
                    "atlas_height": coating["atlas_height"],
                    "grid_size": coating["grid_size"],
                    "mask_source": coating.get("mask_source", "expert_raytexuv_fallback"),
                    "display_semantics": coating.get(
                        "display_semantics", "physical_thickness_um"
                    ),
                    "session_count": 0,
                    "cells": {},
                },
            )
            if (
                aggregate["atlas_width"] != coating["atlas_width"]
                or aggregate["atlas_height"] != coating["atlas_height"]
                or aggregate["grid_size"] != coating["grid_size"]
            ):
                continue
            aggregate["session_count"] += 1
            for cell, value in coating["cells"].items():
                aggregate["cells"].setdefault(cell, []).append(value)

    zones = {}
    for key, value in sorted(zone_samples.items()):
        coating_source = coating_samples.get(key)
        if process != "spray":
            coating = {
                "analysis_status": "not_applicable",
                "reason": "coating_not_applicable",
            }
        elif coating_source is None:
            coating = {
                "analysis_status": "not_available",
                "reason": "heightmap_or_uv_unavailable",
            }
        else:
            minimum_samples = max(1, math.ceil(coating_source["session_count"] * 0.4))
            cells = {
                cell: {
                    "sample_count": len(samples),
                    "q1_um": _quantile(samples, 0.25),
                    "median_um": statistics.median(samples),
                    "q3_um": _quantile(samples, 0.75),
                }
                for cell, samples in sorted(coating_source["cells"].items())
                if len(samples) >= minimum_samples
            }
            coating = {
                "analysis_status": "available" if cells else "not_available",
                "atlas_width": coating_source["atlas_width"],
                "atlas_height": coating_source["atlas_height"],
                "grid_size": coating_source["grid_size"],
                "mask_source": coating_source["mask_source"],
                "display_semantics": coating_source["display_semantics"],
                "session_count": coating_source["session_count"],
                "cells": cells,
            }
            if not cells:
                coating["reason"] = "insufficient_consensus"
        zones[key] = {
            "map_num": value["map_num"],
            "zone_id": value["zone_id"],
            "session_count": max(len(value["speed"]), len(value["distance"])),
            "speed_mms": _distribution(value["speed"]) if value["speed"] else {},
            "distance_mm": _distribution(value["distance"]) if value["distance"] else {},
            "expected_direction": Counter(value["directions"]).most_common(1)[0][0] if value["directions"] else None,
            "coating": coating,
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
    chosen = (expert_orders or {}).get(str(mode_id)) if process == "spray" else None
    if chosen:
        macro_orders, transition_items = _apply_expert_orders(chosen, macro_orders)
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
            "heightmap_scale_status": (
                "rg16_scale_20000"
                if process == "spray" and coating_samples
                else "not_applicable" if process != "spray" else "not_available"
            ),
        },
        "aggregate": {"zones": zones, "macro_orders": macro_orders, "transitions": transition_items},
        "sessions": sessions,
    }


def build_profiles(
    reference_root: Path,
    *,
    process: str,
    coating_grid_size: int = 64,
    zone_reference_root: Path | None = None,
    expert_orders_path: Path | None = None,
) -> list[Path]:
    root = Path(reference_root)
    raw = root / "raw" / process / "learning"
    output_root = root / "generated" / "profiles" / process / "learning"
    output_root.mkdir(parents=True, exist_ok=True)
    zone_catalog: ZoneReferenceCatalog | None = None
    if process == "spray":
        reference_path = (
            Path(zone_reference_root)
            if zone_reference_root is not None
            else ZoneReferenceCatalog.default().root
        )
        try:
            zone_catalog = ZoneReferenceCatalog(reference_path)
        except (OSError, KeyError, TypeError, ValueError):
            zone_catalog = None
    expert_orders = _expert_orders(expert_orders_path)
    outputs: list[Path] = []
    for mode_root in sorted(raw.glob("mode_*")):
        try:
            mode_id = int(mode_root.name.rsplit("_", 1)[-1])
        except ValueError:
            continue
        sessions = [
            session
            for path in sorted(mode_root.rglob("*.csv"))
            if (
                session := _load_session(
                    path,
                    mode_root,
                    process=process,
                    mode_id=mode_id,
                    coating_grid_size=coating_grid_size,
                    zone_catalog=zone_catalog,
                )
            )
        ]
        if not sessions:
            continue
        target = output_root / f"mode_{mode_id:02d}.json"
        temporary = target.with_suffix(f".json.tmp-{os.getpid()}")
        temporary.write_text(
            json.dumps(
                _profile(process, mode_id, mode_root, sessions, expert_orders),
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
    parser.add_argument("--coating-grid-size", type=int, default=64)
    parser.add_argument("--zone-reference-root", type=Path)
    parser.add_argument(
        "--expert-orders", default=str(EXPERT_ZONE_ORDERS),
        help="레벨별 권장 순서·전이 JSON. 빈 문자열이면 raw 세션에서만 낸다.",
    )
    args = parser.parse_args()
    outputs = build_profiles(
        args.reference_root,
        process=args.process,
        coating_grid_size=args.coating_grid_size,
        zone_reference_root=args.zone_reference_root,
        expert_orders_path=Path(args.expert_orders) if args.expert_orders else None,
    )
    print(json.dumps({"process": args.process, "profiles": len(outputs)}, ensure_ascii=False))
    return 0 if outputs else 1


if __name__ == "__main__":
    raise SystemExit(main())
