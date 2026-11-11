from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from trainer.zone_engine.assets import ZoneAssetCatalog
from trainer.zone_engine.contracts import EventRequest, HeightPng


@pytest.fixture()
def asset_catalog(tmp_path: Path) -> ZoneAssetCatalog:
    root = tmp_path / "zone-assets"
    (root / "color").mkdir(parents=True)
    palette = root / "palette.csv"
    with palette.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["zone_id", "hex", "r", "g", "b"])
        writer.writerow([1, "#FF0000", 255, 0, 0])
        writer.writerow([2, "#00FF00", 0, 255, 0])

    label = Image.new("RGB", (4, 4), (0, 0, 0))
    label.putpixel((0, 0), (255, 0, 0))
    zone_two_pixels = [
        (1, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2), (3, 2),
    ]
    for point in zone_two_pixels:
        label.putpixel(point, (0, 255, 0))
    label_path = root / "color" / "Learn01_01.png"
    label.save(label_path)

    palette_sha = hashlib.sha256(palette.read_bytes()).hexdigest()
    label_sha = hashlib.sha256(label_path.read_bytes()).hexdigest()
    manifest = {
        "schema_version": "test-zone-assets/v1",
        "manifest_id": "test-zones-v1",
        "asset_root": ".",
        "palette": {"path": "palette.csv", "sha256": palette_sha},
        "label_maps": [
            {
                "model_key": "Learn01",
                "map_num": 1,
                "path": "color/Learn01_01.png",
                "sha256": label_sha,
                "width": 4,
                "height": 4,
            }
        ],
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    return ZoneAssetCatalog(manifest_path)


def height_png(
    *,
    width: int = 4,
    height: int = 4,
    rgb: tuple[int, int, int] = (2, 43, 255),
    encoding: str = "rg16_b8",
    basis: str = "current_round",
    x0: int = 0,
    y0: int = 0,
    atlas_width: int = 4,
    atlas_height: int = 4,
) -> dict[str, Any]:
    image = Image.new("RGB", (width, height), rgb)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    raw = buffer.getvalue()
    return {
        "data_base64": base64.b64encode(raw).decode("ascii"),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "encoding": encoding,
        "basis": basis,
        "width": width,
        "height": height,
        "origin": "top_left",
        "crop": {
            "x0": x0,
            "y0": y0,
            "width": width,
            "height": height,
            "atlas_width": atlas_width,
            "atlas_height": atlas_height,
        },
    }


def start_event(
    catalog: ZoneAssetCatalog,
    *,
    session_id: str = "session-1",
    event_id: str = "event-start",
    target_um: float | None = 160.0,
    expected_zone_ids: list[str] | None = None,
    round_index: int = 1,
    round_id: str = "round-1",
    family_id: str = "family-1",
) -> EventRequest:
    return EventRequest.model_validate(
        {
            "event_id": event_id,
            "event_type": "session_start",
            "session_id": session_id,
            "payload": {
                "source_instance_id": "test-source",
                "user_id": "user-1",
                "scenario_id": "1",
                "started_at": "2026-11-06T00:00:00Z",
                "evaluation_mode": "corrected_zone_v2",
                "contract_version": "sprayxr-corrected-zone/v2-draft-20261106",
                "target_thickness_um": target_um,
                "target_source": "test" if target_um is not None else None,
                "model_mapping": {
                    "provider_model_key": "Learn01",
                    "work_id": "work-1",
                    "session_family_id": family_id,
                    "round_index": round_index,
                    "round_id": round_id,
                    "geometry_model_key": "Learn01",
                },
                "zone_manifest": {
                    "manifest_id": catalog.manifest_id,
                    "sha256": catalog.manifest_sha256,
                    "model_key": "Learn01",
                    "map_nums": [1],
                },
                "expected_zones": (
                    [
                        {
                            "face_id": "face-1",
                            "map_num": 1,
                            "zone_ids": expected_zone_ids,
                        }
                    ]
                    if expected_zone_ids
                    else []
                ),
            },
        }
    )


def event(
    event_id: str, event_type: str, payload: dict[str, Any], session_id: str = "session-1"
) -> EventRequest:
    return EventRequest.model_validate(
        {
            "event_id": event_id,
            "event_type": event_type,
            "session_id": session_id,
            "payload": payload,
        }
    )
