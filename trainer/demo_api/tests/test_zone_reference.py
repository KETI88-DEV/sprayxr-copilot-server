from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image

from trainer.zone_reference import ZoneReferenceCatalog


def _reference(root: Path) -> Path:
    (root / "color").mkdir(parents=True)
    with (root / "palette.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["zone_id", "hex", "r", "g", "b"])
        writer.writerow([1, "#FF0000", 255, 0, 0])
        writer.writerow([2, "#00FF00", 0, 255, 0])
    with (root / "scenario_map.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["scenario_id", "model_key", "note"])
        writer.writerow([4, "Learn04", "학습"])
        writer.writerow(["Unit8", "", "Unit (Zone 없음)"])
    with (root / "zone_rects.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "model_key", "map_num", "zone_id",
                "rect_x", "rect_y", "rect_w", "rect_h",
                "px_x", "px_y", "px_w", "px_h",
                "u_min", "v_min", "u_max", "v_max", "area_percent",
            ]
        )
        writer.writerow(
            ["Learn04", "01", 1, 0, 0, 1, 1, 0, 0, 4, 4, 0, 0, 1, 1, 50]
        )
        writer.writerow(
            ["Learn04", "01", 2, 0, 0, 1, 1, 0, 0, 4, 4, 0, 0, 1, 1, 50]
        )
    label = Image.new("RGB", (4, 4), (0, 0, 0))
    for y in range(4):
        for x in range(4):
            label.putpixel((x, y), (255, 0, 0) if x == y else (0, 255, 0))
    label.save(root / "color" / "Learn04_01.png")
    return root


def test_zone_reference_resolves_scenario_and_uses_map_scoped_zone_keys(tmp_path: Path) -> None:
    catalog = ZoneReferenceCatalog(_reference(tmp_path))

    assert catalog.resolve_model("4") == "Learn04"
    assert catalog.resolve_model("Unit8") is None
    zone = catalog.zone("Learn04", 1, "1")
    assert zone is not None
    assert (zone.model_key, zone.map_num, zone.zone_id) == ("Learn04", 1, "1")
    assert zone.rgb == (255, 0, 0)


def test_zone_reference_returns_exact_non_rectangular_mask(tmp_path: Path) -> None:
    catalog = ZoneReferenceCatalog(_reference(tmp_path))

    mask = catalog.zone_mask("Learn04", 1, "1", size=(4, 4))

    assert mask is not None
    values = (
        mask.get_flattened_data()
        if hasattr(mask, "get_flattened_data")
        else mask.getdata()
    )
    assert list(values).count(255) == 4
    assert mask.getpixel((0, 0)) == 255
    assert mask.getpixel((1, 0)) == 0


def test_bundled_zone_reference_matches_tsb_delivery() -> None:
    catalog = ZoneReferenceCatalog.default()

    assert catalog.resolve_model("1") == "Learn01"
    assert catalog.resolve_model("10") == "Learn10"
    assert catalog.resolve_model("Unit16") == "Exam2"
    assert catalog.resolve_model("Unit8") is None
    assert len(catalog.zones("Learn04", 1)) == 13
    assert catalog.zone("Learn02", 1, "1") is not None
    assert catalog.zone("Learn02", 1, "3") is not None
