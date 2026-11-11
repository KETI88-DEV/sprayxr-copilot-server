from __future__ import annotations

from trainer.zone_engine.assets import ZoneAssetCatalog


def test_exact_color_mask_is_not_bounding_rectangle(
    asset_catalog: ZoneAssetCatalog,
) -> None:
    zone_one = asset_catalog.zone_mask("Learn01", 1, "1")
    zone_two = asset_catalog.zone_mask("Learn01", 1, "2")
    assert zone_one.values.shape == (4, 4)
    assert int(zone_one.values.sum()) == 1
    assert int(zone_two.values.sum()) == 9
    assert not bool((zone_one.values & zone_two.values).any())
    assert zone_one.sha256 != zone_two.sha256


def test_uv_pixels_use_bottom_left_origin_and_never_clamp(
    asset_catalog: ZoneAssetCatalog,
) -> None:
    # PNG red is at top-left row 0, so bottom-origin y=3 selects it.
    assert asset_catalog.zone_at_uv("Learn01", 1, 0.2, 3.2) == "1"
    assert asset_catalog.zone_at_uv("Learn01", 1, 0.2, 0.2) is None
    assert asset_catalog.zone_at_uv("Learn01", 1, -0.1, 3.0) is None
    assert asset_catalog.zone_at_uv("Learn01", 1, 4.0, 3.0) is None
    assert asset_catalog.zone_at_uv("Learn01", 1, 0.0, 4.0) is None
