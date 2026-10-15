"""TSB 공식 Zone 라벨맵과 시나리오 매핑을 읽는 기준 데이터 모듈."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class ZoneDefinition:
    model_key: str
    map_num: int
    zone_id: str
    rect: tuple[float, float, float, float]
    pixel_rect: tuple[int, int, int, int]
    uv_bounds: tuple[float, float, float, float]
    area_percent: float
    rgb: tuple[int, int, int]
    hex_color: str


class ZoneReferenceCatalog:
    """Read-only access to the vendor supplied Zone definitions.

    Label images use PNG top-left coordinates. ``zone_mask`` keeps that
    orientation so it can be applied directly to the supplied height maps.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self._palette = self._load_palette()
        self._scenarios = self._load_scenarios()
        self._zones = self._load_zones()

    @classmethod
    @lru_cache(maxsize=1)
    def default(cls) -> "ZoneReferenceCatalog":
        root = Path(__file__).resolve().parent / "demo_api" / "assets" / "zone_reference"
        return cls(root)

    def _load_palette(self) -> dict[str, tuple[str, tuple[int, int, int]]]:
        output: dict[str, tuple[str, tuple[int, int, int]]] = {}
        with (self.root / "palette.csv").open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                zone_id = str(row["zone_id"]).strip()
                output[zone_id] = (
                    str(row["hex"]).strip(),
                    (int(row["r"]), int(row["g"]), int(row["b"])),
                )
        return output

    def _load_scenarios(self) -> dict[str, str | None]:
        output: dict[str, str | None] = {}
        with (self.root / "scenario_map.csv").open(
            encoding="utf-8-sig", newline=""
        ) as stream:
            for row in csv.DictReader(stream):
                value = str(row.get("model_key", "")).strip()
                output[str(row["scenario_id"]).strip()] = value or None
        return output

    def _load_zones(self) -> dict[tuple[str, int, str], ZoneDefinition]:
        output: dict[tuple[str, int, str], ZoneDefinition] = {}
        with (self.root / "zone_rects.csv").open(
            encoding="utf-8-sig", newline=""
        ) as stream:
            for row in csv.DictReader(stream):
                model_key = str(row["model_key"]).strip()
                map_num = int(row["map_num"])
                zone_id = str(row["zone_id"]).strip()
                color = self._palette[zone_id]
                definition = ZoneDefinition(
                    model_key=model_key,
                    map_num=map_num,
                    zone_id=zone_id,
                    rect=tuple(float(row[key]) for key in ("rect_x", "rect_y", "rect_w", "rect_h")),
                    pixel_rect=tuple(int(row[key]) for key in ("px_x", "px_y", "px_w", "px_h")),
                    uv_bounds=tuple(float(row[key]) for key in ("u_min", "v_min", "u_max", "v_max")),
                    area_percent=float(row["area_percent"]),
                    rgb=color[1],
                    hex_color=color[0],
                )
                output[(model_key, map_num, zone_id)] = definition
        return output

    def resolve_model(self, scenario_id: str | int) -> str | None:
        return self._scenarios.get(str(scenario_id).strip())

    def zone(
        self, model_key: str, map_num: int | str, zone_id: str | int
    ) -> ZoneDefinition | None:
        return self._zones.get((str(model_key), int(map_num), str(zone_id)))

    def zones(self, model_key: str, map_num: int | str) -> tuple[ZoneDefinition, ...]:
        prefix = (str(model_key), int(map_num))
        return tuple(
            value
            for key, value in sorted(
                self._zones.items(), key=lambda item: (item[0][0], item[0][1], int(item[0][2]))
            )
            if key[:2] == prefix
        )

    @lru_cache(maxsize=32)
    def _label_map(self, model_key: str, map_num: int) -> Image.Image:
        path = self.root / "color" / f"{model_key}_{map_num:02d}.png"
        return Image.open(path).convert("RGB")

    @lru_cache(maxsize=32)
    def _native_mask(self, model_key: str, map_num: int, zone_id: str) -> Image.Image | None:
        definition = self.zone(model_key, map_num, zone_id)
        if definition is None:
            return None
        label = self._label_map(model_key, map_num)
        pixels = np.asarray(label, dtype=np.uint8)
        color = np.asarray(definition.rgb, dtype=np.uint8)
        return Image.fromarray(np.all(pixels == color, axis=2).astype(np.uint8) * 255)

    def zone_mask(
        self,
        model_key: str,
        map_num: int | str,
        zone_id: str | int,
        *,
        size: tuple[int, int] | None = None,
    ) -> Image.Image | None:
        mask = self._native_mask(str(model_key), int(map_num), str(zone_id))
        if mask is None:
            return None
        if size is None or mask.size == size:
            return mask.copy()
        return mask.resize(size, Image.Resampling.NEAREST)

    @lru_cache(maxsize=32)
    def _label_ids(self, model_key: str, map_num: int) -> np.ndarray:
        """라벨맵 화소마다 구역 번호(정수, 구역 아님 = 0). 좌상단 원점."""
        pixels = np.asarray(self._label_map(model_key, map_num), dtype=np.int64)
        codes = pixels[..., 0] << 16 | pixels[..., 1] << 8 | pixels[..., 2]
        ids = np.zeros(codes.shape, dtype=np.int16)
        for zone_id, (_, (r, g, b)) in self._palette.items():
            ids[codes == (r << 16 | g << 8 | b)] = int(zone_id)
        return ids

    def zone_at(self, model_key: str, map_num: int | str, u: float, v: float) -> str | None:
        """착탄 UV(텍스처 픽셀, 좌하단 원점 — TSB CSV `Ray Texture UV X/Y`)가 든 구역. 구역 밖·라벨맵 없음은 None."""
        try:
            ids = self._label_ids(str(model_key), int(map_num))
        except (OSError, ValueError):
            return None
        height, width = ids.shape
        x, y = int(u), height - 1 - int(v)
        if not (0 <= x < width and 0 <= y < height):
            return None
        zone = int(ids[y, x])
        return str(zone) if zone else None
