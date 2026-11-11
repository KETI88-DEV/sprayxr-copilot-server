"""Hash-verified exact-color Zone assets and UV lookup."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .errors import EngineError


@dataclass(frozen=True)
class ZoneMask:
    model_key: str
    map_num: int
    zone_id: str
    values: np.ndarray
    sha256: str
    label_map_sha256: str

    @property
    def width(self) -> int:
        return int(self.values.shape[1])

    @property
    def height(self) -> int:
        return int(self.values.shape[0])


class ZoneAssetCatalog:
    """Read a pinned manifest without resizing or guessing missing faces."""

    def __init__(self, manifest_path: Path, *, asset_root: Path | None = None) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        raw = self.manifest_path.read_bytes()
        self.manifest_sha256 = hashlib.sha256(raw).hexdigest()
        self.manifest: dict[str, Any] = json.loads(raw)
        self.manifest_id = str(self.manifest["manifest_id"])
        declared_root = self.manifest_path.parent / str(self.manifest["asset_root"])
        self.asset_root = (Path(asset_root) if asset_root is not None else declared_root).resolve()
        self._palette = self._load_palette()
        self._maps: dict[tuple[str, int], dict[str, Any]] = {}
        for item in self.manifest["label_maps"]:
            key = (str(item["model_key"]), int(item["map_num"]))
            if key in self._maps:
                raise EngineError("asset_manifest_duplicate_map", f"duplicate map {key}")
            self._maps[key] = dict(item)

    @classmethod
    def default(cls) -> "ZoneAssetCatalog":
        return cls(Path(__file__).resolve().parent / "assets" / "zone_asset_manifest.json")

    def _asset_path(self, relative: str) -> Path:
        candidate = (self.asset_root / relative).resolve()
        if candidate != self.asset_root and self.asset_root not in candidate.parents:
            raise EngineError("asset_path_escape", "asset path escapes the manifest root")
        return candidate

    @staticmethod
    def _verify_file(path: Path, expected_sha256: str) -> bytes:
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise EngineError("asset_unavailable", f"cannot read pinned asset {path.name}") from exc
        actual = hashlib.sha256(raw).hexdigest()
        if actual != expected_sha256:
            raise EngineError("asset_hash_mismatch", f"pinned asset hash mismatch: {path.name}")
        return raw

    def _load_palette(self) -> dict[str, tuple[int, int, int]]:
        spec = self.manifest["palette"]
        path = self._asset_path(str(spec["path"]))
        raw = self._verify_file(path, str(spec["sha256"]))
        output: dict[str, tuple[int, int, int]] = {}
        text = raw.decode("utf-8-sig").splitlines()
        for row in csv.DictReader(text):
            zone_id = str(row["zone_id"]).strip()
            output[zone_id] = (int(row["r"]), int(row["g"]), int(row["b"]))
        if not output:
            raise EngineError("asset_palette_empty", "Zone palette is empty")
        return output

    def verify_session_reference(
        self,
        *,
        manifest_id: str,
        manifest_sha256: str,
        model_key: str,
        map_nums: list[int],
    ) -> None:
        if manifest_id != self.manifest_id or manifest_sha256 != self.manifest_sha256:
            raise EngineError(
                "zone_manifest_mismatch",
                "session Zone manifest identity does not match the engine asset",
            )
        for map_num in map_nums:
            if (model_key, map_num) not in self._maps:
                raise EngineError(
                    "zone_map_unavailable",
                    f"no pinned exact-color map for {model_key} face {map_num}",
                )

    def map_spec(self, model_key: str, map_num: int) -> dict[str, Any]:
        try:
            return self._maps[(model_key, map_num)]
        except KeyError as exc:
            raise EngineError(
                "zone_map_unavailable",
                f"no pinned exact-color map for {model_key} face {map_num}",
            ) from exc

    @lru_cache(maxsize=4)
    def _label_pixels(self, model_key: str, map_num: int) -> np.ndarray:
        spec = self.map_spec(model_key, map_num)
        path = self._asset_path(str(spec["path"]))
        self._verify_file(path, str(spec["sha256"]))
        try:
            with Image.open(path) as image:
                if image.size != (int(spec["width"]), int(spec["height"])):
                    raise EngineError("zone_map_geometry_mismatch", "label map dimensions changed")
                pixels = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        except OSError as exc:
            raise EngineError("zone_map_invalid_png", "exact-color label map is not a PNG") from exc
        allowed = {0}
        allowed.update((r << 16) | (g << 8) | b for r, g, b in self._palette.values())
        codes = (
            pixels[..., 0].astype(np.uint32) << 16
            | pixels[..., 1].astype(np.uint32) << 8
            | pixels[..., 2].astype(np.uint32)
        )
        if not set(int(value) for value in np.unique(codes)).issubset(allowed):
            raise EngineError("zone_map_unknown_color", "label map contains a color outside the palette")
        return pixels

    @lru_cache(maxsize=16)
    def zone_mask(self, model_key: str, map_num: int, zone_id: str) -> ZoneMask:
        try:
            color = np.asarray(self._palette[zone_id], dtype=np.uint8)
        except KeyError as exc:
            raise EngineError("zone_id_unknown", f"unknown Zone {zone_id}") from exc
        spec = self.map_spec(model_key, map_num)
        pixels = self._label_pixels(model_key, map_num)
        values = np.all(pixels == color, axis=2)
        if not bool(np.any(values)):
            raise EngineError(
                "zone_not_on_face",
                f"Zone {zone_id} is not present on {model_key} face {map_num}",
            )
        digest = hashlib.sha256()
        digest.update(b"exact-color-zone-mask/v1\0")
        digest.update(f"{values.shape[1]}x{values.shape[0]}\0".encode("ascii"))
        digest.update(values.astype(np.uint8).tobytes(order="C"))
        return ZoneMask(
            model_key=model_key,
            map_num=map_num,
            zone_id=zone_id,
            values=values,
            sha256=digest.hexdigest(),
            label_map_sha256=str(spec["sha256"]),
        )

    def zone_at_uv(
        self, model_key: str, map_num: int, x: float, y: float
    ) -> str | None:
        """Resolve bottom-left pixel UV without clamping out-of-range values."""
        pixels = self._label_pixels(model_key, map_num)
        height, width = pixels.shape[:2]
        col = math.floor(x)
        bottom_row = math.floor(y)
        row = height - 1 - bottom_row
        if not (0 <= col < width and 0 <= row < height):
            return None
        rgb = tuple(int(value) for value in pixels[row, col])
        if rgb == (0, 0, 0):
            return None
        for zone_id, color in self._palette.items():
            if rgb == color:
                return zone_id
        return None
