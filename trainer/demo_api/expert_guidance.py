"""Deterministic speed and stand-off guidance from expert Zone profiles."""

from __future__ import annotations

import json
import math
import statistics
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from trainer.demo_api.telemetry_contracts import (
    AnalyzeSequenceRequest,
    GuideItem,
    SequenceGuide,
)


@dataclass
class _CachedProfile:
    stamp: tuple[int, int]
    payload: dict[str, Any]


@dataclass
class _StableStatus:
    stable: str
    candidate: str | None = None
    count: int = 0


_MESSAGES = {
    "slow": "이동 속도를 조금 높이세요.",
    "fast": "이동 속도를 조금 낮추세요.",
    "near": "스프레이건을 피도면에서 조금 멀리 이동하세요.",
    "far": "스프레이건을 피도면 쪽으로 가까이 이동하세요.",
    "ok": "현재 상태가 적정합니다.",
}


class ExpertGuidanceEngine:
    def __init__(self, reference_root: Path) -> None:
        self._root = Path(reference_root)
        self._profiles: dict[tuple[str, int], _CachedProfile] = {}
        self._states: dict[tuple[str, str, int, str, str], _StableStatus] = {}
        self._guard = threading.Lock()

    @property
    def reference_root(self) -> Path:
        return self._root

    def _profile_path(self, process: str, mode_id: int) -> Path:
        return (
            self._root
            / "generated"
            / "profiles"
            / process
            / "learning"
            / f"mode_{mode_id:02d}.json"
        )

    def _load_profile(self, process: str, mode_id: int) -> tuple[dict[str, Any] | None, str | None]:
        key = (process, mode_id)
        path = self._profile_path(process, mode_id)
        cached = self._profiles.get(key)
        try:
            stat = path.stat()
            stamp = (stat.st_mtime_ns, stat.st_size)
            if cached is not None and cached.stamp == stamp:
                return cached.payload, None
            payload = json.loads(path.read_text(encoding="utf-8"))
            zones = payload["aggregate"]["zones"]
            if not isinstance(zones, dict):
                raise ValueError("aggregate.zones must be an object")
            self._profiles[key] = _CachedProfile(stamp=stamp, payload=payload)
            return payload, None
        except FileNotFoundError:
            return (cached.payload, None) if cached is not None else (None, "reference_unavailable")
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return (cached.payload, None) if cached is not None else (None, "reference_invalid")

    @staticmethod
    def _zone(payload: dict[str, Any], zone_id: str) -> dict[str, Any] | None:
        zones = payload["aggregate"]["zones"]
        direct = zones.get(zone_id)
        if isinstance(direct, dict):
            return direct
        matches = [zone for zone in zones.values() if str(zone.get("zone_id")) == zone_id]
        return matches[0] if len(matches) == 1 else None

    @staticmethod
    def _number(value: Any) -> float | None:
        if isinstance(value, bool):
            return None
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        return parsed if math.isfinite(parsed) and parsed >= 0 else None

    @classmethod
    def _metric(
        cls,
        request: AnalyzeSequenceRequest,
        *,
        millimetre_key: str,
        metre_key: str,
        frame_key: str,
        unit_key: str,
    ) -> tuple[float | None, str | None]:
        result = request.sequence_result.model_dump(exclude_none=True)
        has_mm = millimetre_key in result
        has_m = metre_key in result
        if has_mm and has_m:
            return None, "invalid_units"
        if has_mm:
            value = cls._number(result.get(millimetre_key))
            return (value, None) if value is not None else (None, "metric_unavailable")
        if has_m:
            value = cls._number(result.get(metre_key))
            return (value * 1000.0, None) if value is not None else (None, "metric_unavailable")

        values: list[float] = []
        for frame in request.sequence_info:
            metrics = frame.gun_metrics.model_dump() if frame.gun_metrics is not None else {}
            raw = cls._number(metrics.get(frame_key))
            if raw is None:
                continue
            unit = metrics.get(unit_key)
            if unit in (None, "mm", "mm/s"):
                values.append(raw)
            elif unit in ("m", "m/s"):
                values.append(raw * 1000.0)
            else:
                return None, "invalid_units"
        return (statistics.median(values), None) if values else (None, "metric_unavailable")

    @staticmethod
    def _limits(reference: dict[str, Any], *, metric: str) -> tuple[float, float] | None:
        try:
            median = float(reference["median"])
            q1 = float(reference["q1"])
            q3 = float(reference["q3"])
        except (KeyError, TypeError, ValueError):
            return None
        if not all(math.isfinite(value) and value >= 0 for value in (median, q1, q3)):
            return None
        minimum = max(median * (0.15 if metric == "speed" else 0.05), 20.0 if metric == "distance" else 1.0)
        padding = max((q3 - q1) * 0.25, minimum)
        return max(0.0, q1 - padding), q3 + padding

    def _stabilize(self, key: tuple[str, str, int, str, str], raw: str) -> str:
        state = self._states.get(key)
        if state is None:
            while len(self._states) >= 4096:
                self._states.pop(next(iter(self._states)))
            self._states[key] = _StableStatus(stable=raw)
            return raw
        if raw == state.stable:
            state.candidate = None
            state.count = 0
            return state.stable
        if state.candidate == raw:
            state.count += 1
        else:
            state.candidate = raw
            state.count = 1
        if state.count >= 2:
            state.stable = raw
            state.candidate = None
            state.count = 0
        return state.stable

    @staticmethod
    def _unavailable(reason: str) -> GuideItem:
        return GuideItem(status=None, message=None, available=False, reason=reason)

    def _item(
        self,
        *,
        request: AnalyzeSequenceRequest,
        zone: dict[str, Any],
        metric: Literal["speed", "distance"],
        value: float | None,
        reason: str | None,
    ) -> GuideItem:
        if reason is not None or value is None:
            return self._unavailable(reason or "metric_unavailable")
        reference_key = "speed_mms" if metric == "speed" else "distance_mm"
        limits = self._limits(zone.get(reference_key, {}), metric=metric)
        if limits is None:
            return self._unavailable("reference_invalid")
        lower, upper = limits
        if metric == "speed":
            raw = "slow" if value < lower else "fast" if value > upper else "ok"
        else:
            raw = "near" if value < lower else "far" if value > upper else "ok"
        mode_id = request.resolved_mode_id
        process = request.process_type
        if mode_id is None or process is None:
            return self._unavailable("reference_context_unavailable")
        stable = self._stabilize((request.session_id, process, mode_id, request.zone_id, metric), raw)
        return GuideItem(status=stable, message=_MESSAGES[stable], available=True, reason=None)

    def evaluate(self, request: AnalyzeSequenceRequest) -> SequenceGuide:
        with self._guard:
            mode_id = request.resolved_mode_id
            process = request.process_type
            if mode_id is None or process is None:
                unavailable = self._unavailable("reference_context_unavailable")
                return SequenceGuide(speed=unavailable, distance=unavailable.model_copy())
            profile, profile_reason = self._load_profile(process, mode_id)
            if profile is None:
                unavailable = self._unavailable(profile_reason or "reference_unavailable")
                return SequenceGuide(speed=unavailable, distance=unavailable.model_copy())
            zone = self._zone(profile, request.zone_id)
            if zone is None:
                unavailable = self._unavailable("reference_unavailable")
                return SequenceGuide(speed=unavailable, distance=unavailable.model_copy())

            speed, speed_reason = self._metric(
                request,
                millimetre_key="avg_speed_mms",
                metre_key="avg_speed_mps",
                frame_key="speed",
                unit_key="speed_unit",
            )
            distance, distance_reason = self._metric(
                request,
                millimetre_key="avg_distance_mm",
                metre_key="avg_distance_m",
                frame_key="distance",
                unit_key="distance_unit",
            )
            return SequenceGuide(
                speed=self._item(
                    request=request,
                    zone=zone,
                    metric="speed",
                    value=speed,
                    reason=speed_reason,
                ),
                distance=self._item(
                    request=request,
                    zone=zone,
                    metric="distance",
                    value=distance,
                    reason=distance_reason,
                ),
            )
