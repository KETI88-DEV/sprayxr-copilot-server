"""Local Ray Point direction at a replay time, retained and authenticated by server."""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import OrderedDict
from dataclasses import dataclass
import json
import math
import re
import secrets
from threading import RLock
from typing import Any

from trainer.demo_api.viewer_context import sign_context, verify_context

LABELS = {"horizontal": "수평", "vertical": "수직", "diagonal": "대각선", "irregular": "비정형"}
_WINDOW = .16  # Seconds either side, within one continuous painted Map/Zone.


def _finite(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _classify(points: list[tuple[float, float, float]]) -> tuple[str | None, str]:
    lengths = {key: 0.0 for key in ("horizontal", "vertical", "diagonal")}
    for a, b in zip(points, points[1:]):
        dx, dy, dz = (b[i] - a[i] for i in range(3))
        distance = math.sqrt(dx*dx + dy*dy + dz*dz)
        if distance < .0002:  # Ignore sub-0.2 mm positional jitter, not a stroke.
            continue
        elevation = math.degrees(math.atan2(abs(dy), math.hypot(dx, dz)))
        key = "horizontal" if elevation <= 22.5 else "vertical" if elevation >= 67.5 else "diagonal"
        lengths[key] += distance
    total = sum(lengths.values())
    if total < .0005:
        return None, "stationary"
    best = max(lengths, key=lengths.get)
    return (best if lengths[best] / total >= .65 else "irregular"), "available"


@dataclass(frozen=True)
class _Stroke:
    times: tuple[float, ...]
    values: tuple[dict, ...]
    map_num: Any
    zone_id: Any


def _strokes(trace: list[dict]) -> tuple[_Stroke, ...]:
    groups: list[list[tuple[float, tuple[float, float, float], Any, Any]]] = []
    previous = None
    for row in trace:
        t, p = row.get("t"), row.get("point")
        identity = (row.get("segment"), row.get("map_num"), row.get("zone_id"))
        if not _finite(t) or t < 0 or not isinstance(p, (list, tuple)) or len(p) != 3 or not all(_finite(v) for v in p):
            previous = None
            continue
        if previous != identity or (groups and groups[-1] and t - groups[-1][-1][0] > .5):
            groups.append([])
        if groups[-1] and t <= groups[-1][-1][0]:
            continue
        groups[-1].append((float(t), tuple(float(v) for v in p), row.get("map_num"), row.get("zone_id")))
        previous = identity
    result = []
    for group in groups:
        times = tuple(row[0] for row in group)
        values = []
        for index, row in enumerate(group):
            left, right = bisect_left(times, row[0]-_WINDOW), bisect_right(times, row[0]+_WINDOW)
            if right-left < 2:
                left, right = max(0,index-1), min(len(group),index+2)
            sample = group[left:right]
            classification, reason = _classify([item[1] for item in sample])
            if len(sample) < 2 or sample[-1][0]-sample[0][0] > .5:
                classification, reason = None, "insufficient_samples"
            values.append({"classification": classification, "label": LABELS.get(classification),
                           "reason_code": reason, "window_start_seconds": sample[0][0], "window_end_seconds": sample[-1][0]})
        if group:
            result.append(_Stroke(times, tuple(values), group[0][2], group[0][3]))
    return tuple(result)


class PlaybackDirectionCache:
    def __init__(self, capacity: int = 64):
        self.capacity = capacity
        self.entries: OrderedDict[str, tuple[dict, tuple[_Stroke, ...], str | None]] = OrderedDict()
        self.keys: dict[str, str] = {}
        self.lock = RLock()

    def build(self, trace: list[dict], record_id: str, duration: float, key: str | None, offset: float = 0, clock_valid: bool = True) -> dict:
        with self.lock:
            token = self.keys.get(key) if key else None
            cached = self.entries.get(token)
            if cached and cached[0]["record_id"] == record_id and cached[0]["duration_seconds"] == duration and cached[0].get("csv_time_offset_seconds")==offset and cached[0].get("clock_valid")==clock_valid:
                self.entries.move_to_end(token)
                return dict(cached[0])
        normalized = [{**row,"t":round(row["t"]-offset,6)} for row in trace if _finite(row.get("t")) and row["t"]>=offset] if clock_valid else []
        strokes = _strokes(normalized)
        claims = {"snapshot_id": secrets.token_urlsafe(18), "record_id": record_id, "duration_seconds": duration, "csv_time_offset_seconds":offset, "clock_valid":clock_valid}
        with self.lock:
            self.entries[claims["snapshot_id"]] = (claims, strokes, key)
            if key:
                self.keys[key] = claims["snapshot_id"]
            while len(self.entries) > self.capacity:
                evicted, (_, _, old_key) = self.entries.popitem(last=False)
                if old_key and self.keys.get(old_key) == evicted:
                    del self.keys[old_key]
        return dict(claims)

    def lookup(self, claims: dict, seconds: Any) -> dict:
        result = {"scope": "current_frame", "analysis_status": "unavailable", "record_id": claims.get("record_id"),
                  "classification": None, "label": None, "basis": "local_ray_point_trajectory"}
        duration = claims.get("duration_seconds")
        if not _finite(seconds) or not _finite(duration) or not 0 <= seconds <= duration:
            return {**result, "reason_code": "invalid_playhead"}
        result["playhead_seconds"] = float(seconds)
        with self.lock:
            item = self.entries.get(claims.get("snapshot_id"))
            if item is None or item[0] != claims:
                return {**result, "reason_code": "snapshot_expired"}
            self.entries.move_to_end(claims["snapshot_id"])
        if not claims.get("clock_valid", True):
            return {**result, "reason_code":"invalid_clock"}
        strokes = item[1]
        starts = tuple(stroke.times[0] for stroke in strokes)
        index = bisect_right(starts, seconds) - 1
        if index < 0 or seconds > strokes[index].times[-1]:
            return {**result, "reason_code": "no_paint_trace"}
        stroke = strokes[index]
        right = bisect_left(stroke.times, seconds)
        nearest = min((max(0,right-1), min(right,len(stroke.times)-1)), key=lambda i:abs(stroke.times[i]-seconds))
        value = stroke.values[nearest]
        return {**result, **value, "analysis_status": "available" if value["classification"] else "unavailable",
                "map_num": stroke.map_num, "zone_id": stroke.zone_id}


_CACHE = PlaybackDirectionCache()


def build_playback_snapshot(trace: list[dict], *, record_id: str, duration_seconds: float, cache_key: str | None = None, time_offset_seconds: float = 0, clock_valid: bool = True) -> tuple[dict, str]:
    claims = _CACHE.build(trace, record_id, float(duration_seconds), cache_key, time_offset_seconds, clock_valid)
    return claims, sign_context("playback_direction", claims)


def playback_direction_from_context(context: str | None) -> dict | None:
    if not isinstance(context, str) or not context.startswith(("motion_viewer_analysis:", "motion_viewer_playback:")):
        return None
    try:
        value = json.loads(context.split(":",1)[1])
        claims, playback = value.get("frame_direction_context"), value.get("playback_context")
        if not isinstance(claims, dict) or not isinstance(playback, dict) or not verify_context("playback_direction",claims,value.get("frame_direction_signature")):
            return None
        return _CACHE.lookup(claims, playback.get("playhead_seconds"))
    except (ValueError,TypeError,AttributeError,OverflowError):
        return None


def playback_direction_answer(value: dict | None) -> str:
    if value is None or value.get("reason_code") == "snapshot_expired":
        return "현재 프레임의 도포 방향 자료가 없습니다. 재생 기록을 다시 불러와 주세요."
    seconds = value.get("playhead_seconds")
    prefix = f"현재 {seconds:.2f}초 프레임" if _finite(seconds) else "현재 프레임"
    if value.get("analysis_status") == "available":
        return prefix + "의 도포 방향은 " + value["label"] + "입니다. 인접한 분사 궤적을 기준으로 판단했습니다."
    if value.get("reason_code") == "stationary":
        return prefix + "은 분사 위치 변화가 작아 도포 방향을 판정할 수 없습니다."
    if value.get("reason_code") == "invalid_clock":
        return "CSV의 프레임 시간 정보를 확인할 수 없어 현재 도포 방향을 판정할 수 없습니다."
    if value.get("reason_code") == "invalid_playhead":
        return "현재 프레임의 재생 시점이 유효하지 않습니다. 재생 위치를 다시 선택해 주세요."
    return prefix + "에는 유효한 분사 궤적이 충분하지 않아 도포 방향을 판정할 수 없습니다."


def requested_direction_scope(prompt: str) -> str:
    if re.search("프레임|순간",prompt):
        return "frame"
    if "전체" in prompt or ("기록" in prompt and re.search("대표|평균|통계",prompt)):
        return "record"
    return "zone" if re.search(r"대표|평균|통계|(?:zone|존|구역)\s*\d+",prompt,re.I) else "frame"


def simple_direction_question(prompt: str) -> bool:
    normalized=re.sub(r"[\s?？.!。]+","",prompt)
    return re.fullmatch(r"(?:현재|지금)?(?:(?:선택한|선택된)?프레임(?:의)?)?(?:도포|도장|분사)방향(?:은|이|을)?(?:뭐야|무엇인가요|어때|알려줘|어떻게돼|뭐지)?",normalized) is not None


def aggregate_direction(context: str | None, prompt: str) -> tuple[str,dict]:
    scope=requested_direction_scope(prompt)
    empty={"scope":"whole_record" if scope=="record" else "zone_aggregate", "analysis_status":"unavailable"}
    try:
        payload=json.loads(context.split(":",1)[1])
        claims=payload.get("dialogue_context")
        if not isinstance(claims,dict) or not verify_context('dialogue',claims,payload.get('dialogue_signature')):
            raise ValueError('Unverified aggregate')
        if scope=='record':
            classification=claims.get('record_direction')
            title='전체 기록의 대표 도포 방향'
        else:
            zone=re.search(r'(?:zone|존|구역)\s*(\d+)',prompt,re.I)
            map_number=re.search(r'map\s*(\d+)',prompt,re.I)
            playback=payload.get('playback_context') or {}
            zone_id=zone.group(1) if zone else str(playback.get('zone_id'))
            map_id=map_number.group(1) if map_number else str(playback.get('map_num'))
            candidates=[item for item in (claims.get('zones') or {}).values() if str(item.get('zone_id'))==zone_id and (map_id=='None' or str(item.get('map_num'))==map_id)]
            row=candidates[0] if len(candidates)==1 else {}
            classification=row.get('direction')
            title=f"Map {row.get('map_num')} · Zone {row.get('zone_id')}의 대표 도포 방향"
        if classification in {"curved_irregular", "mixed"}:
            classification = "irregular"
        if classification in LABELS:
            return title+'은 '+LABELS[classification]+'입니다. 전체 또는 Zone의 대표 통계이며 현재 프레임 방향과 다를 수 있습니다.', {**empty,'analysis_status':'available','classification':classification,'label':LABELS[classification]}
    except (ValueError,TypeError,AttributeError):
        pass
    return '요청한 범위의 대표 도포 방향 자료가 없습니다. 재생 기록과 Zone 선택을 확인해 주세요.',empty
