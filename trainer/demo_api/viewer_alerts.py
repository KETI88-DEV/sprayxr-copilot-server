"""v2 뷰어 재생 알림 — 공개 이상 알림 SSE(`/api/v1/stream/anomalies`)와 같은 판정 함수를 재생 기록에 붙인다.

뷰어는 끝난 CSV 기록을 재생하므로, 분사 구간마다 analyze-sequence 와 같은 본문을 만들어 같은 엔진으로 판정하고
그 시점(재생 초)을 붙여 내려준다. 화면은 이 값을 그리기만 한다.

- 순서·방향: `SequenceAnalysisEngine.decide`(공개 경로 `evaluate`가 도막 진행 문장 게이트를 씌우기 전의 판정).
- 도막: `telemetry_router.coating_alerts`를 도막 패널과 같은 행(`coating_evaluation.target.zones`)에 그대로 적용한다.
  CSV 기록에는 최종 높이맵 한 장뿐이라, 알림 시점은 그 Zone을 마지막으로 칠한 분사의 끝이다(`basis: final_height_map`).
"""
from __future__ import annotations

import copy
import logging
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from pydantic import ValidationError

from trainer.demo_api.session_analysis import painted_zone, read_csv, spray_runs
from trainer.demo_api.telemetry_contracts import AnalyzeSequenceRequest
from trainer.demo_api.telemetry_router import coating_alerts
from trainer.demo_api.viewer_arm_pose import _Bodies, _engine_bodies, _seconds
from trainer.expert_zone_mapping import canonical_expert_zone_id

logger = logging.getLogger(__name__)
_VALUE_KEYS = ("grade", "thickness_um", "target_thickness_um", "target_basis", "status")
# (시퀀스 ID, 시작초, 끝초, 면, canonical Zone). 번호는 자세 막대와 같은 분사 구간 순서다.
Window = tuple[str, float, float, str | None, str]


def _rows(path: Path) -> list[dict[str, str]]:
    return [{(name or "").strip(): value for name, value in row.items()} for row in read_csv(path.read_bytes())]


def _canonical(mode_id: int, legacy: bool, zone: object) -> str:
    # motion_viewer._canonicalize_zone_items 와 같은 규칙. 프로필은 canonical 번호로 저장돼 있다.
    return str(canonical_expert_zone_id("spray", "learning" if legacy else "source", mode_id, zone))


def _stamp(path: Path) -> tuple[str, int, int]:
    stat = path.stat()
    return str(path), stat.st_mtime_ns, stat.st_size


@lru_cache(maxsize=32)
def _windows(path_text: str, mtime: int, size: int, mode_id: int, legacy: bool) -> tuple[Window, ...]:
    del mtime, size  # 파일이 바뀌면 다시 읽도록 열쇠에만 쓴다
    rows = _rows(Path(path_text))
    origin = (_seconds(rows[0], 0.0) or 0.0) if rows else 0.0
    out: list[Window] = []
    for number, (start, end) in enumerate(spray_runs(rows), start=1):
        begin, finish = _seconds(rows[start], origin), _seconds(rows[end], origin)
        if begin is None or finish is None:
            continue
        window = rows[start:end + 1]
        zone = painted_zone(window)
        faces = Counter(str(row.get("Ray Texture Map Number", "")).strip() for row in window if str(row.get("Zone ID", "")).strip() == zone)
        try:
            face: str | None = str(int(float(faces.most_common(1)[0][0])))
        except (IndexError, ValueError):
            face = None
        out.append((f"SEQ-{number:03d}", begin, finish, face, _canonical(mode_id, legacy, zone)))
    return tuple(out)


def _shift(seconds: float, offset: float) -> float:
    return round(max(0.0, seconds - offset), 3)


def spray_windows(path: Path, *, mode_id: int, legacy: bool, offset: float = 0.0) -> tuple[Window, ...]:
    """offset = 재생 프레임 0초가 CSV 첫 행보다 늦은 만큼(첫 행이 빈 기록). 재생 시계로 맞춘다."""
    return tuple((i, _shift(a, offset), _shift(b, offset), f, z) for i, a, b, f, z in _windows(*_stamp(path), mode_id, legacy))


def _item(anomaly_type: str, reason_code: object, sequence_id: str, map_num: object, zone_id: object,
          seek: float, at: float, basis: str, message: object, values: dict[str, Any], key: str) -> dict[str, Any]:
    return {"id": f"{key}:{anomaly_type}", "anomaly_type": anomaly_type, "reason_code": reason_code,
            "sequence_id": sequence_id, "map_num": None if map_num is None else str(map_num), "zone_id": str(zone_id),
            "at_seconds": at, "seek_seconds": seek, "basis": basis, "message": message, "values": values}


def coating_alerts_for(target: Mapping[str, Any], windows: tuple[Window, ...]) -> list[dict[str, Any]]:
    """도막 패널 행마다 공개 도막 알림 함수를 부른다. 과도막·저도막을 같이 받는 Zone은 없어 행 하나로 둘 다 얻는다."""
    if target.get("analysis_status") != "available":
        return []
    last = {zone: (sequence_id, start, end) for sequence_id, start, end, _face, zone in windows}  # 그 Zone의 마지막 분사
    items = []
    for row in target.get("zones") or []:
        hit = last.get(str(row.get("zone_id")))
        if hit is None:   # 그 Zone을 칠한 분사가 없으면 시점을 정할 수 없다
            continue
        for alert in coating_alerts({**row, "left_zone": row}):
            payload = alert["payload"]
            items.append(_item(alert["anomaly_type"], payload["reason_code"], hit[0], payload["map_num"], payload["zone_id"],
                               hit[1], hit[2], "final_height_map", payload["message"],
                               {key: payload[key] for key in _VALUE_KEYS}, alert["dedupe_key"]))
    return sorted(items, key=lambda item: item["at_seconds"])


def _unavailable(reason_code: str) -> dict[str, Any]:
    return {"status": "unavailable", "reason_code": reason_code, "alerts": []}


def sequence_alerts_for(path: Path, *, process: str, mode_id: int, legacy: bool, engine: Any, offset: float = 0.0) -> dict[str, Any]:
    """분사 구간마다 공개 순서·방향 판정. 엔진 예외는 캐시하지 않는다."""
    if process != "spray":   # 본문 tool_type 이 spray_gun 으로 고정이라 블라스팅 기준을 고를 수 없다
        return _unavailable("process_not_supported")
    try:
        result = copy.deepcopy(_sequence_alerts(*_stamp(path), mode_id, legacy, engine))
        for alert in result["alerts"]:
            alert["seek_seconds"], alert["at_seconds"] = _shift(alert["seek_seconds"], offset), _shift(alert["at_seconds"], offset)
        return result
    except Exception:  # noqa: BLE001 - 재생은 판정 장애와 무관하게 열려야 한다
        logger.exception("viewer sequence alerts failed path=%s", path)
        return _unavailable("engine_error")


# shortcut: 열쇠가 CSV 파일뿐이라 전문가 프로필이 바뀌어도 옛 판정을 낸다(최대 32건), 프로필을 실행 중에 갈게 되면 프로필 mtime을 열쇠에 넣는다.
@lru_cache(maxsize=32)
def _sequence_alerts(path_text: str, mtime: int, size: int, mode_id: int, legacy: bool, engine: Any) -> dict[str, Any]:
    rows = _rows(Path(path_text))
    bodies = _engine_bodies(rows, session_id="motion-viewer-alerts", scenario_id=str(mode_id))
    requests = []
    for body in bodies:
        body["zone_id"] = _canonical(mode_id, legacy, body["zone_id"])
        try:
            requests.append((body, AnalyzeSequenceRequest.model_validate(body)))
        except ValidationError:   # 공개 경로도 422로 거부하고 저장하지 않는다
            continue
    if not requests:
        return _unavailable("no_sequences")
    store = _Bodies([body for body, _ in requests])
    windows = {window[0]: window for window in _windows(path_text, mtime, size, mode_id, legacy)}
    alerts, reasons, judged = [], [], False
    for _body, request in requests:
        decision = engine.decide(request, store)
        if decision.feedback.analysis_status != "available":
            reasons.append(decision.feedback.reason_code)
            continue
        judged = True
        window = windows.get(request.sequence_id)
        if decision.anomaly_type is None or window is None:
            continue
        alerts.append(_item(decision.anomaly_type, decision.feedback.reason_code, request.sequence_id, window[3],
                            request.zone_id, window[1], window[2], "sequence", decision.text_guide, {}, request.sequence_id))
    if not judged:
        return _unavailable(next((str(reason) for reason in reasons if reason != "zone_unresolved"), "no_sequences"))
    return {"status": "available", "reason_code": None, "alerts": alerts}
