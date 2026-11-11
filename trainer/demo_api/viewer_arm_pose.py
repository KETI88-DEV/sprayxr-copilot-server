"""뷰어 재생 기록의 자세 유사도 — 분사 시퀀스마다 팔 관절 엔진으로 채점한다.

뷰어의 `pose_evaluation`은 전문가 기록 대비 **작업 유사도**(도구 각도·도막·이격거리 등)다. 여기는 **자세 유사도**다:
세션 종합 분석(`session_analysis.analyze_session`)과 같은 함수로 기록 CSV를 분사 구간으로 나눠 채점하고,
시퀀스·구역별 유사도와 교정을 낸다. 관절 월드 열이 없는 예전 형식 CSV는 `unavailable`이다 — 값을 만들지 않는다.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Mapping

from trainer.demo_api.session_analysis import analyze_session, read_csv, spray_runs

logger = logging.getLogger(__name__)
_CACHE: dict[tuple[Any, ...], dict[str, Any]] = {}
_CACHE_LIMIT = 32


class _NoStoredSequences:
    """뷰어 기록 이름은 DB 세션 ID가 아니다. 저장 시퀀스와 섞지 않고 CSV만 채점한다."""

    def session_bodies(self, session_id: str) -> list[dict[str, Any]]:
        return []


def _percent(score: object) -> int | None:
    return round(float(score) * 100) if isinstance(score, (int, float)) and not isinstance(score, bool) else None


def _seconds(row: Mapping[str, str], origin: float) -> float | None:
    try:
        return round(float(row.get("시간(초)", "")) - origin, 3)
    except (TypeError, ValueError):
        return None


def build_arm_pose_evaluation(
    csv_path: Path, *, record_id: str, mode_id: int, mcp_base_url: str, transport: Any = None
) -> dict[str, Any]:
    stat = csv_path.stat()
    key = (str(csv_path), stat.st_mtime_ns, stat.st_size, mcp_base_url, id(transport))
    if key in _CACHE:
        return _CACHE[key]
    try:
        raw = csv_path.read_bytes()
        analysis = analyze_session(
            session_id=record_id, user_id="motion-viewer", scenario_id=str(mode_id), csv_bytes=raw,
            store=_NoStoredSequences(), mcp_base_url=mcp_base_url, transport=transport,  # type: ignore[arg-type]
        )
        rows = read_csv(raw)
    except Exception:  # noqa: BLE001 - 뷰어 재생은 자세 엔진 장애와 무관하게 열려야 한다
        logger.exception("viewer arm pose evaluation failed record=%s", record_id)
        return {"status": "unavailable", "reason_code": "engine_error"}
    if analysis.analysis_status != "available":
        result: dict[str, Any] = {
            "status": "unavailable",
            "reason_code": "no_scored_sequence" if analysis.sequences else "no_world_joint_sequences",
            "guideline": analysis.guideline,
        }
    else:
        origin = (_seconds(rows[0], 0.0) or 0.0) if rows else 0.0
        runs = {f"SEQ-{n:03d}": run for n, run in enumerate(spray_runs(rows), start=1)}
        sequences = []
        for item in analysis.sequences:
            if item.get("analysis_status") != "available":
                continue
            start, end = runs.get(str(item.get("sequence_id")), (None, None))
            corrections = item.get("corrections") or []
            sequences.append({
                "sequence_id": item.get("sequence_id"), "zone_id": item.get("zone_id"),
                "start_seconds": _seconds(rows[start], origin) if start is not None else None,
                "end_seconds": _seconds(rows[end], origin) if end is not None else None,
                "similarity_percent": _percent(item.get("similarity_score")),
                "correction": next((str(c["label"]) for c in corrections if isinstance(c, Mapping) and c.get("label")), None),
                "confidence": item.get("confidence"),
            })
        guidance = {str(c["zone_id"]): c["correction_guidance"] for c in reversed(analysis.priority_corrections)}
        result = {
            "status": "available",
            "average_similarity_percent": _percent(analysis.average_similarity_score),
            "evaluated_sequence_count": analysis.counts.get("scored", 0),
            "sequence_count": analysis.counts.get("total", 0),
            "guideline": analysis.guideline,
            "zones": {
                str(zone["zone_id"]): {
                    "similarity_percent": _percent(zone["average_similarity_score"]),
                    "evaluated_sequence_count": zone["evaluated_sequence_count"],
                    "guidance": guidance.get(str(zone["zone_id"])),
                }
                for zone in analysis.zone_summary
            },
            "priority_corrections": [c["correction_guidance"] for c in analysis.priority_corrections],
            "sequences": sequences,
        }
    if len(_CACHE) >= _CACHE_LIMIT:
        _CACHE.pop(next(iter(_CACHE)))
    _CACHE[key] = result
    return result


def parse_viewer_arm_pose(value: str | None) -> dict[str, Any] | None:
    """뷰어가 되돌려 보낸 자세 채점 결과. 이 서버의 서명이 맞을 때만 받는다."""
    if not value or not value.startswith(("motion_viewer_analysis:", "motion_viewer_playback:")):
        return None
    try:
        payload = json.loads(value.split(":", 1)[1])
        claims = payload.get("arm_pose_context") if isinstance(payload, dict) else None
        from trainer.demo_api.viewer_context import verify_context
        if isinstance(claims, dict) and verify_context("arm_pose", claims, payload.get("arm_pose_signature")):
            return claims
    except (TypeError, ValueError, OverflowError):
        pass
    return None


def _sequence_at(arm: Mapping[str, Any], seconds: object) -> Mapping[str, Any] | None:
    """재생 시점의 분사 시퀀스. 이동 구간이면 직전에 끝난 분사."""
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)):
        return None
    timed = [s for s in arm.get("sequences", []) if isinstance(s, Mapping) and isinstance(s.get("start_seconds"), (int, float))]
    started = [s for s in timed if s["start_seconds"] <= seconds]
    return max(started, key=lambda s: s["start_seconds"]) if started else None


def arm_pose_answer(arm: Mapping[str, Any], *, current: bool, playback: Mapping[str, Any] | None, zone_id: str | None) -> str:
    """서명된 채점 결과로만 문장을 만든다. `current`는 "지금·이 Zone" 질문이다."""
    if arm.get("status") != "available":
        if arm.get("reason_code") == "no_world_joint_sequences":
            return "이 기록에는 관절 좌표가 없어 숙련자 대비 자세 유사도를 채점할 수 없습니다."
        return "이 기록에서는 숙련자 대비 자세 유사도를 채점할 수 없습니다."
    if current and zone_id is None:
        sequence = _sequence_at(arm, (playback or {}).get("playhead_seconds"))
        if sequence is not None:
            return (f"재생 시점의 분사({sequence['sequence_id']}, Zone {sequence['zone_id']})는 숙련자 팔 자세 대비 "
                    f"자세 유사도 {sequence['similarity_percent']}%입니다. "
                    + (sequence.get("correction") or ("" if sequence.get("confidence") == "limited" else "교정할 관절은 없습니다."))
                    # 엔진이 제한을 건 채점은 정상 자세로 확정하지 않는다.
                    + (" 입력 품질 제한이 있어 참고용 유사도입니다." if sequence.get("confidence") == "limited" else ""))
        zone_id = str(playback["zone_id"]) if playback and playback.get("zone_id") is not None else None
    if zone_id is not None:
        zone = (arm.get("zones") or {}).get(zone_id)
        if not isinstance(zone, Mapping):
            return f"Zone {zone_id}에는 자세를 채점한 분사가 없습니다."
        return (f"Zone {zone_id}의 자세 유사도는 평균 {zone['similarity_percent']}%입니다(분사 {zone['evaluated_sequence_count']}회). "
                + (zone.get("guidance") or "이 Zone에서 반복해 어긋난 관절은 없습니다."))
    priority = arm.get("priority_corrections") or []
    return str(arm.get("guideline") or "") + (f" 우선 교정: {priority[0]}" if priority else "")
