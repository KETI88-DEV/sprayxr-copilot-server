"""뷰어 재생 기록의 자세 유사도 — 분사 시퀀스마다 팔 관절 엔진으로 채점한다.

뷰어의 `pose_evaluation`은 전문가 기록 대비 **작업 유사도**(도구 각도·도막·이격거리 등)다. 여기는 **자세 유사도**다:
세션 종합 분석(`session_analysis.analyze_session`)과 같은 함수로 기록 CSV를 분사 구간으로 나눠 채점하고,
시퀀스·구역별 유사도와 교정을 낸다.

관절은 CSV 형식에 따라 둘 중 하나다. 새 형식(2026-10 아바타)은 관절 월드 열을 그대로 쓴다. Spark 뷰어 기록 같은
예전 형식은 월드 열이 없고 로컬 회전만 있어, 숙련자 기준을 구울 때와 같은 FK로 관절을 만든다(`replay_telemetry`).
이때 건 위치는 손에서 합성하므로 `tool_not_held` 판정은 검증되지 않는다(그 모듈 독스트링).
팔 자세 기준은 스프레이 숙련자 기록으로만 구웠으므로 블라스팅 기록은 채점하지 않는다.
"""
from __future__ import annotations

import json
import logging
import secrets
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from trainer.demo_api.session_analysis import (
    analyze_session, csv_sequences, has_world_joints, painted_zone, read_csv, spray_runs,
)
from trainer.scripts import replay_telemetry

logger = logging.getLogger(__name__)
_USER = "motion-viewer"


class _RetryableEvaluation(Exception):
    """Return an evaluation without retaining a transient engine failure in the LRU."""

    def __init__(self, result: dict[str, Any]) -> None:
        super().__init__("Pose engine temporarily unavailable")
        self.result = result


class _Bodies:
    """`analyze_session`에 넘길 시퀀스 본문. 뷰어 기록은 DB 세션이 아니므로 저장 시퀀스와 섞지 않는다."""

    def __init__(self, bodies: list[dict[str, Any]]) -> None:
        self._bodies = bodies

    def session_bodies(self, session_id: str) -> list[dict[str, Any]]:
        return self._bodies


def _engine_bodies(rows: list[dict[str, str]], *, session_id: str, scenario_id: str) -> list[dict[str, Any]]:
    """분사 구간마다 자세 엔진 본문(`SEQ-001`…, 구간 순서). 구역은 착탄이 가장 많이 난 구역이다."""
    if has_world_joints(rows):
        return csv_sequences(rows, session_id=session_id, user_id=_USER, scenario_id=scenario_id)
    bodies: list[dict[str, Any]] = []
    for number, (start, end) in enumerate(spray_runs(rows), start=1):
        try:
            body, _skipped = replay_telemetry.build_payload(
                rows, (start, end), session_id=session_id, sequence_id=f"SEQ-{number:03d}", max_frames=end - start + 1)
        except (KeyError, ValueError):  # 관절·건·착탄 열이 비거나 빠진 구간은 채점 대상이 아니다
            continue
        body.update(user_id=_USER, scenario_id=scenario_id, zone_id=painted_zone(rows[start:end + 1]))
        bodies.append(body)
    return bodies


def _percent(score: object) -> int | None:
    return round(float(score) * 100) if isinstance(score, (int, float)) and not isinstance(score, bool) else None


def _seconds(row: Mapping[str, str], origin: float) -> float | None:
    try:
        return round(float(row.get("시간(초)", "")) - origin, 3)
    except (TypeError, ValueError):
        return None


def build_arm_pose_evaluation(
    csv_path: Path, *, record_id: str, process: str, mode_id: int, mcp_base_url: str, transport: Any = None
) -> dict[str, Any]:
    if process != "spray":
        return {"status": "unavailable", "reason_code": "process_not_supported"}
    try:
        stat = csv_path.stat()
        return _evaluate(str(csv_path), (stat.st_mtime_ns, stat.st_size), record_id, mode_id, mcp_base_url, transport)
    except _RetryableEvaluation as exc:
        return exc.result
    except Exception:  # noqa: BLE001 - 뷰어 재생은 자세 엔진 장애와 무관하게 열려야 한다. 실패는 캐시하지 않는다.
        logger.exception("viewer arm pose evaluation failed record=%s", record_id)
        return {"status": "unavailable", "reason_code": "engine_error"}


@lru_cache(maxsize=32)
def _evaluate(
    path_text: str, stamp: tuple[int, int], record_id: str, mode_id: int, mcp_base_url: str, transport: Any
) -> dict[str, Any]:
    del stamp  # 파일이 바뀌면 다시 채점하도록 캐시 키에만 쓴다
    # 자세 엔진은 세션 ID별로 상태(굳은 관절 등)를 쌓고 지우는 경로가 없다. 같은 이름을 다시 쓰면 앞선 채점이
    # 다음 채점을 바꾼다(Spark에서 같은 기록이 17회 65% → 16회 66%). 채점마다 새 이름을 쓴다.
    # 엔진은 세션을 최대 64개까지 오래 안 쓴 것부터 버리므로 쌓이지 않는다.
    session_id = f"{_USER}:{record_id}:{secrets.token_hex(4)}"
    rows = [{(name or "").strip(): value for name, value in row.items()} for row in read_csv(Path(path_text).read_bytes())]
    bodies = _engine_bodies(rows, session_id=session_id, scenario_id=str(mode_id))
    analysis = analyze_session(
        session_id=session_id, user_id=_USER, scenario_id=str(mode_id), csv_bytes=None,
        store=_Bodies(bodies), mcp_base_url=mcp_base_url, transport=transport,  # type: ignore[arg-type]
    )
    joint_source = "world" if has_world_joints(rows) else "fk"
    if analysis.analysis_status != "available":
        result: dict[str, Any] = {
            "status": "unavailable",
            "reason_code": "no_scored_sequence" if bodies else "no_joint_sequences",
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
                # 팔 방향 질문은 분사의 교정 전부에서 찾는다(첫 교정만 보면 다른 팔의 교정을 놓친다).
                "corrections": [str(c["label"]) for c in corrections if isinstance(c, Mapping) and c.get("label")],
                # 화면이 교정이 난 관절을 스켈레톤 위에 표시한다.
                "joints": [str(c["joint"]) for c in corrections if isinstance(c, Mapping) and c.get("joint")],
                "confidence": item.get("confidence"),
            })
        guidance = {str(c["zone_id"]): c["correction_guidance"] for c in reversed(analysis.priority_corrections)}
        result = {
            "status": "available",
            "joint_source": joint_source,
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
    if analysis.counts.get("reason:engine_unavailable", 0):
        # analyze_session projects HTTP failures into normal results. An exception keeps
        # the LRU from retaining either a failed result or an incomplete partial score.
        if result["status"] == "unavailable":
            result["reason_code"] = "engine_error"
        raise _RetryableEvaluation(result)
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


_UNAVAILABLE = {
    "process_not_supported": "블라스팅 기록은 숙련자 팔 자세 기준이 없어 자세 유사도를 채점하지 않습니다.",
    "no_joint_sequences": "이 기록에는 자세를 채점할 분사 구간이 없습니다.",
}


def _sequence_at(arm: Mapping[str, Any], seconds: object) -> Mapping[str, Any] | None:
    """재생 시점의 분사 시퀀스. 이동 구간이면 직전에 끝난 분사."""
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)):
        return None
    timed = [s for s in arm.get("sequences", []) if isinstance(s, Mapping) and isinstance(s.get("start_seconds"), (int, float))]
    started = [s for s in timed if s["start_seconds"] <= seconds]
    return max(started, key=lambda s: s["start_seconds"]) if started else None


def _side_answer(arm: Mapping[str, Any], side: str, *, current: bool, playback: Mapping[str, Any] | None, zone_id: str | None) -> str:
    """오른팔·왼팔 질문의 답: 엔진이 그 팔에 낸 교정이 있는 분사를 센다. 교정 문장은 엔진의 것을 그대로 옮긴다."""
    word, name = ("오른쪽", "오른팔") if side == "right" else ("왼쪽", "왼팔")
    scored = [s for s in arm.get("sequences", []) if isinstance(s, Mapping)]
    scope = f"채점한 분사 {len(scored)}회"
    sequence = _sequence_at(arm, (playback or {}).get("playhead_seconds")) if current and zone_id is None else None
    if sequence is not None:
        scored, scope = [sequence], f"재생 시점의 분사({sequence['sequence_id']})"
    elif zone_id is not None:
        scored = [s for s in scored if str(s.get("zone_id")) == zone_id]
        scope = f"Zone {zone_id}의 분사 {len(scored)}회"
    hits = [(s, label) for s in scored for label in s.get("corrections") or [s.get("correction")] if label and word in str(label)]
    limited = sum(s.get("confidence") == "limited" for s in scored)
    note = f" 이 가운데 {limited}회는 입력 품질 제한이 있어 참고용 채점입니다." if limited else ""
    if not hits:
        return f"{scope}에서 {name} 교정은 나오지 않았습니다.{note}"
    count = len({s.get("sequence_id") for s, _ in hits})
    return f"{scope} 중 {count}회에서 {name} 교정이 나왔습니다. 예({hits[0][0].get('sequence_id')}): {hits[0][1]}{note}"


def arm_pose_answer(arm: Mapping[str, Any], *, current: bool, playback: Mapping[str, Any] | None, zone_id: str | None,
                    side: str | None = None) -> str:
    """서명된 채점 결과로만 문장을 만든다. `current`는 "지금·이 Zone" 질문, `side`는 오른팔·왼팔 질문이다."""
    if arm.get("status") != "available":
        return _UNAVAILABLE.get(str(arm.get("reason_code")), "이 기록에서는 숙련자 대비 자세 유사도를 채점할 수 없습니다.")
    if side is not None:
        return _side_answer(arm, side, current=current, playback=playback, zone_id=zone_id)
    if current and zone_id is None:
        sequence = _sequence_at(arm, (playback or {}).get("playhead_seconds"))
        if sequence is not None:
            seconds, end = (playback or {}).get("playhead_seconds"), sequence.get("end_seconds")
            when = "직전 분사" if isinstance(end, (int, float)) and isinstance(seconds, (int, float)) and seconds > end else "재생 시점의 분사"
            limited = sequence.get("confidence") == "limited"
            # 엔진이 제한을 건 채점은 정상 자세로 확정하지 않는다.
            notes = [sequence.get("correction") or ("" if limited else "교정할 관절은 없습니다."),
                     "입력 품질 제한이 있어 참고용 유사도입니다." if limited else ""]
            # 분사 하나가 여러 Zone에 걸칠 수 있어 Zone은 적지 않는다(뷰어 Zone 카드와 어긋나 보인다).
            return (f"{when}({sequence['sequence_id']})는 숙련자 팔 자세 대비 자세 유사도 {sequence['similarity_percent']}%입니다. "
                    + " ".join(note for note in notes if note))
        zone_id = str(playback["zone_id"]) if playback and playback.get("zone_id") is not None else None
    if zone_id is not None:
        zone = (arm.get("zones") or {}).get(zone_id)
        if not isinstance(zone, Mapping):
            return f"Zone {zone_id}에는 자세를 채점한 분사가 없습니다."
        return (f"Zone {zone_id}의 자세 유사도는 평균 {zone['similarity_percent']}%입니다(분사 {zone['evaluated_sequence_count']}회). "
                + (zone.get("guidance") or "이 Zone에서 반복해 어긋난 관절은 없습니다."))
    priority = arm.get("priority_corrections") or []
    return str(arm.get("guideline") or "").replace("평균 유사도", "평균 자세 유사도") + (f" 우선 교정: {priority[0]}" if priority else "")
