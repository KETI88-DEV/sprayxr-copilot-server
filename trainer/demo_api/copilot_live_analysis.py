"""Server-owned summaries for questions asked before session completion."""

from __future__ import annotations

import threading
import time
import weakref
from collections import OrderedDict
from typing import Any

from trainer.demo_api.pose_query import PoseQueryResult, score_body
from trainer.demo_api.sequence_analysis import SequenceAnalysisEngine
from trainer.demo_api.sequence_store import SequenceStore

# 훈련 중 질문은 지금 작업하는 구역(또는 지정한 구역)의 분사를 모두 채점한다(2026-11-11 지시).
# 완료 분석이 세션 전체의 정본이고, 여기 결과는 범위와 채점 수를 함께 밝힌다.
_INVALID_ZONES = {"", "-1", "0", "none", "null", "nan"}


def current_zone(store: SequenceStore, session_id: str) -> str | None:
    """가장 최근에 올라온 분사의 구역 — 지금 작업 중인 구역."""
    for body in reversed(store.session_bodies(session_id)):
        zone = str(body.get("zone_id", "")).strip()
        if zone.lower() not in _INVALID_ZONES:
            return zone
    return None

# 저장된 분사는 바뀌지 않는다. 훈련 한 회(수십 분) 동안 다시 채점하지 않게 오래 둔다.
# 5분·128개였을 때 6초 제한에 걸린 질문마다 채점 수와 평균이 달랐다(2026-11-24: 54회·54% 다음 질문 55%).
_CACHE_LIMIT = 4096
_CACHE_TTL = 3600.0
_GUARD = threading.Lock()
_CACHES: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_WARMING: set[tuple[int, str]] = set()


def _warm_remaining(store: SequenceStore, bodies: list[dict[str, Any]], *, session_id: str,
                    mcp_base_url: str, transport: Any) -> None:
    """시간 제한으로 못 채점한 분사를 뒤에서 마저 채점해 둔다. 다음 질문은 전체 분사로 답한다."""
    key = (id(store), session_id)
    with _GUARD:
        if key in _WARMING:
            return
        _WARMING.add(key)

    def run() -> None:
        try:
            for body in bodies:
                _score_cached(store, body, session_id=session_id, mcp_base_url=mcp_base_url,
                              transport=transport, timeout_sec=2.0)
        finally:
            with _GUARD:
                _WARMING.discard(key)

    threading.Thread(target=run, name="pose-cache-warm", daemon=True).start()


def _score_cached(store: SequenceStore, body: dict[str, Any], *, session_id: str,
                  mcp_base_url: str, transport: Any, timeout_sec: float) -> PoseQueryResult:
    sequence_id = str(body.get("sequence_id", ""))
    key = (session_id, sequence_id, mcp_base_url, id(transport))
    with _GUARD:
        cache = _CACHES.setdefault(store, OrderedDict())
        cached = cache.get(key)
        if cached and time.monotonic() - cached[0] < _CACHE_TTL:
            cache.move_to_end(key)
            return cached[1]
    result = score_body(body, sequence_id=sequence_id, mcp_base_url=mcp_base_url,
                        transport=transport, timeout_sec=timeout_sec, isolated=True)
    # Retry engine outages next time; immutable stored sequences make successful
    # measurements safe to reuse within this short reference refresh window.
    if result.analysis_status == "available":
        with _GUARD:
            cache[key] = (time.monotonic(), result)
            cache.move_to_end(key)
            while len(cache) > _CACHE_LIMIT:
                cache.popitem(last=False)
    return result


def live_analysis(store: SequenceStore, *, session_id: str, scenario_id: str | None,
                  task_type: str, sequence_engine: SequenceAnalysisEngine | None,
                  mcp_base_url: str, transport: Any = None,
                  zone_id: str | None = None) -> tuple[None, dict[str, Any]] | None:
    bodies = store.session_bodies(session_id)
    if not bodies:
        return None
    scenario = scenario_id or next((str(body["scenario_id"]) for body in reversed(bodies)
                                    if body.get("scenario_id")), None)
    analysis: dict[str, Any] = {"source": "live_sequences", "sequence_count": len(bodies)}
    if task_type == "trajectory_recomm":
        if sequence_engine is not None and scenario:
            analysis["sequence_report"] = sequence_engine.report_bodies(bodies, scenario_id=scenario)
        return None, analysis

    from trainer.demo_api.session_analysis import zone_summary

    candidates = [body for body in bodies if str(body.get("zone_id", "")).strip().lower() not in _INVALID_ZONES
                  and (zone_id is None or str(body.get("zone_id")) == zone_id)]
    scored: list[tuple[str, PoseQueryResult]] = []
    reasons: dict[str, str] = {}
    deadline = time.monotonic() + 6.0
    ordered = list(reversed(candidates))
    pending: list[dict[str, Any]] = []
    for index, body in enumerate(ordered):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            pending = ordered[index:]
            _warm_remaining(store, pending, session_id=session_id, mcp_base_url=mcp_base_url, transport=transport)
            break
        result = _score_cached(store, body, session_id=session_id, mcp_base_url=mcp_base_url,
                               transport=transport, timeout_sec=min(2.0, remaining))
        if result.analysis_status == "available" and result.similarity_score is not None:
            scored.append((str(body["zone_id"]), result))
        if result.reason_code:
            reasons[str(body.get("sequence_id"))] = result.reason_code
    analysis["pose_report"] = {
        "analysis_status": "available" if scored else "unavailable",
        # 구역만 채점했을 때의 평균은 세션 평균이 아니다 — 답이 "세션 평균"이라고 부르지 않게 비워 둔다.
        "average_similarity_score": (sum(result.similarity_score for _, result in scored) / len(scored)
                                     if scored and zone_id is None else None),
        "evaluated_sequence_count": len(scored), "zone_summary": zone_summary(scored),
    }
    analysis["live_pose_context"] = {
        "source": "live_sequences", "scope": "zone" if zone_id is not None else "session", "zone_id": zone_id,
        "total_sequence_count": len(bodies), "sampled_sequence_count": len(candidates),
        "evaluated_sequence_count": len(scored), "reason_codes": reasons,
        "pending_sequence_count": len(pending),
    }
    return None, analysis
