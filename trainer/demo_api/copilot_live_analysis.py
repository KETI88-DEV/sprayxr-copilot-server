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

# Never rescore the whole session on a live question. Completion remains the
# authoritative full-session evaluation; a live posture summary is a labeled sample.
LIVE_POSE_SAMPLE = 3
_CACHE_LIMIT = 128
_CACHE_TTL = 300.0
_GUARD = threading.Lock()
_CACHES: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


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
                        transport=transport, timeout_sec=timeout_sec)
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

    candidates = [body for body in bodies if str(body.get("zone_id", "")).strip().lower()
                  not in {"", "-1", "0", "none", "null", "nan"}
                  and (zone_id is None or str(body.get("zone_id")) == zone_id)][-LIVE_POSE_SAMPLE:]
    scored: list[tuple[str, PoseQueryResult]] = []
    reasons: dict[str, str] = {}
    deadline = time.monotonic() + 6.0
    for body in reversed(candidates):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        result = _score_cached(store, body, session_id=session_id, mcp_base_url=mcp_base_url,
                               transport=transport, timeout_sec=min(2.0, remaining))
        if result.analysis_status == "available" and result.similarity_score is not None:
            scored.append((str(body["zone_id"]), result))
        if result.reason_code:
            reasons[str(body.get("sequence_id"))] = result.reason_code
    analysis["pose_report"] = {
        "analysis_status": "available" if scored else "unavailable",
        "average_similarity_score": sum(result.similarity_score for _, result in scored) / len(scored) if scored else None,
        "evaluated_sequence_count": len(scored), "zone_summary": zone_summary(scored),
    }
    analysis["live_pose_context"] = {
        "source": "live_sequences", "scope": "recent_sequences",
        "total_sequence_count": len(bodies), "sampled_sequence_count": len(candidates),
        "evaluated_sequence_count": len(scored), "reason_codes": reasons,
        "sample_limit": LIVE_POSE_SAMPLE,
    }
    return None, analysis
