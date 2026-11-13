"""Describe recorded report progress without treating age as proof of failure."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Mapping

STALE_AFTER = timedelta(hours=24)
PROCESSING_FIELDS = ("processing_started_at","processing_updated_at","processing_finished_at","processing_reason")
REASONS = {"queued":"처리 요청 저장 · 작업 시작 여부는 최근 처리 기록 확인", "processing":"작업 시작 기록 있음 · 현재 실행 여부는 확인 불가", "complete":"처리 완료 기록", "failed":"처리 실패 기록", "report_input_unavailable":"저장된 분석 입력을 읽지 못함", "analysis_failed":"분석 작업 실패 기록"}


def _parse(value: object) -> datetime | None:
    try:
        stamp = datetime.fromisoformat(str(value).replace("Z","+00:00"))
        return stamp.replace(tzinfo=timezone.utc) if stamp.tzinfo is None else stamp.astimezone(timezone.utc)
    except (ValueError,TypeError,OverflowError):
        return None


def _elapsed(delta: timedelta) -> str:
    seconds = max(0,int(delta.total_seconds()))
    days, hours, minutes = seconds//86400, seconds%86400//3600, seconds%3600//60
    if days:
        return f"{days}일 {hours}시간"
    if hours:
        return f"{hours}시간 {minutes}분"
    return f"{minutes}분" if minutes else "1분 미만"


def report_progress(row: Mapping[str,Any], *, now: datetime | None = None) -> dict[str,Any]:
    data = dict(row)
    now = now or datetime.now(timezone.utc)
    status = str(data.get("report_status") or "").strip().lower()
    pending = status in {"pending","processing","running"}
    requested = _parse(data.get("requested_at"))
    elapsed = now - requested if requested and requested <= now else None
    stale = bool(pending and elapsed is not None and elapsed >= STALE_AFTER)
    label = "장기 대기 · 진행 여부 확인 불가" if stale else "분석 대기·진행" if pending else "처리 완료" if status in {"complete","completed","success"} else "처리 실패" if status in {"failed","failure","error"} else "상태 미기록" if not status else f"기타 상태: {status}"
    latest = max((_parse(data.get(field)) for field in PROCESSING_FIELDS[:3] if _parse(data.get(field))),default=None)
    reason = data.get("processing_reason")
    return {"label":label,"stale":stale,"elapsed":_elapsed(elapsed) if pending and elapsed is not None else "알 수 없음" if pending else "해당 없음",
            "latest_at":latest.isoformat() if latest else None,"reason":REASONS.get(str(reason),str(reason)) if reason else "미기록 · 진행 여부 확인 불가" if pending else "미기록"}
