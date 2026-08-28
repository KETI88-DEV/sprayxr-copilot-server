#!/usr/bin/env python3
"""pose lab HTTP 표면. demo API(:8120)와 분리된 테스트 도구다."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any, Final

from fastapi import FastAPI, Query
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from trainer.pose_lab import service

STATIC_DIR = Path(__file__).resolve().parent / "static"

# 원본 CSV 트리의 루트. 비어 있으면 소스를 배선하지 않고 지금까지의 거동을 유지한다.
DATA_ROOT_ENV: Final = "POSE_LAB_DATA_ROOT"
# 한 요청에서 새로 훑을 세션 수의 상한. 세션 하나가 수만 행이라 전수 스캔은 화면을
# 멈춘다. 자른 사실은 응답의 sessions_total과 truncated로 나간다.
DEFAULT_SESSION_LIMIT: Final = 8


class PassRefModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    worker: str = Field(min_length=1)
    date: str = Field(min_length=1)
    level: str = Field(min_length=1)
    session: str = Field(min_length=1)
    pass_index: int = Field(ge=0)


class CompareRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: PassRefModel
    reference: PassRefModel
    uv_grid: int = Field(default=8, ge=1, le=64)
    lateral_deg: float = Field(default=30.0, gt=0.0, le=90.0)
    reach_m: float = Field(default=0.8, gt=0.0)
    samples: int = Field(default=20, ge=2, le=200)


def _source_from_env() -> Any | None:
    """환경변수에 루트가 있을 때만 실데이터 소스를 만든다.

    `create_app()`을 인자 없이 부르는 테스트는 루트가 없으므로 지금까지와 똑같이
    `source=None`으로 돌고 모든 비교가 `no_source`가 된다.
    """
    root = os.environ.get(DATA_ROOT_ENV, "").strip()
    if not root:
        return None
    from trainer.pose_lab import csv_source

    return csv_source.CsvPassSource(Path(root))


def create_app(source: Any | None = None) -> FastAPI:
    app = FastAPI(title="pose lab")
    motion_key = importlib.import_module("motion_key")
    if source is None:
        source = _source_from_env()

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/workers")
    def workers() -> dict[str, list[str]]:
        return {"workers": [] if source is None else source.workers()}

    @app.get("/api/passes")
    def passes(
        worker: str = Query(min_length=1),
        level: str | None = None,
        date: str | None = None,
        limit: int = Query(default=DEFAULT_SESSION_LIMIT, ge=1, le=200),
        uv_grid: int = Query(default=8, ge=1, le=64),
        lateral_deg: float = Query(default=30.0, gt=0.0, le=90.0),
        reach_m: float = Query(default=0.8, gt=0.0),
    ) -> dict[str, object]:
        """한 작업자의 패스 목록. 동작 키를 함께 내어 견줄 수 있는 쌍을 눈으로 고른다."""
        if source is None:
            return {"status": "unavailable", "reason": "no_source"}
        params = motion_key.KeyParams(
            uv_grid=uv_grid, lateral_deg=lateral_deg, reach_m=reach_m
        )
        found, total = source.passes(worker, level=level, date=date, limit=limit)
        return {
            "status": "ok",
            "worker": worker,
            "levels": source.levels(worker),
            "sessions_total": total,
            "sessions_scanned": min(total, limit),
            # 잘렸다는 사실을 숨기면 "이 작업자에게 이 패스뿐"으로 읽힌다.
            "truncated": total > limit,
            # 키 재료는 이 리샘플 N 위에서 뽑았다. 비교를 다른 N으로 돌리면 키가
            # 미세하게 달라질 수 있고, 그때는 비교 응답이 낸 키가 사실이다.
            "samples": source.samples,
            "yaw_assumed": service.ASSUMED_YAW,
            "passes": [
                {
                    "worker": item.worker,
                    "date": item.date,
                    "level": item.level,
                    "session": item.session,
                    "pass_index": entry.pass_index,
                    "frames": entry.frames,
                    "length_mm": entry.length_m * 1000.0,
                    "motion_key": source.key_of(entry, params),
                }
                for item, entry in found
            ],
        }

    @app.post("/api/compare")
    def compare(request: CompareRequest) -> dict[str, object]:
        if source is None:
            return {"status": "unavailable", "reason": "no_source"}
        params = motion_key.KeyParams(
            uv_grid=request.uv_grid,
            lateral_deg=request.lateral_deg,
            reach_m=request.reach_m,
        )
        try:
            return service.compare(
                source,
                service.PassRef(**request.query.model_dump()),
                service.PassRef(**request.reference.model_dump()),
                params=params,
                samples=request.samples,
            )
        except LookupError:
            # 없는 패스를 가리킨 것과 프레임이 모자란 것은 다른 사건이다. 목록이
            # 오래돼 사라진 패스를 "재지 못했다"로 적으면 데이터 탓으로 읽힌다.
            return {"status": "unavailable", "reason": "pass_not_found"}

    if STATIC_DIR.is_dir():
        app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
    return app
