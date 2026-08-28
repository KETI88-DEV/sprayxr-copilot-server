#!/usr/bin/env python3
"""pose lab HTTP 표면. demo API(:8120)와 분리된 테스트 도구다."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from trainer.pose_lab import service

STATIC_DIR = Path(__file__).resolve().parent / "static"


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


def create_app(source: Any | None = None) -> FastAPI:
    app = FastAPI(title="pose lab")
    motion_key = importlib.import_module("motion_key")

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/workers")
    def workers() -> dict[str, list[str]]:
        return {"workers": [] if source is None else source.workers()}

    @app.post("/api/compare")
    def compare(request: CompareRequest) -> dict[str, object]:
        if source is None:
            return {"status": "unavailable", "reason": "no_source"}
        params = motion_key.KeyParams(
            uv_grid=request.uv_grid,
            lateral_deg=request.lateral_deg,
            reach_m=request.reach_m,
        )
        return service.compare(
            source,
            service.PassRef(**request.query.model_dump()),
            service.PassRef(**request.reference.model_dump()),
            params=params,
            samples=request.samples,
        )

    if STATIC_DIR.is_dir():
        app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
    return app
