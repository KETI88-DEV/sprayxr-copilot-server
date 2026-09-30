#!/usr/bin/env python3
"""pose lab HTTP 표면. demo API(:8120)와 분리된 테스트 도구다."""

from __future__ import annotations

import importlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Final

from fastapi import FastAPI, Query
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from trainer.pose_lab import service

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"

# 원본 CSV 트리의 루트. 비어 있으면 소스를 배선하지 않고 지금까지의 거동을 유지한다.
DATA_ROOT_ENV: Final = "POSE_LAB_DATA_ROOT"
# 실측 산포 아티팩트(build_spread_reference의 출력)의 경로. 비어 있거나 파일이 없으면
# service.compare가 잠정 산포로 돈다 — 지금까지의 거동과 같다.
SPREAD_REFERENCE_ENV: Final = "POSE_LAB_SPREAD_REFERENCE"
# 한 요청에서 새로 훑을 세션 수의 상한. 세션 하나가 수만 행이라 전수 스캔은 화면을
# 멈춘다. 자른 사실은 응답의 sessions_total과 truncated로 나간다.
DEFAULT_SESSION_LIMIT: Final = 8
# 숙련자 팔 레퍼런스(build_arm_reference의 출력). **MCP와 같은 환경변수 이름을 쓴다** —
# 랩과 제품이 다른 아티팩트를 보면 눈으로 대조하는 의미가 없다. 없으면 `/api/diagnose`만
# `reference_data_missing`을 내고 나머지 경로는 지금까지대로 돈다.
ARM_REFERENCE_ENV: Final = "ARM_REFERENCE_PATH"
DEFAULT_ARM_REFERENCE: Final = Path("/output/expert_reference/expert_arm_reference.json")


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


def _yaw_source(entries: list[Any]) -> str:
    """이 목록의 동작 키가 실측 요 위에 섰는가.

    요는 이제 HMD 회전에서 뽑은 실측량이라 패스마다 값이 다르다. 목록 응답에 단일
    yaw 값을 실으면 그 하나가 모든 패스의 요인 것처럼 읽히므로 값 대신 출처를 낸다.
    회전 컬럼을 못 읽은 패스는 `service.ASSUMED_YAW`로 떨어져 몸 기준 축이 다시 월드
    방위각이 되는데, `mixed`가 그 사실을 드러낸다.
    """
    if not entries:
        return "assumed"
    measured = sum(1 for entry in entries if entry.key_inputs.yaw is not None)
    if measured == len(entries):
        return "measured"
    return "assumed" if measured == 0 else "mixed"


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


def _reference_spreads_from_env() -> Any | None:
    """환경변수에 경로가 있고 그 경로에 파일이 있을 때만 레퍼런스를 읽는다.

    `create_app()`을 인자 없이 부르는 테스트는 이 환경변수가 없으므로 지금까지처럼
    잠정 산포로 돈다.

    경로가 **설정됐는데** 파일이 없거나 못 읽으면(오타난 경로, 지워진 아티팩트 등)
    잠정 산포로 조용히 떨어뜨리지 않는다 — 그러면 오타난 경로와 환경변수를 아예 안
    준 것이 로그만 봐서는 구별되지 않는다. 폴백 동작 자체는 바꾸지 않는다(계속
    `None`을 낸다) — 여기서 예외를 던지기 시작하지는 않는다.
    """
    path_str = os.environ.get(SPREAD_REFERENCE_ENV, "").strip()
    if not path_str:
        return None
    path = Path(path_str)
    if not path.is_file():
        logger.warning(
            "%s=%s가 설정됐지만 파일이 없다 — 잠정 산포(provisional)로 돈다.",
            SPREAD_REFERENCE_ENV, path,
        )
        return None
    from trainer.pose_lab.spread_reference import SpreadReference

    try:
        loaded = SpreadReference.load(path)
    except (OSError, ValueError) as exc:
        logger.warning(
            "%s=%s를 읽지 못했다(%s) — 잠정 산포(provisional)로 돈다.",
            SPREAD_REFERENCE_ENV, path, exc,
        )
        return None
    # 키 정의가 바뀐 뒤(m1→m3) 옛 아티팩트를 주면 버킷 키가 하나도 맞지 않아 모든 비교가 조용히
    # 잠정 산포로 떨어진다. 오타난 경로와 같은 이유로 경고하고, 섞어 쓰지 않는다.
    expected = importlib.import_module("motion_key").KEY_PREFIX
    prefixes = loaded.key_prefixes()
    if prefixes != {expected}:
        logger.warning(
            "%s=%s의 키 접두어 %s가 현행 %r와 다르다 — 잠정 산포(provisional)로 돈다.",
            SPREAD_REFERENCE_ENV, path, sorted(prefixes), expected,
        )
        return None
    return loaded


def _arm_buckets() -> dict[str, Any] | None:
    """팔 레퍼런스의 버킷. 없거나 깨졌으면 `None` — **지어내지 않는다.**"""
    raw = os.environ.get(ARM_REFERENCE_ENV, "").strip()
    path = Path(raw) if raw else DEFAULT_ARM_REFERENCE
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        logger.warning("팔 레퍼런스를 읽지 못했다: %s", path)
        return None
    buckets = payload.get("buckets")
    return buckets if isinstance(buckets, dict) else None


def create_app(source: Any | None = None) -> FastAPI:
    app = FastAPI(title="pose lab")
    motion_key = importlib.import_module("motion_key")
    if source is None:
        source = _source_from_env()
    reference_spreads = _reference_spreads_from_env()

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
            # 요는 패스마다 실측이므로 목록에 단일 값이 없다. 대신 이 목록의 키가
            # 실측 요 위에 섰는지 대체값으로 떨어진 패스가 섞였는지를 낸다.
            "yaw_source": _yaw_source([entry for _, entry in found]),
            "passes": [
                {
                    "worker": item.worker,
                    "date": item.date,
                    "level": item.level,
                    "session": item.session,
                    "pass_index": entry.pass_index,
                    "frames": entry.frames,
                    "length_mm": entry.length_m * 1000.0,
                    "motion_key": (motion := source.key_of(entry, params)),
                    # 사람이 읽는 이름(동작 유형·길이·작업 기록). 화면은 이것만 쓴다.
                    **service.pass_labels(item.date, item.session, entry.length_m * 1000.0, motion),
                }
                for item, entry in found
            ],
        }

    @app.get("/api/reference")
    def reference(
        worker: str = Query(min_length=1),
        date: str = Query(min_length=1),
        level: str = Query(min_length=1),
        session: str = Query(min_length=1),
        pass_index: int = Query(ge=0),
        uv_grid: int = Query(default=8, ge=1, le=64),
        lateral_deg: float = Query(default=30.0, gt=0.0, le=90.0),
        reach_m: float = Query(default=0.8, gt=0.0),
        samples: int = Query(default=20, ge=2, le=200),
        limit: int = Query(default=50, ge=1, le=500),
    ) -> dict[str, object]:
        """대상 패스의 기준 후보 — 전형기 고정, 같은 레벨, 동작 키 일치, 대상 제외.

        화면은 기준을 고르지 않는다(브리프). 이 목록에서만 고른다 — `[다른 기준 ▸]`이
        `candidates`를 순환할 뿐, 다른 작업자·레벨로 보낼 길이 없다.
        """
        if source is None:
            return {"status": "unavailable", "reason": "no_source"}
        params = motion_key.KeyParams(
            uv_grid=uv_grid, lateral_deg=lateral_deg, reach_m=reach_m
        )
        query = service.PassRef(
            worker=worker, date=date, level=level, session=session, pass_index=pass_index
        )
        return service.reference_candidates(source, query, params, samples=samples, limit=limit)

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
                reference_spreads=reference_spreads,
            )
        except LookupError:
            # 없는 패스를 가리킨 것과 프레임이 모자란 것은 다른 사건이다. 목록이
            # 오래돼 사라진 패스를 "재지 못했다"로 적으면 데이터 탓으로 읽힌다.
            return {"status": "unavailable", "reason": "pass_not_found"}

    @app.get("/api/diagnose")
    def diagnose(
        worker: str = Query(min_length=1),
        date: str = Query(min_length=1),
        level: str = Query(min_length=1),
        session: str = Query(min_length=1),
        pass_index: int = Query(ge=0),
    ) -> dict[str, object]:
        """이 패스를 **배포된 채점 경로**에 그대로 넣으면 무엇이 나오나.

        `/api/compare`와 논리가 다르다. 저쪽은 동작 키로 고른 **기준 패스 하나**에
        대고 재고, 이쪽은 reach 밴드로 고른 숙련자 **버킷**에 DTW로 맞춰 잰다.
        기준 패스를 받지 않는 이유가 그것이다 — 제품에는 그런 입력이 없다.

        레퍼런스는 **요청마다 읽는다.** 아티팩트를 다시 굽고 랩을 재시작하지 않아도
        되게 — 랩은 그걸 보는 도구다.
        """
        if source is None:
            return {"status": "unavailable", "reason": "no_source"}
        return service.diagnose_sequence(
            source,
            service.PassRef(worker=worker, date=date, level=level,
                            session=session, pass_index=pass_index),
            _arm_buckets(),
        )

    if STATIC_DIR.is_dir():
        app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
    return app
