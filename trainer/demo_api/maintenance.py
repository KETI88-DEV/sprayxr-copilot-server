"""점검 배너: 운영 호스트의 ops_lock.py 가 쓴 maintenance.json 을 화면에 알린다.

파일은 호스트 /home/keti/shipcoat/results/ops/maintenance.json = 컨테이너 /output/eval_results/ops/maintenance.json.
읽지 못하거나 기한이 지났으면 점검이 아닌 것으로 본다. 배너 때문에 화면이 깨지면 안 되므로 예외를 던지지 않는다.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse, JSONResponse

DEFAULT_FILE = "/output/eval_results/ops/maintenance.json"
_INACTIVE = {"active": False}


def read_maintenance(path: Path, now: datetime) -> dict[str, object]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or data.get("active") is not True:
            return dict(_INACTIVE)
        until = datetime.fromisoformat(str(data["until"]).replace("Z", "+00:00"))
        if until.tzinfo is None or until <= now:
            return dict(_INACTIVE)
    except (OSError, ValueError, KeyError):
        return dict(_INACTIVE)
    text = {key: data.get(key) if isinstance(data.get(key), str) else "" for key in ("what", "reason", "impact")}
    return {"active": True, **text, "until": until.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}


def create_maintenance_router() -> APIRouter:
    router = APIRouter(include_in_schema=False)

    @router.get("/api/maintenance")
    def maintenance() -> JSONResponse:
        path = Path(os.environ.get("MAINTENANCE_FILE", DEFAULT_FILE))
        return JSONResponse(read_maintenance(path, datetime.now(timezone.utc)), headers={"Cache-Control": "no-store"})

    @router.get("/assets/maintenance_banner.js")
    def banner_script() -> FileResponse:
        return FileResponse(Path(__file__).with_name("static") / "maintenance_banner.js",
                            media_type="text/javascript", headers={"Cache-Control": "no-cache"})

    return router
