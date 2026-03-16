from __future__ import annotations

import argparse
import logging
from typing import cast

import uvicorn
from fastapi import FastAPI

from mcp_servers.anomaly_detection import router as anomaly_detection_router
from mcp_servers.common import iso_timestamp, register_error_handlers
from mcp_servers.dft_judge import router as dft_judge_router
from mcp_servers.location_tracking import router as location_tracking_router
from mcp_servers.pose_estimation import router as pose_estimation_router
from mcp_servers.sequence_validation import router as sequence_validation_router
from mcp_servers.timeseries import router as timeseries_router
from mcp_servers.vlm_analysis import router as vlm_analysis_router


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Ship Painting MCP Server Aggregator")
register_error_handlers(app)

app.include_router(dft_judge_router)
app.include_router(anomaly_detection_router)
app.include_router(sequence_validation_router)
app.include_router(pose_estimation_router)
app.include_router(timeseries_router)
app.include_router(location_tracking_router)
app.include_router(vlm_analysis_router)


@app.get("/health")
def health() -> dict[str, str | int]:
    return {
        "status": "ok",
        "servers": 7,
        "timestamp": iso_timestamp(),
    }


def parse_args() -> tuple[str, int, bool]:
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("--host", default="0.0.0.0")
    _ = parser.add_argument("--port", type=int, default=8100)
    _ = parser.add_argument("--reload", action="store_true")

    raw_args = cast(dict[str, object], vars(parser.parse_args()))
    host_obj = raw_args.get("host", "0.0.0.0")
    port_obj = raw_args.get("port", 8100)
    reload_obj = raw_args.get("reload", False)

    host = host_obj if isinstance(host_obj, str) else "0.0.0.0"
    port = port_obj if isinstance(port_obj, int) else 8100
    reload_mode = reload_obj if isinstance(reload_obj, bool) else False
    return host, port, reload_mode


def main() -> None:
    host, port, reload_mode = parse_args()
    logger.info("Starting MCP server on %s:%s", host, port)
    uvicorn.run("mcp_servers.main:app", host=host, port=port, reload=reload_mode)


if __name__ == "__main__":
    main()
