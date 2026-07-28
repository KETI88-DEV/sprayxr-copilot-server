#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx"]
# ///
# ─── How to run ───
#   데모 API 기동 후 (GPU 불필요):
#     python3 -m trainer.demo_api.app --host 127.0.0.1 --port 8120 &
#     python3 trainer/scripts/sim_dummy_driver.py --api-url http://127.0.0.1:8120 --events 3 --anomaly
#   각 호출은 모니터에 input 1건 + output 1건을 남긴다.
from __future__ import annotations

import argparse
import json

import httpx

_STANDOFF_OUT_OF_BAND_MM = 350.0
_ANGLE_OUT_OF_BAND_DEG = 55.0
_STANDOFF_IN_BAND_MM = 300.0
_ANGLE_IN_BAND_DEG = 72.0


def build_dummy_session(
    session_id: str, events: int, anomaly: bool
) -> tuple[dict[str, object], list[dict[str, object]], dict[str, object]]:
    """Build the start / runtime / session_end payloads for one dummy session."""
    start: dict[str, object] = {
        "session_id": session_id,
        "level_id": "SP0035-L01",
        "worker": {"trainee_id": "t-dummy", "skill_level": "beginner", "handedness": "right"},
        "spec": {"nominal_um": 100.0},
    }

    event_payloads: list[dict[str, object]] = []
    for index in range(events):
        if anomaly and index == 0:
            event_payloads.append(
                {
                    "event_id": f"ev-{index}",
                    "telemetry": {
                        "gun_metrics": {
                            "speed_m_per_s": 0.8,
                            "standoff_mm": _STANDOFF_OUT_OF_BAND_MM,
                            "angle_deg": _ANGLE_OUT_OF_BAND_DEG,
                        }
                    },
                }
            )
        elif index % 2 == 0:
            event_payloads.append(
                {
                    "event_id": f"ev-{index}",
                    "telemetry": {
                        "gun_metrics": {
                            "speed_m_per_s": 0.8,
                            "standoff_mm": _STANDOFF_IN_BAND_MM,
                            "angle_deg": _ANGLE_IN_BAND_DEG,
                        }
                    },
                }
            )
        else:
            event_payloads.append({"event_id": f"ev-{index}", "user_request": "에폭시 도료란?"})

    complete: dict[str, object] = {
        "session_id": session_id,
        "scenario_id": "dummy",
        "mode": "live",
        "zone_id": "A-2",
        "zone_type": "panel",
        "metrics": {
            "avg_dft": 100.0,
            "dft_values": [95.0, 102.0, 88.0],
            "avg_speed": 0.8,
            "speed_variance": 0.05,
            "avg_distance": 0.3,
            "distance_variance": 0.05,
            "avg_angle": 70.0,
            "angle_consistency": 0.9,
            "sequence_correct": not anomaly,
            "under_thickness_count": 1 if anomaly else 0,
        },
    }
    return start, event_payloads, complete


def run(api_url: str, session_id: str, events: int, anomaly: bool) -> None:
    start, event_payloads, complete = build_dummy_session(session_id, events, anomaly)
    base = api_url.rstrip("/")

    with httpx.Client(timeout=10.0) as client:
        started = client.post(f"{base}/api/sim/sessions/start", json=start)
        print("start ->", started.status_code, json.dumps(started.json(), ensure_ascii=False))

        for payload in event_payloads:
            response = client.post(f"{base}/api/sim/sessions/{session_id}/events", json=payload)
            print("event ->", response.status_code, json.dumps(response.json(), ensure_ascii=False))

        completed = client.post(f"{base}/api/sim/sessions/{session_id}/complete", json=complete)
        print("complete ->", completed.status_code, json.dumps(completed.json(), ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Dummy simulator interface driver.")
    _ = parser.add_argument("--api-url", default="http://127.0.0.1:8120")
    _ = parser.add_argument("--session-id", default="dummy-session")
    _ = parser.add_argument("--events", type=int, default=3)
    _ = parser.add_argument("--anomaly", action="store_true")
    args = parser.parse_args()

    run(args.api_url, args.session_id, args.events, args.anomaly)


if __name__ == "__main__":
    main()
