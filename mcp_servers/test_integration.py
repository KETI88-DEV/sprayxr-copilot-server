from __future__ import annotations

import unittest
import sys
from pathlib import Path
from typing import cast

from httpx import Response
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mcp_servers.main import app
from mcp_servers.timeseries import reset_sessions


JsonDict = dict[str, object]


def to_dict(value: object) -> JsonDict:
    if not isinstance(value, dict):
        raise AssertionError("expected dict payload")
    return cast(JsonDict, value)


def response_json(response: Response) -> object:
    data = cast(object, response.json())
    return data


class IntegrationTests(unittest.TestCase):
    def reset_state(self) -> None:
        reset_sessions()

    def test_health(self) -> None:
        self.reset_state()
        client = TestClient(app)
        response = client.get("/health")
        self.assertEqual(response.status_code, 200)
        payload = to_dict(response_json(response))
        self.assertEqual(payload.get("status"), "ok")
        self.assertEqual(payload.get("servers"), 7)
        self.assertIn("timestamp", payload)

    def test_dft_judge_pass(self) -> None:
        self.reset_state()
        client = TestClient(app)
        response = client.post(
            "/dft_judge/evaluate",
            json={
                "measurements": [100.0, 102.0, 98.0, 101.0],
                "spec": {"nominal": 100.0, "tolerance_rule": "90/10"},
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = to_dict(response_json(response))
        result = to_dict(payload["result"])
        self.assertEqual(result.get("pass_"), True)
        self.assertEqual(result.get("violations"), [])

    def test_dft_judge_fail(self) -> None:
        self.reset_state()
        client = TestClient(app)
        response = client.post(
            "/dft_judge/evaluate",
            json={
                "measurements": [80.0, 90.0, 91.0],
                "spec": {"nominal": 100.0, "tolerance_rule": "90/10"},
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = to_dict(response_json(response))
        result = to_dict(payload["result"])
        self.assertEqual(result.get("pass_"), False)
        violations = cast(list[object], result["violations"])
        self.assertGreaterEqual(len(violations), 1)

    def test_dft_judge_missing_required_field(self) -> None:
        self.reset_state()
        client = TestClient(app)
        response = client.post("/dft_judge/evaluate", json={"spec": {"nominal": 100.0}})
        self.assertEqual(response.status_code, 422)

    def test_anomaly_detection_under_thickness(self) -> None:
        self.reset_state()
        client = TestClient(app)
        response = client.post(
            "/anomaly_detection/detect",
            json={
                "image_context": "fixture",
                "zone_id": "B-3",
                "spec": {"nominal_um": 100.0, "tolerance_rule": "90/10"},
                "dft_measurements": [
                    {"value_um": 100.0, "zone_id": "B-3", "position": "center"},
                    {"value_um": 60.0, "zone_id": "B-3", "position": "lower"},
                    {"value_um": 100.0, "zone_id": "B-3", "position": "upper"}
                ],
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = to_dict(response_json(response))
        result = to_dict(payload["result"])
        self.assertEqual(result.get("anomaly_detected"), True)
        self.assertEqual(result.get("anomaly_count"), 1)
        anomalies = cast(list[object], result["anomalies"])
        first_anomaly = to_dict(anomalies[0])
        self.assertEqual(first_anomaly.get("type"), "저도막")
        self.assertEqual(first_anomaly.get("position"), "lower")

    def test_anomaly_detection_no_anomaly(self) -> None:
        self.reset_state()
        client = TestClient(app)
        response = client.post(
            "/anomaly_detection/detect",
            json={
                "image_context": "fixture",
                "zone_id": "B-3",
                "spec": {"nominal_um": 100.0, "tolerance_rule": "90/10"},
                "dft_measurements": [
                    {"value_um": 100.0, "zone_id": "B-3", "position": "center"},
                    {"value_um": 102.0, "zone_id": "B-3", "position": "lower"},
                    {"value_um": 98.0, "zone_id": "B-3", "position": "upper"}
                ],
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = to_dict(response_json(response))
        result = to_dict(payload["result"])
        self.assertEqual(result.get("anomaly_detected"), False)
        self.assertEqual(result.get("anomaly_count"), 0)
        self.assertEqual(result.get("anomalies"), [])

    def test_sequence_validation_match(self) -> None:
        self.reset_state()
        client = TestClient(app)
        response = client.post(
            "/sequence_validation/validate",
            json={
                "structure_type": "block",
                "current_zone_id": "A-1",
                "zone_map": [
                    {"zone_id": "A-1", "block_id": "BLK-A", "zone_type": "수직 패널", "position": "upper", "is_painted": False, "area_m2": 10.0},
                    {"zone_id": "A-2", "block_id": "BLK-A", "zone_type": "수직 패널", "position": "middle", "is_painted": False, "area_m2": 8.0},
                    {"zone_id": "B-1", "block_id": "BLK-B", "zone_type": "수직 패널", "position": "upper", "is_painted": False, "area_m2": 12.0}
                ]
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = to_dict(response_json(response))
        result = to_dict(payload["result"])
        self.assertEqual(result.get("current_block_id"), "BLK-A")
        self.assertEqual(result.get("next_block"), "BLK-A")
        self.assertEqual(result.get("next_zone"), "A-2")
        self.assertEqual(result.get("current_block_remaining"), ["A-2"])

    def test_sequence_validation_mismatch(self) -> None:
        self.reset_state()
        client = TestClient(app)
        response = client.post(
            "/sequence_validation/validate",
            json={
                "structure_type": "block",
                "current_zone_id": "A-1",
                "zone_map": [
                    {"zone_id": "A-1", "block_id": "BLK-A", "zone_type": "수직 패널", "position": "upper", "is_painted": False, "area_m2": 10.0},
                    {"zone_id": "A-2", "block_id": "BLK-A", "zone_type": "수직 패널", "position": "middle", "is_painted": True, "area_m2": 8.0},
                    {"zone_id": "B-1", "block_id": "BLK-B", "zone_type": "수직 패널", "position": "upper", "is_painted": False, "area_m2": 12.0},
                    {"zone_id": "B-2", "block_id": "BLK-B", "zone_type": "모서리", "position": "lower", "is_painted": False, "area_m2": 3.0}
                ]
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = to_dict(response_json(response))
        result = to_dict(payload["result"])
        self.assertEqual(result.get("current_block_id"), "BLK-A")
        self.assertEqual(result.get("next_block"), "BLK-B")
        self.assertEqual(result.get("next_zone"), "B-1")
        self.assertEqual(result.get("block_sequence"), ["BLK-B"])

    def test_pose_estimation_compare(self) -> None:
        self.reset_state()
        client = TestClient(app)
        response = client.post(
            "/pose_estimation/compare",
            json={
                "mode": "compare",
                "block_id": "BLK-A",
                "current_pose": {
                    "joints": {
                        "wrist_r": {"x": 0.15, "y": -0.02, "z": 0.5, "roll": 2.0, "pitch": 18.0, "yaw": 5.0},
                        "elbow_r": {"x": 0.28, "y": -0.1, "z": 0.35, "roll": 0.0, "pitch": 12.0, "yaw": 0.0}
                    }
                }
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = to_dict(response_json(response))
        result = to_dict(payload["result"])
        self.assertEqual(result.get("expert_pose_id"), "EP-001")
        similarity = cast(float, result["similarity_score"])
        self.assertGreaterEqual(similarity, 0.0)
        self.assertLessEqual(similarity, 1.0)
        corrections = cast(list[object], result["corrections"])
        self.assertGreaterEqual(len(corrections), 1)

    def test_pose_estimation_missing_reference_pose(self) -> None:
        self.reset_state()
        client = TestClient(app)
        response = client.post(
            "/pose_estimation/compare",
            json={"current_pose": {"joints": {"shoulder": {"x": 1, "y": 1, "z": 1}}}},
        )
        self.assertEqual(response.status_code, 422)

    def test_timeseries_update(self) -> None:
        self.reset_state()
        client = TestClient(app)
        response = client.post(
            "/timeseries/update",
            json={
                "session_id": "s1",
                "data": {"timestamp": 1, "speed": 1.0, "distance": 10.0, "angle": 15.0},
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = to_dict(response_json(response))
        result = to_dict(payload["result"])
        self.assertEqual(result.get("count"), 1)

    def test_timeseries_lifecycle(self) -> None:
        self.reset_state()
        client = TestClient(app)
        payloads = [
            {"timestamp": 1, "speed": 1.0, "distance": 10.0, "angle": 10.0},
            {"timestamp": 2, "speed": 2.0, "distance": 11.0, "angle": 11.0},
            {"timestamp": 3, "speed": 3.0, "distance": 12.0, "angle": 12.0},
            {"timestamp": 4, "speed": 4.0, "distance": 13.0, "angle": 13.0},
            {"timestamp": 5, "speed": 5.0, "distance": 14.0, "angle": 14.0},
            {"timestamp": 6, "speed": 6.0, "distance": 15.0, "angle": 15.0},
        ]
        for data in payloads:
            response = client.post("/timeseries/update", json={"session_id": "session-a", "data": data})
            self.assertEqual(response.status_code, 200)

        query_response = client.get("/timeseries/query", params={"session_id": "session-a"})
        self.assertEqual(query_response.status_code, 200)
        payload = to_dict(response_json(query_response))
        result = to_dict(payload["result"])
        self.assertEqual(result.get("count"), 6)
        self.assertEqual(result.get("trend"), "increasing")

    def test_timeseries_drops_oldest_session_past_the_limit(self) -> None:
        from mcp_servers import timeseries

        self.reset_state()
        client = TestClient(app)
        data = {"timestamp": 1.0, "speed": 0.5, "distance": 300.0, "angle": 90.0}
        for index in range(timeseries._MAX_SESSIONS + 1):
            client.post("/timeseries/update", json={"session_id": f"s{index}", "data": data})

        def result_of(session_id: str) -> JsonDict:
            response = client.get("/timeseries/query", params={"session_id": session_id})
            return to_dict(to_dict(response_json(response))["result"])

        self.assertEqual(result_of("s0").get("trend"), "no_data")
        self.assertEqual(result_of(f"s{timeseries._MAX_SESSIONS}").get("count"), 1)

    def test_timeseries_keeps_only_latest_points(self) -> None:
        from unittest.mock import patch

        self.reset_state()
        client = TestClient(app)
        with patch("mcp_servers.timeseries._MAX_POINTS", 3):
            for index in range(5):
                data = {"timestamp": float(index), "speed": 0.5, "distance": 300.0, "angle": 90.0}
                client.post("/timeseries/update", json={"session_id": "capped", "data": data})
        response = client.get("/timeseries/query", params={"session_id": "capped"})
        result = to_dict(to_dict(response_json(response))["result"])
        self.assertEqual(result.get("count"), 3)
        self.assertEqual(to_dict(result["latest"]).get("timestamp"), 4.0)

    def test_timeseries_recently_updated_session_survives(self) -> None:
        from mcp_servers import timeseries

        self.reset_state()
        client = TestClient(app)
        data = {"timestamp": 1.0, "speed": 0.5, "distance": 300.0, "angle": 90.0}
        for index in range(timeseries._MAX_SESSIONS):
            client.post("/timeseries/update", json={"session_id": f"s{index}", "data": data})
        client.post("/timeseries/update", json={"session_id": "s0", "data": data})
        client.post("/timeseries/update", json={"session_id": "new", "data": data})

        def result_of(session_id: str) -> JsonDict:
            response = client.get("/timeseries/query", params={"session_id": session_id})
            return to_dict(to_dict(response_json(response))["result"])

        self.assertEqual(result_of("s0").get("count"), 2)
        self.assertEqual(result_of("s1").get("trend"), "no_data")

    def test_timeseries_concurrent_updates_past_the_limit_do_not_fail(self) -> None:
        import sys
        import threading

        from mcp_servers import timeseries
        from mcp_servers.timeseries import DataPoint, TimeSeriesUpdateRequest, update_timeseries

        self.reset_state()
        point = DataPoint(timestamp=1.0, speed=0.5, distance=300.0, angle=90.0)
        errors: list[BaseException] = []

        def hammer() -> None:
            try:
                for index in range(4000):
                    session_id = f"s{index % (timeseries._MAX_SESSIONS + 6)}"
                    update_timeseries(TimeSeriesUpdateRequest(session_id=session_id, data=point))
            except BaseException as error:  # noqa: BLE001 - 스레드 안 실패를 모아 단언한다
                errors.append(error)

        previous = sys.getswitchinterval()
        sys.setswitchinterval(1e-6)
        try:
            threads = [threading.Thread(target=hammer) for _ in range(8)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        finally:
            sys.setswitchinterval(previous)
        self.assertEqual(errors, [])
        self.assertLessEqual(len(timeseries._sessions), timeseries._MAX_SESSIONS)

    def test_timeseries_missing_session(self) -> None:
        self.reset_state()
        client = TestClient(app)
        response = client.get("/timeseries/query", params={"session_id": "missing"})
        self.assertEqual(response.status_code, 200)
        payload = to_dict(response_json(response))
        self.assertEqual(payload.get("status"), "success")
        result = to_dict(payload["result"])
        self.assertEqual(result.get("count"), 0)
        self.assertEqual(result.get("trend"), "no_data")
        self.assertIsNone(result.get("latest"))

    def test_location_tracking_fixture(self) -> None:
        self.reset_state()
        client = TestClient(app)
        response = client.post(
            "/location_tracking/identify",
            json={
                "coordinates": [
                    {"x": 0, "y": 1, "z": 2, "timestamp": 1},
                    {"x": 1, "y": 2, "z": 3, "timestamp": 2},
                ]
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = to_dict(response_json(response))
        result = to_dict(payload["result"])
        self.assertEqual(result.get("zone_id"), "B-3")
        self.assertEqual(result.get("zone_type"), "수직 패널")

    def test_location_tracking_missing_coordinates(self) -> None:
        self.reset_state()
        client = TestClient(app)
        response = client.post("/location_tracking/identify", json={})
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    _ = unittest.main()
