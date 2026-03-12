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
        self.assertEqual(payload.get("servers"), 6)
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
        response = client.post("/dft_judge/evaluate", json={"measurements": [100.0]})
        self.assertEqual(response.status_code, 422)

    def test_anomaly_detection_under_thickness(self) -> None:
        self.reset_state()
        client = TestClient(app)
        response = client.post(
            "/anomaly_detection/detect",
            json={
                "image_context": "fixture",
                "dft_measurements": [100.0, 60.0, 100.0],
                "env": {"temperature": 23.0, "humidity": 55.0},
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = to_dict(response_json(response))
        result = to_dict(payload["result"])
        self.assertEqual(result.get("anomaly_detected"), True)
        self.assertEqual(result.get("anomaly_type"), "under_thickness")

    def test_anomaly_detection_no_anomaly(self) -> None:
        self.reset_state()
        client = TestClient(app)
        response = client.post(
            "/anomaly_detection/detect",
            json={
                "image_context": "fixture",
                "dft_measurements": [100.0, 102.0, 98.0],
                "env": {"temperature": 23.0, "humidity": 55.0},
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = to_dict(response_json(response))
        result = to_dict(payload["result"])
        self.assertEqual(result.get("anomaly_detected"), False)
        self.assertEqual(result.get("anomaly_type"), "none")

    def test_sequence_validation_match(self) -> None:
        self.reset_state()
        client = TestClient(app)
        response = client.post(
            "/sequence_validation/validate",
            json={
                "trajectory": [
                    {"x": 0, "y": 10, "z": 0, "timestamp": 1},
                    {"x": 0, "y": 5, "z": 0, "timestamp": 2},
                    {"x": 0, "y": 2, "z": 0, "timestamp": 3},
                ],
                "zone_id": "B-3",
                "zone_type": "수직 패널",
                "reference_pattern": "top_to_bottom",
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = to_dict(response_json(response))
        result = to_dict(payload["result"])
        self.assertEqual(result.get("is_valid"), True)

    def test_sequence_validation_mismatch(self) -> None:
        self.reset_state()
        client = TestClient(app)
        response = client.post(
            "/sequence_validation/validate",
            json={
                "trajectory": [
                    {"x": 0, "y": 1, "z": 0, "timestamp": 1},
                    {"x": 0, "y": 4, "z": 0, "timestamp": 2},
                    {"x": 0, "y": 8, "z": 0, "timestamp": 3},
                ],
                "zone_id": "B-3",
                "zone_type": "수직 패널",
                "reference_pattern": "top_to_bottom",
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = to_dict(response_json(response))
        result = to_dict(payload["result"])
        self.assertEqual(result.get("is_valid"), False)
        violations = cast(list[object], result["violations"])
        first_violation = to_dict(violations[0])
        self.assertEqual(first_violation.get("reason"), "direction_mismatch")

    def test_pose_estimation_compare(self) -> None:
        self.reset_state()
        client = TestClient(app)
        response = client.post(
            "/pose_estimation/compare",
            json={
                "current_pose": {
                    "joints": {
                        "shoulder": {"x": 1, "y": 1, "z": 1},
                        "elbow": {"x": 2, "y": 2, "z": 2},
                    }
                },
                "reference_pose": {
                    "joints": {
                        "shoulder": {"x": 1.5, "y": 1.0, "z": 0.5},
                        "elbow": {"x": 2.0, "y": 1.5, "z": 2.5},
                    }
                },
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = to_dict(response_json(response))
        result = to_dict(payload["result"])
        similarity = cast(float, result["similarity_score"])
        self.assertGreaterEqual(similarity, 0.0)
        self.assertLessEqual(similarity, 1.0)

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
