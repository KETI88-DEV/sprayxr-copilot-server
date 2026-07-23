#!/usr/bin/env python3

from __future__ import annotations

import unittest
from typing import Any

# Import the class directly to call the staticmethod
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from agent_pipeline import AgentPipeline


class TestAssembleToolSections(unittest.TestCase):
    """Unit tests for AgentPipeline._assemble_tool_sections MCP envelope unwrap fix."""

    def test_dft_judge_success_envelope_unwraps_inner_result(self) -> None:
        """
        Test case 1: dft_judge success envelope with nested business data.
        Verifies that business fields are read from the INNER result, not envelope top level.
        """
        tool_calls = [
            {
                "name": "dft_judge",
                "arguments": {},
                "result": {
                    "status": "success",
                    "result": {
                        "verdict": "pass",
                        "mean": 95.0,
                        "out_of_spec_count": 0,
                        "nominal": 100,
                        "measurements": [94.5, 95.2, 95.1],
                        "rule": "90/10",
                    },
                    "error": None,
                    "timestamp": "2025-07-22T10:00:00Z",
                },
            }
        ]

        sections = AgentPipeline._assemble_tool_sections(tool_calls)

        self.assertIsNotNone(sections["dft_result"])
        self.assertEqual(sections["dft_result"]["verdict"], "pass")
        self.assertEqual(sections["dft_result"]["mean_um"], 95.0)
        self.assertEqual(sections["dft_result"]["out_of_spec_count"], 0)

    def test_pose_estimation_success_envelope(self) -> None:
        """
        Test case 2: pose_estimation success envelope with inner result.
        Verifies similarity_score and corrections are read from inner result.
        """
        tool_calls = [
            {
                "name": "pose_estimation",
                "arguments": {},
                "result": {
                    "status": "success",
                    "result": {
                        "similarity_score": 0.8,
                        "corrections": [
                            {"joint": "shoulder", "deviation_deg": 5.2},
                            {"joint": "elbow", "deviation_deg": 3.1},
                        ],
                    },
                    "error": None,
                    "timestamp": "2025-07-22T10:00:00Z",
                },
            }
        ]

        sections = AgentPipeline._assemble_tool_sections(tool_calls)

        self.assertIsNotNone(sections["pose_comparison"])
        self.assertEqual(sections["pose_comparison"]["similarity_score"], 0.8)
        self.assertEqual(len(sections["pose_comparison"]["corrections"]), 2)
        self.assertEqual(sections["pose_comparison"]["corrections"][0]["joint"], "shoulder")

    def test_error_envelope_skipped(self) -> None:
        """
        Test case 3: error envelope is skipped, section remains None.
        """
        tool_calls = [
            {
                "name": "dft_judge",
                "arguments": {},
                "result": {
                    "status": "error",
                    "result": None,
                    "error": {"code": "SENSOR_FAILURE", "message": "DFT sensor offline"},
                    "timestamp": "2025-07-22T10:00:00Z",
                },
            }
        ]

        sections = AgentPipeline._assemble_tool_sections(tool_calls)

        self.assertIsNone(sections["dft_result"])

    def test_regression_guard_inner_result_only(self) -> None:
        """
        Test case 4: Regression guard.
        Business fields are ONLY at inner result level (not duplicated at envelope top).
        This would FAIL before the fix (when code tried to read from envelope top level).
        """
        tool_calls = [
            {
                "name": "anomaly_detection",
                "arguments": {},
                "result": {
                    "status": "success",
                    "result": {
                        "detected": True,
                        "anomaly_type": "over_thickness",
                        "zone": "zone_A",
                        "cause": "excessive_spray_pressure",
                        "severity": "critical",
                    },
                    "error": None,
                    "timestamp": "2025-07-22T10:00:00Z",
                },
            }
        ]

        sections = AgentPipeline._assemble_tool_sections(tool_calls)

        self.assertIsNotNone(sections["anomaly"])
        self.assertTrue(sections["anomaly"]["detected"])
        self.assertEqual(sections["anomaly"]["type"], "over_thickness")
        self.assertEqual(sections["anomaly"]["zone"], "zone_A")
        self.assertEqual(sections["anomaly"]["cause"], "excessive_spray_pressure")
        self.assertEqual(sections["anomaly"]["severity"], "critical")

    def test_sequence_validation_success_envelope(self) -> None:
        """
        Test case 5: sequence_validation success envelope.
        Verifies next_path section is populated correctly.
        """
        tool_calls = [
            {
                "name": "sequence_validation",
                "arguments": {},
                "result": {
                    "status": "success",
                    "result": {
                        "zone_id": "zone_B",
                        "recommended_direction": "left_to_right",
                        "reference_pattern": "horizontal_sweep",
                        "waypoints": [[0, 0], [1, 0], [2, 0]],
                        "sequence_violation": False,
                    },
                    "error": None,
                    "timestamp": "2025-07-22T10:00:00Z",
                },
            }
        ]

        sections = AgentPipeline._assemble_tool_sections(tool_calls)

        self.assertIsNotNone(sections["next_path"])
        self.assertEqual(sections["next_path"]["zone_id"], "zone_B")
        self.assertEqual(sections["next_path"]["direction"], "left_to_right")
        self.assertEqual(len(sections["next_path"]["waypoints"]), 3)
        self.assertFalse(sections["next_path"]["violation"])

    def test_multiple_tool_calls_mixed_success_error(self) -> None:
        """
        Test case 6: Multiple tool calls with mixed success/error statuses.
        Verifies that only successful calls populate sections.
        """
        tool_calls = [
            {
                "name": "dft_judge",
                "arguments": {},
                "result": {
                    "status": "success",
                    "result": {
                        "verdict": "fail",
                        "mean": 75.0,
                        "out_of_spec_count": 5,
                    },
                    "error": None,
                    "timestamp": "2025-07-22T10:00:00Z",
                },
            },
            {
                "name": "pose_estimation",
                "arguments": {},
                "result": {
                    "status": "error",
                    "result": None,
                    "error": {"code": "CAMERA_ERROR", "message": "Camera not available"},
                    "timestamp": "2025-07-22T10:00:00Z",
                },
            },
            {
                "name": "anomaly_detection",
                "arguments": {},
                "result": {
                    "status": "success",
                    "result": {
                        "detected": False,
                        "anomaly_type": "none",
                        "zone": "zone_C",
                        "cause": "normal_operation",
                        "severity": "info",
                    },
                    "error": None,
                    "timestamp": "2025-07-22T10:00:00Z",
                },
            },
        ]

        sections = AgentPipeline._assemble_tool_sections(tool_calls)

        # dft_judge succeeded
        self.assertIsNotNone(sections["dft_result"])
        self.assertEqual(sections["dft_result"]["verdict"], "fail")
        self.assertEqual(sections["dft_result"]["mean_um"], 75.0)

        # pose_estimation failed
        self.assertIsNone(sections["pose_comparison"])

        # anomaly_detection succeeded
        self.assertIsNotNone(sections["anomaly"])
        self.assertFalse(sections["anomaly"]["detected"])

    def test_missing_inner_result_defaults_to_empty_dict(self) -> None:
        """
        Test case 7: Envelope with missing inner result defaults to empty dict.
        Verifies graceful handling when inner result is None or missing.
        """
        tool_calls = [
            {
                "name": "dft_judge",
                "arguments": {},
                "result": {
                    "status": "success",
                    "result": None,
                    "error": None,
                    "timestamp": "2025-07-22T10:00:00Z",
                },
            }
        ]

        sections = AgentPipeline._assemble_tool_sections(tool_calls)

        self.assertIsNotNone(sections["dft_result"])
        self.assertEqual(sections["dft_result"]["verdict"], "")
        self.assertIsNone(sections["dft_result"]["mean_um"])

    def test_empty_tool_calls_list(self) -> None:
        """
        Test case 8: Empty tool calls list returns all None sections.
        """
        tool_calls: list[dict[str, Any]] = []

        sections = AgentPipeline._assemble_tool_sections(tool_calls)

        self.assertIsNone(sections["pose_comparison"])
        self.assertIsNone(sections["next_path"])
        self.assertIsNone(sections["dft_result"])
        self.assertIsNone(sections["anomaly"])


if __name__ == "__main__":
    unittest.main()
