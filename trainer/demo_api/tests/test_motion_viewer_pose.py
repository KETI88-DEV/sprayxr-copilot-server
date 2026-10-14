from __future__ import annotations

from typing import Any

import pytest

from trainer.demo_api.motion_viewer_pose import build_pose_view
from trainer.demo_api.pose_query import PoseEngineResult


def _body(frame_count: int = 5) -> dict[str, Any]:
    return {
        "session_id": "3_11000346_1",
        "sequence_id": "SEQ-020",
        "scenario_id": "3",
        "zone_id": "7",
        "sequence_info": [
            {
                "index": index,
                "time": f"2026-10-08T00:00:{index:02d}Z",
                "skeleton_joints": {
                    "Head": [0.0, 1.7, index * 0.01],
                    "RightHand": [0.3, 1.2, index * 0.01],
                    "LeftHand": [-0.3, 1.2, index * 0.01],
                },
                "tool_pos_rot": {"pos": [0.35, 1.2, index * 0.01]},
            }
            for index in range(frame_count)
        ],
    }


def _raw(**over: Any) -> PoseEngineResult:
    result: dict[str, Any] = {
        "analysis_status": "available",
        "similarity_score": 0.82,
        "expert_percentile": 70,
        "reference_quality": "multi_worker",
        "support": 150,
        "summary": "숙련자 자세와 대체로 유사합니다.",
        "corrections": [
            {
                "joint": "RightHand",
                "correction_text": "오른손을 5 cm 올리세요.",
                "severity": "medium",
                "evidence": {"sigma": 2.1},
                "target": {"per_frame": [{"index": 2, "point": [0.3, 1.25, 0.02]}]},
            },
            {
                "joint": "LeftHand",
                "correction_text": "왼손을 몸쪽으로 당기세요.",
                "severity": "low",
                "evidence": {"sigma": 1.1},
                "target": {"per_frame": []},
            },
        ],
        "burst": {
            "strokes": [
                {"start": 0, "end": 2, "similarity_score": 0.85},
                {"start": 3, "end": 4, "similarity_score": 0.78},
            ]
        },
        "reference_layer": {
            "RightHand": [[0.2, 0.4, 0.1], [0.2, 0.5, 0.1]],
        },
    }
    result.update(over)
    return PoseEngineResult("available", None, "SEQ-020", result)


def _verified_mesh(_mode: str) -> dict[str, Any]:
    return {
        "available": True,
        "vertices": [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        "triangles": [[0, 1, 2]],
        "reason": None,
    }


def test_build_pose_view_reports_sequence_identity_and_product_score() -> None:
    view = build_pose_view(_body(), _raw(), mesh_provider=_verified_mesh)

    assert view["identity"] == {
        "session_id": "3_11000346_1",
        "sequence_id": "SEQ-020",
        "scenario_id": "3",
        "mode_id": 3,
        "zone_id": "7",
        "unit_label": "SEQ-020",
    }
    assert view["evaluation"]["similarity_score"] == 0.82
    assert view["evaluation"]["expert_percentile"] == 70
    assert view["evaluation"]["verdict"] == "양호"
    assert view["evaluation"]["confidence"] == "높음"
    assert view["evaluation"]["top_correction"]["joint"] == "RightHand"


def test_build_pose_view_labels_strokes_as_secondary_units() -> None:
    view = build_pose_view(_body(), _raw(), mesh_provider=_verified_mesh)

    assert view["identity"]["unit_label"] == "SEQ-020"
    assert [stroke["label"] for stroke in view["strokes"]] == [
        "스트로크 1/2", "스트로크 2/2",
    ]
    assert all(stroke["secondary"] is True for stroke in view["strokes"])


def test_build_pose_view_downsamples_tracks_but_keeps_first_last_and_correction_frames() -> None:
    raw = _raw()
    raw.result["corrections"][0]["target"]["per_frame"] = [
        {"index": 50, "point": [0.3, 1.25, 0.5]}
    ]

    view = build_pose_view(_body(100), raw, mesh_provider=_verified_mesh, max_frames=10)
    indices = [frame["index"] for frame in view["layers"]["measured"]["frames"]]

    assert len(indices) <= 10
    assert {0, 50, 99}.issubset(indices)


def test_build_pose_view_keeps_measured_reference_and_corrected_layers_separate() -> None:
    view = build_pose_view(_body(), _raw(), mesh_provider=_verified_mesh)
    layers = view["layers"]

    assert layers["measured"]["coordinate_space"] == "world"
    assert layers["measured"]["color"] == "#38BDF8"
    assert layers["expert"]["coordinate_space"] == "body_normalized"
    assert layers["expert"]["color"] == "#22C55E"
    assert layers["corrected"]["coordinate_space"] == "world"
    assert layers["corrected"]["color"] == "#FACC15"
    assert layers["measured"] is not layers["expert"]


@pytest.mark.parametrize(
    ("body", "raw", "reason", "message_part"),
    [
        (None, None, "missing_sequence_body", "시퀀스 본문"),
        (
            {**_body(), "sequence_info": [{"index": 0}]},
            PoseEngineResult("not_available", "quality_insufficient", "SEQ-020", {}),
            "quality_insufficient",
            "골격",
        ),
        (
            _body(),
            PoseEngineResult("not_available", "engine_unavailable", "SEQ-020", {}),
            "engine_unavailable",
            "엔진",
        ),
    ],
)
def test_build_pose_view_explains_unavailable_evaluation_without_fake_score(
    body: dict[str, Any] | None,
    raw: PoseEngineResult | None,
    reason: str,
    message_part: str,
) -> None:
    view = build_pose_view(body, raw, mesh_provider=_verified_mesh)

    assert view["evaluation"]["analysis_status"] == "not_available"
    assert view["evaluation"]["reason_code"] == reason
    assert view["evaluation"]["similarity_score"] is None
    assert message_part in view["evaluation"]["message"]


def test_build_pose_view_marks_missing_expert_layer_without_hiding_real_score() -> None:
    raw = _raw()
    raw.result.pop("reference_layer")

    view = build_pose_view(_body(), raw, mesh_provider=_verified_mesh)

    assert view["evaluation"]["similarity_score"] == 0.82
    assert view["layers"]["expert"]["available"] is False
    assert "전문가" in view["layers"]["expert"]["message"]


@pytest.mark.parametrize("mesh", [{"available": False, "reason": "unverified_alignment"}, None])
def test_build_pose_view_uses_ray_point_fallback_for_unusable_mesh(mesh: Any) -> None:
    view = build_pose_view(_body(), _raw(), mesh_provider=lambda _mode: mesh)

    assert view["object_mesh"]["available"] is False
    assert view["object_mesh"]["vertices"] == []
    assert view["object_mesh"]["triangles"] == []
    assert view["object_mesh"]["fallback"] == "ray_point_surface"


def test_build_pose_view_keeps_verified_mesh_triangles() -> None:
    view = build_pose_view(_body(), _raw(), mesh_provider=_verified_mesh)

    assert view["object_mesh"]["available"] is True
    assert view["object_mesh"]["triangles"] == [[0, 1, 2]]
    assert view["object_mesh"]["fallback"] is None
