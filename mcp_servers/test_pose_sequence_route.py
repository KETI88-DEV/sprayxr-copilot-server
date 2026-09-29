#!/usr/bin/env python3
"""`POST /pose_estimation/sequence` 검사.

**실물 샘플을 그대로 넣는다.** 손으로 지어낸 본문은 우리가 상상한 계약을 확인할 뿐이다 —
이 경로를 처음 실물에 물렸을 때 `time`이 ISO-8601 문자열이라 정상 시퀀스가 전부 422였고,
그건 손으로 쓴 본문으로는 잡히지 않았다.

이 파일이 지키려는 것 둘:

1. **점수가 `0.0`이 아니라 `null`이다.** 레퍼런스가 없으면 안 재는 것이지 "많이 다르다"가 아니다.
2. **키 집합이 레퍼런스가 준비된 세계와 같다.** 거부든 통과든 같은 키가 나가야
   소비자가 나중에 고치지 않는다.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from mcp_servers import pose_estimation
from mcp_servers.main import app
from trainer.scripts import posture_sequence

_SAMPLE = Path(__file__).resolve().parents[1] / "docs" / "SprayXR_CoPilot_샘플_2026-09-21.json"

# 계약이 약속한 키. 값이 아니라 **키 집합**이 고정이다.
CONTRACT_KEYS = {
    "mode",
    "sequence_id",
    "analysis_status",
    "reason_code",
    "scoring_set_id",
    "scored_joints",
    "excluded_joints",
    "reference_quality",
    "support",
    "similarity_score",
    "expert_percentile",
    "correction_mode",
    "corrections",
    "input_quality",
    "posture_sequence",
    "summary",
}


@pytest.fixture
def client() -> TestClient:
    # 모듈 전역 저장소라 검사끼리 세션이 새면 기울기·R4c 결과가 앞 검사에 좌우된다.
    pose_estimation._SESSIONS = posture_sequence.SessionStore()
    return TestClient(app)


def _sample_bodies() -> list[dict[str, Any]]:
    doc = json.loads(_SAMPLE.read_text(encoding="utf-8"))
    return [r["body"] for r in doc["requests"] if r["endpoint"] == "POST /copilot/analyze-sequence"]


def _post(client: TestClient, body: dict[str, Any]) -> dict[str, Any]:
    response = client.post("/pose_estimation/sequence", json=body)
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["status"] == "success"
    return payload["result"]


@pytest.mark.skipif(not _SAMPLE.is_file(), reason="실물 샘플이 이 작업 트리에만 있다")
def test_the_real_sample_bodies_are_accepted() -> None:
    """실물 두 시퀀스가 200이어야 한다. 422면 우리가 계약을 잘못 읽은 것이다."""
    bodies = _sample_bodies()
    assert len(bodies) == 2, "샘플의 analyze-sequence 요청 수가 달라졌다"
    with TestClient(app) as client:
        for body in bodies:
            assert client.post("/pose_estimation/sequence", json=body).status_code == 200


@pytest.mark.skipif(not _SAMPLE.is_file(), reason="실물 샘플이 이 작업 트리에만 있다")
def test_iso8601_time_parses(client: TestClient) -> None:
    """`time`이 ISO-8601 문자열이다 — 실물이 그렇다. 숫자로만 받으면 정상 시퀀스가 422가 된다.

    깨뜨리는 법: `SequenceFrame.time`을 `float | None`으로 되돌리면 실패한다.
    """
    body = _sample_bodies()[1]
    assert isinstance(body["sequence_info"][0]["time"], str)
    result = _post(client, body)
    quality = result["input_quality"]["measurements"]
    # 120FPS다. 간격이 읽혔다는 증거는 이 키가 있다는 것이다.
    assert quality["frame_interval_min_s"] == pytest.approx(0.008, abs=1e-3)


@pytest.mark.skipif(not _SAMPLE.is_file(), reason="실물 샘플이 이 작업 트리에만 있다")
def test_seq001_is_rejected_because_the_gun_is_not_in_hand(client: TestClient) -> None:
    """설계 문서 §6이 예고한 판정이다 — R5 `tool_not_held`, 실측 4.961 m."""
    result = _post(client, _sample_bodies()[0])
    assert result["analysis_status"] == "not_available"
    assert result["reason_code"] == "tool_not_held"
    assert result["input_quality"]["measurements"]["hand_tool_gap_m"] == pytest.approx(4.961, abs=1e-3)
    assert result["posture_sequence"]["posture_point_recorded"] is False, (
        "거부한 시퀀스의 자세 점이 세션에 쌓이면 기울기가 오염된다"
    )
    assert result["posture_sequence"]["sequences"] == 0


@pytest.mark.skipif(not _SAMPLE.is_file(), reason="실물 샘플이 이 작업 트리에만 있다")
def test_the_key_set_is_the_same_whether_rejected_or_not(client: TestClient) -> None:
    """거부든 통과든 같은 키가 나간다. 레퍼런스가 준비돼도 키는 그대로고 값만 바뀐다."""
    rejected, accepted = (_post(client, body) for body in _sample_bodies())
    assert set(rejected) == CONTRACT_KEYS
    assert set(accepted) == CONTRACT_KEYS


@pytest.mark.skipif(not _SAMPLE.is_file(), reason="실물 샘플이 이 작업 트리에만 있다")
def test_the_score_is_null_not_zero(client: TestClient) -> None:
    """`0.0`은 '많이 다르다'로 읽힌다. 우리는 재지 않았다.

    깨뜨리는 법: `similarity_score`에 `0.0`을 넣으면 실패한다.
    """
    result = _post(client, _sample_bodies()[1])
    assert result["similarity_score"] is None
    assert result["reference_quality"] == "unavailable"
    assert result["reason_code"] == "reference_rebuild_pending"
    assert result["expert_percentile"] is None


def test_unknown_top_level_keys_fail_loudly(client: TestClient) -> None:
    """최상위는 `forbid`다. 안 읽은 신호를 보낸 쪽이 읽혔다고 믿으면 안 된다."""
    response = client.post(
        "/pose_estimation/sequence",
        json={"session_id": "S", "sequence_id": "Q", "posture_hint": "crouch"},
    )
    assert response.status_code == 422


def test_unknown_frame_keys_are_reported_not_rejected(client: TestClient) -> None:
    """프레임 이하까지 `forbid`로 잠그면 벤더 확장 하나가 시퀀스를 죽인다.

    받되 **세어서 응답에 싣는다** — 조용히 버리는 것과 다르다.
    """
    result = _post(
        client,
        {
            "session_id": "S",
            "sequence_id": "Q",
            "sequence_info": [
                {"index": 1, "skeleton_joints": {"head": [0.0, 1.5, 0.0]}, "gun_pressure_psi": 324}
            ],
        },
    )
    assert result["input_quality"]["unknown_fields"] == ["gun_pressure_psi"]


def test_unknown_joint_names_are_reported(client: TestClient) -> None:
    """모르는 관절 이름도 같은 취지다. `joint_names.to_code`가 `None`을 내는 쪽이다."""
    result = _post(
        client,
        {
            "session_id": "S",
            "sequence_id": "Q",
            "sequence_info": [{"skeleton_joints": {"head": [0.0, 1.5, 0.0], "tailbone": [0, 0, 0]}}],
        },
    )
    assert "tailbone" in result["input_quality"]["unknown_fields"]


def _synthetic(session_id: str, sequence_id: str, head: float, impact: float) -> dict[str, Any]:
    """게이트를 통과하는 최소 시퀀스. 리그 치수는 실측(어깨 사이 0.0567 m)에서 왔다."""
    joints = {
        "pelvis": [0.0, head - 0.75, 0.0],
        "spine": [0.0, head - 0.55, 0.0],
        "chest": [0.0, head - 0.35, 0.0],
        "head": [0.0, head, 0.0],
        "leftshoulder": [-0.02835, head - 0.33, 0.0],
        "rightshoulder": [0.02835, head - 0.33, 0.0],
        "leftupperarm": [-0.15, head - 0.33, 0.0],
        "rightupperarm": [0.15, head - 0.33, 0.0],
        "leftforearm": [-0.15, head - 0.58, 0.0],
        "rightforearm": [0.15, head - 0.58, 0.0],
        "lefthand": [-0.15, head - 0.80, 0.0],
        "righthand": [0.15, head - 0.80, 0.0],
        "lefttoe": [-0.1, head - 1.0, 0.1],
        "righttoe": [0.1, head - 1.0, 0.1],
    }
    frames = []
    for index in range(4):
        moved = {name: list(value) for name, value in joints.items()}
        # 전부 얼면 R4가 잡는다. 손만 움직이는 것이 실제 분사 시퀀스의 모습이다.
        moved["righthand"][0] += 0.01 * index
        frames.append(
            {
                "index": index,
                "time": 1000.0 + 0.009 * index,
                "tool_pos_rot": {"pos": [0.15 + 0.05 * index, head - 0.80, 0.2], "rot": [0, 0, 0]},
                "skeleton_joints": moved,
                "gun_metrics": {"raypoint": [1.0, impact, 0.0]},
            }
        )
    return {"session_id": session_id, "sequence_id": sequence_id, "sequence_info": frames}


def test_the_slope_appears_only_after_enough_sequences(client: TestClient) -> None:
    """추종 기울기는 시퀀스를 가로질러 생긴다 — 하나로는 낼 수 없다.

    착탄이 1 m 오를 때 몸이 0.6 m 오르게 넣으면 그 값이 나와야 한다. 숙련자 실측 중앙값은 +0.654다.
    """
    slope_seen = []
    for index in range(posture_sequence.MIN_SAMPLES_FOR_SLOPE):
        impact = 4.5 + 0.25 * index
        result = _post(client, _synthetic("W1", f"SEQ-{index}", 1.5 + 0.6 * 0.25 * index, impact))
        slope_seen.append(result["posture_sequence"]["tracking_slope"])

    assert slope_seen[:-1] == [None] * (posture_sequence.MIN_SAMPLES_FOR_SLOPE - 1)
    assert slope_seen[-1] == pytest.approx(0.6, abs=1e-6)


def test_sessions_do_not_bleed_into_each_other(client: TestClient) -> None:
    """작업자 두 명이 같은 프로세스를 쓴다. 세션이 섞이면 남의 자세가 내 기울기에 든다."""
    for index in range(posture_sequence.MIN_SAMPLES_FOR_SLOPE):
        _post(client, _synthetic("W1", f"SEQ-{index}", 1.5 + 0.15 * index, 4.5 + 0.25 * index))
    other = _post(client, _synthetic("W2", "SEQ-0", 1.5, 4.5))
    assert other["posture_sequence"]["sequences"] == 1
    assert other["posture_sequence"]["tracking_slope"] is None


def test_every_reason_code_is_in_the_closed_set(client: TestClient) -> None:
    """지어낸 사유가 나가면 소비자가 모르는 값을 받는다.

    깨뜨리는 법: 핸들러에서 `reason`에 없는 문자열을 넣으면 `assert`가 잡는다.
    """
    bodies = [
        {"session_id": "S", "sequence_id": "EMPTY", "sequence_info": []},
        {"session_id": "S", "sequence_id": "THIN", "sequence_info": [{"skeleton_joints": {}}]},
        _synthetic("S", "OK", 1.5, 4.5),
    ]
    for body in bodies:
        result = _post(client, body)
        assert result["reason_code"] in pose_estimation.SEQUENCE_REASON_CODES


def test_too_few_scored_joints_withholds_the_score(client: TestClient) -> None:
    """분모가 관절 수다. 모자란 관절로 낸 점수는 다른 점수와 비교할 수 없다."""
    result = _post(
        client,
        {
            "session_id": "S",
            "sequence_id": "Q",
            "sequence_info": [
                {"skeleton_joints": {"pelvis": [0, 0.75, 0], "chest": [0, 1.15, 0],
                                     "leftshoulder": [-0.02835, 1.17, 0],
                                     "rightshoulder": [0.02835, 1.17, 0]}}
            ],
        },
    )
    assert len(result["scored_joints"]) < posture_sequence.MIN_SCORED_JOINTS
    assert result["reason_code"] == "quality_insufficient"
    assert result["similarity_score"] is None


def test_the_legacy_compare_route_still_answers(client: TestClient) -> None:
    """새 경로를 붙이면서 `/compare`를 건드리지 않았다는 증거."""
    response = client.post(
        "/pose_estimation/compare",
        json={"mode": "recommend", "block_id": "BLK-A"},
    )
    assert response.status_code == 200
    assert response.json()["result"]["recommended_expert_pose_id"] == "EP-001"


@pytest.mark.skipif(not _SAMPLE.is_file(), reason="실물 샘플이 이 작업 트리에만 있다")
def test_a_joint_frozen_twice_in_one_session_drops_out(client: TestClient) -> None:
    """R4c다 — 한 번은 우연이지만 두 번은 그 관절이 안 오는 것이다.

    실물 SEQ-002가 이 모양이다: 골격이 전부 얼었지만 **건도 정지**해서 한 번만으로는
    거부하지 않는다(R4a). 같은 세션에 한 번 더 오면 그때 채점에서 뺀다.

    이 항이 세션 상태가 실제로 쓰인다는 유일한 증거다 — `frozen_before`를 빈 집합으로
    바꾸면 여기서만 실패한다.
    """
    body = _sample_bodies()[1]
    first = _post(client, body)
    assert first["excluded_joints"] == [], "첫 관측만으로 빼면 정지 시퀀스가 전부 죽는다"
    assert first["input_quality"]["status"] == "pass"

    second = _post(client, {**body, "sequence_id": "SEQ-002-again"})
    assert second["input_quality"]["status"] == "degrade"
    assert "skeleton_joint_frozen" in second["input_quality"]["reason_codes"]
    assert "Head" in second["excluded_joints"]
    assert second["reason_code"] == "quality_insufficient"
    assert second["similarity_score"] is None
