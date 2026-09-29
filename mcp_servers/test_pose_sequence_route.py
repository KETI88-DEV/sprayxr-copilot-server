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
import math
import statistics
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from mcp_servers import pose_estimation
from mcp_servers.main import app
from trainer.scripts import arm_reference, joint_names, posture_sequence, world_axes

def _length(point) -> float:
    return math.sqrt(sum(value * value for value in point))


def _unit(point):
    length = _length(point)
    return tuple(value / length for value in point) if length > 1e-9 else (1.0, 0.0, 0.0)


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
    "reference_band",
    "reach_m",
    "support",
    "similarity_score",
    "expert_percentile",
    "correction_mode",
    "scoring_mode",
    "corrections",
    "joint_sigmas",
    "input_quality",
    "posture_sequence",
    "summary",
}


@pytest.fixture
def client(monkeypatch, tmp_path: Path) -> TestClient:
    # 모듈 전역 저장소라 검사끼리 세션이 새면 기울기·R4c 결과가 앞 검사에 좌우된다.
    pose_estimation._SESSIONS = posture_sequence.SessionStore()
    # `/output`은 마운트된 산출물이지 저장소 데이터가 아니다(AGENTS.md). 실물 아티팩트가
    # 있든 없든 같은 결과가 나오도록 **검사마다 자기 레퍼런스를 깐다.**
    monkeypatch.setenv(pose_estimation.ARM_REFERENCE_ENV, str(tmp_path / "none.json"))
    return TestClient(app)


def _write_reference(tmp_path: Path, monkeypatch, *, sd: float = 0.05,
                     n: int = 200, workers: int = 6) -> Path:
    """검사용 숙련자 레퍼런스. 평균은 아래 `_synthetic`이 내는 자세에서 뽑는다."""
    body = _synthetic("REF", "REF", 1.5, 4.5)
    frames, _ = pose_estimation._gate_frames(
        [pose_estimation.SequenceFrame(**frame) for frame in body["sequence_info"]]
    )
    normalized = arm_reference.normalize_sequence([frame.joints for frame in frames])
    assert normalized is not None
    # reach는 **건 기준**이다(착탄 수직 − 건 수직). 하드코딩하지 않고 본문에서 그대로
    # 뽑는다 — 경로가 쓰는 정의가 바뀌면 이 붙박이가 조용히 어긋난다.
    impacts = [
        world_axes.vertical_of(tuple(frame["gun_metrics"]["raypoint"]))
        for frame in body["sequence_info"]
    ]
    guns = [
        world_axes.vertical_of(tuple(frame["tool_pos_rot"]["pos"]))
        for frame in body["sequence_info"]
    ]
    reach = statistics.median(impacts) - statistics.median(guns)

    path = tmp_path / "arm_reference.json"
    path.write_text(json.dumps({
        "version": 1,
        "scoring_set_id": arm_reference.SCORING_SET_ID,
        "reach_band_m": arm_reference.REACH_BAND_M,
        "buckets": {
            arm_reference.reach_band(reach + offset): {
                "n": n,
                "workers": workers,
                "joints": {
                    name: {
                        "mean": list(point),
                        "sd": [sd, sd, sd],
                        "n": n,
                        "radius": {"mean": _length(point), "sd": sd},
                        "heading": {"mean": list(_unit(point)), "angular_sd": 0.40},
                    }
                    for name, point in normalized.items()
                },
            }
            # 이웃 밴드까지 덮는다 — 프레임마다 몸이 조금 움직여 reach가 밴드 경계를 넘나든다.
            for offset in (-arm_reference.REACH_BAND_M, 0.0, arm_reference.REACH_BAND_M)
        },
    }), encoding="utf-8")
    monkeypatch.setenv(pose_estimation.ARM_REFERENCE_ENV, str(path))
    return path


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
def test_the_score_is_null_not_zero_when_there_is_no_reference(client: TestClient) -> None:
    """`0.0`은 '많이 다르다'로 읽힌다. 레퍼런스가 없으면 우리는 재지 않은 것이다.

    깨뜨리는 법: `similarity_score`에 `0.0`을 넣으면 실패한다.
    """
    result = _post(client, _sample_bodies()[1])
    assert result["similarity_score"] is None
    assert result["reference_quality"] == "unavailable"
    assert result["reason_code"] == "reference_data_missing"
    assert result["expert_percentile"] is None
    assert result["corrections"] == []


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
    """게이트를 통과하는 최소 시퀀스. 리그 치수는 실측(어깨 사이 0.0567 m,
    위팔 0.1797 · 아래팔 0.2430 · 손 0.2430)에서 왔다.

    **몸 전체를 프레임마다 조금씩 옮긴다.** 한 관절만 움직이면 나머지가 얼어
    게이트의 R4가 전부 제외해 버린다 — 실제 분사 시퀀스는 그렇지 않다. 강체 평행이동이라
    뼈 길이는 그대로 유지된다(R1).
    """
    joints = {
        "pelvis": [0.0, head - 0.75, 0.0],
        "spine": [0.0, head - 0.55, 0.0],
        "chest": [0.0, head - 0.2126, 0.0],
        "head": [0.0, head, 0.0],
        "leftshoulder": [-0.02835, head, 0.0],
        "rightshoulder": [0.02835, head, 0.0],
        "leftupperarm": [-0.2080, head, 0.0],
        "rightupperarm": [0.2080, head, 0.0],
        "leftforearm": [-0.2080, head - 0.2430, 0.10],
        "rightforearm": [0.2080, head - 0.2430, 0.10],
        "lefthand": [-0.2080, head - 0.4860, 0.20],
        "righthand": [0.2080, head - 0.4860, 0.20],
        "lefttoe": [-0.1, head - 1.0, 0.1],
        "righttoe": [0.1, head - 1.0, 0.1],
    }
    frames = []
    for index in range(4):
        shift = (0.004 * index, 0.002 * index, 0.003 * index)
        moved = {
            name: [value[axis] + shift[axis] for axis in range(3)]
            for name, value in joints.items()
        }
        frames.append(
            {
                "index": index,
                "time": 1000.0 + 0.009 * index,
                "tool_pos_rot": {"pos": [0.25 + 0.05 * index, head - 0.86, 0.22], "rot": [0, 0, 0]},
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
    assert len(result["scored_joints"]) < arm_reference.MIN_SCORED_JOINTS
    assert result["reason_code"] in pose_estimation.SEQUENCE_REASON_CODES
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
    assert second["reason_code"] in pose_estimation.SEQUENCE_REASON_CODES
    assert second["similarity_score"] is None


# --- 레퍼런스가 있을 때 -------------------------------------------------------


def test_a_pose_on_the_expert_mean_scores_high(client, monkeypatch, tmp_path) -> None:
    """레퍼런스가 깔리면 **실제 점수와 교정이 나온다.** 여기가 계약의 나머지 절반이다."""
    _write_reference(tmp_path, monkeypatch)
    result = _post(client, _synthetic("W", "SEQ-0", 1.5, 4.5))
    assert result["analysis_status"] == "available"
    assert result["reason_code"] is None
    assert result["reference_quality"] == "multi_worker"
    assert result["scoring_set_id"] == arm_reference.SCORING_SET_ID
    assert len(result["scored_joints"]) == 6
    assert result["similarity_score"] == pytest.approx(1.0, abs=1e-3)
    assert result["corrections"] == [], "숙련자 평균과 같은 자세에 교정이 나왔다"
    assert result["support"] == 200


def test_a_departed_arm_lowers_the_score_and_names_that_joint(
    client, monkeypatch, tmp_path
) -> None:
    """오른손만 20 cm 옮기면 점수가 내려가고 **그 관절이 지목돼야** 한다."""
    _write_reference(tmp_path, monkeypatch)
    # 같은 자세의 점수와 견준다. 고정 임계와 견주면 배선이 바뀔 때마다 임계를 따라 옮기게 되고,
    # 그러면 검사가 "점수가 내려간다"가 아니라 "내가 적은 숫자와 같다"를 지키게 된다.
    baseline = _post(client, _synthetic("W", "SEQ-base", 1.5, 4.5))["similarity_score"]
    body = _synthetic("W", "SEQ-0", 1.5, 4.5)
    for frame in body["sequence_info"]:
        frame["skeleton_joints"]["righthand"][2] += 0.30
    result = _post(client, body)

    assert baseline is not None and result["similarity_score"] is not None
    assert result["similarity_score"] < baseline - 0.05, (
        f"오른손을 30 cm 옮겼는데 점수가 {baseline:.3f} → {result['similarity_score']:.3f}에 그쳤다"
    )
    assert len(result["corrections"]) == 1, "한 번에 한 관절만 짚는다"
    correction = result["corrections"][0]
    assert correction["joint"] == "RightHand"
    assert correction["label"] == "오른쪽 손목"
    assert correction["severity"] in ("medium", "high")
    assert correction["evidence"]["sigma"] > arm_reference.CORRECTION_SIGMA
    assert correction["evidence"]["distance_m"] == pytest.approx(0.30, abs=0.03)
    assert correction["evidence"]["expected_gain"] > 0


def test_the_correction_carries_no_angle_and_no_direction(
    client, monkeypatch, tmp_path
) -> None:
    """`correction_mode="joint_only"`다 — 관절만 짚고 각도도 방향도 싣지 않는다.

    회전순서가 미확정이고(실물 `rot`이 전 프레임 90° 배수) 방향 단서는 외적 초점을
    내적 초점으로 바꾼다.
    """
    _write_reference(tmp_path, monkeypatch)
    body = _synthetic("W", "SEQ-0", 1.5, 4.5)
    for frame in body["sequence_info"]:
        frame["skeleton_joints"]["righthand"][2] += 0.30
    result = _post(client, body)
    assert result["correction_mode"] == "joint_only"
    correction = result["corrections"][0]
    assert correction["angle_delta_deg"] is None
    assert correction["direction"] is None
    for banned in ("올리", "내리", "왼쪽으로", "오른쪽으로", "앞으로", "뒤로"):
        assert banned not in correction["correction_text"], correction["correction_text"]


def test_a_single_worker_reference_is_degraded_not_available(
    client, monkeypatch, tmp_path
) -> None:
    """1인 산포는 '숙련자 통상 범위'가 아니라 '그 사람의 습관'이다."""
    _write_reference(tmp_path, monkeypatch, workers=1)
    result = _post(client, _synthetic("W", "SEQ-0", 1.5, 4.5))
    assert result["reference_quality"] == "single_worker"
    assert result["analysis_status"] == "degraded"
    assert result["reason_code"] == "reference_single_worker"
    assert result["similarity_score"] is not None, "강등이지 미측정이 아니다"


def test_a_thin_bucket_withholds_the_score(client, monkeypatch, tmp_path) -> None:
    """표본이 얕은 버킷으로 낸 점수는 숙련자 범위가 아니라 우연이다."""
    _write_reference(tmp_path, monkeypatch, n=5)
    result = _post(client, _synthetic("W", "SEQ-0", 1.5, 4.5))
    assert result["reference_quality"] == "unavailable"
    assert result["similarity_score"] is None
    assert result["reason_code"] == "reference_data_missing"


def test_a_sequence_with_no_impact_cannot_pick_a_bucket(
    client, monkeypatch, tmp_path
) -> None:
    """버킷 조건이 reach(착탄 − 머리)다. 미명중이면 조건이 없으니 추정하지 않는다."""
    _write_reference(tmp_path, monkeypatch)
    body = _synthetic("W", "SEQ-0", 1.5, 4.5)
    for frame in body["sequence_info"]:
        frame["gun_metrics"]["raypoint"] = None
    result = _post(client, body)
    assert result["reach_m"] is None
    assert result["reference_band"] is None
    assert result["similarity_score"] is None
    assert result["reason_code"] == "reference_data_missing"



def test_the_composite_route_prefers_phase_scoring_when_it_can(
    client, monkeypatch, tmp_path
) -> None:
    """합본 경로는 **위상 채점을 먼저 시도한다** — 국소 오차를 접기보다 3.7배 잡는다.

    위상 레퍼런스가 있고 착탄이 움직이면 `phase`, 아니면 `fold`로 내려간다.

    깨뜨리는 법: 합본 경로가 늘 `fold`를 쓰면 실패한다.
    """
    import importlib

    stages_test = importlib.import_module("mcp_servers.test_pose_stages_route")
    stages_test._write_phase_reference(
        tmp_path, monkeypatch, stages_test._sweep(stages_test._sample_body("SEQ-002"))
    )
    # 두 사다리가 **갈리는 점수대**를 쓴다. 0.87은 접기 p10(0.90) 아래이고
    # 위상 p90(0.50) 위라 모드를 무시하면 90이 아니라 0이 나온다.
    body = stages_test._sweep(
        stages_test._sample_body("SEQ-002"), arm_shift=(0.0, 0.0, 0.5), first_half_only=False
    )

    result = _post(client, body)
    assert result["scoring_mode"] == "phase", result["reason_code"]
    assert result["similarity_score"] == pytest.approx(0.87, abs=0.05)
    assert result["expert_percentile"] == 90, (
        f"점수 {result['similarity_score']}가 접기 사다리에 대해 매겨졌다"
    )


def test_the_composite_route_falls_back_when_the_gun_is_fixed(
    client, monkeypatch, tmp_path
) -> None:
    """건을 고정하면 정렬할 모양이 없다(실물 SEQ-002). 그때도 **점수는 나와야 한다** —
    위상이 안 된다고 채점을 포기하면 실물 시퀀스 대부분이 미측정이 된다.

    깨뜨리는 법: 위상 실패 시 접기로 안 내려가면 실패한다.
    """
    import importlib

    stages_test = importlib.import_module("mcp_servers.test_pose_stages_route")
    swept = stages_test._sweep(stages_test._sample_body("SEQ-002"))
    stages_test._write_phase_reference(tmp_path, monkeypatch, swept)

    frozen = json.loads(json.dumps(swept))
    for frame in frozen["sequence_info"]:
        frame["gun_metrics"]["raypoint"] = list(
            swept["sequence_info"][0]["gun_metrics"]["raypoint"]
        )
        frame["tool_pos_rot"]["pos"] = list(swept["sequence_info"][0]["tool_pos_rot"]["pos"])
    result = _post(client, frozen)
    assert result["scoring_mode"] == "fold", "정렬이 불가능한데 위상으로 쟀다고 나왔다"
    assert result["similarity_score"] is not None, "접기로 내려가지 않았다"
