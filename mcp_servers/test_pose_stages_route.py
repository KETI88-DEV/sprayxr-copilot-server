#!/usr/bin/env python3
"""세 단계 경로 검사 — ① reference 찾기 · ② DTW 정렬 · ③ 유사도 측정.

**실물 샘플을 변환해서 쓴다.** 샘플의 SEQ-002는 건을 고정해 `raypoint`가 전 프레임
같으므로 정렬할 모양이 없다(그 자체가 `path_missing`으로 확인된다). 그래서 **벤더가 보낸
본문 그대로의 리그·관절 이름·프레임 간격을 유지한 채**, 몸과 착탄을 벽을 따라 쓸어 가는
강체 이동만 더해 움직이는 시퀀스를 만든다.

변환은 강체 평행이동뿐이라 뼈 길이가 보존되고(게이트 R1), 관절 이름·프레임 시각·
`gun_metrics` 모양이 전부 실물 그대로다. **지어낸 것은 '어디로 움직였나'뿐이다.**
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from mcp_servers import pose_estimation
from mcp_servers.main import app
from trainer.scripts import arm_reference, joint_names, posture_sequence, sequence_dtw

_SAMPLE = Path(__file__).resolve().parents[1] / "docs" / "SprayXR_CoPilot_샘플_2026-09-21.json"
pytestmark = pytest.mark.skipif(not _SAMPLE.is_file(), reason="실물 샘플이 이 작업 트리에만 있다")


def _sample_body(sequence_id: str) -> dict[str, Any]:
    doc = json.loads(_SAMPLE.read_text(encoding="utf-8"))
    for request in doc["requests"]:
        if request["endpoint"] == "POST /copilot/analyze-sequence":
            if request["body"]["sequence_id"] == sequence_id:
                return copy.deepcopy(request["body"])
    raise AssertionError(sequence_id)


def _sweep(
    body: dict[str, Any],
    *,
    span_m: float = 0.60,
    arm_shift: tuple[float, float, float] | None = None,
    first_half_only: bool = True,
) -> dict[str, Any]:
    """실물 본문을 **벽을 따라 쓸어 가는** 시퀀스로 바꾼다.

    몸·건·착탄을 같은 만큼 위로 옮긴다(강체). `arm_shift`를 주면 오른손만 더 옮겨
    자세 오차를 심는데, `first_half_only`면 **앞 절반 프레임에만** 심는다 —
    중앙값이 희석하는 바로 그 경우다.
    """
    out = copy.deepcopy(body)
    frames = out["sequence_info"]
    count = len(frames)
    for index, frame in enumerate(frames):
        ratio = index / (count - 1)
        lift = span_m * ratio
        for name, value in frame["skeleton_joints"].items():
            value[1] += lift
        frame["tool_pos_rot"]["pos"][1] += lift
        ray = frame["gun_metrics"].get("raypoint")
        if ray is not None:
            ray[1] += lift
        if arm_shift is not None and (not first_half_only or index < count // 2):
            hand = frame["skeleton_joints"]["righthand"]
            for axis in range(3):
                hand[axis] += arm_shift[axis]
    return out


@pytest.fixture
def client(monkeypatch, tmp_path: Path) -> TestClient:
    pose_estimation._SESSIONS = posture_sequence.SessionStore()
    monkeypatch.setenv(pose_estimation.ARM_REFERENCE_ENV, str(tmp_path / "none.json"))
    return TestClient(app)


def _band_of(body: dict[str, Any]) -> str:
    """이 본문이 떨어지는 밴드. 단계 사이로 넘기는 값과 같은 방식으로 낸다."""
    from trainer.scripts import pose_stages

    frames = [pose_estimation.SequenceFrame(**f) for f in body["sequence_info"]]
    gated, _ = pose_estimation._gate_frames(frames)
    reach = pose_stages.reach_of(
        pose_estimation._impact_points(frames),
        [f.tool_pos for f in gated if f.tool_pos is not None],
    )
    assert reach is not None
    return arm_reference.reach_band(reach)


def _post(client: TestClient, route: str, body: dict[str, Any]) -> dict[str, Any]:
    response = client.post(f"/pose_estimation/{route}", json=body)
    assert response.status_code == 200, response.text
    return response.json()["result"]


def _write_phase_reference(tmp_path: Path, monkeypatch, body: dict[str, Any]) -> str:
    """이 시퀀스 자신을 숙련자로 삼은 레퍼런스. 위상별 분포와 프로토타입을 싣는다."""
    frames, _ = pose_estimation._gate_frames(
        [pose_estimation.SequenceFrame(**frame) for frame in body["sequence_info"]]
    )
    impacts = pose_estimation._impact_points(
        [pose_estimation.SequenceFrame(**frame) for frame in body["sequence_info"]]
    )
    tools = [frame.tool_pos for frame in frames if frame.tool_pos is not None]
    from trainer.scripts import pose_stages

    reach = pose_stages.reach_of(impacts, tools)
    assert reach is not None
    shape = sequence_dtw.path_shape(impacts)
    assert shape is not None, "변환한 시퀀스에 착탄 경로가 없다"
    indices = sequence_dtw.decimate(shape)
    prototype = sequence_dtw.barycenter([[shape[i] for i in indices]], phases=arm_reference.PHASES)
    assert prototype is not None

    path, _ = sequence_dtw.align([shape[i] for i in indices], prototype)
    phases: list[dict[str, Any]] = [{"n": 0, "joints": {}} for _ in range(arm_reference.PHASES)]
    for query_index, phase in path:
        pose = arm_reference.normalize(frames[indices[query_index]].joints)
        if pose is None:
            continue
        entry = phases[phase]
        entry["n"] = 60
        for name, point in pose.items():
            entry["joints"][name] = {
                "mean": list(point),
                "sd": [0.05, 0.05, 0.05],
                "n": 60,
                "radius": {"mean": math.sqrt(sum(v * v for v in point)), "sd": 0.05},
                "heading": {
                    "mean": [v / max(math.sqrt(sum(c * c for c in point)), 1e-9) for v in point],
                    "angular_sd": 0.30,
                },
            }
    bucket = {
        "n": 300,
        "workers": 6,
        "joints": phases[0]["joints"],
        "prototype": [list(point) for point in prototype],
        "prototype_support": {"to_prototype": 0.05, "between": 0.40, "n": 300},
        "phases": phases,
    }
    path_out = tmp_path / "phase_reference.json"
    path_out.write_text(json.dumps({
        "version": 1,
        "scoring_set_id": arm_reference.SCORING_SET_ID,
        "buckets": {arm_reference.reach_band(reach): bucket},
    }), encoding="utf-8")
    monkeypatch.setenv(pose_estimation.ARM_REFERENCE_ENV, str(path_out))
    return arm_reference.reach_band(reach)


# --- 변환 자체가 실물을 보존하는가 -------------------------------------------


def test_the_sweep_keeps_the_vendor_rig_intact() -> None:
    """변환이 강체 평행이동뿐이어야 한다 — 뼈가 늘어나면 게이트가 거부하고,
    그때 우리가 보는 것은 실물이 아니라 우리가 만든 결함이다."""
    original = _sample_body("SEQ-002")
    swept = _sweep(original)
    first = swept["sequence_info"][0]["skeleton_joints"]
    last = swept["sequence_info"][-1]["skeleton_joints"]
    for a, b in (("leftshoulder", "rightshoulder"), ("rightupperarm", "rightforearm")):
        assert math.dist(first[a], first[b]) == pytest.approx(math.dist(last[a], last[b]), abs=1e-9)
    assert set(swept["sequence_info"][0].keys()) == set(original["sequence_info"][0].keys())
    assert swept["sequence_info"][0]["time"] == original["sequence_info"][0]["time"]


def test_the_real_sample_has_no_path_to_align() -> None:
    """건을 고정한 실물 시퀀스에는 정렬할 모양이 없다. 그것이 ②의 `path_missing`이다."""
    body = _sample_body("SEQ-002")
    impacts = pose_estimation._impact_points(
        [pose_estimation.SequenceFrame(**frame) for frame in body["sequence_info"]]
    )
    assert impacts, "착탄점은 있다"
    assert sequence_dtw.path_shape(impacts) is None, "고정된 착탄에 모양이 생겼다"


# --- 단계별 -------------------------------------------------------------------


def test_stage_one_resolves_a_bucket_on_its_own(client, monkeypatch, tmp_path) -> None:
    """①만 불러도 버킷 판정이 나온다 — 정렬·채점 없이."""
    body = _sweep(_sample_body("SEQ-002"))
    band = _write_phase_reference(tmp_path, monkeypatch, body)
    result = _post(client, "reference", body)
    assert result["stage"] == "resolve"
    assert result["ok"] is True
    assert result["band"] == band
    assert result["has_phase_reference"] is True
    assert result["prototype_points"] == arm_reference.PHASES
    assert "similarity_score" not in result, "①이 점수를 냈다 — 단계가 섞였다"


def test_stage_one_refuses_when_there_is_no_reference(client) -> None:
    """레퍼런스가 없으면 ①에서 멈춘다. 뒤 단계가 없는 값을 지어내지 않는다."""
    result = _post(client, "reference", _sweep(_sample_body("SEQ-002")))
    assert result["ok"] is False
    assert result["reason"] == "reference_data_missing"
    assert result["prototype_points"] == 0


def test_stage_two_aligns_and_reports_the_warp(client, monkeypatch, tmp_path) -> None:
    """②는 정렬 결과와 **휜 양**을 낸다. 점수는 내지 않는다."""
    body = _sweep(_sample_body("SEQ-002"))
    _write_phase_reference(tmp_path, monkeypatch, body)
    result = _post(client, "align", body)
    assert result["stage"] == "align"
    assert result["ok"] is True
    assert result["phases"] == arm_reference.PHASES
    assert result["covered_phases"] >= 1
    assert result["warp"] >= 0.0
    assert result["path_span_m"] == pytest.approx(0.60, abs=0.05)
    assert result["frame_to_phase"], "위상 대응이 비어 있다"
    assert "similarity_score" not in result
    # ①의 출력도 같이 실려야 단계를 이어 붙일 수 있다.
    assert result["resolve"]["ok"] is True


def test_stage_two_stops_on_a_fixed_gun(client, monkeypatch, tmp_path) -> None:
    """실물 SEQ-002처럼 착탄이 안 움직이면 ②에서 멈춘다 — 억지로 맞추면 잡음에 맞춘다."""
    swept = _sweep(_sample_body("SEQ-002"))
    _write_phase_reference(tmp_path, monkeypatch, swept)
    frozen = copy.deepcopy(swept)
    for frame in frozen["sequence_info"]:
        frame["gun_metrics"]["raypoint"] = list(
            swept["sequence_info"][0]["gun_metrics"]["raypoint"]
        )
    # 착탄을 얼리면 reach가 달라져 ①에서 먼저 막힌다. ②를 보려면 밴드를 넘겨야 한다 —
    # 단계 출력을 넘기는 손잡이가 바로 이런 때 쓰인다.
    result = _post(client, "align", {**frozen, "band": _band_of(swept)})
    assert result["ok"] is False
    assert result["reason"] == "path_missing", result


def test_stage_three_scores_against_the_phase_reference(client, monkeypatch, tmp_path) -> None:
    """③이 점수와 교정을 낸다. 앞 두 단계 출력도 같이 실린다 — 공유 가능한 형태다."""
    body = _sweep(_sample_body("SEQ-002"))
    _write_phase_reference(tmp_path, monkeypatch, body)
    result = _post(client, "score", body)
    assert result["stage"] == "score"
    assert result["ok"] is True
    assert result["mode"] == "phase"
    assert result["similarity_score"] is not None
    assert result["scored_phases"] >= 1
    assert result["resolve"]["ok"] is True and result["align"]["ok"] is True
    # 자기 자신을 기준으로 쟀으니 거의 1이어야 한다.
    assert result["similarity_score"] > 0.9, result["similarity_score"]
    assert result["corrections"] == []


def test_a_band_from_stage_one_can_be_handed_to_the_later_stages(
    client, monkeypatch, tmp_path
) -> None:
    """①의 출력(`band`)을 넘기면 ②·③이 그것을 쓴다 — 단계 사이 출력 공유의 요점이다."""
    body = _sweep(_sample_body("SEQ-002"))
    band = _write_phase_reference(tmp_path, monkeypatch, body)
    handed = {**body, "band": band}
    assert _post(client, "align", handed)["ok"] is True
    assert _post(client, "score", handed)["ok"] is True

    missing = {**body, "band": "+9.90"}
    result = _post(client, "align", missing)
    assert result["ok"] is False
    assert result["reason"] == "reference_data_missing"


def test_every_stage_reason_is_in_the_closed_set(client, monkeypatch, tmp_path) -> None:
    from trainer.scripts import pose_stages

    body = _sweep(_sample_body("SEQ-002"))
    for route in ("reference", "align", "score"):
        result = _post(client, route, body)          # 레퍼런스 없음
        assert result["reason"] in pose_stages.STAGE_REASONS or result["reason"] is None
    _write_phase_reference(tmp_path, monkeypatch, body)
    for route in ("reference", "align", "score"):
        result = _post(client, route, body)
        assert result["reason"] in pose_stages.STAGE_REASONS or result["reason"] is None


def test_calling_a_stage_does_not_grow_the_session(client, monkeypatch, tmp_path) -> None:
    """단계 경로를 여러 번 불러도 자세 시퀀스가 부풀면 안 된다 — 기울기가 오염된다."""
    body = _sweep(_sample_body("SEQ-002"))
    _write_phase_reference(tmp_path, monkeypatch, body)
    for _ in range(4):
        _post(client, "score", body)
    session = pose_estimation._SESSIONS.get(body["session_id"])
    assert session is None or len(session.samples) == 0


def test_a_first_half_error_is_caught_better_than_folding_the_sequence(
    client, monkeypatch, tmp_path
) -> None:
    """**이것이 DTW를 붙인 이유다.** 앞 절반에만 심은 오차를 시퀀스 접기는 희석한다.

    절대 임계로 재지 않는다 — 임계는 배선이 바뀔 때마다 따라 옮기게 된다. **같은 오차를
    전 구간에 심었을 때와 앞 절반에만 심었을 때의 점수 하락을 견준다.** 접어서 재면
    앞 절반 오차의 하락이 전 구간의 절반 이하로 줄지만, 위상으로 재면 그렇지 않아야 한다.
    """
    shift = (0.0, 0.0, 0.35)
    clean = _sweep(_sample_body("SEQ-002"))
    _write_phase_reference(tmp_path, monkeypatch, clean)
    base = _post(client, "score", clean)["similarity_score"]

    whole = _post(
        client, "score",
        _sweep(_sample_body("SEQ-002"), arm_shift=shift, first_half_only=False),
    )["similarity_score"]
    half = _post(
        client, "score",
        _sweep(_sample_body("SEQ-002"), arm_shift=shift, first_half_only=True),
    )["similarity_score"]

    assert base is not None and whole is not None and half is not None
    drop_whole = base - whole
    drop_half = base - half
    assert drop_whole > 0.02, f"전 구간 오차조차 안 잡힌다({drop_whole:.4f})"
    # 앞 절반은 프레임의 45%에만 심었다. 접어서 재면 그보다 훨씬 덜 잡힌다.
    assert drop_half > drop_whole * 0.35, (
        f"앞 절반 오차가 희석됐다: 전 구간 {drop_whole:.4f} vs 앞 절반 {drop_half:.4f}"
    )


def test_stage_three_refuses_a_bucket_without_a_phase_reference(
    client, monkeypatch, tmp_path
) -> None:
    """위상 레퍼런스가 없는 버킷에서는 ③이 멈춘다 — 점 레퍼런스로 몰래 내려가지 않는다.

    깨뜨리는 법: `pose_stages.score`의 `phase_reference_missing` 검사를 빼면 실패한다.
    """
    body = _sweep(_sample_body("SEQ-002"))
    band = _write_phase_reference(tmp_path, monkeypatch, body)
    stripped = json.loads((tmp_path / "phase_reference.json").read_text(encoding="utf-8"))
    del stripped["buckets"][band]["phases"]
    (tmp_path / "phase_reference.json").write_text(
        json.dumps(stripped), encoding="utf-8"
    )
    result = _post(client, "score", body)
    assert result["ok"] is False
    assert result["reason"] == "phase_reference_missing"
    assert result["similarity_score"] is None


def test_stage_one_refuses_a_bucket_that_is_too_thin(client, monkeypatch, tmp_path) -> None:
    """표본이 얕은 버킷은 ①에서 막는다. 뒤 단계가 그 위에 점수를 쌓으면 안 된다.

    깨뜨리는 법: `resolve`의 `quality == "unavailable"` 검사를 빼면 실패한다.
    """
    body = _sweep(_sample_body("SEQ-002"))
    band = _write_phase_reference(tmp_path, monkeypatch, body)
    thin = json.loads((tmp_path / "phase_reference.json").read_text(encoding="utf-8"))
    thin["buckets"][band]["n"] = arm_reference.MIN_BUCKET_N - 1
    (tmp_path / "phase_reference.json").write_text(json.dumps(thin), encoding="utf-8")

    resolved = _post(client, "reference", body)
    assert resolved["ok"] is False
    assert resolved["reason"] == "reference_data_missing"
    assert _post(client, "score", body)["ok"] is False


def test_scored_phases_counts_only_phases_that_had_frames(
    client, monkeypatch, tmp_path
) -> None:
    """대응이 없는 위상을 빈 자세로 채우면 `scored_phases`가 실제보다 커진다 —
    '몇 구간을 봤나'가 부풀어 소비자가 근거를 과신한다.

    DTW 경로는 프로토타입의 **모든** 위상을 지나므로 경로만으로는 이 경우가 안 생긴다.
    그래서 `pose_stages.score`를 직접 불러 일부 위상에 대응이 없는 정렬을 넣는다.

    깨뜨리는 법: `score`에서 빈 위상을 `{}`로 채우면 12가 나와 실패한다.
    """
    from trainer.scripts import pose_stages

    body = _sweep(_sample_body("SEQ-002"))
    band = _write_phase_reference(tmp_path, monkeypatch, body)
    bucket = json.loads((tmp_path / "phase_reference.json").read_text(encoding="utf-8"))
    bucket = bucket["buckets"][band]

    frames, _ = pose_estimation._gate_frames(
        [pose_estimation.SequenceFrame(**frame) for frame in body["sequence_info"]]
    )
    # `score`는 **월드** 관절을 받는다 — 교정 계획을 월드로 되돌리려면 몸 프레임이 필요하다.
    poses = [dict(frame.joints) for frame in frames]

    covered = 5
    partial = pose_stages.AlignResult(
        ok=True,
        reason=None,
        phase_of=tuple((index, index) for index in range(covered)),
        phases=arm_reference.PHASES,
        covered_phases=covered,
    )
    result = pose_stages.score(poses, partial, bucket, labels={})
    assert result.ok is True
    assert result.scored_phases == covered, (
        f"대응이 {covered}위상뿐인데 {result.scored_phases}위상을 쟀다고 나왔다"
    )

    whole = pose_stages.AlignResult(
        ok=True, reason=None,
        phase_of=tuple((min(i, len(poses) - 1), i) for i in range(arm_reference.PHASES)),
        phases=arm_reference.PHASES, covered_phases=arm_reference.PHASES,
    )
    assert pose_stages.score(poses, whole, bucket, labels={}).scored_phases == arm_reference.PHASES


def test_a_handed_band_does_not_overwrite_the_measured_reach(
    client, monkeypatch, tmp_path
) -> None:
    """`band`를 넘겨도 응답의 `reach_m`은 **이 시퀀스에서 잰 값**이어야 한다.

    밴드의 공칭값으로 덮어쓰면 `reach_m`이 "잰 값"이 아니라 "고른 칸의 이름"이 되고,
    밴드를 잘못 넘긴 것을 호출자가 못 본다.

    깨뜨리는 법: `_resolved_of`에서 `replace(..., reach_m=prepared.reach)`를 빼면
    밴드 공칭값이 나와 실패한다.
    """
    body = _sweep(_sample_body("SEQ-002"))
    band = _write_phase_reference(tmp_path, monkeypatch, body)
    measured = _post(client, "reference", body)["reach_m"]
    assert measured is not None

    handed = _post(client, "align", {**body, "band": band})["resolve"]
    assert handed["band"] == band
    assert handed["reach_m"] == pytest.approx(measured, abs=1e-9), (
        "밴드 공칭값이 실측 reach를 덮었다"
    )
    assert handed["reach_m"] != pytest.approx(float(band), abs=1e-6), (
        "이 검사가 공허하다 — 실측이 마침 밴드 경계값이다"
    )


def test_the_target_is_world_coordinates_the_simulator_can_apply(
    client, monkeypatch, tmp_path
) -> None:
    """**Co-Pilot이 시뮬레이터에 넘기는 값이다.** 우리 몸 프레임이 아니라 월드여야 한다.

    몸 프레임(어깨 중점 원점, 어깨축·몸통축)은 이 모듈 안의 약속이다. 그 좌표로 내보내면
    받는 쪽이 우리 프레임을 다시 구현해야 하고, 그 순간 두 구현이 갈린다.

    깨뜨리는 법: `correction_plan`이 `to_world`를 안 거치면 좌표가 원점 근처로 나와 실패한다.
    """
    # 0.35 m면 σ 1.33으로 문턱(1.5) 아래다 — 교정이 안 나간다. 0.70 m면 σ 2.90이다.
    body = _sweep(_sample_body("SEQ-002"), arm_shift=(0.0, 0.0, 0.70), first_half_only=False)
    _write_phase_reference(tmp_path, monkeypatch, _sweep(_sample_body("SEQ-002")))
    result = _post(client, "score", body)
    assert result["corrections"], "교란을 넣었는데 교정이 없다"
    target = result["corrections"][0]["target"]
    assert target is not None
    assert target["frame"] == "world"
    assert target["mode"] in ("radius", "heading")
    assert target["per_frame"], "프레임별 값이 비었다 — 시뮬레이터가 쓸 것이 없다"

    joint = result["corrections"][0]["joint"]
    step = target["per_frame"][0]
    frames = [pose_estimation.SequenceFrame(**f) for f in body["sequence_info"]]
    gated, _ = pose_estimation._gate_frames(frames)
    actual = gated[step["index"]].joints[joint]
    assert step["current"] == pytest.approx(actual, abs=1e-3), (
        "`current`가 그 프레임의 실제 월드 좌표가 아니다"
    )
    # 월드 좌표는 원점 근처가 아니다 — 이 아바타는 4 m 높이에 서 있다.
    assert abs(step["target"][1]) > 1.0


def test_a_target_that_breaks_a_bone_says_it_needs_ik(client, monkeypatch, tmp_path) -> None:
    """관절 하나만 목표로 옮기면 부모와의 뼈 길이가 달라진다 — 우리는 사슬을 모델링하지 않는다.

    좌표만 주고 끝내면 받는 쪽이 **불가능한 골격**을 만든다. 깨지는 양을 같이 실어
    IK가 필요하다는 것을 알려야 한다.

    깨뜨리는 법: `_feasibility`가 늘 `requires_ik: False`를 내면 실패한다.
    """
    # 0.35 m면 σ 1.33으로 문턱(1.5) 아래다 — 교정이 안 나간다. 0.70 m면 σ 2.90이다.
    body = _sweep(_sample_body("SEQ-002"), arm_shift=(0.0, 0.0, 0.70), first_half_only=False)
    _write_phase_reference(tmp_path, monkeypatch, _sweep(_sample_body("SEQ-002")))
    target = _post(client, "score", body)["corrections"][0]["target"]
    assert target["parent_joint"] is not None
    assert target["bone_break_m"] is not None
    assert target["requires_ik"] == (
        target["bone_break_m"] > pose_estimation.pose_stages.BONE_BREAK_TOLERANCE_M
    )
    step = target["per_frame"][0]
    assert "parent" in step, "IK를 풀 부모 좌표가 없다"


def test_the_promised_gain_matches_what_the_target_delivers(
    client, monkeypatch, tmp_path
) -> None:
    """**교정 정확도의 내적 일관성.** 예고한 상승분이 지시를 적용했을 때와 같아야 한다.

    예전에는 `expected_gain`이 "σ를 0으로 만들면"이라는 상한이었는데 `target`은 주범
    성분 하나만 고쳤다 — 실측에서 예고 0.0579에 실현 0.0284였다. 예고가 지시를 따라가야
    검증이 뜻을 갖는다.

    이것은 **효능이 아니다.** 실제 작업자가 이 교정을 받고 나아지는지는 런타임 데이터가
    있어야 한다.

    깨뜨리는 법: `score`에서 `expected_gain`을 `target` 적용이 아닌 σ→0으로 되돌리면 실패한다.
    """
    # 0.35 m면 σ 1.33으로 문턱(1.5) 아래다 — 교정이 안 나간다. 0.70 m면 σ 2.90이다.
    body = _sweep(_sample_body("SEQ-002"), arm_shift=(0.0, 0.0, 0.70), first_half_only=False)
    _write_phase_reference(tmp_path, monkeypatch, _sweep(_sample_body("SEQ-002")))
    result = _post(client, "verify", body)
    assert result["ok"] is True, result
    assert result["expected_gain"] is not None
    assert result["realized_gain"] == pytest.approx(result["expected_gain"], abs=1e-6), (
        f"예고 {result['expected_gain']} 와 실현 {result['realized_gain']} 이 다르다"
    )
    assert result["gain_error"] == pytest.approx(0.0, abs=1e-6)
    assert result["after"] > result["before"], "교정을 적용했는데 점수가 안 올랐다"


def test_verify_says_so_when_there_is_nothing_to_correct(client, monkeypatch, tmp_path) -> None:
    """고칠 곳이 없으면 잴 것도 없다. 그것도 정직한 결과다."""
    clean = _sweep(_sample_body("SEQ-002"))
    _write_phase_reference(tmp_path, monkeypatch, clean)
    result = _post(client, "verify", clean)
    assert result["ok"] is False
    assert result["reason"] == "no_correction"
    assert result["before"] is not None
