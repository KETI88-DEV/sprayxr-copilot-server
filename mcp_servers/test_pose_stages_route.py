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
    poses = [arm_reference.normalize(frame.joints) or {} for frame in frames]

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
