#!/usr/bin/env python3
"""관절 어휘 통일 검사.

실물 샘플에 대고 검사한다 — 어휘표를 손으로 적어 놓고 그 표에 대고 검사하면
표가 틀렸을 때 아무것도 실패하지 않는다.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPTS_DIR = Path(__file__).resolve().parents[1]
for path in (str(_REPO_ROOT), str(_SCRIPTS_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

joint_names = importlib.import_module("joint_names")

SAMPLE = _REPO_ROOT / "docs" / "SprayXR_CoPilot_샘플_2026-09-21.json"


def _sample_joint_names() -> set[str]:
    """실물 샘플이 실제로 보내는 관절 이름."""
    payload = json.loads(SAMPLE.read_text(encoding="utf-8"))
    found: set[str] = set()
    for request in payload["requests"]:
        body = request.get("body") or {}
        for frame in body.get("sequence_info") or []:
            found.update(frame.get("skeleton_joints") or {})
    return found


def test_every_real_sim_joint_maps_to_a_canonical_name() -> None:
    """실물 SIM이 보내는 이름이 하나도 빠짐없이 정본으로 옮겨져야 한다.

    하나라도 `None`이면 그 관절은 채점에서 조용히 사라진다 — 점수의 분모가 관절 수이므로
    사라진 것이 점수를 바꾸는데 아무 데도 안 적힌다.

    깨뜨리는 법: `CODE_JOINTS`에서 `RightToe`를 빼면 이 검사가 실패해야 한다.
    """
    names = _sample_joint_names()
    assert names, "샘플에서 관절 이름을 하나도 못 읽었다 — 검사가 아무것도 재지 않는다"
    unmapped = sorted(name for name in names if joint_names.to_code(name) is None)
    assert unmapped == [], f"정본으로 못 옮긴 SIM 이름: {unmapped}"


def test_the_sample_sends_exactly_the_canonical_set() -> None:
    """샘플의 관절 집합이 정본 20종과 정확히 같다 — 모자라지도 남지도 않는다."""
    mapped = {joint_names.to_code(name) for name in _sample_joint_names()}
    assert mapped == set(joint_names.CODE_JOINTS)


@pytest.mark.parametrize("code", joint_names.CODE_JOINTS)
def test_sim_and_csv_names_round_trip(code: str) -> None:
    """정본 → SIM/CSV → 정본이 원래 이름으로 돌아온다."""
    sim = joint_names.to_sim(code)
    csv = joint_names.to_csv(code)
    assert sim is not None and csv is not None
    assert joint_names.to_code(sim) == code
    assert joint_names.to_code(csv) == code


def test_csv_name_matches_the_capture_header_spelling() -> None:
    """CSV 이름은 대문자 앞에 공백을 넣은 형태다 — 첫 글자 앞에는 넣지 않는다.

    깨뜨리는 법: `to_csv`에서 첫 글자 예외를 없애면 `" Right Fore Arm"`이 되어 실패한다.
    """
    assert joint_names.to_csv("RightForeArm") == "Right Fore Arm"
    assert joint_names.to_csv("Pelvis") == "Pelvis"


def test_unknown_names_return_none_instead_of_raising() -> None:
    """모르는 이름에 예외를 던지면 프레임 하나가 시퀀스를 죽인다."""
    for name in ("", "elbow", "RightElbow", "머리", "neck", "Neck"):
        assert joint_names.to_code(name) is None, name


def test_neck_is_deliberately_absent() -> None:
    """`Neck`은 런타임·CSV 어디에도 없다. BVH의 `Neck`은 비율로 끼워 넣은 표시용 값이다.

    측정량이 아닌 값을 채점에 넣으면 σ의 출처가 데이터가 아니라 우리 상수가 된다.
    `pose_similarity.SCORING_JOINTS`가 선언 12종·실효 11종이던 괴리가 여기서 끝난다.
    """
    assert "Neck" not in joint_names.CODE_JOINTS
    assert joint_names.to_code("Neck") is None


def test_deprecated_mcp_aliases_are_absorbed() -> None:
    """`mcp_servers/pose_estimation.py:29`가 쓰던 이름도 받는다 — 흡수만 하고 새로 쓰지 않는다."""
    assert joint_names.to_code("wrist_r") == "RightHand"
    assert joint_names.to_code("elbow_l") == "LeftForeArm"
    assert joint_names.to_code("shoulder_r") == "RightShoulder"


def test_legs_are_canonical_but_not_scorable() -> None:
    """다리는 받아서 보관하되 채점·교정에서 뺀다 — 트래커가 없어 IK 산출물이다."""
    assert joint_names.to_code("leftthigh") == "LeftThigh"
    assert joint_names.is_scorable("LeftThigh") is False
    assert joint_names.is_scorable("RightHand") is True
    assert len(joint_names.LEG_JOINTS) == 8


def test_position_accepts_both_array_and_object_forms() -> None:
    """실물은 3원소 배열, 계약 문서는 {pos, rot} 객체다. 둘 중 하나를 422로 만들면 정상 데이터가 죽는다."""
    assert joint_names.parse_position([0.4207, 4.1166, 0.6457]) == (0.4207, 4.1166, 0.6457)
    assert joint_names.parse_position({"pos": [1.0, 2.0, 3.0], "rot": [270.0, 0.0, 0.0]}) == (1.0, 2.0, 3.0)


def test_position_rejects_malformed_values() -> None:
    """길이가 틀리거나 수가 아니면 `None`이다. `True`는 수가 아니다(파이썬에서 bool은 int의 하위형이다).

    깨뜨리는 법: `parse_position`의 bool 검사를 지우면 `[True, 1, 2]`가 `(1.0, 1.0, 2.0)`이 되어 실패한다.
    """
    for bad in (
        [1.0, 2.0],
        [1.0, 2.0, 3.0, 4.0],
        "abc",
        None,
        {"rot": [1, 2, 3]},
        [True, 1.0, 2.0],
        [None, 1.0, 2.0],
    ):
        assert joint_names.parse_position(bad) is None, bad


def test_rotation_is_only_available_on_the_object_form() -> None:
    """회전은 원문 보관용이다. 3원소 배열로 온 값에는 회전이 없다."""
    assert joint_names.parse_rotation({"pos": [0, 0, 0], "rot": [270.0, 0.0, 0.0]}) == (270.0, 0.0, 0.0)
    assert joint_names.parse_rotation([1.0, 2.0, 3.0]) is None


def test_positions_drops_unknown_quietly_and_reports_them() -> None:
    """모르는 이름은 조용히 빠지고, 빠진 것은 따로 보고된다 — 빠진 줄 모르는 것이 결함이다."""
    raw = {
        "chest": [0.1, 0.2, 0.3],
        "Right Fore Arm": [1.0, 1.1, 1.2],
        "neck": [9.0, 9.0, 9.0],
        "새관절": [0.0, 0.0, 0.0],
        "head": "형식오류",
    }
    mapped = joint_names.positions(raw)
    assert mapped == {"Chest": (0.1, 0.2, 0.3), "RightForeArm": (1.0, 1.1, 1.2)}
    assert joint_names.unknown_names(raw) == ["neck", "새관절"]


def test_real_sample_positions_parse_for_every_frame() -> None:
    """실물 전 프레임에서 20종이 전부 파싱된다 — 하나라도 빠지면 채점 분모가 프레임마다 달라진다.

    **20을 리터럴로 적는다.** `len(CODE_JOINTS)`와 비교하면 어휘가 줄 때 기대값도 같이 줄어
    어휘 축소를 볼 수 없다(자기참조). 실제로 `RightToe`를 빼는 돌연변이에서 이 검사만 통과했다.
    """
    expected = 20
    assert len(joint_names.CODE_JOINTS) == expected, "정본 어휘가 20종에서 바뀌었다 — 의도한 변경인지 확인하라"
    payload = json.loads(SAMPLE.read_text(encoding="utf-8"))
    frames = 0
    for request in payload["requests"]:
        for frame in (request.get("body") or {}).get("sequence_info") or []:
            joints = frame.get("skeleton_joints") or {}
            if not joints:
                continue
            frames += 1
            assert len(joint_names.positions(joints)) == expected
    assert frames >= 18, f"검사한 프레임이 {frames}개뿐이다 — 샘플을 못 읽고 있다"
