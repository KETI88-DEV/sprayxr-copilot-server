"""몸통 비례(U4)를 고정한다.

이 모듈이 지키는 성질은 둘이다.

1. **상수의 출처.** `body_proportions.BVH_*_M`은 전형기 BVH 79파일에서 잰 OFFSET 노름 중앙값이다.
   여기서 그 파일을 **다시 읽어 다시 재서** 같은 값인지 본다(출처와 다른 라벨을 막는다).
2. **앵커 오차 불변.** 골반→가슴만 다시 재면 세 앵커(머리·오른손·왼손)의 상대 위치가 그대로라
   배치 어긋남이 **정확히 그대로**다. 합성 프레임으로 그 분기를 직접 고정하고, 가슴→머리까지
   함께 늘리면 **바뀐다**는 대조도 같이 잰다 — 실패할 수 없는 단언을 막는다.
"""

from __future__ import annotations

import glob
import importlib
import math
import statistics
import sys
from pathlib import Path

import numpy as np
import pytest

_SCRIPTS = str(Path(__file__).resolve().parents[3] / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from trainer.pose_lab import body_proportions as bp
from trainer.pose_lab import skeleton

rotation_math = importlib.import_module("rotation_math")

BVH_GLOB = "/data/sources/professionals/bvh_전형기/*/*.bvh"
BVH_TOL_M = 1e-6


# --- 1. 상수의 출처 ---------------------------------------------------------------------


def _bvh_medians() -> dict[str, float]:
    bvh_skeleton = importlib.import_module("bvh_skeleton")
    wanted = set(bp.BVH_TORSO_M) | set(bp.BVH_ARM_M) | set(bp.BVH_LEG_M)
    seen: dict[str, list[float]] = {name: [] for name in wanted}
    for path in sorted(glob.glob(BVH_GLOB)):
        clip = bvh_skeleton.parse_bvh(path)
        for joint in clip.joints:
            if joint.name in seen:
                seen[joint.name].append(math.hypot(*joint.offset))
    return {name: statistics.median(values) for name, values in seen.items() if values}


def test_bvh_constants_are_remeasured_from_the_bvh_files() -> None:
    """상수 15개를 BVH 파일에서 다시 재 같은 값인지 본다. 파일이 없으면 이 검사만 건너뛴다."""
    if not glob.glob(BVH_GLOB):
        pytest.skip(f"BVH 파일이 없다({BVH_GLOB})")
    measured = _bvh_medians()
    quoted = {**bp.BVH_TORSO_M, **bp.BVH_ARM_M, **bp.BVH_LEG_M}
    assert set(quoted) <= set(measured), sorted(set(quoted) - set(measured))
    for name, said in quoted.items():
        assert abs(measured[name] - said) <= BVH_TOL_M, (name, said, measured[name])


def test_bvh_constant_check_can_fail_when_a_number_drifts() -> None:
    """상수 하나를 1mm 틀리게 적으면 위 검사가 실제로 잡는다."""
    if not glob.glob(BVH_GLOB):
        pytest.skip(f"BVH 파일이 없다({BVH_GLOB})")
    measured = _bvh_medians()
    drifted = bp.BVH_TORSO_M["Neck"] + 0.001
    assert abs(measured["Neck"] - drifted) > BVH_TOL_M


def test_target_is_the_geometric_mean_of_the_arm_and_leg_derivations() -> None:
    """목표 길이가 독스트링이 적은 대로 `sqrt(L_arm · L_leg)`인지 — 손으로 다시 계산해 본다."""
    offsets = _csv_like_offsets()
    arm = sum(math.hypot(*offsets[name]) for name in bp.ARM_CHAINS[0])
    leg = sum(math.hypot(*offsets[name]) for name in bp.LEG_CHAINS[0])
    expected = math.sqrt((arm / bp.BVH_ARM_RATIO) * (leg / bp.BVH_LEG_RATIO))
    assert bp.target_lower_torso_m(offsets) == pytest.approx(expected, rel=1e-12)


def test_rescaling_touches_only_the_two_lower_torso_bones() -> None:
    offsets = _csv_like_offsets()
    out = bp.rig_offsets(offsets)
    changed = {
        name
        for name in offsets
        if math.hypot(*out[name]) != pytest.approx(math.hypot(*offsets[name]), abs=1e-12)
    }
    assert changed == set(bp.RESCALED_BONES), changed
    # 방향은 그대로다 — 스칼라만 곱한다.
    for name in bp.RESCALED_BONES:
        a = np.asarray(offsets[name], dtype=float)
        b = np.asarray(out[name], dtype=float)
        cosine = float(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))
        assert cosine == pytest.approx(1.0, abs=1e-12), name


def test_lower_torso_reaches_the_bvh_derived_target() -> None:
    offsets = _csv_like_offsets()
    out = bp.rig_offsets(offsets)
    total = sum(math.hypot(*out[name]) for name in bp.RESCALED_BONES)
    assert total == pytest.approx(bp.target_lower_torso_m(offsets), rel=1e-12)
    # 마디별 배분도 BVH 몫 그대로다.
    share_total = sum(bp.LOWER_TORSO_SHARE_M[name] for name in bp.RESCALED_BONES)
    for name in bp.RESCALED_BONES:
        want = total * bp.LOWER_TORSO_SHARE_M[name] / share_total
        assert math.hypot(*out[name]) == pytest.approx(want, rel=1e-12), name


def test_inserted_joints_lie_on_the_segment_at_the_bvh_split() -> None:
    joints = {
        "Spine": (0.0, 1.0, 0.0),
        "Chest": (0.0, 1.3, 0.4),
        "Head": (0.2, 1.7, 0.4),
    }
    out = bp.inserted_joints(joints)
    assert set(out) == {"Spine1", "Neck"}
    for name, start, end, ratio in bp.INSERTED_JOINTS:
        a = np.asarray(joints[start], dtype=float)
        b = np.asarray(joints[end], dtype=float)
        assert np.allclose(np.asarray(out[name]), a + (b - a) * ratio, atol=1e-12), name
    # 비율은 BVH가 그 선분을 가르는 몫이다(지어낸 수가 아니다).
    assert bp.INSERTED_JOINTS[0][3] == pytest.approx(
        bp.BVH_TORSO_M["Spine1"] / (bp.BVH_TORSO_M["Spine1"] + bp.BVH_TORSO_M["Chest"])
    )
    assert bp.INSERTED_JOINTS[1][3] == pytest.approx(
        bp.BVH_TORSO_M["Neck"] / (bp.BVH_TORSO_M["Neck"] + bp.BVH_TORSO_M["Head"])
    )


def test_inserted_joints_are_skipped_when_an_end_is_missing() -> None:
    assert bp.inserted_joints({"Spine": (0.0, 0.0, 0.0)}) == {}


def test_display_bones_replaces_the_three_stem_bones_with_five() -> None:
    joints = {name: (0.0, 0.0, 0.0) for name in ("Pelvis", "Spine", "Spine1", "Chest", "Neck", "Head")}
    bones = [["Pelvis", "Spine"], ["Spine", "Chest"], ["Chest", "Head"], ["Chest", "LeftShoulder"]]
    out = bp.display_bones(bones, joints)
    assert out == [list(pair) for pair in bp.TORSO_STEM_BONES] + [["Chest", "LeftShoulder"]]
    # 그려 넣은 관절이 없으면 원래 줄기를 남긴다.
    assert bp.display_bones(bones, {"Pelvis": (0.0, 0.0, 0.0)}) == bones


# --- 3. 머리 구체 지름(F2) ---------------------------------------------------------------
#
# `skeleton_view.js`의 옛 `HEAD_SPHERE_RADIUS = 0.09`(지름 0.18m)는 그려지는 골반→머리 0.409m의
# 44%였다 — 사람 머리는 정수리~골반 길이의 17~22%다(BVH 실측: Head 0.097643m / 골반→머리 총합
# 0.561368m = 17.4%). 화면이 풍선처럼 보인 원인이 렌더 버그가 아니라 상수였다는 뜻이다.


def test_head_diameter_ratio_is_the_bvh_head_bones_share_of_the_full_chain() -> None:
    """`HEAD_DIAMETER_RATIO`가 BVH 다섯 마디 합에서 `Head`가 차지하는 몫과 같은지 다시 계산해 본다."""
    total = sum(bp.BVH_TORSO_M.values())
    assert total == pytest.approx(0.561368, abs=1e-6)
    expected = bp.BVH_TORSO_M["Head"] / total
    assert bp.HEAD_DIAMETER_RATIO == pytest.approx(expected, rel=1e-12)
    # 사람 범위(17~22%) 안에 있는가 — 옛 44%와는 확연히 다르다.
    assert 0.17 < bp.HEAD_DIAMETER_RATIO < 0.22, bp.HEAD_DIAMETER_RATIO


def test_head_diameter_ratio_check_can_fail_against_the_old_44_percent() -> None:
    """옛 상수(지름 0.18m, 골반→머리 0.409m)가 내는 44%는 이 비례 범위 **밖**이어야 한다."""
    old_ratio = 0.18 / 0.409
    assert not (0.17 < old_ratio < 0.22), old_ratio


def test_head_diameter_m_is_torso_length_times_the_bvh_ratio() -> None:
    """머리 지름이 **이 리그의** 골반→머리 길이에 실측 비례를 곱한 값인지 — 고정 상수가 아닌지 본다."""
    offsets = _csv_like_offsets()
    torso = bp.torso_length_m(offsets)
    assert torso == pytest.approx(0.3474, abs=1e-3)  # 독스트링의 옛 골반→머리(다시 재기 전)
    assert bp.head_diameter_m(offsets) == pytest.approx(torso * bp.HEAD_DIAMETER_RATIO, rel=1e-12)
    # 리그가 달라지면(예: `rig_offsets`로 다시 잰 골반→가슴) 지름도 같이 바뀐다 — 고정 상수라면
    # 안 바뀔 것이다.
    rescaled = bp.rig_offsets(offsets)
    assert bp.torso_length_m(rescaled) != pytest.approx(torso, rel=1e-9)
    assert bp.head_diameter_m(rescaled) != pytest.approx(bp.head_diameter_m(offsets), rel=1e-9)
    assert bp.head_diameter_m(rescaled) == pytest.approx(bp.torso_length_m(rescaled) * bp.HEAD_DIAMETER_RATIO, rel=1e-12)


def test_head_diameter_m_uses_torso_length_not_a_different_definition() -> None:
    """`torso_length_m`이 `TORSO_BONES`(Spine·Chest·Head)만 재는지 — 다리·팔이 섞이면 안 된다."""
    offsets = _csv_like_offsets()
    with_legs_and_arms = bp.torso_length_m(offsets)
    trimmed = {k: v for k, v in offsets.items() if k in bp.TORSO_BONES}
    assert bp.torso_length_m(trimmed) == pytest.approx(with_legs_and_arms, rel=1e-12)
    # 팔·다리를 지워도 값이 그대로여야 실제로 세 마디만 쓴다는 증거다.
    assert set(trimmed) < set(offsets)


# --- 2. 앵커 오차 불변 ------------------------------------------------------------------


def _csv_like_offsets() -> dict[str, bp.Vec3]:
    """CSV 리그와 같은 마디 길이를 가진 바인드 오프셋(합성). 실측 길이는 `body_proportions` 독스트링."""
    out: dict[str, bp.Vec3] = {
        "Spine": (0.0, 0.10530, 0.0),
        "Chest": (0.0, 0.10530, 0.0),
        "Head": (0.0, 0.13680, 0.0),
    }
    for side, sign in (("Left", 1.0), ("Right", -1.0)):
        out[f"{side}Shoulder"] = (sign * 0.0500, 0.0500, 0.0)
        out[f"{side}UpperArm"] = (sign * 0.1200, 0.0, 0.0)
        out[f"{side}ForeArm"] = (sign * 0.2600, 0.0, 0.0)
        out[f"{side}Hand"] = (sign * 0.2141, 0.0, 0.0)
        out[f"{side}Thigh"] = (sign * 0.0900, -0.0400, 0.0)
        out[f"{side}Calf"] = (0.0, -0.4400, 0.0)
        out[f"{side}Foot"] = (0.0, -0.3974, 0.0)
        out[f"{side}Toe"] = (0.0, 0.0, 0.1500)
    return out


_CHAIN_PARENT = {
    "Spine": "Pelvis",
    "Chest": "Spine",
    "Head": "Chest",
    "LeftShoulder": "Chest",
    "LeftUpperArm": "LeftShoulder",
    "LeftForeArm": "LeftUpperArm",
    "LeftHand": "LeftForeArm",
    "RightShoulder": "Chest",
    "RightUpperArm": "RightShoulder",
    "RightForeArm": "RightUpperArm",
    "RightHand": "RightForeArm",
}


def _fk(offsets: dict[str, bp.Vec3], angles: dict[str, float]) -> tuple[dict, dict]:
    """합성 FK: 관절마다 z축 회전 `angles[name]`(도)만 준다. (위치, 회전) 사전을 낸다.

    `service`가 쓰는 `pose_fk_csv`와 같은 사슬(`Pelvis → Spine → Chest → {Head, 어깨}`)이라
    여기서 재는 불변이 서버의 불변이다.
    """
    positions = {"Pelvis": np.zeros(3)}
    rotations = {"Pelvis": _rz(angles.get("Pelvis", 0.0))}
    for name, parent in _CHAIN_PARENT.items():
        rotations[name] = rotations[parent] @ _rz(angles.get(name, 0.0))
        positions[name] = positions[parent] + rotations[parent] @ np.asarray(offsets[name], dtype=float)
    return (
        {k: tuple(float(x) for x in v) for k, v in positions.items()},
        {k: tuple(tuple(float(x) for x in row) for row in v) for k, v in rotations.items()},
    )


def _rz(deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


_ANGLES = {"Pelvis": 7.0, "Spine": -4.0, "Chest": 11.0, "Head": -13.0,
           "LeftShoulder": 8.0, "LeftUpperArm": -21.0, "LeftForeArm": 17.0, "LeftHand": 5.0,
           "RightShoulder": -6.0, "RightUpperArm": 25.0, "RightForeArm": -19.0, "RightHand": -3.0}


def _measured_and_mounts(offsets: dict[str, bp.Vec3]) -> tuple[dict, skeleton.Mounts]:
    """합성 실측 강체와 장착 상수. 실측은 **원본 리그**로 만든 뒤 일부러 조금 어긋나게 둔다 —
    어긋남이 0이면 "그대로다"가 실패할 수 없는 단언이 된다."""
    local_pos, local_rot = _fk(offsets, _ANGLES)
    mounts_map: dict[str, skeleton.AnchorMount] = {}
    measured: dict[str, tuple] = {}
    wobble = {"head": (0.021, -0.014, 0.009), "right": (-0.033, 0.012, -0.022), "left": (0.017, 0.028, -0.011)}
    for point, joint in skeleton.ANCHOR_JOINTS.items():
        rot = np.asarray(local_rot[joint], dtype=float)
        mount_offset = (0.05, 0.03, -0.02) if point == "head" else (0.11, -0.04, 0.06)
        mounts_map[skeleton.mount_key(point, "한길숏건")] = skeleton.AnchorMount(
            rotation=tuple(tuple(float(x) for x in row) for row in np.eye(3)), offset=mount_offset
        )
        marker = np.asarray(local_pos[joint], dtype=float) + rot @ np.asarray(mount_offset, dtype=float)
        marker = marker + np.asarray(wobble[point], dtype=float)
        measured[point] = (tuple(float(x) for x in marker), tuple(tuple(float(x) for x in row) for row in rot))
    return measured, skeleton.mounts_for(mounts_map, "한길숏건")


def _anchor_misses(offsets: dict[str, bp.Vec3]) -> dict[str, float]:
    measured, mounts = _measured_and_mounts(_csv_like_offsets())
    local_pos, local_rot = _fk(offsets, _ANGLES)
    placed = skeleton.place_oriented(local_pos, local_rot, measured, mounts)
    assert placed is not None
    return {fit.point: fit.miss_m for fit in placed.anchors}


def test_rescaling_the_lower_torso_leaves_every_anchor_miss_exactly_alone() -> None:
    """골반→가슴만 다시 재면 세 앵커 어긋남이 그대로다 — 합성 프레임에서 분기를 직접 고정한다.

    `place_oriented`의 평행이동 `t`가 세 앵커의 **공통** 변위를 흡수한다. 머리·양손이 전부
    `Chest` 아래에 달려 있으므로 `Spine`·`Chest`를 늘리면 셋이 같은 벡터만큼 움직인다.
    """
    base = _csv_like_offsets()
    before = _anchor_misses(base)
    after = _anchor_misses(bp.rig_offsets(base))
    assert set(before) == {"head", "right", "left"}
    assert min(before.values()) > 1e-3, before  # 어긋남이 0이 아니어야 비교가 의미 있다
    for point in before:
        assert after[point] == pytest.approx(before[point], abs=1e-12), (point, before, after)


def test_stretching_chest_to_head_would_change_the_anchor_miss() -> None:
    """대조군: 가슴→머리까지 BVH 비례로 늘리면 앵커 어긋남이 **바뀐다**.

    위 검사가 무엇을 잡는지 이 대조가 정한다 — 늘려도 안 바뀌면 위 단언은 실패할 수 없다.
    가슴→머리를 늘리지 않는 이유가 여기 있다(`body_proportions` 독스트링의 실측).
    """
    base = _csv_like_offsets()
    before = _anchor_misses(base)
    stretched = dict(bp.rig_offsets(base))
    x, y, z = stretched["Head"]
    factor = (bp.BVH_TORSO_M["Neck"] + bp.BVH_TORSO_M["Head"]) / bp.BVH_TORSO_M["Head"]
    stretched["Head"] = (x * factor, y * factor, z * factor)
    after = _anchor_misses(stretched)
    assert after["head"] != pytest.approx(before["head"], abs=1e-6), (before, after)


def test_rescaling_the_lower_torso_leaves_the_fitted_mounts_alone() -> None:
    """장착 상수 적합도 그대로다 — 적합이 보는 것이 앵커 **사이의 차**(FK_a − FK_b)이기 때문이다.

    그래서 `skeleton_mounts.json`을 다시 맞추지 않아도 된다.
    """
    base = _csv_like_offsets()
    measured, _mounts = _measured_and_mounts(base)
    samples = []
    for offsets in (base, bp.rig_offsets(base)):
        local_pos, local_rot = _fk(offsets, _ANGLES)
        samples.append(
            [skeleton.FrameSample(local_pos=local_pos, local_rot=local_rot, measured=measured,
                                  group="한길숏건", index=0, time_sec=0.0)]
        )
    table_before = skeleton.fit_mounts([samples[0]])
    table_after = skeleton.fit_mounts([samples[1]])
    assert set(table_before) == set(table_after), (sorted(table_before), sorted(table_after))
    for key, mount in table_before.items():
        assert np.allclose(np.asarray(table_after[key].offset), np.asarray(mount.offset), atol=1e-12), key


# --- 살(반투명 부피) ----------------------------------------------------------------


def test_flesh_is_not_drawn_without_a_head_diameter() -> None:
    """리그 크기를 모르면 살을 **지어내지 않는다** — 화면은 막대 인형으로 남는다."""
    assert bp.flesh_block(None) is None
    assert bp.flesh_block(0.0) is None


def test_flesh_scales_with_the_same_denominator_as_the_head() -> None:
    """머리 표식과 **같은 분모**로 줄여야 머리와 몸이 따로 커지지 않는다.

    깨뜨리는 법: 배율을 고정 상수로 박으면 리그 크기가 바뀌어도 살이 안 따라온다.
    """
    adult = bp.flesh_block(bp.ADULT_HEAD_DIAMETER_M)
    half = bp.flesh_block(bp.ADULT_HEAD_DIAMETER_M / 2)
    assert adult is not None and half is not None
    assert adult["scale"] == pytest.approx(1.0)
    assert half["scale"] == pytest.approx(0.5)
    for name, (r0, r1) in bp.FLESH_RADIUS_ADULT_M.items():
        assert adult["segments"][name] == pytest.approx([r0, r1], abs=1e-4)
        assert half["segments"][name] == pytest.approx([r0 / 2, r1 / 2], abs=1e-4)


def _triples(value, path="flesh"):
    """숫자 3개짜리 배열의 위치. 화면의 좌표 경계가 이런 배열을 좌표로 보고 Z를 뒤집는다."""
    found = []
    if isinstance(value, list):
        if len(value) == 3 and all(isinstance(item, (int, float)) for item in value):
            found.append(path)
        for index, item in enumerate(value):
            found.extend(_triples(item, f"{path}[{index}]"))
    elif isinstance(value, dict):
        for key, item in value.items():
            found.extend(_triples(item, f"{path}.{key}"))
    return found


def test_flesh_never_carries_a_three_number_array() -> None:
    """3원소 숫자 배열을 실으면 화면 경계(`sceneViewFromUnity`)가 좌표로 알고 뒤집는다 —
    반지름 셋이 조용히 부호가 바뀐다.

    깨뜨리는 법: 가로 덩어리를 `[from, to, r]`이 아니라 `[r0, r1, r2]`로 실으면 실패한다.
    """
    block = bp.flesh_block(0.07)
    assert block is not None
    assert _triples(block) == []


def test_every_flesh_name_is_a_joint_the_screen_can_draw() -> None:
    """부위 이름은 **골격에 있는 관절**이어야 한다. 없는 이름을 주면 화면이 조용히 건너뛰어
    그 부위만 막대로 남는다.

    깨뜨리는 법: `Spine1`을 `Spine2`로 오타 내면 실패한다.
    """
    from trainer.scripts import skeleton_gate

    known = {name for pair in skeleton_gate.BONES for name in pair}
    known |= {joint for joint, *_ in bp.INSERTED_JOINTS}
    block = bp.flesh_block(0.07)
    assert block is not None
    names = set(block["segments"])
    names |= {item["from"] for item in block["bridges"]} | {item["to"] for item in block["bridges"]}
    names |= {item["from"] for item in block["palms"]} | {item["to"] for item in block["palms"]}
    head = block["head"]
    names |= {head["from"], head["fallback_from"], head["to"]}
    assert names - known == set(), f"골격에 없는 부위: {sorted(names - known)}"
