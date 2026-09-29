#!/usr/bin/env python3
"""팔 레퍼런스 검사.

이 파일이 지키는 것은 하나로 요약된다 — **두 리그의 차이가 점수에 들어오면 안 된다.**
CSV-FK와 SIM은 `Chest–Shoulder` 길이가 7.50배 어긋난다. 그 차이가 σ에 새면 우리는
자세 오차가 아니라 리그를 재게 된다.
"""

from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path

import pytest
from typing import Final

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPTS_DIR = Path(__file__).resolve().parents[1]
for path in (str(_REPO_ROOT), str(_SCRIPTS_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

arm_reference = importlib.import_module("arm_reference")
measured_score = importlib.import_module("measured_score")


def _pose(chest_scale: float = 1.0) -> dict[str, tuple[float, float, float]]:
    """어깨 0.0567 m·위팔 0.1797·아래팔 0.2430·손 0.2430 — 두 리그가 공유하는 실측 치수."""
    return {
        "LeftShoulder": (-0.02835, 1.40, 0.0),
        "RightShoulder": (0.02835, 1.40, 0.0),
        "Chest": (0.0, 1.40 - 0.2126 * chest_scale, 0.0),
        "LeftUpperArm": (-0.2080, 1.40, 0.0),
        "RightUpperArm": (0.2080, 1.40, 0.0),
        "LeftForeArm": (-0.2080, 1.40 - 0.2430, 0.10),
        "RightForeArm": (0.2080, 1.40 - 0.2430, 0.10),
        "LeftHand": (-0.2080, 1.40 - 0.4860, 0.20),
        "RightHand": (0.2080, 1.40 - 0.4860, 0.20),
    }


def _rotate_y(point, degrees: float):
    angle = math.radians(degrees)
    cos, sin = math.cos(angle), math.sin(angle)
    return (point[0] * cos + point[2] * sin, point[1], -point[0] * sin + point[2] * cos)


def test_the_chest_bone_length_cannot_move_the_result() -> None:
    """이 파일의 존재 이유다.

    두 리그의 `Chest–Shoulder`가 7.50배 어긋난다. `Chest`를 **방향으로만** 쓰므로
    길이를 7.5배로 늘려도 정규화 좌표가 바뀌지 않아야 한다.

    깨뜨리는 법: `body_frame`에서 `torso`를 단위벡터로 만들지 않고 날것 길이로 쓰면 실패한다.
    """
    base = arm_reference.normalize(_pose(chest_scale=1.0))
    stretched = arm_reference.normalize(_pose(chest_scale=7.5))
    assert base is not None and stretched is not None
    for joint, point in base.items():
        assert stretched[joint] == pytest.approx(point, abs=1e-9), joint


def test_normalization_is_invariant_to_where_the_body_stands_and_faces() -> None:
    """CSV-FK 공간은 월드 정렬이 아니다(어깨축의 월드 수직 성분 중앙 0.77).

    같은 자세를 옮기고 돌려도 같은 값이 나와야 두 좌표계를 한 자로 잴 수 있다.
    """
    base = arm_reference.normalize(_pose())
    moved = arm_reference.normalize(
        {
            name: tuple(c + o for c, o in zip(_rotate_y(point, 37.0), (3.2, -1.1, 0.7)))
            for name, point in _pose().items()
        }
    )
    assert base is not None and moved is not None
    for joint, point in base.items():
        assert moved[joint] == pytest.approx(point, abs=1e-9), joint


def test_size_is_deliberately_not_normalized() -> None:
    """팔 뼈 길이가 두 리그에서 같으므로 크기를 지우지 않는다 — 값이 그대로 미터다.

    크기까지 정규화하면 '팔을 덜 뻗었다'가 지워진다. 그건 재려는 대상이다.
    """
    base = arm_reference.normalize(_pose())
    assert base is not None
    # 정규화 좌표의 크기가 곧 어깨 중점까지의 **실제 미터**여야 한다.
    origin = (0.0, 1.40, 0.0)
    expected = math.dist(_pose()["RightHand"], origin)
    assert math.sqrt(sum(c * c for c in base["RightHand"])) == pytest.approx(expected, abs=1e-9)
    assert expected > 0.5, "이 자세의 손은 어깨에서 0.5 m 넘게 떨어져 있다"

    pose = _pose()
    pose["RightHand"] = (0.2080, 1.40 - 0.30, 0.20)  # 팔을 덜 뻗었다
    shortened = arm_reference.normalize(pose)
    assert shortened is not None
    shorter = math.sqrt(sum(c * c for c in shortened["RightHand"]))
    # 줄어든 자세에서도 크기가 실제 미터와 같아야 한다 — 단위벡터로 지우면 둘 다 1.0이 된다.
    assert shorter == pytest.approx(math.dist(pose["RightHand"], origin), abs=1e-9)
    assert shorter < expected, "팔을 덜 뻗었는데 크기가 지워졌다"


def test_the_frame_axes_point_where_the_names_say() -> None:
    """`+lateral`은 오른쪽, `+torso`는 머리 쪽. 실물 SIM 샘플로 확인한 방향이다."""
    normalized = arm_reference.normalize(_pose())
    assert normalized is not None
    assert normalized["RightHand"][0] > 0 > normalized["LeftHand"][0]
    frame = arm_reference.body_frame(_pose())
    assert frame is not None
    chest_torso = sum(
        (_pose()["Chest"][i] - frame.origin[i]) * frame.axes[1][i] for i in range(3)
    )
    assert chest_torso < 0, "Chest가 어깨보다 위에 있다고 나왔다"


def test_a_frame_cannot_be_built_without_its_joints() -> None:
    """프레임을 못 세우면 `None`이다. 추정한 축으로 재면 σ의 출처가 데이터가 아니다."""
    pose = _pose()
    del pose["Chest"]
    assert arm_reference.body_frame(pose) is None
    assert arm_reference.normalize(pose) is None


def test_shoulders_are_not_scored() -> None:
    """어깨는 원점과 x축을 정의한다 — 정규화 후 정보가 없다. 넣으면 분모만 커진다."""
    assert "LeftShoulder" not in arm_reference.SCORING_JOINTS_ARM
    assert "RightShoulder" not in arm_reference.SCORING_JOINTS_ARM
    assert len(arm_reference.SCORING_JOINTS_ARM) == 6


def test_the_torso_and_head_are_not_scored() -> None:
    """리그가 어긋나는 뼈들이다. 넣으면 σ가 자세가 아니라 리그 차이를 잰다."""
    for joint in ("Spine", "Chest", "Head", "Pelvis"):
        assert joint not in arm_reference.SCORING_JOINTS_ARM


def _bucket(sd: float = 0.05, n: int = 200, workers: int = 6) -> dict:
    return {
        "n": n,
        "workers": workers,
        "joints": {
            name: {"mean": [0.0, 0.0, 0.0], "sd": [sd, sd, sd], "n": n}
            for name in arm_reference.SCORING_JOINTS_ARM
        },
    }


def test_a_pose_on_the_expert_mean_scores_one() -> None:
    normalized = {name: (0.0, 0.0, 0.0) for name in arm_reference.SCORING_JOINTS_ARM}
    sigmas = arm_reference.joint_sigmas(normalized, _bucket())
    assert len(sigmas) == 6
    assert arm_reference.similarity_score(sigmas) == pytest.approx(1.0)


def test_the_score_falls_as_the_pose_departs() -> None:
    scores = []
    for offset in (0.0, 0.05, 0.10, 0.20):
        normalized = {name: (offset, 0.0, 0.0) for name in arm_reference.SCORING_JOINTS_ARM}
        value = arm_reference.similarity_score(arm_reference.joint_sigmas(normalized, _bucket()))
        assert value is not None
        scores.append(value)
    assert scores == sorted(scores, reverse=True)
    assert scores[0] > scores[-1]


def test_no_measurable_joint_gives_null_not_zero() -> None:
    """`0.0`은 '많이 다르다'로 읽힌다. 우리는 재지 않았다.

    깨뜨리는 법: `similarity_score`가 빈 목록에 `0.0`을 내면 실패한다.
    """
    assert arm_reference.similarity_score([]) is None
    assert arm_reference.joint_sigmas({}, _bucket()) == []


def test_the_gun_hand_weighs_more_than_the_other() -> None:
    """건을 쥔 쪽이 분사를 정한다. 가중치는 `measured_score`에서 읽어 온다 — 여기서 다시 적지 않는다."""
    right = arm_reference._weight_of("RightHand")
    left = arm_reference._weight_of("LeftHand")
    assert right == measured_score.DEFAULT_WEIGHTS["right"]
    assert left == measured_score.DEFAULT_WEIGHTS["left"]
    assert right > left

    off_right = {name: (0.2 if name.startswith("Right") else 0.0, 0.0, 0.0)
                 for name in arm_reference.SCORING_JOINTS_ARM}
    off_left = {name: (0.2 if name.startswith("Left") else 0.0, 0.0, 0.0)
                for name in arm_reference.SCORING_JOINTS_ARM}
    worse = arm_reference.similarity_score(arm_reference.joint_sigmas(off_right, _bucket()))
    milder = arm_reference.similarity_score(arm_reference.joint_sigmas(off_left, _bucket()))
    assert worse is not None and milder is not None
    assert worse < milder, "오른팔이 틀어졌는데 왼팔과 같은 벌점을 받았다"


def test_sigma_has_a_floor_so_a_tight_bucket_cannot_explode_it() -> None:
    """산포가 0에 가까운 버킷에서 σ가 무한히 부풀면 전원이 교정 대상이 된다."""
    normalized = {name: (0.01, 0.0, 0.0) for name in arm_reference.SCORING_JOINTS_ARM}
    sigmas = arm_reference.joint_sigmas(normalized, _bucket(sd=1e-9))
    assert all(item.sigma <= arm_reference.MAX_SIGMA for item in sigmas)
    assert sigmas[0].axis_sigmas[0] == pytest.approx(0.01 / arm_reference.SIGMA_FLOOR_M)


def test_reach_bands_are_a_grid_with_stable_edges() -> None:
    """격자는 `REACH_BAND_M` 위에 있다. 폭을 바꿔도 성질은 같아야 한다."""
    step = arm_reference.REACH_BAND_M
    assert arm_reference.reach_band(0.0) == "+0.00"
    assert arm_reference.reach_band(step * 0.5) == f"{0.0:+.2f}"
    assert arm_reference.reach_band(step) == f"{step:+.2f}"
    assert arm_reference.reach_band(-step * 0.5) == f"{-step:+.2f}"
    assert arm_reference.reach_band(-step) == f"{-step:+.2f}"


def test_band_edges_survive_floating_point() -> None:
    """`2.8 / 0.2`는 13.999999999999998이다. 그냥 내림하면 경계값이 한 칸 아래로 떨어진다.

    격자 위의 값은 **자기 칸**에 들어가야 한다. 실제로 이 결함이 검사를 통해 잡혔다 —
    reach 2.997이 `+2.80`이 아니라 `+2.60`으로 갔다.

    깨뜨리는 법: `reach_band`에서 `round(...)`를 빼면 실패한다.
    """
    step = arm_reference.REACH_BAND_M
    for index in range(-10, 11):
        edge = index * step
        assert arm_reference.reach_band(edge) == f"{edge:+.2f}", edge


def test_the_quality_ladder_refuses_to_promote_one_worker() -> None:
    """1인 산포는 '숙련자 통상 범위'가 아니라 '그 사람의 습관'이다.

    깨뜨리는 법: `reference_quality`가 작업자 수를 안 보면 실패한다.
    """
    assert arm_reference.reference_quality({"n": 500, "workers": 1}) == "single_worker"
    assert arm_reference.reference_quality({"n": 500, "workers": 6}) == "multi_worker"
    assert arm_reference.reference_quality({"n": 5, "workers": 6}) == "unavailable"
    assert arm_reference.reference_quality({"n": 40, "workers": 2}) == "coarse_bucket"


def test_only_one_joint_is_named_at_a_time() -> None:
    """동시에 여러 단서를 주면 수행은 좋아지고 학습은 나빠진다(Raisbeck & Diekfuss 2017)."""
    normalized = {name: (0.5, 0.5, 0.5) for name in arm_reference.SCORING_JOINTS_ARM}
    sigmas = arm_reference.joint_sigmas(normalized, _bucket())
    corrections = arm_reference.build_corrections(sigmas, {})
    assert len(corrections) == 1
    assert arm_reference.MAX_CORRECTIONS == 1


def test_a_pose_inside_the_expert_range_gets_no_correction() -> None:
    """문턱 아래를 짚으면 '항상 뭔가 틀렸다'가 되어 단서가 신호를 잃는다."""
    normalized = {name: (0.01, 0.0, 0.0) for name in arm_reference.SCORING_JOINTS_ARM}
    sigmas = arm_reference.joint_sigmas(normalized, _bucket())
    assert arm_reference.build_corrections(sigmas, {}) == []


def test_the_worst_joint_is_the_one_named() -> None:
    normalized = {name: (0.0, 0.0, 0.0) for name in arm_reference.SCORING_JOINTS_ARM}
    normalized["LeftForeArm"] = (0.40, 0.0, 0.0)
    sigmas = arm_reference.joint_sigmas(normalized, _bucket())
    corrections = arm_reference.build_corrections(sigmas, {"LeftForeArm": "왼쪽 팔꿈치"})
    assert corrections[0].joint == "LeftForeArm"
    assert corrections[0].label == "왼쪽 팔꿈치"
    assert corrections[0].severity == "high"


def test_expected_gain_is_positive_and_bounded() -> None:
    """모의값이다 — 이 관절을 숙련자 평균에 맞췄을 때 점수가 얼마나 오르는지."""
    normalized = {name: (0.0, 0.0, 0.0) for name in arm_reference.SCORING_JOINTS_ARM}
    normalized["RightHand"] = (0.25, 0.0, 0.0)
    sigmas = arm_reference.joint_sigmas(normalized, _bucket())
    base = arm_reference.similarity_score(sigmas)
    correction = arm_reference.build_corrections(sigmas, {})[0]
    assert base is not None and correction.expected_gain is not None
    assert correction.expected_gain > 0
    assert base + correction.expected_gain <= 1.0 + 1e-9


def test_a_typical_expert_sits_near_one_sigma_not_root_three() -> None:
    """축마다 정확히 1σ인 자세는 관절 σ가 **1.0**이어야 한다.

    제곱합 제곱근으로 내면 √3 = 1.73이 되고 `exp(−W²/2)`가 0.22로 떨어진다 —
    **통상적인** 숙련자가 낮은 점수를 받는다. 실제로 숙련자 27패스 p50이 0.016이었다.

    깨뜨리는 법: `joint_sigmas`에서 `total / 3.0`의 나눗셈을 빼면 실패한다.
    """
    sd = 0.05
    normalized = {name: (sd, sd, sd) for name in arm_reference.SCORING_JOINTS_ARM}
    sigmas = arm_reference.joint_sigmas(normalized, _bucket(sd=sd))
    assert all(item.axis_sigmas == pytest.approx((1.0, 1.0, 1.0)) for item in sigmas)
    assert all(item.sigma == pytest.approx(1.0) for item in sigmas), "축 3개가 σ를 부풀렸다"
    score = arm_reference.similarity_score(sigmas)
    assert score == pytest.approx(math.exp(-0.5), abs=1e-9)
    assert score > 0.6, "축마다 1σ인 통상적인 숙련자가 0.6 아래를 받았다"


# 단조 회전을 쓰면 안 된다 — 축마다의 중앙값이 **가운데 프레임의 값과 정확히 같아져서**
# 날것 중앙값 경로가 우연히 옳은 답을 낸다. 그러면 대조가 공허해진다(실제로 그랬다).
# 왔다 갔다 하는 회전이라야 축마다 다른 프레임이 뽑혀 진짜로 섞인 자세가 만들어진다.
_WOBBLE_DEGREES: Final = (0.0, 40.0, -40.0, 20.0, -20.0)


def _rotated_frames() -> list[dict[str, tuple[float, float, float]]]:
    """몸이 프레임마다 돌아가는 시퀀스. 자세 자체는 한 번도 바뀌지 않는다."""
    base = _pose()
    return [
        {name: _rotate_y(point, angle) for name, point in base.items()}
        for angle in _WOBBLE_DEGREES
    ]


def test_normalizing_comes_before_taking_the_median() -> None:
    """순서가 뒤집히면 프레임마다 다른 몸 방향이 한 점에 섞인다.

    몸이 돌기만 하고 자세는 그대로인 시퀀스를 넣으면, 제대로 된 순서는 **정지한 자세와
    같은 값**을 내야 한다. 날것 중앙값을 먼저 내면 그렇지 않다.

    이 결함이 실제로 숙련자를 숙련자 기준에 대고도 0.008점으로 만들었다.

    깨뜨리는 법: `normalize_sequence`가 `normalize(joints)` 대신 날것 `joints`를
    모으게 하면 실패한다.
    """
    still = arm_reference.normalize_sequence([_pose()])
    turning = arm_reference.normalize_sequence(_rotated_frames())
    assert still is not None and turning is not None
    for joint, point in still.items():
        assert turning[joint] == pytest.approx(point, abs=1e-9), joint

    # 대조 — **왜** 순서가 문제인지를 직접 보인다. 회전이 섞인 날것 좌표의 중앙값은
    # 어느 프레임의 자세도 아니고, 강체가 아니다: 뼈가 줄어든다. 줄어든 골격으로
    # 프레임을 세우면 그 위에서 잰 σ의 출처가 자세가 아니게 된다.
    raw: dict[str, list[list[float]]] = {}
    for frame in _rotated_frames():
        for name, point in frame.items():
            slot = raw.setdefault(name, [[], [], []])
            for index in range(3):
                slot[index].append(point[index])
    blended = {
        name: tuple(sorted(axis)[len(axis) // 2] for axis in axes) for name, axes in raw.items()
    }
    true_width = math.dist(_pose()["LeftShoulder"], _pose()["RightShoulder"])
    blended_width = math.dist(blended["LeftShoulder"], blended["RightShoulder"])
    assert true_width == pytest.approx(0.0567, abs=1e-4)
    assert blended_width < true_width - 1e-3, (
        "날것 중앙값이 강체를 보존했다 — 이 대조가 공허하다"
    )


def test_the_ladder_refuses_to_form_from_too_few_scores() -> None:
    """표본이 모자라면 사다리를 만들지 않는다 — 없는 분포를 지어내면 백분위가 소설이 된다.

    깨뜨리는 법: `percentile_ladder`의 `MIN_BUCKET_N` 검사를 빼면 실패한다.
    """
    assert arm_reference.percentile_ladder([0.5, 0.6, 0.7]) == {}
    ladder = arm_reference.percentile_ladder([i / 100 for i in range(arm_reference.MIN_BUCKET_N)])
    assert set(ladder) == {f"p{p}" for p in arm_reference.PERCENTILE_POINTS}


def test_a_percentile_from_a_stale_ladder_is_refused() -> None:
    """사다리는 **잰 당시의 버킷**에만 유효하다.

    평균·산포만 다시 굽고 사다리를 그대로 두면 틀린 백분위가 조용히 나간다.
    `reference_pass_count`가 그것을 막는다.

    깨뜨리는 법: `expert_percentile`의 대조를 빼면 실패한다.
    """
    ladder = {"p10": 0.2, "p25": 0.4, "p50": 0.6, "p75": 0.8, "p90": 0.9}
    fresh = {"n": 200, "expert_scores": {**ladder, "reference_pass_count": 200}}
    stale = {"n": 240, "expert_scores": {**ladder, "reference_pass_count": 200}}
    assert arm_reference.expert_percentile(fresh, 0.65) == 50
    assert arm_reference.expert_percentile(fresh, 0.05) == 0
    assert arm_reference.expert_percentile(fresh, 0.95) == 90
    assert arm_reference.expert_percentile(stale, 0.65) is None, "버킷이 바뀐 사다리를 썼다"
    assert arm_reference.expert_percentile({"n": 200}, 0.65) is None


def _bucket_with_shape(radius: float = 0.50, radius_sd: float = 0.07,
                       angular_sd: float = 0.80) -> dict:
    """반경·방향을 실은 버킷. 값은 레벨1 실측(손 반경 0.513 ± 0.074 m, 방향 SD 0.83 rad)에서 왔다."""
    bucket = _bucket()
    for name, entry in bucket["joints"].items():
        entry["radius"] = {"mean": radius, "sd": radius_sd}
        entry["heading"] = {"mean": [1.0, 0.0, 0.0], "angular_sd": angular_sd}
        entry["mean"] = [radius, 0.0, 0.0]
        entry["sd"] = [0.2, 0.2, 0.2]
    return bucket


def test_over_extension_is_caught_even_where_the_axes_are_loose() -> None:
    """손의 산포는 96%가 **방향** 성분이라 축별 SD가 0.2 m대로 크다.

    축으로만 재면 그 큰 SD가 뻗음을 희석해 15 cm 과신전이 1σ도 안 된다 — 실측에서
    손을 45 cm 옮겨도 교정이 안 나갔다. 반경을 따로 재면 잡힌다.

    깨뜨리는 법: `joint_sigmas`에서 `radius` 가지를 지우면 축 RMS로 내려가 실패한다.
    """
    bucket = _bucket_with_shape()
    # 반경 방향으로 15 cm 더 뻗은 자세. 방향은 기준과 같다.
    normalized = {name: (0.65, 0.0, 0.0) for name in arm_reference.SCORING_JOINTS_ARM}
    sigmas = arm_reference.joint_sigmas(normalized, bucket)
    item = sigmas[0]
    assert item.radial_sigma == pytest.approx(0.15 / 0.07, abs=1e-6)
    assert item.angular_sigma == pytest.approx(0.0, abs=1e-9), "방향은 그대로인데 σ가 붙었다"
    assert item.sigma > arm_reference.CORRECTION_SIGMA, "15 cm 과신전이 문턱을 못 넘었다"

    # 대조 — 같은 편차를 축으로만 재면 문턱을 못 넘는다는 것을 직접 보인다.
    axis_only = math.sqrt(((0.15 / 0.2) ** 2) / 3.0)
    assert axis_only < arm_reference.CORRECTION_SIGMA


def test_pointing_elsewhere_shows_up_as_direction_not_reach() -> None:
    """둘은 직교한다. 방향만 틀면 반경 σ가 0이어야 한다 — 그래야 증거가 원인을 가리킨다."""
    bucket = _bucket_with_shape()
    turned = (0.0, 0.50, 0.0)  # 같은 반경, 90° 돌아간 방향
    sigmas = arm_reference.joint_sigmas(
        {name: turned for name in arm_reference.SCORING_JOINTS_ARM}, bucket
    )
    item = sigmas[0]
    assert item.radial_sigma == pytest.approx(0.0, abs=1e-9)
    assert item.angular_sigma == pytest.approx((math.pi / 2) / 0.80, abs=1e-6)
    assert item.radius_delta_m == pytest.approx(0.0, abs=1e-9)


def test_reach_and_direction_combine_as_their_rms() -> None:
    """둘 다 1σ면 관절 σ는 1.0이다 — 성분 수가 점수를 흔들면 안 된다."""
    bucket = _bucket_with_shape(radius_sd=0.10, angular_sd=0.50)
    length = 0.60  # 반경 +0.10 = 1σ
    angle = 0.50   # 방향 0.50 rad = 1σ
    point = (length * math.cos(angle), length * math.sin(angle), 0.0)
    item = arm_reference.joint_sigmas(
        {name: point for name in arm_reference.SCORING_JOINTS_ARM}, bucket
    )[0]
    assert item.radial_sigma == pytest.approx(1.0, abs=1e-6)
    assert item.angular_sigma == pytest.approx(1.0, abs=1e-6)
    assert item.sigma == pytest.approx(1.0, abs=1e-6)


def test_an_old_reference_without_reach_still_scores() -> None:
    """반경·방향을 싣기 전에 구운 아티팩트가 남아 있다. 죽지 말고 축 RMS로 내려간다."""
    plain = _bucket()
    assert "radius" not in plain["joints"]["RightHand"]
    item = arm_reference.joint_sigmas(
        {name: (0.05, 0.05, 0.05) for name in arm_reference.SCORING_JOINTS_ARM}, plain
    )[0]
    assert item.radial_sigma is None and item.angular_sigma is None
    assert item.sigma == pytest.approx(1.0, abs=1e-9)


def test_the_angular_floor_stops_a_tight_bucket_from_exploding() -> None:
    """방향 산포가 0에 가까운 버킷에서 각 σ가 무한히 부풀면 전원이 교정 대상이 된다."""
    bucket = _bucket_with_shape(angular_sd=1e-9)
    point = (0.50 * math.cos(0.05), 0.50 * math.sin(0.05), 0.0)
    item = arm_reference.joint_sigmas(
        {name: point for name in arm_reference.SCORING_JOINTS_ARM}, bucket
    )[0]
    assert item.angular_sigma == pytest.approx(0.05 / arm_reference.ANGULAR_SD_FLOOR, abs=1e-6)


def test_the_builder_records_a_rigid_bone_as_a_tight_radius() -> None:
    """위팔은 어깨에 0.1797 m 뼈로 묶여 있다 — 반경 SD가 0에 가깝게 나와야 한다.

    레벨1 실측이 0.002~0.004 m다. 이 값이 커지면 굽는 쪽이 반경을 잘못 쌓고 있는 것이다.
    """
    accum = arm_reference.BucketAccum()
    for angle in (0.0, 0.4, -0.4, 0.9, -0.9, 0.2):
        point = (0.2 * math.cos(angle), 0.2 * math.sin(angle), 0.0)
        accum.add({"RightUpperArm": point}, "w")
    entry = accum.payload()["joints"]["RightUpperArm"]
    assert entry["radius"]["mean"] == pytest.approx(0.2, abs=1e-9)
    assert entry["radius"]["sd"] == pytest.approx(0.0, abs=1e-6), "강체인데 반경이 흔들렸다"
    assert entry["heading"]["angular_sd"] > 0.1, "방향은 흩어져 있는데 0으로 나왔다"
