"""물려받은 산포 측정의 순수 부분. **측정 전에** 통제가 작동하는지 본다.

사전등록: `docs/measurements/2026-09-30-sigma-inherited-spread-prereg.md`

코퍼스는 읽지 않는다. 읽는 부분(`collect`)은 CSV 코퍼스가 없는 검사 환경에서 돌릴
수 없고, 그 대신 **판정을 내리는 계산**이 심은 구조를 실제로 집어내는지를 본다.

여기서 가장 중요한 것은 **통제가 실패할 수 있는가**다. 상관된 열을 모으면 유효
차원은 늘 작아지므로, 작업자 섞기 통제가 그것을 걸러내지 못하면 이 측정은 늘
"중복이다"를 낸다.
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sigma_inherited as si  # noqa: E402


def _spin(axis: str, angle_rad: float, point: tuple[float, float, float]):
    """축 하나를 중심으로 회전. 사슬을 통째로 돌려 '물려받음'을 만든다."""
    cos, sin = math.cos(angle_rad), math.sin(angle_rad)
    x, y, z = point
    if axis == "y":
        return (cos * x + sin * z, y, -sin * x + cos * z)
    return (x, cos * y - sin * z, sin * y + cos * z)


def _rows(*, chain_jitter_deg: float, wrist_jitter_deg: float,
          workers: int = 6, per_worker: int = 40, seed: int = 1) -> list[dict[str, object]]:
    """어깨에서 사슬을 `chain_jitter_deg`만큼 흔들고, 손목만 `wrist_jitter_deg` 더 흔든다.

    `chain_jitter`만 크면 손의 절대 산포는 크지만 **부모 기준으로는 작다** —
    물려받은 경우다. `wrist_jitter`만 크면 반대다.
    """
    rng = random.Random(seed)
    shoulder = (0.028, 0.0, 0.0)
    upper = (0.208, 0.0, 0.0)
    fore = (0.208, -0.243, 0.0)
    hand = (0.208, -0.486, 0.0)

    out: list[dict[str, object]] = []
    for worker in range(workers):
        for _ in range(per_worker):
            chain = math.radians(rng.gauss(0.0, chain_jitter_deg))
            wrist = math.radians(rng.gauss(0.0, wrist_jitter_deg))
            pose = {
                "RightShoulder": shoulder,
                "RightUpperArm": _spin("y", chain, upper),
                "RightForeArm": _spin("y", chain, fore),
                # 손은 사슬 회전 **위에** 손목 회전이 더 얹힌다.
                "RightHand": _spin("y", chain, _spin("x", wrist, hand)),
            }
            out.append({"worker": f"W{worker}", "pose": pose})
    return out


def test_angular_spread_is_zero_for_a_single_direction() -> None:
    points = [(0.0, -1.0, 0.0)] * 10
    assert si.angular_spread_deg(points) == pytest.approx(0.0, abs=1e-9)


def test_angular_spread_grows_with_the_planted_jitter() -> None:
    """심은 흔들림이 커지면 산포도 커져야 한다 — 자가 자가 아니면 뒤 전부가 무의미하다."""
    rng = random.Random(3)
    small = [_spin("x", math.radians(rng.gauss(0.0, 2.0)), (0.0, -1.0, 0.0)) for _ in range(400)]
    rng = random.Random(3)
    large = [_spin("x", math.radians(rng.gauss(0.0, 20.0)), (0.0, -1.0, 0.0)) for _ in range(400)]
    assert si.angular_spread_deg(small) < si.angular_spread_deg(large)


def test_segment_angles_read_the_chain_not_the_origin() -> None:
    """어깨 마디는 **위팔→아래팔**, 팔꿈치 각은 두 마디 사이 각이다.

    곧게 편 팔은 팔꿈치 각이 0°다. 원점 기준으로 재면 이 값이 안 나온다.

    깨뜨리는 법: 마디를 원점 기준 위치로 바꾸면 각이 0이 아니게 되어 실패한다.
    """
    straight = {
        "RightUpperArm": (0.208, 0.0, 0.0),
        "RightForeArm": (0.208, -0.243, 0.0),
        "RightHand": (0.208, -0.486, 0.0),
    }
    direction, elbow = si.segment_angles(straight, "Right")
    assert elbow == pytest.approx(0.0, abs=1e-6), "곧은 팔인데 팔꿈치가 굽었다고 나온다"
    assert direction == pytest.approx((0.0, -1.0, 0.0), abs=1e-9)

    bent = dict(straight, RightHand=(0.451, -0.243, 0.0))
    _, bent_elbow = si.segment_angles(bent, "Right")
    assert bent_elbow == pytest.approx(90.0, abs=1e-6)


def test_effective_dimensions_counts_one_for_a_rigid_chain() -> None:
    """모든 열이 같이 움직이면 유효 차원은 1이어야 한다.

    깨뜨리는 법: 고윳값 정렬이 뒤집히면 1이 아니라 6이 나온다.
    """
    rng = random.Random(5)
    matrix = [[value] * 6 for value in (rng.gauss(0.0, 1.0) for _ in range(300))]
    assert si.effective_dimensions(matrix) == 1


def test_effective_dimensions_counts_all_for_independent_columns() -> None:
    """열이 서로 무관하면 유효 차원이 열 수에 가까워야 한다."""
    rng = random.Random(5)
    matrix = [[rng.gauss(0.0, 1.0) for _ in range(6)] for _ in range(600)]
    assert si.effective_dimensions(matrix) >= 5


def test_shuffling_within_workers_breaks_a_real_correlation() -> None:
    """**통제다.** 같은 패스의 관절이 함께 움직여 생긴 상관이라면, 짝을 흩으면
    무너져 유효 차원이 늘어야 한다.

    깨뜨리는 법: `_shuffle_columns`가 열을 안 섞으면(같은 순열을 모든 열에 쓰면)
    상관이 그대로 남아 여기서 실패한다.
    """
    rng = random.Random(5)
    matrix = [[value + rng.gauss(0.0, 0.05) for _ in range(6)]
              for value in (rng.gauss(0.0, 1.0) for _ in range(600))]
    workers = [f"W{index % 6}" for index in range(len(matrix))]
    assert si.effective_dimensions(matrix) == 1

    shuffled = si._shuffle_columns(matrix, workers, seed=7)
    assert si.effective_dimensions(shuffled) >= int(si.SHUFFLED_DIMS_MIN), (
        "짝을 흩었는데 상관이 안 무너졌다 — 통제가 작동하지 않는다"
    )


def test_shuffling_keeps_every_value_inside_its_worker() -> None:
    """작업자 간 차이를 상관으로 되살리면 안 된다 — 섞기는 작업자 **안에서**만."""
    matrix = [[float(index)] * 6 for index in range(12)]
    workers = [f"W{index // 6}" for index in range(12)]
    shuffled = si._shuffle_columns(matrix, workers, seed=7)
    for worker in ("W0", "W1"):
        before = sorted(matrix[i][0] for i, w in enumerate(workers) if w == worker)
        after = sorted(shuffled[i][0] for i, w in enumerate(workers) if w == worker)
        assert before == after, f"{worker}의 값이 다른 작업자로 넘어갔다"


def test_the_verdict_table_matches_what_was_registered() -> None:
    """개정 등록한 표 그대로인지. 문턱을 나중에 옮기면 여기서 실패한다."""
    assert (si.DOMINANCE, si.VARIANCE_COVERED, si.REDUNDANT_DIMS, si.SHUFFLED_DIMS_MIN) == (
        1.5, 0.90, 4, 5.0
    )
    assert si.verdict({"shoulder_direction_sd_deg": 30.0,
                       "elbow_angle_sd_deg": 10.0}).startswith("①")
    assert si.verdict({"shoulder_direction_sd_deg": 10.0,
                       "elbow_angle_sd_deg": 30.0}).startswith("②")
    assert si.verdict({"shoulder_direction_sd_deg": 12.0,
                       "elbow_angle_sd_deg": 10.0}).startswith("③")
    assert si.verdict({"shoulder_direction_sd_deg": None,
                       "elbow_angle_sd_deg": 10.0}) == "미측정"


def test_the_summary_flags_a_broken_control() -> None:
    """짝을 흩었는데도 차원이 안 늘면 요약이 **결론을 내지 말라고** 말해야 한다.

    깨뜨리는 법: 경고 줄을 지우면 실패한다.
    """
    def _payload(shuffled_dims: int) -> dict[str, object]:
        return {
            "passes": 600, "workers": 6, "arms": 1200, "enough_passes": True,
            "shoulder_direction_sd_deg": 30.0, "elbow_angle_sd_deg": 10.0,
            "elbow_angle_median_deg": 40.0, "joints": list(si.arm_reference.SCORING_JOINTS_ARM),
            "correlation": [[1.0] * 6 for _ in range(6)], "mean_abs_correlation": 1.0,
            "effective_dimensions": 2, "shuffled_effective_dimensions": shuffled_dims,
            "verdict": "①′ 어깨가 지배한다 — 손 단서는 중복이다",
        }

    assert "계산의 부산물" in "\n".join(si.summarize(_payload(2)))
    assert "계산의 부산물" not in "\n".join(si.summarize(_payload(6)))
    assert "중복이다" in "\n".join(si.summarize(_payload(6)))


def test_the_summary_flags_a_thin_sample() -> None:
    payload = {"passes": 40, "workers": 6, "arms": 80, "joints": [],
               "verdict": "미측정", "enough_passes": False}
    assert str(si.MIN_PASSES) in "\n".join(si.summarize(payload))
