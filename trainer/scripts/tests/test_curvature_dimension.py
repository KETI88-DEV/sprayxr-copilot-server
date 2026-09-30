"""굽이 차원 실험의 순수 부분. **측정 전에** 판정기가 실패할 수 있는지 본다.

사전등록: `docs/measurements/2026-09-30-stroke-curvature-dimension-prereg.md`

여기서 코퍼스는 읽지 않는다. 읽는 부분(`collect`)은 CSV 코퍼스가 없는 검사 환경에서
돌릴 수 없고, 그 대신 **판정을 내리는 계산**이 심은 효과를 실제로 집어내는지를 본다.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import curvature_dimension as cd  # noqa: E402


def _rows(
    *, workers: int = 6, per_worker: int = 40, effect: float = 0.0, seed: int = 1
) -> list[dict[str, object]]:
    """굽이가 클수록 σ가 `effect`만큼 오르는 합성 표본.

    굽이와 σ 말고는 아무 구조도 심지 않는다 — 길이는 굽이와 **무관**하게 둔다.
    그래야 길이 대조가 통제로 작동하는지 볼 수 있다.
    """
    rng = random.Random(seed)
    out: list[dict[str, object]] = []
    for worker in range(workers):
        for index in range(per_worker):
            ratio = index / (per_worker - 1)  # 0.0 ~ 1.0
            out.append({
                "worker": f"W{worker}",
                "band": "-0.10",
                "straightness": 1.0 + ratio,
                "length_m": rng.uniform(0.3, 1.2),
                "random": rng.random(),
                "sigma": 1.0 + effect * ratio + rng.gauss(0.0, 0.01),
            })
    return out


def test_percentile_matches_a_known_ladder() -> None:
    values = [float(v) for v in range(101)]  # 0..100
    assert cd._percentile(values, 50.0) == pytest.approx(50.0)
    assert cd._percentile(values, 90.0) == pytest.approx(90.0)


def test_no_effect_gives_a_delta_near_zero_and_a_ci_covering_zero() -> None:
    """효과를 안 심으면 Δ가 0 근처여야 하고 구간이 0을 덮어야 한다.

    깨뜨리는 법: `_delta`가 두 무리를 뒤집어 잡으면 부호가 커져 실패한다.
    """
    arm = cd.bootstrap(_rows(effect=0.0), "straightness", seed=7)
    assert arm is not None
    assert abs(arm["delta"]) < 0.05
    assert arm["ci_low"] <= 0.0 <= arm["ci_high"]
    assert cd.verdict(arm) == "② 싣기만 한다"


def test_a_planted_effect_is_recovered_with_the_right_sign() -> None:
    """굽은 쪽 σ를 1.0 올리면 Δ가 양수로, 등록 문턱 위로 나와야 한다.

    깨뜨리는 법: `_delta`에서 `high - low`를 `low - high`로 바꾸면 실패한다.
    """
    arm = cd.bootstrap(_rows(effect=1.0), "straightness", seed=7)
    assert arm is not None
    # 굽이 p50 이하의 중앙은 ratio 0.25, p90 이상의 중앙은 ratio 0.95 근처다.
    assert arm["delta"] == pytest.approx(0.70, abs=0.08)
    assert arm["ci_low"] > 0.0, "효과를 심었는데 구간이 0을 덮는다"
    assert cd.verdict(arm) == "① 차원으로 넣는다"


def test_the_random_control_stays_null_when_the_effect_is_real() -> None:
    """굽이에 진짜 효과가 있어도 **난수 대조는 0이어야** 측정 장치가 성한 것이다.

    깨뜨리는 법: `_delta`가 `key`를 무시하고 늘 굽이로 나누면 난수 대조가 같이 살아나
    여기서 실패한다.
    """
    control = cd.bootstrap(_rows(effect=1.0), "random", seed=7)
    assert control is not None
    assert abs(control["delta"]) < 0.20
    assert cd.verdict(control) == "② 싣기만 한다"


def test_the_length_control_stays_null_when_length_is_unrelated() -> None:
    """길이를 굽이와 무관하게 심었으므로 길이 대조도 0이어야 한다."""
    control = cd.bootstrap(_rows(effect=1.0), "length_m", seed=7)
    assert control is not None
    assert abs(control["delta"]) < 0.20


def test_a_worker_only_effect_widens_the_interval() -> None:
    """효과가 **한 작업자에게만** 있으면 구간이 0을 덮고 **넓어야** 한다.

    이것이 작업자 클러스터 부트스트랩을 쓰는 이유다. 실제로 σ가 작업자별로 군집해
    위양성률이 0.015~0.259로 17배 벌어진 전례가 있다.

    점추정은 움직이지 않는다 — 중앙값이라 6인 중 1인의 효과에 둔하다. **구간이 그
    사실을 말해 준다**: 상한이 0.5σ를 넘어 "한 사람이 몰고 갈 수도 있다"를 인정한다.

    깨뜨리는 법: `bootstrap`의 재표집 단위를 패스로 바꾸면 구간이 90배 좁아져
    (실측 폭 1.395 → 0.016) 아래 두 단언이 모두 실패한다.
    """
    rows = _rows(effect=0.0, seed=3)
    for row in rows:
        if row["worker"] == "W0":
            row["sigma"] = float(row["sigma"]) + 3.0 * (float(row["straightness"]) - 1.0)

    arm = cd.bootstrap(rows, "straightness", seed=7)
    assert arm is not None
    assert arm["ci_low"] <= 0.0 <= arm["ci_high"], "한 작업자만의 효과인데 구간이 0을 뺐다"
    assert arm["ci_high"] > 0.5, (
        f"상한 {arm['ci_high']:.3f}이 작다 — 한 사람이 몰고 갈 여지를 구간이 안 담았다"
    )
    assert cd.verdict(arm) == "② 싣기만 한다"

    # 같은 자료를 **패스 단위**로 재표집하면 군집이 사라져 구간이 무너진다.
    rng = random.Random(7)
    draws = []
    for _ in range(cd.BOOTSTRAP_ROUNDS):
        sample = [rows[rng.randrange(len(rows))] for _ in range(len(rows))]
        drawn = cd._delta(sample, "straightness")
        if drawn is not None:
            draws.append(drawn["delta"])
    naive = cd._percentile(draws, cd.CI_HIGH) - cd._percentile(draws, cd.CI_LOW)
    clustered = arm["ci_high"] - arm["ci_low"]
    assert clustered > naive * 10.0, (
        f"군집 폭 {clustered:.4f}이 패스 단위 폭 {naive:.4f}의 10배도 안 된다"
    )


def test_the_verdict_table_matches_what_was_registered() -> None:
    """등록한 표 그대로인지. 문턱을 나중에 옮기면 여기서 실패한다."""
    assert cd.DELTA_DIMENSION == 0.50
    assert cd.DELTA_NEGLIGIBLE == 0.25
    ci = {"ci_low": 0.1, "ci_high": 0.9}
    assert cd.verdict({"delta": 0.60, **ci}) == "① 차원으로 넣는다"
    assert cd.verdict({"delta": 0.10, **ci}) == "② 싣기만 한다"
    assert cd.verdict({"delta": 0.35, **ci}) == "③ 미결 — 넣지 않는다"
    # 구간이 0을 덮으면 크기와 무관하게 ②다.
    assert cd.verdict({"delta": 5.0, "ci_low": -0.2, "ci_high": 9.0}) == "② 싣기만 한다"
    # 음수도 그대로 읽는다 — 부호를 뒤집지 않는다.
    assert cd.verdict({"delta": -0.80, "ci_low": -1.2, "ci_high": -0.3}) == "② 싣기만 한다"


def test_the_summary_flags_a_broken_instrument() -> None:
    """난수 대조가 ①을 통과하면 요약이 **폐기하라고 말해야** 한다.

    깨뜨리는 법: 경고 줄을 지우면 실패한다.
    """
    payload = {
        "passes": 900, "workers": 6, "by_band": {"-0.10": 900},
        "straightness_ladder": {"p50": 1.0},
        "arms": {
            "straightness": {"delta": 0.6, "ci_low": 0.2, "ci_high": 1.0,
                             "low_n": 1, "high_n": 1,
                             "low_median_sigma": 1.0, "high_median_sigma": 1.6},
            "random": {"delta": 0.7, "ci_low": 0.3, "ci_high": 1.1,
                       "low_n": 1, "high_n": 1,
                       "low_median_sigma": 1.0, "high_median_sigma": 1.7},
        },
        "verdict": "① 차원으로 넣는다", "enough_passes": True,
    }
    text = "\n".join(cd.summarize(payload))
    assert "측정 장치가 고장났다" in text
    assert "폐기" in text


def test_the_summary_flags_a_thin_sample() -> None:
    """등록 하한에 못 미치면 요약이 그렇게 말해야 한다."""
    payload = {
        "passes": 40, "workers": 6, "by_band": {"-0.10": 40},
        "straightness_ladder": {}, "arms": {}, "verdict": "② 싣기만 한다",
        "enough_passes": False,
    }
    text = "\n".join(cd.summarize(payload))
    assert str(cd.MIN_PASSES) in text
    assert "그대로 믿지 않는다" in text
