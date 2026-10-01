"""몸 제약 개선율 측정의 순수 부분 — 측정 전에 실현 규칙과 집계가 실패할 수 있는지 본다.

사전등록: `docs/measurements/2026-10-01-body-constrained-improvement-prereg.md`
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import body_constrained_study as bcs  # noqa: E402
import measured_score  # noqa: E402

FRAME = measured_score.WORLD_FRAME


def _body() -> dict[str, tuple[float, float, float]]:
    return {
        "Pelvis": (0.0, 1.0, 0.0), "Chest": (0.0, 1.4, 0.0), "Neck": (0.0, 1.55, 0.0), "Head": (0.0, 1.65, 0.0),
        "RightShoulder": (0.1, 1.45, 0.0), "RightUpperArm": (0.2, 1.45, 0.0), "RightForeArm": (0.2, 1.17, 0.05),
        "RightHand": (0.2, 0.92, 0.1),
        "LeftShoulder": (-0.1, 1.45, 0.0), "LeftUpperArm": (-0.2, 1.45, 0.0), "LeftForeArm": (-0.2, 1.17, 0.05),
        "LeftHand": (-0.2, 0.92, 0.1),
        "LeftThigh": (-0.1, 0.95, 0.0), "LeftCalf": (-0.1, 0.55, 0.12), "LeftFoot": (-0.1, 0.08, 0.0),
        "LeftToe": (-0.1, 0.0, 0.12),
        "RightThigh": (0.1, 0.95, 0.0), "RightCalf": (0.1, 0.55, 0.12), "RightFoot": (0.1, 0.08, 0.0),
        "RightToe": (0.1, 0.0, 0.12),
    }


def _cue(point: str, unit: tuple[float, float, float], cm: float) -> Any:
    return measured_score.AxisCorrection(point=point, axis="vertical", magnitude_cm=cm, direction="", sigma=1.0, unit=unit)


def _tracks() -> dict[str, list[Any]]:
    return {p: [(0.1 * i, 0.2, 0.3 + 0.01 * i) for i in range(5)] for p in measured_score.POINT_ORDER}


def test_reachable_hand_cues_realize_exactly_like_the_independent_model() -> None:
    """팔이 닿고 머리 안내가 없으면 몸 제약 실현 = 기존 점 독립 실현(재현). 측정 장치가 차이를 지어내지 않는다.

    깨뜨리는 법: 실현 이동을 손이 아닌 관절(예: 팔꿈치)에서 재면 실패한다.
    """
    chosen = [_cue("right", (0.0, 1.0, 0.0), 5.0), _cue("right", (0.0, 0.0, 1.0), -4.0)]
    shifts, short = bcs.realized_shifts(_body(), chosen, FRAME)
    model = measured_score.apply_corrections(_tracks(), chosen, FRAME)
    realized = measured_score.shift_tracks(_tracks(), shifts)
    for point in ("right", "head", "left"):
        flat = [v for p in realized[point] for v in p]
        expected = [v for p in model[point] for v in p]
        assert flat == pytest.approx(expected, abs=1e-9), point
    assert short["right"] == pytest.approx(0.0, abs=1e-6)


def test_stepping_carries_the_uncued_hands_unlike_the_independent_model() -> None:
    """옮겨 서기(머리 수평)에서 안내 없는 손은 몸을 따라간다 — 점 독립 모델은 제자리에 둔다. 두 값이 갈리는 곳이다."""
    chosen = [_cue("head", (1.0, 0.0, 0.0), 20.0)]
    shifts, _short = bcs.realized_shifts(_body(), chosen, FRAME)
    assert shifts["left"] == pytest.approx((0.20, 0.0, 0.0))
    assert shifts["right"] == pytest.approx((0.20, 0.0, 0.0))
    model = measured_score.apply_corrections(_tracks(), chosen, FRAME)
    assert model["left"] == _tracks()["left"]


def test_out_of_reach_cues_are_realized_short_and_measured() -> None:
    """손이 닿지 않거나 발을 디딘 채 못 서면, 실현 이동이 안내보다 짧고 그 거리가 기록된다."""
    chosen = [_cue("right", (0.0, 0.0, 1.0), 120.0), _cue("head", (0.0, 1.0, 0.0), 60.0)]
    shifts, short = bcs.realized_shifts(_body(), chosen, FRAME)
    assert short["right"] > 10.0 and short["head"] > 10.0
    assert shifts["head"][1] < 0.60


def _rec(worker: str, model: float, body: float | None, skeleton: bool = True) -> dict[str, Any]:
    return {"status": "ok", "worker": worker, "skeleton": skeleton, "issued.values": 4,
            "issued.same": model, "issued.same_body": body, "issued.split": model, "issued.split_body": body,
            "issued.short_hand_cm": 0.0, "issued.short_head_cm": 2.0}


def test_aggregate_compares_only_pairs_that_have_both_values() -> None:
    """두 값이 다 있는 쌍끼리만 견준다. 골격이 없는 쌍은 점 독립 평균(`model_all_ok`)에만 들어간다.

    깨뜨리는 법: 몸 제약 열을 모든 쌍 기준으로 내면 모집단이 달라져 n이 어긋나 실패한다.
    """
    records = [_rec("A", 0.8, 0.9), _rec("A", 0.6, 0.5), _rec("B", 0.2, None, skeleton=False)]
    report = bcs.aggregate(records, ["issued"])
    block = report["policies"]["issued"]["split"]
    assert block["model"]["n"] == 2 and block["body"]["n"] == 2
    assert block["model"]["mean"] == pytest.approx(0.7)
    assert block["body"]["mean"] == pytest.approx(0.7)
    assert block["difference"]["mean"] == pytest.approx(0.0)
    assert block["body_lower_share"] == pytest.approx(0.5)
    assert block["model_all_ok"]["n"] == 3
    assert report["policies"]["issued"]["short_head_over_1cm_share"] == pytest.approx(1.0)


def test_csv_reports_percent_against_both_targets(tmp_path: Path) -> None:
    import csv

    records = [_rec("A", 0.9, 0.95) for _ in range(12)]
    report = bcs.aggregate(records, ["issued"])
    written = bcs.write_csvs(records, report, ["issued"], tmp_path)
    assert {p.name for p in written} == {"정확도_평가방법.csv", "몸제약_차이.csv", "못미친거리.csv", "작업자별.csv",
                                         "쌍별_원자료.csv"}
    with (tmp_path / "정확도_평가방법.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    body = next(r for r in rows if r["교정 실현"] == "몸 제약" and r["채점 방식"].startswith("교정·채점 기준 분리"))
    assert float(body["정확도 평균(%)"]) == pytest.approx(95.0)
    assert (body["2차년도 70% 달성"], body["3차년도 90% 달성"]) == ("True", "True")


def test_stay_keeps_an_uncued_hand_where_it_was_when_the_body_steps() -> None:
    """사후 민감도 가정 `stay`: 옮겨 서도 안내 없는 손은 벽 기준 원래 자리(닿는 데까지)를 지킨다 — `follow`와 갈린다.

    깨뜨리는 법: `stay`에서도 손을 몸에 붙여 옮기면 이동이 20cm가 되어 실패한다.
    """
    chosen = [_cue("head", (1.0, 0.0, 0.0), 10.0)]
    follow, _ = bcs.realized_shifts(_body(), chosen, FRAME, uncued_hands="follow")
    stay, short = bcs.realized_shifts(_body(), chosen, FRAME, uncued_hands="stay")
    assert follow["right"] == pytest.approx((0.10, 0.0, 0.0))
    assert stay["right"] == pytest.approx((0.0, 0.0, 0.0), abs=1e-6)
    assert stay["head"] == pytest.approx((0.10, 0.0, 0.0))
