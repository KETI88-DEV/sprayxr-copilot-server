"""3차년도 교정 정책과 전수 집계 — 측정 전에 판정기가 실패할 수 있는지 본다.

사전등록: `docs/year3/2026-10-01-six-value-correction-prereg.md`
"""

from __future__ import annotations

import csv
import importlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

measured_score = importlib.import_module("measured_score")

from trainer.pose_lab import service  # noqa: E402
from trainer.pose_lab.service import PassRef  # noqa: E402
from trainer.pose_lab.tests import test_scoring_isolation as isolation  # noqa: E402
from trainer.year3 import correction_policies as cp  # noqa: E402
from trainer.year3 import six_value_study as study  # noqa: E402

FRAME = measured_score.WORLD_FRAME
WEIGHTS = measured_score.DEFAULT_WEIGHTS
SPREADS = measured_score.uniform_spreads(0.05)


def _tracks(offsets: dict[str, tuple[float, float, float]]) -> tuple[dict[str, list[Any]], dict[str, list[Any]]]:
    """기준은 원점, 질의는 점마다 `offsets`만큼 떨어져 있다(격자 4칸)."""
    reference = {p: [(0.0, 0.0, 0.0)] * 4 for p in measured_score.POINT_ORDER}
    query = {p: [offsets.get(p, (0.0, 0.0, 0.0))] * 4 for p in measured_score.POINT_ORDER}
    return query, reference


def _deltas(query: dict[str, list[Any]], reference: dict[str, list[Any]]) -> list[Any]:
    return measured_score.score_track(query, reference, SPREADS, WEIGHTS, FRAME).deltas


def _call(policy: Any, offsets: dict[str, tuple[float, float, float]]) -> list[Any]:
    query, reference = _tracks(offsets)
    return list(policy(query, reference, SPREADS, WEIGHTS, FRAME, _deltas(query, reference)))


def test_candidates_include_the_left_hand_and_every_head_axis() -> None:
    """2차년도 제품은 왼손을 옮기지 않고 머리는 축 하나다. 3차년도 후보는 셋 다 연다.

    깨뜨리는 법: `candidate_values`가 `issued_axes`를 그대로 내면 왼손·머리 둘째 축이 빠져 실패한다.
    """
    offsets = {"right": (0.10, 0.08, 0.0), "head": (0.06, 0.05, 0.0), "left": (0.07, 0.0, 0.0)}
    values = _call(lambda *a: cp.candidate_values(a[5], a[4]), offsets)
    assert {(c.point, c.axis) for c in values} >= {("left", "wall"), ("head", "wall"), ("head", "lateral")}
    product = _call(cp.product_policy, offsets)
    assert all(c.point != "left" for c in product)
    assert sum(1 for c in product if c.point == "head") == 1


def test_candidates_keep_the_product_gates() -> None:
    """σ 문턱과 1cm 하한은 제품과 같다 — 0.5cm 이탈은 후보가 아니다."""
    values = _call(lambda *a: cp.candidate_values(a[5], a[4]), {"left": (0.005, 0.0, 0.0)})
    assert values == []


def test_every_eligible_policy_adjusts_at_most_six_values() -> None:
    """9개 값이 다 문턱을 넘어도 P1·P2는 6개를 넘지 않는다. P3만 넘는다(참고용).

    깨뜨리는 법: `MAX_VALUES`를 무시하면 9가 나와 실패한다.
    """
    offsets = {"right": (0.10, 0.08, 0.06), "head": (0.09, 0.07, 0.05), "left": (0.08, 0.06, 0.04)}
    assert len(_call(cp.sigma_top_policy, offsets)) == 6
    assert len(_call(cp.greedy_policy, offsets)) == 6
    assert len(_call(cp.all_values_policy, offsets)) == 9


def test_greedy_prefers_the_value_that_raises_similarity_most() -> None:
    """σ가 같아도 가중이 큰 점(오른손 1.0 > 왼손 0.15)을 먼저 고친다 — σ 순서와 다르다.

    왼손 σ를 오른손보다 크게 두면 σ 순서는 왼손이 먼저지만 유사도는 오른손을 고칠 때 더 오른다.
    깨뜨리는 법: 탐욕 선택이 점수 대신 σ로 고르면 첫 값이 왼손이 되어 실패한다.
    """
    offsets = {"right": (0.10, 0.0, 0.0), "left": (0.20, 0.0, 0.0)}
    assert _call(cp.sigma_top_policy, offsets)[0].point == "left"
    assert _call(cp.greedy_policy, offsets)[0].point == "right"


def test_greedy_stops_when_nothing_raises_the_score() -> None:
    """이탈이 없으면 아무것도 옮기지 않는다."""
    assert _call(cp.greedy_policy, {}) == []


def test_correcting_every_value_reaches_the_target() -> None:
    """모든 축을 평균 이탈만큼 옮기면 유사도가 1이 된다 — 그래서 값 수 제한(6개)이 정확도의 천장을 정한다.

    이 성질을 결과 문서에 밝힌다: P3(값 제한 없음)의 100% 근처는 교정의 품질이 아니라 정의에서 나온다.
    """
    offsets = {"right": (0.10, 0.08, 0.06), "head": (0.09, 0.07, 0.05), "left": (0.08, 0.06, 0.04)}
    query, reference = _tracks(offsets)
    chosen = cp.all_values_policy(query, reference, SPREADS, WEIGHTS, FRAME, _deltas(query, reference))
    after = measured_score.score_track(
        measured_score.apply_corrections(query, chosen, FRAME), reference, SPREADS, WEIGHTS, FRAME
    ).score
    assert after == pytest.approx(1.0)


@pytest.mark.parametrize("name", [n for n in isolation.CASE_NAMES if not n.startswith("synthetic")])
def test_p0_reproduces_the_product_improvement_exactly(name: str) -> None:
    """연결부(`correction_policies`)로 제품 정책을 돌리면 `improvement_all`과 같아야 한다 — same·split 둘 다.

    실데이터 쌍이라 split이 계산된다. 깨뜨리는 법: 연결부가 split 정책을 A가 아니라 전체 산포로 고르면
    split 값이 달라져 실패한다.
    """
    case = next(c for c in isolation._cases() if c["name"] == name)
    source, kwargs = isolation._source_for(case)
    body = service.compare(
        source, PassRef(**case["query"]), PassRef(**case["reference"]), include_view=False,
        correction_policies={"P0_product": cp.product_policy}, **kwargs,
    )
    measured = body["measured"]
    p0 = measured["policies"]["P0_product"]
    assert measured["improvement_all"]["split"] is not None
    assert p0["improvement"]["same"] == pytest.approx(measured["improvement_all"]["same"], abs=1e-12)
    assert p0["improvement"]["split"] == pytest.approx(measured["improvement_all"]["split"], abs=1e-12)
    assert p0["score_after"] == pytest.approx(measured["score_after_all"], abs=1e-12)


def test_compare_without_policies_has_no_policy_field() -> None:
    """정책을 주지 않으면 응답에 새 필드가 없다 — 2차년도 화면·채점 격리 기준선이 그대로다."""
    case = next(c for c in isolation._cases() if c["name"].startswith("synthetic"))
    body = isolation._run(case, include_view=False)
    assert "policies" not in body["measured"]


def _record(**policy_values: Any) -> dict[str, Any]:
    out: dict[str, Any] = {"status": "ok", "query": "q", "worker": "A", "score": 0.5,
                           "same_product_all": 0.8, "split_product_all": 0.7}
    for name in cp.POLICIES:
        out.update({f"{name}.values": 4, f"{name}.values_split": 4, f"{name}.points": "right+head",
                    f"{name}.score_after": 0.9, f"{name}.same": 0.8, f"{name}.split": 0.7})
    out.update(policy_values)
    return out


def test_reproduction_check_can_fail() -> None:
    """P0이 제품 값과 다르면 집계가 그 쌍을 센다. 깨뜨리는 법: 비교를 지우면 0이 나와 실패한다."""
    good = study.aggregate([_record()])
    assert good["reproduction_mismatches"] == {"same": 0, "split": 0}
    bad = study.aggregate([_record(**{"P0_product.split": 0.71})])
    assert bad["reproduction_mismatches"] == {"same": 0, "split": 1}


def test_accuracy_sheet_refuses_a_verdict_for_more_than_six_values(tmp_path: Path) -> None:
    """값이 6개를 넘는 정책(P3)은 평균이 높아도 연차 목표 칸이 비어야 한다 — 평가 조건 밖이다.

    쌍 12개(10회 이상). P2 split 0.95 → 3차년도 달성, P3 split 0.99지만 값 9개 → 판정 없음.
    깨뜨리는 법: `eligible`에서 6개 조건을 빼면 P3 칸이 True가 되어 실패한다.
    """
    records = [_record(**{"P2_greedy6.split": 0.95, "P3_all_values.split": 0.99,
                          "P3_all_values.values": 9, "P3_all_values.values_split": 9}) for _ in range(12)]
    report = study.aggregate(records)
    study.write_csvs(records, report, tmp_path)
    with (tmp_path / "정확도_평가방법.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = {(r["교정 정책"], r["채점 방식"]): r for r in csv.DictReader(handle)}
    p2 = rows[(study.POLICY_NAMES["P2_greedy6"], "교정·채점 기준 분리(split)")]
    assert float(p2["정확도 평균(%)"]) == pytest.approx(95.0)
    assert (p2["2차년도 70% 달성"], p2["3차년도 90% 달성"]) == ("True", "True")
    p3 = rows[(study.POLICY_NAMES["P3_all_values"], "교정·채점 기준 분리(split)")]
    assert (p3["6개 이하"], p3["3차년도 90% 달성"]) == ("False", "")
    assert json.loads(json.dumps(report))["policies"]["P3_all_values"]["values_max"] == 9


@pytest.mark.parametrize("name", [n for n in isolation.CASE_NAMES if not n.startswith("synthetic")][:1])
def test_split_side_selects_with_the_a_half_not_the_full_reference(name: str) -> None:
    """split은 교정을 **A 산포·A 이탈**로 골라야 한다 — 전체 산포로 고르면 채점(B)과 겹치는 정보로 고른다.

    제품 정책(P0)으로는 이 차이가 거의 드러나지 않는다(이동량은 산포와 무관하고 σ 문턱만 산포를 본다).
    탐욕 정책은 산포로 점수를 재며 고르므로 차이가 크다. 그래서 정책이 받은 인자를 직접 본다.
    깨뜨리는 법: 연결부가 split 쪽에 `spreads_used`/`before.deltas`를 넘기면 두 호출이 같아 실패한다.
    """
    calls: list[tuple[Any, Any]] = []

    def spy(query: Any, reference: Any, spreads: Any, weights: Any, frame: Any, deltas: Any) -> list[Any]:
        calls.append((spreads, deltas))
        return []

    case = next(c for c in isolation._cases() if c["name"] == name)
    source, kwargs = isolation._source_for(case)
    body = service.compare(
        source, PassRef(**case["query"]), PassRef(**case["reference"]), include_view=False,
        correction_policies={"spy": spy}, **kwargs,
    )
    assert body["measured"]["split_reason"] is None
    assert len(calls) == 2, "same 한 번, split 한 번이어야 한다"
    (spreads_f, deltas_f), (spreads_a, deltas_a) = calls
    assert spreads_a != spreads_f
    assert [d.axis_sigmas for d in deltas_a] != [d.axis_sigmas for d in deltas_f]
