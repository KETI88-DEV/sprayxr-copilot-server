"""3차년도 교정 정책 — 주요 관절 값 6개 이하를 조정한다.

과제 평가 방법(사용자 제공 2026-10-01): 작업자의 기존 자세와 목표 자세를 입력으로, **6개 이하의 주요 관절
값을 조정**하는 교정 알고리즘을 적용한 뒤 유사도로 개선율을 낸다. 정확도(%) = (S_corrected − S0)/(1 − S0) × 100
의 평균, 10회 이상. 3차년도 목표 90% 이상.

"관절 값 하나" = 한 점(오른손·머리·왼손)의 한 축(벽·위아래·진행) 이동량이다. 2차년도 제품 알고리즘
(`measured_score.issued_axes`)은 오른손 축 최대 3개와 머리 1개, 즉 4개까지 쓰고 왼손은 옮기지 않는다.

후보 값은 제품과 같은 문턱을 지킨다: 그 축의 σ가 `CORRECTION_SIGMA_THRESHOLD`를 넘고, 크기가
`MIN_CORRECTION_CM` 이상이며, 그 축 방향을 정할 수 있어야 한다(`measured_score._correction_candidates`).

정책 시그니처는 `measured_score.CorrectionPolicy`다.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Final, Mapping, Sequence

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

measured_score = importlib.import_module("measured_score")

MAX_VALUES: Final = 6  # 평가 방법의 "6개 이하"


def candidate_values(deltas: Sequence[object], frame: object) -> list[object]:
    """세 점 × 세 축 중 문턱을 넘는 값 전부. σ 내림차순(같으면 점·축 순서로 고정)."""
    out = []
    for delta in deltas:
        axis_sigmas = delta.axis_sigmas  # type: ignore[attr-defined]
        for axis, magnitude_cm, direction, unit in measured_score._correction_candidates(
            delta.local_delta, frame  # type: ignore[attr-defined]
        ):
            sigma = float(axis_sigmas.get(axis, 0.0))
            if sigma <= measured_score.CORRECTION_SIGMA_THRESHOLD:
                continue
            out.append(measured_score.AxisCorrection(
                point=delta.point, axis=axis, magnitude_cm=magnitude_cm,  # type: ignore[attr-defined]
                direction=direction, sigma=sigma, unit=unit,
            ))
    order = {name: i for i, name in enumerate(measured_score.POINT_ORDER)}
    axis_order = {name: i for i, name in enumerate(measured_score.AXIS_ORDER)}
    return sorted(out, key=lambda c: (-c.sigma, order[c.point], axis_order[c.axis]))


def product_policy(query, reference, spreads, weights, frame, deltas):  # noqa: ANN001, ANN201
    """P0 — 2차년도 제품 알고리즘 그대로(값 ≤ 4). `improvement_all`과 같아야 한다(재현 확인)."""
    return measured_score.issued_axes(deltas, frame)


def sigma_top_policy(query, reference, spreads, weights, frame, deltas):  # noqa: ANN001, ANN201
    """P1 — 후보를 σ 순으로 6개."""
    return candidate_values(deltas, frame)[:MAX_VALUES]


def greedy_policy(query, reference, spreads, weights, frame, deltas):  # noqa: ANN001, ANN201
    """P2(주 판정) — 목표와의 유사도를 가장 많이 올리는 값을 하나씩 더한다. 6개 또는 더 오르지 않으면 멈춘다.

    고를 때 보는 점수는 **선택 쪽 산포**(same이면 전체, split이면 A)로 잰다 — 채점 쪽(B)을 보지 않는다.
    동점이면 σ 순서(후보 목록 순서)가 앞선 것을 고른다.
    """
    pool = candidate_values(deltas, frame)
    chosen: list[object] = []
    current = measured_score.score_track(query, reference, spreads, weights, frame).score
    while pool and len(chosen) < MAX_VALUES:
        best_index, best_score = -1, current
        for index, candidate in enumerate(pool):
            score = measured_score.score_track(
                measured_score.apply_corrections(query, [*chosen, candidate], frame),
                reference, spreads, weights, frame,
            ).score
            if score > best_score:
                best_index, best_score = index, score
        if best_index < 0:
            break
        chosen.append(pool.pop(best_index))
        current = best_score
    return chosen


def all_values_policy(query, reference, spreads, weights, frame, deltas):  # noqa: ANN001, ANN201
    """P3(참고, 6개 제한 밖) — 문턱을 넘은 값 전부(≤ 9). 평가 방법의 조건을 넘으므로 판정에 쓰지 않는다."""
    return candidate_values(deltas, frame)


POLICIES: Final[Mapping[str, object]] = {
    "P0_product": product_policy,
    "P1_sigma_top6": sigma_top_policy,
    "P2_greedy6": greedy_policy,
    "P3_all_values": all_values_policy,
}
PRIMARY: Final = "P2_greedy6"
