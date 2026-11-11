"""Conservative, revision-bound behavior evidence and coaching candidates."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np


COACHING_RULE_VERSION = "sprayxr-zone-coaching-rules/1"


def _stable_id(prefix: str, namespace: str, suffix: str) -> str:
    digest = hashlib.sha256(f"{namespace}\0{suffix}".encode("utf-8")).hexdigest()
    return f"{prefix}-{digest[:24]}"


def summarize_samples(values: list[float]) -> dict[str, float | int] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
        "median": float(np.median(array)),
    }


def build_coaching(
    *,
    namespace: str,
    quality: dict[str, Any],
    behavior_values: dict[str, list[float]],
    behavior_source_ids: dict[str, list[str]],
    observation_event_id: str,
    contract_version: str,
    revision: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Return structured candidates without asserting causality or an optimum."""
    evidence: list[dict[str, Any]] = []
    grade = quality["grade"]
    grade_labels = {
        "below": "목표 미만",
        "normal": "정상",
        "above": "목표 초과",
        None: "산정 보류",
    }
    quality_id = _stable_id("ev-quality", namespace, "grade")
    evidence.append(
        {
            "evidence_id": quality_id,
            "evidence_type": "quality_measurement",
            "statement": (
                "정확 컬러 마스크와 보정 계약으로 계산한 Zone 품질 등급은 "
                f"{grade_labels.get(grade, '산정 보류')}입니다. "
                "이 측정만으로 행동 원인을 확정하지 않습니다."
            ),
            "value": grade,
            "unit": None,
            "source_event_ids": [observation_event_id],
            "contract_version": contract_version,
            "revision": revision,
        }
    )
    if quality["painted_mean_um"] is not None:
        mean_id = _stable_id("ev-quality", namespace, "painted-mean")
        evidence.append(
            {
                "evidence_id": mean_id,
                "evidence_type": "quality_measurement",
                "statement": (
                    "명시된 관측 범위에서 유효 도포 픽셀만 집계한 "
                    f"보정 도막 평균은 {quality['painted_mean_um']:g} µm입니다."
                ),
                "value": quality["painted_mean_um"],
                "unit": "um",
                "source_event_ids": [observation_event_id],
                "contract_version": contract_version,
                "revision": revision,
            }
        )

    stats: dict[str, Any] = {}
    variable_units = {"speed_mm_s": "mm/s", "distance_mm": "mm"}
    variable_labels = {"speed_mm_s": "건 이동 속도", "distance_mm": "도장 거리"}
    variable_objects = {"speed_mm_s": "건 이동 속도를", "distance_mm": "도장 거리를"}
    behavior_evidence_ids: dict[str, str] = {}
    for variable in ("speed_mm_s", "distance_mm"):
        summary = summarize_samples(behavior_values.get(variable, []))
        stats[variable] = summary
        if summary is None:
            continue
        evidence_id = _stable_id("ev-behavior", namespace, variable)
        behavior_evidence_ids[variable] = evidence_id
        evidence.append(
            {
                "evidence_id": evidence_id,
                "evidence_type": "behavior_observation",
                "statement": (
                    f"이 Zone의 Height 촬영 시각 이전 분사 중 프레임 "
                    f"{summary['count']}개에서 계산한 {variable_labels[variable]} 평균은 "
                    f"{summary['mean']:g} {variable_units[variable]}입니다."
                ),
                "value": summary["mean"],
                "unit": variable_units[variable],
                "source_event_ids": sorted(set(behavior_source_ids.get(variable, []))),
                "contract_version": contract_version,
                "revision": revision,
            }
        )

    causes: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    if quality["analysis_status"] != "available":
        reason_code = quality["reason_codes"][0] if quality["reason_codes"] else "quality_withheld"
        coaching_status = "withheld"
    elif grade == "normal":
        reason_code = "normal_grade_no_corrective_candidate"
        coaching_status = "available"
    elif grade not in {"below", "above"}:
        reason_code = "quality_grade_unavailable"
        coaching_status = "withheld"
    elif not behavior_evidence_ids:
        reason_code = "behavior_evidence_missing"
        coaching_status = "withheld"
    else:
        reason_code = None
        coaching_status = "available"
        directions = {
            "below": {"speed_mm_s": "decrease", "distance_mm": "decrease"},
            "above": {"speed_mm_s": "increase", "distance_mm": "increase"},
        }
        for priority, variable in enumerate(("speed_mm_s", "distance_mm"), start=1):
            evidence_id = behavior_evidence_ids.get(variable)
            if evidence_id is None:
                continue
            cause_id = _stable_id("cause", namespace, variable)
            action_id = _stable_id("action", namespace, variable)
            direction = directions[grade][variable]
            causes.append(
                {
                    "cause_id": cause_id,
                    "statement": (
                        f"{variable_labels[variable]}는 품질 규칙과 함께 검토할 수 있는 "
                        "조절 후보입니다. 관측 시퀀스만으로 원인이라고 확정하지 않습니다."
                    ),
                    "variable": variable,
                    "evidence_ids": [quality_id, evidence_id],
                    "evidence_kind": "quality_rule_candidate",
                    "causal_status": "candidate_not_established",
                }
            )
            actions.append(
                {
                    "action_id": action_id,
                    "statement": (
                        f"{variable_objects[variable]} "
                        f"{'낮추는' if direction == 'decrease' else '높이는'} 방향의 "
                        "통제된 조정을 검토하세요. 기대 효과나 최적값은 확인되지 않았습니다."
                    ),
                    "variable": variable,
                    "direction": direction,
                    "priority": priority,
                    "evidence_ids": [quality_id, evidence_id],
                    "evidence_kind": "quality_rule_candidate",
                    "target_range": None,
                    "expected_effect": None,
                }
            )

    coaching = {
        "analysis_status": coaching_status,
        "reason_code": reason_code,
        "causes": causes,
        "actions": actions,
        "primary_action_id": actions[0]["action_id"] if actions else None,
    }
    return coaching, evidence, stats
