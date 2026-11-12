from __future__ import annotations

from statistics import mean, stdev
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from mcp_servers.common import MCPException, MCPResponse, success_response
from trainer.demo_api import coating_assessment


router = APIRouter(prefix="/anomaly_detection", tags=["anomaly_detection"])


class DftPoint(BaseModel):
    value_um: float
    zone_id: str = ""
    position: str = ""


class TrajectoryPoint(BaseModel):
    x: float
    y: float
    z: float


class TimeseriesData(BaseModel):
    timestamps: list[float] = Field(default_factory=list)
    standoff_distance_mm: list[float] = Field(default_factory=list)
    gun_speed_m_per_s: list[float] = Field(default_factory=list)
    spray_angle_deg: list[float] = Field(default_factory=list)
    trajectory: list[TrajectoryPoint] = Field(default_factory=list)


class Spec(BaseModel):
    nominal_um: float | None = None
    tolerance_rule: str | None = None  # 학습된 호출 모양을 받기만 한다. 판정에는 쓰지 않는다.


class AnomalyRequest(BaseModel):
    dft_measurements: list[DftPoint] = Field(min_length=1)
    timeseries: Optional[TimeseriesData] = None
    zone_id: str = ""
    spec: Optional[Spec] = None
    image_context: str = ""


def _list_anomalies(measurements: list[DftPoint], assessed: dict) -> list[dict]:
    # Zone 등급이 정상이면 낱낱 점이 목표 아래여도 이상으로 싣지 않는다(PR #8은 평균으로 판정한다).
    if assessed["grade"] not in ("below", "above"):
        return []
    nominal = assessed["target_thickness_um"]
    return [
        {"zone_id": m.zone_id, "position": m.position, "value_um": m.value_um,
         "type": coating_assessment.GRADE_WORDS[grade],
         "deviation_pct": round((m.value_um - nominal) / nominal * 100, 1)}
        for m, grade in zip(measurements, assessed["point_grades"])
        if grade in ("below", "above")
    ]


def _analyze_distribution(anomalies: list[dict]) -> str:
    if not anomalies:
        return "이상 없음"
    position_counts: dict[str, int] = {}
    for a in anomalies:
        pos = a.get("position", "unknown")
        position_counts[pos] = position_counts.get(pos, 0) + 1
    dominant = max(position_counts, key=lambda k: position_counts[k])
    pos_label = {"upper": "상부", "middle": "중간부", "lower": "하부", "left": "좌측", "right": "우측", "center": "중앙부"}.get(dominant, dominant)
    # 점마다 등급 낱말이 같다(Zone 등급이 below면 저도막만, above면 과도막만 나올 수 있다).
    return f"{pos_label} 영역에 {anomalies[0]['type']} 측정점 집중 ({position_counts[dominant]}건)"


_FACTOR = {"speed_mm_s": "gun_speed", "distance_mm": "standoff_distance"}


def _root_causes(adjustments: list[dict]) -> list[dict]:
    # 각도 판정은 PR #8에 없어 원인 후보에서 뺀다. 근거 없는 신뢰도 수도 싣지 않는다.
    return [
        {"cause": coating_assessment.ADJUSTMENT_WORDS[(a["variable"], a["direction"])],
         "factor": _FACTOR[a["variable"]], "evidence_kind": a["evidence_kind"]}
        for a in sorted(adjustments, key=lambda a: a["priority"])
    ]


@router.post("/detect", response_model=MCPResponse)
def detect_anomaly(payload: AnomalyRequest) -> MCPResponse:
    if not payload.dft_measurements:
        raise MCPException(code="INVALID_PARAMS", message="dft_measurements must not be empty", status_code=400)

    values = [m.value_um for m in payload.dft_measurements]
    nominal = payload.spec.nominal_um if payload.spec else None
    if nominal is not None and nominal <= 0:
        raise MCPException(code="INVALID_PARAMS", message="nominal_um must be greater than zero", status_code=400)
    ts = payload.timeseries
    assessed = coating_assessment.assess_readings(
        values, nominal,
        speed_mm_s=[v * 1000.0 for v in ts.gun_speed_m_per_s] if ts else (),
        distance_mm=ts.standoff_distance_mm if ts else ())
    grade = assessed["grade"]

    anomalies = _list_anomalies(payload.dft_measurements, assessed)
    root_causes = _root_causes(assessed["adjustments"])

    result = {
        "anomaly_detected": grade in ("below", "above"),
        "anomaly_count": len(anomalies),
        "anomalies": anomalies,
        "distribution_pattern": _analyze_distribution(anomalies),
        "root_causes": root_causes,
        "anomaly_type": coating_assessment.GRADE_WORDS[grade] if grade in ("below", "above") else "",
        "zone": payload.zone_id,
        "cause": coating_assessment.adjustment_sentence(assessed["adjustments"]) or "",
        "grade": grade,
        "status": assessed["status"],
        "target_thickness_um": assessed["target_thickness_um"],
        "target_basis": assessed["target_basis"],
        "normal_band_um": assessed["normal_band_um"],
        "six_band_counts": assessed["six_band_counts"],
        "criterion_version": coating_assessment.ASSESSMENT_VERSION,
        "stats": {
            "mean_um": round(mean(values), 1),
            "stdev_um": round(stdev(values), 1) if len(values) > 1 else 0,
            "min_um": min(values),
            "max_um": max(values),
            "nominal_um": assessed["target_thickness_um"],
        },
        "zone_id": payload.zone_id,
        "image_context": payload.image_context,
    }
    return success_response(result=result)
