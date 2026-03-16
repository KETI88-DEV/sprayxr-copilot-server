from __future__ import annotations

from statistics import mean, stdev
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from mcp_servers.common import MCPException, MCPResponse, success_response


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
    nominal_um: float = 100.0
    tolerance_rule: str = "90/10"


class AnomalyRequest(BaseModel):
    dft_measurements: list[DftPoint] = Field(min_length=1)
    timeseries: Optional[TimeseriesData] = None
    zone_id: str = ""
    spec: Optional[Spec] = None
    image_context: str = ""


def _detect_anomalies(measurements: list[DftPoint], nominal: float) -> list[dict]:
    under_limit = nominal * 0.9
    over_limit = nominal * 1.2
    anomalies = []
    for m in measurements:
        if m.value_um < under_limit:
            anomalies.append({"zone_id": m.zone_id, "position": m.position, "value_um": m.value_um, "type": "저도막", "deviation_pct": round((m.value_um - nominal) / nominal * 100, 1)})
        elif m.value_um > over_limit:
            anomalies.append({"zone_id": m.zone_id, "position": m.position, "value_um": m.value_um, "type": "과도막", "deviation_pct": round((m.value_um - nominal) / nominal * 100, 1)})
    return anomalies


def _analyze_distribution(anomalies: list[dict]) -> str:
    if not anomalies:
        return "이상 없음"
    position_counts: dict[str, int] = {}
    for a in anomalies:
        pos = a.get("position", "unknown")
        position_counts[pos] = position_counts.get(pos, 0) + 1
    dominant = max(position_counts, key=lambda k: position_counts[k]) if position_counts else "unknown"
    pos_label = {"upper": "상부", "middle": "중간부", "lower": "하부", "left": "좌측", "right": "우측", "center": "중앙부"}.get(dominant, dominant)
    dominant_type = anomalies[0]["type"]
    return f"{pos_label} 영역에 {dominant_type} 집중 ({position_counts[dominant]}건)"


def _infer_root_causes(timeseries: Optional[TimeseriesData], anomalies: list[dict]) -> list[dict]:
    causes = []
    if not timeseries or not anomalies:
        return causes

    if timeseries.gun_speed_m_per_s and len(timeseries.gun_speed_m_per_s) >= 2:
        speeds = timeseries.gun_speed_m_per_s
        speed_mean = mean(speeds)
        first_half = mean(speeds[:len(speeds)//2])
        second_half = mean(speeds[len(speeds)//2:])
        if second_half < first_half * 0.85:
            causes.append({"cause": f"후반부 이동속도 감소 ({first_half:.2f} → {second_half:.2f} m/s)", "factor": "gun_speed", "confidence": 0.85})
        elif second_half > first_half * 1.15:
            causes.append({"cause": f"후반부 이동속도 증가 ({first_half:.2f} → {second_half:.2f} m/s)", "factor": "gun_speed", "confidence": 0.80})

    if timeseries.standoff_distance_mm and len(timeseries.standoff_distance_mm) >= 2:
        dists = timeseries.standoff_distance_mm
        dist_std = stdev(dists) if len(dists) > 1 else 0
        if dist_std > mean(dists) * 0.15:
            causes.append({"cause": f"이격거리 편차 과다 (표준편차 {dist_std:.0f}mm)", "factor": "standoff_distance", "confidence": 0.75})

    if timeseries.spray_angle_deg and len(timeseries.spray_angle_deg) >= 2:
        angles = timeseries.spray_angle_deg
        low_angles = [a for a in angles if a < 60]
        if len(low_angles) > len(angles) * 0.3:
            causes.append({"cause": f"분사각도 부족 구간 {len(low_angles)}건 ({min(angles):.0f}°~)", "factor": "spray_angle", "confidence": 0.70})

    if not causes and anomalies:
        causes.append({"cause": "시계열 데이터 부족으로 원인 특정 불가, 추가 센서 데이터 필요", "factor": "unknown", "confidence": 0.3})

    return causes


@router.post("/detect", response_model=MCPResponse)
def detect_anomaly(payload: AnomalyRequest) -> MCPResponse:
    if not payload.dft_measurements:
        raise MCPException(code="INVALID_PARAMS", message="dft_measurements must not be empty", status_code=400)

    nominal = payload.spec.nominal_um if payload.spec else mean([m.value_um for m in payload.dft_measurements])
    values = [m.value_um for m in payload.dft_measurements]

    anomalies = _detect_anomalies(payload.dft_measurements, nominal)
    distribution_pattern = _analyze_distribution(anomalies)
    root_causes = _infer_root_causes(payload.timeseries, anomalies)

    result = {
        "anomaly_detected": len(anomalies) > 0,
        "anomaly_count": len(anomalies),
        "anomalies": anomalies,
        "distribution_pattern": distribution_pattern,
        "root_causes": root_causes,
        "stats": {
            "mean_um": round(mean(values), 1),
            "stdev_um": round(stdev(values), 1) if len(values) > 1 else 0,
            "min_um": min(values),
            "max_um": max(values),
            "nominal_um": nominal,
        },
        "zone_id": payload.zone_id,
        "image_context": payload.image_context,
    }
    return success_response(result=result)
