"""`/motion-viewer`용 자세 평가 읽기 모델.

점수는 MCP 제품 엔진 결과를 그대로 읽고, 이 모듈은 표시용 레이어와 설명만
구성한다. 좌표계가 다른 레이어를 강제로 합치지 않도록 각 레이어에 좌표계를
명시한다.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import statistics
from typing import Any

from trainer.demo_api.pose_query import PoseEngineResult

MeshProvider = Callable[[str], dict[str, Any] | None]

_SEVERITY = {"critical": 4, "high": 3, "medium": 2, "low": 1}
_REASON_MESSAGES = {
    "missing_sequence_body": "저장된 시퀀스 본문을 찾지 못했습니다.",
    "quality_insufficient": "평가에 필요한 골격 프레임이 충분하지 않습니다.",
    "insufficient_frames": "평가에 필요한 골격 프레임이 충분하지 않습니다.",
    "reference_data_missing": "전문가 자세 기준 데이터가 연결되지 않았습니다.",
    "engine_unavailable": "자세 분석 엔진에 연결할 수 없습니다.",
}


def _default_mesh_provider(mode_id: str) -> dict[str, Any] | None:
    from trainer.pose_lab import service  # 큰 FBX 의존성은 실제 호출 시에만 읽는다.

    return service.mesh_block(mode_id)


def _mode_id(body: Mapping[str, Any]) -> int | None:
    explicit = body.get("mode_id")
    if isinstance(explicit, int) and 1 <= explicit <= 10:
        return explicit
    scenario = str(body.get("scenario_id") or "")
    return int(scenario) if scenario.isdigit() and 1 <= int(scenario) <= 10 else None


def _correction_indices(result: Mapping[str, Any]) -> set[int]:
    found: set[int] = set()
    for correction in result.get("corrections") or []:
        if not isinstance(correction, Mapping):
            continue
        target = correction.get("target")
        if not isinstance(target, Mapping):
            continue
        for entry in target.get("per_frame") or []:
            if isinstance(entry, Mapping) and isinstance(entry.get("index"), int):
                found.add(int(entry["index"]))
    return found


def _sample_positions(length: int, limit: int, required: set[int]) -> list[int]:
    if length <= 0:
        return []
    limit = max(2, int(limit))
    required_positions = {0, length - 1}
    required_positions.update(index for index in required if 0 <= index < length)
    if len(required_positions) >= limit:
        middle = sorted(required_positions - {0, length - 1})[: max(0, limit - 2)]
        return sorted({0, length - 1, *middle})
    if length <= limit:
        return list(range(length))
    budget = limit - len(required_positions)
    candidates = [index for index in range(length) if index not in required_positions]
    if budget > 0 and candidates:
        for slot in range(budget):
            position = round(slot * (len(candidates) - 1) / max(1, budget - 1))
            required_positions.add(candidates[position])
    return sorted(required_positions)[:limit]


def _frames(body: Mapping[str, Any], result: Mapping[str, Any], limit: int) -> list[dict[str, Any]]:
    source = body.get("sequence_info")
    if not isinstance(source, Sequence) or isinstance(source, (str, bytes)):
        return []
    raw_frames = [frame for frame in source if isinstance(frame, Mapping)]
    positions = _sample_positions(len(raw_frames), limit, _correction_indices(result))
    output: list[dict[str, Any]] = []
    for position in positions:
        frame = raw_frames[position]
        index = frame.get("index") if isinstance(frame.get("index"), int) else position
        joints = frame.get("skeleton_joints")
        tool = frame.get("tool_pos_rot")
        output.append(
            {
                "index": int(index),
                "time": frame.get("time"),
                "joints": dict(joints) if isinstance(joints, Mapping) else {},
                "tool_position": (
                    list(tool.get("pos"))
                    if isinstance(tool, Mapping) and isinstance(tool.get("pos"), Sequence)
                    else None
                ),
            }
        )
    return output


def _top_correction(result: Mapping[str, Any]) -> dict[str, Any] | None:
    corrections = [item for item in result.get("corrections") or [] if isinstance(item, Mapping)]
    if not corrections:
        return None

    def key(item: Mapping[str, Any]) -> tuple[int, float]:
        evidence = item.get("evidence")
        sigma = evidence.get("sigma") if isinstance(evidence, Mapping) else 0.0
        return (_SEVERITY.get(str(item.get("severity") or "").lower(), 0), float(sigma or 0.0))

    chosen = max(corrections, key=key)
    return {
        "joint": str(chosen.get("joint") or ""),
        "message": str(chosen.get("correction_text") or chosen.get("label") or ""),
        "severity": chosen.get("severity"),
        "direction": chosen.get("direction"),
    }


def _confidence(result: Mapping[str, Any]) -> str:
    quality = str(result.get("reference_quality") or "")
    support = int(result.get("support") or 0)
    input_quality = result.get("input_quality")
    input_status = input_quality.get("status") if isinstance(input_quality, Mapping) else "accept"
    if quality in {"multi_worker", "validated"} and support >= 100 and input_status == "accept":
        return "높음"
    if quality and quality != "unavailable" and input_status != "reject":
        return "보통"
    return "낮음"


def _verdict(result: Mapping[str, Any]) -> str:
    percentile = result.get("expert_percentile")
    score = result.get("similarity_score")
    value = float(percentile) / 100.0 if isinstance(percentile, (int, float)) else score
    if not isinstance(value, (int, float)):
        return "판정 불가"
    if value >= 0.70:
        return "양호"
    if value >= 0.40:
        return "개선 필요"
    return "집중 교정 필요"


def _strokes(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    burst = result.get("burst")
    if not isinstance(burst, Mapping):
        return []
    source = burst.get("strokes") or burst.get("per_stroke") or []
    strokes = [item for item in source if isinstance(item, Mapping)]
    total = len(strokes)
    return [
        {"index": index, "label": f"스트로크 {index + 1}/{total}", "secondary": True, **dict(item)}
        for index, item in enumerate(strokes)
    ]


def _corrected_layer(result: Mapping[str, Any]) -> dict[str, Any]:
    targets: list[dict[str, Any]] = []
    for correction in result.get("corrections") or []:
        if not isinstance(correction, Mapping):
            continue
        target = correction.get("target")
        if not isinstance(target, Mapping):
            continue
        for point in target.get("per_frame") or []:
            if isinstance(point, Mapping):
                targets.append({"joint": correction.get("joint"), **dict(point)})
    return {
        "available": bool(targets),
        "coordinate_space": "world",
        "color": "#FACC15",
        "targets": targets,
    }


def _expert_layer(result: Mapping[str, Any]) -> dict[str, Any]:
    layer = result.get("reference_layer")
    if not isinstance(layer, Mapping) or not layer:
        return {
            "available": False,
            "coordinate_space": "body_normalized",
            "color": "#22C55E",
            "tracks": {},
            "message": "전문가 기준 골격 좌표는 현재 엔진 응답에 포함되지 않았습니다.",
        }
    return {
        "available": True,
        "coordinate_space": "body_normalized",
        "color": "#22C55E",
        "tracks": dict(layer),
        "message": "몸 기준 정규 좌표이므로 작업자 월드 좌표와 강제로 겹치지 않습니다.",
    }


def _mesh(mode: int | None, provider: MeshProvider) -> dict[str, Any]:
    supplied = provider(str(mode)) if mode is not None else None
    if not isinstance(supplied, Mapping) or not supplied.get("available"):
        reason = supplied.get("reason") if isinstance(supplied, Mapping) else "no_alignment_record"
        return {
            "available": False,
            "vertices": [],
            "triangles": [],
            "reason": reason,
            "fallback": "ray_point_surface",
        }
    vertices = supplied.get("vertices") if isinstance(supplied.get("vertices"), list) else []
    triangles = supplied.get("triangles") if isinstance(supplied.get("triangles"), list) else []
    return {
        **dict(supplied),
        "available": bool(vertices and triangles),
        "vertices": vertices,
        "triangles": triangles,
        "fallback": None if vertices and triangles else "ray_point_surface",
    }


def aggregate_zone_results(results: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """이미 완료된 시퀀스 결과를 Zone별로 집계한다.

    Zone을 하나의 가상 시퀀스로 다시 채점하지 않는다. 사용할 수 없는 결과도
    분모와 개수에는 남기되 점수 중앙값에는 섞지 않는다.
    """
    grouped: dict[str, list[Mapping[str, Any]]] = {"all": list(results)}
    for item in results:
        identity = item.get("identity")
        zone_id = str(identity.get("zone_id") or "unknown") if isinstance(identity, Mapping) else "unknown"
        grouped.setdefault(zone_id, []).append(item)

    output: dict[str, dict[str, Any]] = {}
    confidence_rank = {"낮음": 0, "보통": 1, "높음": 2}
    for zone_id, items in grouped.items():
        available: list[Mapping[str, Any]] = []
        scores: list[float] = []
        for item in items:
            evaluation = item.get("evaluation")
            if not isinstance(evaluation, Mapping):
                continue
            score = evaluation.get("similarity_score")
            if evaluation.get("analysis_status") == "available" and isinstance(score, (int, float)):
                available.append(item)
                scores.append(float(score))
        ratio = len(available) / len(items) if items else 0.0
        correction_candidates: list[tuple[float, Mapping[str, Any]]] = []
        for item in available:
            evaluation = item["evaluation"]
            correction = evaluation.get("top_correction")
            score = evaluation.get("similarity_score")
            if isinstance(correction, Mapping) and isinstance(score, (int, float)):
                correction_candidates.append((float(score), correction))
        worst = min(correction_candidates, key=lambda pair: pair[0])[1] if correction_candidates else None
        if ratio < 0.5 or not available:
            confidence = "낮음"
        else:
            confidence = min(
                (str(item["evaluation"].get("confidence") or "낮음") for item in available),
                key=lambda value: confidence_rank.get(value, 0),
            )
        output[zone_id] = {
            "zone_id": zone_id,
            "sequence_count": len(items),
            "available_count": len(available),
            "unavailable_count": len(items) - len(available),
            "available_ratio": ratio,
            "median_score": statistics.median(scores) if scores else None,
            "worst_correction": dict(worst) if worst is not None else None,
            "confidence": confidence,
            "sequence_ids": [
                str(item.get("identity", {}).get("sequence_id") or "")
                for item in items
            ],
        }
    return output


def build_pose_view(
    body: dict[str, Any] | None,
    raw: PoseEngineResult | None,
    *,
    mesh_provider: MeshProvider = _default_mesh_provider,
    max_frames: int = 240,
) -> dict[str, Any]:
    """저장된 TSB 시퀀스와 제품 엔진 결과를 브라우저 읽기 모델로 변환한다."""
    content: dict[str, Any] = body or {}
    result = raw.result if raw is not None else {}
    reason = (
        "missing_sequence_body"
        if body is None
        else (raw.reason_code if raw is not None and raw.analysis_status != "available" else None)
    )
    status = "not_available" if reason else "available"
    sequence_id = str(content.get("sequence_id") or (raw.sequence_id if raw else ""))
    scenario_id = str(content.get("scenario_id") or "")
    mode = _mode_id(content)
    frames = _frames(content, result, max_frames)
    if status == "available" and not any(frame["joints"] for frame in frames):
        status, reason = "not_available", "quality_insufficient"
    score = result.get("similarity_score") if status == "available" else None
    evaluation = {
        "analysis_status": status,
        "reason_code": reason,
        "similarity_score": score,
        "expert_percentile": result.get("expert_percentile") if status == "available" else None,
        "verdict": _verdict(result) if status == "available" else "판정 불가",
        "confidence": _confidence(result) if status == "available" else "낮음",
        "top_correction": _top_correction(result) if status == "available" else None,
        "summary": result.get("summary") if status == "available" else None,
        "message": (
            str(result.get("summary") or "자세 평가가 완료되었습니다.")
            if status == "available"
            else _REASON_MESSAGES.get(str(reason), f"자세 평가를 사용할 수 없습니다: {reason}")
        ),
    }
    return {
        "identity": {
            "session_id": str(content.get("session_id") or ""),
            "sequence_id": sequence_id,
            "scenario_id": scenario_id,
            "mode_id": mode,
            "zone_id": str(content.get("zone_id") or ""),
            "unit_label": sequence_id,
        },
        "evaluation": evaluation,
        "strokes": _strokes(result),
        "layers": {
            "measured": {
                "available": bool(frames),
                "coordinate_space": "world",
                "color": "#38BDF8",
                "frames": frames,
            },
            "expert": _expert_layer(result),
            "corrected": _corrected_layer(result),
        },
        "object_mesh": _mesh(mode, mesh_provider),
    }
