"""Document serialized output contracts; never change requests or runtime responses."""

from __future__ import annotations

from copy import deepcopy
import json
from typing import Any

from trainer.demo_api.v1_contracts import (
    ReportDetail, ReportStatusData, SessionCompleteData, UserProfile,
)

_MARKER = "### 출력 파라미터"
_METHODS = {"get", "post", "put", "patch", "delete", "options", "head"}
_SSE = "/api/v1/stream/anomalies/{session_id}"
_WAV = "/api/v1/test/assets/query-voice-ko.wav"
_PAYLOADS = {
    ("post", "/api/v1/sessions/complete"): SessionCompleteData,
    ("get", "/api/v1/sessions/{session_id}/reports/{report_id}"): ReportStatusData,
    ("get", "/api/v1/reports/{report_id}"): ReportDetail,
    ("get", "/api/v1/users"): UserProfile,
    ("post", "/api/v1/users"): UserProfile,
    ("get", "/api/v1/users/{user_id}"): UserProfile,
}
_MEANINGS = {
    "status": "처리 또는 판정 상태입니다. 아래 허용값과 해당 영역의 사유를 함께 확인합니다.",
    "code": "응답 처리 코드입니다. v1 JSON 봉투에서는 HTTP 상태 코드와 함께 확인합니다.",
    "message": "처리 결과 또는 현재 영역의 짧은 안내입니다.",
    "data": "API별 결과 본문입니다. v1 ERROR 응답은 null입니다.",
    "request_id": "개별 호출의 추적 식별자입니다.",
    "session_id": "결과가 속한 훈련 세션 식별자입니다.",
    "user_id": "훈련자 식별자입니다.",
    "scenario_id": "콘텐츠 식별자입니다. 학습 1~10·실전·Unit 형식을 구분합니다.",
    "sequence_id": "세션 안의 연속 분사 구간 식별자입니다.",
    "report_id": "완료 업로드로 발급된 리포트 식별자입니다.",
    "zone_id": "해당 구역의 문자열 ID입니다. 여러 면에서는 map_num과 함께 구분합니다.",
    "map_num": "텍스처 면 번호입니다. Zone ID는 면마다 다시 매겨집니다.",
    "generated_at": "서버가 결과를 생성한 시각입니다(ISO 8601).",
    "started_at": "훈련 시작 시각입니다(ISO 8601).",
    "ended_at": "훈련 종료 시각입니다. 아직 종료되지 않았으면 null입니다.",
    "requested_at": "리포트 생성을 요청한 시각입니다(ISO 8601).",
    "latency_ms": "이번 요청의 처리 시간(ms)입니다. 응답 시간 보장값은 아닙니다.",
    "stt_latency_ms": "오디오 샘플레이트 변환과 음성 인식 처리 시간(ms)입니다.",
    "transcript": "업로드한 음성을 인식한 질문 원문입니다.",
    "answer": "사용자에게 표시할 답변입니다. 제한·보류 안내도 포함될 수 있습니다.",
    "language": "질문 또는 인식한 음성의 언어 코드입니다.",
    "response_language": "답변의 언어 코드입니다.",
    "degraded": "true이면 생성·검토 제한 또는 근거 부족이 있어 답변을 확정 분석으로 사용하지 않습니다.",
    "resolved_intents": "서버가 실제로 처리한 질문 의도 목록입니다.",
    "context_status": "현재 질문에 사용할 훈련 문맥의 이용 가능 여부입니다.",
    "retrieval_used": "이번 답변에 지식 검색을 실제 사용했는지 나타냅니다.",
    "citations": "답변에 연결된 검색 문서 조각 ID 목록입니다. 빈 목록은 인용 출처가 없음을 뜻합니다.",
    "analysis_status": "available은 해당 분석을 사용할 수 있다는 뜻이며 정상 수행 점수라는 뜻은 아닙니다.",
    "reason_code": "분석 제한·미평가 사유 코드입니다. null은 추가 사유가 없다는 뜻이며 정상 판정 보장은 아닙니다.",
    "reason": "이 항목을 제공하지 못하거나 조정한 이유입니다.",
    "available": "현재 항목의 평가·안내를 사용할 수 있는지 나타냅니다. false를 정상으로 해석하지 않습니다.",
    "guideline": "해당 분석 영역의 안내입니다. 다른 영역의 판정과 구분합니다.",
    "algorithm_version": "결과 계산에 사용한 알고리즘 버전입니다.",
    "source_artifact_id": "분석에 사용한 CSV·이미지 등 원본 자료 식별자입니다.",
    "stored": "시퀀스 본문이 새로 저장되었는지 나타냅니다.",
    "duplicate": "이미 수신한 동일 시퀀스의 재전송인지 나타냅니다.",
    "frames_received": "이번 본문에서 수신한 프레임 수입니다.",
    "frames_accepted": "이번 호출에서 새로 수용한 프레임 수입니다.",
    "sequence_data_ack": "시퀀스 저장 확인입니다. 저장 성공과 분석 가능 여부는 별개입니다.",
    "coating_feedback": "현재 도막 평가입니다. 미평가 시 상태·사유를 확인하고 수치를 추정하지 않습니다.",
    "sequence_feedback": "현재 시퀀스의 순서·방향 분석 가능 상태입니다.",
    "text_guide": "이번 분사에 대한 통합 안내입니다. 제공할 안내가 없으면 null입니다.",
    "coating_progress": "현재 Zone의 도막 진행 판정입니다. 평균 두께만으로 다음 Zone 이동을 허용하지 않습니다.",
    "next_zone_id": "현재 구역의 진행 판정을 통과한 뒤 안내하는 다음 구역입니다. 없거나 미판정이면 null입니다.",
    "recommendation": "현재 Zone 유지·이동 등 진행 권고입니다.",
    "completion_percent": "해당 판정 기준에 대한 구역 진행 비율(0~100%)입니다.",
    "coverage_percent": "공식 Zone 면적 중 정상 두께를 충족한 면적 비율(0~100%)입니다.",
    "overpaint_percent": "공식 Zone 면적 중 상한을 초과한 면적 비율(0~100%)입니다.",
    "thickness_um": "판정에 사용하는 칠한 곳 평균 두께(µm)입니다. 미평가·미도장은 null일 수 있습니다.",
    "target_thickness_um": "목표 두께 또는 정상범위 하한(µm)입니다.",
    "normal_band_um": "정상범위 [하한, 상한](µm)입니다. 기본 160~200이며 양쪽 경계를 포함합니다.",
    "target_basis": "session_target은 세션 지정 목표, default_target은 기본 목표를 사용했다는 뜻입니다.",
    "comparison_basis": "목표 두께 판정과 전문가 최종 도막 참고 비교의 기준을 구분합니다.",
    "unpainted_fraction": "구역의 미도장 면적 비율(0~1)입니다. 1은 전체 미도장입니다.",
    "six_band_counts": "6개 두께 구간의 화소 수입니다. 키별 수와 면적 비율을 혼동하지 않습니다.",
    "adjustments": "구조화된 조절 후보 목록입니다. 가변 항목이며 원인 확정이나 무조건 실행 지시가 아닙니다.",
    "session_status": "ACTIVE는 진행 중, COMPLETED는 세션 종료입니다. 분석 리포트 완료 여부와 별개입니다.",
    "report_status": "pending은 분석 결과 대기, complete는 처리 완료, failed는 처리 실패입니다. complete여도 개별 영역은 미평가일 수 있습니다.",
    "user_created": "이번 요청에서 사용자 레코드를 새로 만들었는지 나타냅니다.",
    "first_zone_id": "등록된 전문가 순서의 첫 Zone입니다. 등록된 기준이 없으면 null이며 임의 구역을 안내하지 않습니다.",
    "first_zone_guidance": "선택한 시나리오·공정에 등록된 첫 구역과 작업 순서 안내입니다.",
    "recommended_flow": "등록된 전문가 작업 순서입니다. 시나리오·공정에 따라 달라집니다.",
    "first_visit_flow": "이번 세션에서 각 구역을 처음 방문한 순서입니다.",
    "adherence_rate": "작업 순서 유사·준수 비율(0~1)입니다. 도포 방향 비율과 별도로 봅니다.",
    "direction_adherence_rate": "평가한 분사 중 권장 도포 방향을 따른 비율(0~1)입니다.",
    "average_similarity_score": "채점 가능한 시퀀스만 평균한 자세 유사도(0~1)입니다.",
    "evaluated_sequence_count": "실제로 평가한 시퀀스 수이며 미판정 시퀀스는 점수의 분모에서 제외합니다.",
    "limited_sequence_count": "평가에 포함되었지만 비교 범위·입력 품질 등의 제한이 있는 시퀀스 수입니다.",
    "confidence": "limited이면 참고용 비교입니다. 교정 항목이 없어도 정상 자세를 보장하지 않습니다.",
    "priority_corrections": "우선순위가 높은 자세 교정 목록입니다. 안내의 수치와 부위를 함께 확인합니다.",
    "correction_guidance": "해당 부위·구역의 구체적인 교정 안내입니다.",
    "worst_zones": "목표 범위에서 크게 벗어난 도장 구역입니다. 빈 목록이어도 미도장 구역이 남을 수 있습니다.",
    "expert_comparison": "전문가 최종 도막과 비교한 참고 결과입니다. 목표 기준 저도막·과도막 판정과 구분합니다.",
    "llm_briefing": "구조화된 분석을 요약한 총평입니다. 미평가 영역의 수치를 새로 만들지 않습니다.",
    "test_context": "모의·테스트 호출에서 실제 사용한 입력과 확인할 결과입니다. 실제 평가값과 구분합니다.",
    "input_used": "테스트 내부에서 사용한 입력입니다. 키는 테스트마다 달라지는 가변 object입니다.",
    "evaluation_mode": "모의 이벤트·설명용 리포트 등 현재 결과의 생성 방식을 나타냅니다.",
    "evaluation_focus": "테스트에서 확인할 사항이며 분석 점수나 사용자 입력 요구가 아닙니다.",
    "workflow_status": "모의 연동 예제의 진행 상태입니다. 실제 세션·분석 상태와 구분합니다.",
    "next_endpoint": "모의 연동에서 다음으로 호출할 API 경로입니다.",
    "telemetry_csv_bytes": "수신한 CSV 업로드의 크기(byte)입니다.",
    "height_map_image_bytes": "높이맵 업로드 크기(byte)입니다. 파일을 보내지 않으면 null입니다.",
    "name": "훈련자 표시 이름입니다. 수집하지 않은 값은 null입니다.",
    "skill_level": "등록된 숙련도 분류입니다. 수행 점수를 보장하는 값은 아닙니다.",
    "dominant_hand": "주 사용 손입니다. unknown은 과거 자료에 기록이 없다는 뜻입니다.",
}


def _resolve(node: dict[str, Any], components: dict[str, Any]) -> dict[str, Any]:
    ref = node.get("$ref")
    if isinstance(ref, str):
        return {**components.get(ref.rsplit("/", 1)[-1], {}), **{k: v for k, v in node.items() if k != "$ref"}}
    return node


def _branches(node: dict[str, Any]) -> list[dict[str, Any]]:
    return [branch for branch in node.get("anyOf", node.get("oneOf", [])) if isinstance(branch, dict)]


def _label(node: dict[str, Any], components: dict[str, Any]) -> str:
    node = _resolve(node, components)
    branches = _branches(node)
    if branches:
        return " | ".join(dict.fromkeys(_label(branch, components) for branch in branches))
    kind = node.get("type", "object" if "properties" in node else "value")
    return f"array<{_label(node.get('items', {}), components)}>" if kind == "array" else str(kind)


def _structure(node: dict[str, Any], components: dict[str, Any]) -> dict[str, Any]:
    node = _resolve(node, components)
    branches = _branches(node)
    return _structure(next((branch for branch in branches if branch.get("type") != "null"), {}), components) if branches else node


def _enum_values(node: dict[str, Any], components: dict[str, Any]) -> list[Any]:
    node = _resolve(node, components)
    values = list(node.get("enum", []))
    for branch in _branches(node):
        for value in _enum_values(branch, components):
            if value not in values:
                values.append(value)
    return values


def _meaning(path: str, name: str, node: dict[str, Any]) -> str:
    expert = ".expert_comparison." in path or name == "expert_completion_percent"
    if expert:
        measures = {
            "coverage_percent": "원본 셀 평균이 q1−0.5µm 이상인 셀의 비율",
            "overpaint_percent": "원본 셀 평균이 q3+max(q3−q1, q3×0.1, 5µm)+0.5µm를 초과한 셀의 비율",
            "completion_percent": "셀별 min(원본 셀 평균/q1, 1)의 평균×100",
            "expert_completion_percent": "셀별 min(원본 셀 평균/q1, 1)의 평균×100",
        }
        if name in measures:
            return f"유효 전문가 격자 기준 {measures[name]}(0~100%, 보정 전 두께)입니다. 목표 두께·이동 게이트와 무관한 참고값입니다."
        if name == "status":
            return "전문가 최종 도막과 비교한 참고 상태입니다. normal=전문가 기준 충족, under_app=전문가보다 얇음, over_app=전문가보다 두꺼움이며 목표 정상범위 판정과 구분합니다."
        if name in {"normal", "under_app", "over_app"}:
            label = {"normal": "전문가 기준을 충족한", "under_app": "전문가보다 얇은", "over_app": "전문가보다 두꺼운"}[name]
            return f"{label} 구역 수입니다. 목표 두께 기준의 등급 개수가 아닙니다."
    return str(node.get("description") or _MEANINGS.get(name) or f"{name} 응답 항목입니다.")


def _field_lines(node: dict[str, Any], components: dict[str, Any], path: str = "",
                 ancestors: frozenset[str] = frozenset(), depth: int = 0) -> list[str]:
    if depth > 14:
        return [f"  - `{path}` : 깊은 구조는 아래 응답 Schema에서 확인합니다."]
    ref = node.get("$ref")
    if ref and ref in ancestors:
        return [f"  - `{path}` : 같은 구조가 반복되는 재귀 항목입니다. Schema를 확인합니다."]
    visited = ancestors | {ref} if isinstance(ref, str) else ancestors
    structured = _structure(node, components)
    if structured.get("type") == "array":
        return _field_lines(structured.get("items", {}), components, f"{path}[]", visited, depth + 1)
    properties = structured.get("properties", {})
    if not properties:
        if structured.get("type") == "object" or structured.get("additionalProperties"):
            return [f"  - `{path or 'body'}.{{key}}` : 가변 object입니다. 키·하위 값은 결과 종류에 따라 달라지며 고정 항목을 보장하지 않습니다."]
        return []
    required = set(structured.get("required", []))
    lines: list[str] = []
    for name, child in properties.items():
        child_path = f"{path}.{name}" if path else name
        child_schema = _resolve(child, components)
        description = " ".join(_meaning(child_path, name, child_schema).split())
        values = _enum_values(child, components)
        if values:
            description += " 허용값: " + ", ".join(f"`{value}`" for value in values) + "."
        presence = "필수 키" if name in required else "생략 가능"
        indent = "  " if path else ""
        lines.append(f"{indent}- `{child_path}({_label(child, components)})` : {description} **[{presence}]**")
        lines.extend(_field_lines(child, components, child_path, visited, depth + 1))
    return lines


def _infer_example(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return {"type": "object", "properties": {key: _infer_example(child) for key, child in value.items()}}
    if isinstance(value, list):
        return {"type": "array", "items": _infer_example(value[0]) if value else {}}
    return {"type": "null" if value is None else "boolean" if isinstance(value, bool) else
            "integer" if isinstance(value, int) else "number" if isinstance(value, float) else "string"}


def _install_payload(path: str, method: str, media: dict[str, Any], components: dict[str, Any]) -> None:
    model = _PAYLOADS.get((method, path))
    if model is None:
        return
    generated = model.model_json_schema(mode="serialization", ref_template="#/components/schemas/Output{model}")
    for name, definition in generated.pop("$defs", {}).items():
        components.setdefault(f"Output{name}", definition)
    components.setdefault(f"Output{model.__name__}", generated)
    payload: dict[str, Any] = {"$ref": f"#/components/schemas/Output{model.__name__}"}
    if path == "/api/v1/users" and method == "get":
        payload = {"type": "array", "items": payload}
    envelope = deepcopy(_resolve(media.get("schema", {}), components))
    envelope.setdefault("properties", {})["data"] = payload
    media["schema"] = envelope


def _sample(node: dict[str, Any], components: dict[str, Any], path: str = "", depth: int = 0) -> Any:
    node = _resolve(node, components)
    if depth > 14:
        return None
    branches = _branches(node)
    if branches:
        if path != "data" and any(branch.get("type") == "null" for branch in branches):
            return None
        return _sample(next((branch for branch in branches if branch.get("type") != "null"), {}), components, path, depth + 1)
    if "const" in node:
        return node["const"]
    if "enum" in node:
        values = node["enum"]
        return next((value for value in ("SUCCESS", "not_available", "unavailable", "pending", "ACTIVE", "novice", "right", "ko") if value in values), values[0])
    kind = node.get("type")
    if kind == "object" or "properties" in node:
        return {name: _sample(child, components, f"{path}.{name}" if path else name, depth + 1)
                for name, child in node.get("properties", {}).items()}
    if kind == "array":
        return []
    if kind == "null":
        return None
    if kind == "boolean":
        return path.endswith(("degraded", "stored"))
    if kind in {"integer", "number"}:
        return 200 if path == "code" else node.get("minimum", 0)
    leaf = path.rsplit(".", 1)[-1]
    if leaf.endswith("_at"):
        return "2026-11-16T00:00:00Z"
    return {"session_id": "3_11000346_1", "user_id": "1234", "scenario_id": "3", "sequence_id": "SEQ-001",
            "zone_id": "3", "report_id": "RPT-example", "request_id": "REQ-EXAMPLE", "message": "형식 확인용 예시"}.get(leaf, "예시")


def _response_example(path: str, node: dict[str, Any], components: dict[str, Any]) -> Any:
    example = _sample(node, components)
    if not isinstance(example, dict):
        return example
    if "message" in example:
        example["message"] = "형식 확인용 응답 예시"
    data = example.get("data")
    if not isinstance(data, dict):
        return example
    if path.endswith(("/sessions/start", "/sessions/end")):
        ended = path.endswith("/end")
        data.update(session_status="COMPLETED" if ended else "ACTIVE", started_at="2026-11-16T00:00:00Z",
                    ended_at="2026-11-16T00:01:00Z" if ended else None,
                    target_thickness_um=160.0, target_thickness_min_um=160.0, target_thickness_max_um=200.0)
        if not ended:
            data["first_zone_guidance"] = {"analysis_status": "not_available", "reason_code": "sequence_reference_unavailable",
                "process": "spray", "zone_id": None, "recommended_flow": [], "reference_basis": None,
                "message": "등록된 작업 순서 기준이 없습니다."}
    if "query" in path:
        data.update(answer="분석 자료가 없어 현재 상태를 판단할 수 없습니다.", degraded=True,
                    retrieval_used=False, citations=[], resolved_intents=["knowledge_search"])
        if "transcript" in data:
            data["transcript"] = "지금 자세는 어떤가요?"
    for name in ("coating_feedback", "sequence_feedback"):
        feedback = data.get(name)
        if isinstance(feedback, dict):
            feedback["reason_code"] = "sequence_reference_unavailable" if name == "sequence_feedback" else (
                "coating_engine_pending_integration" if path.endswith("/telemetry/sequence") else "coating_reference_unavailable")
    for item in data.get("guide", {}).values():
        if isinstance(item, dict):
            item["reason"] = "reference_unavailable"
    if isinstance(data.get("coating_progress"), dict):
        data["coating_progress"]["reason"] = "coating_reference_unavailable"
    return example


def _stream_documentation() -> str:
    return "\n".join([
        _MARKER, "", "**HTTP 200 · `text/event-stream`**. JSON 성공 봉투가 아닌 지속 연결입니다.", "",
        "- `id(integer)` : 재연결 시 Last-Event-ID에 보내는 이벤트 번호입니다.",
        "- `event(string)` : 이상 알림의 이벤트 이름은 `anomaly_detection`입니다.",
        "- `data(object)` : 한 이벤트의 JSON 데이터입니다.",
        "  - `data.session_id(string)` : 훈련 세션 식별자입니다.",
        "  - `data.sequence_id(string)` : 이상이 발생한 시퀀스 식별자입니다.",
        "  - `data.anomaly_type(string)` : 이상 종류입니다. 추가 payload 키는 이벤트별 가변 항목입니다.",
        "- `: keep-alive` : 연결 유지 주석이며 이상 발생이나 새 분석 결과가 아닙니다. 이상이 없으면 알림 이벤트가 없을 수 있습니다.",
        "", "### 응답 예시", "", "이상 이벤트의 전송 형식 예시이며 실제 발생을 뜻하지 않습니다.", "", "```text",
        'id: 1\nevent: anomaly_detection\ndata: {"session_id":"3_11000346_1","sequence_id":"SEQ-001","anomaly_type":"direction_deviation","message":"도포 방향을 확인하세요."}\n\n: keep-alive', "```",
    ])


def add_openapi_output_overviews(schema: dict[str, Any]) -> None:
    """Enrich cached Swagger metadata once; payloads and route handlers are untouched."""
    components = schema.setdefault("components", {}).setdefault("schemas", {})
    for path, operations in schema.get("paths", {}).items():
        for method, operation in operations.items():
            if method not in _METHODS or not isinstance(operation, dict) or _MARKER in operation.get("description", ""):
                continue
            responses = operation.setdefault("responses", {})
            success = next((value for code, value in responses.items() if str(code).startswith("2")), None)
            if not isinstance(success, dict):
                continue
            if path == _SSE:
                success["content"] = {"text/event-stream": {"schema": {"type": "string"}}}
                documentation = _stream_documentation()
            elif path == _WAV:
                success["content"] = {"audio/wav": {"schema": {"type": "string", "format": "binary"}}}
                documentation = f"{_MARKER}\n\n- `body(file)` : HTTP 200 · `audio/wav` 바이너리입니다. JSON 봉투나 분석 결과가 아닙니다.\n- `Content-Disposition(header)` : 다운로드 파일명 `query-voice-ko.wav`입니다.\n\n### 응답 예시\n\nWAV PCM16 · 16 kHz · mono 음성 파일이 내려옵니다. 바이너리이므로 JSON 예시는 제공하지 않습니다."
            else:
                media = success.setdefault("content", {}).setdefault("application/json", {})
                _install_payload(path, method, media, components)
                response_schema = media.get("schema", {})
                example = media.get("example")
                if example is None:
                    example = _response_example(path, response_schema, components)
                    media["example"] = example
                shown_schema = _infer_example(example) if "/test/" in path and isinstance(example, dict) else response_schema
                fields = _field_lines(shown_schema, components)
                if not fields:
                    fields = ["- `body(value)` : 고정 응답 필드가 선언되지 않은 가변 결과입니다. 실제 응답과 제공된 예시를 확인하세요."]
                notes = ["**읽는 법:** v1 `SUCCESS`·HTTP 200은 요청 처리를 뜻하며 분석 가능·완료·정상 판정을 보장하지 않습니다.",
                         "- `available`은 분석 가능, `not_available`·`unavailable`은 미평가/제한, `pending`은 결과 대기입니다. 상태와 `reason_code`를 함께 확인하세요.",
                         "- `null`은 미제공·미판정 등 해당 필드의 빈 값입니다. 0점·정상·0µm로 바꾸어 해석하지 않습니다. `degraded=true`는 제한 답변입니다."]
                if path.endswith("/analyze-sequence"):
                    notes.append("- 도막 정상범위는 기본 160~200µm(경계 포함)입니다. 다음 구역 안내는 공식 Zone 면적의 정상범위 충족 95% 이상 등 진행 조건을 확인하며 평균 두께만으로 이동하지 않습니다.")
                if "/test/" in path:
                    notes.append("- **모의·테스트 예시:** `test_context`의 생성 방식을 확인하세요. 형식·UI 확인용 수치는 실측 결과가 아니며 실제 실행값과 다를 수 있습니다. 아래 예시에서 관찰되는 키는 테스트별 가변 항목입니다.")
                documentation = "\n".join([_MARKER, "", *notes, "", *fields, "", "### 응답 예시", "",
                    "기존 등록 예시 또는 형식 확인용 미평가 예시입니다. 고정 Zone·식별자·시간·수치는 운영 데이터가 아니며 실제 시나리오·공정·입력에 따라 달라집니다.", "", "```json", json.dumps(example, ensure_ascii=False, indent=2), "```"])
            operation["description"] = str(operation.get("description", "")).rstrip() + "\n\n" + documentation
