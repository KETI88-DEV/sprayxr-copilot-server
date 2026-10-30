"""Show the server's test inputs before Execute, without inventing request bodies."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

_MARKER = "### 실행 전 확인: 자동 사용 입력"
_METHODS = {"get", "post", "put", "patch", "delete"}


def _values(value: Any, path: str = "") -> Iterator[tuple[str, Any, str]]:
    if isinstance(value, dict):
        if not value:
            yield path, value, "object"
        for key, child in value.items():
            yield from _values(child, f"{path}.{key}" if path else str(key))
    elif isinstance(value, list):
        if not value:
            yield path, value, "array"
        for index, child in enumerate(value):
            yield from _values(child, f"{path}[{index}]")
    else:
        kind = ("null" if value is None else "boolean" if isinstance(value, bool) else
                "integer" if isinstance(value, int) else "number" if isinstance(value, float) else "string")
        yield path, value, kind


def _context(operation: dict[str, Any]) -> dict[str, Any]:
    response = operation.get("responses", {}).get("200", {})
    media = response.get("content", {}).get("application/json", {})
    example = media.get("example")
    if not isinstance(example, dict):
        samples = media.get("examples", {})
        example = next((item.get("value") for item in samples.values() if isinstance(item, dict)), {})
    return example.get("data", {}).get("test_context", {}) if isinstance(example, dict) else {}


def add_test_input_overviews(schema: dict[str, Any]) -> None:
    """Append accurate, typed internal inputs to every visible /api/v1/test operation."""
    for path, operations in schema.get("paths", {}).items():
        if not path.startswith("/api/v1/test/"):
            continue
        for method, operation in operations.items():
            if method not in _METHODS or not isinstance(operation, dict):
                continue
            description = str(operation.get("description", ""))
            if _MARKER in description:
                continue
            context = _context(operation)
            used = context.get("input_used")
            if not isinstance(used, dict):
                used = {"request_parameters": "없음"}
                if path.endswith("/assets/query-voice-ko.wav"):
                    used["asset"] = {
                        "filename": "query-voice-ko.wav", "media_type": "audio/wav",
                        "format": "WAV PCM16 mono", "sample_rate_hz": 16000,
                        "spoken_text": "스프레이 도장 시 적정 이격거리를 알려주세요.",
                        "max_duration_seconds": 15,
                    }
            parameters = operation.get("parameters", [])
            query_names = {item.get("name") for item in parameters if item.get("in") == "query"}
            lines = [
                _MARKER, f"실행할 테스트 API: **`{method.upper()} {path}`**.",
                ("아래 query 항목은 화면에서 수정할 수 있으며 나머지는 서버 내부 파라미터입니다."
                 if query_names else "사용자 입력이 없는 API입니다. 아래 값은 서버 내부 파라미터이며 실제 요청 본문을 보내지 않습니다."),
                "표와 JSON은 실행 전 예시입니다. 자동 생성 식별자·시간은 런타임에 정해지며 실제 값은 응답의 `data.test_context.input_used`에서 확인합니다.",
            ]
            if context.get("input_summary"):
                lines.append(str(context["input_summary"]))
            if context.get("evaluation_focus"):
                lines.append("확인할 결과: " + str(context["evaluation_focus"]))
            lines.extend(["", "자동 입력 항목(하위 키 포함):", ""])
            dynamic_fields = {"user_id", "session_id", "sequence_id", "report_id", "started_at", "ended_at", "requested_at", "created_at", "time"}
            for name, value, kind in _values(used):
                leaf = name.rsplit(".", 1)[-1]
                note = "런타임 생성값의 예시" if "/workflow/" in path and leaf in dynamic_fields else "고정 기본값"
                if leaf in query_names:
                    note = "query 기본값 — 화면에서 변경 가능"
                lines.append(f"- `{name} ({kind})` = `{json.dumps(value, ensure_ascii=False)}` — {note}")
            lines.extend(["", "자동 입력 JSON(설명용이며 추가 requestBody를 만들지 않습니다):", "", "```json",
                          json.dumps(used, ensure_ascii=False, indent=2), "```"])
            if path.endswith("/assets/query-voice-ko.wav"):
                lines.append("이 GET은 JSON 응답 대신 WAV 바이너리를 내려줍니다. 위 JSON은 다운로드 자산의 설명용 입력입니다.")
            operation["description"] = description + "\n\n" + "\n\n".join(lines[:5]) + "\n" + "\n".join(lines[5:])
