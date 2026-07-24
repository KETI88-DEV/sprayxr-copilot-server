#!/usr/bin/env python3

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import logging
import os
import re
import uuid
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

TOOL_CALL_REGEX = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", flags=re.DOTALL)
TOOL_CALL_STRIP_REGEX = re.compile(r"<tool_call>.*?</tool_call>", flags=re.DOTALL)
THINK_BLOCK_REGEX = re.compile(r"<think>.*?</think>", flags=re.DOTALL)

SYNTHESIS_INSTRUCTION = (
    "도구 호출이 끝났다. 이제 도구 결과와 문서 근거를 통합하여 최종 한국어 코칭 답변을 작성하라. "
    "추가 도구 호출은 하지 마라. 반드시 아래 구조를 정확히 이 순서로 사용하라:\n"
    "[분석] 도구 결과와 측정값을 종합한 상황 분석 (2-3문장)\n"
    "[판정] 합격/불합격 또는 정상/이상 한 줄 결론\n"
    "[원인] 문제가 있으면 근본 원인 1-2가지 (없으면 생략)\n"
    "[조치] 구체적 행동 항목을 번호 목록으로 1-3개\n"
    "[근거] 참조 문서 ID를 'DOC-xxx-Cxxx' 형식으로 명시\n\n"
    "예시:\n"
    "[분석] DFT 측정값 [82, 88, 140]μm을 90/10 규칙으로 판정한 결과, 90% 기준(90μm) 미만이 2건입니다.\n"
    "[판정] 불합격\n"
    "[원인] 하부 구간 스프레이 거리 과다로 도료 도달량이 부족했습니다.\n"
    "[조치]\n"
    "1. 분사 거리를 400~600mm로 유지하십시오.\n"
    "2. 미달 구간을 재도장 후 재측정하십시오.\n"
    "[근거] 참조 문서 ID: DOC-doc_0421-C0007"
)


def _is_type_match(value: Any, expected_type: str) -> bool:
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "array":
        return isinstance(value, list)
    if expected_type == "object":
        return isinstance(value, dict)
    return True


def _coerce_scalar(value: Any, expected_type: str) -> Any:
    if expected_type == "boolean" and isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "false"}:
            return lowered == "true"

    if expected_type in {"number", "integer"} and isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return value
        try:
            numeric = float(stripped)
        except ValueError:
            return value

        if expected_type == "integer":
            if numeric.is_integer():
                return int(numeric)
            return value

        if numeric.is_integer():
            return int(numeric)
        return numeric

    return value


def _coerce_value(value: Any, schema: dict[str, Any]) -> Any:
    expected_type = schema.get("type")

    if expected_type == "array":
        items_schema = schema.get("items", {})
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.startswith("[") and stripped.endswith("]"):
                try:
                    parsed = json.loads(stripped)
                    if isinstance(parsed, list):
                        value = parsed
                except json.JSONDecodeError:
                    pass
            elif "," in stripped:
                parts = [part.strip() for part in stripped.split(",") if part.strip()]
                value = [_coerce_value(part, items_schema) for part in parts]

        if isinstance(value, list):
            return [_coerce_value(item, items_schema) for item in value]
        return value

    if expected_type == "object":
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.startswith("{") and stripped.endswith("}"):
                try:
                    parsed = json.loads(stripped)
                    if isinstance(parsed, dict):
                        value = parsed
                except json.JSONDecodeError:
                    pass

        if isinstance(value, dict):
            properties = schema.get("properties", {})
            additional_properties = schema.get("additionalProperties")
            coerced: dict[str, Any] = {}
            for key, item in value.items():
                if key in properties:
                    coerced[key] = _coerce_value(item, properties[key])
                elif isinstance(additional_properties, dict):
                    coerced[key] = _coerce_value(item, additional_properties)
                else:
                    coerced[key] = item
            return coerced
        return value

    if isinstance(expected_type, str):
        return _coerce_scalar(value, expected_type)
    return value


def coerce_parameter_types(params: dict[str, Any], schema_properties: dict[str, Any]) -> dict[str, Any]:
    """Auto-convert common parameter type mismatches using schema hints."""
    coerced: dict[str, Any] = {}
    for key, value in params.items():
        schema = schema_properties.get(key)
        if isinstance(schema, dict):
            coerced[key] = _coerce_value(value, schema)
        else:
            coerced[key] = value
    return coerced


def fill_default_parameters(
    params: dict[str, Any],
    schema_properties: dict[str, Any],
    required: list[str],
) -> dict[str, Any]:
    """Fill schema defaults and raise on missing required fields."""
    if not isinstance(params, dict):
        raise ValueError("Parameters must be an object")

    filled = dict(params)

    for key, schema in schema_properties.items():
        if key not in filled:
            if "default" in schema:
                filled[key] = schema["default"]
            elif key in required:
                raise ValueError(f"Missing required parameter: {key}")
            continue

        value = filled[key]
        expected_type = schema.get("type")
        if expected_type == "object" and isinstance(value, dict):
            child_props = schema.get("properties", {})
            child_required = schema.get("required", [])
            filled[key] = fill_default_parameters(value, child_props, child_required)
        elif expected_type == "array" and isinstance(value, list):
            item_schema = schema.get("items", {})
            if item_schema.get("type") == "object":
                child_props = item_schema.get("properties", {})
                child_required = item_schema.get("required", [])
                filled[key] = [
                    fill_default_parameters(item, child_props, child_required) if isinstance(item, dict) else item
                    for item in value
                ]

    return filled


def _validate_value(path: str, value: Any, schema: dict[str, Any], errors: list[str]) -> None:
    expected_type = schema.get("type")

    if isinstance(expected_type, str) and not _is_type_match(value, expected_type):
        errors.append(
            f"{path}: expected {expected_type}, got {type(value).__name__}"
        )
        return

    enum_values = schema.get("enum")
    if isinstance(enum_values, list) and value not in enum_values:
        errors.append(f"{path}: value {value!r} not in enum {enum_values}")

    if expected_type == "object" and isinstance(value, dict):
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        additional_properties = schema.get("additionalProperties")

        for req_key in required:
            if req_key not in value:
                errors.append(f"{path}.{req_key}: missing required field")

        for key, item in value.items():
            child_path = f"{path}.{key}"
            if key in properties:
                _validate_value(child_path, item, properties[key], errors)
            elif isinstance(additional_properties, dict):
                _validate_value(child_path, item, additional_properties, errors)

    if expected_type == "array" and isinstance(value, list):
        item_schema = schema.get("items", {})
        for idx, item in enumerate(value):
            _validate_value(f"{path}[{idx}]", item, item_schema, errors)


def validate_tool_parameters(
    tool_name: str,
    params: dict[str, Any],
    schema_dir: str | None = None,
) -> dict[str, Any]:
    """Validate params against tool schema and return corrected payload metadata."""
    if schema_dir is None:
        schema_dir = "/workspace/trainer/configs/tool_schemas"

    errors: list[str] = []

    schema_path = Path(schema_dir) / f"{tool_name}.json"
    if not schema_path.is_file():
        return {
            "is_valid": False,
            "errors": [f"Schema not found for tool '{tool_name}': {schema_path}"],
            "corrected_params": params,
        }

    try:
        with schema_path.open("r", encoding="utf-8") as fh:
            schema = json.load(fh)
    except json.JSONDecodeError as exc:
        return {
            "is_valid": False,
            "errors": [f"Invalid schema JSON for tool '{tool_name}': {exc}"],
            "corrected_params": params,
        }

    parameters_schema = schema.get("function", {}).get("parameters", {})
    schema_properties = parameters_schema.get("properties", {})
    required = parameters_schema.get("required", [])

    corrected_params = coerce_parameter_types(copy.deepcopy(params), schema_properties)

    try:
        corrected_params = fill_default_parameters(corrected_params, schema_properties, required)
    except ValueError as exc:
        errors.append(str(exc))

    _validate_value("params", corrected_params, parameters_schema, errors)

    return {
        "is_valid": len(errors) == 0,
        "errors": errors,
        "corrected_params": corrected_params,
    }


class ToolCallingError(Exception):
    """Represents adapter-level failures that should stop the current turn."""


class ToolValidationError(ToolCallingError):
    """Represents malformed or unsupported tool calls before network dispatch."""


class ToolInfrastructureError(ToolCallingError):
    """Represents transport/runtime faults while calling MCP endpoints."""


class ToolCallingAdapter:
    """Bridges Qwen3 native function-calling format with MCP HTTP endpoints."""

    def __init__(
        self,
        schema_dir: str | Path,
        protocol_path: str | Path,
        mcp_base_url: str | None = None,
        pass_native_tools: bool | None = None,
    ) -> None:
        self.logger = logger
        self.schema_dir = Path(schema_dir)
        self.protocol_path = Path(protocol_path)
        self.protocol = self._load_json_file(self.protocol_path)
        self.tool_schemas = self._load_tool_schemas(self.schema_dir)

        self.registered_tools: set[str] = set(self.protocol.get("registered_tools", []))
        self.endpoint_mapping: dict[str, str] = self.protocol.get("endpoint_mapping", {})

        retry_policy = self.protocol.get("retry_policy", {})
        self.max_retries = int(retry_policy.get("max_retries", 0))
        self.retry_on = set(retry_policy.get("retry_on", []))
        self.backoff_ms = int(retry_policy.get("backoff_ms", 0))

        request_format = self.protocol.get("request_format", {})
        self.timeout_ms = int(request_format.get("timeout_ms", {}).get("default", 10000))

        base_url = mcp_base_url or self.protocol.get("mcp_server_base_url", "http://localhost:8100")
        self.mcp_base_url = base_url.rstrip("/")

        if pass_native_tools is None:
            pass_native_tools = os.environ.get("TOOL_ADAPTER_NATIVE_TOOLS", "1") != "0"
        self.pass_native_tools = pass_native_tools
        self.force_synthesis = os.environ.get("TOOL_ADAPTER_FORCE_SYNTHESIS", "1") != "0"

    @staticmethod
    def _load_json_file(path: Path) -> dict[str, Any]:
        if not path.is_file():
            raise ToolValidationError(f"JSON file not found: {path}")
        try:
            with path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except json.JSONDecodeError as exc:
            raise ToolValidationError(f"Invalid JSON file: {path}") from exc
        if not isinstance(data, dict):
            raise ToolValidationError(f"Expected JSON object in: {path}")
        return data

    @staticmethod
    def _strip_think_blocks(text: str) -> str:
        return THINK_BLOCK_REGEX.sub("", text).strip()

    def _load_tool_schemas(self, schema_dir: Path) -> list[dict[str, Any]]:
        if not schema_dir.is_dir():
            raise ToolValidationError(f"Schema directory not found: {schema_dir}")

        schemas: list[dict[str, Any]] = []
        for json_path in sorted(schema_dir.glob("*.json")):
            schema = self._load_json_file(json_path)
            function_data = schema.get("function", {})
            tool_name = function_data.get("name")
            if not tool_name:
                raise ToolValidationError(f"Missing function.name in schema: {json_path}")
            schemas.append(schema)

        if not schemas:
            raise ToolValidationError(f"No tool schema files found in: {schema_dir}")
        return schemas

    def get_tools_param(self) -> list[dict[str, Any]]:
        """Provides OpenAI-compatible tool definitions expected by Qwen3 templates."""
        return self.tool_schemas

    @staticmethod
    def _extract_tool_call_blocks(raw_text: str) -> list[str]:
        blocks: list[str] = []
        pos = 0
        while True:
            start = raw_text.find("<tool_call>", pos)
            if start == -1:
                break
            brace_start = raw_text.find("{", start)
            if brace_start == -1:
                break
            depth = 0
            in_str = False
            escape = False
            end = -1
            for i in range(brace_start, len(raw_text)):
                ch = raw_text[i]
                if escape:
                    escape = False
                    continue
                if ch == "\\":
                    escape = True
                    continue
                if ch == '"':
                    in_str = not in_str
                    continue
                if in_str:
                    continue
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            if end == -1:
                break
            blocks.append(raw_text[brace_start:end])
            pos = end
        return blocks

    def parse_tool_calls(self, raw_text: str) -> list[dict[str, Any]]:
        """Extracts tool calls robustly, handling nested-object args and missing closing tags."""
        if not raw_text:
            return []

        tool_calls: list[dict[str, Any]] = []
        for block in self._extract_tool_call_blocks(raw_text):
            try:
                payload = json.loads(block)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed tool_call JSON (len=%d): %s...", len(block), block[:120])
                continue

            if not isinstance(payload, dict):
                logger.warning("Skipping non-object tool_call payload")
                continue

            tool_name = payload.get("name")
            if not isinstance(tool_name, str) or not tool_name:
                logger.warning("Skipping tool_call with missing name")
                continue

            arguments = payload.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    logger.warning("Skipping tool_call with unparseable string arguments for '%s'", tool_name)
                    continue
            if not isinstance(arguments, dict):
                logger.warning("Skipping tool_call with non-object arguments for '%s'", tool_name)
                continue

            tool_calls.append(
                {
                    "name": tool_name,
                    "arguments": arguments,
                    "call_id": str(uuid.uuid4()),
                }
            )
        return tool_calls

    def format_tool_results(self, results: list[dict[str, Any]]) -> list[dict[str, str]]:
        """Normalizes tool outputs into role=tool messages for follow-up generation."""
        messages: list[dict[str, str]] = []
        for result in results:
            call_id = str(result.get("call_id") or str(uuid.uuid4()))
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": json.dumps(result, ensure_ascii=False),
                }
            )
        return messages

    def _resolve_endpoint(self, tool_name: str, arguments: dict[str, Any]) -> str:
        endpoint_template = self.endpoint_mapping.get(tool_name)
        if not endpoint_template:
            raise ToolValidationError(f"No endpoint mapping found for tool: {tool_name}")

        if "{action}" in endpoint_template:
            action = arguments.get("action")
            if not isinstance(action, str) or not action:
                raise ToolValidationError("timeseries tool requires non-empty 'action' argument")
            endpoint = endpoint_template.format(action=action)
        else:
            endpoint = endpoint_template
        return f"{self.mcp_base_url}{endpoint}"

    async def dispatch_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Executes protocol-compliant MCP calls with retry boundaries for transient faults."""
        call_id = str(uuid.uuid4())

        if tool_name not in self.registered_tools:
            logger.warning("Unregistered tool requested: %s — returning error result", tool_name)
            return {
                "call_id": call_id,
                "status": "error",
                "error": {
                    "code": "TOOL_NOT_FOUND",
                    "message": f"Unregistered tool: {tool_name}",
                    "details": {"available": sorted(self.registered_tools)},
                },
            }

        if not isinstance(arguments, dict):
            return {
                "call_id": call_id,
                "status": "error",
                "error": {
                    "code": "INVALID_PARAMS",
                    "message": f"Arguments must be a dictionary for {tool_name}",
                    "details": {},
                },
            }

        validation_result = validate_tool_parameters(
            tool_name=tool_name,
            params=arguments,
            schema_dir=str(self.schema_dir),
        )
        corrected_arguments = validation_result.get("corrected_params", arguments)
        validation_errors = validation_result.get("errors", [])
        if not validation_result.get("is_valid", False):
            self.logger.warning(
                "Tool parameter validation failed for '%s': %s",
                tool_name,
                validation_errors,
            )
            return {
                "call_id": call_id,
                "status": "error",
                "error": {
                    "code": "INVALID_PARAMS",
                    "message": f"Invalid parameters for {tool_name}",
                    "details": {"errors": validation_errors},
                },
            }

        if corrected_arguments != arguments:
            self.logger.warning(
                "Tool parameters auto-corrected for '%s': original=%s corrected=%s",
                tool_name,
                arguments,
                corrected_arguments,
            )
        elif validation_errors:
            self.logger.warning(
                "Tool parameter validation warnings for '%s': %s",
                tool_name,
                validation_errors,
            )

        mcp_body = dict(corrected_arguments)
        if session_id is not None:
            mcp_body.setdefault("session_id", session_id)

        url = self._resolve_endpoint(tool_name, corrected_arguments)
        timeout_seconds = self.timeout_ms / 1000.0

        for attempt in range(self.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=timeout_seconds) as client:
                    response = await client.post(url, json=mcp_body)
                response.raise_for_status()

                response_data = response.json()
                if not isinstance(response_data, dict):
                    raise ToolInfrastructureError("MCP response must be a JSON object")
                response_data.setdefault("call_id", call_id)
                return response_data

            except httpx.TimeoutException as exc:
                error_code = "TIMEOUT"
                if attempt < self.max_retries and error_code in self.retry_on:
                    await asyncio.sleep(self.backoff_ms / 1000.0)
                    continue
                return {
                    "call_id": call_id,
                    "status": "error",
                    "error": {
                        "code": error_code,
                        "message": f"MCP timeout for {tool_name}",
                        "details": {"exception": str(exc)},
                    },
                }

            except httpx.HTTPStatusError as exc:
                status_code = exc.response.status_code
                error_code = "SERVER_ERROR" if status_code >= 500 else "INVALID_PARAMS"
                if attempt < self.max_retries and error_code in self.retry_on:
                    await asyncio.sleep(self.backoff_ms / 1000.0)
                    continue
                return {
                    "call_id": call_id,
                    "status": "error",
                    "error": {
                        "code": error_code,
                        "message": f"MCP HTTP error {status_code} for {tool_name}",
                        "details": {"response": exc.response.text},
                    },
                }

            except (httpx.RequestError, ValueError) as exc:
                raise ToolInfrastructureError(f"MCP request failed for {tool_name}") from exc

        return {
            "call_id": call_id,
            "status": "error",
            "error": {
                "code": "SERVER_ERROR",
                "message": f"MCP request exhausted retries for {tool_name}",
                "details": {},
            },
        }

    async def _generate_model_text(
        self,
        messages: list[dict[str, Any]],
        model: Any,
        tokenizer: Any,
        enable_thinking: bool,
        max_tokens: int | None = None,
        gen_params: dict[str, Any] | None = None,
    ) -> str:
        if hasattr(model, "generate_response"):
            response = model.generate_response(messages=messages, tools=self.get_tools_param())
            if asyncio.iscoroutine(response):
                response = await response
            if not isinstance(response, str):
                raise ToolInfrastructureError("Model generate_response must return a string")
            return response

        template_kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": enable_thinking,
        }
        if self.pass_native_tools:
            template_kwargs["tools"] = self.get_tools_param()
        prompt_text = tokenizer.apply_chat_template(messages, **template_kwargs)

        inputs = tokenizer([prompt_text], return_tensors="pt", return_attention_mask=True)
        if hasattr(inputs, "to") and hasattr(model, "device"):
            inputs = inputs.to(model.device)

        gen_max_tokens = max_tokens or 1024
        gp = gen_params or {}
        do_sample = bool(gp.get("do_sample", True))
        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": gen_max_tokens,
            "pad_token_id": tokenizer.eos_token_id,
            "repetition_penalty": float(gp.get("repetition_penalty", 1.15)),
            "do_sample": do_sample,
        }
        if do_sample:
            generate_kwargs["temperature"] = float(gp.get("temperature", 0.3))
            generate_kwargs["top_p"] = float(gp.get("top_p", 0.8))
            generate_kwargs["top_k"] = int(gp.get("top_k", 20))

        def _run_generation() -> Any:
            return model.generate(**inputs, **generate_kwargs)

        output_ids = await asyncio.to_thread(_run_generation)
        generated = output_ids[0][inputs.input_ids.shape[1] :]
        return tokenizer.decode(generated, skip_special_tokens=True)

    async def _generate_synthesis_text(
        self,
        messages: list[dict[str, Any]],
        model: Any,
        tokenizer: Any,
        max_tokens: int | None = None,
        gen_params: dict[str, Any] | None = None,
    ) -> str:
        synthesis_messages = list(messages)
        synthesis_messages.append({"role": "user", "content": SYNTHESIS_INSTRUCTION})

        if hasattr(model, "generate_response"):
            response = model.generate_response(messages=synthesis_messages, tools=None)
            if asyncio.iscoroutine(response):
                response = await response
            if not isinstance(response, str):
                raise ToolInfrastructureError("Model generate_response must return a string")
            return TOOL_CALL_STRIP_REGEX.sub("", response).strip()

        prompt_text = tokenizer.apply_chat_template(
            synthesis_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer([prompt_text], return_tensors="pt", return_attention_mask=True)
        if hasattr(inputs, "to") and hasattr(model, "device"):
            inputs = inputs.to(model.device)

        gen_max_tokens = max_tokens or 1024
        gp = gen_params or {}
        do_sample = bool(gp.get("do_sample", True))
        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": gen_max_tokens,
            "pad_token_id": tokenizer.eos_token_id,
            "repetition_penalty": float(gp.get("repetition_penalty", 1.15)),
            "do_sample": do_sample,
        }
        if do_sample:
            generate_kwargs["temperature"] = float(gp.get("temperature", 0.3))
            generate_kwargs["top_p"] = float(gp.get("top_p", 0.8))
            generate_kwargs["top_k"] = int(gp.get("top_k", 20))

        def _run_generation() -> Any:
            return model.generate(**inputs, **generate_kwargs)

        output_ids = await asyncio.to_thread(_run_generation)
        generated = output_ids[0][inputs.input_ids.shape[1] :]
        text = tokenizer.decode(generated, skip_special_tokens=True)
        return TOOL_CALL_STRIP_REGEX.sub("", self._strip_think_blocks(text)).strip()

    async def run_tool_loop(
        self,
        messages: list[dict[str, Any]],
        model: Any,
        tokenizer: Any,
        enable_thinking: bool = False,
        max_rounds: int = 5,
        max_tokens: int | None = None,
        gen_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Runs the multi-turn tool loop until the model emits a final natural-language answer."""
        if max_rounds <= 0:
            raise ToolValidationError("max_rounds must be greater than zero")

        conversation = list(messages)
        collected_tool_results: list[dict[str, Any]] = []

        for round_idx in range(max_rounds):
            raw_text = await self._generate_model_text(
                messages=conversation,
                model=model,
                tokenizer=tokenizer,
                enable_thinking=enable_thinking,
                max_tokens=max_tokens,
                gen_params=gen_params,
            )
            tool_calls = self.parse_tool_calls(raw_text)

            if not tool_calls:
                final_text = self._strip_think_blocks(raw_text) if enable_thinking else raw_text.strip()
                stripped = TOOL_CALL_STRIP_REGEX.sub("", final_text).strip()
                if self.force_synthesis and collected_tool_results and not stripped:
                    final_text = await self._generate_synthesis_text(
                        conversation, model, tokenizer, max_tokens=max_tokens, gen_params=gen_params,
                    )
                else:
                    final_text = stripped
                conversation.append({"role": "assistant", "content": final_text})
                return {
                    "final_response": final_text,
                    "messages": conversation,
                    "tool_results": collected_tool_results,
                    "rounds": round_idx + 1,
                }

            conversation.append({"role": "assistant", "content": raw_text})

            dispatch_tasks = [
                self.dispatch_tool_call(call["name"], call["arguments"]) for call in tool_calls
            ]
            dispatch_results = await asyncio.gather(*dispatch_tasks)

            normalized_results: list[dict[str, Any]] = []
            for call, result in zip(tool_calls, dispatch_results):
                item = dict(result)
                item.setdefault("call_id", call["call_id"])
                item.setdefault("tool_name", call["name"])
                normalized_results.append(item)

            collected_tool_results.extend(normalized_results)
            conversation.extend(self.format_tool_results(normalized_results))

        logger.warning(
            "Tool loop reached max_rounds=%d without final answer; forcing synthesis",
            max_rounds,
        )

        if self.force_synthesis and collected_tool_results:
            final_text = await self._generate_synthesis_text(
                conversation, model, tokenizer, max_tokens=max_tokens, gen_params=gen_params,
            )
            conversation.append({"role": "assistant", "content": final_text})
            return {
                "final_response": final_text,
                "messages": conversation,
                "tool_results": collected_tool_results,
                "rounds": max_rounds,
                "truncated": True,
                "synthesized": True,
            }

        last_assistant = ""
        for msg in reversed(conversation):
            if msg.get("role") == "assistant":
                last_assistant = TOOL_CALL_STRIP_REGEX.sub(
                    "", self._strip_think_blocks(msg.get("content", ""))
                ).strip()
                break

        return {
            "final_response": last_assistant,
            "messages": conversation,
            "tool_results": collected_tool_results,
            "rounds": max_rounds,
            "truncated": True,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen3 tool-calling adapter for MCP dispatch.")
    parser.add_argument(
        "--schema-dir",
        type=Path,
        default=Path("/workspace/trainer/configs/tool_schemas"),
        help="Directory containing tool schema JSON files.",
    )
    parser.add_argument(
        "--protocol-path",
        type=Path,
        default=Path("/workspace/trainer/configs/tool_calling_protocol.json"),
        help="Tool-calling protocol JSON path.",
    )
    parser.add_argument(
        "--mcp-base-url",
        type=str,
        default=None,
        help="Optional MCP base URL override.",
    )
    parser.add_argument(
        "--parse-text",
        type=str,
        default=None,
        help="Raw model text for validating tool_call parsing.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    try:
        adapter = ToolCallingAdapter(
            schema_dir=args.schema_dir,
            protocol_path=args.protocol_path,
            mcp_base_url=args.mcp_base_url,
        )
    except ToolCallingError:
        logger.exception("Failed to initialize tool-calling adapter")
        return 1

    logger.info("Loaded %d tool schemas", len(adapter.get_tools_param()))

    if args.parse_text is not None:
        try:
            parsed = adapter.parse_tool_calls(args.parse_text)
        except ToolCallingError:
            logger.exception("Failed to parse tool calls")
            return 1
        print(json.dumps(parsed, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
