#!/usr/bin/env python3
"""HF 토크나이저와 vLLM ``/tokenize``의 토큰 ID가 같은지 확인한다 (agent-api 컨테이너 안에서 실행).

agent-api는 ``llm.tokenizer_path``의 HF 토크나이저로 프롬프트 예산을 세고(prompt_budget.py),
vLLM은 자기 토크나이저와 채팅 템플릿으로 ``--max-model-len``을 강제한다. 둘이 다르면 예산이 틀리고,
템플릿 차이는 곧 학습 분포와 다른 입력이다.

비교는 prompt_budget.TokenizerTokenCounter와 같은 규약이다:
``apply_chat_template(tokenize=True, add_generation_prompt=True, enable_thinking=False)``.

    docker compose run --rm --no-deps agent-api python3 deploy/spark/scripts/check_tokenizer_parity.py

종료 코드: 0 전 사례 일치, 1 불일치, 2 설정·연결 오류.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import httpx
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from trainer.scripts.openai_backend import OpenAIBackendSettings  # noqa: E402

_TOOL_CALL = (
    '<tool_call>{"name": "dft_judge", "arguments": {"measurements": [95, 102, 88], '
    '"spec": {"nominal": 100, "tolerance_rule": "90/10"}}}</tool_call>'
)


def build_cases(config: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    prompt = config.get("prompt") or {}
    context = str(prompt.get("context_template", "{context}")).format(
        context="[DOC-001-C0001] 무기질 아연 실리케이트 프라이머는 블라스팅 직후 4시간 이내에 도포한다.\n"
        "[DOC-002-C0003] 상대습도 85% 이상이거나 강판 온도가 이슬점 +3℃ 미만이면 도장을 중지한다."
    )
    question = str(prompt.get("user_template", "{question}")).format(question="무기질 아연 프라이머의 도장 조건은? (DFT 75㎛)")
    environment = json.dumps({"zone_id": "A-1", "dft_um": [95, 102, 88]}, ensure_ascii=False, indent=2)
    chat = [
        {
            "name": "fast_route_rag",
            "messages": [
                {"role": "system", "content": str(prompt["fast_system"])},
                {"role": "user", "content": f"{context}\n\n{question}"},
            ],
        },
        {
            "name": "tool_route_environment",
            "messages": [
                {"role": "system", "content": str(prompt["system"])},
                {"role": "user", "content": f"[환경 데이터]\n{environment}\n\n질문: A-1 구역 DFT 판정해줘"},
            ],
        },
        {
            "name": "tool_round_replay",
            "messages": [
                {"role": "system", "content": str(prompt["system"])},
                {"role": "user", "content": "질문: DFT 95, 102, 88 판정해줘"},
                {"role": "assistant", "content": _TOOL_CALL},
                {"role": "tool", "content": json.dumps({"status": "success", "result": {"verdict": "합격"}}, ensure_ascii=False)},
            ],
        },
    ]
    text = [
        {"name": "raw_korean_units", "prompt": "선박 도장 DFT 75㎛ — 이슬점 +3℃, RH≤85%, 도막 두께 판정"},
        {"name": "raw_protocol_tokens", "prompt": f"{_TOOL_CALL}\n<think>\n\n</think>\n[답변]: 확인"},
    ]
    return chat, text


def _ids(encoded: Any) -> list[int]:
    ids = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
    if ids and isinstance(ids[0], (list, tuple)):
        raise ValueError("expected a single sequence")
    return [int(token) for token in ids]


def first_difference(left: Sequence[int], right: Sequence[int]) -> int | None:
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return None if len(left) == len(right) else min(len(left), len(right))


def _compare(name: str, hf_ids: list[int], vllm_ids: list[int], tokenizer: Any) -> dict[str, Any]:
    diff = first_difference(hf_ids, vllm_ids)
    result: dict[str, Any] = {"name": name, "equal": diff is None, "hf_tokens": len(hf_ids), "vllm_tokens": len(vllm_ids)}
    if diff is not None:
        window = slice(max(0, diff - 3), diff + 4)
        result["first_difference"] = diff
        result["hf_around"] = tokenizer.convert_ids_to_tokens(hf_ids[window])
        result["vllm_around"] = tokenizer.convert_ids_to_tokens(vllm_ids[window])
    return result


def run(config: Mapping[str, Any], vllm_root: str | None) -> tuple[int, list[dict[str, Any]]]:
    from transformers import AutoTokenizer

    settings = OpenAIBackendSettings.from_config(config)
    tokenizer = AutoTokenizer.from_pretrained(str(config["llm"]["tokenizer_path"]), trust_remote_code=True)
    root = (vllm_root or settings.server_root).rstrip("/")
    chat_cases, text_cases = build_cases(config)
    results: list[dict[str, Any]] = []
    with httpx.Client(headers=settings.headers(), timeout=30.0) as client:
        for case in chat_cases:
            hf_ids = _ids(
                tokenizer.apply_chat_template(case["messages"], tokenize=True, add_generation_prompt=True, enable_thinking=False)
            )
            response = client.post(
                f"{root}/tokenize",
                json={
                    "model": settings.served_model_name,
                    "messages": case["messages"],
                    "add_generation_prompt": True,
                    "chat_template_kwargs": {"enable_thinking": False},
                    "add_special_tokens": False,
                },
            )
            response.raise_for_status()
            results.append(_compare(case["name"], hf_ids, _ids(response.json()["tokens"]), tokenizer))
        for case in text_cases:
            hf_ids = _ids(tokenizer(case["prompt"], add_special_tokens=False))
            response = client.post(
                f"{root}/tokenize",
                json={"model": settings.served_model_name, "prompt": case["prompt"], "add_special_tokens": False},
            )
            response.raise_for_status()
            results.append(_compare(case["name"], hf_ids, _ids(response.json()["tokens"]), tokenizer))
    return (0 if all(item["equal"] for item in results) else 1), results


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path(os.environ.get("CONFIG_PATH", "")))
    parser.add_argument("--vllm-root", default=None, help="default: llm.base_url without /v1")
    args = parser.parse_args(argv)
    try:
        config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        code, results = run(config, args.vllm_root)
    except (OSError, ValueError, KeyError, httpx.HTTPError) as exc:
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False))
        return 2
    print(json.dumps({"ok": code == 0, "cases": results}, ensure_ascii=False, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
