"""PR #8 도막 정의가 내부 채팅 프롬프트·도구 설명·합성 지시문에 반영됐는지 지킨다."""

from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

from trainer.demo_api import coating_assessment
from trainer.scripts import tool_calling_adapter as adapter

ROOT = Path(__file__).resolve().parents[3]
SPARK = ROOT / "trainer/configs/rag_config_spark_vllm.yaml"
FEWSHOT = ROOT / "trainer/configs/rag_config_fewshot_27b.yaml"
SCHEMAS = ROOT / "trainer/configs/tool_schemas"


def _system(path: Path) -> str:
    return yaml.safe_load(path.read_text(encoding="utf-8"))["prompt"]["system"]


def _schema(name: str) -> dict:
    return json.loads((SCHEMAS / f"{name}.json").read_text(encoding="utf-8"))["function"]


def test_fast_definition_uses_the_training_grade_rule() -> None:
    text = (ROOT / "trainer/scripts/agent_pipeline.py").read_text(encoding="utf-8")
    assert coating_assessment.OVER_PHRASE in text
    assert "규정 두께를 초과한 도막" not in text


def test_both_serving_configs_share_the_system_prompt() -> None:
    assert _system(SPARK) == _system(FEWSHOT)


def test_system_prompt_examples_follow_the_new_rule() -> None:
    system = _system(SPARK)
    for gone in ('"90/10"', "B-3", '"nominal": 100', "근본 원인"):
        assert gone not in system
    assert '"nominal": 160' in system
    calls = [json.loads(m) for m in re.findall(r"<tool_call>(\{.*?\})</tool_call>", system, re.S)]
    by_name = {c["name"]: c["arguments"] for c in calls}
    for name in ("dft_judge", "anomaly_detection"):
        params = _schema(name)["parameters"]
        adapter.fill_default_parameters(by_name[name], params["properties"], params.get("required", []))


def test_schemas_no_longer_require_the_tolerance_rule() -> None:
    params = _schema("dft_judge")["parameters"]
    assert params["required"] == ["measurements"]
    assert "required" not in params["properties"]["spec"]
    adapter.fill_default_parameters({"measurements": [150]}, params["properties"], params["required"])


def test_synthesis_example_matches_the_mcp_output() -> None:
    assert "90/10" not in adapter.SYNTHESIS_INSTRUCTION
    r = coating_assessment.assess_readings([150, 158, 171], 160)
    assert f"{r['painted_mean_um']:.1f}μm" in adapter.SYNTHESIS_INSTRUCTION
    assert f"목표 {r['target_thickness_um']:.0f}μm" in adapter.SYNTHESIS_INSTRUCTION
    # 예시 입력은 속도·거리 표본이 없어 조절 후보가 없다. 예시에 후보 문장이 있으면 안 된다.
    assert r["adjustments"] == [] and "조절 후보는" not in adapter.SYNTHESIS_INSTRUCTION
