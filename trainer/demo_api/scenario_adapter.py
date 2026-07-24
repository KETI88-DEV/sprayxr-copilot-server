from __future__ import annotations

from importlib import import_module
from typing import Any


def classify_question(text: str) -> str:
    demo_app = import_module("trainer.scripts.demo_app")
    classify = getattr(demo_app, "_classify_question")
    return classify(text)


def get_demo_scenario(question: str) -> dict[str, Any]:
    demo_app = import_module("trainer.scripts.demo_app")
    scenario_map: dict[str, dict[str, Any]] = getattr(demo_app, "DEMO_SCENARIOS")
    scenario_type = classify_question(question)
    return scenario_map.get(scenario_type, scenario_map["general"])
