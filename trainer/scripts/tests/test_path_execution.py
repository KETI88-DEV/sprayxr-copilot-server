#!/usr/bin/env python3
"""README대로 `python3 trainer/scripts/<script>.py`로 실행해도 `trainer` 패키지를 찾는지 본다.

경로로 실행하면 sys.path[0]은 trainer/scripts이고 저장소 루트는 없다. pytest 안에서는
루트가 이미 잡혀 있어 `from trainer.scripts...` 회귀가 드러나지 않으므로, PYTHONPATH를
비운 새 프로세스에서 확인한다. `--help`가 import보다 먼저 끝나는 스크립트는 경로 실행과
같은 sys.path(`-P`로 cwd를 빼고 scripts만 앞에 둔다)에서 늦은 import를 직접 부른다.

agent_pipeline import는 torch를 올리므로 모든 프로세스를 한꺼번에 띄워 기다린다. GPU는 쓰지 않는다.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = REPO_ROOT / "trainer" / "scripts"
TIMEOUT_S = 600

HELP_SCRIPTS = (
    "run_multiseed_eval",
    "run_agentic_eval",
    "bench_latency",
    "generate_agentic_responses",
    "agent_pipeline",
    "prompt_budget_study",
)

_DEFERRED_AGENT_IMPORT = (
    "import {mod} as m; m._ensure_scripts_path(); import importlib; importlib.import_module('agent_pipeline')"
)
IMPORT_CHECKS = {
    # --help는 parse_args에서 끝나고 agent_pipeline은 그 뒤에 불린다.
    "generate_agentic_responses->agent_pipeline": _DEFERRED_AGENT_IMPORT.format(mod="generate_agentic_responses"),
    "demo_agent->agent_pipeline": _DEFERRED_AGENT_IMPORT.format(mod="demo_agent"),
    "session_coaching->agent_pipeline": _DEFERRED_AGENT_IMPORT.format(mod="session_coaching"),
    # argparse가 없어 경로로 띄우면 Gradio가 뜬다. 모듈 import만 본다.
    "demo_app": "import demo_app",
    "test_response_contract": "import test_response_contract",
}


def _clean_env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["CUDA_VISIBLE_DEVICES"] = ""
    return env


@pytest.fixture(scope="module")
def results() -> Iterator[dict[str, subprocess.CompletedProcess[str]]]:
    commands: dict[str, list[str]] = {
        f"help:{name}": [sys.executable, f"trainer/scripts/{name}.py", "--help"] for name in HELP_SCRIPTS
    }
    for label, code in IMPORT_CHECKS.items():
        prelude = f"import sys; sys.path.insert(0, {str(SCRIPTS_DIR)!r}); "
        commands[f"import:{label}"] = [sys.executable, "-P", "-c", prelude + code]

    env = _clean_env()
    procs = {
        key: subprocess.Popen(
            cmd, cwd=REPO_ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        for key, cmd in commands.items()
    }
    done: dict[str, subprocess.CompletedProcess[str]] = {}
    try:
        for key, proc in procs.items():
            out, err = proc.communicate(timeout=TIMEOUT_S)
            done[key] = subprocess.CompletedProcess(commands[key], proc.returncode, out, err)
        yield done
    finally:
        for proc in procs.values():
            if proc.poll() is None:
                proc.kill()


def test_path_import_check_does_not_see_repo_root() -> None:
    """검사 자체가 루트를 몰래 보면 회귀를 못 잡는다."""
    code = f"import sys; sys.path.insert(0, {str(SCRIPTS_DIR)!r}); import trainer"
    proc = subprocess.run(
        [sys.executable, "-P", "-c", code],
        cwd=REPO_ROOT, env=_clean_env(), capture_output=True, text=True, timeout=TIMEOUT_S, check=False,
    )
    assert proc.returncode != 0
    assert "No module named 'trainer'" in proc.stderr


@pytest.mark.parametrize("name", HELP_SCRIPTS)
def test_help_runs_by_path(results: dict[str, subprocess.CompletedProcess[str]], name: str) -> None:
    proc = results[f"help:{name}"]
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "usage:" in proc.stdout


@pytest.mark.parametrize("label", sorted(IMPORT_CHECKS))
def test_deferred_imports_resolve_under_path_execution(
    results: dict[str, subprocess.CompletedProcess[str]], label: str
) -> None:
    proc = results[f"import:{label}"]
    assert proc.returncode == 0, proc.stderr[-2000:]
