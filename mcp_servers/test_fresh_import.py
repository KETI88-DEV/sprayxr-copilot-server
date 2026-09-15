"""MCP 서버가 새 프로세스에서 뜨는지 확인한다.

같은 pytest 세션 안의 다른 테스트가 trainer/scripts를 sys.path에 얹으면
형제 모듈의 최상위 이름 import가 우연히 풀려 결함이 가려진다. 그래서
빈 PYTHONPATH로 띄운 별도 프로세스에서 운영과 같은 방식으로 import한다.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_main_imports_in_fresh_process_without_pythonpath() -> None:
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    env["PYTHONPATH"] = ""
    completed = subprocess.run(
        [sys.executable, "-c", "import mcp_servers.main"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
