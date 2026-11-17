"""교체 스크립트가 맨 처음 부르는 잠금 확인. 자기 잠금이 아니면 아무것도 바꾸기 전에 멈춘다.

    from ops_lock_guard import require_lock   # 같은 폴더에 두거나, 이 함수를 그대로 복사해 쓴다
    require_lock("KURE", "vllm")

셸 스크립트에서는: python3 /home/keti/shipcoat/ops/ops_lock.py check --by KURE --what vllm || exit 1
"""
from __future__ import annotations

import os
import subprocess
import sys


def require_lock(by: str, what: str) -> None:
    script = os.environ.get("OPS_LOCK", "/home/keti/shipcoat/ops/ops_lock.py")
    if subprocess.run([sys.executable, script, "check", "--by", by, "--what", what]).returncode != 0:
        raise SystemExit("운영 잠금을 확인하지 못해 교체를 시작하지 않는다(ops_lock.py status 로 확인)")
