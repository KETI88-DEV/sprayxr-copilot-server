from __future__ import annotations

import os
import subprocess
import sys


def test_agent_pipeline_disables_transformers_async_loading_on_import() -> None:
    environment = dict(os.environ)
    environment.pop("HF_DEACTIVATE_ASYNC_LOAD", None)

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import os; "
                "import trainer.scripts.agent_pipeline; "
                "print(os.environ.get('HF_DEACTIVATE_ASYNC_LOAD', 'missing'))"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.stdout.strip() == "1"
