"""The single generation slot shared by every live entry point.

The in-process HF model and the single-stream model server both serve one
generation at a time. ``AgentPipeline.run_streaming`` (``/api/chat``) and
``POST /api/v1/copilot/query`` acquire this slot inside their worker thread, so
async cancellation cannot free it while an orphaned generation is still running.

It lives in its own module so the copilot router can share it without importing
torch through ``agent_pipeline``.
"""

from __future__ import annotations

import threading
from typing import Final

GENERATION_SLOT: Final[threading.Semaphore] = threading.Semaphore(1)

# Resident quick and detailed servers schedule independently. RAG modules remain shared.
QUICK_GENERATION_SLOT: Final[threading.Semaphore] = threading.Semaphore(2)
RAG_SLOT: Final[threading.Semaphore] = threading.Semaphore(1)
