"""Bounded process-local reuse for identical quick inference and validated translations."""
from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import OrderedDict
from collections.abc import Callable


def inference_key(*parts: object) -> str:
    return hashlib.sha256(json.dumps(parts, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


class TextCache:
    """Process-local LRU with fixed TTL; keys are hashes, never raw prompts."""
    def __init__(self, *, capacity: int, ttl: float, clock: Callable[[], float] = time.monotonic):
        self.capacity, self.ttl, self.clock = capacity, ttl, clock
        self._items: OrderedDict[str, tuple[float, str]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> str | None:
        with self._lock:
            item = self._items.get(key)
            if item is None:
                return None
            if item[0] <= self.clock():
                del self._items[key]
                return None
            self._items.move_to_end(key)
            return item[1]

    def put(self, key: str, text: str) -> None:
        if not text or len(text) > 24000:
            return
        with self._lock:
            self._items[key] = (self.clock() + self.ttl, text)
            self._items.move_to_end(key)
            while len(self._items) > self.capacity:
                self._items.popitem(last=False)
