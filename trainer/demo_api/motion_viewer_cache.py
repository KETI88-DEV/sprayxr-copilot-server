"""Opaque, short-lived source revisions for payloads retained by the browser.

This registry holds file stamps, never playback payloads. A server restart or
registry eviction safely forces a reload rather than claiming a server cache hit.
"""
from collections import OrderedDict
from pathlib import Path
from threading import Lock
from time import monotonic
from uuid import uuid4


def file_revision(path: Path) -> tuple[int, int, int, int]:
    """The same observable revision for validation and computed cache keys."""
    stat = Path(path).stat()
    return stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size, stat.st_ino


class PayloadRevisions:
    def __init__(self, *, max_entries=64, ttl_seconds=300, now=monotonic):
        self._entries = OrderedDict()
        self._lock = Lock()
        self._max_entries, self._ttl, self._now = max_entries, ttl_seconds, now

    @staticmethod
    def stamp(paths):
        result = []
        for path in sorted({Path(item).resolve() for item in paths}):
            try:
                value = file_revision(path)
            except OSError:
                value = None
            result.append((path, value))
        return tuple(result)

    def register(self, stamp):
        token = uuid4().hex
        with self._lock:
            self._entries[token] = (self._now(), stamp)
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)
        return token

    def valid(self, token):
        with self._lock:
            entry = self._entries.get(token)
            if entry is None:
                return False
            created, stamp = entry
            if self._now() - created >= self._ttl or self.stamp(path for path, _ in stamp) != stamp:
                self._entries.pop(token, None)
                return False
            return True
