"""Domain errors translated to stable HTTP responses by the service."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EngineError(Exception):
    reason_code: str
    message: str
    status_code: int = 422

    def __str__(self) -> str:
        return self.message


class ConflictError(EngineError):
    def __init__(self, reason_code: str, message: str) -> None:
        super().__init__(reason_code=reason_code, message=message, status_code=409)


class NotFoundError(EngineError):
    def __init__(self, reason_code: str, message: str) -> None:
        super().__init__(reason_code=reason_code, message=message, status_code=404)
