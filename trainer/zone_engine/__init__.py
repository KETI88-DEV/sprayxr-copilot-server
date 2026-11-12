"""Persistent corrected Zone thickness engine.

The package is intentionally independent from the legacy coating evaluator.  Its
public boundary is the HTTP service in :mod:`trainer.zone_engine.service`.
"""

from .engine import CorrectedZoneEngine

__all__ = ["CorrectedZoneEngine"]
