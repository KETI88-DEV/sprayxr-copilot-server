from __future__ import annotations

import logging
import re
from typing import Final

logger = logging.getLogger(__name__)

_HEADING_RE: Final[re.Pattern[str]] = re.compile(r"^[ \t]*\[(답변|분석|판정|원인|조치|근거)\][ \t]*:?[ \t]*")
_ANSWER_LABELS: Final[frozenset[str]] = frozenset({"답변", "분석"})
_UNSUPPORTED_LABELS: Final[frozenset[str]] = frozenset({"판정", "원인", "조치"})
_VERDICT_LABEL: Final[str] = "판정"
_EVIDENCE_LABEL: Final[str] = "근거"

_Section = tuple[str, list[str]]


def _split_sections(response: str) -> list[_Section] | None:
    sections: list[_Section] = []
    current: _Section | None = None

    for line in response.split("\n"):
        match = _HEADING_RE.match(line)
        if match is not None:
            if current is not None:
                sections.append(current)
            current = (match.group(1), [line])
            continue
        if current is None:
            if line.strip():
                return None
            continue
        current[1].append(line)

    if current is not None:
        sections.append(current)
    return sections or None


def _is_canonical_diagnostic_scaffold(labels: list[str]) -> bool:
    if len(set(labels)) != len(labels):
        return False
    if labels[0] not in _ANSWER_LABELS:
        return False
    return _VERDICT_LABEL in labels and labels[-1] == _EVIDENCE_LABEL


def normalize_fast_knowledge_response(response: str) -> str:
    """Strip verdict/cause/action from a toolless knowledge answer; any other shape passes through."""
    sections = _split_sections(response)
    if sections is None:
        return response

    labels = [label for label, _ in sections]
    if not _is_canonical_diagnostic_scaffold(labels):
        return response

    kept: list[str] = []
    dropped: list[str] = []
    for label, lines in sections:
        if label in _UNSUPPORTED_LABELS:
            dropped.append(label)
            continue
        block = list(lines)
        if label == "분석":
            block[0] = block[0].replace("[분석]", "[답변]", 1)
        kept.extend(block)

    logger.info("fast answer normalized, dropped unsupported sections: %s", ",".join(dropped))
    return "\n".join(kept).strip()
