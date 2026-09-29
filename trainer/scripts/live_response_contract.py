"""Knowledge-answer contract for the fast (toolless) route.

``parse_knowledge_answer`` is total: every input, however malformed, comes back as
``[답변] <body>`` optionally followed by ``\\n[근거] <ids>``. There is no pass-through
branch, because a pass-through does nothing in exactly the cases that need repair
(measured: 18.3% of 240 stored v6.1 responses deviate from the trained scaffold).

Invariant the callers rely on: **every ``DOC-...`` id left in ``answer`` is in
``citations``**. Ids the model cited but that were not retrieved for this request are
removed from the prose and reported in ``dropped`` — the answer text and the contract's
``citations`` array can no longer disagree.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Final

logger = logging.getLogger(__name__)

_HEADING_RE: Final[re.Pattern[str]] = re.compile(r"^[ \t]*\[(답변|분석|판정|원인|조치|근거)\][ \t]*:?[ \t]*")
_ANSWER_LABELS: Final[frozenset[str]] = frozenset({"답변", "분석"})
_EVIDENCE_LABEL: Final[str] = "근거"
_ANSWER_HEADING: Final[str] = "[답변]"
_EVIDENCE_HEADING: Final[str] = "[근거]"

# Same shape as ``agent_pipeline._DOC_ID_RE``; both must accept the identical id set.
_DOC_ID_RE: Final[re.Pattern[str]] = re.compile(r"DOC-[a-zA-Z0-9_-]+-C\d+")

_EMPTY_BRACKETS_RE: Final[re.Pattern[str]] = re.compile(r"[(\[]\s*[,;·、]?\s*[)\]]")
_DANGLING_SEP_RE: Final[re.Pattern[str]] = re.compile(r"(?m)[ \t]*[,;](?=[ \t]*(?:[,;)\]]|$))")
_MULTISPACE_RE: Final[re.Pattern[str]] = re.compile(r"[ \t]{2,}")
_BRACKET_INNER_SPACE_RE: Final[re.Pattern[str]] = re.compile(r"(?<=[(\[])[ \t]+|[ \t]+(?=[)\]])")

# Stated as our failure, not as a claim about the documents: when the model produced no
# answer section at all we do not know whether the corpus covers the question.
FAILED_ANSWER: Final[str] = "답변을 생성하지 못했습니다."

_Section = tuple[str, list[str]]


@dataclass(frozen=True)
class KnowledgeAnswer:
    """``answer`` is contract-shaped; ``citations`` is exactly the ids it cites."""

    answer: str
    citations: tuple[str, ...]
    dropped: tuple[str, ...]
    dropped_sections: tuple[str, ...]
    repair_path: str


def _split_with_preamble(response: str) -> tuple[str, list[_Section]]:
    """``(prose before the first heading, sections)``; empty section list means no headings.

    The preamble is kept rather than rejected: sending a response with a preamble down the
    heading-less path would leave ``[분석]`` markers nested inside the answer body.
    """
    preamble: list[str] = []
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
            preamble.append(line)
            continue
        current[1].append(line)

    if current is not None:
        sections.append(current)
    return "\n".join(preamble).strip(), sections


def _section_body(lines: list[str]) -> str:
    head = _HEADING_RE.sub("", lines[0], count=1)
    return "\n".join([head, *lines[1:]]).strip()


def _strip_ids(text: str, ids: Iterable[str]) -> str:
    """Removes the given ids and the separators they leave behind."""
    out = text
    for doc_id in ids:
        out = out.replace(doc_id, "")
    out = _EMPTY_BRACKETS_RE.sub("", out)
    out = _DANGLING_SEP_RE.sub("", out)
    out = _MULTISPACE_RE.sub(" ", out)
    out = _BRACKET_INNER_SPACE_RE.sub("", out)
    return "\n".join(line.rstrip() for line in out.split("\n")).strip()


def _ordered_unique(ids: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(ids))


def parse_knowledge_answer(raw: str, retrieved: Sequence[str]) -> KnowledgeAnswer:
    """Normalizes a toolless knowledge answer and reconciles its citations with ``retrieved``.

    ``retrieved`` is the chunk id list actually searched for this request; an empty
    sequence (``use_rag='N'`` or no hits) removes every id from the prose, so a request
    that was given no documents cannot answer with document ids.
    """
    text = raw.strip()
    if not text:
        logger.warning("knowledge answer empty, repair_path=empty")
        return KnowledgeAnswer(
            answer=f"{_ANSWER_HEADING} {FAILED_ANSWER}",
            citations=(),
            dropped=(),
            dropped_sections=(),
            repair_path="empty",
        )

    preamble, sections = _split_with_preamble(text)
    dropped_sections: list[str] = []
    evidence_text = ""

    if not sections:
        body = text
        repair_path = "fallback"
    else:
        answer_parts: list[str] = [preamble] if preamble else []
        evidence_parts: list[str] = []
        for label, lines in sections:
            part = _section_body(lines)
            if label == _EVIDENCE_LABEL:
                evidence_parts.append(part)
            elif label in _ANSWER_LABELS:
                answer_parts.append(part)
            else:
                dropped_sections.append(label)
        evidence_text = "\n".join(part for part in evidence_parts if part)
        body = "\n".join(part for part in answer_parts if part)
        if not answer_parts:
            # Only a verdict/cause/action reached us. The fast route must not publish a
            # verdict it has no measurements for, and there is nothing else to show.
            logger.warning(
                "knowledge answer had no answer section, labels=%s",
                ",".join(label for label, _ in sections),
            )
            return KnowledgeAnswer(
                answer=f"{_ANSWER_HEADING} {FAILED_ANSWER}",
                citations=(),
                dropped=tuple(_ordered_unique(_DOC_ID_RE.findall(text))),
                dropped_sections=tuple(dropped_sections),
                repair_path="no_answer_section",
            )
        repair_path = "sections"

    retrieved_set = set(retrieved)
    cited = _ordered_unique(_DOC_ID_RE.findall(evidence_text) + _DOC_ID_RE.findall(body))
    validated = [doc_id for doc_id in cited if doc_id in retrieved_set]
    dropped = [doc_id for doc_id in cited if doc_id not in retrieved_set]

    body = _strip_ids(body, dropped)
    if not body:
        body = FAILED_ANSWER
        repair_path = "body_empty"

    answer = f"{_ANSWER_HEADING} {body}"
    if validated:
        answer = f"{answer}\n{_EVIDENCE_HEADING} {', '.join(validated)}"

    if dropped:
        logger.warning(
            "knowledge answer cited outside the evidence, dropped=%s retrieved=%d",
            ",".join(dropped),
            len(retrieved_set),
        )
    if dropped_sections:
        logger.info("knowledge answer dropped unsupported sections: %s", ",".join(dropped_sections))

    return KnowledgeAnswer(
        answer=answer,
        citations=tuple(validated),
        dropped=tuple(dropped),
        dropped_sections=tuple(dropped_sections),
        repair_path=repair_path,
    )


def document_ids(text: str) -> list[str]:
    """Document ids appearing in ``text``, in order, deduplicated."""
    return _ordered_unique(_DOC_ID_RE.findall(text))


def is_contract_shaped(answer: str) -> bool:
    """Whether ``answer`` is exactly ``[답변] <body>`` plus at most one trailing ``[근거]`` line.

    This is the measurement instrument for the structure-conformance gate. It judges the
    published shape only; whether the ids are grounded is ``parse_knowledge_answer``'s job.
    """
    lines = answer.split("\n")
    head = lines[0] if lines else ""
    if not head.startswith(f"{_ANSWER_HEADING} ") or not head[len(_ANSWER_HEADING) + 1 :].strip():
        return False

    evidence_seen = False
    for line in lines[1:]:
        if line.startswith(f"{_EVIDENCE_HEADING} "):
            if evidence_seen:
                return False
            evidence_seen = True
            parts = [part.strip() for part in line[len(_EVIDENCE_HEADING) + 1 :].split(",")]
            if not parts or not all(_DOC_ID_RE.fullmatch(part) for part in parts):
                return False
            continue
        if evidence_seen:
            # Nothing may follow the evidence line; it closes the answer.
            return False
        if _HEADING_RE.match(line):
            return False
    return True


def retrieved_chunk_ids(rag_chunks: Sequence[object]) -> list[str]:
    """Chunk ids from an ``AgentPipeline.run`` result's ``rag_chunks``, order preserved."""
    ids: list[str] = []
    for chunk in rag_chunks:
        if not isinstance(chunk, dict):
            continue
        chunk_id = str(chunk.get("chunk_id", "")).strip()
        if chunk_id:
            ids.append(chunk_id)
    return _ordered_unique(ids)
