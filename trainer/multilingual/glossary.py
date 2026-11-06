"""Source-attested coating terms shared by every translation path.

Matching ignores spacing, case, width and ё/е because documents and the model
write one term several ways (건도막 두께 / 건조도막두께, плёнки / пленки).
Abbreviations such as DFT are never listed: a translation may keep them.
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Final

GLOSSARY: Final[dict[str, Any]] = json.loads(Path(__file__).with_name("glossary.json").read_text(encoding="utf-8"))


def _key(text: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text)).casefold().replace("ё", "е")


def forms(term: dict[str, Any], language: str) -> list[str]:
    """The rendering first, then other attested spellings; empty when the language has none."""
    rendering = term["renderings"].get(language)
    return [rendering, *term.get("variants", {}).get(language, [])] if rendering else []


def _found(term: dict[str, Any], language: str, key: str) -> str | None:
    return next((form for form in forms(term, language) if _key(form) in key), None)


def term_pairs(text: str, source: str, target: str) -> list[dict[str, str]]:
    """Terms the text uses that have a target rendering, as translator hints."""
    key = _key(text)
    pairs = []
    for term in GLOSSARY["terms"]:
        found = _found(term, source, key)
        if found and term["renderings"].get(target):
            pairs.append({"source": found, "target": term["renderings"][target]})
    return pairs


def missing_terms(source_text: str, translated: str, source: str, target: str) -> list[str]:
    """Target renderings a faithful translation of source_text must contain but does not."""
    source_key, target_key = _key(source_text), _key(translated)
    return [
        term["renderings"][target]
        for term in GLOSSARY["terms"]
        if term["renderings"].get(target) and _found(term, source, source_key) and not _found(term, target, target_key)
    ]
