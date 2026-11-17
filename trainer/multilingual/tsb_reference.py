"""Bounded optional TSB term hints, separate from the source-attested glossary."""
from __future__ import annotations

import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Mapping, Sequence

from trainer.multilingual.tsb_import import LANGUAGE_POLICY, normalized


@lru_cache(maxsize=1)
def load_corpus() -> dict:
    return json.loads(Path(__file__).with_name('tsb_reference.json').read_text(encoding='utf-8'))


def reference_version() -> str:
    return str(load_corpus()['version'])


def _match(term: str, text: str, language: str):
    term = unicodedata.normalize('NFKC',term).strip()
    pattern = r'\s*'.join(re.escape(word) for word in term.split())
    if language in ('en','vi','id','uz'):
        pattern = r'(?<![A-Za-z])' + pattern + r'(?![A-Za-z])'
    return re.search(pattern,unicodedata.normalize('NFKC',text),re.I)


def reference_hints(text: str, source: str, target: str, *,
                    protected_pairs: Sequence[Mapping[str,str]] = (), limit: int = 6) -> list[dict]:
    if source==target or source in LANGUAGE_POLICY or target in LANGUAGE_POLICY or limit<=0:
        return []
    matches = []
    for term in load_corpus()['reference_terms']:
        forms = {'ko':term['ko'],**term['translations']}
        if not forms.get(source) or not forms.get(target):
            continue
        found = _match(forms[source],text,source)
        if found is None:
            continue
        key = normalized(forms[source])
        if any(key in normalized(pair['source']) or normalized(pair['source']) in key for pair in protected_pairs):
            continue
        matches.append((found,term,forms))
    hints,spans = [],[]
    for found,term,forms in sorted(matches,key=lambda item:-(item[0].end()-item[0].start())):
        if any(found.start()<end and found.end()>start for start,end in spans):
            continue
        if any(item['source']==found[0] and item['target']==forms[target] for item in hints):
            continue
        hints.append({'source':found[0],'target':forms[target],'status':'user_provided_reference',
                      'provenance':term['source']})
        spans.append(found.span())
        if len(hints)>=min(limit,6):
            break
    return hints
