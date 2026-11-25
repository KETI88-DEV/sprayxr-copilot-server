"""Exact reviewed terms for explicit translation questions; never a general answerer.

Sentence and unknown-term requests remain parseable for the existing translator.
Only an exact glossary rendering/variant can produce a direct dictionary answer.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import re
import unicodedata
from typing import Any, Literal

from trainer.multilingual import glossary as verified_glossary
from trainer.multilingual import tsb_reference
from trainer.multilingual.tsb_import import LANGUAGE_POLICY


@dataclass(frozen=True)
class TranslationRequest:
    source_text: str
    target_language: str
    source_language: str | None = None
    require_standard: bool = False


@dataclass(frozen=True)
class GlossaryTranslation:
    answer: str
    source_text: str
    source_language: str | None
    target_language: str
    source_kind: Literal["verified_glossary", "tsb_reference"]
    provenance: dict[str, Any]


_LANGUAGES = {
    "ko": ("한국어", "한국말", "Korean", "tiếng Hàn"),
    "en": ("영어", "English", "tiếng Anh"),
    "vi": ("베트남어", "Vietnamese", "tiếng Việt"),
    "id": ("인도네시아어", "인니어", "Indonesian", "tiếng Indonesia"),
    "th": ("태국어", "Thai", "tiếng Thái"),
    "ne": ("네팔어", "Nepali", "tiếng Nepal"),
    "my": ("미얀마어", "버마어", "Burmese", "tiếng Myanmar"),
    "uz": ("우즈베크어", "우즈벡어", "우즈베크", "우즈벡", "Uzbek", "tiếng Uzbek"),
    "si": ("싱할라어", "Sinhala", "tiếng Sinhala"),
    "ru": ("러시아어", "Russian", "tiếng Nga"),
    "zh": ("중국어", "Chinese", "tiếng Trung"),
    "km": ("크메르어", "캄보디아어", "Khmer", "tiếng Khmer"),
}
_ALIASES = {alias.casefold(): code for code, aliases in _LANGUAGES.items() for alias in aliases}
_LANG = "(?:" + "|".join(
    r"(?<![A-Za-z])" + re.escape(alias) + r"(?![A-Za-z])" if alias.isascii() else re.escape(alias)
    for alias in sorted(_ALIASES, key=len, reverse=True))
_LANG += r"|(?<![A-Za-z])(?:" + "|".join(_LANGUAGES) + r")(?:[-_][A-Za-z0-9]{2,8}){0,2}(?![A-Za-z0-9_-]))"
_END = r"\s*[.!?。！？]*\s*\Z"
_QUESTION = r"(?:뭐(?:야|예요|에요)|뭐라고\s*(?:해(?:요)?|하나요|하죠)|어떻게\s*(?:말해|말하나요|표현해)|알려\s*(?:줘|주세요))"
_ACTION = r"(?:하면|번역(?:해\s*(?:줘|주세요)|해|하라|해\s*줄래|해\s*주시겠어요|\s*바람)?|" + _QUESTION + ")"
_WHAT_IS = rf"what\s+is\s+(?P<subject>.+?)\s+in\s+(?P<language>{_LANG}){_END}"
_LABEL_ONLY_NOTE = (r"(?:\s*[.!。]\s*(?:말\s*붙이지\s*말고|설명(?:은\s*빼고|\s*없이)|"
                    r"(?:항목명|용어|라벨|표기)\s*만))?")
_QUOTES = {'"': '"', "'": "'", "“": "”", "‘": "’", "`": "`", "«": "»", "「": "」", "『": "』"}
_QUOTED = "(?P<subject>" + "|".join(
    rf"{re.escape(start)}[^{re.escape(end)}\n]+{re.escape(end)}" for start, end in _QUOTES.items()
) + ")"
_QUOTE_MARKS = re.escape("".join(_QUOTES))
_QUOTE_QUALIFIER = (rf"(?:[^{_QUOTE_MARKS}\n.!?]{{1,44}}?"
                    r"(?:의|인|문구|문장|라벨|표시|안내|에\s*(?:쓸|넣을|표시할)|용어(?:집|표)에서)\s*)?")
_TERM_QUALIFIER = r"(?:(?:도장\s*현장\s*)?(?:현장\s*)?(?:용어표|용어집)(?:를\s*기준으로|에\s*따라|로)\s*)?"
_SHORT_LABEL = r"(?:(?:한\s*)?(?:용어|단어(?:\s*/\s*구)?|구절|표기|명칭)\s*만)"
_PATTERNS = tuple((re.compile(pattern, re.IGNORECASE | re.DOTALL), intent) for pattern, intent in (
    (rf"{_QUOTE_QUALIFIER}{_QUOTED}(?:을|를)?\s*(?P<language>{_LANG})\s*(?:도장\s*)?"
     rf"(?:규격|표준|공식)\s*용어(?:로|로만)\s*(?:정확히\s*)?{_ACTION}{_END}", "standard"),
    (rf"{_QUOTE_QUALIFIER}{_QUOTED}(?:을|를)?\s*(?P<language>{_LANG})\s*(?:으로|로)\s*"
     rf"(?:[,，]\s*{_SHORT_LABEL}(?:\s*{_QUESTION})?|번역해\s*(?:줘|주세요|줄래|주시겠어요))"
     rf"{_LABEL_ONLY_NOTE}{_END}", "explicit"),
    (rf"{_QUOTE_QUALIFIER}{_QUOTED}(?:을|를)?\s*(?P<language>{_LANG})\s*(?:으로|로)?\s*"
     rf"{_SHORT_LABEL}\s*(?:{_QUESTION})?{_LABEL_ONLY_NOTE}{_END}", "label"),
    (rf"{_QUOTED}(?:이?라고)\s*(?:써|쓰여|적혀)\s*있는\s*(?:메뉴|버튼|라벨|문구)(?:가|를|의)?\s*"
     rf"(?:무슨\s*뜻인지\s*)?(?P<language>{_LANG})\s*(?:으로|로)\s*(?:짧게\s*)?"
     rf"(?:바꿔\s*쓰고\s*싶어(?:요)?|바꿔\s*(?:줘|주세요)|옮겨\s*(?:줘|주세요)){_END}", "explicit"),
    (rf"{_TERM_QUALIFIER}(?P<subject>.+?)\s*[,，]?\s*(?P<language>{_LANG})\s*(?:으로|로)?\s*"
     rf"{_SHORT_LABEL}\s*(?:{_QUESTION})?{_END}", "label"),
    (rf"(?:(?:vui\s+lòng|xin(?:\s+hãy)?|hãy)\s+)?dịch(?:\s+câu|\s+cụm\s+từ)?\s*{_QUOTED}\s*"
     rf"sang\s+(?P<language>{_LANG})(?:\s*[,，]\s*chỉ\s*(?:cần\s*)?(?:câu|bản)\s+dịch)?{_END}", "explicit"),
    (rf"(?:please\s+)?translate\s+(?P<subject>.+?)\s+from\s+(?P<source_language>{_LANG})\s+(?:into|to)\s+(?P<language>{_LANG}){_END}", "explicit"),
    (rf"(?:please\s+)?translate\s+(?P<subject>.+?)\s+(?:into|to)\s+(?P<language>{_LANG}){_END}", "explicit"),
    (_WHAT_IS, "weak"),
    (rf"how\s+do\s+you\s+say\s+(?P<subject>.+?)\s+in\s+(?P<language>{_LANG}){_END}", "explicit"),
    (rf"(?P<subject>.+?)\s*(?P<language>{_LANG})\s*(?:으로|로)\s*(?P<action>{_ACTION}){_END}", "action"),
    (rf"(?P<language>{_LANG})\s*(?:으로|로)\s*(?P<subject>.+?)\s*(?P<action>{_ACTION}){_END}", "action"),
    (rf"(?P<subject>.+?)\s*(?P<language>{_LANG})\s*(?:표현|번역|명칭|단어)(?:은|는|이|가)?\s*(?:{_QUESTION})?{_END}", "explicit"),
    (rf"(?P<subject>.+?)\s*(?P<language>{_LANG})\s*표기(?:은|는|이|가|을|를)?"
     rf"\s*(?:만\s*)?(?:{_QUESTION})?{_LABEL_ONLY_NOTE}{_END}", "label"),
    (rf"(?P<subject>.+?)\s*(?P<language>{_LANG})\s*(?:으로|로){_END}", "bare"),
))
_WEAK_ACTIONS = {"알려줘", "알려주세요", "뭐야", "뭐예요", "뭐에요"}
_MULTIPLE_TARGETS = re.compile(rf"(?<![A-Za-z]){_LANG}\s*(?:와|과|및|[,/])\s*\Z", re.IGNORECASE)
_FOLLOWUP = re.compile(
    rf"(?:아\s*[,，]\s*)?(?:(?:그럼|그러면|그렇다면|이번에는|이번엔)\s*)?"
    rf"(?P<language>{_LANG})\s*(?:(?:으로|로)\s*(?:{_ACTION})?|표기\s*(?:으로|로)?\s*만){_END}",
    re.IGNORECASE,
)
_APOSTROPHES = str.maketrans("‘’ʻʼ", "''''")


def _language(value: str) -> str | None:
    key = value.casefold()
    if key in _ALIASES:
        return _ALIASES[key]
    code = key.replace("_", "-").split("-", 1)[0]
    return code if code in _LANGUAGES else None


def _subject(value: str) -> str | None:
    text = value.strip()
    if not text:
        return None
    if text[0] in _QUOTES:
        # Remove only a request particle outside the closing quotation mark.
        quoted = text
        if len(quoted) > 1 and quoted[-1] in "을를은는의":
            quoted = quoted[:-1].rstrip()
        if len(quoted) < 2 or quoted[-1] != _QUOTES[quoted[0]]:
            return None
        text = quoted[1:-1]
    elif text[-1] in _QUOTES.values():
        return None
    elif len(text) > 1 and text[-1] in "을를은는의":
        # A Korean word can itself end in 의/을/은. Preserve the whole input
        # first; shorten only when the remaining prefix is an actual reviewed
        # form, not a guessed Korean stem (주의 -> 주, 마을 -> 마).
        prefix = text[:-1].rstrip()
        if not _registered_form(text) and _registered_form(prefix):
            text = prefix
    return text if text.strip() else None


def _registered_form(text: str) -> bool:
    key = _key(text)
    for term in verified_glossary.GLOSSARY["terms"]:
        if any(_key(form) == key for language in term["renderings"]
               for form in verified_glossary.forms(term, language)):
            return True
    for term in tsb_reference.load_corpus().get("reference_terms", []):
        if term.get("status", "user_provided_reference") != "user_provided_reference":
            continue
        renderings = {"ko": term["ko"], **term["translations"]}
        if any(language not in LANGUAGE_POLICY and _key(form) == key for language, form in renderings.items()):
            return True
    return False


def _explicit_request(prompt: str) -> TranslationRequest | None:
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 8000 or "\x00" in prompt:
        return None
    for pattern, intent in _PATTERNS:
        found = pattern.fullmatch(prompt.strip())
        if found is None or _MULTIPLE_TARGETS.search(found["subject"]):
            continue
        text = _subject(found["subject"])
        target = _language(found["language"])
        source = found.groupdict().get("source_language")
        if text is not None and target is not None:
            action = re.sub(r"\s+", "", found.groupdict().get("action") or "")
            if (intent in {"bare", "weak", "label"} or action in _WEAK_ACTIONS) and not (
                found["subject"].lstrip()[0] in _QUOTES or _registered_form(text)
            ):
                continue
            return TranslationRequest(text, target, _language(source) if source else None,
                                      require_standard=intent == "standard")
    return None


_OUTPUT_ACTION = (r"(?:(?:설명|답변|대답|표현|작성)(?:해\s*(?:줘|주세요)|해\s*줄래|하세요)|"
                  r"알려\s*(?:줘|주세요)|써\s*(?:줘|주세요)|부탁(?:해(?:요)?|드려요|드립니다|합니다))")
_OUTPUT_MODIFIERS = r"(?:(?:좀|간단히|자세히|짧게|만)\s*)*"
_OUTPUT_SUFFIX = re.compile(rf"(?P<language>{_LANG})\s*(?:으로|로)\s*{_OUTPUT_MODIFIERS}{_OUTPUT_ACTION}{_END}", re.IGNORECASE)
_OUTPUT_PREFIX = re.compile(rf"(?P<language>{_LANG})\s*(?:으로|로)\s*(?P<question>.+?)\s*{_OUTPUT_ACTION}{_END}", re.IGNORECASE | re.DOTALL)
_OUTPUT_ENGLISH = re.compile(rf"(?:please\s+)?(?:answer|respond|explain)(?:\s+.+?)?\s+in\s+(?P<language>{_LANG}){_END}", re.IGNORECASE | re.DOTALL)
_OUTPUT_WHAT_IS = re.compile(_WHAT_IS, re.IGNORECASE | re.DOTALL)
_SOURCE_DESCRIPTOR = re.compile(r"^(?:된|쓰인|쓴|말한|말하는|적힌|작성된|작성한|작성하는|만들어진|기록된|설명한|설명하는|대답한|진행한|진행하는)(?:\s|$)")
_DETACHED_OUTPUT_PREFERENCE = re.compile(
    rf"(?P<question>.+?[?!.])\s+(?:(?:답변|대답|설명)만\s+)?(?P<language>{_LANG})\s*(?:으로|로)\s*"
    rf"{_OUTPUT_MODIFIERS}{_OUTPUT_ACTION}{_END}", re.IGNORECASE | re.DOTALL,
)


def requested_response_language(prompt: str) -> str | None:
    """Report a supported explicit output-language instruction, without editing it.

    This is separate from translation intent: an instruction to explain a coating
    topic in English must keep the original knowledge question on the RAG path.
    Mere language mentions, quoted instructions and hypothetical endings do not
    supply an output-language override.
    """
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 8000 or "\x00" in prompt:
        return None
    text = prompt.strip()
    found = _OUTPUT_SUFFIX.search(text) or _OUTPUT_ENGLISH.fullmatch(text)
    if found is not None:
        return _language(found["language"])
    found = _OUTPUT_PREFIX.fullmatch(text)
    if found is not None and not _SOURCE_DESCRIPTOR.match(found["question"].lstrip()):
        return _language(found["language"])
    found = _OUTPUT_WHAT_IS.fullmatch(text)
    if found is not None and _explicit_request(text) is None:
        # Keep a cause/difference/recommendation question on its normal route.
        # Exact or quoted terms remain translation requests in _explicit_request.
        return _language(found["language"])
    return None


def question_without_output_preference(prompt: str) -> str:
    """Remove only a detached instruction about how to write the answer."""
    preference = _DETACHED_OUTPUT_PREFERENCE.fullmatch(prompt.strip())
    return preference["question"] if preference is not None else prompt


def parse_translation_request(
    prompt: str, *, recent_turns: Sequence[Mapping[str, object]] = (),
) -> TranslationRequest | None:
    """Parse an explicit target, optionally resolving a target-only follow-up.

    History must be scoped by the caller to the same user/session/conversation
    and ordered oldest first. Only user source text is reused; assistant answers
    are never dictionary evidence. A non-translation user turn ends the topic.
    No language detection occurs. Sentence callers must apply the parsed target
    to translation/output checks, even if the surrounding Query defaults to ko.
    """
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 8000 or "\x00" in prompt:
        return None
    followup = _FOLLOWUP.fullmatch(prompt.strip())
    if followup is None:
        return _explicit_request(prompt)
    target = _language(followup["language"])
    if target is None or not isinstance(recent_turns, Sequence) or isinstance(recent_turns, (str, bytes)):
        return None
    for turn in reversed(recent_turns[-20:]):
        if not isinstance(turn, Mapping) or turn.get("role") != "user":
            continue
        previous = turn.get("content")
        if not isinstance(previous, str) or len(previous) > 8000 or "\x00" in previous:
            return None
        if _FOLLOWUP.fullmatch(previous.strip()):
            continue
        request = _explicit_request(previous)
        if request is None:
            return None
        return TranslationRequest(request.source_text, target, request.source_language, request.require_standard)
    return None


def unverified_standard_answer(request: TranslationRequest) -> str:
    language_name = _LANGUAGES[request.target_language][0]
    return (f"‘{request.source_text}’에 해당하는 검증된 {language_name} 규격 용어를 확인할 수 없습니다. "
            "현장 표현의 뜻이나 해당 규격 문서를 알려 주세요.")


def _key(text: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text)).casefold().translate(_APOSTROPHES).replace("ё", "е")


def _result(request: TranslationRequest, candidates: list[dict],
            kind: Literal["verified_glossary", "tsb_reference"]) -> GlossaryTranslation | None:
    # An identical foreign form may belong to different concepts. Never pick an
    # arbitrary target rendering when their reviewed translations disagree.
    if not candidates or len({_key(item["answer"]) for item in candidates}) != 1:
        return None
    chosen = candidates[0]
    languages = sorted({item["source_language"] for item in candidates})
    provenance = {**chosen["provenance"], "matched_source_languages": languages}
    return GlossaryTranslation(chosen["answer"], request.source_text,
                               languages[0] if len(languages) == 1 else None,
                               request.target_language, kind, provenance)


def _documents(source: str | None, target: str | None) -> dict[str, str]:
    cited = set(re.findall(r"\b[A-Z]+\d+\b", (source or "") + "," + (target or "")))
    documents = verified_glossary.GLOSSARY.get("sources", {})
    return {code: documents[code] for code in sorted(cited) if code in documents}


def lookup_translation_question(
    prompt: str, *, recent_turns: Sequence[Mapping[str, object]] = (),
) -> GlossaryTranslation | None:
    """Return a direct answer only for an exact, available reviewed term pair.

    A None result never grants dictionary authority to an LLM answer. The caller
    can parse the request separately and use its normal translation path instead.
    No archived raw CSV rows or quarantined cells are searched here.
    """
    request = parse_translation_request(prompt, recent_turns=recent_turns)
    if request is None:
        return None
    key = _key(request.source_text)
    candidates = []
    for term in verified_glossary.GLOSSARY["terms"]:
        answer = term["renderings"].get(request.target_language)
        if not answer:
            continue
        for language in term["renderings"]:
            if request.source_language and language != request.source_language:
                continue
            if any(_key(form) == key for form in verified_glossary.forms(term, language)):
                source_attestation = term.get("sources", {}).get(language)
                target_attestation = term.get("sources", {}).get(request.target_language)
                candidates.append({"answer": answer, "source_language": language, "provenance": {
                    "term": term["term"], "version": verified_glossary.GLOSSARY["version"],
                    "source_attestation": source_attestation,
                    "target_attestation": target_attestation,
                    "documents": _documents(source_attestation, target_attestation),
                }})
    if candidates:
        return _result(request, candidates, "verified_glossary")
    # The policy concerns CSV evidence only: Russian still has verified glossary
    # renderings above despite having no source column in the user-provided CSVs.
    if request.require_standard or request.target_language in LANGUAGE_POLICY:
        return None
    corpus = tsb_reference.load_corpus()
    for term in corpus.get("reference_terms", []):
        if term.get("status", "user_provided_reference") != "user_provided_reference":
            continue
        renderings = {"ko": term["ko"], **term["translations"]}
        answer = renderings.get(request.target_language)
        if not answer:
            continue
        for language, form in renderings.items():
            if language in LANGUAGE_POLICY or (request.source_language and language != request.source_language):
                continue
            if _key(form) == key:
                candidates.append({"answer": answer, "source_language": language, "provenance": {
                    **term["source"], "version": corpus["version"], "status": "user_provided_reference",
                }})
    return _result(request, candidates, "tsb_reference")
