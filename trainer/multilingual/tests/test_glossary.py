"""The shared glossary finds terms however they are spaced and catches translations that drop them."""
from __future__ import annotations

import re

import pytest

from trainer.multilingual import glossary
from trainer.multilingual.glossary import GLOSSARY, missing_terms, term_pairs


@pytest.mark.parametrize("answer", ["건도막 두께를 확인하세요.", "건도막두께를 확인하세요.", "건조도막두께를 확인하세요."])
def test_korean_spellings_from_the_corpus_and_the_tds_trigger_the_term(answer: str) -> None:
    assert [pair["target"] for pair in term_pairs(answer, "ko", "en")] == ["dry film thickness"]


def test_spacing_does_not_hide_a_term() -> None:
    assert term_pairs("표면 처리 후 도장하세요.", "ko", "en") == [{"source": "표면처리", "target": "surface preparation"}]


def test_a_translation_without_the_rendering_is_reported_missing() -> None:
    source = "표면 처리 후 건도막 두께를 확인하세요."
    assert missing_terms(source, "After surface preparation, check the film.", "ko", "en") == ["dry film thickness"]
    assert missing_terms(source, "After Surface Preparation, check the dry film  thickness.", "ko", "en") == []


def test_variants_and_cyrillic_yo_count_and_a_missing_rendering_skips_the_term(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(glossary, "GLOSSARY", {"terms": [{
        "term": "DFT", "renderings": {"ko": "건도막 두께", "ru": "толщина сухой плёнки"},
        "variants": {"ru": ["толщины сухой плёнки"]},
    }]})
    assert glossary.term_pairs("건도막 두께", "ko", "en") == []
    assert glossary.missing_terms("건도막 두께", "Значение толщины сухой пленки 80 мкм.", "ko", "ru") == []
    assert glossary.missing_terms("건도막 두께", "Проверьте толщину сухой пленки.", "ko", "ru") == ["толщина сухой плёнки"]


def test_russian_case_forms_of_an_attested_term_count() -> None:
    # Live 27B output (2026-11-06) that was wrongly rejected before the case forms were listed.
    source = "에어리스 스프레이 도장 시 주의사항은 다음과 같습니다:"
    assert missing_terms(source, "При безвоздушном распылении следует соблюдать следующие меры предосторожности:", "ko", "ru") == []


def test_every_rendering_cites_a_listed_source_and_no_form_is_an_abbreviation() -> None:
    for term in GLOSSARY["terms"]:
        assert {"ko", "en", "zh"} <= term["renderings"].keys(), term["term"]
        for language in term["renderings"]:
            for reference in re.split(r",\s*", term["sources"][language]):
                assert reference.split(":")[0] in GLOSSARY["sources"], (term["term"], language, reference)
        for language, values in term.get("variants", {}).items():
            assert language in term["renderings"], (term["term"], language)
        for value in [*term["renderings"].values(), *(v for vs in term.get("variants", {}).values() for v in vs)]:
            assert not re.fullmatch(r"[A-Z]{2,5}", value), value
