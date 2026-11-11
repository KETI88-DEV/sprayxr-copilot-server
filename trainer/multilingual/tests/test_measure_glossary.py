import json

from trainer.multilingual import measure_glossary as mg


def test_every_case_carries_a_glossary_term_except_non_dft_uzbek():
    for source, target, text in mg.cases():
        assert mg.term_pairs(text, source, target) or target == "uz", (source, target, text)


def test_glossary_toggle_changes_only_the_prompt_and_scores_kept_terms():
    prompts = []

    def fake(messages, _regex):
        prompts.append(messages[-1]["content"])
        return json.dumps({"text": "Check the surface before airless spray."})

    on = mg.measure(fake, "ko", "en", "에어리스 스프레이 작업 전에 표면처리 상태를 점검하세요.", True)
    off = mg.measure(fake, "ko", "en", "에어리스 스프레이 작업 전에 표면처리 상태를 점검하세요.", False)
    assert '"glossary"' in prompts[0] and '"glossary"' not in prompts[1]
    assert (on["expected_terms"], on["kept_terms"], on["missed"]) == (2, 1, "surface preparation")
    assert mg.summarize([on, off]) == {
        "ko->en off": {"terms_kept": "1/2", "delivered": "1/1"},
        "ko->en on": {"terms_kept": "1/2", "delivered": "0/1"},
    }
