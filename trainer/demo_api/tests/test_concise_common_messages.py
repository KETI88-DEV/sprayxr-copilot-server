"""Keep essential facts when compacting common source and safety messages."""
from trainer.demo_api import coating_query as coating
from trainer.demo_api import safety_guard as safety

def test_high_pressure_advice_is_brief_but_keeps_prohibition_hazard_and_actions():
    text=safety.HIGH_PRESSURE_ANSWER
    assert len(text)<=95
    for value in ("손", "막", "고압", "피부", "눈", "멈추", "압력", "지침", "관리자"):
        assert value in text
    assert safety.CLASSES["hand_tip"].answer==text
    for language in ("en","vi","id","th","ne","my","uz","si","ru","zh","km"):
        assert safety.fixed_answer_translation(text,language)
    assert safety.fixed_answer_translation("다른 답변","vi") is None

def test_generic_safety_preserves_stop_release_and_verification_in_a_short_answer():
    text=safety.CONSERVATIVE_ANSWER
    assert len(text)<=65
    for value in ("하지 마세요", "장비", "압력", "지침", "관리자"):
        assert value in text

def test_final_coating_note_is_short_but_distinguishes_the_playback_instant():
    assert len(coating._VIEWER_NOTE)<=23
    assert "최종" in coating._VIEWER_NOTE and "재생 시점" in coating._VIEWER_NOTE

def test_missing_target_assessment_reasons_remain_specific_and_compact():
    for message in coating._TARGET_MISSING.values():
        assert len(message)<=40 and "판정" in message
    assert "원본" in coating._TARGET_MISSING["height_map_unavailable"]
    assert "크기" in coating._TARGET_MISSING["height_map_size_mismatch"]
    assert "Zone" in coating._TARGET_MISSING["zone_reference_unavailable"]
