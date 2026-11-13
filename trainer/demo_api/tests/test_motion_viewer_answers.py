"""모션 뷰어 질문 창의 답 — 2026-11-11 Spark 카나리에서 뷰어 화면으로 재현한 결함을 막는다."""
from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from trainer.demo_api import dialogue, motion_viewer, pose_query, viewer_arm_pose
from trainer.demo_api.copilot_router import create_copilot_router
from trainer.demo_api.sequence_store import SequenceStore
from trainer.demo_api.viewer_context import sign_context, verify_context

ZONES = {
    "1:8": {"map_num": "1", "zone_id": "8", "work_similarity_percent": 81, "distance_mm": 389.4, "reference_distance_mm": 407.8,
            "speed_mms": 2020.0, "reference_speed_mms": 2726.5, "corrections": ["도구를 피도면에서 약 9.9cm 가까이 이동하세요."]},
    "1:5": {"map_num": "1", "zone_id": "5", "work_similarity_percent": 100, "distance_mm": 300.0, "corrections": []},
}
CLAIMS = {"work_similarity_percent": 88, "zones": ZONES, "record_id": "4_1", "scenario_id": "4", "process": "spray"}
PLAYBACK = {"map_num": "1", "zone_id": "8", "source_index": 60, "playhead_seconds": 15.5}
POSE = {"status": "available", "similarity_percent": 70, "zones": {
    "1:8": {"map_num": "1", "zone_id": "8", "similarity_percent": 81, "corrections": ["도구를 피도면에서 약 9.9cm 가까이 이동하세요."]},
    "1:3": {"map_num": "1", "zone_id": "3", "similarity_percent": 40, "corrections": []}},
    "corrections": [{"message": "도구를 피도면에서 약 9.9cm 가까이 이동하세요."}]}
ARM = {
    "status": "available", "average_similarity_percent": 70, "evaluated_sequence_count": 2, "sequence_count": 2,
    "guideline": "분사 2회 중 2회의 자세를 숙련 작업자와 비교했습니다. 평균 유사도 70%.", "zones": {}, "priority_corrections": [],
    "sequences": [
        {"sequence_id": "SEQ-001", "zone_id": "8", "start_seconds": -0.0, "end_seconds": 20.0, "similarity_percent": 60, "confidence": "limited",
         "correction": "왼쪽 손목이 숙련자 평균보다 26 cm 멀리 있었습니다.",
         "corrections": ["왼쪽 손목이 숙련자 평균보다 26 cm 멀리 있었습니다.", "오른쪽 팔꿈치가 숙련자 평균보다 22 cm 가까이 있었습니다."]},
        {"sequence_id": "SEQ-002", "zone_id": "3", "start_seconds": 30.0, "end_seconds": 40.0, "similarity_percent": 80, "confidence": None,
         "correction": None, "corrections": []},
    ],
}


def _browser(value: Any) -> Any:
    """브라우저의 JSON.stringify(JSON.parse(...))가 하는 일: -0.0은 0이 된다."""
    return json.loads(json.dumps(value).replace("-0.0", "0"))


def _playback_context(playback: dict[str, Any] | None = PLAYBACK) -> str:
    return "motion_viewer_playback:" + json.dumps({
        "dialogue_context": CLAIMS, "dialogue_signature": sign_context("dialogue", CLAIMS), "playback_context": playback})


def _pose_context(*, arm: dict[str, Any] | None = ARM) -> str:
    payload = {"analysis_context": POSE, "signature": sign_context("pose_recomm", POSE), "playback_context": PLAYBACK}
    if arm is not None:
        payload.update(arm_pose_context=_browser(arm), arm_pose_signature=sign_context("arm_pose", arm))
    return "motion_viewer_analysis:" + json.dumps(payload)


class _Pipeline:
    """대화 계획 모델이 무엇을 묻든 도포 방향으로 분류한다(Spark 8B가 속도·Zone 질문에 그랬다)."""

    def __init__(self) -> None:
        self.plans: list[str] = []

    def plan_copilot_dialogue(self, prompt: str, **kwargs: object) -> dict[str, str]:
        self.plans.append(prompt)
        return {"kind": "lookup", "intent": "order_direction", "scope": "playback"}

    def converse_copilot(self, prompt: str, **kwargs: object) -> str:
        return "대화 답입니다."


def _ask(pipeline: _Pipeline, prompt: str, *, context: str, task: str = "knowledge_search") -> dict[str, Any]:
    app = FastAPI()
    app.include_router(create_copilot_router(lambda: pipeline, store=SequenceStore(":memory:")))
    response = TestClient(app).post("/api/v1/copilot/query", json={
        "user_id": "u", "session_id": "4_1", "scenario_id": "4", "task_type": task, "prompt": prompt, "context": context})
    assert response.status_code == 200, response.text
    return {**response.json()["data"], "answer_source": response.headers["x-copilot-answer-source"]}


def test_signature_survives_the_browser_turning_negative_zero_into_zero() -> None:
    assert verify_context("arm_pose", _browser(ARM), sign_context("arm_pose", ARM))
    assert viewer_arm_pose.parse_viewer_arm_pose(_pose_context()) is not None  # 예전에는 자세 유사도 답이 조용히 빠졌다


def test_signed_zone_facts_carry_the_record_and_expert_speed() -> None:
    payload = {"pose_evaluation": {"zones": {"1:8": {"map_num": "1", "zone_id": "8", "similarity_percent": 81, "metrics": [], "corrections": []}}},
               "zone_statistics": {"1:8": {"selected_speed_mms": 2020.0, "median_speed_mms": 2726.5, "median_distance_mm": 407.762}}}
    zone = dialogue.viewer_dialogue_context(payload)["zones"]["1:8"]
    assert (zone["speed_mms"], zone["reference_speed_mms"], zone["reference_distance_mm"]) == (2020.0, 2726.5, 407.762)


def test_record_measurement_questions_quote_the_signed_zone_values() -> None:
    speed = dialogue.viewer_measurement_answer(_playback_context(), "지금 속도는 적당해?")
    assert speed is not None and "Map 1 · Zone 8" in speed and "2,020 mm/s" in speed and "2,726 mm/s" in speed and "-26%" in speed
    assert "순간 측정값이 아닙니다" in speed and "이격거리" not in speed
    both = dialogue.viewer_measurement_answer(_playback_context(), "이 시퀀스에서 분사 속도와 거리는 어땠어?")
    assert both is not None and "2,020 mm/s" in both and "389 mm" in both and "408 mm" in both and "µm" not in both
    named = dialogue.viewer_measurement_answer(_playback_context(), "Zone 5 이격거리는 어땠어?")
    assert named is not None and "Zone 5" in named and "300 mm" in named and "전문가" not in named  # 전문가 값이 없으면 견주지 않는다


@pytest.mark.parametrize("prompt", ["적정 이격거리는?", "적정 분사 속도는?", "분사 속도가 너무 빠르면 도막에 어떤 영향이 있어?"])
def test_general_knowledge_questions_are_left_to_the_documents(prompt: str) -> None:
    assert dialogue.viewer_measurement_answer(_playback_context(), prompt) is None


def test_measurement_answer_needs_a_signed_record_and_a_zone() -> None:
    forged = _playback_context().replace('"speed_mms": 2020.0', '"speed_mms": 600.0')
    assert dialogue.viewer_measurement_answer(forged, "지금 속도는 적당해?") is None
    moving = dialogue.viewer_measurement_answer(_playback_context({**PLAYBACK, "map_num": None, "zone_id": None}), "지금 속도는 적당해?")
    assert moving is not None and "분사 구간으로 이동" in moving and "mm/s" not in moving


def test_evaluation_question_in_the_viewer_names_the_lowest_zone_from_the_signed_record() -> None:
    answer = dialogue.viewer_findings_answer(_playback_context())
    assert answer is not None and "Zone 8" in answer and "81%" in answer and "9.9cm" in answer
    assert dialogue.viewer_findings_answer("motion_viewer_playback:{}") is None


def test_findings_answer_adds_one_coating_line_from_the_signed_target() -> None:
    from trainer.demo_api.tests.test_query_question_routing import target_assessment
    plain = dialogue.viewer_findings_answer(_playback_context())
    coating = {"analysis_status": "available", "scope": "final_record", "comparison_basis": "expert_zone_median",
               "zones": [], "target": target_assessment()}
    payload = json.loads(_playback_context().split(":", 1)[1])
    payload.update(coating_context=coating, coating_signature=sign_context("coating_status", coating))
    answer = dialogue.viewer_findings_answer("motion_viewer_playback:" + json.dumps(payload))
    assert answer == f"{plain}\n도막(기본 목표 160µm 기준): 저도막 Zone 1, 과도막 Zone 2."
    payload["coating_context"]["target"]["zones"][0]["grade"] = "normal"  # 서명과 맞지 않으면 붙이지 않는다
    assert dialogue.viewer_findings_answer("motion_viewer_playback:" + json.dumps(payload)) == plain


def test_record_speed_question_in_the_viewer_never_reaches_the_model() -> None:
    pipeline = _Pipeline()
    data = _ask(pipeline, "지금 속도는 적당해?", context=_playback_context())
    assert pipeline.plans == [] and data["retrieval_used"] is False
    assert "2,020 mm/s" in data["answer"] and "방향" not in data["answer"]
    assert data["answer_source"] == "analysis"  # 화면이 채점 결과에 모델 이름을 붙이지 않게 한다


def test_only_a_question_about_direction_gets_the_direction_answer() -> None:
    pipeline = _Pipeline()
    talk = _ask(pipeline, "여기는 어때 보여?", context=_playback_context())
    assert "방향" not in talk["answer"] and talk["answer_source"] == "conversation"
    assert "방향" in _ask(pipeline, "여기 칠하는 방향이 맞는 거야?", context=_playback_context())["answer"]
    assert pipeline.plans == ["여기는 어때 보여?", "여기 칠하는 방향이 맞는 거야?"]


def test_zone_number_question_on_the_pose_tab_is_answered_from_the_signed_zone() -> None:
    pipeline = _Pipeline()
    data = _ask(pipeline, "Zone 3는 어때?", context=_pose_context(), task="pose_recomm")
    assert pipeline.plans == [] and "Zone 3 작업 유사도는 40%" in data["answer"] and "방향" not in data["answer"]


def test_unscored_body_part_is_declined_on_the_signed_viewer_path() -> None:
    data = _ask(_Pipeline(), "팔꿈치를 더 펴야 해, 굽혀야 해?", context=_pose_context(), task="pose_recomm")
    assert "확인할 수 없습니다" in data["answer"] and "작업 유사도" not in data["answer"]


def test_arm_side_question_is_answered_from_every_correction_of_the_arm_scoring() -> None:
    right = _ask(_Pipeline(), "오른손을 어떻게 고쳐?", context=_pose_context(), task="pose_recomm")["answer"]
    assert "채점한 분사 2회 중 1회에서 오른팔 교정이 나왔습니다" in right and "SEQ-001" in right and "오른쪽 팔꿈치" in right
    assert "특정된" not in right and "참고용" in right  # 첫 교정(왼쪽 손목)만 보면 오른팔 교정을 놓친다
    assert "왼팔 교정은 나오지 않았습니다" in viewer_arm_pose.arm_pose_answer(
        {**ARM, "sequences": [ARM["sequences"][1]]}, current=False, playback=None, zone_id=None, side="left")
    waiting = _ask(_Pipeline(), "오른손을 어떻게 고쳐?", context=_pose_context(arm=None), task="pose_recomm")["answer"]
    assert "팔 자세 채점 결과를 받지 못해" in waiting


def test_whole_record_arm_answer_names_which_similarity_it_reports() -> None:
    whole = _ask(_Pipeline(), "전체 작업 유사도는?", context=_pose_context(), task="pose_recomm")["answer"]
    assert "평균 자세 유사도 70%" in whole and "전체 작업 유사도는 70%" in whole


@pytest.mark.parametrize(("metric", "expected"), [
    ({"actual": 140.0, "reference": 15.0, "actual_label": "깊게 굽힌 자세", "reference_label": "서 있는 자세"},
     "전문가는 이 Zone에서 주로 서 있는 자세였습니다(이 기록은 깊게 굽힌 자세)."),
    ({"actual": 140.0, "reference": 15.0, "actual_label": "서 있는 자세", "reference_label": "서 있는 자세"}, None),
    ({"actual": 30.0, "reference": 12.5, "actual_label": "서 있는 자세", "reference_label": "서 있는 자세"}, "몸통 기울기를 약 17.5° 더 세우세요."),
])
def test_torso_advice_never_asks_for_an_impossible_angle(metric: dict[str, Any], expected: str | None) -> None:
    assert motion_viewer._correction_message({"key": "posture", **metric}) == expected


def test_zone_answer_names_the_basis_it_was_judged_against_and_compares_what_it_shows() -> None:
    analysis = {
        "pose_report": {"average_similarity_score": 0.5126, "zone_summary": [
            {"zone_id": "1", "average_similarity_score": 0.513, "evaluated_sequence_count": 6, "problem_joints": []}]},
        "coating_report": {"comparison_basis": "session_target", "target_thickness_um": 160.0},
        "coating_zones": [{"zone_id": "1", "status": "under_app", "thickness_um": 33.0, "coverage_percent": 0.0}],
    }
    answer = pose_query.answer_zone_question(("RPT-1", analysis), context=None).answer
    assert "51%로 세션 평균과 같습니다" in answer and "높습니다" not in answer
    assert "세션 목표 두께 160µm보다 부족합니다" in answer and "전문가 기준" not in answer
    analysis["coating_report"] = {"comparison_basis": "expert_reference", "target_thickness_um": None}
    assert "전문가 기준보다 부족합니다" in pose_query.answer_zone_question(("RPT-1", analysis), context=None).answer


def test_v2_page_keeps_every_element_the_playback_scripts_use() -> None:
    document = motion_viewer.viewer_v2_document()
    ids = set(__import__("re").findall(r'id="([A-Za-z0-9_]+)"', document))
    scripts = motion_viewer._VIEWER_HTML[motion_viewer._VIEWER_HTML.index("<script>"):]
    needed = set(__import__("re").findall(r"\$\('([A-Za-z0-9_]+)'\)", scripts))
    assert needed and needed <= ids and "function loadMotion" in document and "/*CORE_SCRIPT*/" not in document
    assert {"poseUiTemplate", "zoneTimeline", "realtimeHudSpeed", "coatingCanvas", "v2Ask"} <= ids
    assert {"v2CoatBasisTarget", "v2CoatBasisExpert"} <= ids


def test_v2_question_column_offers_the_same_answer_languages_as_the_first_viewer() -> None:
    import re
    from pathlib import Path

    static = Path(motion_viewer.__file__).with_name("static")
    pairs = re.compile(r"\['([a-z]{2})', '([a-z]{2})', ")
    first = pairs.findall((static / "motion_viewer_chat.js").read_text(encoding="utf-8"))
    second = pairs.findall((static / "motion_viewer_v2.js").read_text(encoding="utf-8"))
    assert len(first) == 12 and second == first
    assert all((static / "flags" / f"{country}.svg").is_file() for _, country in second)
    assert "response_language: sent.language[0]" in (static / "motion_viewer_v2.js").read_text(encoding="utf-8")


def _v2_static(name: str) -> str:
    from pathlib import Path
    return (Path(motion_viewer.__file__).with_name("static") / name).read_text(encoding="utf-8")


def test_v2_coating_feature_names_both_bases_in_visible_text() -> None:
    page = _v2_static("motion_viewer_v2.html")
    assert '<button id="v2CoatBasisTarget" type="button" aria-pressed="true">목표 대비</button>' in page
    assert '<button id="v2CoatBasisExpert" type="button" aria-pressed="false">전문가 대비</button>' in page


def test_v2_script_draws_the_target_view_from_the_signed_target_block() -> None:
    script = _v2_static("motion_viewer_v2.js")
    assert "coating_evaluation?.target" in script
    for key in ("six_band_counts", "unpainted_fraction", "band_edges_um", "normal_band_um", "grade_labels", "band_labels", "basis_label"):
        assert key in script, key
    assert "reference_um" in script   # 전문가 대비 그림도 남아 있다(D5)


def test_v2_expert_map_button_has_a_visible_label() -> None:
    import re
    page = _v2_static("motion_viewer_v2.html")
    bars = re.search(r'<button id="v2CoatBarsView"[^>]*>(.*?)</button>', page).group(1)
    expert_map = re.search(r'<button id="v2CoatMapView"[^>]*>(.*?)</button>', page).group(1)
    assert "편차" in re.sub(r"<[^>]+>", "", bars)
    assert "전문가 지도" in re.sub(r"<[^>]+>", "", expert_map)


def test_v2_keeps_sending_the_whole_signed_coating_block() -> None:
    assert "coating_context: payload.coating_evaluation ?? null" in _v2_static("motion_viewer_v2.js")


def _css_rule(css: str, selector: str) -> str:
    import re
    return re.search(r"(?m)^" + re.escape(selector) + r"\s*\{([^}]*)\}", css).group(1)


def test_v2_question_pins_sit_above_the_slider_that_follows_them_in_the_dom() -> None:
    import re
    page, css = _v2_static("motion_viewer_v2.html"), _v2_static("motion_viewer_v2.css")
    assert page.index('id="v2Pins"') < page.index('id="timeline"')   # 슬라이더가 뒤라서 z-index 없이는 핀을 덮는다
    pins = re.search(r"z-index:\s*(\d+)", _css_rule(css, ".pins"))
    slider = re.search(r"z-index:\s*(\d+)", _css_rule(css, "#timeline"))
    assert pins and slider and int(pins.group(1)) > int(slider.group(1)) >= 1


def test_v2_coating_list_follows_the_current_zone_only_when_asked_and_not_while_the_user_scrolls() -> None:
    script = _v2_static("motion_viewer_v2.js")
    assert "function markCurrentZone(follow = false)" in script
    assert "!follow" in script and "coatScrolledByUser" in script
    assert script.count("markCurrentZone(true)") == 2   # Zone·맵이 바뀔 때 한 곳 + 사용자 스크롤 뒤 따라잡기
    assert "previous.zone_id !== anchor.zone_id || previous.map_num !== anchor.map_num) markCurrentZone(true)" in script   # 분사만 바뀔 때는 아님
    assert "'scroll'" in script and "coatCatchUp" in script
    assert script.count("markCurrentZone();") >= 3      # 목록·순서를 그릴 때는 강조만


def test_v2_hud_corrections_do_not_say_small_difference_for_a_zone_the_coating_panel_flags() -> None:
    # 2026-11-13 재디자인: 장면 아래 Zone 줄 대신 HUD 교정 칸이 같은 문장을 보인다.
    script = _v2_static("motion_viewer_v2.js")
    note = script[script.index("function renderHudFix"):script.index("$('v2HudFixList')")]
    assert "targetOf()?.zones" in note and "'below'" in note and "'above'" in note and "grade_labels" in note
    assert note.index("coatNote ||") < note.index("전문가 기록과 차이가 작습니다")   # 도막 등급이 먼저


def test_v2_coating_target_rows_use_server_words_and_the_shared_sort() -> None:
    # 2026-11-13 재디자인 C1: 눈금 축 대신 Zone마다 6구간 100% 막대와 서버 등급 글자. 순서는 순수 함수 coatOrder.
    script = _v2_static("motion_viewer_v2.js")
    rows = script[script.index("function renderTargetCoating"):script.index("function renderExpertCoating")]
    assert "L.coatOrder(target.zones" in rows and "labels[grade]" in rows and "bandLabels" in rows
    assert "아래로 ${below}개 더" in script
