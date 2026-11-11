from trainer.coating_explanation_service.service import catalogs, selection_schema, validate_selection, render


def sample(grade="below"):
    return {"assessment_id": "a", "revision": 1, "zone_key": {"zone_id": 2}, "quality": {"grade": grade}, "evidence": [{"evidence_id": "q", "statement": "도포부 평균은 100µm입니다.", "value": 100}, {"evidence_id": "s", "statement": "관측 이동 속도는 400mm/s입니다.", "value": 400}], "coaching": {"causes": [{"cause_id": "c", "statement": "이동 속도와 저도막의 관련 가능성을 확인합니다.", "evidence_ids": ["q", "s"], "causal_status": "candidate_not_established"}], "actions": [{"action_id": "a", "statement": "다음 작업에서 이동 속도를 낮추는 방향을 검토합니다.", "variable": "speed_mm_s", "direction": "decrease", "target_range": None, "expected_effect": None, "evidence_ids": ["q", "s"]}]}}


def selection():
    return {"evidence_ids": ["q"], "cause_ids": ["c"], "action_ids": ["a"], "presentation": "observations_then_actions"}


def test_unknown_action_and_free_prose_rejected():
    c = catalogs(sample()); s = selection()
    s["action_ids"] = ["invented-speed-target"]
    assert not validate_selection(s, c)[0]
    s = selection(); s["text"] = "원인은 확정되었으며 300mm/s가 최적입니다."
    assert not validate_selection(s, c)[0]


def test_causal_guarantee_and_numeric_target_not_admitted():
    a = sample(); a["coaching"]["causes"][0]["causal_status"] = "proven"
    a["coaching"]["actions"][0]["target_range"] = [200, 300]
    c = catalogs(a)
    assert c["causes"] == [] and c["actions"] == []


def test_action_evidence_is_included_in_display_even_if_not_selected_as_fact():
    a = sample(); c = catalogs(a); s = selection()
    assert validate_selection(s, c)[0]
    text, ids = render(a, "zone", s, c)
    assert ids == ["q", "s"]
    assert "400mm/s" in text and "속도를 낮추" in text
    assert "원인이 입증된 것은 아닙니다" in text
    assert "최적 수치나 개선 효과" in text


def test_normal_does_not_invent_causes():
    a = sample("normal"); a["coaching"] = {"causes": [], "actions": []}
    c = catalogs(a)
    s = {"evidence_ids": ["q"], "cause_ids": [], "action_ids": [], "presentation": "observations_then_actions"}
    assert validate_selection(s, c)[0]
    text, _ = render(a, "zone", s, c)
    assert "정상도막" in text and "원인 후보" not in text and "다음 작업 안내" not in text
    assert selection_schema(c)["properties"]["action_ids"]["maxItems"] == 0


def test_missing_target_withholds_grade_and_session_scope_is_current():
    a = sample(None); a["coaching"] = {"causes": [], "actions": []}
    c = catalogs(a)
    s = {"evidence_ids": ["q"], "cause_ids": [], "action_ids": [], "presentation": "observations_then_actions"}
    text, _ = render(a, "zone", s, c)
    assert "판정은 확정하지 않습니다" in text
    text, _ = render(a, "session", s, c)
    assert "현재 세션" in text and "전체 구역" not in text


def test_duplicate_ids_and_missing_available_action_rejected():
    c = catalogs(sample()); s = selection()
    s["evidence_ids"] = ["q", "q"]
    assert not validate_selection(s, c)[0]
    s = selection(); s["action_ids"] = []
    assert not validate_selection(s, c)[0]


def test_internal_lineage_stays_out_of_user_coaching():
    a = sample()
    a["evidence"].append({"evidence_id": "internal-hash", "evidence_type": "lineage", "statement": "SHA-256 internal lineage hash"})
    c = catalogs(a)
    assert all(e["evidence_id"] != "internal-hash" for e in c["evidence"])
    text, ids = render(a, "zone", selection(), c)
    assert "SHA-256" not in text and "internal-hash" not in ids
