"""Session and Replay queries must use records from the requested scope."""

import json
from types import SimpleNamespace

import pytest

from trainer.demo_api.sequence_store import SequenceStore
from trainer.demo_api.tests.test_scenario_reference_query import Pipeline, URL, client, profile, query


def _saved_run(use_report):
    store = SequenceStore(":memory:")
    report = {
        "sequence_report": {
            "analysis_status": "available", "recommended_flow": ["2", "1", "3"],
            "first_visit_flow": ["3", "1", "2"], "skipped_zone_count": 0, "deviation_count": 1,
            "adherence_rate": 1.0, "direction_adherence_rate": .25,
            "direction_summary": {"evaluated_sequence_count": 4, "matched_sequence_count": 1},
            "zone_summary": [{"zone_id": "3", "visit_count": 2, "revisit_count": 1, "deviation_count": 1}],
        },
        "coating_report": {"analysis_status": "available", "normal_band_um": [145, 190],
                           "coating_distribution_index": None, "root_cause": None,
                           "source_artifact_id": "QA-Map02.png",
                           "distribution": {"below": 20, "in_band": 70, "above": 10}},
        "coating_zones": [
            {"zone_id": "1", "map_num": 1, "thickness_um": 120, "grade": "below"},
            {"zone_id": "2", "map_num": 1, "thickness_um": 220, "grade": "above"},
            {"zone_id": "3", "map_num": 1, "thickness_um": 170, "grade": "normal"},
        ],
        "pose_report": {"analysis_status": "unavailable", "average_similarity_score": None, "evaluated_sequence_count": 0},
    }
    stored = SimpleNamespace(user_id="COPILOT-DEMO-USER", scenario_id="3", tool_type="spray_gun",
                             target_thickness_um=145, target_thickness_min_um=145,
                             target_thickness_max_um=190, target_basis="session_target",
                             session_status="COMPLETED" if use_report else "STARTED")
    reports = SimpleNamespace(get_session=lambda _: stored,
                              latest_complete_analysis=lambda _: ("RPT-RUN", report) if use_report else None,
                              get_completion=lambda _: SimpleNamespace(
                                  session_id="RUN", report_status="complete",
                                  payload={"user_id": stored.user_id, "scenario_id": "3", "vr_score": 78, "is_pass": True,
                                           "avg_distance_mm": 271},
                              ))
    visits = zip(("3", "1", "3", "2"), (210, 330, 260, 290), (280, 360, 380, 420))
    for index, (zone, distance, speed) in enumerate(visits, start=1):
        sequence_id = f"SEQ-{index:03d}"
        body = {"scenario_id": "3", "user_id": stored.user_id, "sequence_id": sequence_id, "zone_id": zone,
                "sequence_result": {"avg_distance_mm": distance, "avg_speed_mms": speed},
                "sequence_setting": {"tool_type": "spray_gun"}}
        store.record(session_key="RUN", sequence_id=sequence_id, body_sha256=sequence_id,
                     frames=1, received_at="now", body=json.dumps(body))
    return store, reports


@pytest.mark.parametrize("prompt, fragments", [
    ("직전 분사의 평균 이동 속도랑 이격거리를 cm로 환산해 줘.", ("42cm/s", "29cm")),
    ("방금 저장된 분사 시퀀스의 식별자는 뭐야?", ("SEQ-004",)),
    ("기본과 같은 값이어도 이 세션의 정상 범위는 시작 입력에서 생략했나요?", ("145", "190", "명시")),
])
def test_query_reuses_stored_sequence_measurements_and_limits(tmp_path, prompt, fragments):
    store, reports = _saved_run(False)
    pipeline = Pipeline()
    response = query(client(tmp_path, store=store, reports=reports, pipeline=pipeline),
                     session_id="RUN", prompt=prompt, summarize=False)
    assert all(fragment in response["answer"] for fragment in fragments)
    assert not pipeline.calls


@pytest.mark.parametrize("prompt, fragments", [
    ("완료 기록의 최초 방문 순서와 재방문을 포함한 흐름은 각각 어떻게 돼?", ("3→1→2", "3→1→3→2")),
    ("리플레이 3번 구역 첫 분사와 다시 와서 분사했을 때의 평균 이격거리 차이는?", ("210mm", "260mm", "50mm")),
    ("분사별 평균 중 이격거리가 가장 컸던 구역과 값은?", ("1번", "330mm")),
    ("VR에서 받은 78점은 자세 유사도인가요?", ("78점", "자세 유사도", "확인할 수 없습니다")),
    ("종료 리포트에서 2번 구역의 평균 두께가 상한보다 몇 마이크로 높은가요?", ("220µm", "190µm", "30µm")),
    ("완료 리포트의 도막 분포 지수가 없으면 0으로 적어도 될까요?", ("미산출", "0점")),
])
def test_completed_query_uses_its_report_and_observed_scope(tmp_path, prompt, fragments):
    store, reports = _saved_run(True)
    pipeline = Pipeline()
    response = query(client(tmp_path, store=store, reports=reports, pipeline=pipeline),
                     session_id="RUN", prompt=prompt, summarize=False)
    assert all(fragment in response["answer"] for fragment in fragments)
    assert not pipeline.calls


def test_distributed_thickness_projection_uses_each_pixels_cap(tmp_path):
    store, reports = _saved_run(False)
    response = query(client(tmp_path, store=store, reports=reports, pipeline=Pipeline()), session_id="RUN", summarize=False,
                     prompt="가정입니다. 구역 30%는 260µm, 70%는 72.5µm일 때 목표 두께 도달 면적과 진행률이 각각 얼마인가요?")
    assert "145µm 이상인 면적은 30%" in response["answer"]
    assert "도막 진행률은 65%" in response["answer"]


def test_coating_query_distinguishes_submitted_average_and_corrected_surface(tmp_path):
    from trainer.demo_api.tests.test_zone_completion_gate import engine, request

    progress = engine(tmp_path)
    req = request(value=145)
    req.sequence_result.avg_thickness_um = 200
    store = SequenceStore(":memory:")
    store.record(session_key=req.session_id, sequence_id=req.sequence_id, body_sha256="source", frames=1,
                 received_at="now", body=req.model_dump_json())
    response = query(client(tmp_path, store=store, coating_engine=progress), session_id=req.session_id,
                     user_id=req.user_id, prompt="이번 분사 요약 평균과 구역의 누적 높이맵 평균을 각각 알려줘.", summarize=False)
    assert "분사 요약에 들어온 평균은 200µm" in response["answer"]
    assert "도포부 평균은 13" in response["answer"]


@pytest.mark.parametrize("snapshots", [1, 2])
def test_coating_change_requires_two_comparable_surface_observations(tmp_path, snapshots):
    from trainer.demo_api.tests.test_zone_completion_gate import engine, request

    progress = engine(tmp_path)
    store = SequenceStore(":memory:")
    for i, value in enumerate((140, 190)[:snapshots], start=1):
        req = request(i, value=value)
        store.record(session_key=req.session_id, sequence_id=req.sequence_id, body_sha256=req.sequence_id,
                     frames=1, received_at="now", body=req.model_dump_json())
    response = query(client(tmp_path, store=store, coating_engine=progress), session_id=req.session_id,
                     user_id=req.user_id, prompt="방금 기록한 1번 구역 도막의 전후 값 차이를 알려줘.", summarize=False)
    if snapshots == 1:
        assert response["context_status"] == "not_available"
        assert "같은 구역에서 비교할 두 시점" in response["answer"]
    else:
        assert response["context_status"] == "available"
        assert "1번 구역의 칠한 곳 평균 보정 두께" in response["answer"]
        assert "증가" in response["answer"]


def test_grinding_plan_uses_registered_grinding_data_only(tmp_path):
    profile(tmp_path, "spray", ["1", "4", "7"])
    profile(tmp_path, "grinding", ["9", "8"])
    response = query(client(tmp_path), prompt="스프레이 말고, 학습모드 3의 그라인딩 전용 순서를 알려줘.", summarize=False)
    assert "그라인딩" in response["answer"] and "9 → 8" in response["answer"]
    assert "1 → 4 → 7" not in response["answer"]


@pytest.mark.parametrize("prompt, fragments", [
    ("최종 도막 리포트가 읽은 높이맵의 파일명은 뭔가요?", ("QA-Map02.png", "높이맵")),
    ("이 리포트에서 과도막 구역 수 비율과 과도막 면적 비율은 각각 얼마인가요?", ("1/3", "33.3%", "10/100", "10.0%")),
    ("완료 보고서에서 순서 유사도와 방향 준수율은 왜 다른가요? 방향 평가와 일치 건수도 알려 주세요.", ("100%", "25%", "4회", "1회")),
    ("완료 기록 기준 어느 구역에 두 번 방문했고 재방문과 순서 이탈은 각각 몇 번인가요?", ("3번", "방문 2회", "재방문 1회", "순서 이탈 1회")),
    ("저장된 분사 평균 이격거리의 중앙값을 리포트 평균과 구별해 줘.", ("275mm", "271mm")),
    ("SEQ-003의 구역과 저장된 평균 이격거리를 알려줘.", ("SEQ-003", "3번", "260mm")),
    ("SEQ-005의 이격거리를 찾아줘.", ("SEQ-005", "없습니다")),
    ("리플레이 분석이 완료됐나요? VR 점수와 합격도 확인해 주세요.", ("완료", "78점", "합격")),
    ("자세를 못 채점했다면 분사 기록도 없는지, 리포트 채점 건수와 같이 비교해 주세요.", ("저장된 분사 4개", "자세 채점 0개")),
    ("3번에 처음 갔을 때와 다시 갔을 때 이격거리와 이동 속도의 차이가 최종 도막의 원인인가요?",
     ("210mm", "260mm", "50mm 증가", "280mm/s", "380mm/s", "100mm/s 증가", "원인을 입증할 수는 없습니다")),
])
def test_query_report_and_raw_record_scopes_use_authoritative_rows(tmp_path, prompt, fragments):
    store, reports = _saved_run(True)
    pipeline = Pipeline()
    response = query(client(tmp_path, store=store, reports=reports, pipeline=pipeline),
                     session_id="RUN", prompt=prompt, summarize=False)
    assert all(fragment in response["answer"] for fragment in fragments), response["answer"]
    assert "RPT-RUN" not in response["answer"]
    assert not pipeline.calls


def test_completed_vr_result_cannot_carry_to_another_session_in_one_conversation(tmp_path):
    store, reports = _saved_run(True)
    old_report = reports.latest_complete_analysis
    reports.latest_complete_analysis = lambda session_id: old_report(session_id) if session_id == "RUN" else None
    pipeline = Pipeline()
    app_client = client(tmp_path, store=store, reports=reports, pipeline=pipeline)

    def submit(session_id, prompt):
        response = app_client.post(URL, json={
            "user_id": "COPILOT-DEMO-USER", "scenario_id": "3", "session_id": session_id,
            "task_type": "knowledge_search", "prompt": prompt,
        }, headers={"X-Copilot-Conversation": "scoped-result-test"})
        assert response.status_code == 200
        return response.json()["data"]

    old = submit("RUN", "완료 리포트 분석 상태와 VR 시뮬레이터 점수, 합격 여부만 알려 줘.")
    assert "78점(합격)" in old["answer"]
    fresh = submit("NEW", "새 훈련에도 앞 기록의 그 점수와 합격이 나왔나요?")
    assert fresh["context_status"] == "not_available"
    assert "78" not in fresh["answer"] and "아직 확인할 수 없습니다" in fresh["answer"]
    assert not pipeline.calls
