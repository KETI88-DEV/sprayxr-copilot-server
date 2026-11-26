"""실시간 안내(text_guide)는 시뮬레이터 한 줄(60자)에 들어가고 판정·조치를 잃지 않는다."""
from __future__ import annotations

from pathlib import Path
import re

import pytest

from trainer.demo_api.quick_answer import QUICK_LIMIT
from trainer.demo_api.sequence_analysis import SequenceDecision, gate_sequence_decision
from trainer.demo_api.telemetry_contracts import CoatingProgress, EngineFeedback
from trainer.demo_api.tests.test_zone_completion_gate import analyze, request, thin_mean


def _one_line(text: str | None) -> str:
    assert text and "\n" not in text and len(text) <= QUICK_LIMIT, text
    assert not re.search(r"(?:하세요|해주세요|해 주세요|바랍니다|입니다|하십시오)", text), text
    return text


@pytest.mark.parametrize(("req", "words", "absent"), [
    (request(valid_count=85), ("1구역", "진행률 85%", "2구역으로 이동"), ()),
    (request(valid_count=50), ("1구역", "진행률 50.0%", "보완"), ("다음", "이동하세요")),
    (thin_mean(), ("1구역", "진행률 85%", "2구역으로 이동"), ("보완",)),
    (request(value=300), ("1구역", "과도막", "진행률 85%", "추가 분사 중지", "2구역으로 이동"), ("일시 중지",)),
])
def test_coating_guides_fit_one_line_and_keep_the_gate(tmp_path: Path, req, words, absent) -> None:
    text = _one_line(analyze(tmp_path, req)["text_guide"])
    assert all(word in text for word in words), text
    assert not any(word in text for word in absent), text


def test_unknown_gate_guide_fits_one_line_without_a_move(tmp_path: Path) -> None:
    req = request()
    req.sequence_result.cur_sequence_heatmap = None
    text = _one_line(analyze(tmp_path, req)["text_guide"])
    assert "현재 1구역" in text and "이동 전" in text and "다음" not in text


def _progress(status: str, *, coverage: float, over: float = 0.0, mean: float = 150.0,
              next_zone: str | None = None) -> CoatingProgress:
    return CoatingProgress(
        zone_id="39", available=True, status=status,
        recommendation={"complete": "move_next_zone", "overpaint_risk": "stop_current_zone"}.get(
            status, "continue_current_zone"),
        completion_percent=coverage, coverage_percent=coverage, overpaint_percent=over,
        next_zone_id=next_zone, message="원문", reason=None, comparison_basis="session_target",
        target_thickness_um=160.0, assessment={"thickness_um": mean})


WORST = [
    _progress("complete", coverage=100.0, mean=199.0, next_zone="38"),
    _progress("complete", coverage=100.0, mean=199.0),
    _progress("overpaint_risk", coverage=12.3, over=99.9, mean=1600.0),
    _progress("needs_more", coverage=84.9, mean=159.96),
    _progress("needs_more", coverage=45.5, over=54.5),  # 목표 이상+상한 초과 길이 시험. 둘 다 두 자리 소수
    _progress("needs_more", coverage=0.1, over=99.9),
    _progress("needs_more", coverage=84.9),
    None,
]


@pytest.mark.parametrize("progress", WORST)
@pytest.mark.parametrize("direction", [None, "39구역: 대각선 방향으로 도장하세요."])
def test_longest_zone_numbers_and_direction_still_fit(progress, direction) -> None:
    req = request().model_copy(update={"zone_id": "39"})
    reason = "direction_deviation" if direction else None
    decision = SequenceDecision(EngineFeedback(analysis_status="available", reason_code=reason),
                                direction or "순서 적정.", "direction_deviation" if direction else None, direction)
    result = gate_sequence_decision(req, decision, progress)
    text = _one_line(result.text_guide)
    if direction:
        assert text.startswith("39구역: 대각선 도포 권장.")
        assert result.detail_guide.startswith(direction)
    if progress is not None and progress.status == "overpaint_risk":
        assert "일시 중지" in text and "부족한 곳만 보완" in text and "추가 분사 중지" not in text
    if progress is not None and progress.status == "complete":
        if progress.next_zone_id:
            assert "38구역으로 이동" in text
        else:
            assert "구역 완료" in text and "이동" not in text
    if progress is not None and progress.status == "needs_more" and progress.overpaint_percent:
        assert "과도막 면적" in text and "일시 중지" in text and "부족한 곳만 보완" in text


@pytest.mark.parametrize("mode", ["quick", "detailed"])
def test_current_next_query_answers_directly_and_keeps_the_gate_payload(tmp_path: Path, mode: str) -> None:
    from types import SimpleNamespace

    from trainer.demo_api.reference_order_query import current_next_answer
    from trainer.demo_api.sequence_analysis import SequenceAnalysisEngine
    from trainer.demo_api.sequence_store import SequenceStore
    from trainer.demo_api.tests.test_zone_completion_gate import engine

    req = thin_mean()
    progress = engine(tmp_path)
    store = SequenceStore(":memory:")
    store.record(session_key=req.session_id, sequence_id=req.sequence_id, body_sha256="q",
                 frames=0, received_at="now", body=req.model_dump_json())
    query = SimpleNamespace(session_id=req.session_id, context=None, scenario_id=None,
                            prompt="다음 구역은?", response_mode=mode)
    text, scope, payload = current_next_answer(
        query, store, sequence_engine=SequenceAnalysisEngine(tmp_path), coating_engine=progress)
    assert text == "2구역으로 이동하세요."
    assert scope["next_zone_id"] == payload["next_zone_id"] == "2"
    assert payload["status"] == "complete" and payload["recommendation"] == "move_next_zone"
    assert "정상범위 160~200µm" in payload["message"]
