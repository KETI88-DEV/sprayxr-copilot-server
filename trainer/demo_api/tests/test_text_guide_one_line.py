"""실시간 안내(text_guide)는 시뮬레이터 한 줄(60자)에 들어가고 판정·조치를 잃지 않는다."""
from __future__ import annotations

from pathlib import Path

import pytest

from trainer.demo_api.quick_answer import QUICK_LIMIT
from trainer.demo_api.sequence_analysis import SequenceDecision, gate_sequence_decision
from trainer.demo_api.telemetry_contracts import CoatingProgress, EngineFeedback
from trainer.demo_api.tests.test_zone_completion_gate import analyze, request, thin_mean


def _one_line(text: str | None) -> str:
    assert text and "\n" not in text and len(text) <= QUICK_LIMIT, text
    return text


@pytest.mark.parametrize(("req", "words", "absent"), [
    (request(valid_count=95), ("Zone 1", "95%", "Zone 2로 이동"), ()),
    (request(valid_count=50), ("Zone 1", "정상범위", "50.0%", "보완"), ("다음", "이동하세요")),
    (thin_mean(), ("Zone 1", "저도막", "153", "보완"), ("다음", "이동하세요")),
    (request(value=300), ("Zone 1", "과도막", "분사 중지"), ("다음", "이동하세요")),
])
def test_coating_guides_fit_one_line_and_keep_the_gate(tmp_path: Path, req, words, absent) -> None:
    text = _one_line(analyze(tmp_path, req)["text_guide"])
    assert all(word in text for word in words), text
    assert not any(word in text for word in absent), text


def test_unknown_gate_guide_fits_one_line_without_a_move(tmp_path: Path) -> None:
    req = request()
    req.sequence_result.cur_sequence_heatmap = None
    text = _one_line(analyze(tmp_path, req)["text_guide"])
    assert "현재 Zone 1" in text and "이동 전" in text and "다음" not in text


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
    _progress("overpaint_risk", coverage=12.3, over=99.9, mean=1600.0),
    _progress("needs_more", coverage=100.0, mean=159.96),
    _progress("needs_more", coverage=99.9, over=99.9),
    _progress("needs_more", coverage=94.9),
    None,
]


@pytest.mark.parametrize("progress", WORST)
@pytest.mark.parametrize("direction", [None, "Zone 39: 대각선 방향으로 도장하세요."])
def test_longest_zone_numbers_and_direction_still_fit(progress, direction) -> None:
    req = request().model_copy(update={"zone_id": "39"})
    reason = "direction_deviation" if direction else None
    decision = SequenceDecision(EngineFeedback(analysis_status="available", reason_code=reason),
                                direction or "순서 적정.", "direction_deviation" if direction else None, direction)
    text = _one_line(gate_sequence_decision(req, decision, progress).text_guide)
    if direction:
        assert text.startswith(direction)
    if progress is not None and progress.status == "overpaint_risk":
        assert "분사 중지" in text
    if progress is not None and progress.status == "complete":
        assert "Zone 38로 이동" in text


@pytest.mark.parametrize("mode", ["quick", "detailed"])
def test_current_next_query_uses_the_one_line_guide_only_when_quick(tmp_path: Path, mode: str) -> None:
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
    text, _, payload = current_next_answer(
        query, store, sequence_engine=SequenceAnalysisEngine(tmp_path), coating_engine=progress)
    if mode == "quick":
        assert "저도막" in _one_line(text)
    else:
        assert text.endswith(payload["message"]) and "정상범위 160~200µm" in text
