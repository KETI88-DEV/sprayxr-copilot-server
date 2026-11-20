import pytest

from trainer.demo_api import training_verification as verification
from trainer.demo_api.tests.test_training_verification import fixtures


@pytest.mark.parametrize('valid,stale,expected', [(0, False, False), (1000, False, False),
    (1199, False, False), (1200, False, True), (1201, False, True), (1201, True, False)])
def test_exam_quantity_uses_current_1200_threshold_without_changing_history(valid, stale, expected):
    result = verification._exam_metadata(valid=valid, stale=stale, historical_target=1000)
    assert result == {'target': 1200, 'population': 'quality_complete',
                      'technical_goal_met': expected, 'historical_run_target': 1000}


def test_exam_presentation_does_not_rewrite_existing_audit_or_expertise(tmp_path):
    _, store, _ = fixtures(tmp_path)
    store.audit()
    saved = store.root / 'runs' / (store.status()['run_id'] + '.json')
    before = saved.read_bytes()
    result = store.status()
    assert result['target'] == 1000
    assert result['exam']['target'] == 1200
    assert result['exam']['population'] == 'quality_complete'
    assert result['exam']['technical_goal_met'] is False
    assert saved.read_bytes() == before


def test_exam_not_started_still_exposes_current_quantity_target(tmp_path):
    _, store, _ = fixtures(tmp_path)
    result = store.status()
    assert result['phase'] == 'not_started'
    assert result['exam']['target'] == 1200
    assert result['exam']['technical_goal_met'] is False
