"""Short answers retain requested facts, full lists and decision caveats."""
import re
import pytest
from trainer.demo_api.record_query import RecordFacts, RecordPart, answer_part
from trainer.demo_api.pose_query import answer_order_question
from trainer.demo_api.viewer_arm_pose import zone_pose_summary
from trainer.demo_api.viewer_progress import next_answer, status_answer
from trainer.demo_api.tests.viewer_record import ask


def _facts():
    sequences = [{'sequence_id':f'SEQ-{i:03}', 'zone_id':str(i), 'similarity_percent':40+i,
                  'confidence':'limited' if i == 30 else 'normal'} for i in range(1,31)]
    return RecordFacts('', {}, {}, None, {'status':'available', 'average_similarity_percent':55.5, 'sequences':sequences})


def _part(text):
    return RecordPart(text, 'pose_judgement', (), 'none', None, None, 'any', (), False)


def test_overall_pose_answer_summarizes_instead_of_dumping_every_sequence():
    answer, _ = answer_part(_facts(), _part('전체 자세 유사도는?'), task_type='pose_recomm')
    assert '56%' in answer and '30회' in answer and '참고용' in answer and '70%' in answer
    assert 'SEQ-001' not in answer and len(answer) < 250


def test_an_explicit_per_sequence_request_keeps_every_score():
    answer, _ = answer_part(_facts(), _part('분사별 자세 유사도를 모두 보여줘'), task_type='pose_recomm')
    for i in range(1,31):
        assert f'SEQ-{i:03}' in answer and f'{40+i}%' in answer


@pytest.mark.parametrize('prompt, expected', [
    ('이 기록에서 전문가보다 멀리 칠한 Zone은 어디야?', {'1'}),
    ('이 기록에서 전문가보다 멀리 칠한 Zone과 비슷한 Zone은 어디야?', {'1','2'}),
    ('이 기록에서 전문가보다 멀리, 가까이, 비슷하게 칠한 Zone을 알려줘', {'1','2','3'}),
])
def test_distance_filter_keeps_only_the_explicitly_requested_relations(prompt, expected):
    claims = {'zones':{str(i):{'zone_id':str(i), 'distance_mm':distance, 'reference_distance_mm':100}
                       for i, distance in [(1,200),(2,105),(3,50)]}}
    facts = RecordFacts('', {}, claims, None, None)
    part = RecordPart(prompt, 'distance_filter', (), 'none', 'higher', None, 'any', (), False)
    answer, _ = answer_part(facts, part, task_type='knowledge_search')
    assert set(re.findall(r'Zone (\d)', answer)) == expected
    for zone, value in [('1','200'),('2','105'),('3','50')]:
        if zone in expected:
            assert value + ' mm' in answer and '전문가 100 mm' in answer


def test_reference_order_preserves_the_whole_flow_without_repeating_the_start():
    flow = [str(i) for i in range(1,14)]
    report = {'analysis_status':'available', 'recommended_flow':flow, 'first_visit_flow':[],
              'reference_session_count':6, 'position_distribution':[
                  [{'zone_id':'1','probability':1.0}], [{'zone_id':'2','probability':1.0}]]}
    answer = answer_order_question(('r', {'sequence_report':report}), prompt='권장 Zone 순서는?', intent='order_reference').answer
    assert ' → '.join(flow) in answer
    assert answer.count('Zone 1') == 1
    assert '우선 Zone' not in answer


def test_explicit_multi_order_request_keeps_both_complete_flows_and_requested_scores():
    report = {'analysis_status':'available', 'recommended_flow':['1','3','2'], 'first_visit_flow':['3','1','2'],
              'adherence_rate':.75, 'best_reference_similarity':.8, 'position_similarity':.6}
    answer = answer_order_question(('r', {'sequence_report':report}),
        prompt='권장 순서와 실제 순서, 유사도 수치를 모두 알려줘', intent='order_reference').answer
    assert '1 → 3 → 2' in answer and '3 → 1 → 2' in answer
    assert all(value in answer for value in ['75%', '80%', '60%'])


def test_multi_order_request_does_not_equate_visiting_every_zone_with_coating_completion():
    report = {'analysis_status':'available', 'recommended_flow':['1','2'], 'first_visit_flow':['1','2']}
    rows = [{'zone_id':'1', 'map_num':1, 'gate_status':'overpaint_risk'}]
    answer = answer_order_question(('r', {'sequence_report':report, 'coating_zones':rows}),
        prompt='권장 순서와 빠진 구역을 알려줘', intent='order_missing').answer
    assert '1 → 2' in answer and '미방문: 없음' in answer
    assert 'Zone 1' in answer and '더 칠하면 안 되는' in answer
    assert '세션이 끝났을 때' in answer


def test_multi_order_request_keeps_the_explicitly_named_similarity_metrics():
    report = {'analysis_status':'available', 'recommended_flow':['1','3','2'], 'first_visit_flow':['3','1','2'],
              'adherence_rate':.75, 'best_reference_similarity':.8, 'position_similarity':.6}
    answer = answer_order_question(('r', {'sequence_report':report}),
        prompt='권장 순서와 실제 순서, 가장 가까운 전문가 수행 유사도와 순번별 유사도는?', intent='order_reference').answer
    assert '1 → 3 → 2' in answer and '3 → 1 → 2' in answer
    assert '80%' in answer and '60%' in answer


@pytest.mark.parametrize('adherence', [None, .75])
def test_multi_order_request_distinguishes_the_record_direction_from_direction_adherence(adherence):
    from trainer.demo_api.copilot_router import _viewer_answer
    context = {'recommended_flow':['1','3','2'], 'selected_flow':['3','1','2'],
               'direction_summary':{'classification':'horizontal','label':'수평 중심'}}
    if adherence is not None:
        context['direction_adherence_rate'] = adherence
    answer = _viewer_answer('trajectory_recomm', '권장 순서와 실제 순서, 도포 방향을 알려줘', context, 'order_reference')
    assert '1 → 3 → 2' in answer and '3 → 1 → 2' in answer
    assert '수평' in answer and '방향 평가 자료 없음' not in answer
    assert ('75%' in answer) == (adherence is not None)


def _arm():
    return {'status':'available', 'zones':{str(i):{'similarity_percent':80-i,
        'guidance':f'교정 안내 {i}', 'evaluated_sequence_count':1} for i in range(1,8)}}


def test_a_worst_zone_question_omits_other_score_families_and_repeated_guidance():
    answer = zone_pose_summary(_arm(), [{'zone_id':'2','similarity_percent':10}], prompt='가장 문제가 큰 Zone은?')
    assert 'Zone 7' in answer and '73%' in answer and '교정 안내 7' in answer
    assert '작업 유사도' not in answer and '교정 안내 6' not in answer


def test_an_explicit_all_zone_request_preserves_each_zone_and_its_guidance():
    answer = zone_pose_summary(_arm(), prompt='모든 Zone별 자세 유사도와 개선점을 보여줘')
    for i in range(1,8):
        assert f'Zone {i}' in answer and f'교정 안내 {i}' in answer


def test_next_zone_answer_puts_the_action_first_and_keeps_the_final_map_limit():
    claims = {'reference_order':['1','2','3'], 'actual_order':['1'],
              'expert_transitions':[{'from':'1','to':'2','count':6}]}
    answer, scope = next_answer(claims, None, {'zone_id':'1','playhead_seconds':2.0})
    assert answer.startswith('다음은 Zone 2')
    assert '현재' in answer and '도막' in answer and ('최종' in answer or '세션이 끝났을 때' in answer)
    assert scope['next_zone_id'] == '2' and scope['visited_zone_ids'] == ['1']


def test_explicit_all_zone_progress_retains_all_unvisited_zone_ids():
    order = [str(i) for i in range(1,20)]
    answer, scope = status_answer({'reference_order':order, 'actual_order':[]}, None, prompt='아직 안 칠한 모든 Zone 목록은?')
    for zone in order:
        assert re.search(rf'(?<!\d){zone}(?!\d)', answer)
    assert '외' not in answer
    assert scope['unvisited_zone_ids'] == order


@pytest.mark.parametrize('mode', ['quick', 'detailed'])
@pytest.mark.parametrize('prompt', ['지금 자세는 어때?', '자세 유사도는 얼마야?'])
def test_pose_only_question_does_not_append_the_other_work_score(mode, prompt):
    answer = ask(prompt, tab='pose', mode=mode)['answer']
    assert '자세 유사도' in answer and '작업 유사도' not in answer


@pytest.mark.parametrize('mode', ['quick', 'detailed'])
def test_work_score_only_question_keeps_the_work_metric(mode):
    answer = ask('전체 작업 유사도는?', tab='pose', mode=mode)['answer']
    assert '작업 유사도' in answer and '87%' in answer
    assert '자세 유사도' not in answer


@pytest.mark.parametrize('mode', ['quick', 'detailed'])
def test_explicitly_requested_pose_and_work_scores_are_both_kept(mode):
    answer = ask('전체 자세 유사도와 작업 유사도를 함께 알려줘', tab='pose', mode=mode)['answer']
    assert '자세 유사도' in answer and '작업 유사도' in answer
