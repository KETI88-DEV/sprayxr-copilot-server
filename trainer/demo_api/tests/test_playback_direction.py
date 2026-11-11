import json
import math
import pytest
from trainer.demo_api.viewer_context import sign_context


def trace(points, *, start=0, segment=0):
    return [{"t":round(start+i*.1,6),"point":p,"segment":segment,"map_num":"1","zone_id":"1"} for i,p in enumerate(points)]


def snapshot_context(points, time=.1, *, duration=2, cache_key=None):
    from trainer.demo_api.playback_direction import build_playback_snapshot
    claims,signature=build_playback_snapshot(points, record_id="RECORD", duration_seconds=duration, cache_key=cache_key)
    return "motion_viewer_playback:"+json.dumps({"frame_direction_context":claims,"frame_direction_signature":signature,
        "playback_context":{"playhead_seconds":time,"source_index":0,"map_num":"1","zone_id":"1"}})


@pytest.mark.parametrize("points,expected,label",[
    ([[0,1,0],[.01,1,0],[.02,1,0]],"horizontal","수평"),
    ([[0,1,0],[0,1.01,0],[0,1.02,0]],"vertical","수직"),
    ([[0,1,0],[0,1,.01],[0,1,.02]],"horizontal","수평"),
    ([[0,1,0],[.01,1.01,0],[.02,1.02,0]],"diagonal","대각선"),
    ([[0,1,0],[.01,1,0],[.01,1.01,0]],"irregular","비정형"),
    ([[0,1,0],[.01,1,0],[0,1,0]],"horizontal","수평"),
])
def test_selected_time_uses_local_3d_ray_motion(points,expected,label):
    from trainer.demo_api.playback_direction import playback_direction_from_context
    result=playback_direction_from_context(snapshot_context(trace(points)))
    assert result["analysis_status"]=="available"
    assert result["classification"]==expected and result["label"]==label
    assert result["scope"]=="current_frame"


def test_two_strokes_have_different_directions_and_no_bridging_in_between():
    from trainer.demo_api.playback_direction import playback_direction_from_context
    points=trace([[0,1,0],[.01,1,0],[.02,1,0]])+trace([[0,1,0],[0,1.01,0],[0,1.02,0]],start=1,segment=1)
    assert playback_direction_from_context(snapshot_context(points,.1))["label"]=="수평"
    assert playback_direction_from_context(snapshot_context(points,1.1))["label"]=="수직"
    middle=playback_direction_from_context(snapshot_context(points,.6))
    assert middle["analysis_status"]=="unavailable" and middle["reason_code"]=="no_paint_trace"


def test_stationary_paint_does_not_invent_a_direction():
    from trainer.demo_api.playback_direction import playback_direction_from_context
    result=playback_direction_from_context(snapshot_context(trace([[0,1,0]]*3)))
    assert result["analysis_status"]=="unavailable" and result["reason_code"]=="stationary"


@pytest.mark.parametrize('seconds',[True,-1,float('nan'),float('inf'),3])
def test_invalid_playhead_cannot_select_measurements(seconds):
    from trainer.demo_api.playback_direction import playback_direction_from_context
    result=playback_direction_from_context(snapshot_context(trace([[0,1,0],[.01,1,0]]),seconds))
    assert result["analysis_status"]=="unavailable" and result["reason_code"]=="invalid_playhead"


def test_unsigned_or_tampered_snapshot_is_not_a_measurement():
    from trainer.demo_api.playback_direction import playback_direction_from_context
    context=snapshot_context(trace([[0,1,0],[.01,1,0]]))
    value=json.loads(context.split(':',1)[1]);value['frame_direction_context']['record_id']='OTHER'
    assert playback_direction_from_context('motion_viewer_playback:'+json.dumps(value)) is None


@pytest.mark.parametrize('task',['knowledge_search','pose_recomm','trajectory_recomm'])
@pytest.mark.parametrize('mode',['quick','detailed'])
def test_direction_question_answers_selected_frame_on_every_tab_without_llm(task,mode):
    from trainer.demo_api.tests.test_dialogue_conversation import client,body
    from trainer.demo_api.sequence_store import SequenceStore
    context=snapshot_context(trace([[0,1,0],[0,1.01,0],[0,1.02,0]]))
    result=client(None,SequenceStore(':memory:')).post('/api/v1/copilot/query',json=body('도포 방향은?',task_type=task,response_mode=mode,context=context))
    assert result.status_code==200,result.text
    data=result.json()['data']
    assert '수직' in data['answer'] and '대표' not in data['answer'] and 'vertical' not in data['answer']
    assert data['sequence_context']['scope']=='current_frame' and data['retrieval_used'] is False


def test_missing_frame_snapshot_does_not_fall_back_to_zone_average():
    from trainer.demo_api.tests.test_dialogue_conversation import client,body
    from trainer.demo_api.sequence_store import SequenceStore
    facts={'record_direction':'diagonal','zones':{'1:1':{'direction':'horizontal'}}}
    context='motion_viewer_playback:'+json.dumps({'dialogue_context':facts,'dialogue_signature':sign_context('dialogue',facts),'playback_context':{'playhead_seconds':.1,'map_num':'1','zone_id':'1'}})
    result=client(None,SequenceStore(':memory:')).post('/api/v1/copilot/query',json=body('도포 방향은?',task_type='knowledge_search',context=context)).json()['data']
    assert '현재 프레임' in result['answer'] and '수평' not in result['answer'] and '대각선' not in result['answer']


def test_missing_timestamps_inside_same_segment_are_not_a_local_measurement():
    from trainer.demo_api.playback_direction import playback_direction_from_context
    points=trace([[0,1,0],[0,1.01,0],[0,1.02,0]])+trace([[0,1.03,0],[0,1.04,0],[0,1.05,0]],start=1)
    assert playback_direction_from_context(snapshot_context(points,.6))["analysis_status"]=="unavailable"


@pytest.mark.parametrize("question",["현재 도포 방향은?","도포 방향?","지금 도포 방향은 뭐야?"])
@pytest.mark.parametrize("task",["knowledge_search","pose_recomm","trajectory_recomm"])
def test_current_direction_does_not_need_a_planner(question,task):
    from trainer.demo_api.tests.test_dialogue_conversation import client,body
    from trainer.demo_api.sequence_store import SequenceStore
    context=snapshot_context(trace([[0,1,0],[0,1.01,0],[0,1.02,0]]))
    response=client(None,SequenceStore(":memory:")).post('/api/v1/copilot/query',json=body(question,task_type=task,context=context))
    assert response.status_code==200,response.text
    assert "수직" in response.json()["data"]["answer"]


def test_explicit_zone_aggregate_is_not_answered_with_frame_measurement():
    from trainer.demo_api.tests.test_dialogue_conversation import client,body,DialoguePipeline
    from trainer.demo_api.sequence_store import SequenceStore
    pipeline=DialoguePipeline()
    pipeline.plan_copilot_dialogue=lambda *a,**kw:{"kind":"lookup","intent":"order_direction","scope":"zone"}
    value=json.loads(snapshot_context(trace([[0,1,0],[0,1.01,0],[0,1.02,0]])).split(':',1)[1])
    claims={"record_direction":"diagonal","zones":{"1:1":{"map_num":"1","zone_id":"1","direction":"horizontal"}}}
    value.update(dialogue_context=claims,dialogue_signature=sign_context('dialogue',claims))
    response=client(pipeline,SequenceStore(":memory:")).post('/api/v1/copilot/query',json=body('선택 Zone의 대표 도포 방향은?',context='motion_viewer_playback:'+json.dumps(value)))
    answer=response.json()['data']['answer']
    assert '수평' in answer and '대표' in answer and '수직' not in answer


def test_csv_clock_offset_is_aligned_with_normalized_viewer_playhead():
    from trainer.demo_api.playback_direction import build_playback_snapshot,playback_direction_from_context
    claims,signature=build_playback_snapshot(trace([[0,1,0],[0,1.01,0],[0,1.02,0]],start=4),record_id='SHIFTED',duration_seconds=.2,time_offset_seconds=4)
    ctx='motion_viewer_playback:'+json.dumps({'frame_direction_context':claims,'frame_direction_signature':signature,'playback_context':{'playhead_seconds':.1}})
    assert playback_direction_from_context(ctx)['label']=='수직'


def test_missing_csv_clock_does_not_invent_a_current_frame_direction():
    from trainer.demo_api.playback_direction import build_playback_snapshot,playback_direction_from_context
    claims,signature=build_playback_snapshot(trace([[0,1,0],[0,1.01,0]]),record_id='NO-CLOCK',duration_seconds=1,clock_valid=False)
    ctx='motion_viewer_playback:'+json.dumps({'frame_direction_context':claims,'frame_direction_signature':signature,'playback_context':{'playhead_seconds':.1}})
    result=playback_direction_from_context(ctx)
    assert result['analysis_status']=='unavailable' and result['reason_code']=='invalid_clock'


def test_curved_record_aggregate_uses_korean_irregular_label():
    from trainer.demo_api.playback_direction import aggregate_direction
    claims={'record_direction':'curved_irregular'}
    context='motion_viewer_playback:'+json.dumps({'dialogue_context':claims,'dialogue_signature':sign_context('dialogue',claims)})
    answer,result=aggregate_direction(context,'전체 기록의 대표 도포 방향은?')
    assert result['analysis_status']=='available' and result['classification']=='irregular' and '비정형' in answer
