"""Dataset selectors stay dependent and Query receives authenticated selected-record facts."""
import hashlib
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from trainer.demo_api.copilot_router import create_copilot_router
from trainer.demo_api.dialogue import viewer_claims, authenticated_facts
from trainer.demo_api.query_chat import create_query_chat_router
from trainer.demo_api.query_chat_data import QueryChatDatasets
from trainer.demo_api.sequence_store import SequenceStore
from trainer.demo_api.training_data_catalog import TrainingDataCatalog


@pytest.fixture
def datasets(tmp_path):
    output = tmp_path / 'datasets'
    (output / 'records').mkdir(parents=True)
    definitions = [('alice', '훈련자 A', 'spray', '3', 301), ('alice', '훈련자 A', 'spray', '7', 701),
                   ('alice', '훈련자 A', 'blasting', '3', 302), ('bob', '훈련자 B', 'grinding', '5', 501),
                   ('alice', '훈련자 A', 'spray', '1-R-A', 111)]
    rows=[]
    for i, (worker, name, process, mode, speed) in enumerate(definitions, 1):
        dataset_id = f'{i:032x}'
        meta = {'dataset_id': dataset_id, 'dataset_name': f'{mode}_{worker}_{i}', 'worker_id': worker,
                'worker_name': name, 'process': process, 'scenario_id': mode, 'mode_label': f'학습모드 {mode}',
                'acquired_at': '2026-11-16', 'quality_status': 'needs_review', 'zone_id_available': i != 4,
                'zone_count': 2 if i != 4 else 0, 'heightmap_count': 0, 'frame_count': 100}
        features = {'frame_count': 100, 'duration_seconds': 12, 'sequence_count': 4,
                    'zone_order': [{'map_num': 1, 'zone_id': '3'}, {'map_num': 1, 'zone_id': '1'}] if i != 4 else [],
                    'zones': [{'map_num': 1, 'zone_id': '3', 'frame_count': 50,
                               'distance_mm': {'median': 250}, 'speed_mms': {'median': speed}},
                              {'map_num': 1, 'zone_id': '1', 'frame_count': 50,
                               'distance_mm': {'median': 220}, 'speed_mms': {'median': speed+10}}] if i != 4 else []}
        artifact = {'metadata': meta, 'features': features, 'coating': {'maps': []}, 'quality': {'status': 'needs_review'}}
        content = json.dumps(artifact, ensure_ascii=False).encode()
        (output / 'records' / f'{dataset_id}.json').write_bytes(content)
        rows.append({**meta, 'artifact_sha256': hashlib.sha256(content).hexdigest(), 'json_bytes': len(content)})
    (output / 'catalog.json').write_text(json.dumps({'records': rows}), encoding='utf-8')
    return TrainingDataCatalog(output, tmp_path / 'source')


@pytest.fixture
def client(datasets):
    app = FastAPI()
    app.include_router(create_query_chat_router(datasets))
    app.include_router(create_copilot_router(None, store=SequenceStore(':memory:')))
    return TestClient(app)


def test_cascading_options_never_mix_workers_processes_or_modes(client):
    root = client.get('/query-chat/api/options').json()
    assert {row['id'] for row in root['workers']} == {'alice', 'bob'}
    assert root['processes'] == root['modes'] == root['sessions'] == []
    selected = client.get('/query-chat/api/options?worker_id=alice').json()
    assert {row['id'] for row in selected['processes']} == {'spray', 'blasting'}
    assert selected['sessions'] == []
    selected = client.get('/query-chat/api/options?worker_id=alice&process=spray').json()
    assert {row['id'] for row in selected['modes']} == {'3', '7'}
    assert selected['sessions'] == []
    selected = client.get('/query-chat/api/options?worker_id=alice&process=spray&scenario_id=7').json()
    assert [row['id'] for row in selected['sessions']] == [f'{2:032x}']
    assert client.get('/query-chat/api/options?worker_id=bob&process=spray&scenario_id=3').json()['sessions'] == []


def test_selected_context_matches_artifact_and_is_signed(client):
    result = client.get(f'/query-chat/api/context?dataset_id={2:032x}&zone_key=1:3').json()
    assert result['scenario_id'] == '7' and result['session_id'] == 'CHATDATA:' + f'{2:032x}'
    assert {row['id'] for row in result['zones']} == {'1:1', '1:3'}
    assert result['selection']['record'] == '7_alice_2'
    assert result['selection']['quality_status'] == 'needs_review'
    payload, facts = viewer_claims(result['context'])
    assert payload['playback_context']['zone_id'] == '3'
    assert facts['scenario_id'] == '7' and facts['process'] == 'spray'
    assert facts['zones']['1:3']['speed_mms'] == 701
    assert facts['actual_order'] == ['3', '1']
    assert authenticated_facts(result['context'])['selected_zone']['distance_mm'] == 250
    assert viewer_claims(result['context'].replace('701.0', '999.0')) is None
    assert result['available']['pose'] is False


def test_query_reads_selected_zone_instead_of_unrelated_live_session(client):
    selected = client.get(f'/query-chat/api/context?dataset_id={2:032x}&zone_key=1:3').json()
    response = client.post('/api/v1/copilot/query', json={
        'user_id': 'query-chat', 'session_id': selected['session_id'], 'scenario_id': selected['scenario_id'],
        'task_type': 'knowledge_search', 'prompt': '지금 Zone 3 속도는?', 'context': selected['context']})
    assert response.status_code == 200, response.text
    answer = response.json()['data']['answer']
    assert '701' in answer and 'Zone 3' in answer
    assert '301' not in answer


def test_missing_zone_data_still_allows_record_context(client):
    response = client.get(f'/query-chat/api/context?dataset_id={4:032x}')
    assert response.status_code == 200
    data = response.json()
    assert data['zones'] == [] and data['selection']['zone_id_available'] is False
    assert data['selection']['process'] == 'grinding'
    assert data['available']['zone_statistics'] is False
    assert client.get(f'/query-chat/api/context?dataset_id={4:032x}&zone_key=1:3').status_code == 422


def test_invalid_ids_and_cross_record_zone_are_rejected(client):
    assert client.get('/query-chat/api/context?dataset_id=../../etc/passwd').status_code == 422
    assert client.get('/query-chat/api/context?dataset_id=' + 'f'*32).status_code == 404
    assert client.get(f'/query-chat/api/context?dataset_id={1:032x}&zone_key=2:3').status_code == 422
    assert client.get('/query-chat/api/options?scenario_id=11').status_code == 422


def test_catalog_refreshes_when_published_catalog_changes(datasets):
    service = QueryChatDatasets(datasets)
    assert len(service.options()['workers']) == 2
    path = datasets.root / 'catalog.json'
    index = json.loads(path.read_text())
    index['records'] = [row for row in index['records'] if row['worker_id'] != 'bob']
    path.write_text(json.dumps(index))
    assert [row['id'] for row in service.options()['workers']] == ['alice']


def test_repeated_zone_numbers_never_mix_measurements_between_faces(datasets):
    dataset_id = f'{1:032x}'
    path = datasets.root / 'records' / f'{dataset_id}.json'
    artifact = json.loads(path.read_text(encoding='utf-8'))
    artifact['features']['zones'].append({'map_num': 2, 'zone_id': '3', 'frame_count': 50,
                                          'distance_mm': {'median': 450}, 'speed_mms': {'median': 999}})
    content = json.dumps(artifact, ensure_ascii=False).encode()
    path.write_bytes(content)
    index_path = datasets.root / 'catalog.json'
    index = json.loads(index_path.read_text())
    index['records'][0].update(json_bytes=len(content), artifact_sha256=hashlib.sha256(content).hexdigest())
    index_path.write_text(json.dumps(index))
    service = QueryChatDatasets(datasets)
    whole = service.context(dataset_id)
    assert whole['selection_requires_zone'] is True
    claims = viewer_claims(whole['context'])[1]
    assert claims['ambiguous_zone_ids'] == ['3']
    assert 'speed_mms' not in claims['zones']['1:3'] and 'speed_mms' not in claims['zones']['2:3']
    selected = service.context(dataset_id, '2:3')
    claims = viewer_claims(selected['context'])[1]
    assert claims['zones']['2:3']['speed_mms'] == 999
    assert '1:3' not in claims['zones']
    assert selected['selection_requires_zone'] is False
    assert {row['id'] for row in selected['zones']} == {'1:1', '1:3', '2:3'}
    app = FastAPI()
    app.include_router(create_copilot_router(None, store=SequenceStore(':memory:')))
    client = TestClient(app)
    def ask(context):
        return client.post('/api/v1/copilot/query', json={'user_id': 'query-chat', 'task_type': 'knowledge_search',
            'prompt': '지금 Zone 3 속도는?', 'context': context}).json()['data']['answer']
    assert '면과 Zone을 선택' in ask(whole['context'])
    assert '301' not in ask(whole['context']) and '999' not in ask(whole['context'])
    assert '999' in ask(selected['context'])


def test_known_legacy_and_sql_worker_ids_share_one_selector_without_changing_source(datasets):
    index_path = datasets.root / 'catalog.json'
    index = json.loads(index_path.read_text())
    for row, worker in zip(index['records'][:2], ['jeon_hyeonggi', 'tsb_user_86bc']):
        row.update(worker_id=worker, worker_name='전형기', scenario_id='3')
        path = datasets.root / 'records' / f"{row['dataset_id']}.json"
        artifact = json.loads(path.read_text(encoding='utf-8'))
        artifact['metadata'].update(worker_id=worker, worker_name='전형기', scenario_id='3')
        content = json.dumps(artifact, ensure_ascii=False).encode()
        path.write_bytes(content)
        row.update(json_bytes=len(content), artifact_sha256=hashlib.sha256(content).hexdigest())
    index_path.write_text(json.dumps(index))
    service = QueryChatDatasets(datasets)
    options = service.options('jeon_hyeonggi', 'spray', '3')
    assert [row for row in options['workers'] if row['label'] == '전형기'] == [{'id': 'jeon_hyeonggi', 'label': '전형기'}]
    assert {row['id'] for row in options['sessions']} == {f'{1:032x}', f'{2:032x}'}
    selection = service.context(f'{2:032x}')['selection']
    assert selection['worker_id'] == 'jeon_hyeonggi'
    assert selection['source_worker_id'] == 'tsb_user_86bc'
