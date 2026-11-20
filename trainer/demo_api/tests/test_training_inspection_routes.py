from trainer.demo_api.tests.test_db_admin import admin  # noqa: F401: shared isolated fixture


def test_exam_assets_are_public_and_cached_with_revalidation(admin):
    client, _, _ = admin
    for suffix in ('js', 'css'):
        response = client.get('/admin/assets/training_inspection.' + suffix)
        assert response.status_code == 200
        assert response.headers['cache-control'] == 'no-cache'


def test_csv_transformation_assets_are_public_without_exposing_api_routes(admin):
    client, _, _ = admin
    for asset in ('training_transform.js', 'training_transform.css', 'training_transform_scene.js'):
        response = client.get('/admin/assets/' + asset)
        assert response.status_code == 200
        assert response.headers['cache-control'] == 'no-cache'
    assert not any('training_transform' in path for path in client.get('/openapi.json').json()['paths'])


def test_exam_transformation_shell_preserves_verification_and_read_only_controls(admin):
    client, _, _ = admin
    body = client.get('/admin/training-data?view=verification').text
    for element in ('tt-root', 'tt-record', 'tt-source', 'tt-visual', 'tt-json', 'tt-status',
                    'tt-view-zone', 'tt-view-metrics', 'tt-view-coating', 'tv-kpi-valid', 'tv-start'):
        assert f'id="{element}"' in body
    assert '훈련 기록 한 건이 학습 정보로 바뀌는 과정' in body
    assert '학습데이터 JSON' in body
    assert body.index('/admin/assets/training_data.js') < body.index('/admin/assets/training_transform_scene.js')
    assert body.index('/admin/assets/training_transform_scene.js') < body.index('/admin/assets/training_transform.js')
    assert body.index('/admin/assets/training_transform.js') < body.index('/admin/assets/training_verification.js')


def test_exam_actions_share_existing_admin_boundary_and_stay_out_of_swagger(admin):
    client, _, _ = admin
    assert client.post('/admin/api/training-data/' + 'a' * 32 + '/reconvert').status_code == 401
    assert not any('inspection' in path or 'reconvert' in path for path in client.get('/openapi.json').json()['paths'])
