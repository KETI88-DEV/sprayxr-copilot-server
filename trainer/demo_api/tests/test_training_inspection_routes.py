from trainer.demo_api.tests.test_db_admin import admin  # noqa: F401: shared isolated fixture


def test_exam_assets_are_public_and_cached_with_revalidation(admin):
    client, _, _ = admin
    for suffix in ('js', 'css'):
        response = client.get('/admin/assets/training_inspection.' + suffix)
        assert response.status_code == 200
        assert response.headers['cache-control'] == 'no-cache'


def test_exam_actions_share_existing_admin_boundary_and_stay_out_of_swagger(admin):
    client, _, _ = admin
    assert client.post('/admin/api/training-data/' + 'a' * 32 + '/reconvert').status_code == 401
    assert not any('inspection' in path or 'reconvert' in path for path in client.get('/openapi.json').json()['paths'])
