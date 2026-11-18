"""Only source-validated, already retained browser payloads are reusable."""
import os

from trainer.demo_api.tests.test_motion_viewer import _client


def test_motion_payload_validation_token_expires_when_source_changes(tmp_path):
    client = _client(tmp_path)
    query = {'process': 'spray', 'mode_id': 4, 'session_id': '4_1001_1'}
    payload = client.get('/motion-viewer/api/motion', params=query).json()
    token = payload['viewer_cache']['token']
    assert client.get('/motion-viewer/api/cache-status', params={'token': token}).json() == {'valid': True}
    source = tmp_path / 'raw/spray/learning/mode_04/4_1001_1.csv'
    before = source.stat()
    os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000))
    assert client.get('/motion-viewer/api/cache-status', params={'token': token}).json() == {'valid': False}
    assert client.get('/motion-viewer/api/cache-status', params={'token': 'unknown'}).json() == {'valid': False}


def test_motion_payload_validation_also_watches_reference_profile(tmp_path):
    client = _client(tmp_path)
    payload = client.get('/motion-viewer/api/motion', params={
        'process': 'spray', 'mode_id': 4, 'session_id': '4_1001_1'}).json()
    profile = tmp_path / 'generated/profiles/spray/learning/mode_04.json'
    profile.write_text(profile.read_text(encoding='utf-8') + '\n', encoding='utf-8')
    assert client.get('/motion-viewer/api/cache-status', params={
        'token': payload['viewer_cache']['token']}).json() == {'valid': False}


def test_payload_validation_registry_is_bounded_and_timed(tmp_path):
    from trainer.demo_api.motion_viewer_cache import PayloadRevisions
    source = tmp_path / 'a.csv'; source.write_text('source')
    clock = [0.0]
    registry = PayloadRevisions(max_entries=1, ttl_seconds=10, now=lambda: clock[0])
    stamp = registry.stamp([source])
    first = registry.register(stamp)
    assert registry.valid(first)
    second = registry.register(stamp)
    assert not registry.valid(first) and registry.valid(second)
    clock[0] = 11
    assert not registry.valid(second)


def test_viewer_shells_load_real_payload_cache_and_do_not_claim_server_payload_cache(tmp_path):
    client = _client(tmp_path)
    for page in ['/motion-viewer/v1', '/motion-viewer']:
        html = client.get(page).text
        assert 'src="/motion-viewer/assets/motion_viewer_cache.js"' in html
        assert "payloadCache.load(motionUrl()," in html
        assert 'id="recordCacheLegend"' in html and '✓ 이 창에 보관됨' in html
        for key in ['loadingProgress', 'loadingStage', 'loadingUnits', 'loadingCancel']:
            assert f'id="{key}"' in html
    assert client.get('/motion-viewer/assets/motion_viewer_cache.js').status_code == 200
    payload = client.get('/motion-viewer/api/motion', params={
        'process': 'spray', 'mode_id': 4, 'session_id': '4_1001_1'}).json()
    assert payload['viewer_cache']['scope'] == 'browser_payload'


def test_initial_motion_load_builds_training_summary_once(monkeypatch, tmp_path):
    from trainer.demo_api.training_motion_catalog import TrainingMotionCatalog
    original = TrainingMotionCatalog.profile
    reads = []
    def profile(self, *args, **kwargs):
        reads.append(args[:3])
        return original(self, *args, **kwargs)
    monkeypatch.setattr(TrainingMotionCatalog, 'profile', profile)
    response = _client(tmp_path).get('/motion-viewer/api/motion', params={
        'process': 'spray', 'mode_id': 4, 'session_id': '4_1001_1'})
    assert response.status_code == 200
    assert len(reads) == 1, 'freshness checks must not rebuild a large training summary twice'


def test_profile_change_during_motion_build_does_not_receive_reuse_token(monkeypatch, tmp_path):
    from trainer.demo_api import motion_viewer as viewer
    client = _client(tmp_path)
    original = viewer._frames
    changed = []
    def frames(*args, **kwargs):
        if not changed:
            profile = tmp_path / 'generated/profiles/spray/learning/mode_04.json'
            profile.write_text(profile.read_text(encoding='utf-8') + '\n', encoding='utf-8')
            changed.append(True)
        return original(*args, **kwargs)
    monkeypatch.setattr(viewer, '_frames', frames)
    response = client.get('/motion-viewer/api/motion', params={
        'process': 'spray', 'mode_id': 4, 'session_id': '4_1001_1'})
    assert response.status_code == 200
    assert 'viewer_cache' not in response.json()
