"""Progress reflects consumed CSV rows, actual list units, and bounded job state."""
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from uuid import uuid4

import pytest

from trainer.demo_api.tests.test_motion_viewer import _client


def test_progress_store_is_private_bounded_timed_and_cancellable():
    from trainer.demo_api.motion_viewer_progress import LoadProgressStore, LoadStopped
    clock = [0.0]
    store = LoadProgressStore(capacity=1, ttl_seconds=10, now=lambda: clock[0])
    first = store.start(uuid4().hex)
    first.update('csv_frames', completed=2, total=5, unit='rows')
    snapshot = store.snapshot(first.load_id)
    assert snapshot['completed'] == 2 and snapshot['total'] == 5
    assert snapshot['unit'] == 'rows' and snapshot['status'] == 'running'
    with pytest.raises(ValueError):
        store.start(first.load_id)
    second = store.start(uuid4().hex)
    with pytest.raises(LoadStopped):
        first.checkpoint()
    assert store.snapshot(first.load_id)['status'] == 'unavailable'
    second.cancel()
    assert store.snapshot(second.load_id)['status'] == 'cancelled'
    clock[0] = 11
    assert store.snapshot(second.load_id)['status'] == 'unavailable'


def test_csv_counter_reports_real_rows_and_does_not_invent_unknown_totals():
    from trainer.demo_api.motion_viewer_progress import LoadProgressStore, bind_progress, progress_rows
    store = LoadProgressStore(); job = store.start(uuid4().hex)
    with bind_progress(job):
        rows = progress_rows(iter([10, 20, 30]), stage='csv_count', path='private/not-returned.csv')
        assert next(rows) == 10
        assert next(rows) == 20
        assert store.snapshot(job.load_id)['completed'] == 1
        assert store.snapshot(job.load_id)['total'] is None
        assert list(rows) == [30]
        assert store.snapshot(job.load_id)['completed'] == 3
        assert store.snapshot(job.load_id)['phase_complete'] is True
        known = progress_rows(iter([1, 2, 3]), stage='csv_frames', path='private/not-returned.csv')
        next(known); next(known)
        snapshot = store.snapshot(job.load_id)
        assert snapshot['completed'] == 1 and snapshot['total'] == 3
        assert 'private' not in str(snapshot)
        assert list(known) == [3]


def test_progress_endpoint_observes_live_csv_work_and_completion(monkeypatch, tmp_path):
    from trainer.demo_api import motion_viewer as viewer
    client = _client(tmp_path)
    entered, release = Event(), Event()
    original = viewer._world_joints
    calls = []
    def paused(row):
        calls.append(True)
        if len(calls) == 2:
            entered.set()
            assert release.wait(10)
        return original(row)
    monkeypatch.setattr(viewer, '_world_joints', paused)
    load_id = uuid4().hex
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(client.get, '/motion-viewer/api/motion',
            params={'process': 'spray', 'mode_id': 4, 'session_id': '4_1001_1'},
            headers={'X-Motion-Load-ID': load_id})
        try:
            assert entered.wait(10)
            snapshot = client.get('/motion-viewer/api/load-progress', params={'load_id': load_id}).json()
            assert snapshot['stage'] == 'csv_frames'
            assert snapshot['completed'] == 1 and snapshot['total'] == 5
            assert '4_1001_1' not in str(snapshot) and str(tmp_path) not in str(snapshot)
        finally:
            release.set()
        assert future.result(timeout=10).status_code == 200
    assert client.get('/motion-viewer/api/load-progress', params={'load_id': load_id}).json()['status'] == 'completed'


def test_failed_load_has_terminal_private_progress_state(tmp_path):
    client = _client(tmp_path); load_id = uuid4().hex
    response = client.get('/motion-viewer/api/motion', params={
        'process': 'spray', 'mode_id': 4, 'session_id': 'missing-private-session'},
        headers={'X-Motion-Load-ID': load_id})
    assert response.status_code == 404
    result = client.get('/motion-viewer/api/load-progress', params={'load_id': load_id}).json()
    assert result['status'] == 'failed'
    assert 'missing-private-session' not in str(result)


def test_duplicate_load_id_does_not_reuse_or_overwrite_finished_state(tmp_path):
    client = _client(tmp_path); load_id = uuid4().hex
    query = {'process': 'spray', 'mode_id': 4, 'session_id': '4_1001_1'}
    assert client.get('/motion-viewer/api/motion', params=query, headers={'X-Motion-Load-ID': load_id}).status_code == 200
    assert client.get('/motion-viewer/api/motion', params=query, headers={'X-Motion-Load-ID': load_id}).status_code == 409
    assert client.get('/motion-viewer/api/load-progress', params={'load_id': load_id}).json()['status'] == 'completed'


def test_disconnect_cancels_server_worker_at_its_next_checkpoint(monkeypatch, tmp_path):
    import asyncio
    from fastapi import HTTPException
    from trainer.demo_api import motion_viewer as viewer
    client = _client(tmp_path)
    endpoint = next(route.endpoint for route in client.app.routes if getattr(route, 'path', '') == '/motion-viewer/api/motion')
    entered, release = Event(), Event()
    original = viewer._world_joints
    def paused(row):
        entered.set()
        assert release.wait(10)
        return original(row)
    monkeypatch.setattr(viewer, '_world_joints', paused)
    class LeftClient:
        async def is_disconnected(self):
            return entered.is_set()
    load_id = uuid4().hex
    async def run():
        try:
            with pytest.raises(HTTPException) as error:
                await endpoint(request=LeftClient(), worker_id='jeon_hyeonggi', process='spray', mode_id=4,
                               session_id='4_1001_1', load_id=load_id)
            assert error.value.status_code == 499
        finally:
            release.set()
        await asyncio.sleep(0.05)
    asyncio.run(run())
    assert client.get('/motion-viewer/api/load-progress', params={'load_id': load_id}).json()['status'] == 'cancelled'


def test_job_cancellation_interrupts_rows_without_retaining_partial_success():
    from trainer.demo_api.motion_viewer_progress import LoadProgressStore, LoadStopped, bind_progress, progress_rows
    store = LoadProgressStore(); job = store.start(uuid4().hex)
    with bind_progress(job):
        rows = progress_rows(iter(range(100)), stage='csv_frames', path='private')
        assert next(rows) == 0
        job.cancel()
        with pytest.raises(LoadStopped):
            next(rows)
    assert store.snapshot(job.load_id)['status'] == 'cancelled'


def test_blank_and_quoted_multiline_csv_records_use_one_consistent_total(monkeypatch, tmp_path):
    import csv
    from trainer.demo_api import motion_viewer as viewer
    from trainer.demo_api.motion_viewer_progress import LoadProgressStore, LoadJob, bind_progress
    from trainer.demo_api.tests.test_motion_viewer import _write_csv
    source = tmp_path / 'mixed.csv'; _write_csv(source, offset=0)
    with source.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle); fields = reader.fieldnames + ['note']; rows = list(reader)
    rows[0]['note'] = 'quoted first line\nquoted second line'
    with source.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for index, row in enumerate(rows):
            if index in {0, 2}:
                handle.write('\r\n')  # DictReader skips these empty records.
            writer.writerow(row)
    snapshots = []
    original = LoadJob.update
    def observe(self, stage, **kwargs):
        original(self, stage, **kwargs)
        if stage == 'csv_frames':
            snapshots.append(self.store.snapshot(self.load_id))
    monkeypatch.setattr(LoadJob, 'update', observe)
    store = LoadProgressStore(); job = store.start(uuid4().hex)
    with bind_progress(job):
        frames = viewer._frames(source, 5, 'spray')
    assert len(frames) == 5
    assert snapshots and all(value['total'] == 5 for value in snapshots)
    assert snapshots[-1]['completed'] == 5 and snapshots[-1]['phase_complete'] is True
