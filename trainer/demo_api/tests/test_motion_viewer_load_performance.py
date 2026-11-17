"""Viewer selection reuses verified summaries without rescanning the catalog."""
from collections import Counter
import cProfile
from pathlib import Path

import pytest

from trainer.demo_api import motion_viewer as mv
from trainer.demo_api.tests.test_training_motion_catalog import publish
from trainer.demo_api.training_data_catalog import TrainingDataCatalog
from trainer.demo_api.training_motion_catalog import TrainingMotionCatalog


def test_repeated_viewer_profiles_read_each_unchanged_artifact_once(monkeypatch, tmp_path):
    """Motion, alerts and arm-pose open the same selection independently."""
    output = tmp_path / 'built'
    first = publish(tmp_path, output, session='4_2001_1')
    second = publish(tmp_path, output, session='4_2002_1')
    motion = TrainingMotionCatalog(tmp_path, TrainingDataCatalog(output, tmp_path))
    reads = Counter()
    original_open = Path.open

    def counted_open(path, mode='r', *args, **kwargs):
        if path.is_relative_to(output) and 'r' in mode:
            reads[path.name] += 1
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', counted_open)
    for _ in range(3):
        profile = motion.profile('new_worker', 'spray', 4)
        assert [session['session_id'] for session in profile['sessions']] == ['4_2001_1', '4_2002_1']
        assert profile['aggregate']['zones']['1:1']['speed_mms']['median'] == 300
    assert reads['catalog.json'] == 1
    assert reads[first['dataset_id'] + '.json'] == 1
    assert reads[second['dataset_id'] + '.json'] == 1


def test_cached_viewer_session_does_not_survive_artifact_tampering(monkeypatch, tmp_path):
    """Cache hits must still check the file stamp and its registered digest."""
    output = tmp_path / 'built'
    row = publish(tmp_path, output)
    motion = TrainingMotionCatalog(tmp_path, TrainingDataCatalog(output, tmp_path))
    assert motion.profile('new_worker', 'spray', 4)['sessions'][0]['analysis']['zones']['1:1']['speed_mms'] == 300
    artifact_path = output / 'records' / (row['dataset_id'] + '.json')
    before = artifact_path.stat()
    content = artifact_path.read_bytes().replace(b'300', b'900')
    artifact_path.write_bytes(content)
    # Make the update observable even on filesystems with coarse clock ticks.
    import os
    os.utime(artifact_path, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000))
    assert motion.profile('new_worker', 'spray', 4) is None


def test_cached_viewer_session_refreshes_on_catalog_republication(tmp_path):
    output = tmp_path / 'built'
    publish(tmp_path, output)
    motion = TrainingMotionCatalog(tmp_path, TrainingDataCatalog(output, tmp_path))
    first = motion.profile('new_worker', 'spray', 4)
    first['sessions'][0]['analysis']['zones']['1:1']['speed_mms'] = 900
    # A consumer must not mutate the summary retained for another API request.
    assert motion.profile('new_worker', 'spray', 4)['sessions'][0]['analysis']['zones']['1:1']['speed_mms'] == 300
    publish(tmp_path, output, session='4_2002_1')
    assert [session['session_id'] for session in motion.profile('new_worker', 'spray', 4)['sessions']] == ['4_2001_1', '4_2002_1']


def test_normalized_row_keeps_casefold_and_last_duplicate_header_value():
    row = {' Zone ID ': '2', 'ZONE id': '7', ' Straße ': 'folded',
           ' RAY POINT X ': '1.25', ' Ray Point Y': '2.5', 'Ray Point Z ': '3.75'}
    normalized = mv._normalized_row(row)
    assert dict(normalized) == {'zone id': '7', 'strasse': 'folded',
                                'ray point x': '1.25', 'ray point y': '2.5', 'ray point z': '3.75'}
    assert normalized.get('absent') is None
    assert normalized.get('absent', 'fallback') == 'fallback'
    assert mv._ray_point(row) == [1.25, 2.5, 3.75]
    with pytest.raises(TypeError):
        normalized['zone id'] = '9'
    assert row['ZONE id'] == '7'


def test_repeated_csv_headers_are_casefolded_once_instead_of_for_every_frame():
    """Wide recording headers cost millions of redundant casefold calls."""
    headers = [f' unused motion performance column {index} ' for index in range(160)]
    headers += [' Zone ID ', ' RAY POINT X ', ' RAY POINT Y ', ' RAY POINT Z ']
    profiler = cProfile.Profile()
    profiler.enable()
    try:
        for frame in range(20):
            row = dict.fromkeys(headers, 'unused')
            row.update({' Zone ID ': str(frame), ' RAY POINT X ': '1', ' RAY POINT Y ': '2', ' RAY POINT Z ': '3'})
            assert mv._normalized_row(row).get('zone id') == str(frame)
            assert mv._ray_point(row) == [1.0, 2.0, 3.0]
    finally:
        profiler.disable()
    folds = sum(entry.callcount for entry in profiler.getstats()
                if isinstance(entry.code, str) and 'casefold' in entry.code)
    assert folds <= len(headers), f'{folds} header casefold calls for {len(headers)} distinct headers'
