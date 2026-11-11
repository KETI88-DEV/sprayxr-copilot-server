"""Raw CSV features stay useful without inventing Zone or map identifiers."""

import base64
import csv
import json
import struct
import zlib

import pytest

from trainer.demo_api.training_data_build import _features


def write_record(tmp_path, columns, rows, name='record.csv'):
    path = tmp_path / name
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(rows)
    return path


def png_payload(*, large=False):
    """A genuine cropped PNG, with optional metadata exceeding csv's default limit."""
    def chunk(kind, data):
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data))
    raw = b'\x89PNG\r\n\x1a\n'
    raw += chunk(b'IHDR', struct.pack('>IIBBBBB', 491, 308, 8, 2, 0, 0, 0))
    if large:
        text_bytes = 150_000 if large is True else large
        raw += chunk(b'tEXt', b'description\x00' + b'x' * text_bytes)
    raw += chunk(b'IDAT', zlib.compress((b'\0' + b'\0' * (491 * 3)) * 308))
    raw += chunk(b'IEND', b'')
    return {'map_num': '01', 'x0': 125, 'y0': 87, 'width': 1472, 'height': 924,
            'img_width': 491, 'img_height': 308,
            'format': 'v2;thickness=RG16;alpha=B8;scale_um=20000',
            'encoding': 'png;base64', 'data': base64.b64encode(raw).decode('ascii')}


def test_raw_without_zone_retains_global_metrics_and_source_bytes(tmp_path):
    path = write_record(tmp_path,
        ['시간(초)', '분사', 'Distance (Input)', 'Speed (Input)', 'Distance (Real)', 'Speed (Real)',
         'Ray Point X', 'Ray Point Y', 'Ray Point Z'],
        [[0, 'Off', 999, 999, 999, 999, 99, 99, 99],
         [1, 'On', 200, 400, 210, 410, 1, 2, 3],
         [2, 'On', 400, 800, 430, 830, 2, 4, 6],
         [3, 'Off', 999, 999, 999, 999, 99, 99, 99],
         [4, 'On', -1, 'nan', '', '', '', '', '']])
    original = path.read_bytes()
    result = _features(path, 'spray', '3')
    assert result['frame_count'] == 5 and result['spraying_frame_count'] == 3
    assert result['sequence_count'] == 2 and result['duration_seconds'] == 4
    assert result['zone_id_column_present'] is False
    assert result['zone_id_available'] is False and result['zone_id_status'] == 'absent'
    assert result['mapped_spraying_frame_count'] == 0 and result['unmapped_spraying_frames'] == 3
    assert result['zones'] == [] and result['zone_order'] == []
    assert result['global_metrics']['distance_mm'] == {
        'count': 2, 'mean': 300, 'median': 300, 'p25': 250, 'p75': 350}
    assert result['global_metrics']['speed_mms']['median'] == 600
    assert result['global_metrics']['distance_real_mm']['median'] == 320
    assert result['global_metrics']['speed_real_mms']['median'] == 620
    assert result['global_metrics']['ray_point_bounds_m'] == {'min': [1, 2, 3], 'max': [2, 4, 6]}
    assert path.read_bytes() == original


def test_zone_header_with_no_positive_zone_is_absent(tmp_path):
    path = write_record(tmp_path, ['time', '분사', 'Zone ID'],
                        [[i, 'On', value] for i, value in enumerate(['', '-1', '0', 'nan', 'null', 'bad', '-2'])])
    result = _features(path, 'spray', '3')
    assert result['zone_id_column_present'] is True
    assert result['zone_id_available'] is False and result['zone_id_status'] == 'absent'
    assert result['mapped_spraying_frame_count'] == 0
    assert result['unmapped_spraying_frames'] == 7 and result['zones'] == []


def test_real_zone_and_zero_padded_map_are_normalized_without_legacy_remapping(tmp_path):
    path = write_record(tmp_path, ['time', ' 분사 ', ' Zone ID ', 'Ray Texture Map Number', 'Distance (Input)'],
                        [[0, 'On', '1', '01', 200], [1, 'On', '1', 1, 400], [2, 'On', '-1', 1, 600]])
    result = _features(path, 'spray', '2')
    assert result['zone_id_available'] is True and result['zone_id_status'] == 'present'
    assert result['mapped_spraying_frame_count'] == 2 and result['unmapped_spraying_frames'] == 1
    assert result['zones'][0]['zone_id'] == '1' and result['zones'][0]['map_num'] == 1
    assert len(result['zones']) == 1 and result['zones'][0]['frame_count'] == 2
    assert result['zones'][0]['distance_mm']['median'] == 300
    assert result['global_metrics']['distance_mm']['median'] == 400


def test_real_zone_on_non_spraying_frame_still_establishes_source_availability(tmp_path):
    path = write_record(tmp_path, ['time', '분사', 'Zone ID'], [[0, 'Off', '2'], [1, 'On', '-1']])
    result = _features(path, 'grinding', '3')
    assert result['zone_id_available'] is True and result['zone_id_status'] == 'present'
    assert result['mapped_spraying_frame_count'] == 0 and result['zones'] == []


def test_named_map_labels_are_retained_and_do_not_split_spray_runs(tmp_path):
    path = write_record(tmp_path, ['time', '분사', 'Zone ID', 'Ray Texture Map Number'],
        [[0, 'On', '1', 'Grind TrainingMetal'], [1, 'On', '2', 'Grind TrainingMetal'],
         [2, 'On', '1', 'Grind OtherMetal'], [3, 'Off', '-1', ''],
         [4, 'On', '1', 'Grind TrainingMetal']])
    result = _features(path, 'grinding', '3')
    assert result['sequence_count'] == 2
    assert [(zone['map_num'], zone['map_label'], zone['zone_id'], zone['frame_count'])
            for zone in result['zones']] == [
        (None, 'Grind TrainingMetal', '1', 2), (None, 'Grind TrainingMetal', '2', 1),
        (None, 'Grind OtherMetal', '1', 1)]
    assert result['zone_order'][0] == {'map_num': None, 'map_label': 'Grind TrainingMetal', 'zone_id': '1'}


def test_unmapped_raw_record_retains_first_seen_named_and_numeric_maps(tmp_path):
    path = write_record(tmp_path, ['time', '분사', 'Ray Texture Map Number'],
        [[0, 'On', 'Grind TrainingMetal'], [1, 'On', 'Grind OtherMetal'],
         [2, 'On', 'Grind TrainingMetal'], [3, 'On', '01'], [4, 'On', '1'], [5, 'On', '']])
    result = _features(path, 'grinding', '3')
    assert result['zone_id_available'] is False and result['zones'] == []
    assert result['sequence_count'] == 1 and result['unmapped_spraying_frames'] == 6
    assert result['map_order'] == [
        {'map_num': None, 'map_label': 'Grind TrainingMetal'},
        {'map_num': None, 'map_label': 'Grind OtherMetal'}, {'map_num': 1}]
    assert result['map_labels'] == ['Grind TrainingMetal', 'Grind OtherMetal']


def test_sparse_heatmaps_keep_original_png_crop_metadata_and_latest_spray_run(tmp_path):
    image = png_payload()
    payload = json.dumps([image])
    path = write_record(tmp_path, ['time', '분사', 'cur_sequence_heatmap'],
        [[0, 'Off', payload], [1, 'On', '[]'], [2, 'On', payload], [3, 'Off', payload],
         [4, 'Off', ''], [5, 'On', '[]'], [6, 'Off', payload]])
    result = _features(path, 'spray', '3')
    assert result['sequence_count'] == 2 and result['sequence_heatmap_count'] == 4
    assert result['sequence_heatmap_invalid_count'] == 0
    assert result['sequence_heatmaps'] == [
        {'frame_index': 0, 'time_seconds': 0, 'spray_run_index': None, 'maps': [image]},
        {'frame_index': 2, 'time_seconds': 2, 'spray_run_index': 1, 'maps': [image]},
        {'frame_index': 3, 'time_seconds': 3, 'spray_run_index': 1, 'maps': [image]},
        {'frame_index': 6, 'time_seconds': 6, 'spray_run_index': 2, 'maps': [image]},
    ]
    assert all('sequence_id' not in entry for entry in result['sequence_heatmaps'])


@pytest.mark.parametrize('bad_value', ['json', 'object', 'element', 'base64', 'png', 'dimensions', 'nonfinite', 'overflow'])
def test_malformed_heatmaps_are_counted_without_discarding_frame_metrics(tmp_path, bad_value):
    if bad_value in {'json', 'object', 'element'}:
        cell = {'json': '{broken', 'object': '{}', 'element': '[1]'}[bad_value]
    else:
        image = png_payload()
        if bad_value == 'base64':
            image['data'] = '!bad!'
        elif bad_value == 'png':
            image['data'] = base64.b64encode(b'a valid base64 string but no PNG').decode('ascii')
        elif bad_value == 'dimensions':
            image['img_width'] = 490
        elif bad_value == 'nonfinite':
            image['x0'] = float('nan')
        else:
            image['x0'] = 10 ** 400
        cell = json.dumps([image])
    path = write_record(tmp_path, ['time', '분사', 'Distance (Input)', 'cur_sequence_heatmap'],
                        [[0, 'On', 250, cell], [1, 'On', 350, '[]']])
    result = _features(path, 'spray', '3')
    assert result['frame_count'] == 2 and result['global_metrics']['distance_mm']['median'] == 300
    assert result['sequence_heatmap_count'] == 0 and result['sequence_heatmaps'] == []
    assert result['sequence_heatmap_invalid_count'] == 1


def test_wide_csv_accepts_sparse_png_larger_than_default_csv_field_limit(tmp_path):
    image = png_payload(large=True)
    columns = ['time', '분사'] + [f'unused_{i}' for i in range(169)] + ['cur_sequence_heatmap', 'Zone ID']
    path = write_record(tmp_path, columns, [[0, 'On'] + ['unused'] * 169 + [json.dumps([image]), '1']])
    assert len(json.dumps([image])) > 131072 and len(columns) == 173
    result = _features(path, 'spray', '3')
    assert result['frame_count'] == 1 and result['zone_id_available'] is True
    assert result['sequence_heatmap_count'] == 1
    assert result['sequence_heatmaps'][0]['maps'][0] == image


def test_oversized_valid_png_is_rejected_without_losing_metrics(tmp_path):
    image = png_payload(large=17 * 1024 * 1024)
    path = write_record(tmp_path, ['time', '분사', 'Distance (Input)', 'cur_sequence_heatmap'],
                        [[0, 'On', 250, json.dumps([image])]])
    result = _features(path, 'spray', '3')
    assert result['global_metrics']['distance_mm']['median'] == 250
    assert result['sequence_heatmap_count'] == 0 and result['sequence_heatmap_invalid_count'] == 1


def test_header_only_csv_has_clear_no_frames_error(tmp_path):
    path = write_record(tmp_path, ['time', '분사'], [])
    with pytest.raises(ValueError, match='프레임'):
        _features(path, 'spray', '3')
