"""Cascading recording selectors and authenticated, stored dataset facts for Query."""
from __future__ import annotations

import json
import math
import threading
from collections import Counter
from typing import Any, Mapping

from fastapi import HTTPException

from trainer.demo_api.training_data_catalog import TrainingDataCatalog
from trainer.demo_api.training_worker_identity import selection_worker_id
from trainer.demo_api.viewer_context import sign_context

PROCESSES = {'spray': '스프레이', 'blasting': '블라스팅', 'grinding': '그라인딩'}


def number(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


def representative(value: Any) -> float | None:
    if isinstance(value, Mapping):
        return number(value.get('median')) if number(value.get('median')) is not None else number(value.get('mean'))
    return number(value)


def zone_identity(row: Mapping) -> tuple[str, str] | None:
    zone = str(row.get('zone_id') or '').strip()
    if not zone.isdigit() or int(zone) <= 0:
        return None
    face = row.get('map_num')
    if isinstance(face, (float, int)) and not isinstance(face, bool) and math.isfinite(face) and face >= 0 and float(face).is_integer():
        face = str(int(face))
    elif face is None:
        face = str(row.get('map_label') or 'unknown')
    else:
        face = str(face)
    return face, str(int(zone))


class QueryChatDatasets:
    """Refresh selectors when the catalog is republished; parse only one selected artifact."""
    def __init__(self, catalog: TrainingDataCatalog | None = None):
        self.catalog = catalog or TrainingDataCatalog()
        self._lock = threading.Lock()
        self._fingerprint: tuple | None = None
        self._cached: list[dict] = []

    def rows(self) -> list[dict]:
        try:
            stat = (self.catalog.root / 'catalog.json').stat()
            fingerprint = (stat.st_mtime_ns, stat.st_size, stat.st_ino)
        except OSError:
            return []
        with self._lock:
            if fingerprint != self._fingerprint:
                self._cached = [{**row, 'scenario_id': str(int(row['scenario_id']))} for row in self.catalog._rows(self.catalog._index())
                                if row.get('process') in PROCESSES
                                and str(row.get('scenario_id', '')).isdigit()
                                and 1 <= int(row['scenario_id']) <= 10]
                self._fingerprint = fingerprint
            return self._cached

    def options(self, worker_id='all', process='all', scenario_id='all') -> dict:
        rows = self.rows()
        workers = {selection_worker_id(row.get('worker_id'), row.get('worker_name')):
                   str(row.get('worker_name') or row['worker_id']) for row in rows}
        chosen = [row for row in rows if selection_worker_id(row.get('worker_id'), row.get('worker_name')) == worker_id]
        processes = [{'id': key, 'label': label} for key, label in PROCESSES.items()
                     if any(row['process'] == key for row in chosen)]
        chosen = [row for row in chosen if row.get('process') == process]
        modes = sorted({str(row['scenario_id']) for row in chosen}, key=int)
        chosen = [row for row in chosen if str(row.get('scenario_id')) == scenario_id]
        chosen.sort(key=lambda row: (str(row.get('acquired_at') or ''), str(row.get('dataset_name') or ''), row['dataset_id']), reverse=True)
        return {'workers': [{'id': key, 'label': label} for key, label in sorted(workers.items(), key=lambda item: (item[1], item[0]))],
                'processes': processes, 'modes': [{'id': mode, 'label': f'훈련모드 {mode}'} for mode in modes],
                'sessions': [{'id': row['dataset_id'], 'label': ' · '.join(filter(None, [row.get('acquired_at'), row.get('dataset_name')])),
                              'session_id': row.get('dataset_name'), 'quality_status': row.get('quality_status'),
                              'zone_id_available': row.get('zone_id_available', False)} for row in chosen],
                'selected': {'worker_id': worker_id, 'process': process, 'scenario_id': scenario_id},
                'record_count': len(rows)}

    def _reference(self, process: str, scenario: str) -> dict:
        try:
            path = self.catalog.sources / 'generated' / 'profiles' / process / 'learning' / f'mode_{int(scenario):02d}.json'
            profile = json.loads(path.read_text(encoding='utf-8'))
            return profile if isinstance(profile, dict) else {}
        except (OSError, ValueError, TypeError):
            return {}

    def context(self, dataset_id: str, zone_key: str = 'all') -> dict:
        item = next((row for row in self.rows() if row['dataset_id'] == dataset_id), None)
        if item is None:
            raise HTTPException(404, '선택한 학습 데이터를 찾을 수 없습니다. 목록을 새로고침하세요.')
        try:
            artifact = self.catalog.detail(dataset_id)
        except (OSError, ValueError, KeyError):
            raise HTTPException(409, '학습 데이터 원본 검증에 실패했습니다. 목록을 새로고침하세요.') from None
        meta, features = artifact.get('metadata') or {}, artifact.get('features') or {}
        profile = self._reference(item['process'], str(item['scenario_id']))
        reference = (profile.get('aggregate') or {}).get('zones') or {}
        zones: dict[str, dict] = {}
        for row in features.get('zones') or []:
            if not isinstance(row, Mapping) or (identity := zone_identity(row)) is None:
                continue
            face, zone = identity
            key = f'{face}:{zone}'
            expert = reference.get(key) or {}
            zones[key] = {'map_num': face, 'zone_id': zone, 'frame_count': row.get('frame_count'),
                          # Measured Real columns, as in the expert profiles; feature `speed_mms` is the simulator Input (mostly 1600).
                          'distance_mm': representative(row.get('distance_real_mm')), 'speed_mms': representative(row.get('speed_real_mms')),
                          'reference_distance_mm': representative(expert.get('distance_mm')),
                          'reference_speed_mms': representative(expert.get('speed_mms')),
                          'direction': None, 'posture_label': None, 'metrics': [], 'corrections': []}
        coating_rows = []
        coating_measurements = []
        for face in (artifact.get('coating') or {}).get('maps') or []:
            for row in face.get('zones') or []:
                if not isinstance(row, Mapping) or (identity := zone_identity({**row, 'map_num': face.get('map_num')})) is None:
                    continue
                map_num, zone = identity
                key = f'{map_num}:{zone}'
                target = zones.setdefault(key, {'map_num': map_num, 'zone_id': zone, 'metrics': [], 'corrections': []})
                summary = row.get('summary') or {}
                actual, expert = number(summary.get('median_um')), number((row.get('reference') or {}).get('median_um'))
                if actual is not None:
                    target['thickness_um'] = actual
                    coating_measurements.append({'map_num': map_num, 'zone_id': zone, 'median_um': actual,
                                                 'mean_um': number(summary.get('mean_um')),
                                                 'coverage_percent': number(summary.get('coverage_percent'))})
                if actual is not None and expert is not None and expert > 0:
                    coating_rows.append({'map_num': map_num, 'zone_id': zone, 'actual_um': actual, 'reference_um': expert})
        zone_options = [{'id': key, 'label': (f'Zone {row["zone_id"]} · Map {row["map_num"]}' if row['map_num'] != 'unknown'
                                            else f'Zone {row["zone_id"]} · 면 미기록')}
                        for key, row in sorted(zones.items(), key=lambda entry: (entry[1]['map_num'], int(entry[1]['zone_id'])))]
        if zone_key != 'all' and zone_key not in zones:
            raise HTTPException(422, '선택한 Zone은 이 훈련 기록에 없습니다.')
        # The legacy Query fact reader keys measurements by Zone number alone.
        # A map selector may disambiguate one number; otherwise do not let the
        # last JSON row silently replace a different face's measurements.
        repeated = {zone for zone, count in Counter(row['zone_id'] for row in zones.values()).items() if count > 1}
        selected_zone = zones.get(zone_key)
        ambiguous = repeated - ({selected_zone['zone_id']} if selected_zone else set())
        signed_zones = {}
        numeric_keys = {'distance_mm', 'speed_mms', 'reference_distance_mm', 'reference_speed_mms', 'thickness_um'}
        for key, row in zones.items():
            if selected_zone and row['zone_id'] == selected_zone['zone_id'] and key != zone_key:
                continue
            signed_zones[key] = ({k: v for k, v in row.items() if k not in numeric_keys}
                                 if row['zone_id'] in ambiguous else row)
        actual_order = []
        actual_map_order = []
        for row in features.get('zone_order') or []:
            if not isinstance(row, Mapping) or (identity := zone_identity(row)) is None:
                continue
            face, zone = identity
            actual_order.append(zone)
            actual_map_order.append({'map_num': face, 'zone_id': zone})
        facts = {'record_id': item.get('dataset_name'), 'scenario_id': str(item['scenario_id']),
                 'worker_name': item.get('worker_name'), 'process': item['process'],
                 'worker_id': selection_worker_id(item.get('worker_id'), item.get('worker_name')),
                 'source_worker_id': item.get('worker_id'),
                 'actual_order': actual_order, 'actual_map_order': actual_map_order,
                 'reference_order': None, 'record_direction': None, 'zones': signed_zones,
                 'ambiguous_zone_ids': sorted(ambiguous, key=int),
                 'zone_resolution_note': ('동일 Zone 번호가 여러 면에 있습니다. 면·Zone을 선택한 뒤 비교하세요.' if ambiguous else None),
                 'quality_status': item.get('quality_status'), 'dataset_id': dataset_id,
                 'duration_seconds': number(features.get('duration_seconds')), 'sequence_count': features.get('sequence_count'),
                 'frame_count': features.get('frame_count'), 'global_metrics': features.get('global_metrics') or {},
                 'coating_measurements': coating_measurements,
                 'measurement_scope': '선택한 훈련 기록·Zone의 취득 데이터 대표 통계입니다. 재생 순간의 값이 아닙니다. 자세 점수와 도포 방향은 이 특징 데이터에 저장되지 않았습니다.'}
        coating = {'analysis_status': 'available' if coating_rows else 'not_available', 'scope': 'final_record',
                   'comparison_basis': 'expert_zone_median', 'zones': coating_rows,
                   'target': {'analysis_status': 'not_available', 'reason_code': 'target_assessment_not_loaded'}}
        playback = {'playhead_seconds': 0, 'source_index': 0, 'map_num': None, 'zone_id': None}
        if zone_key != 'all':
            playback.update(map_num=zones[zone_key]['map_num'], zone_id=zones[zone_key]['zone_id'])
        payload = {'record': item.get('dataset_name'), 'dataset_id': dataset_id,
                   'dialogue_context': facts, 'dialogue_signature': sign_context('dialogue', facts),
                   'coating_context': coating, 'coating_signature': sign_context('coating_status', coating),
                   'playback_context': playback}
        return {'dataset_id': dataset_id, 'scenario_id': str(item['scenario_id']), 'session_id': 'CHATDATA:' + dataset_id,
                'context': 'motion_viewer_playback:' + json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(',', ':')),
                'zones': zone_options,
                'selection': {'worker_name': item.get('worker_name'),
                              'worker_id': selection_worker_id(item.get('worker_id'), item.get('worker_name')),
                              'source_worker_id': item.get('worker_id'),
                              'process': item['process'], 'mode_label': item.get('mode_label'),
                              'record': item.get('dataset_name'), 'quality_status': item.get('quality_status'),
                              'zone_key': zone_key, 'zone_id_available': item.get('zone_id_available', False)},
                'selection_requires_zone': bool(ambiguous),
                'warning': facts['zone_resolution_note'],
                'available': {'zone_statistics': bool(zones), 'coating': bool(coating_measurements), 'pose': False, 'direction': False}}
