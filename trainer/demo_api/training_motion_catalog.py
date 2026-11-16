"""Read completed, Zone-mapped training artifacts as viewer recording profiles.

This is a display adapter. It never writes expert guidance profiles or trains a
model. Catalog listing reads index metadata only; recording features are loaded
when the corresponding worker/mode is opened.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
import statistics
import threading
from collections import Counter
from pathlib import Path

from trainer.demo_api.training_data_catalog import TrainingDataCatalog
from trainer.demo_api.training_worker_identity import selection_worker_id


class TrainingMotionCatalog:
    def __init__(self, source_root: Path, catalog: TrainingDataCatalog | None = None):
        self.source_root = Path(source_root).resolve()
        self.catalog = catalog or TrainingDataCatalog(source_root=self.source_root)
        self._fingerprint = None
        self._records = []
        self._lock = threading.Lock()

    @staticmethod
    def worker_id(value) -> str:
        value = str(value or '')
        return value if re.fullmatch(r'[A-Za-z0-9_-]+', value) else 'dataset_' + hashlib.sha256(value.encode()).hexdigest()[:24]

    @staticmethod
    def session_id(row: dict) -> str:
        value = row.get('session_id') or row.get('dataset_name') or row.get('dataset_id')
        # Imported converted filenames contain this suffix; their logical session
        # ID is separately recorded by current builders and always takes priority.
        if not row.get('session_id') and row.get('source_kind') == 'converted':
            value = str(value).removesuffix('_Height')
        return str(value)

    def records(self) -> list[dict]:
        path = self.catalog.root / 'catalog.json'
        try:
            stat = path.stat()
            fingerprint = (stat.st_mtime_ns, stat.st_size)
        except OSError:
            return []
        with self._lock:
            if fingerprint != self._fingerprint:
                index = self.catalog._index()
                self._records = [row for row in self.catalog._rows(index)
                    if row.get('quality_status') == 'complete' and row.get('zone_id_available') is True
                    and row.get('process') in {'spray', 'blasting', 'grinding'}
                    and str(row.get('scenario_id', '')).isdigit() and 1 <= int(row['scenario_id']) <= 10]
                self._fingerprint = fingerprint
            return list(self._records)

    def workers(self) -> list[dict]:
        workers = {}
        for row in self.records():
            key = self.worker_id(selection_worker_id(row.get('worker_id'), row.get('worker_name')))
            workers.setdefault(key, {'worker_id': key, 'display_name': str(row.get('worker_name') or row.get('worker_id') or '미기록'),
                                     'role': 'expert' if row.get('role') == 'expert' else 'trainee',
                                     'root': self.source_root, 'catalog_worker_id': row.get('worker_id')})
        return list(workers.values())

    def selections(self, worker_id: str, process: str, mode_id: int) -> list[dict]:
        records = [row for row in self.records()
                   if self.worker_id(selection_worker_id(row.get('worker_id'), row.get('worker_name'))) == worker_id
                   and row['process'] == process and int(row['scenario_id']) == mode_id]
        # A catalog is already deduplicated, but prefer the converted artifact
        # if a legacy/current publication contains the same logical session.
        unique = {}
        for row in sorted(records, key=lambda r: r.get('source_kind') == 'converted'):
            unique[self.session_id(row)] = row
        return list(unique.values())

    def session_ids(self, worker_id: str, process: str, mode_id: int) -> list[str]:
        return sorted(self.session_id(row) for row in self.selections(worker_id, process, mode_id))

    def _session(self, row: dict) -> dict | None:
        try:
            artifact = self.catalog.detail(row['dataset_id'])
        except (OSError, ValueError, KeyError):
            return None
        meta, features = artifact.get('metadata') or {}, artifact.get('features') or {}
        source = (artifact.get('provenance') or {}).get('source_csv') or {}
        relative = source.get('relative_path')
        if not isinstance(relative, str):
            return None
        candidate = (self.source_root / relative).resolve()
        if not candidate.is_relative_to(self.source_root) or not candidate.is_file():
            return None
        zones = {}
        for zone in features.get('zones') or []:
            if zone.get('map_num') is None or zone.get('zone_id') is None:
                continue
            key = f"{zone['map_num']}:{zone['zone_id']}"
            # Expert profiles (build_expert_reference) use the measured Real columns; the
            # feature `speed_mms` is the simulator Input setting (mostly a fixed 1600).
            zones[key] = {'map_num': zone['map_num'], 'zone_id': str(zone['zone_id']),
                'speed_mms': (zone.get('speed_real_mms') or {}).get('median'),
                'distance_mm': (zone.get('distance_real_mm') or {}).get('median')}
        # Features produced for old foreman files were canonicalized at build
        # time, while playback still reads their original CSV Zone numbers.
        legacy = row.get('source_kind') != 'converted' and str(relative).startswith('raw/')
        return {'session_id': self.session_id(row), 'dataset_id': row['dataset_id'],
                'source_worker_id': row.get('worker_id'),
                'zone_id_policy': 'legacy_expert' if legacy else 'source',
                'sources': {'catalog_relative_path': relative},
                'heightmaps': (artifact.get('provenance') or {}).get('heightmaps') or [],
                'target_thickness_um': meta.get('target_thickness_um'),
                'analysis': {'macro_order': features.get('zone_order') or [], 'zones': zones}}

    def profile(self, worker_id: str, process: str, mode_id: int, base: dict | None = None) -> dict | None:
        rows = self.selections(worker_id, process, mode_id)
        if not rows:
            return copy.deepcopy(base) if base is not None else None
        sessions = {str(s['session_id']): copy.deepcopy(s) for s in (base or {}).get('sessions', [])}
        for row in rows:
            session = self._session(row)
            if session is not None:
                sessions[session['session_id']] = session
        if not sessions:
            return None
        profile = copy.deepcopy(base) if base is not None else {
            'schema_version': 1, 'identity': {'process': process, 'mode_id': mode_id,
                'content_type': 'learning', 'zone_id_policy': 'source'},
            'quality': {'status': 'registered_training_data'}, 'aggregate': {'zones': {}, 'macro_orders': []}}
        profile['sessions'] = list(sessions.values())
        if base is None:
            orders = Counter(json.dumps(s['analysis']['macro_order'], sort_keys=True) for s in sessions.values())
            profile['aggregate']['macro_orders'] = [{'order': json.loads(order), 'session_count': count,
                'fraction': count / len(sessions)} for order, count in orders.most_common()]
            values = {}
            for session in sessions.values():
                for key, zone in session['analysis']['zones'].items():
                    bucket = values.setdefault(key, {'map_num': zone['map_num'], 'zone_id': zone['zone_id'],
                                                     'speed_mms': [], 'distance_mm': []})
                    for name in ['speed_mms', 'distance_mm']:
                        if isinstance(zone.get(name), (int, float)):
                            bucket[name].append(zone[name])
            profile['aggregate']['zones'] = {key: {'map_num': z['map_num'], 'zone_id': z['zone_id'],
                **{name: {'median': statistics.median(z[name]) if z[name] else None} for name in ['speed_mms','distance_mm']},
                'expected_direction': 'unavailable'} for key, z in values.items()}
        return profile
