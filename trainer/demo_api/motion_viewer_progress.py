"""Bounded, private progress for actual viewer work; no timing estimates."""
from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from threading import Event, RLock
from time import monotonic


STAGES = {
    'profile': '기록 목록과 원본 확인', 'csv_count': 'CSV 전체 행 수 확인',
    'csv_frames': 'CSV 동작 프레임 처리', 'csv_trace': 'CSV 도포 경로 처리',
    'csv_cells': 'CSV 도포 점군 통계', 'csv_strokes': 'CSV Zone 경로 통계',
    'csv_surfaces': 'CSV Zone 표면 통계', 'record_prepare': '비교 기록 확인',
    'expert_prepare': '전문가 기록 확인', 'reference_statistics': '비교 기록 통계',
    'zone_evaluation': 'Zone 유사도 비교', 'coating': '도막 분석',
    'response': '응답 자료 조립', 'completed': '서버 계산 완료',
    'failed': '불러오기 실패', 'cancelled': '불러오기 취소',
}
_CURRENT = ContextVar('motion_viewer_load_progress', default=None)


class LoadStopped(Exception):
    """The client left, or its bounded progress job expired/was evicted."""


class LoadJob:
    def __init__(self, store, load_id):
        self.store, self.load_id = store, load_id
        self.stopped = Event()
        self.row_totals = {}  # Private paths never enter the public snapshot.
        self.touched = store.now()
        self.state = {'status': 'running', 'stage': 'profile', 'stage_label': STAGES['profile'],
                      'completed': 0, 'total': None, 'unit': None, 'phase_complete': False}

    def checkpoint(self):
        if self.stopped.is_set():
            raise LoadStopped()
        with self.store.lock:
            if (self.store.entries.get(self.load_id) is not self
                    or self.store.now() - self.touched >= self.store.ttl):
                self.stopped.set()
                raise LoadStopped()

    def update(self, stage, *, completed=0, total=None, unit=None, phase_complete=False):
        if stage not in STAGES or unit not in {None, 'rows', 'records', 'zones'}:
            raise ValueError('Unsupported progress stage or unit')
        self.checkpoint()
        with self.store.lock:
            # A changed input can outgrow an earlier count; do not cap or fake it.
            if total is not None and completed > total:
                total = None
            self.state.update(stage=stage, stage_label=STAGES[stage], completed=completed,
                              total=total, unit=unit, phase_complete=phase_complete)
            self.touched = self.store.now()

    def finish(self, status):
        with self.store.lock:
            if self.store.entries.get(self.load_id) is not self:
                return
            self.state.update(status=status, stage=status, stage_label=STAGES[status],
                              completed=0, total=None, unit=None, phase_complete=True)
            self.row_totals.clear()
            self.touched = self.store.now()

    def cancel(self):
        self.stopped.set()
        self.finish('cancelled')


class LoadProgressStore:
    def __init__(self, *, capacity=64, ttl_seconds=300, now=monotonic):
        self.capacity, self.ttl, self.now = capacity, ttl_seconds, now
        self.lock, self.entries = RLock(), OrderedDict()

    def _expire(self):
        for key, job in list(self.entries.items()):
            if self.now() - job.touched >= self.ttl:
                job.stopped.set()
                del self.entries[key]

    def start(self, load_id):
        with self.lock:
            self._expire()
            if load_id in self.entries:
                raise ValueError('Load ID already used')
            job = LoadJob(self, load_id)
            self.entries[load_id] = job
            while len(self.entries) > self.capacity:
                _, removed = self.entries.popitem(last=False)
                removed.stopped.set()
            return job

    def snapshot(self, load_id):
        with self.lock:
            self._expire()
            job = self.entries.get(load_id)
            return dict(job.state) if job is not None else {'status': 'unavailable', 'reason_code': 'expired_or_missing'}


@contextmanager
def bind_progress(job):
    token = _CURRENT.set(job)
    try:
        yield job
    finally:
        _CURRENT.reset(token)


def run_load(job, callback):
    with bind_progress(job):
        try:
            job.checkpoint()
            result = callback()
            job.checkpoint()
        except LoadStopped:
            job.cancel()
            raise
        except BaseException:
            job.finish('cancelled' if job.stopped.is_set() else 'failed')
            raise
        job.finish('completed')
        return result


def stage(name):
    job = _CURRENT.get()
    if job is not None:
        job.update(name)


def progress_rows(rows, *, stage, path):
    job = _CURRENT.get()
    if job is None:
        yield from rows
        return
    total = job.row_totals.get(str(path))
    job.update(stage, total=total, unit='rows')
    completed = 0
    for row in rows:
        if job.stopped.is_set():
            raise LoadStopped()
        yield row
        completed += 1
        if completed <= 2 or completed % 512 == 0:
            job.update(stage, completed=completed, total=total, unit='rows')
    job.row_totals[str(path)] = completed
    job.update(stage, completed=completed, total=completed, unit='rows', phase_complete=True)


def progress_units(items, *, stage, unit='records'):
    job = _CURRENT.get()
    if job is None:
        yield from items
        return
    total = len(items)
    job.update(stage, total=total, unit=unit)
    for index, item in enumerate(items):
        job.checkpoint()
        yield item
        job.update(stage, completed=index + 1, total=total, unit=unit)
    job.update(stage, completed=total, total=total, unit=unit, phase_complete=True)


def record_row_total(path, total):
    job = _CURRENT.get()
    if job is not None:
        job.row_totals[str(path)] = total
