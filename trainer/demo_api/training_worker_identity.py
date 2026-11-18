"""Presentation groups without changing source worker identities.

Only explicitly unidentified names share a bucket. Equal known display names
remain separate, apart from the two pre-existing legacy selector aliases.
"""
import unicodedata


_KNOWN_WORKERS = {'전형기': 'jeon_hyeonggi', '김경호': 'kim_kyungho'}
UNKNOWN_WORKER_ID = '__unknown_workers__'
UNKNOWN_WORKER_LABEL = '이름 미확인'


def is_unknown_worker(worker_id: object, worker_name: object = '') -> bool:
    name = unicodedata.normalize('NFKC', str(worker_name or '')).strip().casefold()
    prefix, delimiter, _ = name.partition('·')
    placeholder = bool(delimiter) and prefix.strip() == '이름 미확인'
    return (not name or name in {'미기록', '미확인', '이름 미확인', 'unknown'}
            or placeholder)


def worker_group_id(worker_id: object, worker_name: object = '') -> str:
    return UNKNOWN_WORKER_ID if is_unknown_worker(worker_id, worker_name) else str(worker_id or '')


def worker_group_label(worker_id: object, worker_name: object = '') -> str:
    return UNKNOWN_WORKER_LABEL if is_unknown_worker(worker_id, worker_name) else str(worker_name).strip()


def worker_sort_key(worker_id: object, worker_name: object = '') -> tuple[bool, str, str]:
    """Stable Korean name order, independent of the host's installed locales."""
    return (is_unknown_worker(worker_id, worker_name),
            unicodedata.normalize('NFKC', worker_group_label(worker_id, worker_name)).casefold(),
            str(worker_id or ''))


def selection_worker_id(worker_id: object, worker_name: object = '') -> str:
    return _KNOWN_WORKERS.get(str(worker_name or '').strip(), str(worker_id or ''))
