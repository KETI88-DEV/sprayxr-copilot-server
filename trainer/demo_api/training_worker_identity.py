"""Stable selector aliases for the two explicitly identified legacy workers.

Catalog source IDs stay unchanged. Other equal display names are not merged.
"""
_KNOWN_WORKERS = {'전형기': 'jeon_hyeonggi', '김경호': 'kim_kyungho'}


def selection_worker_id(worker_id: object, worker_name: object = '') -> str:
    return _KNOWN_WORKERS.get(str(worker_name or '').strip(), str(worker_id or ''))
