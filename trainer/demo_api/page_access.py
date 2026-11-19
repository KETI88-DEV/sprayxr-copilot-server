"""Private, bounded HTML page visits; headers describe navigation, never identity.

The file is separate from the public DB viewer. No URL queries, cookies, user
agents, raw referrers, forwarded addresses or trainee identifiers are persisted.
"""
from __future__ import annotations

import logging
import json
import os
import sqlite3
import stat
import threading
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from ipaddress import ip_address, ip_network
from urllib.parse import urlsplit

from anyio import to_thread
from pydantic import BaseModel, ConfigDict, Field, field_validator

from trainer.demo_api.request_provenance import valid_ip

log = logging.getLogger(__name__)
PAGES = {'/': 'home', '/home': 'home', '/docs': 'docs', '/redoc': 'docs',
         '/admin': 'admin', '/admin/manage': 'admin', '/admin/sessions': 'admin',
         '/admin/sessions/view': 'admin', '/admin/feedback': 'admin', '/admin/training-data': 'admin',
         '/admin/query-feedback': 'admin', '/feedback': 'feedback',
         '/motion-viewer': 'motion-viewer', '/motion-viewer/v1': 'motion-viewer',
         '/motion-viewer/v2': 'motion-viewer', '/query-chat': 'query-chat'}
SOURCES = {'direct', 'internal', 'external'}
SCHEMA = """CREATE TABLE IF NOT EXISTS page_access (
    id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL,
    client_ip TEXT, page TEXT NOT NULL, path TEXT NOT NULL, source TEXT NOT NULL,
    source_basis TEXT NOT NULL, identity TEXT NOT NULL DEFAULT 'unconfirmed');
    CREATE INDEX IF NOT EXISTS page_access_time ON page_access(created_at);
    CREATE INDEX IF NOT EXISTS page_access_page ON page_access(page, id);
    CREATE INDEX IF NOT EXISTS page_access_ip ON page_access(client_ip, id);
"""


class _IPLabelRule(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    network: str = Field(min_length=1, max_length=64)
    label: str = Field(min_length=1, max_length=80)
    basis: str = Field(min_length=1, max_length=240)

    @field_validator('label', 'basis')
    @classmethod
    def public_text(cls, value: str) -> str:
        if not value.strip() or not value.isprintable():
            raise ValueError('plain public label text required')
        return value.strip()


class _IPLabelConfig(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    version: int = Field(ge=1, le=1)
    rules: list[_IPLabelRule] = Field(max_length=128)


def _unique_json_object(pairs):
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError('duplicate rule fields')
    return result


def _load_network_rules() -> list:
    """Read current operator rules once per query; never expose file details."""
    configured = os.environ.get('PAGE_ACCESS_IP_LABELS_PATH')
    if not configured:
        return []
    try:
        path = Path(configured)
        metadata = path.stat()
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > 65_536:
            return []
        with path.open('rb') as source:
            raw = source.read(65_537)
        if len(raw) > 65_536:
            return []
        config = _IPLabelConfig.model_validate(json.loads(raw, object_pairs_hook=_unique_json_object))
        rules, seen = [], set()
        for item in config.rules:
            network = ip_network(item.network, strict=False)
            if network in seen:
                return []  # Ambiguous duplicate networks fail closed as one config.
            seen.add(network)
            rules.append((network, item.label, item.basis))
        return sorted(rules, key=lambda item: item[0].prefixlen, reverse=True)
    except (OSError, ValueError, RecursionError):
        return []


def _network_hint(client_ip: str | None, rules: list) -> dict:
    try:
        address = ip_address(client_ip) if client_ip else None
    except ValueError:
        address = None
    if address is not None:
        for network, label, basis in rules:
            if address.version == network.version and address in network:
                return {'network_label': label + ' (추정)', 'network_basis': '현재 규칙·조회 시 분류: ' + basis}
    return {'network_label': '미확인', 'network_basis': '현재 IP망 규칙의 일치 근거 없음'}


def utc_stamp(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat(timespec='microseconds')


class PageAccessStore:
    def __init__(self, path: Path, *, max_rows: int = 100_000, retention_days: int = 30):
        if max_rows < 1 or retention_days < 1:
            raise ValueError('positive page access limits required')
        self.path = Path(path).resolve()
        self.max_rows, self.retention_days = max_rows, retention_days
        self._guard = threading.Lock()
        self._ready = False

    @classmethod
    def for_database(cls, database: Path) -> PageAccessStore:
        database = Path(database).resolve()
        path = Path(os.environ.get('PAGE_ACCESS_DB_PATH', str(database) + '.page-access.sqlite3')).resolve()
        if path == database:
            raise ValueError('page access storage must be separate from the public database')
        return cls(path)

    def append(self, *, created_at: str, client_ip: str | None, path: str, source: str, source_basis: str):
        if path not in PAGES or source not in SOURCES or source_basis not in {'unavailable', 'request_header_estimate'}:
            raise ValueError('invalid page access event')
        stamp = utc_stamp(datetime.fromisoformat(created_at))
        cutoff = utc_stamp(datetime.now(timezone.utc) - timedelta(days=self.retention_days))
        with self._guard:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with closing(sqlite3.connect(self.path, timeout=0.5)) as conn:
                if not self._ready:
                    conn.execute('PRAGMA journal_mode=WAL')
                    conn.executescript(SCHEMA)
                with conn:
                    cursor = conn.execute('INSERT INTO page_access(created_at,client_ip,page,path,source,source_basis) VALUES(?,?,?,?,?,?)',
                                          (stamp, valid_ip(client_ip), PAGES[path], path, source, source_basis))
                    # Indexed deletes keep storage bounded and reuse SQLite free pages.
                    conn.execute('DELETE FROM page_access WHERE id <= ?', (cursor.lastrowid - self.max_rows,))
                    conn.execute('DELETE FROM page_access WHERE created_at < ?', (cutoff,))
                self._ready = True

    def records(self, *, page: int = 1, limit: int = 50, page_filter: str = 'all', client_ip: str = '',
                source: str = 'all', since: datetime | None = None, until: datetime | None = None) -> dict:
        limit, page = max(1, min(limit, 100)), max(1, page)
        result = {'rows': [], 'total': 0, 'page': 1, 'limit': limit,
                  'retention_days': self.retention_days, 'max_rows': self.max_rows,
                  'network_classification_basis': 'current_rules_at_read'}
        if not self.path.is_file():
            return result
        where = ['created_at >= ?']
        params = [utc_stamp(datetime.now(timezone.utc) - timedelta(days=self.retention_days))]
        for column, value in [('page', page_filter), ('source', source)]:
            if value != 'all':
                where.append(column + ' = ?')
                params.append(value)
        if client_ip:
            where.append('client_ip = ?')
            params.append(valid_ip(client_ip) or '')
        for operator, bound in [('>=', since), ('<=', until)]:
            if bound is not None:
                where.append('created_at ' + operator + ' ?')
                params.append(utc_stamp(bound))
        with closing(sqlite3.connect(f'file:{self.path}?mode=ro', uri=True, timeout=0.5)) as conn:
            conn.row_factory = sqlite3.Row
            # One read snapshot keeps the count and requested page consistent.
            conn.execute('BEGIN')
            if not conn.execute("SELECT 1 FROM sqlite_master WHERE name='page_access' AND type='table'").fetchone():
                return result
            predicate = ' AND '.join(where)
            total = conn.execute('SELECT COUNT(*) FROM page_access WHERE ' + predicate, params).fetchone()[0]
            page = min(page, max(1, (total + limit - 1) // limit))
            rows = conn.execute('SELECT created_at,client_ip,page,path,source,source_basis,identity FROM page_access WHERE '
                                + predicate + ' ORDER BY id DESC LIMIT ? OFFSET ?', [*params, limit, (page - 1) * limit])
            rules = _load_network_rules()
            return {**result, 'rows': [{**dict(row), **_network_hint(row['client_ip'], rules)} for row in rows],
                    'total': total, 'page': page}


def navigation_source(scope: dict) -> tuple[str, str]:
    headers = {key.lower(): value.decode('latin-1')[:2048] for key, value in scope.get('headers', [])}
    try:
        referrer = urlsplit(headers.get(b'referer', ''))
        if referrer.scheme in {'http', 'https'} and referrer.hostname:
            # Host and referrer are only navigation hints, explicitly unverified.
            current = urlsplit(scope.get('scheme', 'http') + '://' + headers.get(b'host', ''))
            same = (referrer.scheme, referrer.hostname, referrer.port or (443 if referrer.scheme == 'https' else 80)) == (
                current.scheme, current.hostname, current.port or (443 if current.scheme == 'https' else 80))
            return ('internal' if same else 'external'), 'request_header_estimate'
    except ValueError:
        pass
    return 'direct', 'unavailable'


class PageAccessMiddleware:
    def __init__(self, app, store: PageAccessStore):
        self.app, self.store = app, store

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http' or scope.get('method') != 'GET' or scope.get('path') not in PAGES:
            return await self.app(scope, receive, send)
        headers = dict(scope.get('headers', []))
        if b'prefetch' in headers.get(b'sec-purpose', b'').lower() or b'prefetch' in headers.get(b'purpose', b'').lower():
            return await self.app(scope, receive, send)
        timestamp = utc_stamp(datetime.now(timezone.utc))
        successful_html = False

        async def observe(message):
            nonlocal successful_html
            if message['type'] == 'http.response.start':
                content_type = dict(message.get('headers', [])).get(b'content-type', b'').lower()
                successful_html = 200 <= message['status'] < 300 and content_type.split(b';', 1)[0].strip() == b'text/html'
            await send(message)

        await self.app(scope, receive, observe)
        if successful_html:
            peer = scope.get('client')
            client_ip = valid_ip(peer[0]) if peer else None
            source, basis = navigation_source(scope)
            try:
                await to_thread.run_sync(lambda: self.store.append(created_at=timestamp, client_ip=client_ip,
                    path=scope['path'], source=source, source_basis=basis))
            except (OSError, sqlite3.Error):
                # Operational logging failure must not change a served page response.
                log.warning('Page access recording unavailable')
