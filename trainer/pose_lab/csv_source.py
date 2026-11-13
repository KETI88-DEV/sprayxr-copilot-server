#!/usr/bin/env python3
"""원본 CSV 트리를 패스 단위로 여는 `PassSource`.

여기까지가 실험대가 실데이터를 보는 유일한 통로다. 앞단(`service.compare`)은 패스 한
건의 행 목록만 알면 되고, 그 행이 어느 파일 어디에서 왔는지는 이 모듈만 안다.

세 가지를 이 모듈이 짊어진다.

**세그멘테이션은 새로 만들지 않는다.** 분사 On이 이어지는 구간(버스트)을 잡고 그 안의
착탄점 궤적을 `pass_segmentation.split_pass_spans`로 자른다. 파이프라인의 나머지가
전부 이 규칙 위에 서 있다 — 버스트 하나에 스트로크가 평균 4개 들어가므로 버스트를
패스로 쓰면 이탈 중앙값이 5배로 부푼다(`pass_segmentation` 독스트링).

**실전모드는 범위 밖이다.** 파일명이 `1-A`/`1-B`/`1-C`로 시작하는 세션은 건너뛴다.
학습모드 레벨 1..10만 훑는다.

**게으르게 훑고 캐시한다.** 세션 하나가 수만 행이고 작업자 한 명이 수십 세션이다.
요청마다 전수 스캔하면 화면이 못 쓰게 된다. 그래서 (1) 작업자·세션 목록은 파일명만
읽어 만들고, (2) 패스 색인은 세션 단위로 한 번만 만들어 들고 있고, (3) 원본 행은 작은
LRU로만 들고 있는다(세션 하나가 250 MB 규모라 무제한 캐시는 메모리를 먹는다).

동작 키는 색인에 문자열로 굳히지 않는다. 판정 경계(UV 격자·좌우 경계각·거리 임계)가
요청마다 달라지므로 재료(`service.KeyInputs`)만 들고 있다가 요청 시점에 조립한다.
그래야 화면에서 격자를 바꿔도 CSV를 다시 읽지 않는다.
"""

from __future__ import annotations

import importlib
import math
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Iterator, Sequence

from trainer.pose_lab import service
from trainer.pose_lab.service import PassRef, RawPass

# service가 trainer/scripts를 sys.path에 얹은 뒤라 이름으로 바로 잡힌다.
pass_segmentation = importlib.import_module("pass_segmentation")
simulator_csv = importlib.import_module("simulator_csv")

Vec3 = tuple[float, float, float]

SEQUENCES_SUFFIX: Final = "(Sequences)"
# `<이름>_<YYYYMMDD>` 디렉터리. 전형기만 `전형기/전형기_<YYYYMMDD>`로 한 단계 깊다.
SESSION_DIR: Final = re.compile(r"^(?P<worker>.+)_(?P<date>\d{8})$")
# 학습모드 레벨. `1-A`/`1-B`/`1-C`는 실전모드라 여기에 걸리지 않는다.
MIN_LEVEL: Final = 1
MAX_LEVEL: Final = 10

DEFAULT_ROWS_CACHE: Final = 2
DEFAULT_SESSION_LIMIT: Final = 8

# 패스의 순 변위 하한. `pass_segmentation.MIN_PASS_LENGTH_MM`과 같은 기준이다.
MIN_PASS_LENGTH_M: Final = pass_segmentation.MIN_PASS_LENGTH_MM / 1000.0


@dataclass(frozen=True, slots=True)
class SessionFile:
    """세션 파일 하나. 파일명만 읽어 만든다 — CSV를 열지 않는다."""

    worker: str
    date: str
    level: str
    session: str
    path: Path


@dataclass(frozen=True, slots=True)
class PassIndexEntry:
    """패스 한 건의 색인. 행 자체가 아니라 행 구간과 키 재료만 들고 있다."""

    pass_index: int
    row_start: int
    row_end: int
    frames: int
    length_m: float
    key_inputs: Any


class CsvPassSource:
    """`service.PassSource` 구현. 루트는 작업자 디렉터리들이 놓인 폴더다."""

    def __init__(
        self,
        root: Path,
        rows_cache: int = DEFAULT_ROWS_CACHE,
        samples: int = service.stroke_frame.DEFAULT_SAMPLES,
    ) -> None:
        self.root = Path(root)
        # 색인 시점의 리샘플 N. 키 재료가 이 격자 위에서 나오므로 화면이 다른 N으로
        # 비교하면 키가 미세하게 달라질 수 있다. 응답이 이 값을 함께 낸다.
        self.samples = samples
        self._rows_cache_size = max(1, rows_cache)
        self._sessions: list[SessionFile] | None = None
        self._rows: OrderedDict[Path, list[dict[str, str]]] = OrderedDict()
        self._index: dict[Path, list[PassIndexEntry]] = {}

    # ---- 목록: 파일명만 읽는다 -------------------------------------------------

    def sessions(self) -> list[SessionFile]:
        if self._sessions is None:
            self._sessions = sorted(
                _discover(self.root),
                key=lambda item: (item.worker, item.date, _level_order(item.level), item.session),
            )
        return self._sessions

    def workers(self) -> list[str]:
        return sorted({item.worker for item in self.sessions()})

    def levels(self, worker: str) -> list[str]:
        found = {item.level for item in self.sessions() if item.worker == worker}
        return sorted(found, key=_level_order)

    # ---- 패스: 세션 단위로 한 번만 훑는다 -------------------------------------

    def passes(
        self,
        worker: str,
        level: str | None = None,
        date: str | None = None,
        limit: int = DEFAULT_SESSION_LIMIT,
    ) -> tuple[list[tuple[SessionFile, PassIndexEntry]], int]:
        """(세션, 패스) 쌍과 조건에 맞는 세션 총수.

        총수를 함께 내는 이유는 `limit`이 목록을 자른 사실을 화면이 말할 수 있게
        하기 위함이다. 자른 것을 숨기면 "이 작업자에게 이 패스뿐"으로 읽힌다.
        """
        matched = [
            item
            for item in self.sessions()
            if item.worker == worker
            and (level is None or item.level == level)
            and (date is None or item.date == date)
        ]
        found: list[tuple[SessionFile, PassIndexEntry]] = []
        for item in matched[: max(0, limit)]:
            found.extend((item, entry) for entry in self.index_of(item))
        return found, len(matched)

    def index_of(self, item: SessionFile) -> list[PassIndexEntry]:
        cached = self._index.get(item.path)
        if cached is None:
            cached = self._build_index(item)
            self._index[item.path] = cached
        return cached

    def _build_index(self, item: SessionFile) -> list[PassIndexEntry]:
        rows = self._read(item.path)
        frames = service.measured_pose.measured_frames(rows)
        offsets = _row_offsets(rows)
        entries: list[PassIndexEntry] = []
        for burst in _bursts(frames):
            hits = [frame.hit for frame in burst]
            for start, end in pass_segmentation.split_pass_spans(hits):
                span = burst[start : end + 1]
                entry = _entry_for(span, offsets, len(entries), self.samples)
                if entry is not None:
                    entries.append(entry)
        return entries

    # ---- 적재 ------------------------------------------------------------------

    def load(self, ref: PassRef) -> RawPass:
        item = self._session_for(ref)
        if item is None:
            raise LookupError(f"session not found: {ref}")
        entries = self.index_of(item)
        if not 0 <= ref.pass_index < len(entries):
            raise LookupError(f"pass not found: {ref}")
        entry = entries[ref.pass_index]
        rows = self._read(item.path)
        return RawPass(rows=rows[entry.row_start : entry.row_end + 1])

    def key_of(self, entry: PassIndexEntry, params: Any) -> str:
        return service.key_from(entry.key_inputs, params)

    def _session_for(self, ref: PassRef) -> SessionFile | None:
        for item in self.sessions():
            if (
                item.worker == ref.worker
                and item.date == ref.date
                and item.level == ref.level
                and item.session == ref.session
            ):
                return item
        return None

    def _read(self, path: Path) -> list[dict[str, str]]:
        rows = self._rows.get(path)
        if rows is None:
            rows = simulator_csv.read_rows(path)
            self._rows[path] = rows
            while len(self._rows) > self._rows_cache_size:
                _ = self._rows.popitem(last=False)
        else:
            self._rows.move_to_end(path)
        return rows


def _level_order(level: str) -> tuple[int, str]:
    return (int(level), level) if level.isdigit() else (MAX_LEVEL + 1, level)


def _discover(root: Path) -> Iterator[SessionFile]:
    """루트 밑의 세션 파일. 전형기처럼 한 단계 더 들어간 배치도 받는다."""
    if not root.is_dir():
        return
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        if SESSION_DIR.match(entry.name):
            yield from _files_in(entry)
            continue
        for nested in sorted(entry.iterdir()):
            if nested.is_dir() and SESSION_DIR.match(nested.name):
                yield from _files_in(nested)


def _files_in(date_dir: Path) -> Iterator[SessionFile]:
    matched = SESSION_DIR.match(date_dir.name)
    if matched is None:
        return
    sequences = date_dir / f"{date_dir.name}{SEQUENCES_SUFFIX}"
    if not sequences.is_dir():
        return
    for path in sorted(sequences.glob("*.csv")):
        parts = path.stem.split("_")
        if len(parts) < 2:
            continue
        level = parts[0]
        # 실전모드(`1-A` 등)는 범위 밖이다. 숫자가 아니라 걸러진다.
        if not level.isdigit() or not MIN_LEVEL <= int(level) <= MAX_LEVEL:
            continue
        yield SessionFile(
            worker=matched.group("worker"),
            date=matched.group("date"),
            level=level,
            session="_".join(parts[1:]),
            path=path,
        )


def _row_offsets(rows: Sequence[dict[str, str]]) -> dict[int, int]:
    """프레임 인덱스 → 행 위치. `measured_pose`가 인덱스를 뽑는 규칙과 같아야 한다."""
    offsets: dict[int, int] = {}
    for offset, row in enumerate(rows):
        try:
            key = int(row["인덱스"])
        except (KeyError, ValueError):
            key = offset
        offsets.setdefault(key, offset)
    return offsets


def _bursts(frames: Sequence[Any]) -> Iterator[list[Any]]:
    """분사 중이고 착탄점이 있는 연속 구간. `build_pass_reference.iter_bursts`와 같은 규칙이다."""
    current: list[Any] = []
    for frame in frames:
        if frame.spraying and frame.hit is not None:
            current.append(frame)
            continue
        if current:
            yield current
            current = []
    if current:
        yield current


def _entry_for(
    span: Sequence[Any], offsets: dict[int, int], pass_index: int, samples: int
) -> PassIndexEntry | None:
    """패스 하나를 색인 항목으로. 잴 수 없는 패스는 목록에 올리지 않는다.

    거르는 조건이 둘이다. 순 변위가 `MIN_PASS_LENGTH_MM` 미만인 구간은 설계 §5.4가
    패스로 세지 않는다. 그리고 이 구간의 오른손 이동이 `MIN_ACTIVE_SPREAD_M` 미만이면
    `measured_pose.active_gun`이 잘라낸 행만 보고는 활성 건을 못 고른다 — 목록에는
    있는데 비교하면 "재지 못했다"가 나오는 패스가 된다.
    """
    if len(span) < 2:
        return None
    first, last = span[0], span[-1]
    if first.hit is None or last.hit is None:
        return None
    length = math.dist(first.hit, last.hit)
    if length < MIN_PASS_LENGTH_M:
        return None
    spread = max(
        max(frame.right[axis] for frame in span) - min(frame.right[axis] for frame in span)
        for axis in range(3)
    )
    if spread < service.measured_pose.MIN_ACTIVE_SPREAD_M:
        return None

    start = offsets.get(first.index)
    end = offsets.get(last.index)
    if start is None or end is None or end < start:
        return None

    # 원 프레임(`span`)을 같이 넘긴다 — 키의 UV 칸은 호장 중점의 원 프레임 UV다(m3, D1).
    inputs = service.key_inputs(service.stroke_frame.resample_frames(span, samples), raw_frames=span)
    if inputs is None:
        return None
    return PassIndexEntry(
        pass_index=pass_index,
        row_start=start,
        row_end=end,
        frames=len(span),
        length_m=length,
        key_inputs=inputs,
    )
