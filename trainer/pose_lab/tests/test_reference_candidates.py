#!/usr/bin/env python3
# pyright: reportAny=false, reportUnusedCallResult=false
"""`service.reference_candidates` — 기준은 고를 수 없다(태스크 11 브리프).

전형기 고정 · 대상과 같은 레벨 · 동작 키 일치 · 대상 자신 제외를 서버가 강제하는지
고정한다. `/api/reference`(app.py)는 이 함수를 그대로 감싸는 얇은 HTTP 표면이라
따로 테스트하지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from trainer.pose_lab import service
from trainer.pose_lab.service import PassRef, RawPass
from trainer.pose_lab.tests.test_app import _rows


@dataclass(frozen=True, slots=True)
class _Item:
    worker: str
    date: str
    level: str
    session: str


@dataclass(frozen=True, slots=True)
class _Entry:
    pass_index: int
    rows: list[dict[str, str]]

    @property
    def frames(self) -> int:
        return len(self.rows)

    @property
    def length_m(self) -> float:
        return 1.0


class FakeCandidateSource:
    """`reference_candidates`가 실제로 쓰는 표면(workers/load/passes/key_of)만
    구현한다 — `PassSource` 프로토콜보다 넓다(`csv_source.CsvPassSource`와 같다)."""

    def __init__(self) -> None:
        # worker "전형기": 레벨10에 대상과 같은 키의 패스 둘 + 다른 레벨의 패스 하나.
        # worker "다른전문가": 전형기가 아니므로 후보에 절대 들어가면 안 된다.
        self._entries: dict[tuple[str, str, str, str, int], list[dict[str, str]]] = {
            ("전형기", "20250101", "10", "s1", 0): _rows(0.0),
            ("전형기", "20250102", "10", "s2", 0): _rows(0.0),
            ("전형기", "20250103", "10", "s3", 0): _rows(0.0, descending=True),  # 다른 키
            ("전형기", "20250104", "7", "s4", 0): _rows(0.0),  # 다른 레벨
            ("다른전문가", "20250105", "10", "s5", 0): _rows(0.0),
        }

    def workers(self) -> list[str]:
        return sorted({key[0] for key in self._entries})

    def load(self, ref: PassRef) -> RawPass:
        key = (ref.worker, ref.date, ref.level, ref.session, ref.pass_index)
        rows = self._entries.get(key)
        if rows is None:
            raise LookupError(f"not found: {ref}")
        return RawPass(rows=rows)

    def passes(
        self, worker: str, level: str | None = None, date: str | None = None, limit: int = 50
    ) -> tuple[list[tuple[_Item, _Entry]], int]:
        matched = [
            (_Item(worker=w, date=d, level=lv, session=s), _Entry(pass_index=pi, rows=rows))
            for (w, d, lv, s, pi), rows in self._entries.items()
            if w == worker and (level is None or lv == level)
        ]
        return matched[:limit], len(matched)

    def key_of(self, entry: _Entry, params: object) -> str | None:
        frames = service.pass_frames(RawPass(rows=entry.rows))
        inputs = service.key_inputs(frames)
        return None if inputs is None else service.key_from(inputs, params)


class FakeSourceWithoutExpert:
    def workers(self) -> list[str]:
        return ["A", "B"]

    def load(self, ref: PassRef) -> RawPass:  # pragma: no cover - 안 불림
        raise LookupError(ref)

    def passes(self, worker, level=None, date=None, limit=50):  # pragma: no cover
        return [], 0

    def key_of(self, entry, params):  # pragma: no cover
        return None


QUERY_REF = PassRef(worker="A", date="20250110", level="10", session="q", pass_index=0)


def test_candidates_are_expert_only_same_level_matching_key_excluding_self() -> None:
    source = FakeCandidateSource()
    source._entries[("A", "20250110", "10", "q", 0)] = _rows(0.0)  # 대상 자신

    result = service.reference_candidates(source, QUERY_REF)
    assert result["status"] == "ok"
    assert result["expert_worker"] == "전형기"
    workers = {c["worker"] for c in result["candidates"]}
    assert workers == {"전형기"}, "전형기가 아닌 작업자가 섞이면 안 된다"
    levels = {c["level"] for c in result["candidates"]}
    assert levels == {"10"}, "다른 레벨이 섞이면 안 된다"
    keys = {c["motion_key"] for c in result["candidates"]}
    assert keys == {result["query_key"]}, "동작 키가 다른 패스가 섞이면 안 된다"
    assert len(result["candidates"]) == 2, "같은 키의 두 전형기 패스만 남아야 한다"


def test_target_pass_itself_is_excluded_when_query_is_the_expert() -> None:
    """대상이 전형기 본인의 패스여도 그 패스 자신만 빠지고 나머지는 후보로 남는다."""
    source = FakeCandidateSource()
    self_query = PassRef(worker="전형기", date="20250101", level="10", session="s1", pass_index=0)
    result = service.reference_candidates(source, self_query)
    assert result["status"] == "ok"
    refs = {(c["worker"], c["date"], c["session"], c["pass_index"]) for c in result["candidates"]}
    assert ("전형기", "20250101", "s1", 0) not in refs, "대상 자신과 비교하면 편차가 0이 된다"
    assert ("전형기", "20250102", "s2", 0) in refs


def test_no_expert_worker_is_a_distinct_reason() -> None:
    """전형기가 이 데이터셋에 아예 없으면 '기준 없음'이 아니라 그 사실 자체를 말한다."""
    result = service.reference_candidates(FakeSourceWithoutExpert(), QUERY_REF)
    assert result == {"status": "unavailable", "reason": "no_expert_worker"}


def test_pass_not_found_when_the_target_pass_does_not_exist() -> None:
    source = FakeCandidateSource()
    missing = PassRef(worker="A", date="20250110", level="10", session="missing", pass_index=0)
    result = service.reference_candidates(source, missing)
    assert result == {"status": "unavailable", "reason": "pass_not_found"}


def test_empty_candidates_list_is_still_ok_status() -> None:
    """일치하는 전형기 패스가 없으면 '없다'와 '재지 못했다'는 다른 사건이다 — 빈
    목록이지 unavailable이 아니다(브리프: "이 동작키의 전형기 패스가 없다")."""
    source = FakeCandidateSource()
    del source._entries[("전형기", "20250101", "10", "s1", 0)]
    del source._entries[("전형기", "20250102", "10", "s2", 0)]
    source._entries[("A", "20250110", "10", "q", 0)] = _rows(0.0)
    result = service.reference_candidates(source, QUERY_REF)
    assert result["status"] == "ok"
    assert result["candidates"] == []
