#!/usr/bin/env python3
# pyright: reportAny=false, reportUnusedCallResult=false
"""`service.reference_candidates` — 기준은 고를 수 없다(태스크 11 브리프).

전형기 고정 · 대상과 같은 레벨 · 동작 키 일치 · 대상 자신 제외를 서버가 강제하는지
고정한다. `/api/reference`(app.py)는 이 함수를 그대로 감싸는 얇은 HTTP 표면이라
따로 테스트하지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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
    hide_hit: bool = False

    @property
    def frames(self) -> int:
        return len(self.rows)

    @property
    def length_m(self) -> float:
        return 1.0

    @property
    def key_inputs(self) -> object | None:
        """색인(`csv_source.PassIndexEntry.key_inputs`)과 같은 자리 — 착탄 중점 정렬이 읽는다.

        `hide_hit`이면 None을 낸다. 색인에 올랐는데 `hit`이 없는 후보(§1.1이 "현재 코드에선
        0이어야 한다"고 적은 경우)를 만들어, 그런 후보가 버려지지 않고 맨 뒤에 서는지 본다.
        """
        if self.hide_hit:
            return None
        _frames, inputs = service.pass_key_inputs(RawPass(rows=self.rows))
        return inputs


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
        _frames, inputs = service.pass_key_inputs(RawPass(rows=entry.rows))
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


# --- 완화 사다리(S8) ----------------------------------------------------------------
#
# 채점 쪽에는 uv8→uv4→uv2→uv 무시 사다리가 있는데 기준 **패스**를 고르는 쪽에는 없어서
# `key_from(...)` 완전 일치만 봤다. 그래서 "유사한 기준이 없다"가 자주 떴다. 이제 같은 사다리를
# 타되, **어느 단에서 맞았는지 반드시 말한다** — 엄밀 일치와 느슨한 일치를 구별하지 못하면
# 화면이 "같은 동작"이라고 거짓을 말하게 된다.


class LadderFakeSource(FakeCandidateSource):
    """전형기 패스의 **벽면 UV 칸만** 대상과 어긋나게 둔다.

    uv8 키에서는 칸이 달라 하나도 안 맞고, uv4 이하에서는 전부 맞는다. 합성으로 분기를 고정하는
    이유는 실데이터의 uv 셀 경계를 테스트가 다시 계산하지 않기 위해서다(실데이터 근거는 아래
    `test_real_passes_recover_a_reference_only_after_relaxing`).
    """

    def key_of(self, entry: _Entry, params: object) -> str | None:
        key = super().key_of(entry, params)
        if key is None or int(getattr(params, "uv_grid", 8)) < 8:
            return key
        parts = key.split(":")
        parts[3] = "99x99"
        return ":".join(parts)


def _with_query(source: FakeCandidateSource) -> FakeCandidateSource:
    """대상 패스 자신을 원본에 넣는다 — 없으면 `pass_not_found`라 사다리까지 가지도 못한다."""
    source._entries[("A", "20250110", "10", "q", 0)] = _rows(0.0)  # noqa: SLF001
    return source


def test_ladder_relaxes_and_says_which_rung_matched() -> None:
    result = service.reference_candidates(_with_query(LadderFakeSource()), QUERY_REF)
    assert result["status"] == "ok"
    assert result["candidates"], "사다리를 내리면 후보가 살아나야 한다"
    assert result["match_granularity"] == "uv4"
    assert result["match_label"] == "벽면 4×4 칸까지 일치"
    assert result["strict_granularity"] == "uv8"
    assert result["relaxed"] is True, "느슨한 일치를 엄밀 일치처럼 말하면 안 된다"
    counts = {row["granularity"]: row["count"] for row in result["ladder"]}
    assert counts["uv8"] == 0, "엄밀 일치가 0이었다는 사실을 숨기지 않는다"
    assert counts["uv4"] == len(result["candidates"])
    assert [row["granularity"] for row in result["ladder"]] == ["uv8", "uv4", "uv2", "uv_off"]
    # 후보마다 자기가 어느 단에서 걸렸는지 달고 나온다.
    assert {c["match_granularity"] for c in result["candidates"]} == {"uv4"}


def test_strict_match_is_not_reported_as_relaxed() -> None:
    """엄밀 일치로 맞은 패스는 relaxed가 False다 — 둘을 구별 못 하면 화면이 거짓을 말한다."""
    result = service.reference_candidates(_with_query(FakeCandidateSource()), QUERY_REF)
    assert result["match_granularity"] == "uv8"
    assert result["match_label"] == "벽면 8×8 칸까지 일치"
    assert result["relaxed"] is False
    assert {c["match_granularity"] for c in result["candidates"]} == {"uv8"}


def test_no_rung_matches_is_still_ok_status_with_an_empty_ladder() -> None:
    """어느 단에서도 못 찾으면 사다리 네 단이 전부 0이고 맞은 단은 None이다 — 기본값을 채우지 않는다."""
    source = FakeCandidateSource()
    for key in list(source._entries):  # noqa: SLF001
        if key[0] == "전형기":
            source._entries[key] = _rows(0.0, descending=True)  # noqa: SLF001
    source._entries[("A", "20250110", "10", "q", 0)] = _rows(0.0)  # noqa: SLF001
    result = service.reference_candidates(source, QUERY_REF)
    assert result["status"] == "ok"
    assert result["candidates"] == []
    assert result["match_granularity"] is None and result["match_label"] is None
    assert result["relaxed"] is False
    assert all(row["count"] == 0 for row in result["ladder"]), result["ladder"]


def test_the_target_pass_is_excluded_at_every_rung() -> None:
    """자기 자신은 느슨한 단에서도 후보가 아니다 — 들어오면 편차 0이 되어 화면이 거짓을 말한다."""
    source = LadderFakeSource()
    self_query = PassRef(worker="전형기", date="20250101", level="10", session="s1", pass_index=0)
    result = service.reference_candidates(source, self_query)
    assert result["status"] == "ok"
    refs = {(c["worker"], c["date"], c["session"], c["pass_index"]) for c in result["candidates"]}
    assert ("전형기", "20250101", "s1", 0) not in refs
    assert result["relaxed"] is True


def test_real_passes_recover_a_reference_only_after_relaxing() -> None:
    """실데이터 근거(S8). 엄밀 일치로 후보가 0인 패스가 사다리를 내리면 살아나고, 화면이 그 단을 말한다.

    김경호 20250831: L2 1000390_1 #5는 uv8에서 0건인데 uv4에서 살아나고, L3 1000391_1 #10은
    uv8·uv4·uv2 모두 0건인데 uv 칸을 빼면 살아난다. 실데이터가 없으면 skip하지 않고 실패한다.
    (m1 시절 예시였던 L2 #0은 D1(키 m3, 중점 UV) 뒤로 uv8에서 2건이 나와 엄밀 일치가 됐다.)
    """
    from trainer.pose_lab.tests import test_scoring_isolation as isolation

    source, _kwargs = isolation._source_for({"name": "real"})  # noqa: SLF001
    expected = {
        ("2", "1000390_1", 5): "uv4",
        ("3", "1000391_1", 10): "uv_off",
    }
    for (level, session, index), rung in expected.items():
        query = PassRef(worker="김경호", date="20250831", level=level, session=session, pass_index=index)
        result = service.reference_candidates(source, query)
        assert result["status"] == "ok", (level, result)
        counts = {row["granularity"]: row["count"] for row in result["ladder"]}
        assert counts["uv8"] == 0, (level, counts)
        assert result["match_granularity"] == rung, (level, counts)
        assert result["relaxed"] is True
        assert result["candidates"], (level, counts)
        # 후보는 여전히 전형기·같은 레벨이다 — 느슨해진 것은 동작 키뿐이다.
        assert {c["worker"] for c in result["candidates"]} == {"전형기"}
        assert {c["level"] for c in result["candidates"]} == {level}


# --- S8이 **제품 수준에서** 도는가: 고른 기준이 비교까지 살아남는가 -----------------------------
#
# 검증 라운드가 잡은 결함: `reference_candidates`는 사다리로 후보를 냈는데 `service.compare()`가
# 여전히 호출자 uv_grid 기준 완전 일치를 요구해 `{status:'unavailable', reason:'motion_key_mismatch'}`
# 를 냈다. 완화로 고른 기준 4/4가 전부 거기서 죽었고, 화면은 한 자리에서 서로 반대되는 말을 했다 —
# 기준 줄은 "동작 키 완화: 벽면 4×4 칸까지 일치"를 경고색으로 띄우는데 [평가]를 누르면 3D가 열리지
# 않고 "동작 키가 달라 비교하지 않습니다."만 남았다. 커밋 메시지의 "7패스가 살아난다"도 그래서
# 과장이었다 — 기준을 고르는 데까지만 살아났다.


def test_motion_key_match_walks_the_same_ladder_as_the_candidate_search() -> None:
    """게이트와 후보 검색이 **같은 사다리**를 탄다. 갈리면 고른 단과 허락하는 단이 달라진다."""
    source = FakeCandidateSource()
    query = PassRef(worker="A", date="20250110", level="10", session="q", pass_index=0)
    source._entries[(query.worker, query.date, query.level, query.session, query.pass_index)] = _rows(0.0)
    params = service.motion_key.KeyParams()
    result = service.reference_candidates(source, query, params)
    rungs = [row["granularity"] for row in result["ladder"]]
    assert rungs == ["uv8", "uv4", "uv2", "uv_off"], rungs

    _frames, inputs = service.pass_key_inputs(source.load(query), 20)
    same = service.motion_key_match(inputs, inputs, params)
    assert [row["granularity"] for row in same["ladder"]] == rungs
    # 자기 자신과는 가장 엄밀한 단에서 만난다 — 완화라고 부르지 않는다.
    assert same["granularity"] == "uv8" and same["relaxed"] is False
    assert same["strict_granularity"] == "uv8"


def test_compare_accepts_a_relaxed_reference_and_says_which_rung() -> None:
    """실데이터: 사다리가 고른 기준으로 **비교가 실제로 돈다**(S8).

    김경호 20250831 L2 1000390_1 #5는 uv8에서 후보 0건이고 uv4에서 1건이 나온다(m3 키). 그 기준으로
    `compare`를 부르면 `status == "ok"`여야 하고, 응답이 어느 단에서 만났는지(`key_match`)를
    말해야 한다. 이전 판은 여기서 `{status:'unavailable', reason:'motion_key_mismatch'}`를 냈다 —
    화면은 3D를 열지 못하고 "동작 키가 달라 비교하지 않습니다."만 남겼다.
    """
    from trainer.pose_lab.tests import test_scoring_isolation as isolation

    source, kwargs = isolation._source_for({"name": "real"})  # noqa: SLF001
    cases = {("2", "1000390_1", 5): "uv4", ("3", "1000391_1", 10): "uv_off"}
    for (level, session, index), rung in cases.items():
        query = PassRef(worker="김경호", date="20250831", level=level, session=session, pass_index=index)
        found = service.reference_candidates(source, query)
        assert found["match_granularity"] == rung and found["relaxed"] is True, (level, found["ladder"])
        first = found["candidates"][0]
        reference = PassRef(**{key: first[key] for key in ("worker", "date", "level", "session", "pass_index")})

        body = service.compare(source, query, reference, **kwargs)
        assert body["status"] == "ok", (level, body.get("reason"), body.get("key_match"))
        match = body["key_match"]
        # 고른 단과 허락한 단이 같다 — 화면 두 자리가 같은 말을 한다.
        assert match["granularity"] == rung, (level, match)
        assert match["relaxed"] is True and match["strict_granularity"] == "uv8"
        assert match["label"] == found["match_label"]
        # 엄밀 단은 실제로 맞지 않았다는 사실이 사다리에 그대로 남는다.
        walked = {row["granularity"]: row["matched"] for row in match["ladder"]}
        assert walked["uv8"] is False, (level, walked)
        assert walked[rung] is True
        # 3D 뷰가 실제로 열린다 — 이 결함의 사용자 증상이 "3D가 아예 안 뜬다"였다.
        assert body["view"]["layers"]["query"]["joints"], level
        assert body["view"]["layers"]["reference"]["joints"], level


def test_strict_pair_is_not_reported_as_relaxed() -> None:
    """엄밀 일치로 맺은 쌍은 완화라고 부르지 않는다 — 구별을 잃으면 화면이 거짓을 말한다."""
    from trainer.pose_lab.tests import test_scoring_isolation as isolation

    source, kwargs = isolation._source_for({"name": "real"})  # noqa: SLF001
    query = PassRef(worker="김경호", date="20250831", level="1", session="1000389_1", pass_index=0)
    found = service.reference_candidates(source, query)
    assert found["match_granularity"] == "uv8" and found["relaxed"] is False
    first = found["candidates"][0]
    reference = PassRef(**{key: first[key] for key in ("worker", "date", "level", "session", "pass_index")})
    body = service.compare(source, query, reference, **kwargs)
    assert body["status"] == "ok"
    assert body["key_match"]["granularity"] == "uv8"
    assert body["key_match"]["relaxed"] is False
    assert body["key_match"]["ladder"][0]["matched"] is True


def test_compare_still_refuses_a_pair_that_meets_at_no_rung() -> None:
    """어느 단에서도 만나지 않으면 여전히 거절한다 — 사다리는 게이트를 없앤 것이 아니다."""
    source = FakeCandidateSource()
    query = PassRef(worker="A", date="20250110", level="10", session="q", pass_index=0)
    source._entries[(query.worker, query.date, query.level, query.session, query.pass_index)] = _rows(0.0)
    unrelated = PassRef(worker="전형기", date="20250103", level="10", session="s3", pass_index=0)
    body = service.compare(source, query, unrelated)
    assert body["status"] == "unavailable"
    assert body["reason"] == "motion_key_mismatch"
    # 거절할 때도 **어느 단을 훑었는지** 말한다. 조용히 없애면 왜 못 맺었는지 알 수 없다.
    match = body["key_match"]
    assert match["granularity"] is None and match["label"] is None
    assert [row["granularity"] for row in match["ladder"]] == ["uv8", "uv4", "uv2", "uv_off"]
    assert all(row["matched"] is False for row in match["ladder"]), match["ladder"]


# --- 착탄 중점 거리 순 정렬(P1, 사전등록 2026-09-23) --------------------------------
#
# 화면은 `candidates[0]`을 자동 기준으로 보여 주고 "[다른 기준 ▸]"로 그 목록을 순환한다.
# 목록이 세션 파일 순서로 서 있으면 착탄 위치와 무관한 패스가 먼저 나온다(실측: 자동 기준까지
# 0.360 m인데 후보 전체의 거리 중앙값은 0.195 m였다). 그래서 목록을 **질의와의 착탄 중점
# 거리** 순으로 세운다. 정렬은 `service.reference_candidates` **한 곳**에만 있고 화면과
# 개선율 연구가 그것을 나눠 쓴다.


class _StubInputs:
    """`KeyInputs`에서 정렬이 읽는 칸(`hit`)만 가진 대역."""

    def __init__(self, hit: tuple[float, float, float] | None) -> None:
        self.hit = hit


@dataclass(frozen=True, slots=True)
class _StubEntry:
    pass_index: int
    hit: tuple[float, float, float] | None

    @property
    def frames(self) -> int:
        return 20

    @property
    def length_m(self) -> float:
        return 1.0

    @property
    def key_inputs(self) -> object | None:
        return None if self.hit is None else _StubInputs(self.hit)


class OrderingFakeSource:
    """후보의 **착탄 중점만** 정해 두는 대역. 동작 키는 전부 질의와 같아 첫 단(uv8)에서 멈춘다.

    거리를 정렬이 실제로 무엇으로 재는지 고정하려고, 키 일치·레벨·자기 제외 같은 다른 조건을
    전부 통과시켜 둔다. `reverse`는 `passes`가 후보를 내는 **입력 순서**만 뒤집는다 —
    정렬이 입력 순서에 의존하는지 보는 손잡이다.
    """

    def __init__(self, rows: list[tuple[str, str, int, tuple[float, float, float] | None]],
                 reverse: bool = False) -> None:
        self.rows = rows
        self.reverse = reverse
        self._query_rows = _rows(0.0)
        _frames, self.query_inputs = service.pass_key_inputs(RawPass(rows=self._query_rows))
        assert self.query_inputs is not None

    def workers(self) -> list[str]:
        return ["A", service.EXPERT_WORKER]

    def load(self, ref: PassRef) -> RawPass:
        if ref != ORDER_QUERY:
            raise LookupError(f"not found: {ref}")
        return RawPass(rows=self._query_rows)

    def passes(self, worker: str, level: str | None = None, date: str | None = None, limit: int = 50):
        if worker != service.EXPERT_WORKER:
            return [], 0
        found = [
            (_Item(worker=worker, date=date_, level="10", session=session),
             _StubEntry(pass_index=index, hit=hit))
            for date_, session, index, hit in self.rows
        ]
        if self.reverse:
            found = found[::-1]
        return found[:limit], len(found)

    def key_of(self, entry: _StubEntry, params: object) -> str | None:
        # 질의와 **같은 재료**로 키를 만든다 — 모든 후보가 어느 단에서도 질의와 일치한다.
        return service.key_from(self.query_inputs, params)


ORDER_QUERY = PassRef(worker="A", date="20250110", level="10", session="q", pass_index=0)


def _shifted(base: tuple[float, float, float], delta: float) -> tuple[float, float, float]:
    """질의 착탄점에서 한 축으로 `delta` m 떨어진 점. 거리는 정확히 |delta|다."""
    return (base[0] + delta, base[1], base[2])


def _order_of(result: dict[str, object]) -> list[tuple[str, str, int]]:
    return [(c["date"], c["session"], c["pass_index"]) for c in result["candidates"]]


def test_nearest_candidate_comes_first_and_distances_never_decrease() -> None:
    """성질 2(§7-2): `candidates[0]`이 가장 가깝고, 목록의 거리 수열이 비내림차순이다. (F4)"""
    source = OrderingFakeSource([])
    base = source.query_inputs.hit
    source.rows = [
        ("20250101", "far", 0, _shifted(base, 0.9)),
        ("20250102", "mid", 1, _shifted(base, 0.4)),
        ("20250103", "near", 2, _shifted(base, 0.05)),
    ]
    result = service.reference_candidates(source, ORDER_QUERY)
    assert result["status"] == "ok"
    distances = [c["hit_distance_m"] for c in result["candidates"]]
    assert distances == pytest.approx([0.05, 0.4, 0.9]), distances
    assert distances[0] == pytest.approx(min(distances)), "첫 후보가 가장 가깝지 않다"
    assert distances == sorted(distances), f"거리 수열이 비내림차순이 아니다: {distances}"
    assert _order_of(result)[0] == ("20250103", "near", 2)


def test_ties_break_by_date_session_pass_index_and_the_call_is_deterministic() -> None:
    """성질 3(§7-3): 거리가 정확히 같으면 `(date, session, pass_index)` 순이고,
    입력 순서를 뒤집어도·두 번 불러도 결과가 같다. 정렬에 난수는 없다."""
    source = OrderingFakeSource([])
    base = source.query_inputs.hit
    same = _shifted(base, 0.25)
    source.rows = [
        ("20250103", "s3", 0, same),
        ("20250101", "s1", 7, same),
        ("20250101", "s1", 2, same),
        ("20250102", "s2", 0, same),
    ]
    expected = [("20250101", "s1", 2), ("20250101", "s1", 7), ("20250102", "s2", 0), ("20250103", "s3", 0)]
    forward = service.reference_candidates(source, ORDER_QUERY)
    again = service.reference_candidates(source, ORDER_QUERY)
    assert _order_of(forward) == expected, _order_of(forward)
    assert _order_of(again) == expected, "같은 호출 두 번이 다른 순서를 냈다"
    source.reverse = True
    backward = service.reference_candidates(source, ORDER_QUERY)
    assert _order_of(backward) == expected, (
        f"입력 순서를 뒤집었더니 동률의 순서가 바뀌었다: {_order_of(backward)}"
    )
    assert [c["hit_distance_m"] for c in forward["candidates"]] == pytest.approx([0.25] * 4)


def test_a_candidate_without_a_hit_is_kept_and_placed_last() -> None:
    """성질 4(§7-4): 거리를 못 구하는 후보는 **버리지 않고** 맨 뒤에 둔다. (F5)

    버리면 후보 집합이 바뀌어 화면의 "N개 중 k번째"가 틀리고, 정렬이 고르는 일에 끼어든 것이 된다.
    """
    source = OrderingFakeSource([])
    base = source.query_inputs.hit
    source.rows = [
        ("20250101", "nohit", 0, None),
        ("20250102", "far", 0, _shifted(base, 0.8)),
        ("20250103", "near", 0, _shifted(base, 0.1)),
    ]
    result = service.reference_candidates(source, ORDER_QUERY)
    assert len(result["candidates"]) == 3, "거리 없는 후보가 조용히 사라졌다"
    assert _order_of(result) == [("20250103", "near", 0), ("20250102", "far", 0), ("20250101", "nohit", 0)]
    assert result["candidates"][-1]["hit_distance_m"] is None
    assert result["ladder"][0]["count"] == 3


def test_a_non_finite_distance_is_treated_as_missing() -> None:
    """`nan`/`inf`도 '거리를 못 구한 후보'다 — 정렬 키에 그대로 넣으면 순서가 정의되지 않는다."""
    source = OrderingFakeSource([])
    base = source.query_inputs.hit
    source.rows = [
        ("20250101", "nan", 0, (float("nan"), base[1], base[2])),
        ("20250102", "near", 0, _shifted(base, 0.1)),
    ]
    result = service.reference_candidates(source, ORDER_QUERY)
    assert _order_of(result) == [("20250102", "near", 0), ("20250101", "nan", 0)]
    assert result["candidates"][-1]["hit_distance_m"] is None


# --- 실데이터 대조(P1) --------------------------------------------------------------

REAL_ROOT = Path("/data/sources/professionals")
_REAL: dict[str, object] = {}

# 사용자 화면 패스와 그 정렬 전 자동 기준. 출처 `/output/samelevel/screen_impact.json`(직전 사전등록 R6).
R6_QUERY = PassRef(worker="김경호", date="20250831", level="1", session="1000389_1", pass_index=0)
R6_REFERENCE = PassRef(worker="전형기", date="20250904", level="1", session="1000564_1", pass_index=15)
R6_DISTANCE_M = 0.360


def _real_source():
    if not REAL_ROOT.is_dir():
        pytest.fail(f"실데이터가 없다({REAL_ROOT}) — 이 대조는 skip하지 않는다")
    from trainer.pose_lab import csv_source

    if "source" not in _REAL:
        _REAL["source"] = csv_source.CsvPassSource(REAL_ROOT)
    return _REAL["source"]


def _real_entry(source, ref: PassRef):
    item = next(
        s for s in source.sessions()
        if (s.worker, s.date, s.level, s.session) == (ref.worker, ref.date, ref.level, ref.session)
    )
    return next(e for e in source.index_of(item) if e.pass_index == ref.pass_index)


def _real_queries() -> list[PassRef]:
    """사용자 화면 세션의 실패스 몇 개. 레벨 1·3, 후보가 실제로 여럿인 것들이다."""
    return [
        PassRef(worker="김경호", date="20250831", level="1", session="1000389_1", pass_index=index)
        for index in (0, 1, 3, 5, 9)
    ] + [
        PassRef(worker="김경호", date="20250831", level="3", session="1000391_1", pass_index=index)
        for index in (0, 4, 10)
    ]


def _refs_of(result) -> list[PassRef]:
    return [PassRef(**{k: c[k] for k in ("worker", "date", "level", "session", "pass_index")})
            for c in result["candidates"]]


def test_sorting_changes_the_order_and_nothing_else_on_real_files(monkeypatch) -> None:
    """성질 1(§7-1, 실데이터): 정렬은 **순서만** 바꾼다. (F1·F2)

    대조군은 `_by_hit_distance`를 상수 키로 바꾼 같은 함수다 — `list.sort`가 안정 정렬이라
    그때의 목록은 정렬 도입 **전** 순서(세션 파일 순서 × 패스 순번)와 같다.
    후보의 `PassRef` 다중집합과 `match_granularity`·`ladder`·`truncated`·`sessions_scanned`·
    `sessions_total`·`query_key`가 모두 같아야 한다.
    """
    source = _real_source()
    sorted_results = {q: service.reference_candidates(source, q) for q in _real_queries()}
    monkeypatch.setattr(service, "_by_hit_distance", lambda row: 0)
    control_results = {q: service.reference_candidates(source, q) for q in _real_queries()}

    reordered = 0
    for query, after in sorted_results.items():
        before = control_results[query]
        assert after["status"] == before["status"] == "ok", query
        assert sorted(_refs_of(after), key=str) == sorted(_refs_of(before), key=str), (
            f"정렬이 후보 집합을 바꿨다: {query}"
        )
        for field in ("match_granularity", "ladder", "truncated", "sessions_scanned",
                      "sessions_total", "query_key", "match_label", "relaxed"):
            assert after[field] == before[field], f"정렬이 {field}를 바꿨다: {query}"
        if _refs_of(after) != _refs_of(before):
            reordered += 1
    assert reordered, "어느 질의에서도 순서가 바뀌지 않았다 — 이 대조가 공허하다"


def test_the_first_candidate_is_the_nearest_on_real_files() -> None:
    """성질 2(§7-2, 실데이터): 모든 질의에서 `candidates[0]`의 거리가 후보 최솟값이고
    거리 수열이 비내림차순이다. 실데이터에 거리 없는 후보가 있으면 그 사실이 드러난다(F5)."""
    source = _real_source()
    missing = 0
    for query in _real_queries():
        result = service.reference_candidates(source, query)
        distances = [c["hit_distance_m"] for c in result["candidates"]]
        assert len(distances) > 1, f"후보가 하나뿐이면 순서 단언이 공허하다: {query}"
        missing += sum(d is None for d in distances)
        known = [d for d in distances if d is not None]
        assert distances[: len(known)] == known, f"거리 없는 후보가 앞에 섰다: {query}"
        assert known == sorted(known), f"거리 수열이 비내림차순이 아니다: {query} {known}"
        assert known[0] == min(known), f"첫 후보가 가장 가깝지 않다: {query}"
    assert missing == 0, f"거리를 못 구한 후보 {missing}개 — §1.1의 전제가 틀렸다(F5)"


def test_the_distance_definition_reproduces_the_r6_measurement() -> None:
    """성질 7(§7-7): 김경호 L1 1000389_1 #0 ↔ 전형기 1000564_1 #15의 §1.1 거리가
    직전 사전등록 R6의 0.360 m를 소수 셋째 자리까지 재현한다. (F13)

    두 문서의 숫자를 나란히 읽을 수 있으려면 거리의 정의가 같아야 한다.
    """
    source = _real_source()
    query_entry = _real_entry(source, R6_QUERY)
    reference_entry = _real_entry(source, R6_REFERENCE)
    distance = service.hit_distance_m(query_entry.key_inputs.hit, reference_entry)
    assert distance == pytest.approx(R6_DISTANCE_M, abs=5e-4), (
        f"R6이 잰 0.360 m가 재현되지 않는다: {distance}"
    )
