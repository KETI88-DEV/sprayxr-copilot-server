#!/usr/bin/env python3
# pyright: reportAny=false, reportUnusedCallResult=false

from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

improvement_study = importlib.import_module("improvement_study")
world_axes = importlib.import_module("world_axes")
measured_score = importlib.import_module("measured_score")


def test_summary_reports_median_quartiles_and_threshold_share() -> None:
    summary = improvement_study.summarize([0.1, 0.2, 0.5, 0.75, 0.8, 0.9], threshold=0.70)
    assert summary["n"] == 6
    assert math.isclose(summary["median"], 0.625, abs_tol=1e-9)
    assert math.isclose(summary["at_or_above"], 0.5, abs_tol=1e-9)


def test_none_values_are_excluded_not_zeroed() -> None:
    """계산되지 않은 split을 0으로 넣으면 분포가 근거 없이 내려간다."""
    summary = improvement_study.summarize([0.8, None, 0.9], threshold=0.70)
    assert summary["n"] == 2
    assert math.isclose(summary["at_or_above"], 1.0, abs_tol=1e-9)


def test_empty_input_reports_none_not_zero() -> None:
    summary = improvement_study.summarize([], threshold=0.70)
    assert summary["n"] == 0
    assert summary["median"] is None
    assert summary["at_or_above"] is None


def test_quartiles_are_reported() -> None:
    summary = improvement_study.summarize([0.1, 0.2, 0.3, 0.4], threshold=0.70)
    assert summary["q1"] is not None
    assert summary["q3"] is not None
    assert summary["q1"] <= summary["median"] <= summary["q3"]


def _unknown_outcome() -> object:
    """발동 여부를 재지 않은 결과 하나. 민감도 실행이 내는 모양이다."""
    return improvement_study.Outcome(
        query=None, reference=None, key="m3:horizontal:right:1x2:center-near-high",
        status="ok", reason=None, same=0.2, split=0.1, split_reason=None,
        granularity="uv2", reference_n=30, split_n_a=15, split_n_b=15,
        gate="fail", gate_reason=None, normalization="measured",
        fired=None, split_fired=None, ceiling=0.5,
        before_same_score=0.3, before_split_score=0.3,
    )


def test_share_fired_is_none_when_nothing_was_measured() -> None:
    """아무것도 재지 않았을 때의 0.0은 '하나도 안 발동했다'로 읽힌다. 그건 거짓이다.

    민감도 실행은 compute_extras=False라 발동 여부를 아예 계산하지 않는다. 그때
    전체 건수로 나누면 0/149 = 0.0이 나오고, 옆에 unknown=149가 적혀 있어도 그
    0.0은 측정값처럼 읽힌다.
    """
    unknown = [_unknown_outcome() for _ in range(3)]
    block = improvement_study._fire_rate_block([], [], unknown, lambda o: o.same)
    assert block["unknown"] == 3
    assert block["share_fired"] is None


def test_share_fired_divides_by_what_was_actually_measured() -> None:
    """분모는 전체가 아니라 발동 여부를 아는 건수다."""
    fired = [_unknown_outcome()]
    not_fired = [_unknown_outcome()]
    unknown = [_unknown_outcome() for _ in range(8)]
    block = improvement_study._fire_rate_block(fired, not_fired, unknown, lambda o: o.same)
    assert math.isclose(block["share_fired"], 0.5, abs_tol=1e-9)


def test_ceiling_leaves_the_left_hand_and_the_head_axes_that_are_never_issued() -> None:
    """천장은 '**발부할 수 있는 것**을 전부 완벽히 따랐을 때'다.

    왼손은 지시 대상이 아니다. 머리는 자세 문장 **하나**뿐이라 `CORRECTABLE["head"]`의
    두 축 중 σ가 큰 하나만 0이 되고, 나머지 하나와 lateral은 남는다. 머리의 두 축을
    모두 0으로 치면 천장이 시스템이 닿을 수 없는 값 위에 앉는다.

    왼손의 미클립 값(4.0·√3 ≈ 6.928)은 `measured_score.MAX_SIGMA`(6.0)를 넘는다 —
    `score_track`이 절대 내지 않을 스케일이므로 `MAX_SIGMA`에서 클립된 값을 고정한다
    (Fix round 2, F1). 클립을 빼먹으면 이 값이 6.928로 나와 이 assert가 실패한다.
    """
    sigmas = {
        "right": {"wall": 3.0, "vertical": 3.0, "lateral": 3.0},
        "head": {"wall": 2.0, "vertical": 3.0, "lateral": 2.5},
        "left": {"wall": 4.0, "vertical": 4.0, "lateral": 4.0},
    }
    residual = improvement_study.ceiling_residual_sigma(sigmas)
    assert residual["left"] == pytest.approx(measured_score.MAX_SIGMA)
    # vertical(3.0)만 0이 된다 — wall 2.0과 lateral 2.5가 남는다.
    assert residual["head"] == pytest.approx(math.sqrt(2.0**2 + 2.5**2))
    assert residual["right"] == pytest.approx(0.0)


def test_the_head_ceiling_zeroes_exactly_one_axis() -> None:
    """머리의 두 자세 축을 다 0으로 치는 구현이면 이 검사가 실패해야 한다.

    앞 검사는 잔차 값 하나만 고정하므로 0이 된 축의 **개수**가 바뀌어도 값이 우연히
    맞을 수 있다. 여기서는 두 자세 축을 같은 값으로 두고 개수를 직접 잰다 — 하나만
    0이면 남는 것은 그 값 하나(2.0), 둘 다 0이면 0이다.
    """
    sigmas = {"head": {"wall": 2.0, "vertical": 2.0, "lateral": 0.0}}
    residual = improvement_study.ceiling_residual_sigma(sigmas)
    assert residual["head"] == pytest.approx(2.0)


def test_the_scorer_issues_exactly_one_head_axis() -> None:
    """천장이 모형으로 삼는 사실 자체를 잰다 — 머리는 문장 하나만 나간다.

    `CORRECTABLE["head"]`가 `STANCE_AXES`와 같은지만 보는 검사는 축의 **개수**를 보지
    못한다. 여기서는 두 자세 축이 모두 임계를 크게 넘는 델타를 주고, 실제로 발부되는
    머리 교정이 하나인지 센다. 머리가 다축 발부로 바뀌면 이 검사가 실패하고, 그때는
    `SINGLE_AXIS_POINTS`에서 머리를 빼야 한다는 뜻이다.
    """
    delta = measured_score.PointDelta(
        point="head",
        sigma=5.0,
        local_delta=(0.4, 0.0, 0.3),
        samples=10,
        axis_sigmas={"wall": 4.0, "vertical": 3.0, "lateral": 0.0},
    )
    issued = measured_score.issued_axes([delta])
    assert [correction.point for correction in issued] == ["head"]


def test_ceiling_residual_sigma_clamps_at_the_scorers_max_sigma() -> None:
    """`score_track`은 결합 σ를 `min(MAX_SIGMA, ·)`로 클립한다 — 천장도 같은 클립이
    있어야, 채점기가 절대 낼 수 없는 스케일 위에 천장이 앉지 않는다.

    클립이 없으면 sqrt(5²×3) ≈ 8.660이 그대로 남아 이 assert가 실패한다.
    """
    sigmas = {"left": {"wall": 5.0, "vertical": 5.0, "lateral": 5.0}}
    residual = improvement_study.ceiling_residual_sigma(sigmas)
    assert residual["left"] == pytest.approx(measured_score.MAX_SIGMA)


def test_correctable_head_axes_match_the_scorer() -> None:
    """천장이 쓰는 '지시 가능한 축'과 실제 지시 규칙이 갈라지면 천장이 틀린다."""
    assert improvement_study.CORRECTABLE["head"] == measured_score.STANCE_AXES


def test_cache_fingerprint_covers_the_axis_convention(tmp_path, monkeypatch) -> None:
    """캐시는 CSV 트리뿐 아니라 **키를 만든 규약**에도 묶여야 한다.

    캐시가 담는 것은 패스 구간과 동작 키이고, 동작 키는 world_axes.UP_INDEX에 의존한다.
    트리만 지문으로 삼으면 옛 축으로 지어진 캐시가 트리가 그대로인 한 조용히 통과하고,
    그러면 낡은 버킷에서 표집해 비교 전체가 소리 없이 무효가 된다 — 축 수정 직후
    실제로 그 일이 일어났다.
    """
    (tmp_path / "a.csv").write_text("x", encoding="utf-8")
    before = improvement_study._fingerprint(tmp_path)
    monkeypatch.setattr(world_axes, "UP_INDEX", 2)
    after = improvement_study._fingerprint(tmp_path)
    assert before != after, "규약이 바뀌었는데 지문이 같다 — 낡은 캐시가 통과한다"
    assert before["up_index"] != after["up_index"]


# --- 쌍 규칙 M0(사전등록 docs/measurements/2026-09-22-same-level-pairs-d1-prereg.md §1, §5) ----------

import random  # noqa: E402
from collections import defaultdict  # noqa: E402

from trainer.pose_lab import service  # noqa: E402
from trainer.pose_lab.service import PassRef  # noqa: E402

motion_key = importlib.import_module("motion_key")

REAL_ROOT = Path("/data/sources/professionals")


def _head_select_pairs(candidates, expert_worker, pairs, seed):
    """HEAD `2e71c4b`의 `select_pairs` 본문을 그대로 옮긴 것(바꾸지 않는다). 옛 규칙 O의 비트 재현 기준."""
    rng = random.Random(seed)
    shuffled = list(candidates)
    rng.shuffle(shuffled)
    queries = shuffled[: max(0, pairs)]

    by_key_expert = defaultdict(list)
    for candidate in candidates:
        if candidate.worker == expert_worker:
            by_key_expert[candidate.key].append(candidate)

    result = []
    for query in queries:
        options = [c for c in by_key_expert.get(query.key, []) if c.ref != query.ref]
        reference = rng.choice(options).ref if options else None
        result.append(improvement_study.Pair(query=query.ref, reference=reference, key=query.key))
    return result


def _synthetic_candidates() -> list:
    """두 작업자 × 레벨 3개 × 세션 2개 × 패스 6개, 키 3종. 같은 키가 레벨을 넘나든다."""
    out = []
    for worker in ("전형기", "김경호"):
        for level in ("1", "2", "3"):
            for session in ("s1", "s2"):
                for index in range(6):
                    ref = PassRef(worker=worker, date="2025010" + level, level=level, session=session, pass_index=index)
                    out.append(improvement_study.Candidate(ref=ref, worker=worker, key=f"m3:k{index % 3}"))
    return out


def test_legacy_rule_reproduces_head_select_pairs_exactly() -> None:
    """성질 6: `--pair-rule legacy`는 HEAD의 `select_pairs`와 같은 쌍을 낸다(79.21% 비트 재현의 전제)."""
    candidates = _synthetic_candidates()
    for seed in (1, 20260828):
        legacy = improvement_study.build_pairs("legacy", None, candidates, "전형기", 40, seed)
        head = _head_select_pairs(candidates, "전형기", 40, seed)
        assert legacy == head, "legacy 규칙이 HEAD select_pairs와 다른 쌍을 낸다 — 옛 기준선 재현이 깨진다"
    # 이 합성 표본에서 옛 규칙은 실제로 다른 레벨 짝을 낸다 — 위 동등성이 공허하지 않다는 확인.
    assert any(p.reference is not None and p.reference.level != p.query.level for p in legacy)


def test_query_sample_is_the_legacy_shuffle() -> None:
    """F2의 전제: N·O·O+L 어느 규칙이든 질의 표본은 같은 셔플에서 나온다."""
    candidates = _synthetic_candidates()
    sample = [c.ref for c in improvement_study.sample_queries(candidates, 40, 20260828)]
    head = [p.query for p in _head_select_pairs(candidates, "전형기", 40, 20260828)]
    same_level = [p.query for p in improvement_study.select_pairs_legacy_same_level(candidates, "전형기", 40, 20260828)]
    assert sample == head == same_level


def _fake_candidates_fn(counts: dict):
    """질의별 후보 수를 정해 두는 `reference_candidates` 대역. 후보는 질의와 같은 레벨의 전형기 패스다."""
    def fn(source, ref, params, samples, limit):
        assert samples == improvement_study.SCREEN_SAMPLES and limit == improvement_study.SCREEN_LIMIT
        rows = [
            {"worker": "전형기", "date": f"2025{k:04d}", "level": ref.level, "session": f"e{k}", "pass_index": k}
            for k in range(counts.get(ref, 5))
        ]
        return {"status": "ok", "match_granularity": "uv8", "truncated": False, "candidates": rows}
    return fn


def test_per_query_stream_is_isolated_from_other_queries() -> None:
    """성질 3: 한 질의의 후보 수가 바뀌어도 다른 질의가 받는 기준은 그대로다(질의별 난수열)."""
    candidates = _synthetic_candidates()
    queries = [c.ref for c in improvement_study.sample_queries(candidates, 24, 7)]
    first = queries[0]
    base = improvement_study.select_pairs_screen(
        None, candidates, "전형기", 24, 7, candidates_fn=_fake_candidates_fn({first: 0})
    )
    moved = improvement_study.select_pairs_screen(
        None, candidates, "전형기", 24, 7, candidates_fn=_fake_candidates_fn({first: 9})
    )
    assert base[0].reference is None and moved[0].reference is not None
    changed = [a.query for a, b in zip(base[1:], moved[1:]) if a.reference != b.reference]
    assert changed == [], f"첫 질의의 후보 수만 바꿨는데 다른 질의 {len(changed)}개의 기준이 바뀌었다"


def test_seed_ref_changes_only_the_choice_not_the_candidate_set() -> None:
    """귀무 재추첨(§3.2)은 기준 선택만 바꾼다 — 질의와 후보 집합은 같다."""
    candidates = _synthetic_candidates()
    fn = _fake_candidates_fn({})
    a = improvement_study.select_pairs_screen(None, candidates, "전형기", 24, 7, seed_ref=1, candidates_fn=fn)
    b = improvement_study.select_pairs_screen(None, candidates, "전형기", 24, 7, seed_ref=2, candidates_fn=fn)
    assert [p.query for p in a] == [p.query for p in b]
    assert [p.candidates for p in a] == [p.candidates for p in b]
    assert [p.reference for p in a] != [p.reference for p in b]


def test_fingerprint_carries_the_key_prefix(tmp_path, monkeypatch) -> None:
    """성질 4: 키 정의(접두어)가 바뀌면 캐시 지문이 달라져 옛 키 캐시가 통과하지 못한다."""
    (tmp_path / "a.csv").write_text("x", encoding="utf-8")
    before = improvement_study._fingerprint(tmp_path)
    assert before["key_prefix"] == motion_key.KEY_PREFIX
    monkeypatch.setattr(improvement_study.motion_key, "KEY_PREFIX", "m9")
    after = improvement_study._fingerprint(tmp_path)
    assert before != after, "키 접두어가 바뀌었는데 지문이 같다 — 옛 키 캐시가 조용히 통과한다"


def test_key_prefix_guard_rejects_stale_keys() -> None:
    """F6: 캐시·쌍 키 중 하나라도 현행 접두어가 아니면 실행을 멈춘다."""
    good = _synthetic_candidates()
    improvement_study.assert_key_prefix(good, [])
    stale = [*good, improvement_study.Candidate(ref=good[0].ref, worker="x", key="m0:old")]
    with pytest.raises(RuntimeError):
        improvement_study.assert_key_prefix(stale, [])


def test_artifact_prefix_guard_rejects_an_old_key_artifact() -> None:
    """F6(산포 쪽): 산포 아티팩트에 현행이 아닌 접두어 버킷이 하나라도 있으면 실행을 멈춘다.

    현행 접두어만 든 아티팩트는 통과해야 한다 — 늘 던지는 검사는 실패할 수 없는 단언과 같다.
    """
    from trainer.pose_lab.spread_reference import SpreadReference

    prefix = improvement_study.motion_key.KEY_PREFIX
    current = {"worker": "전형기", "granularities": {"uv8": {f"{prefix}:a:b:1x1:c": {}}, "uv_off": {f"{prefix}:a:b:na:c": {}}}}
    improvement_study.assert_artifact_prefix(SpreadReference(current), "current.json")
    mixed = {"worker": "전형기", "granularities": {**current["granularities"], "uv4": {"m1:a:b:1x1:c": {}}}}
    with pytest.raises(RuntimeError, match="m1"):
        improvement_study.assert_artifact_prefix(SpreadReference(mixed), "mixed.json")


def test_ladder_mismatch_is_unavailable_not_dropped(monkeypatch) -> None:
    """F4: 후보를 고른 단과 compare가 허락한 단이 갈리면 unavailable(ladder_mismatch)로 남긴다."""
    q = PassRef(worker="김경호", date="20250101", level="1", session="s", pass_index=0)
    r = PassRef(worker="전형기", date="20250102", level="1", session="t", pass_index=1)
    pair = improvement_study.Pair(query=q, reference=r, key="m3:x", reference_granularity="uv8", candidate_count=1)
    monkeypatch.setattr(
        improvement_study.service, "compare",
        lambda *a, **k: {"status": "ok", "key_match": {"granularity": "uv4"}},
    )
    outcome = improvement_study.run_pair(None, None, pair, compute_extras=False)
    assert outcome.status == "unavailable" and outcome.reason == "ladder_mismatch"
    assert outcome.reference_granularity == "uv8" and outcome.key_match_granularity == "uv4"


_REAL_SOURCE: dict = {}


def _real_queries():
    """실데이터 질의 몇 개. 레벨 2·3, 다른 작업자 둘과 전형기 자신(같은 세션의 다른 패스가 후보로 남는가)."""
    if not REAL_ROOT.is_dir():
        pytest.fail(f"실데이터가 없다({REAL_ROOT}) — 이 대조는 skip하지 않는다")
    from trainer.pose_lab import csv_source

    if "source" not in _REAL_SOURCE:
        _REAL_SOURCE["source"] = csv_source.CsvPassSource(REAL_ROOT)
    source = _REAL_SOURCE["source"]
    params = motion_key.KeyParams()
    picks = [("김경호", "2", 0), ("김경호", "3", 10), ("전형기", "2", 3), ("전형기", "3", 5)]
    candidates = []
    for worker, level, index in picks:
        found, _total = source.passes(worker, level=level, limit=1)
        item, entry = next((i, e) for i, e in found if e.pass_index == index)
        ref = PassRef(worker=item.worker, date=item.date, level=item.level, session=item.session, pass_index=index)
        candidates.append(improvement_study.Candidate(ref=ref, worker=worker, key=source.key_of(entry, params)))
    return source, candidates


def test_screen_rule_matches_the_screen_on_real_files() -> None:
    """성질 1·2(실데이터): 연구의 쌍 규칙 N이 고른 기준은 화면(`/api/reference`)이 내는 후보 목록 안에 있고,
    후보 집합은 순서까지 같으며, N0은 화면이 처음 보이는 candidates[0]과 같다.
    기준은 전부 전형기·질의와 같은 레벨·질의 자신이 아니다 — 다른 레벨 짝이 하나라도 나오면 실패한다.
    """
    from fastapi.testclient import TestClient

    from trainer.pose_lab.app import create_app

    source, candidates = _real_queries()
    n = len(candidates)
    screen = improvement_study.select_pairs_screen(source, candidates, "전형기", n, 20260828)
    first = improvement_study.select_pairs_screen(source, candidates, "전형기", n, 20260828, choose="first")
    client = TestClient(create_app(source=source))
    assert sorted((p.query for p in screen), key=str) == sorted((c.ref for c in candidates), key=str)
    for pair, pair0 in zip(screen, first):
        q = pair.query
        body = client.get(
            "/api/reference",
            params={"worker": q.worker, "date": q.date, "level": q.level, "session": q.session, "pass_index": q.pass_index},
        ).json()
        assert body["status"] == "ok", body
        shown = tuple(improvement_study._row_ref(row) for row in body["candidates"])
        assert shown, f"실데이터 질의 {q}에 화면 후보가 없다 — 대조가 공허해진다"
        assert pair.candidates == shown, f"연구의 후보 집합이 화면과 다르다: {q}"
        assert pair.reference in shown
        assert pair.reference.level == q.level, f"다른 레벨 짝: 질의 L{q.level} ↔ 기준 L{pair.reference.level}"
        assert all(ref.level == q.level for ref in pair.candidates), f"후보에 다른 레벨이 섞였다: {q}"
        assert pair.reference.worker == service.EXPERT_WORKER
        assert pair.reference != q
        assert q not in pair.candidates
        assert pair0.reference == shown[0], f"N0이 화면의 candidates[0]과 다르다: {q}"
        assert pair.reference_granularity == body["match_granularity"]
        assert pair.truncated is False and body["truncated"] is False


def test_random_choice_is_independent_of_the_response_order() -> None:
    """성질 6(사전등록 2026-09-23 §7-6): `choose="random"`은 후보 목록의 **순서**와 무관하다.

    `select_pairs_screen`이 뽑기 전에 후보를 `(date, session, pass_index)`로 다시 세우기
    때문이다. 이것이 귀무 8회(§3.3)를 정렬 도입 **전** 코드에서 돌려도 되는 근거다.
    깨지면 R2의 귀무 분포가 P1과 같은 코드 위에서 잰 값이 아니게 된다(F6).
    """
    candidates = _synthetic_candidates()
    forward = _fake_candidates_fn({})

    def backward(source, ref, params, samples, limit):
        found = forward(source, ref, params, samples, limit)
        return {**found, "candidates": list(reversed(found["candidates"]))}

    a = improvement_study.select_pairs_screen(None, candidates, "전형기", 24, 7, candidates_fn=forward)
    b = improvement_study.select_pairs_screen(None, candidates, "전형기", 24, 7, candidates_fn=backward)
    assert [p.reference for p in a] == [p.reference for p in b], (
        "응답 순서를 뒤집었더니 --choose random의 기준이 바뀌었다 — 귀무 분포를 정렬 코드 위에서 다시 재야 한다"
    )
    # 공허하지 않다는 확인: 같은 대역에서 choose="first"는 순서를 그대로 탄다.
    a0 = improvement_study.select_pairs_screen(None, candidates, "전형기", 24, 7, choose="first", candidates_fn=forward)
    b0 = improvement_study.select_pairs_screen(None, candidates, "전형기", 24, 7, choose="first", candidates_fn=backward)
    assert [p.reference for p in a0] != [p.reference for p in b0], (
        "대역이 순서를 실제로 뒤집지 못했다 — 위 단언이 공허하다"
    )
