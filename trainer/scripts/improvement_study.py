#!/usr/bin/env python3
# pyright: reportAny=false, reportUnusedCallResult=false
"""70% 목표(goal_year2)에 답하는 표를 낸다.

이 모듈의 산출물은 기능이 아니라 숫자와 결론이다. "한 스트로크를 전문가와 비교해
자세 유사도를 재고 교정 피드백을 낸다 — 그리고 교정 적용 후 자세 유사도 개선율이
70%에 도달해야 한다"는 계약 문구를 실데이터로 판정한다.

## same과 split

`service.compare`는 개선율을 두 잣대로 낸다.

- **same**: 교정을 고른 분포로 다시 채점한다. 교정된 관절이 정의상 그 분포의 평균에
  가까워지므로 향상이 **자기충족으로 보장**된다. 이것은 결과가 아니라 상한이다.
- **split**: 날짜를 반으로 갈라 A(교정 선택)와 B(채점)를 분리하고, 질의 자신의 날짜는
  A/B 양쪽에서 뺀다. 자기충족이 없는 정직한 숫자다. **70% 판정은 split으로만 낸다.**

## 표집 규칙 (숫자를 보기 전에 고정한다)

1. `trainer.pose_lab.csv_source.CsvPassSource`로 `--root` 아래 전수(모든 작업자,
   레벨 1~10, 학습모드만)를 색인한다 — 패스마다 우8(`motion_key.KeyParams()` 기본
   격자) 키를 `CsvPassSource.key_of`로 뽑는다. 이 값이 `service.compare`가 내부에서
   같은 기본 params로 다시 계산하는 키와 같아야 쌍이 "동작 키 불일치"로 버려지지
   않는다(둘 다 `stroke_frame.DEFAULT_SAMPLES` 리샘플 위에서 나온다).
2. `random.Random(--seed)`로 전체 후보를 셔플하고 앞 `--pairs`개를 질의로 뽑는다 —
   **비복원**(같은 패스가 질의로 두 번 뽑히지 않는다).
3. 질의마다, 같은 우8 키를 가진 **레퍼런스 아티팩트의 작업자(전형기)** 소속 후보
   중에서(질의 자신과 동일한 패스는 제외) `rng.choice`로 레퍼런스 패스 하나를 고른다.
   후보가 없으면 그 쌍은 `status="no_expert_reference"`로 기록하고 버리지 않는다 —
   분모에서 조용히 빼면 "레퍼런스가 없어 재지 못한 몫"이 숨는다.
4. `--pairs`(기본 200)와 `--seed`(기본 20260828)는 CLI 인자로 코드에 먼저 고정한다.
   결과를 본 뒤 유리한 절단·부분집합·문턱을 고르지 않는다.

## None은 0이 아니다

계산되지 않은 `split`(예: `insufficient_dates`/`insufficient_samples`)은 분포에서
제외한다. 0으로 채우면 "교정이 악화시켰다"로 잘못 읽힌다. `summarize()`가 이 규칙을
지킨다 — `n`은 실제로 값이 있는 표본 수이지 시도한 쌍 수가 아니다.

## 사전등록 상수 (건드리지 않는다)

`CORRECTION_SIGMA_THRESHOLD`(2.0), `SIGMA_FLOOR_M`(0.02), `MAX_SIGMA`(6.0),
`GATE_RATIO`(0.5), `DEFAULT_ERROR_M`(0.144), `N_MIN`(20), `PAIR_SIGMA_FACTOR`(√2)는
전부 `measured_score.py`/`service.py`/`spread_reference.py`가 단일 출처다. 이 모듈은
그것들을 읽기만 하고 재정의하지 않는다. 반쪽 두께 계층화(§half_thickness)의 문턱도
새로 고르지 않고 이미 사전등록된 `N_MIN`(20)을 그대로 재사용한다.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import math
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

motion_key = importlib.import_module("motion_key")

from trainer.pose_lab import csv_source as csv_source_module  # noqa: E402
from trainer.pose_lab import service  # noqa: E402
from trainer.pose_lab.service import PassRef  # noqa: E402
from trainer.pose_lab.spread_reference import N_MIN, SpreadReference  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_PAIRS: Final = 200
DEFAULT_SEED: Final = 20260828
THRESHOLD: Final = 0.70
# 반쪽 두께 계층화 문턱. 새로 고르지 않는다 — build_spread_reference의 N_MIN을 그대로 쓴다.
THIN_HALF_THRESHOLD: Final = N_MIN


# --- Step 1의 집계 함수 --------------------------------------------------------


def _percentile(sorted_values: Sequence[float], fraction: float) -> float:
    """선형보간 백분위(넘파이 기본 'linear' 방식과 같다). `median`도 이 함수로 낸다."""
    if len(sorted_values) == 1:
        return sorted_values[0]
    index = fraction * (len(sorted_values) - 1)
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return sorted_values[int(index)]
    frac = index - lower
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * frac


def summarize(values: Sequence[float | None], threshold: float) -> dict[str, float | int | None]:
    """분포 요약: n·중앙값·사분위·문턱 이상 비율. `None`은 표본에서 뺀다(0으로 넣지 않는다)."""
    present = sorted(value for value in values if value is not None)
    n = len(present)
    if n == 0:
        return {"n": 0, "median": None, "q1": None, "q3": None, "at_or_above": None}
    return {
        "n": n,
        "median": _percentile(present, 0.5),
        "q1": _percentile(present, 0.25),
        "q3": _percentile(present, 0.75),
        "at_or_above": sum(1 for value in present if value >= threshold) / n,
    }


# --- 표집 ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Candidate:
    ref: PassRef
    worker: str
    key: str


def index_corpus(root: Path, rows_cache: int = 8) -> tuple[csv_source_module.CsvPassSource, list[Candidate]]:
    """전수 색인. 우8 키(`compare()`의 기본 params와 같은 격자)로 후보를 만든다.

    `CsvPassSource.index_of`는 세션당 한 번만 CSV를 파싱하고, 색인 항목은 행이 아니라
    구간·키 재료만 들고 있는다 — 그래서 479세션 전수를 훑어도 메모리에 원본 행이
    누적되지 않는다.
    """
    source = csv_source_module.CsvPassSource(root, rows_cache=rows_cache)
    params = motion_key.KeyParams()
    candidates: list[Candidate] = []
    for item in source.sessions():
        for entry in source.index_of(item):
            ref = PassRef(
                worker=item.worker, date=item.date, level=item.level,
                session=item.session, pass_index=entry.pass_index,
            )
            candidates.append(Candidate(ref=ref, worker=item.worker, key=source.key_of(entry, params)))
    return source, candidates


def _candidate_to_json(candidate: Candidate) -> dict[str, Any]:
    ref = candidate.ref
    return {
        "worker": ref.worker, "date": ref.date, "level": ref.level,
        "session": ref.session, "pass_index": ref.pass_index,
        "cand_worker": candidate.worker, "key": candidate.key,
    }


def _candidate_from_json(payload: Mapping[str, Any]) -> Candidate:
    ref = PassRef(
        worker=str(payload["worker"]), date=str(payload["date"]), level=str(payload["level"]),
        session=str(payload["session"]), pass_index=int(payload["pass_index"]),
    )
    return Candidate(ref=ref, worker=str(payload["cand_worker"]), key=str(payload["key"]))


def index_corpus_cached(
    root: Path, cache_path: Path | None, rows_cache: int = 8
) -> tuple[csv_source_module.CsvPassSource, list[Candidate]]:
    """`index_corpus`의 결과를 파일로 캐시한다. 표본 규칙을 바꾸지 않는다 — 전수 색인은

    입력(루트 CSV 트리)이 고정이면 항상 같은 후보 목록을 내는 순수 함수이므로, 그
    출력을 디스크에 굳혀 재사용해도 어떤 패스가 후보가 되는지·어떤 키를 갖는지는
    전혀 달라지지 않는다. 느린 것은 479개 세션 CSV(약 15.5GB)의 1회 파싱뿐이고,
    그 결과(패스 구간·모션 키)만 캐시한다 — `select_pairs`의 seed·셔플·선정 로직은
    캐시 유무와 무관하게 그대로 돈다.
    """
    if cache_path is not None and cache_path.is_file():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        if payload.get("root") == str(root):
            candidates = [_candidate_from_json(item) for item in payload["candidates"]]
            source = csv_source_module.CsvPassSource(root, rows_cache=rows_cache)
            logger.info("loaded %d cached candidates from %s", len(candidates), cache_path)
            return source, candidates
        logger.warning("index cache %s was built for a different root; rebuilding", cache_path)
    source, candidates = index_corpus(root, rows_cache=rows_cache)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(
                {"root": str(root), "candidates": [_candidate_to_json(c) for c in candidates]},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        logger.info("wrote %d candidates to index cache %s", len(candidates), cache_path)
    return source, candidates


@dataclass(frozen=True, slots=True)
class Pair:
    query: PassRef
    reference: PassRef | None
    key: str


def select_pairs(
    candidates: Sequence[Candidate], expert_worker: str, pairs: int, seed: int
) -> list[Pair]:
    """모듈 독스트링 "표집 규칙"을 그대로 구현한다. 결과를 본 뒤 바꾸지 않는다."""
    rng = random.Random(seed)
    shuffled = list(candidates)
    rng.shuffle(shuffled)
    queries = shuffled[: max(0, pairs)]

    by_key_expert: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        if candidate.worker == expert_worker:
            by_key_expert[candidate.key].append(candidate)

    result: list[Pair] = []
    for query in queries:
        options = [c for c in by_key_expert.get(query.key, []) if c.ref != query.ref]
        reference = rng.choice(options).ref if options else None
        result.append(Pair(query=query.ref, reference=reference, key=query.key))
    return result


# --- 실행 --------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Outcome:
    query: PassRef
    reference: PassRef | None
    key: str
    status: str  # "ok" | "no_expert_reference" | "unavailable"
    reason: str | None
    same: float | None
    split: float | None
    split_reason: str | None
    granularity: str | None
    reference_n: int | None
    split_n_a: int | None
    split_n_b: int | None
    gate: str | None
    gate_reason: str | None
    normalization: str | None
    fired: bool | None


def run_pair(
    source: csv_source_module.CsvPassSource, reference_spreads: SpreadReference, pair: Pair
) -> Outcome:
    if pair.reference is None:
        return Outcome(
            pair.query, None, pair.key, "no_expert_reference", "no_same_key_expert_pass",
            None, None, None, None, None, None, None, None, None, None, None,
        )
    body = service.compare(
        source, pair.query, pair.reference, reference_spreads=reference_spreads
    )
    if body.get("status") != "ok":
        return Outcome(
            pair.query, pair.reference, pair.key, "unavailable", str(body.get("reason")),
            None, None, None, None, None, None, None, None, None, None, None,
        )
    measured = body["measured"]
    improvement = measured["improvement"]
    fired = bool(measured.get("finding"))
    return Outcome(
        query=pair.query, reference=pair.reference, key=pair.key, status="ok", reason=None,
        same=improvement.get("same"), split=improvement.get("split"),
        split_reason=measured.get("split_reason"), granularity=measured.get("granularity"),
        reference_n=measured.get("reference_n"), split_n_a=measured.get("split_n_a"),
        split_n_b=measured.get("split_n_b"), gate=measured.get("gate"),
        gate_reason=measured.get("gate_reason"), normalization=measured.get("normalization"),
        fired=fired,
    )


def run_study(
    root: Path,
    reference_path: Path,
    pairs: int,
    seed: int,
    progress: bool = False,
    index_cache: Path | None = None,
) -> tuple[list[Outcome], str]:
    reference_spreads = SpreadReference.load(reference_path)
    expert_worker = reference_spreads.worker
    source, candidates = index_corpus_cached(root, index_cache)
    logger.info("indexed %d candidate passes across %d sessions", len(candidates), len(source.sessions()))
    selected = select_pairs(candidates, expert_worker, pairs, seed)
    outcomes: list[Outcome] = []
    for index, pair in enumerate(selected):
        outcomes.append(run_pair(source, reference_spreads, pair))
        if progress and (index + 1) % 20 == 0:
            print(f"  {index + 1}/{len(selected)}", file=sys.stderr, flush=True)
    return outcomes, expert_worker


# --- 집계·보고 ----------------------------------------------------------------


def _label(value: object) -> str:
    return "none" if value is None else str(value)


def build_report(
    outcomes: Sequence[Outcome], pairs: int, seed: int, expert_worker: str, root: Path, reference_path: Path
) -> dict[str, Any]:
    ok = [item for item in outcomes if item.status == "ok"]
    status_counts = Counter(item.status for item in outcomes)

    granularity_counts = Counter(item.granularity for item in ok)
    per_layer = {
        _label(layer): {
            "n": count,
            "same": summarize([item.same for item in ok if item.granularity == layer], THRESHOLD),
            "split": summarize([item.split for item in ok if item.granularity == layer], THRESHOLD),
        }
        for layer, count in granularity_counts.items()
    }

    split_reason_counts = Counter(item.split_reason for item in ok)

    fired = [item for item in ok if item.fired]
    not_fired = [item for item in ok if item.fired is False]

    thickness_known = [
        item for item in ok if item.split_n_a is not None and item.split_n_b is not None
    ]
    thin = [item for item in thickness_known if min(item.split_n_a, item.split_n_b) < THIN_HALF_THRESHOLD]
    thick = [item for item in thickness_known if min(item.split_n_a, item.split_n_b) >= THIN_HALF_THRESHOLD]

    gate_counts = Counter(item.gate for item in ok)
    gate_by_layer: dict[str, Counter] = defaultdict(Counter)
    for item in ok:
        gate_by_layer[_label(item.granularity)][_label(item.gate)] += 1
    gate_determined = gate_counts.get("pass", 0) + gate_counts.get("fail", 0)

    return {
        "selection_rule": {
            "root": str(root),
            "reference_artifact": str(reference_path),
            "expert_worker": expert_worker,
            "pairs_requested": pairs,
            "seed": seed,
            "sampling": (
                "질의: 전수 후보(모든 작업자, 우8 키 부착)를 seed로 셔플해 앞 N개를 "
                "비복원으로 뽑는다. 레퍼런스: 질의와 같은 우8 키를 가진 전형기(레퍼런스 "
                "아티팩트의 worker) 후보 중 질의 자신을 뺀 나머지에서 rng.choice로 하나. "
                "후보가 없으면 status=no_expert_reference로 기록하고 버리지 않는다."
            ),
        },
        "status_counts": {_label(key): value for key, value in status_counts.items()},
        "n_attempted": len(outcomes),
        "n_ok": len(ok),
        "same": summarize([item.same for item in ok], THRESHOLD),
        "split": summarize([item.split for item in ok], THRESHOLD),
        "granularity_counts": {_label(key): value for key, value in granularity_counts.items()},
        "per_layer": per_layer,
        "split_reason_counts": {_label(key): value for key, value in split_reason_counts.items()},
        "correction_fire_rate": {
            "fired": len(fired),
            "not_fired": len(not_fired),
            "share_fired": (len(fired) / len(ok)) if ok else None,
            "same_fired": summarize([item.same for item in fired], THRESHOLD),
            "same_not_fired": summarize([item.same for item in not_fired], THRESHOLD),
            "split_fired": summarize([item.split for item in fired], THRESHOLD),
            "split_not_fired": summarize([item.split for item in not_fired], THRESHOLD),
        },
        "half_thickness": {
            "thin_threshold_n_min": THIN_HALF_THRESHOLD,
            "n_with_known_thickness": len(thickness_known),
            "thin_n": len(thin),
            "thick_n": len(thick),
            "split_thin": summarize([item.split for item in thin], THRESHOLD),
            "split_thick": summarize([item.split for item in thick], THRESHOLD),
        },
        "gate": {
            "counts": {_label(key): value for key, value in gate_counts.items()},
            "pass_rate_of_determined": (
                gate_counts.get("pass", 0) / gate_determined if gate_determined else None
            ),
            "by_layer": {layer: dict(counter) for layer, counter in gate_by_layer.items()},
        },
    }


def _fmt(value: float | int | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def render_summary(report: Mapping[str, Any]) -> str:
    lines: list[str] = []
    rule = report["selection_rule"]
    lines.append(
        f"쌍 {rule['pairs_requested']}개 요청, seed={rule['seed']}, "
        f"전형기={rule['expert_worker']}"
    )
    lines.append(f"상태: {report['status_counts']}")
    for name in ("same", "split"):
        entry = report[name]
        lines.append(
            f"{name}: n={entry['n']} median={_fmt(entry['median'])} "
            f"q1={_fmt(entry['q1'])} q3={_fmt(entry['q3'])} "
            f"≥{THRESHOLD:.0%}={_fmt(entry['at_or_above'])}"
        )
    lines.append(f"입도별 n: {report['granularity_counts']}")
    lines.append(f"split_reason: {report['split_reason_counts']}")
    fire = report["correction_fire_rate"]
    lines.append(f"교정 발동률: {fire['share_fired']} (발동 {fire['fired']} / 미발동 {fire['not_fired']})")
    gate = report["gate"]
    lines.append(f"REQ-19 게이트: {gate['counts']} pass_rate={gate['pass_rate_of_determined']}")
    return "\n".join(lines)


# --- CLI ---------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=DEFAULT_PAIRS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--index-cache", type=Path, default=None,
        help="전수 색인(패스 구간·모션 키) 결과를 이 경로에 캐시하고 재사용한다. "
        "표본 규칙을 바꾸지 않는다 — 느린 CSV 파싱만 건너뛴다.",
    )
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--log-level", default="WARNING", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")
    if not args.root.is_dir():
        logger.error("root not found: %s", args.root)
        return 1
    if not args.reference.is_file():
        logger.error("reference artifact not found: %s", args.reference)
        return 1

    outcomes, expert_worker = run_study(
        args.root, args.reference, args.pairs, args.seed, args.progress, args.index_cache
    )
    report = build_report(outcomes, args.pairs, args.seed, expert_worker, args.root, args.reference)
    print(render_summary(report))

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
