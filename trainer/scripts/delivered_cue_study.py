#!/usr/bin/env python3
# pyright: reportAny=false, reportUnusedCallResult=false
"""전달되는 안내 한 건만 따랐을 때의 자세 유사도 개선 — **전수**(색인된 모든 패스).

사전등록: `docs/measurements/2026-10-01-delivered-cue-improvement-prereg.md`

작업자가 듣는 것은 σ 1위 안내 한 건이다(`measured_score.DELIVERED_CUES`). `service.compare`는
993e30e부터 `score_after`·`improvement`를 그 한 건으로, `score_after_all`·`improvement_all`을
발부된 축 전부(옛 정의, 91.58% 계열)로 낸다. 이 스크립트는 두 정의를 **같은 쌍**에서 나란히 잰다.

- 질의: `improvement_study.index_corpus_cached`가 내는 후보 **전부**(표집하지 않는다).
- 기준: `improvement_study.select_pairs_screen(choose="first")` — 화면이 처음 보이는 기준 패스
  (C1의 `screen_first`와 같은 규칙). 쌍 규칙을 여기서 다시 쓰지 않는다.
- 병렬: 세션 단위로 샤드를 나눈다(`shard_of`). 같은 세션의 패스가 한 프로세스에 모여 CSV
  캐시가 맞는다. 샤드마다 JSONL로 한 줄씩 쓰고, 다시 돌리면 이미 쓴 질의는 건너뛴다.
- 집계(`aggregate`)는 순수 함수다 — None은 표본에서 빼고 0으로 채우지 않는다.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import logging
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Final, Iterable, Mapping, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

measured_score = importlib.import_module("measured_score")
improvement_study = importlib.import_module("improvement_study")

logger = logging.getLogger(__name__)

THRESHOLD: Final = improvement_study.THRESHOLD  # 0.70, goal_year2
# (보고 이름, 레코드 키 접미사, 교정 후 점수 키). 1건이 주 판정, all_issued는 비교용.
DEFINITIONS: Final[tuple[tuple[str, str, str], ...]] = (
    ("delivered", "", "score_after"),
    ("all_issued", "_all", "score_after_all"),
)


def shard_of(session_key: str, shards: int) -> int:
    """세션 → 샤드. 파이썬 `hash`는 실행마다 바뀌므로 쓰지 않는다."""
    digest = hashlib.sha256(session_key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % shards


def _ref_key(ref: Any) -> str:
    return f"{ref.worker}|{ref.date}|{ref.level}|{ref.session}|{ref.pass_index}"


def record_of(query: Any, reference: Any, body: Mapping[str, Any] | None, reason: str | None) -> dict[str, Any]:
    """한 쌍의 원자료. 집계에 필요한 것과 추적에 필요한 것만."""
    out: dict[str, Any] = {
        "query": _ref_key(query),
        "worker": query.worker,
        "level": query.level,
        "reference": _ref_key(reference) if reference is not None else None,
    }
    if body is None:
        out.update({"status": "no_expert_reference", "reason": reason})
        return out
    if body.get("status") != "ok":
        out.update({"status": "unavailable", "reason": str(body.get("reason"))})
        return out
    measured = body["measured"]
    axes = measured.get("axes") or []
    first = axes[0] if axes else None
    improvement = measured.get("improvement") or {}
    improvement_all = measured.get("improvement_all") or {}
    out.update({
        "status": "ok",
        "score": measured.get("score"),
        "score_after": measured.get("score_after"),
        "score_after_all": measured.get("score_after_all"),
        "same": improvement.get("same"),
        "split": improvement.get("split"),
        "same_all": improvement_all.get("same"),
        "split_all": improvement_all.get("split"),
        "split_reason": measured.get("split_reason"),
        "issued": len(axes),
        "delivered_point": first.get("point") if isinstance(first, Mapping) else None,
        "delivered_axis": first.get("axis") if isinstance(first, Mapping) else None,
    })
    return out


# --- 집계 (순수) --------------------------------------------------------------


def describe(values: Iterable[float | None], threshold: float | None = THRESHOLD) -> dict[str, Any]:
    """평균·중앙값·사분위·문턱 이상 비율·0 미만(악화) 비율. None은 빼고 0으로 채우지 않는다."""
    present = sorted(float(v) for v in values if v is not None and math.isfinite(float(v)))
    n = len(present)
    if n == 0:
        return {"n": 0, "mean": None, "median": None, "q1": None, "q3": None,
                "at_or_above": None, "below_zero": None}
    pct = improvement_study._percentile
    return {
        "n": n,
        "mean": sum(present) / n,
        "median": pct(present, 0.5),
        "q1": pct(present, 0.25),
        "q3": pct(present, 0.75),
        "at_or_above": (sum(1 for v in present if v >= threshold) / n) if threshold is not None else None,
        "below_zero": sum(1 for v in present if v < 0.0) / n,
    }


def _rise(record: Mapping[str, Any], after_key: str) -> float | None:
    before, after = record.get("score"), record.get(after_key)
    if before is None or after is None:
        return None
    return float(after) - float(before)


def _block(population: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, suffix, after_key in DEFINITIONS:
        out[name] = {
            "split": describe(r.get("split" + suffix) for r in population),
            "same": describe(r.get("same" + suffix) for r in population),
            # 화면의 "유사도가 오른 폭"(전체 기준 S1 − S0).
            "rise": describe((_rise(r, after_key) for r in population), threshold=None),
        }
    return out


def aggregate(records: Sequence[Mapping[str, Any]], expert_worker: str) -> dict[str, Any]:
    ok = [r for r in records if r.get("status") == "ok"]
    status: dict[str, int] = defaultdict(int)
    for r in records:
        status[str(r.get("status"))] += 1
    issued = [r for r in ok if (r.get("issued") or 0) > 0]

    by_worker: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for r in issued:
        by_worker[str(r.get("worker"))].append(r)
    delivered_point: dict[str, int] = defaultdict(int)
    for r in issued:
        delivered_point[f"{r.get('delivered_point')}/{r.get('delivered_axis')}"] += 1

    return {
        "n_records": len(records),
        "status_counts": dict(status),
        "n_ok": len(ok),
        "n_issued": len(issued),
        "score_before": describe((r.get("score") for r in ok), threshold=None),
        # 주 판정 모집단: 안내가 하나라도 나간 쌍. 안 나간 쌍은 S1 = S0이라 "고칠 것 없음"이지
        # 개선 실패가 아니다 — 섞으면 평균이 0 쪽으로 끌린다. 섞은 값도 `all_ok`로 같이 낸다.
        "issued_only": _block(issued),
        "all_ok": _block(ok),
        "excluding_expert_self": _block([r for r in issued if r.get("worker") != expert_worker]),
        "delivered_point_counts": dict(sorted(delivered_point.items(), key=lambda kv: -kv[1])),
        "per_worker": {
            worker: {
                "n": len(rows),
                "delivered_split": describe(r.get("split") for r in rows),
                "all_issued_split": describe(r.get("split_all") for r in rows),
            }
            for worker, rows in sorted(by_worker.items())
        },
    }


# --- CSV ------------------------------------------------------------------
#
# 엑셀에서 한글이 깨지지 않게 UTF-8 BOM(`utf-8-sig`)으로 쓴다. 숫자는 반올림하지 않는다 —
# 표시 자릿수는 읽는 쪽이 정한다. 빈 값(None)은 빈 칸이다(0이 아니다).

PAIR_COLUMNS: Final[tuple[tuple[str, str], ...]] = (
    ("query", "질의 패스(작업자|날짜|레벨|세션|패스)"),
    ("worker", "작업자"),
    ("level", "레벨"),
    ("reference", "기준 패스"),
    ("status", "상태"),
    ("reason", "사유"),
    ("issued", "발부 안내 수"),
    ("delivered_point", "전달 안내 부위"),
    ("delivered_axis", "전달 안내 축"),
    ("score", "교정 전 유사도"),
    ("score_after", "안내 1건 적용 후 유사도"),
    ("score_after_all", "발부 전부 적용 후 유사도"),
    ("split", "1건 개선율(split)"),
    ("same", "1건 개선율(same)"),
    ("split_all", "전부 개선율(split)"),
    ("same_all", "전부 개선율(same)"),
    ("split_reason", "split 없음 사유"),
)
POINT_NAMES: Final = {"right": "오른손", "head": "머리", "left": "왼손"}
AXIS_NAMES: Final = {"wall": "벽 쪽", "vertical": "위아래", "lateral": "좌우(진행 방향)"}
STAT_COLUMNS: Final[tuple[tuple[str, str], ...]] = (
    ("n", "n"), ("mean", "평균"), ("median", "중앙값"), ("q1", "1사분위"), ("q3", "3사분위"),
    ("at_or_above", "0.70 이상 비율"), ("below_zero", "0 미만 비율"),
)
POPULATION_NAMES: Final = {
    "issued_only": "안내가 나간 쌍(주 판정)",
    "all_ok": "채점된 쌍 전부",
    "excluding_expert_self": "안내가 나간 쌍 중 전형기 자신 제외",
}
DEFINITION_NAMES: Final = {"delivered": "전달 안내 1건", "all_issued": "발부 전부"}
METRIC_NAMES: Final = {"split": "개선율(split)", "same": "개선율(same)", "rise": "유사도 상승폭"}
BEFORE_BANDS: Final[tuple[float, ...]] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
# 과제 평가 방법(사용자 제공 2026-10-01): 정확도(%) = (S_corrected − S0)/(1 − S0) × 100 의 **평균**,
# 10회 이상 적용, 조정하는 주요 관절 값은 6개 이하. 2차년도 70%, 3차년도 90% 이상.
TARGETS: Final[tuple[tuple[str, float], ...]] = (("2차년도 70%", 0.70), ("3차년도 90%", 0.90))
MIN_TRIALS: Final = 10
MAX_ADJUSTED_VALUES: Final = 6


def _cell(value: Any) -> Any:
    return "" if value is None else value


def _write(path: Path, header: Sequence[str], rows: Iterable[Sequence[Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row in rows:
            writer.writerow([_cell(value) for value in row])


def _group_means(records: Sequence[Mapping[str, Any]], key: Any) -> list[tuple[Any, dict[str, Any]]]:
    groups: dict[Any, list[Mapping[str, Any]]] = defaultdict(list)
    for r in records:
        groups[key(r)].append(r)
    out = []
    for name, rows in groups.items():
        out.append((name, {
            "n": len(rows),
            "split": describe(r.get("split") for r in rows),
            "split_all": describe(r.get("split_all") for r in rows),
            "rise": describe((_rise(r, "score_after") for r in rows), threshold=None),
            "rise_all": describe((_rise(r, "score_after_all") for r in rows), threshold=None),
        }))
    return out


GROUP_HEADER: Final = (
    "쌍 수", "1건 개선율(split) 평균", "1건 개선율(split) 중앙값", "1건 0.70 이상 비율",
    "전부 개선율(split) 평균", "전부 0.70 이상 비율", "1건 유사도 상승폭 평균", "전부 유사도 상승폭 평균",
)


def _group_row(stats: Mapping[str, Any]) -> list[Any]:
    return [
        stats["n"], stats["split"]["mean"], stats["split"]["median"], stats["split"]["at_or_above"],
        stats["split_all"]["mean"], stats["split_all"]["at_or_above"],
        stats["rise"]["mean"], stats["rise_all"]["mean"],
    ]


def _band_of(score: Any) -> str | None:
    if score is None:
        return None
    value = float(score)
    for lo, hi in zip(BEFORE_BANDS, BEFORE_BANDS[1:]):
        if lo <= value < hi or (hi == BEFORE_BANDS[-1] and value == hi):
            return f"{lo:.1f}–{hi:.1f}"
    return None


def write_csvs(records: Sequence[Mapping[str, Any]], report: Mapping[str, Any], out_dir: Path) -> list[Path]:
    """쌍별 원자료 1개와 요약표 4개를 쓴다. 쓴 경로를 돌려준다."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    path = out_dir / "쌍별_원자료.csv"
    korean = {"delivered_point": POINT_NAMES, "delivered_axis": AXIS_NAMES}

    def pair_row(r: Mapping[str, Any]) -> list[Any]:
        return [korean[key].get(r.get(key), r.get(key)) if key in korean else r.get(key)
                for key, _label in PAIR_COLUMNS]

    _write(path, [label for _key, label in PAIR_COLUMNS], (pair_row(r) for r in records))
    written.append(path)

    path = out_dir / "정확도_평가방법.csv"
    issued_ok = [r for r in records if r.get("status") == "ok" and (r.get("issued") or 0) > 0]
    adjusted = {
        "delivered": [min(int(r.get("issued") or 0), 1) for r in issued_ok],
        "all_issued": [int(r.get("issued") or 0) for r in issued_ok],
    }
    rows = []
    for definition, def_name in DEFINITION_NAMES.items():
        counts = adjusted[definition]
        for metric, metric_name in (("same", "목표 자세로 교정·채점(same)"), ("split", "교정·채점 기준 분리(split)")):
            stats = report["issued_only"][definition][metric]
            mean = stats.get("mean")
            rows.append([
                def_name, metric_name, max(counts) if counts else None,
                (max(counts) <= MAX_ADJUSTED_VALUES) if counts else None,
                stats.get("n"), (stats.get("n") or 0) >= MIN_TRIALS,
                None if mean is None else mean * 100.0,
                *[(None if mean is None else mean >= target) for _name, target in TARGETS],
            ])
    _write(path, ["적용한 교정", "채점 방식", "조정한 값 최대 개수", "6개 이하",
                  "적용 횟수", "10회 이상", "정확도 평균(%)", *(f"{name} 달성" for name, _t in TARGETS)], rows)
    written.append(path)

    path = out_dir / "요약.csv"
    rows = []
    for population, pop_name in POPULATION_NAMES.items():
        for definition, def_name in DEFINITION_NAMES.items():
            for metric, metric_name in METRIC_NAMES.items():
                stats = report[population][definition][metric]
                rows.append([pop_name, def_name, metric_name, *(stats.get(key) for key, _ in STAT_COLUMNS)])
    _write(path, ["모집단", "적용한 안내", "지표", *(label for _key, label in STAT_COLUMNS)], rows)
    written.append(path)

    issued = [r for r in records if r.get("status") == "ok" and (r.get("issued") or 0) > 0]

    path = out_dir / "전달_안내별.csv"
    groups = _group_means(issued, lambda r: (r.get("delivered_point"), r.get("delivered_axis")))
    groups.sort(key=lambda item: -item[1]["n"])
    _write(path, ["부위", "축", *GROUP_HEADER],
           ([POINT_NAMES.get(p, p), AXIS_NAMES.get(a, a), *_group_row(st)] for (p, a), st in groups))
    written.append(path)

    path = out_dir / "작업자별.csv"
    groups = sorted(_group_means(issued, lambda r: r.get("worker")), key=lambda item: str(item[0]))
    _write(path, ["작업자", *GROUP_HEADER], ([worker, *_group_row(st)] for worker, st in groups))
    written.append(path)

    path = out_dir / "교정전_유사도_구간별.csv"
    groups = sorted(
        (item for item in _group_means(issued, lambda r: _band_of(r.get("score"))) if item[0] is not None),
        key=lambda item: str(item[0]),
    )
    _write(path, ["교정 전 유사도 구간", *GROUP_HEADER], ([band, *_group_row(st)] for band, st in groups))
    written.append(path)
    return written


# --- 실행 -------------------------------------------------------------------


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            # 중단으로 잘린 마지막 줄. 그 질의는 다음 실행에서 다시 돈다.
            continue
    return rows


def run_shard(args: argparse.Namespace) -> int:
    from trainer.pose_lab import service
    from trainer.pose_lab.spread_reference import SpreadReference

    reference_spreads = SpreadReference.load(args.reference)
    improvement_study.assert_artifact_prefix(reference_spreads, args.reference)
    source, candidates = improvement_study.index_corpus_cached(args.root, args.index_cache)
    mine = [
        c for c in candidates
        if shard_of(f"{c.ref.worker}|{c.ref.date}|{c.ref.level}|{c.ref.session}", args.shards) == args.shard
    ]
    mine.sort(key=lambda c: (c.ref.worker, c.ref.date, c.ref.level, c.ref.session, c.ref.pass_index))
    if args.limit is not None:
        mine = mine[: args.limit]
    out_path = args.out_dir / f"shard_{args.shard:02d}_of_{args.shards:02d}.jsonl"
    done = {row["query"] for row in read_jsonl(out_path)}
    todo = [c for c in mine if _ref_key(c.ref) not in done]
    logger.warning("shard %d/%d: %d queries, %d done, %d to go", args.shard, args.shards, len(mine), len(done), len(todo))
    start = time.monotonic()
    with out_path.open("a", encoding="utf-8") as handle:
        for index, candidate in enumerate(todo, start=1):
            (pair,) = improvement_study.select_pairs_screen(
                source, [candidate], reference_spreads.worker, 1, 0, choose="first"
            )
            body: Mapping[str, Any] | None = None
            if pair.reference is not None:
                try:
                    body = service.compare(
                        source, pair.query, pair.reference,
                        reference_spreads=reference_spreads, include_view=False,
                    )
                except Exception as exc:  # 한 쌍의 실패가 샤드를 멈추지 않게 — 사유를 남긴다.
                    body = {"status": "error", "reason": f"{type(exc).__name__}: {exc}"}
            reason = pair.candidate_reason or ("no_screen_candidate" if pair.reference is None else None)
            handle.write(json.dumps(record_of(pair.query, pair.reference, body, reason), ensure_ascii=False) + "\n")
            handle.flush()
            if index % 200 == 0:
                rate = index / (time.monotonic() - start)
                logger.warning("shard %d: %d/%d (%.2f/s)", args.shard, index, len(todo), rate)
    logger.warning("shard %d: finished in %.0fs", args.shard, time.monotonic() - start)
    return 0


def run_aggregate(args: argparse.Namespace) -> int:
    from trainer.pose_lab.spread_reference import SpreadReference

    records: list[dict[str, Any]] = []
    for path in sorted(args.out_dir.glob("shard_*.jsonl")):
        records.extend(read_jsonl(path))
    expert = SpreadReference.load(args.reference).worker
    report = aggregate(records, expert)
    report["delivered_cues"] = measured_score.DELIVERED_CUES
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.report is not None:
        args.report.write_text(text, encoding="utf-8")
    if args.csv_dir is not None:
        for path in write_csvs(records, report, args.csv_dir):
            logger.warning("wrote %s", path)
    print(text)
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("shard", "aggregate"))
    parser.add_argument("--root", type=Path)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--index-cache", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None, help="샤드당 질의 상한(시간 재기용)")
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--csv-dir", type=Path, default=None, help="쌍별 원자료·요약표 CSV를 쓸 폴더")
    parser.add_argument("--log-level", default="WARNING")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "shard":
        return run_shard(args)
    return run_aggregate(args)


if __name__ == "__main__":
    raise SystemExit(main())
