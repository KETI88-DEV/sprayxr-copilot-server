#!/usr/bin/env python3
# pyright: reportAny=false, reportUnusedCallResult=false
"""3차년도 — 관절 값 6개 이하 교정의 정확도, 전수(색인된 모든 패스).

사전등록: `docs/year3/2026-10-01-six-value-correction-prereg.md`
산출물: `/output/year3/six_values/`(샤드 JSONL, 집계 JSON, CSV)

쌍 규칙·표본·병렬 방식은 2차년도 전수 측정(`trainer/scripts/delivered_cue_study.py`)과 같다 — 같은
64,098 질의, 같은 기준 패스(`screen_first`). 쌍마다 `service.compare(correction_policies=POLICIES)`를
한 번 부르고 정책별 same·split 개선율을 적는다.

실행(컨테이너, 저장소 루트):
    python3 -m trainer.year3.six_value_study shard --shards 20 --shard N ...
    python3 -m trainer.year3.six_value_study aggregate ... --csv-dir /output/year3/six_values/csv
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import logging
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SCRIPTS = _REPO_ROOT / "trainer" / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from trainer.year3 import correction_policies  # noqa: E402

dcs = importlib.import_module("delivered_cue_study")
improvement_study = importlib.import_module("improvement_study")

logger = logging.getLogger(__name__)

TARGETS: Final[tuple[tuple[str, float], ...]] = (("2차년도 70%", 0.70), ("3차년도 90%", 0.90))
MIN_TRIALS: Final = 10
MAX_VALUES: Final = correction_policies.MAX_VALUES
POLICY_NAMES: Final = {
    "P0_product": "P0 2차년도 제품 알고리즘(값 ≤4)",
    "P1_sigma_top6": "P1 σ 상위 6개",
    "P2_greedy6": "P2 유사도 탐욕 선택 6개(주 판정)",
    "P3_all_values": "P3 문턱 넘은 값 전부(6개 초과 가능, 참고)",
}
# P0 재현 확인 허용 오차. 같은 함수를 같은 순서로 부르므로 0이어야 하지만, 이동량을 더하는 순서가
# 달라질 수 있는 부동소수 오차만 허용한다.
REPRODUCTION_TOLERANCE: Final = 1e-9


def record_of(query: Any, reference: Any, body: Mapping[str, Any] | None, reason: str | None) -> dict[str, Any]:
    out: dict[str, Any] = {
        "query": dcs._ref_key(query), "worker": query.worker, "level": query.level,
        "reference": dcs._ref_key(reference) if reference is not None else None,
    }
    if body is None:
        out.update({"status": "no_expert_reference", "reason": reason})
        return out
    if body.get("status") != "ok":
        out.update({"status": "unavailable", "reason": str(body.get("reason"))})
        return out
    measured = body["measured"]
    out.update({
        "status": "ok",
        "score": measured.get("score"),
        "same_product_all": (measured.get("improvement_all") or {}).get("same"),
        "split_product_all": (measured.get("improvement_all") or {}).get("split"),
    })
    for name, result in (measured.get("policies") or {}).items():
        out[f"{name}.values"] = result.get("values")
        out[f"{name}.values_split"] = result.get("values_split")
        out[f"{name}.points"] = "+".join(result.get("points") or [])
        out[f"{name}.score_after"] = result.get("score_after")
        out[f"{name}.same"] = (result.get("improvement") or {}).get("same")
        out[f"{name}.split"] = (result.get("improvement") or {}).get("split")
    return out


def reproduction_mismatches(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """P0이 제품의 `improvement_all`과 다른 쌍 수. 0이 아니면 연결부가 다른 것을 재고 있다."""
    counts = {"same": 0, "split": 0}
    for r in records:
        if r.get("status") != "ok":
            continue
        for kind in ("same", "split"):
            a, b = r.get(f"P0_product.{kind}"), r.get(f"{kind}_product_all")
            if (a is None) != (b is None) or (a is not None and abs(float(a) - float(b)) > REPRODUCTION_TOLERANCE):
                counts[kind] += 1
    return counts


def aggregate(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ok = [r for r in records if r.get("status") == "ok"]
    status = Counter(str(r.get("status")) for r in records)
    policies: dict[str, Any] = {}
    for name in correction_policies.POLICIES:
        values = [r.get(f"{name}.values") for r in ok if r.get(f"{name}.values") is not None]
        values_split = [r.get(f"{name}.values_split") for r in ok if r.get(f"{name}.values_split") is not None]
        points = Counter()
        for r in ok:
            for point in str(r.get(f"{name}.points") or "").split("+"):
                if point:
                    points[point] += 1
        policies[name] = {
            "same": dcs.describe(r.get(f"{name}.same") for r in ok),
            "split": dcs.describe(r.get(f"{name}.split") for r in ok),
            "values_max": max([*values, *values_split]) if values or values_split else None,
            "values_histogram": dict(sorted(Counter(values).items())),
            "pairs_touching_point": dict(points),
        }
    return {
        "n_records": len(records),
        "status_counts": dict(status),
        "n_ok": len(ok),
        "reproduction_mismatches": reproduction_mismatches(records),
        "policies": policies,
        "primary": correction_policies.PRIMARY,
    }


def write_csvs(records: Sequence[Mapping[str, Any]], report: Mapping[str, Any], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    path = out_dir / "정확도_평가방법.csv"
    rows = []
    for name, label in POLICY_NAMES.items():
        block = report["policies"][name]
        values_max = block["values_max"]
        within = None if values_max is None else values_max <= MAX_VALUES
        for kind, kind_label in (("same", "목표 자세로 교정·채점(same)"), ("split", "교정·채점 기준 분리(split)")):
            stats = block[kind]
            mean = stats.get("mean")
            eligible = bool(within) and (stats.get("n") or 0) >= MIN_TRIALS
            rows.append([
                label, kind_label, values_max, within, stats.get("n"), (stats.get("n") or 0) >= MIN_TRIALS,
                None if mean is None else mean * 100.0, stats.get("median"),
                *[(None if mean is None or not eligible else mean >= target) for _n, target in TARGETS],
            ])
    dcs._write(path, ["교정 정책", "채점 방식", "조정한 값 최대 개수", "6개 이하", "적용 횟수", "10회 이상",
                      "정확도 평균(%)", "개선율 중앙값", *(f"{n} 달성" for n, _t in TARGETS)], rows)
    written.append(path)

    path = out_dir / "조정값수_분포.csv"
    counts = sorted({k for name in POLICY_NAMES for k in report["policies"][name]["values_histogram"]})
    dcs._write(path, ["교정 정책", *(f"값 {k}개" for k in counts)],
               ([POLICY_NAMES[name], *(report["policies"][name]["values_histogram"].get(k, 0) for k in counts)]
                for name in POLICY_NAMES))
    written.append(path)

    path = out_dir / "조정부위_사용.csv"
    point_names = dcs.POINT_NAMES
    dcs._write(path, ["교정 정책", *(point_names[p] for p in ("right", "head", "left"))],
               ([POLICY_NAMES[name], *(report["policies"][name]["pairs_touching_point"].get(p, 0)
                                       for p in ("right", "head", "left"))] for name in POLICY_NAMES))
    written.append(path)

    ok = [r for r in records if r.get("status") == "ok"]
    group_header = ["쌍 수", *(f"{name.split('_')[0]} 정확도 {kind}(%)" for name in POLICY_NAMES for kind in ("split", "same"))]

    def group_row(rows: Sequence[Mapping[str, Any]]) -> list[Any]:
        out: list[Any] = [len(rows)]
        for name in POLICY_NAMES:
            for kind in ("split", "same"):
                mean = dcs.describe(r.get(f"{name}.{kind}") for r in rows)["mean"]
                out.append(None if mean is None else mean * 100.0)
        return out

    by_worker: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for r in ok:
        by_worker[str(r.get("worker"))].append(r)
    path = out_dir / "작업자별.csv"
    dcs._write(path, ["작업자", *group_header], ([w, *group_row(rows)] for w, rows in sorted(by_worker.items())))
    written.append(path)

    by_band: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for r in ok:
        band = dcs._band_of(r.get("score"))
        if band is not None:
            by_band[band].append(r)
    path = out_dir / "교정전_유사도_구간별.csv"
    dcs._write(path, ["교정 전 유사도 구간", *group_header], ([b, *group_row(rows)] for b, rows in sorted(by_band.items())))
    written.append(path)

    path = out_dir / "쌍별_원자료.csv"
    base = [("query", "질의 패스(작업자|날짜|레벨|세션|패스)"), ("worker", "작업자"), ("level", "레벨"),
            ("reference", "기준 패스"), ("status", "상태"), ("reason", "사유"), ("score", "교정 전 유사도")]
    per_policy = [("values", "조정 값 수(same)"), ("values_split", "조정 값 수(split)"), ("points", "조정 부위"),
                  ("score_after", "교정 후 유사도"), ("same", "개선율(same)"), ("split", "개선율(split)")]
    columns = base + [(f"{name}.{key}", f"{name.split('_')[0]} {label}")
                      for name in POLICY_NAMES for key, label in per_policy]
    korean_points = lambda text: "+".join(point_names.get(p, p) for p in str(text).split("+") if p)  # noqa: E731
    dcs._write(path, [label for _k, label in columns],
               ([korean_points(r.get(k)) if k.endswith(".points") and r.get(k) else r.get(k) for k, _l in columns]
                for r in records))
    written.append(path)
    return written


def run_shard(args: argparse.Namespace) -> int:
    from trainer.pose_lab import service
    from trainer.pose_lab.spread_reference import SpreadReference

    reference_spreads = SpreadReference.load(args.reference)
    improvement_study.assert_artifact_prefix(reference_spreads, args.reference)
    source, candidates = improvement_study.index_corpus_cached(args.root, args.index_cache)
    mine = [c for c in candidates
            if dcs.shard_of(f"{c.ref.worker}|{c.ref.date}|{c.ref.level}|{c.ref.session}", args.shards) == args.shard]
    mine.sort(key=lambda c: (c.ref.worker, c.ref.date, c.ref.level, c.ref.session, c.ref.pass_index))
    if args.limit is not None:
        mine = mine[: args.limit]
    out_path = args.out_dir / f"shard_{args.shard:02d}_of_{args.shards:02d}.jsonl"
    done = {row["query"] for row in dcs.read_jsonl(out_path)}
    todo = [c for c in mine if dcs._ref_key(c.ref) not in done]
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
                        source, pair.query, pair.reference, reference_spreads=reference_spreads,
                        include_view=False, correction_policies=correction_policies.POLICIES,
                    )
                except Exception as exc:  # 한 쌍의 실패가 샤드를 멈추지 않게 — 사유를 남긴다.
                    body = {"status": "error", "reason": f"{type(exc).__name__}: {exc}"}
            reason = pair.candidate_reason or ("no_screen_candidate" if pair.reference is None else None)
            handle.write(json.dumps(record_of(pair.query, pair.reference, body, reason), ensure_ascii=False) + "\n")
            handle.flush()
            if index % 500 == 0:
                logger.warning("shard %d: %d/%d (%.2f/s)", args.shard, index, len(todo), index / (time.monotonic() - start))
    logger.warning("shard %d: finished in %.0fs", args.shard, time.monotonic() - start)
    return 0


def run_aggregate(args: argparse.Namespace) -> int:
    records: list[dict[str, Any]] = []
    for path in sorted(args.out_dir.glob("shard_*.jsonl")):
        records.extend(dcs.read_jsonl(path))
    report = aggregate(records)
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
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--index-cache", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--csv-dir", type=Path, default=None)
    parser.add_argument("--log-level", default="WARNING")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    return run_shard(args) if args.mode == "shard" else run_aggregate(args)


if __name__ == "__main__":
    raise SystemExit(main())
