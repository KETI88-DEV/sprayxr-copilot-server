#!/usr/bin/env python3
# pyright: reportAny=false, reportUnusedCallResult=false
"""몸 제약 개선율 — 교정을 **사람 몸으로 따라 했을 때**의 정확도를 전수로 다시 잰다.

사전등록: `docs/measurements/2026-10-01-body-constrained-improvement-prereg.md`
(3차년도 정책은 `docs/year3/2026-10-01-body-constrained-prereg.md`)

기존 개선율(`service.compare`)은 채점하는 세 점(오른손 건·머리 HMD·왼손 플래시)을 **서로 독립으로** 안내한 만큼
옮긴다. 팔 길이 제약이 없고, 안내가 없는 점은 몸이 옮겨 서거나 일어서도 제자리라고 본다. 이 스크립트는 같은 쌍·
같은 채점기로, 교정을 몸으로 실현한 뒤 다시 잰다.

몸으로 실현하기(`trainer/pose_lab/correction_demo.corrected_body`, 시연 화면과 같은 규칙):
- 손 안내: 손을 안내한 곳으로 보내고 팔꿈치를 다시 푼다. 팔 길이로 닿지 않으면 쇄골 길이를 지킨 채 어깨를 최대
  35° 돌리고, 그래도 모자라면 못 미친 채 둔다.
- 옮겨 서기(머리 수평): 몸 전체가 옮겨 간다 — 안내 없는 손도 따라간다.
- 자세 높이기·낮추기(머리 수직): 발을 디딘 채 골반 위가 오르내리고, 다리가 다 펴지거나 다 접히는 데서 멈춘다.
그 몸에서 오른손·머리·왼손 **관절이 실제로 움직인 만큼** 세 점의 궤적 전체를 옮기고(평행 이동) 다시 채점한다.

한계(결과 문서에 함께 적는다): 몸은 패스의 대표 순간 하나의 골격이다(채점은 패스 평균). 장비(건·HMD·플래시)의
이동을 손·머리 관절의 이동으로 대신하고 회전은 무시한다. 다리는 추정이다.

산출: 샤드 JSONL(`--out-dir`), 집계 JSON·CSV(`aggregate --csv-dir`).
"""

from __future__ import annotations

import argparse
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
_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

measured_score = importlib.import_module("measured_score")
pose_similarity = importlib.import_module("pose_similarity")
improvement_study = importlib.import_module("improvement_study")
dcs = importlib.import_module("delivered_cue_study")

from trainer.pose_lab import correction_demo as cd  # noqa: E402

logger = logging.getLogger(__name__)

THRESHOLDS: Final[tuple[tuple[str, float], ...]] = (("2차년도 70%", 0.70), ("3차년도 90%", 0.90))
MIN_TRIALS: Final = 10
MAX_VALUES: Final = 6
POLICY_LABELS: Final = {
    "issued": "교정 알고리즘 안내 전부(관절 값 ≤4)",
    "delivered": "작업자에게 전달되는 안내 1건",
    "six_values": "관절 값 6개 이하 교정(P2)",
}
REALIZATION_LABELS: Final = {"model": "점 독립(기존)", "body": "몸 제약"}


def policies() -> dict[str, Any]:
    """측정할 정책. 3차년도 정책은 `trainer/year3`가 있을 때만 들어간다."""
    out: dict[str, Any] = {"issued": cd.issued_policy, "delivered": cd.delivered_policy}
    try:
        from trainer.year3 import correction_policies

        out["six_values"] = correction_policies.POLICIES[correction_policies.PRIMARY]
    except ImportError:
        logger.warning("trainer.year3 not available; six_values skipped")
    return out


# --- 몸으로 실현하기 (순수) ---------------------------------------------------------


# 안내 없는 손의 가정. 등록한 주 측정은 `follow`, `stay`는 결과를 본 뒤 정한 사후 민감도 분석이다.
UNCUED_HANDS: str = "follow"


def realized_shifts(
    body: Mapping[str, Sequence[float]], chosen: Sequence[Any], frame: Any, uncued_hands: str | None = None
) -> tuple[dict[str, tuple[float, float, float]], dict[str, float]]:
    """교정을 몸으로 따랐을 때 채점 지점이 실제로 옮겨 간 양(스트로크 프레임 성분)과, 안내에 못 미친 거리(cm)."""
    after = cd.corrected_body(body, chosen, uncued_hands=uncued_hands or UNCUED_HANDS)
    intended = cd.point_shifts(chosen)
    shifts: dict[str, tuple[float, float, float]] = {}
    short: dict[str, float] = {}
    for point, joint in cd.ANCHOR_JOINT.items():
        if joint not in body or joint not in after:
            continue
        moved = cd._sub(after[joint], body[joint])
        shifts[point] = (
            measured_score._dot(moved, frame.normal),
            measured_score._dot(moved, frame.tangent),
            measured_score._dot(moved, frame.up),
        )
        if point in intended:
            short[point] = cd._norm(cd._sub(moved, intended[point])) * 100.0
    return shifts, short


def body_score(inputs: Mapping[str, Any], body: Mapping[str, Sequence[float]]) -> tuple[float, dict[str, float]]:
    shifts, short = realized_shifts(body, inputs["chosen"], inputs["frame"])
    tracks = measured_score.shift_tracks(inputs["query_track"], shifts)
    score = measured_score.score_track(
        tracks, inputs["reference_track"], inputs["score_spreads"], inputs["weights"], inputs["frame"]
    ).score
    return score, short


def _gap(before: float | None, after: float | None) -> float | None:
    if before is None or after is None:
        return None
    return pose_similarity.gap_closure(before, after)


# --- 한 쌍 ------------------------------------------------------------------------


def record_of(source: Any, reference_spreads: Any, pair: Any, chosen_policies: Mapping[str, Any]) -> dict[str, Any]:
    from trainer.pose_lab import service

    out: dict[str, Any] = {
        "query": dcs._ref_key(pair.query), "worker": pair.query.worker, "level": pair.query.level,
        "reference": dcs._ref_key(pair.reference) if pair.reference is not None else None,
    }
    if pair.reference is None:
        out.update({"status": "no_expert_reference", "reason": pair.candidate_reason or "no_screen_candidate"})
        return out
    observed: dict[tuple[str, str], dict[str, Any]] = {}

    def observe(name: str, side: str, inputs: Mapping[str, Any]) -> None:
        observed[(name, side)] = dict(inputs)

    try:
        body = service.compare(
            source, pair.query, pair.reference, reference_spreads=reference_spreads, include_view=True,
            correction_policies=chosen_policies, policy_observer=observe,
        )
    except Exception as exc:  # 한 쌍의 실패가 샤드를 멈추지 않게 — 사유를 남긴다.
        out.update({"status": "error", "reason": f"{type(exc).__name__}: {exc}"})
        return out
    if body.get("status") != "ok":
        out.update({"status": "unavailable", "reason": str(body.get("reason"))})
        return out
    measured = body["measured"]
    view = body.get("view") or {}
    skeleton = cd._skeleton((view.get("layers") or {}).get("query"), view.get("fit"))
    out.update({"status": "ok", "score": measured.get("score"), "skeleton": skeleton is not None})
    for name in chosen_policies:
        result = measured["policies"][name]
        out[f"{name}.values"] = result.get("values")
        out[f"{name}.same"] = result["improvement"].get("same")
        out[f"{name}.split"] = result["improvement"].get("split")
        if skeleton is None:
            continue
        for side in ("same", "split"):
            inputs = observed.get((name, side))
            if inputs is None or inputs.get("before_score") is None:
                continue
            score, short = body_score(inputs, skeleton)
            out[f"{name}.{side}_body"] = _gap(inputs["before_score"], score)
            if side == "same":
                out[f"{name}.short_hand_cm"] = round(max(short.get("right", 0.0), short.get("left", 0.0)), 3)
                out[f"{name}.short_head_cm"] = round(short.get("head", 0.0), 3)
    return out


# --- 집계 (순수) ------------------------------------------------------------------


def _share(rows: Sequence[Mapping[str, Any]], key: str, threshold: float) -> float | None:
    present = [float(r[key]) for r in rows if r.get(key) is not None]
    return sum(1 for v in present if v >= threshold) / len(present) if present else None


def aggregate(records: Sequence[Mapping[str, Any]], names: Sequence[str]) -> dict[str, Any]:
    ok = [r for r in records if r.get("status") == "ok"]
    with_body = [r for r in ok if r.get("skeleton")]
    out: dict[str, Any] = {
        "n_records": len(records),
        "status_counts": dict(Counter(str(r.get("status")) for r in records)),
        "n_ok": len(ok),
        "n_with_skeleton": len(with_body),
        "policies": {},
    }
    for name in names:
        block: dict[str, Any] = {}
        for side in ("same", "split"):
            # 같은 쌍에서 견준다 — 두 값이 모두 있는 쌍만 센다.
            paired = [r for r in with_body
                      if r.get(f"{name}.{side}") is not None and r.get(f"{name}.{side}_body") is not None]
            block[side] = {
                "model_all_ok": dcs.describe(r.get(f"{name}.{side}") for r in ok),
                "model": dcs.describe(r[f"{name}.{side}"] for r in paired),
                "body": dcs.describe(r[f"{name}.{side}_body"] for r in paired),
                "difference": dcs.describe(
                    (r[f"{name}.{side}_body"] - r[f"{name}.{side}"] for r in paired), threshold=None),
                "body_lower_share": (
                    sum(1 for r in paired if r[f"{name}.{side}_body"] < r[f"{name}.{side}"] - 1e-9) / len(paired)
                ) if paired else None,
            }
        values = [r.get(f"{name}.values") for r in ok if r.get(f"{name}.values") is not None]
        block["values_max"] = max(values) if values else None
        block["short_hand_cm"] = dcs.describe((r.get(f"{name}.short_hand_cm") for r in with_body), threshold=None)
        block["short_head_cm"] = dcs.describe((r.get(f"{name}.short_head_cm") for r in with_body), threshold=None)
        block["short_hand_over_1cm_share"] = _share(with_body, f"{name}.short_hand_cm", 1.0)
        block["short_head_over_1cm_share"] = _share(with_body, f"{name}.short_head_cm", 1.0)
        out["policies"][name] = block
    by_worker: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for r in with_body:
        by_worker[str(r.get("worker"))].append(r)
    per_worker: dict[str, Any] = {}
    for worker, rows in sorted(by_worker.items()):
        entry: dict[str, Any] = {"n": len(rows)}
        for name in names:
            entry[name] = {
                side: {
                    "model": dcs.describe(r.get(f"{name}.{side}") for r in rows)["mean"],
                    "body": dcs.describe(r.get(f"{name}.{side}_body") for r in rows)["mean"],
                }
                for side in ("same", "split")
            }
        per_worker[worker] = entry
    out["per_worker"] = per_worker
    return out


def _pct(value: float | None) -> float | None:
    return None if value is None else value * 100.0


def write_csvs(records: Sequence[Mapping[str, Any]], report: Mapping[str, Any], names: Sequence[str], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    path = out_dir / "정확도_평가방법.csv"
    rows = []
    for name in names:
        block = report["policies"][name]
        within = None if block["values_max"] is None else block["values_max"] <= MAX_VALUES
        for side, side_label in (("same", "목표 자세로 교정·채점(same)"), ("split", "교정·채점 기준 분리(split)")):
            for realization in ("model", "body"):
                stats = block[side][realization]
                mean = stats.get("mean")
                eligible = bool(within) and (stats.get("n") or 0) >= MIN_TRIALS
                rows.append([
                    POLICY_LABELS[name], side_label, REALIZATION_LABELS[realization], block["values_max"], within,
                    stats.get("n"), _pct(mean), stats.get("median"),
                    *[(None if mean is None or not eligible else mean >= t) for _l, t in THRESHOLDS],
                ])
    dcs._write(path, ["교정", "채점 방식", "교정 실현", "조정한 값 최대 개수", "6개 이하", "적용 횟수",
                      "정확도 평균(%)", "개선율 중앙값", *(f"{label} 달성" for label, _t in THRESHOLDS)], rows)
    written.append(path)

    path = out_dir / "몸제약_차이.csv"
    rows = []
    for name in names:
        block = report["policies"][name]
        for side in ("same", "split"):
            diff = block[side]["difference"]
            rows.append([POLICY_LABELS[name], side, diff.get("n"), _pct(diff.get("mean")), _pct(diff.get("median")),
                         _pct(diff.get("q1")), _pct(diff.get("q3")), block[side]["body_lower_share"]])
    dcs._write(path, ["교정", "채점 방식", "쌍 수", "차이 평균(%p, 몸−독립)", "차이 중앙값(%p)", "1사분위(%p)",
                      "3사분위(%p)", "몸 제약이 더 낮은 쌍 비율"], rows)
    written.append(path)

    path = out_dir / "못미친거리.csv"
    rows = []
    for name in names:
        block = report["policies"][name]
        for part, key, share_key in (("손(팔 길이)", "short_hand_cm", "short_hand_over_1cm_share"),
                                     ("머리(발 디딤)", "short_head_cm", "short_head_over_1cm_share")):
            stats = block[key]
            rows.append([POLICY_LABELS[name], part, stats.get("n"), stats.get("mean"), stats.get("median"),
                         stats.get("q3"), block[share_key]])
    dcs._write(path, ["교정", "부위", "쌍 수", "평균(cm)", "중앙값(cm)", "3사분위(cm)", "1cm 이상 못 미친 쌍 비율"], rows)
    written.append(path)

    path = out_dir / "작업자별.csv"
    header = ["작업자", "쌍 수"] + [
        f"[{POLICY_LABELS[name]}] {side} {REALIZATION_LABELS[realization]}(%)"
        for name in names for side in ("same", "split") for realization in ("model", "body")
    ]
    rows = []
    for worker, block in report["per_worker"].items():
        row: list[Any] = [worker, block["n"]]
        for name in names:
            for side in ("same", "split"):
                for realization in ("model", "body"):
                    row.append(_pct(block[name][side][realization]))
        rows.append(row)
    dcs._write(path, header, rows)
    written.append(path)

    path = out_dir / "쌍별_원자료.csv"
    base = [("query", "질의 패스"), ("worker", "작업자"), ("level", "레벨"), ("reference", "기준 패스"),
            ("status", "상태"), ("reason", "사유"), ("score", "교정 전 유사도"), ("skeleton", "골격 있음")]
    per = [("values", "조정 값 수"), ("same", "same 점 독립"), ("same_body", "same 몸 제약"),
           ("split", "split 점 독립"), ("split_body", "split 몸 제약"),
           ("short_hand_cm", "손 못 미친 거리(cm)"), ("short_head_cm", "머리 못 미친 거리(cm)")]
    columns = base + [(f"{name}.{key}", f"[{POLICY_LABELS[name]}] {label}") for name in names for key, label in per]
    dcs._write(path, [label for _k, label in columns], ([r.get(k) for k, _l in columns] for r in records))
    written.append(path)
    return written


# --- 실행 -------------------------------------------------------------------------


def run_shard(args: argparse.Namespace) -> int:
    from trainer.pose_lab.spread_reference import SpreadReference

    reference_spreads = SpreadReference.load(args.reference)
    improvement_study.assert_artifact_prefix(reference_spreads, args.reference)
    source, candidates = improvement_study.index_corpus_cached(args.root, args.index_cache)
    chosen_policies = policies()
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
            record = record_of(source, reference_spreads, pair, chosen_policies)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            if index % 200 == 0:
                logger.warning("shard %d: %d/%d (%.2f/s)", args.shard, index, len(todo),
                               index / (time.monotonic() - start))
    logger.warning("shard %d: finished in %.0fs", args.shard, time.monotonic() - start)
    return 0


def run_aggregate(args: argparse.Namespace) -> int:
    records: list[dict[str, Any]] = []
    for path in sorted(args.out_dir.glob("shard_*.jsonl")):
        records.extend(dcs.read_jsonl(path))
    names = [name for name in args.policies.split(",") if name]
    report = aggregate(records, names)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text, encoding="utf-8")
    if args.csv_dir is not None:
        for path in write_csvs(records, report, names, args.csv_dir):
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
    parser.add_argument("--policies", default="issued,delivered", help="집계할 정책(쉼표). 3차년도는 six_values")
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--csv-dir", type=Path, default=None)
    parser.add_argument("--uncued-hands", choices=("follow", "stay"), default="follow",
                        help="안내 없는 손: follow(등록한 주 측정) / stay(사후 민감도 분석)")
    parser.add_argument("--log-level", default="WARNING")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    global UNCUED_HANDS
    UNCUED_HANDS = args.uncued_hands
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    return run_shard(args) if args.mode == "shard" else run_aggregate(args)


if __name__ == "__main__":
    raise SystemExit(main())
