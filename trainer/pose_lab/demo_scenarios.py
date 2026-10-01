#!/usr/bin/env python3
"""교정 시연 화면의 예시 쌍을 **규칙으로** 고른다 — 보기 좋은 쌍을 손으로 고르지 않는다.

규칙(2026-10-01, 고르기 전에 정함): 2차년도 전수 결과(`/output/delivered_cue/shard_*.jsonl`)에서 전달된
안내의 종류(부위 × 축, 6가지)마다, 안내 1건 split 개선율이 **그 종류의 중앙값에 가장 가까운** 쌍 하나.
같으면 질의 키 순서가 앞선 것. 결과는 `demo_scenarios.json`에 굳혀 커밋한다.

    python3 trainer/pose_lab/demo_scenarios.py /output/delivered_cue > trainer/pose_lab/demo_scenarios.json
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

POINT_LABELS = {"right": "오른손", "head": "머리"}
AXIS_LABELS = {"wall": "벽 쪽", "vertical": "위아래", "lateral": "진행 방향"}
ORDER = [("right", "vertical"), ("right", "wall"), ("right", "lateral"),
         ("head", "vertical"), ("head", "wall"), ("head", "lateral")]


def _ref(key: str) -> dict[str, Any]:
    worker, date, level, session, pass_index = key.split("|")
    return {"worker": worker, "date": date, "level": level, "session": session, "pass_index": int(pass_index)}


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    return ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2


def select(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for r in records:
        if r.get("status") == "ok" and r.get("split") is not None and r.get("reference"):
            groups[(str(r.get("delivered_point")), str(r.get("delivered_axis")))].append(r)
    out = []
    for point, axis in ORDER:
        rows = groups.get((point, axis))
        if not rows:
            continue
        median = _median([float(r["split"]) for r in rows])
        best = min(rows, key=lambda r: (abs(float(r["split"]) - median), str(r["query"])))
        out.append({
            "label": f"{POINT_LABELS[point]} · {AXIS_LABELS[axis]} 안내 (대표값)",
            "rule": "안내 종류별 1건 split 개선율 중앙값에 가장 가까운 쌍",
            "group_n": len(rows),
            "group_median_split": median,
            "query": _ref(str(best["query"])),
            "reference": _ref(str(best["reference"])),
        })
    return out


def main(argv: Sequence[str]) -> int:
    directory = Path(argv[1])
    records = [json.loads(line) for path in sorted(directory.glob("shard_*.jsonl"))
               for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(json.dumps({"scenarios": select(records)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
