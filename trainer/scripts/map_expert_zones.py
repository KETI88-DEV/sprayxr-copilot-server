#!/usr/bin/env python3
"""전문가 CSV에 TSB 라벨맵으로 구역을 붙인다 — 구역별 자세 기준과 권장 순서의 재료.

옛 전문가 CSV(`/data/sources/professionals`, 2025-08~09)에는 `Zone ID` 열이 없고 착탄 UV는 있다. 프레임마다 착탄 UV가 든
구역을 라벨맵에서 읽고, 분사 구간(시퀀스)의 구역은 런타임과 같은 규칙 — 착탄이 가장 많이 난 구역
(`session_analysis.painted_zone`) — 으로 정한다. 구간도 런타임과 같은 `spray_runs`로 자른다.

학습 레벨 1~10(파일명 앞 번호 = scenario_id → Learn01~10)만 쓴다. 1-A·1-B·1-C는 라벨맵과 맞지 않아(착탄의 3~7%만
구역 안, 2026-10-15) 무엇인지 확인될 때까지 뺀다. 원본은 고치지 않고 `--out` 아래에 표 세 개를 쓴다.

    python3 -m trainer.scripts.map_expert_zones --source /data/sources/professionals --out /output/zone_data/professionals
"""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path

from trainer.demo_api.session_analysis import NO_ZONE, SPRAY_COLUMN, painted_zone, spray_runs
from trainer.zone_reference import ZoneReferenceCatalog

_UV = ("Ray Texture UV X", "Ray Texture UV Y")
_MAP = "Ray Texture Map Number"


def learning_scenario(path: Path) -> str | None:
    """파일명 앞 번호가 1~10이면 그 scenario_id. 1-A 같은 이름은 None."""
    prefix = path.name.split("_", 1)[0]
    return prefix if prefix.isdigit() and 1 <= int(prefix) <= 10 else None


def frame_zones(path: Path, catalog: ZoneReferenceCatalog, model_key: str) -> list[dict[str, str]]:
    """행마다 `분사`와 착탄 구역(`Zone ID`, 구역 밖·착탄 없음은 -1)."""
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = [name.strip() for name in next(reader)]
        spray, u_col, v_col, map_col = (header.index(name) for name in (SPRAY_COLUMN, *_UV, _MAP))
        rows: list[dict[str, str]] = []
        for row in reader:
            zone = None
            try:
                u, v, map_num = float(row[u_col]), float(row[v_col]), int(float(row[map_col]))
                if u > 0 or v > 0:
                    zone = catalog.zone_at(model_key, map_num, u, v)
            except (ValueError, IndexError):
                pass
            rows.append({SPRAY_COLUMN: row[spray] if spray < len(row) else "", "Zone ID": zone or NO_ZONE})
    return rows


def first_visit_flow(zones: Sequence[str]) -> list[str]:
    """시퀀스 구역들에서 처음 칠한 순서. 미판정(-1)은 뺀다."""
    return list(dict.fromkeys(zone for zone in zones if zone != NO_ZONE))


def _write(path: Path, header: list[str], rows: list[list[object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, default=Path("/data/sources/professionals"))
    parser.add_argument("--out", type=Path, default=Path("/output/zone_data/professionals"))
    args = parser.parse_args(argv)
    catalog = ZoneReferenceCatalog.default()
    args.out.mkdir(parents=True, exist_ok=True)

    sequence_rows: list[list[object]] = []
    flow_rows: list[list[object]] = []
    flows: dict[str, list[list[str]]] = defaultdict(list)
    skipped = 0
    for path in sorted(args.source.rglob("*.csv")):
        scenario = learning_scenario(path)
        model_key = catalog.resolve_model(scenario) if scenario else None
        if model_key is None:
            skipped += 1
            continue
        worker = path.relative_to(args.source).parts[0]
        rows = frame_zones(path, catalog, model_key)
        zones: list[str] = []
        for number, (start, end) in enumerate(spray_runs(rows), start=1):
            window = rows[start:end + 1]
            zone = painted_zone(window)
            hits = Counter(row["Zone ID"] for row in window if row["Zone ID"] != NO_ZONE)
            zones.append(zone)
            sequence_rows.append([
                worker, path.name, scenario, f"SEQ-{number:03d}", start + 1, end + 1, len(window), zone,
                round(hits[zone] / sum(hits.values()) * 100, 1) if hits else 0.0,
                len(window) - sum(hits.values()),
            ])
        flow = first_visit_flow(zones)
        flows[scenario].append(flow)
        flow_rows.append([worker, path.name, scenario, len(zones), " → ".join(flow)])

    order_rows: list[list[object]] = []
    for scenario in sorted(flows, key=int):
        model_key = catalog.resolve_model(scenario) or ""
        total = len(catalog.zones(model_key, 1))
        files = flows[scenario]
        common, common_count = Counter(" → ".join(flow) for flow in files).most_common(1)[0]
        ranks: dict[str, list[int]] = defaultdict(list)
        for flow in files:
            for rank, zone in enumerate(flow, start=1):
                ranks[zone].append(rank)
        by_rank = sorted(ranks, key=lambda zone: (statistics.mean(ranks[zone]), int(zone)))
        order_rows.append([
            scenario, model_key, len(files), total,
            sum(1 for flow in files if len(flow) == total),
            common, common_count,
            " → ".join(by_rank),
            " ".join(f"{zone}:{statistics.mean(ranks[zone]):.1f}({len(ranks[zone])})" for zone in by_rank),
        ])

    _write(args.out / "시퀀스_구역.csv",
           ["작업자", "파일", "시나리오", "시퀀스", "시작행", "끝행", "프레임", "구역", "구역착탄비율%", "구역밖프레임"],
           sequence_rows)
    _write(args.out / "파일별_방문순서.csv", ["작업자", "파일", "시나리오", "시퀀스수", "처음칠한순서"], flow_rows)
    _write(args.out / "권장순서_후보.csv",
           ["시나리오", "모델", "파일수", "구역수", "전구역방문파일수", "최빈순서", "최빈순서파일수",
            "평균순위순서", "구역별평균순위(방문파일수)"],
           order_rows)
    print(f"파일 {len(flow_rows)}개, 시퀀스 {len(sequence_rows)}개, 제외 {skipped}개 → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
