"""`POST /api/v1/sessions/complete`의 세션 종합 자세 분석 — 리포트의 `pose_report`(명세서 p.15-16).

complete는 **세션 종합 분석**이다. 훈련생이 시퀀스마다 `pose_recomm`을 물었든 안 물었든, 세션의 모든 분사 시퀀스를
채점해 평균 자세 유사도와 가이드라인을 낸다. 정본은 complete에 첨부되는 전체 원시 CSV다.

시퀀스를 모으는 순서(상황별, 2026-10-08 샘플로 설계):
1. CSV를 분사 트리거(`분사` = On) 구간으로 나눈다. 시퀀스 경계는 시뮬레이터의 정의와 같다 — 샘플 세션에서 32구간이
   `analyze-sequence` 32건과 프레임 수·순서·건 위치까지 같았다. 구간 순서가 곧 `SEQ-001`… 번호다.
2. 같은 번호의 시퀀스가 이미 올라와 있으면(`analyze-sequence`) **그 본문**을 쓴다 — 시뮬레이터가 보낸 원본이다.
3. 올라오지 않은 시퀀스(네트워크 끊김·앱 재시작·연동 실패)는 CSV 구간으로 같은 모양의 본문을 만들어 채점한다.
4. CSV가 예전 형식(관절 월드 열 없음)이거나 읽을 수 없으면 올라온 시퀀스만 쓴다.
5. 도막에 명중하지 않은 분사(`zone_id` = -1)는 평균에서 뺀다 — 자세를 판정할 작업이 아니다. 개수는 남긴다.

구역별 집계(`zone_summary`·`priority_corrections`): 시퀀스는 **착탄이 가장 많이 난 (면, 구역)**을 칠한 것으로 본다.
CSV의 프레임별 `Zone ID`로 정한다 — 시뮬레이터 본문에는 프레임별 구역이 없다(2026-10-08 샘플 32건은 본문 `zone_id`와
CSV 다수결이 모두 같았다). Zone 번호는 레벨 안에서 유일하고 레벨은 리포트 머리(`scenario_id`)에 있다. 어긋난 관절은
엔진이 교정을 낸 관절 중 그 구역에서 **가장 자주 나온 하나**다 — LLM 브리핑이 여러 관절에 흩어지지 않게. 거리는 엔진
evidence의 숙련자와의 거리다. `angle_issues`는 비운다 — 엔진이 축별 각도·방향을 아직 내지 않는다
(오일러 회전순서 미확정, `pose_stages._correction_dict`). 값을 지어내지 않는다.

채점은 질의와 **같은 함수**(`pose_query.score_body`, MCP 자세 엔진)다. 판정할 수 없는 시퀀스(엔진의 reason_code)는
평균에서 빼고 사유별로 센다. 0으로 채우지 않는다. 채점된 시퀀스가 하나도 없으면 `unavailable`이다.
"""

from __future__ import annotations

import csv
import gzip
import io
import logging
import statistics
import zlib
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Final, Mapping, Sequence

import httpx

from trainer.demo_api.pose_query import PoseQueryResult, score_body
from trainer.demo_api.sequence_store import SequenceStore
from trainer.demo_api.report_jobs import CSV_INFLATED_LIMIT, InputTooLarge
from trainer.scripts import pose_similarity, simulator_csv

logger = logging.getLogger(__name__)

ALGORITHM_VERSION: Final = "pose-session-v2"
SPRAY_COLUMN: Final = "분사"
SPRAY_ON: Final = "On"
WORLD_SUFFIX: Final = " (World)"
GUN_GROUPS: Final[tuple[str, ...]] = ("한길숏건", "한길폴건", "반도숏건", "반도폴건")
NO_ZONE: Final = "-1"
# 분석 자료의 출처. 리포트 본문에는 싣지 않고(계약 키가 아니다) 완료 레코드에 남긴다.
SOURCE_STORED: Final = "stored"
SOURCE_CSV: Final = "csv"
ZONE_COLUMN: Final = "Zone ID"
# 리포트에 싣는 교정 항목 수. 명세서는 "제한된 개수"라고만 한다.
PRIORITY_LIMIT: Final = 3


def inflate_csv(raw: bytes, *, limit: int | None = None) -> bytes:
    limit = CSV_INFLATED_LIMIT if limit is None else limit
    if raw[:2] == b"\x1f\x8b":
        with gzip.GzipFile(fileobj=io.BytesIO(raw)) as handle:
            raw = handle.read(limit + 1)
    if len(raw) > limit:
        raise InputTooLarge("telemetry CSV exceeds inflated size limit")
    return raw


def read_csv(raw: bytes) -> list[dict[str, str]]:
    """원시 CSV(또는 `.csv.gz`) 바이트를 행으로. 압축 여부는 매직 바이트로 가린다. 못 읽으면 빈 목록."""
    try:
        raw = inflate_csv(raw)
        return list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))
    except (OSError, UnicodeDecodeError, csv.Error, EOFError, InputTooLarge, zlib.error):
        logger.warning("telemetry csv unreadable (%d bytes)", len(raw))
        return []


def has_world_joints(rows: Sequence[Mapping[str, str]]) -> bool:
    return bool(rows) and f"{simulator_csv.JOINT_COLUMNS[0]} Position X{WORLD_SUFFIX}" in rows[0]


def spray_runs(rows: Sequence[Mapping[str, str]]) -> list[tuple[int, int]]:
    """분사 트리거 구간 [시작, 끝] 행 번호(끝 포함). 시뮬레이터의 시퀀스 경계와 같다."""
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, row in enumerate(rows):
        on = row.get(SPRAY_COLUMN) == SPRAY_ON
        if on and start is None:
            start = index
        elif not on and start is not None:
            runs.append((start, index - 1))
            start = None
    if start is not None:
        runs.append((start, len(rows) - 1))
    return runs


def _float(row: Mapping[str, str], column: str) -> float | None:
    value = row.get(column)
    if value is None or str(value).strip() == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _vec(row: Mapping[str, str], columns: Sequence[str]) -> list[float] | None:
    values = [_float(row, column) for column in columns]
    if any(v is None for v in values):
        return None
    return [float(v) for v in values if v is not None]


def _column(row: Mapping[str, str], name: str) -> str:
    """머리글 앞뒤 공백이 섞여 온다(`' Ray Point Y'`). 공백·대소문자를 무시하고 찾는다."""
    for key in row:
        if key.strip().lower() == name.lower():
            return key
    return name


def active_gun(rows: Sequence[Mapping[str, str]]) -> str | None:
    """분사 중 가장 많이 움직인 건 강체. 쓰지 않는 건은 제자리에 있다."""
    best: tuple[float, str] | None = None
    spraying = [row for row in rows if row.get(SPRAY_COLUMN) == SPRAY_ON] or list(rows)
    for group in GUN_GROUPS:
        points = [p for p in (_vec(row, [f"{group} Position {a}" for a in "XYZ"]) for row in spraying) if p]
        if len(points) < 2:
            continue
        spread = sum(max(p[i] for p in points) - min(p[i] for p in points) for i in range(3))
        if best is None or spread > best[0]:
            best = (spread, group)
    return None if best is None else best[1]


def _frame(row: Mapping[str, str], gun: str, index: int) -> dict[str, Any]:
    joints: dict[str, list[float]] = {}
    for column in simulator_csv.JOINT_COLUMNS:
        position = _vec(row, [f"{column} Position {a}{WORLD_SUFFIX}" for a in "XYZ"])
        if position is not None:
            joints[simulator_csv.JOINT_KEYS[column].lower()] = position
    map_num = _float(row, "Ray Texture Map Number")
    return {
        "index": index,  # 시뮬레이터처럼 시퀀스 안에서 1부터
        "time": _float(row, "시간(초)"),
        "tool_pos_rot": {
            "pos": _vec(row, [f"{gun} Position {a}" for a in "XYZ"]),
            "rot": _vec(row, [f"{gun} Rotation {a}" for a in "XYZ"]),
        },
        "skeleton_joints": joints,
        "gun_metrics": {
            "speed": _float(row, "Speed (Real)"),
            "distance": _float(row, "Distance (Real)"),
            "raypoint": _vec(row, [_column(row, f"Ray Point {a}") for a in "XYZ"]),
            "raytexmapnum": None if map_num is None else int(map_num),
            "raytexuv": _vec(row, [_column(row, "Ray Texture UV X"), _column(row, "Ray Texture UV Y")]),
        },
    }


def painted_zone(rows: Sequence[Mapping[str, str]]) -> str:
    """구간에서 착탄이 가장 많이 난 구역. 미판정 -1은 세지 않고, 전부 -1이면 -1."""
    hits = [str(row.get(ZONE_COLUMN, "")).strip() for row in rows]
    hits = [zone for zone in hits if zone and zone != NO_ZONE]
    return Counter(hits).most_common(1)[0][0] if hits else NO_ZONE


def csv_sequences(
    rows: Sequence[Mapping[str, str]], *, session_id: str, user_id: str, scenario_id: str
) -> list[dict[str, Any]]:
    """CSV 분사 구간마다 `analyze-sequence`와 같은 모양의 본문. 관절 월드 열이 없으면(예전 형식) 빈 목록."""
    if not has_world_joints(rows):
        return []
    gun = active_gun(rows)
    if gun is None:
        return []
    out: list[dict[str, Any]] = []
    for number, (start, end) in enumerate(spray_runs(rows), start=1):
        window = rows[start:end + 1]
        out.append({
            "session_id": session_id,
            "user_id": user_id,
            "scenario_id": scenario_id,
            "zone_id": painted_zone(window),
            "sequence_id": f"SEQ-{number:03d}",
            "sequence_setting": {"tool_type": "spray_gun"},
            "sequence_info": [_frame(row, gun, i) for i, row in enumerate(window, start=1)],
            "sequence_result": {},
        })
    return out


@dataclass
class SessionPoseAnalysis:
    analysis_status: str  # "available" | "unavailable"
    average_similarity_score: float | None
    guideline: str
    sequences: list[dict[str, Any]] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    source_bodies: list[dict[str, Any]] = field(default_factory=list, repr=False)
    zone_summary: list[dict[str, Any]] = field(default_factory=list)
    priority_corrections: list[dict[str, Any]] = field(default_factory=list)

    def pose_report(self) -> dict[str, Any]:
        """명세서 p.15-16 `pose_report`의 키만."""
        report = {
            "analysis_status": self.analysis_status,
            "algorithm_version": ALGORITHM_VERSION,
            "guideline": self.guideline,
            "evaluated_sequence_count": self.counts.get("scored", 0),
            "average_similarity_score": self.average_similarity_score,
            "zone_summary": self.zone_summary,
            "priority_corrections": self.priority_corrections,
        }
        limited = [item for item in self.sequences if item.get("analysis_status") == "available" and item.get("confidence") == "limited"]
        if limited:
            report.update({
                "confidence": "limited", "limited_sequence_count": len(limited),
                "reason_codes": sorted({str(item["reason_code"]) for item in limited if item.get("reason_code")}),
            })
        return report


def _zone_order(zone: str) -> tuple[int, str]:
    return (int(zone) if zone.isdigit() else 1_000_000, zone)


def zone_summary(scored: Sequence[tuple[str, PoseQueryResult]]) -> list[dict[str, Any]]:
    """구역마다 채점된 시퀀스 수·평균 유사도·가장 자주 어긋난 관절 하나(교정이 난 시퀀스 수 / 그 구역 시퀀스 수)."""
    by_zone: dict[str, list[PoseQueryResult]] = {}
    for key, result in scored:
        by_zone.setdefault(key, []).append(result)
    out: list[dict[str, Any]] = []
    for key in sorted(by_zone, key=_zone_order):
        results = by_zone[key]
        distances: dict[str, list[float]] = {}
        counts: Counter[str] = Counter()
        for result in results:
            # 한 시퀀스에서 같은 관절은 한 번만 센다.
            for joint in dict.fromkeys(str(c.get("joint")) for c in result.corrections if c.get("joint")):
                counts[joint] += 1
                if joint in result.joint_distances:
                    distances.setdefault(joint, []).append(result.joint_distances[joint])
        problem_joints = [
            {
                "joint": joint,
                "occurrence_count": count,
                "occurrence_rate": round(count / len(results), 3),
                "average_distance": round(statistics.mean(distances[joint]), 3) if distances.get(joint) else None,
                "angle_issues": [],
            }
            for joint, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:1]
        ]
        out.append({
            "zone_id": key,
            "evaluated_sequence_count": len(results),
            "average_similarity_score": round(
                statistics.mean(r.similarity_score for r in results if r.similarity_score is not None), 3
            ),
            "problem_joints": problem_joints,
        })
    return out


def joint_measures(scored: Sequence[tuple[str, PoseQueryResult]]) -> dict[tuple[str, str], list[tuple[str, float]]]:
    """(구역, 관절) → 분사마다 교정 문장이 쓴 척도. 리포트에는 싣지 않고 우선 교정 문장만 쓴다."""
    out: dict[tuple[str, str], list[tuple[str, float]]] = {}
    for zone, result in scored:
        for joint, measure in result.joint_measures.items():
            out.setdefault((zone, joint), []).append(measure)
    return out


def _particle(word: str) -> str:
    last = ord(word[-1]) - 0xAC00 if word else -1
    return "이" if 0 <= last < 11172 and last % 28 else "가"


def _measured_phrase(label: str, measures: Sequence[tuple[str, float]]) -> str | None:
    """교정 문장(arm_reference.Correction.text)과 같은 척도·같은 낱말로 평균을 적는다. 반경·방향이 섞이면 많은 쪽."""
    modes = Counter(mode for mode, _ in measures)
    if not modes:
        return None
    mode = modes.most_common(1)[0][0]
    values = [value for kind, value in measures if kind == mode]
    average = statistics.mean(values)
    subject = f"{label}{_particle(label)}"
    if mode == "radius":
        return f"{subject} 숙련자 평균보다 평균 {abs(average) * 100:.0f}cm {'멀리' if average > 0 else '가까이'} 있었습니다."
    return f"{subject} 향한 방향이 숙련자 평균과 평균 {average:.0f}° 달랐습니다."


def priority_corrections(
    zones: Sequence[Mapping[str, Any]], limit: int = PRIORITY_LIMIT,
    *, measures: Mapping[tuple[str, str], Sequence[tuple[str, float]]] | None = None,
) -> list[dict[str, Any]]:
    """발생률이 높은 (구역, 관절) 순. 같으면 횟수가 많은 쪽.

    `measures`가 있으면 분사별 교정 문장과 같은 척도(반경 차·방향 차)로 적는다. 관절 위치 거리(`average_distance`)로
    적으면 "20 cm 가까이"인 교정이 "평균 35cm"로 보였다(2026-11-14 운영 시험).
    """
    pairs = sorted(
        ((zone, joint) for zone in zones for joint in zone["problem_joints"]),
        key=lambda pair: (-pair[1]["occurrence_rate"], -pair[1]["occurrence_count"]),
    )
    out: list[dict[str, Any]] = []
    for priority, (zone, joint) in enumerate(pairs[:limit], start=1):
        label = pose_similarity.JOINT_LABELS.get(joint["joint"], joint["joint"])
        distance = joint["average_distance"]
        lead = f"{zone['zone_id']}번 구역 분사 {zone['evaluated_sequence_count']}회 중 {joint['occurrence_count']}회에서 "
        measured = _measured_phrase(label, (measures or {}).get((str(zone["zone_id"]), joint["joint"]), ()))
        guidance = lead + measured if measured else (
            lead + f"{label} 위치가 숙련자 범위를 벗어났습니다"
            + (f"(평균 {distance * 100:.0f}cm)." if distance is not None else ".")
        )
        out.append({
            "priority": priority,
            "zone_id": zone["zone_id"],
            "joint": joint["joint"],
            "occurrence_rate": joint["occurrence_rate"],
            "correction_guidance": guidance,
        })
    return out


def _guideline(scored: Sequence[PoseQueryResult], total: int, not_scored: int) -> str:
    average = statistics.mean(r.similarity_score for r in scored if r.similarity_score is not None)
    parts = [f"분사 {total}회 중 {len(scored)}회의 자세를 숙련 작업자와 비교했습니다. 평균 유사도 {average * 100:.0f}%."]
    # 분사마다 첫 교정(전달되는 하나)의 부위를 센다. 문장은 분사마다 각도가 달라 같은 교정도 다른 문장이다.
    joints = Counter(
        str(c.get("joint")) for r in scored for c in r.corrections[:1] if isinstance(c, Mapping) and c.get("joint")
    )
    if joints:
        joint, times = joints.most_common(1)[0]
        parts.append(f"교정이 가장 자주 나온 부위는 {pose_similarity.JOINT_LABELS.get(joint, joint)}입니다({times}회).")
    elif not any(result.confidence == "limited" for result in scored):
        parts.append("비교한 분사 모두 숙련 작업자의 통상 범위 안입니다.")
    limited = sum(result.confidence == "limited" for result in scored)
    if limited:
        parts.append(f"이 중 {limited}회는 입력 품질 또는 비교 범위의 제한이 있어 참고용 유사도입니다. 교정 항목이 없더라도 정상 자세를 확정하지 않습니다.")
    if not_scored:
        parts.append(f"자세를 판정할 수 없는 분사 {not_scored}회는 평균에서 뺐습니다.")
    return " ".join(parts)


def analyze_session(
    *,
    session_id: str,
    user_id: str,
    scenario_id: str,
    csv_bytes: bytes | None,
    store: SequenceStore,
    mcp_base_url: str,
    transport: httpx.BaseTransport | None = None,
) -> SessionPoseAnalysis:
    """세션의 모든 분사 시퀀스를 채점해 평균 자세 유사도를 낸다(모듈 독스트링의 순서)."""
    rows = read_csv(csv_bytes) if csv_bytes else []
    from_csv = {
        body["sequence_id"]: body
        for body in csv_sequences(rows, session_id=session_id, user_id=user_id, scenario_id=scenario_id)
    }
    stored_bodies = store.session_bodies(session_id)
    stored_by_id = {str(body.get("sequence_id")): body for body in stored_bodies if body.get("sequence_id")}
    ordered_ids = list(from_csv) if from_csv else list(stored_by_id)
    if from_csv:
        ordered_ids.extend(sequence_id for sequence_id in stored_by_id if sequence_id not in from_csv)
    # 시퀀스 번호 → CSV에서 착탄이 가장 많이 난 구역. `Zone ID` 열이 없는 CSV는 본문 `zone_id`를 쓴다.
    painted = (
        {f"SEQ-{n:03d}": painted_zone(rows[s:e + 1]) for n, (s, e) in enumerate(spray_runs(rows), start=1)}
        if rows and ZONE_COLUMN in rows[0] else {}
    )
    targets: list[tuple[str, str, dict[str, Any]]] = []
    for sequence_id in ordered_ids:
        stored = stored_by_id.get(sequence_id)
        if stored is not None:
            targets.append((sequence_id, SOURCE_STORED, stored))
        elif sequence_id in from_csv:
            targets.append((sequence_id, SOURCE_CSV, from_csv[sequence_id]))

    counts: Counter[str] = Counter()
    counts["total"] = len(targets)
    reasons: Counter[str] = Counter()
    scored: list[PoseQueryResult] = []
    zoned: list[tuple[str, PoseQueryResult]] = []
    details: list[dict[str, Any]] = []
    for sequence_id, source, body in targets:
        counts[f"from_{source}"] += 1
        zone = painted.get(sequence_id) or str(body.get("zone_id", "")).strip()
        where = {"zone_id": zone}
        if zone in {NO_ZONE, ""}:
            counts["no_zone"] += 1
            details.append({
                "sequence_id": sequence_id, "source": source, **where,
                "analysis_status": "skipped", "reason_code": "no_zone",
            })
            continue
        result = score_body(body, sequence_id=sequence_id, mcp_base_url=mcp_base_url, transport=transport)
        details.append({"source": source, **where, **result.as_context()})
        if result.analysis_status == "available" and result.similarity_score is not None:
            scored.append(result)
            zoned.append((zone, result))
        else:
            reasons[str(result.reason_code)] += 1
            counts["not_scored"] += 1
    counts["scored"] = len(scored)
    count_map = dict(counts) | {f"reason:{k}": v for k, v in reasons.items()}

    if not scored:
        guideline = (
            "분석할 분사 기록이 없습니다." if not targets
            else f"분사 {len(targets)}회 중 자세를 판정할 수 있는 분사가 없습니다."
        )
        return SessionPoseAnalysis("unavailable", None, guideline, details, count_map, [body for _, _, body in targets])
    average = statistics.mean(r.similarity_score for r in scored if r.similarity_score is not None)
    guideline = _guideline(scored, len(targets) - counts["no_zone"], counts["not_scored"])
    zones = zone_summary(zoned)
    return SessionPoseAnalysis(
        "available", round(average, 4), guideline, details, count_map, [body for _, _, body in targets],
        zone_summary=zones, priority_corrections=priority_corrections(zones, measures=joint_measures(zoned)),
    )
