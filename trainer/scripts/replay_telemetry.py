#!/usr/bin/env python3
"""기록된 숙련자 세션을 v2.5 `POST /api/v1/telemetry/sequence` 입력으로 재생한다.

실물 벤더 샘플은 **건을 고정해 촬영**했으므로(샘플 `notes/avg_speed_mms`) 골격이 한
시퀀스 안에서 전혀 움직이지 않고, 두 번째 시퀀스부터 `skeleton_joint_frozen`으로 채점이
막힌다. 움직이는 캡처가 오기 전까지 이 드라이버가 그 자리를 메운다 — 코퍼스
(`/data/sources/professionals`, 605세션)는 실제 작업자 움직임이다.

**관절은 FK로 만든다.** CSV의 관절 Position 컬럼은 Pelvis를 뺀 19관절이 프레임 불변인
**부모 기준 바인드 오프셋**이고 Pelvis Position도 월드가 아니다(`pose_fk_csv` 모듈
독스트링). 그대로 실으면 뼈 길이가 0인 마디가 생기고 발끝→머리가 음수가 되어 게이트가
거부한다. 그래서 레퍼런스를 만든 방식(`build_arm_reference`)과 같게
`pose_fk_csv.forward_kinematics`를 쓴다 — 루트를 원점에 둔 CSV 자체 리그 좌표다.

**`tool_pos_rot.pos`는 합성이다.** FK 관절은 루트 원점 공간이고 CSV 건·착탄점은 장면
좌표다. 한 공간이 아니므로 실제 건 좌표를 실으면 `tool_not_held`가 거부한다. 손 위치를
그대로 쓰면 간격이 0이 되어 게이트 하한(`HAND_TOOL_MIN_M` 0.05 m)에 걸리므로, **전완을
이어 손잡이 길이만큼 앞**에 둔다. 길이는 실물 SEQ-002가 보인 `hand_tool_gap_m` 0.3585 m를
쓴다 — 하드웨어 치수 L은 아직 모른다.

**착탄점은 건과 같은 평행이동으로 함께 옮긴다**(`_impact_beside_tool`). `reach = 착탄
수직 − 건 수직`은 차이라서 평행이동에 불변이고, CSV의 건과 착탄은 **둘 다 장면 좌표**라
그 차이가 실측값이다. 건만 옮기고 착탄을 두면 두 공간의 차를 빼는 셈이 되어 reach 가
튄다 — 그것이 이 파일의 이전 판이 "점수가 나오지 않는다"고 적은 이유였고, 원인은 공간이
아니라 **한쪽만 옮긴 것**이었다.

실측(1-A_1000452_1 첫 구간 40프레임):

| 착탄점 처리 | reach_m | band | support | similarity_score |
|---|---|---|---|---|
| 장면 좌표 그대로(이전 판) | 4.8542 | — | — | `reference_data_missing` |
| 건과 같은 평행이동 | 0.167 | +0.10 | 5073 | 0.7237 |

**따라서 이 드라이버가 검증하는 것과 못 하는 것.**

* 검증한다 — 엔드포인트 배선, 모니터 기록, 중복 식별, 골격 게이트(움직임·뼈·높이·간격),
  `reach` 버킷 선택, 레퍼런스 조회, **점수 산출과 교정 생성**
* 못 한다 — `tool_not_held`(건 위치가 손에서 합성되어 간격이 설계상 항상 범위 안이다),
  수평 방향이 걸리는 규칙(FK 리그와 장면의 정합을 모르므로 수직 성분만 보증한다)

실물 운용 전에는 여전히 **이동하며 분사한 global 캡처**가 필요하다. 이 드라이버는 그
전까지 채점 경로를 돌려보는 수단이지, 좌표 정합을 대신하지 않는다.

    python3 trainer/scripts/replay_telemetry.py --list
    python3 trainer/scripts/replay_telemetry.py --session 1-A_1000452_1 --spans 2 --dry-run
    python3 trainer/scripts/replay_telemetry.py --session 1-A_1000452_1 --spans 2 \
        --api-url http://127.0.0.1:58000
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pose_fk_csv  # noqa: E402  CSV 자체 리그로 도는 FK

CORPUS_ROOT: Final = Path("/data/sources/professionals")
TELEMETRY_PATH: Final = "/api/v1/telemetry/sequence"

# trainer.scripts.simulator_csv.JOINT_COLUMNS 와 같은 20종. v2.5 키는 소문자 무공백이다
# (실물 샘플 표기).
JOINT_COLUMNS: Final[tuple[str, ...]] = (
    "Pelvis", "Spine", "Chest",
    "Left Shoulder", "Left Upper Arm", "Left Fore Arm", "Left Hand",
    "Right Shoulder", "Right Upper Arm", "Right Fore Arm", "Right Hand",
    "Left Thigh", "Left Calf", "Left Foot", "Left Toe",
    "Right Thigh", "Right Calf", "Right Foot", "Right Toe",
    "Head",
)
HAND_COLUMN: Final = "Right Hand"
GUN_COLUMN: Final = "한길숏건"
RAY_COLUMNS: Final[tuple[str, str, str]] = ("Ray point X", "Ray Point Y", "Ray Point Z")
TIME_BASE: Final = datetime(2026, 9, 21, 0, 0, 0, tzinfo=timezone.utc)

# 실물 SEQ-002 의 hand_tool_gap_m. 하드웨어 치수 L 이 오면 이 값을 대체한다.
HAND_TOOL_GAP_M: Final = 0.3585

SYNTHESIS_NOTICE: Final = (
    "관절은 pose_fk_csv.forward_kinematics(루트 원점)로 만든 뒤 CSV Pelvis Position 만큼 옮겼다.\n"
    f"  tool_pos_rot.pos 는 전완을 이어 {HAND_TOOL_GAP_M} m 앞에 합성했다(FK 공간과 장면 좌표가 다르다).\n"
    "  gun_metrics.raypoint 는 건과 같은 평행이동으로 옮겨 건→착탄 벡터를 보존했다 — reach 는 실측이다.\n"
    "  따라서 tool_not_held 와 수평 방향 규칙은 이 입력으로 검증되지 않는다."
)


def _key(column: str) -> str:
    return column.replace(" ", "").lower()


def _vec(row: dict[str, str], column: str, kind: str) -> list[float]:
    return [float(row[f"{column} {kind} {axis}"]) for axis in ("X", "Y", "Z")]


def _tool_from_hand(hand: Sequence[float], forearm: Sequence[float]) -> list[float]:
    """손잡이를 전완 방향으로 ``HAND_TOOL_GAP_M`` 앞에 둔다.

    간격 0 은 게이트 하한(0.05 m)에 걸린다. 방향은 전완→손 벡터를 잇는데, 손에 쥔 건이
    대략 그 방향을 향한다는 가정이다 — 하드웨어 치수 L 과 실제 파지 방향은 아직 모른다.
    """
    direction = [float(hand[i]) - float(forearm[i]) for i in range(3)]
    norm = sum(component * component for component in direction) ** 0.5
    if norm <= 1e-9:
        return [float(hand[0]) + HAND_TOOL_GAP_M, float(hand[1]), float(hand[2])]
    scale = HAND_TOOL_GAP_M / norm
    return [float(hand[i]) + direction[i] * scale for i in range(3)]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = [column.strip() for column in next(reader)]
        return [dict(zip(header, row)) for row in reader if row]


def spray_spans(rows: Sequence[dict[str, str]]) -> list[tuple[int, int]]:
    """분사 On 구간. 트리거 ON→OFF 하나가 계약의 시퀀스 하나다(§7.1)."""
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for index, row in enumerate(rows):
        on = row.get("분사", "").strip() == "On"
        if on and start is None:
            start = index
        elif not on and start is not None:
            spans.append((start, index - 1))
            start = None
    if start is not None:
        spans.append((start, len(rows) - 1))
    return spans




def _root_motion(row: Mapping[str, str]) -> list[float]:
    """CSV가 기록한 Pelvis 이동. FK가 버리는 **실제 움직임**을 되살린다.

    `pose_fk_csv.forward_kinematics` 는 루트를 원점에 두므로 Pelvis 가 모든 프레임에서
    `(0,0,0)` 이다. 그러면 골격 게이트가 "이 관절은 움직이지 않는다"고 보고 `degrade` 를
    매긴다(`skeleton_joint_frozen`). 그런데 CSV 의 Pelvis Position 은 실제로 변한다 —
    1-A_1000452_1 첫 구간 90프레임에서 고유값 81개, X·Y 범위 0.11 m 다.

    **점수는 바뀌지 않는다(실측).** 같은 구간에서 루트 고정 0.7293 / 루트 이동 0.7293 으로
    동일하고 밴드도 `+0.10` 로 같다. 채점이 Pelvis 를 쓰지 않기 때문이다 — Pelvis 는
    `excluded_joints` 이고 채점 관절은 팔 6개다. 즉 이 `degrade` 는 작업자의 성질이 아니라
    우리 합성의 부작용이었고, 되살리면 `available` 이 된다.

    Pelvis Position 자체는 월드 좌표가 아니다(`pose_fk_csv` 독스트링). 쓰는 것은 절대값이
    아니라 **프레임 간 변화**다.
    """
    return [float(row[f"Pelvis Position {axis}"]) for axis in "XYZ"]


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def _register_impacts(frames: list[dict[str, Any]]) -> None:
    """착탄점을 장면 좌표에서 FK 공간으로 **시퀀스당 상수 하나**로 옮긴다(제자리 수정).

    `reach = 착탄 수직 − 건 수직`은 차이라서 평행이동에 불변이다(`arm_reference` 독스트링:
    "둘 다 월드라 차이에서 바닥 상수가 지워진다"). CSV의 건과 착탄은 **둘 다 장면 좌표**라
    그 차이가 실측값이다. 건만 FK 공간으로 합성하고 착탄을 장면 좌표로 두면 두 공간의 차를
    빼는 셈이 되어 reach 가 몇 미터로 튄다(실측 4.8542 → `reference_data_missing`).

    **상수여야 하는 이유.** `pose_stages.reach_of` 는 착탄과 건의 중앙값을 **각각 따로**
    취한다. 프레임마다 다른 델타로 옮기면 두 중앙값이 독립적으로 움직여 차이가 보존되지
    않는다 — 실측으로 한 구간의 reach 가 0.1863 대신 0.5421 이 되어 밴드가 `+0.10` 이 아닌
    `+0.50`(support 202)으로 잘못 잡혔다. 축별 중앙값으로 맞추면 중앙값끼리 정확히 상쇄되어
    reach 가 CSV 실측과 같아진다.

    **수직 성분만 보증한다.** 두 공간의 정합(registration)을 모르므로 수평 방향이 골격에
    대해 맞는지는 보증하지 못한다. 지금 채점은 reach 의 수직 성분만 쓴다.
    """
    pairs = [(f["tool_pos_rot"]["pos"], f["gun_metrics"].pop("_gun_scene")) for f in frames]
    if not pairs:
        return
    offset = [
        _median([tool[axis] for tool, _ in pairs]) - _median([gun[axis] for _, gun in pairs])
        for axis in range(3)
    ]
    for frame in frames:
        raypoint = frame["gun_metrics"]["raypoint"]
        frame["gun_metrics"]["raypoint"] = [raypoint[axis] + offset[axis] for axis in range(3)]


def build_frame(
    row: dict[str, str], index: int, span_start_sec: float, offsets: Mapping[str, Any]
) -> dict[str, Any] | None:
    """착탄점이 없는 프레임은 ``None``. 미명중 프레임을 섞으면 구간 성질이 흐려진다."""
    if not row.get(RAY_COLUMNS[0], "").strip():
        return None
    seconds = float(row["시간(초)"])
    stamp = TIME_BASE + timedelta(seconds=seconds)
    # FK: 루트 원점 공간. 키는 "RightHand" 형태라 v2.5 소문자 키로 바꾼다.
    fk = pose_fk_csv.forward_kinematics(row, offsets)
    root = _root_motion(row)
    joints = {name.lower(): [pos[axis] + root[axis] for axis in range(3)] for name, pos in fk.items()}
    hand = joints.get("righthand")
    forearm = joints.get("rightforearm")
    if hand is None or forearm is None:
        return None
    # 도구는 **옮긴 뒤의** 손에서 합성한다. 먼저 합성하면 손↔건 간격이 루트 이동만큼 틀어진다.
    tool = _tool_from_hand(hand, forearm)
    return {
        "index": index,
        "time": stamp.strftime("%Y-%m-%dT%H:%M:%S.") + f"{stamp.microsecond // 1000:03d}Z",
        # pos 는 합성(전완 연장), rot 은 CSV 건 오브젝트의 실제 회전이다.
        "tool_pos_rot": {"pos": tool, "rot": _vec(row, GUN_COLUMN, "Rotation")},
        "skeleton_joints": joints,
        "gun_metrics": {
            "speed": float(row["Speed (Real)"]),
            "distance": float(row["Distance (Real)"]),
            # 장면 좌표 그대로 둔다. FK 공간으로의 이동은 시퀀스 단위 상수로
            # `_register_impacts` 가 한 번에 한다 — 프레임마다 옮기면 reach 가 어긋난다.
            "raypoint": [float(row[column]) for column in RAY_COLUMNS],
            "_gun_scene": _vec(row, GUN_COLUMN, "Position"),
            "raytexmapnum": int(float(row["Ray Texture Map Number"])),
            "raytexuv": [float(row["Ray Texture UV X"]), float(row["Ray Texture UV Y"])],
            # 명세서 p.5: spraytime 은 누적 분사 시간(ms, integer)이다.
            "spraytime": int(round((seconds - span_start_sec) * 1000)),
        },
    }


def build_payload(
    rows: Sequence[dict[str, str]],
    span: tuple[int, int],
    *,
    session_id: str,
    sequence_id: str,
    max_frames: int,
) -> tuple[dict[str, Any], int]:
    """``(payload, 착탄점이 없어 건너뛴 프레임 수)``. 프레임은 연속으로 잘라 간격을 보존한다."""
    start, end = span
    span_start_sec = float(rows[start]["시간(초)"])
    # 바인드 오프셋은 프레임 불변이므로 세션 첫 행에서 한 번 읽는다.
    offsets = pose_fk_csv.bind_offsets(rows[0])
    frames: list[dict[str, Any]] = []
    skipped = 0
    for row in rows[start : end + 1]:
        frame = build_frame(row, len(frames), span_start_sec, offsets)
        if frame is None:
            skipped += 1
            continue
        frames.append(frame)
        if len(frames) >= max_frames:
            break
    _register_impacts(frames)
    # 명세서 p.5 의 최상위 8키는 전부 필수다. REPLAY- 접두로 재생분을 실물과 구별한다.
    payload = {
        "session_id": session_id,
        "user_id": session_id.split("_")[0] if "_" in session_id else session_id,
        "scenario_id": "1",
        "zone_id": "1",
        "sequence_id": sequence_id,
        "sequence_setting": {"tool_type": "spray_gun", "month": 9},
        "sequence_info": frames,
        "sequence_result": {},
    }
    return payload, skipped


def post(api_url: str, payload: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        api_url.rstrip("/") + TELEMETRY_PATH,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def find_session(name: str) -> Path:
    matches = sorted(CORPUS_ROOT.glob(f"**/{name}.csv"))
    if not matches:
        raise SystemExit(f"세션을 찾지 못했다: {name} (아래 --list 로 확인)")
    return matches[0]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    _ = parser.add_argument("--list", action="store_true", help="코퍼스 세션 이름을 나열한다")
    _ = parser.add_argument("--session", help="세션 CSV 이름(확장자 없이)")
    _ = parser.add_argument("--spans", type=int, default=2, help="보낼 트리거 구간 수")
    _ = parser.add_argument("--max-frames", type=int, default=90, help="구간당 프레임 상한(90 FPS 배치)")
    _ = parser.add_argument("--api-url", default="", help="비우면 전송하지 않는다")
    _ = parser.add_argument("--dry-run", action="store_true", help="페이로드를 파일로 쓰고 끝낸다")
    # 기본값을 임시 디렉터리로 둔다 — 공유 저장소 루트에 산출물을 남기지 않는다.
    _ = parser.add_argument("--out-dir", default=tempfile.gettempdir(), help="--dry-run 산출 위치")
    args = parser.parse_args(argv)

    if args.list:
        names = sorted(path.stem for path in CORPUS_ROOT.glob("**/*.csv"))
        print(f"{len(names)}개 세션")
        for name in names[:40]:
            print(" ", name)
        if len(names) > 40:
            print(f"  ... 그리고 {len(names) - 40}개 더")
        return 0

    if not args.session:
        parser.error("--session 또는 --list 가 필요하다")

    print("[합성 고지]", SYNTHESIS_NOTICE)
    print(
        "[한계] 검증되는 것 — 엔드포인트 배선·모니터·중복 식별·골격 게이트, 그리고 reach\n"
        "  버킷 선택·레퍼런스 조회·점수 산출까지. reach 는 CSV 장면 좌표에서 잰 실측값을\n"
        "  평행이동으로 옮긴 것이라 보존된다.\n"
        "  검증되지 않는 것 — tool_not_held(건 위치를 손에서 합성하므로 간격이 설계상 범위\n"
        "  안이다), 수평 방향이 걸리는 규칙(두 공간의 정합을 모르므로 수직 성분만 보증한다)."
    )
    path = find_session(args.session)
    rows = read_rows(path)
    spans = spray_spans(rows)
    print(f"세션 {path.name}: {len(rows)}행, 분사 구간 {len(spans)}개")

    sent = 0
    for order, span in enumerate(spans, start=1):
        if sent >= args.spans:
            break
        sequence_id = f"REPLAY-{args.session}-{order:03d}"
        payload, skipped = build_payload(
            rows, span, session_id=f"REPLAY-{args.session}",
            sequence_id=sequence_id, max_frames=args.max_frames,
        )
        frames = len(payload["sequence_info"])
        if frames == 0:
            print(f"  구간 {order}: 착탄점 있는 프레임이 없어 건너뜀")
            continue
        sent += 1
        joints = {tuple(frame["skeleton_joints"]["righthand"]) for frame in payload["sequence_info"]}
        print(f"  구간 {order}: {frames}프레임 (미명중 {skipped} 제외), 오른손 고유 위치 {len(joints)}개")

        if args.dry_run or not args.api_url:
            out = Path(args.out_dir) / f"replay-{sequence_id}.json"
            _ = out.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            print(f"    → {out}")
            continue

        data = (post(args.api_url, payload).get("data") or {})
        ack = data.get("sequence_data_ack") or {}
        pose = data.get("pose_feedback") or {}
        print(
            f"    ack stored={ack.get('stored')} duplicate={ack.get('duplicate')}"
            f" accepted={ack.get('frames_accepted')}"
        )
        print(
            f"    pose {pose.get('analysis_status')} reason={pose.get('reason_code')}"
            f" similarity={pose.get('similarity_score')} corrections={len(pose.get('corrections') or [])}"
        )
        if data.get("text_guide"):
            print(f"    guide {data['text_guide']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
