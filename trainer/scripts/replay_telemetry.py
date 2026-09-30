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

**그래서 이 드라이버로는 점수가 나오지 않는다(실측).** 골격 게이트는 통과한다 — 뼈 길이
일정(전완 0.2430 m, sd 0.00000), 높이 0.9528 m, 손↔건 0.3585 m, 오른손 고유 위치 81/90.
그런데 `reach_m`이 **4.5286**이 되어(착탄 3.955 장면좌표 − 건 −1.076 FK공간) 레퍼런스에
해당 버킷이 없고 `reference_quality: "unavailable"`, `support: null`이 된다. 실물 SEQ-002는
reach −0.0322, band −0.10, support 14245였다.

**원인은 한 필드가 두 규칙에 서로 다른 공간을 요구하는 것이다.** `tool_pos_rot.pos`는
R5(손↔건 간격)에서 **골격과 같은 공간**이어야 하고, `reach` 버킷 선택에서는 착탄점과
**같은 장면 공간**이어야 한다. CSV는 관절이 local이라 두 조건을 한 값으로 만족시킬 수
없다. global 형식(실물 샘플)에서는 전부 한 공간이라 모순이 없다.

**따라서 이 드라이버가 검증하는 것과 못 하는 것.**

* 검증한다 — 엔드포인트 배선, 모니터 기록, 중복 식별, 골격 게이트(움직임·뼈·높이·간격)
* 못 한다 — `reach` 버킷 선택, 따라서 **점수 자체**, `tool_not_held`, 분사각

점수 경로를 열려면 **이동하며 분사한 global 캡처**가 필요하다. CSV 코퍼스는 local이고
FK 공간과 장면 공간의 정합(registration)을 우리가 갖고 있지 않다.

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
    "관절은 pose_fk_csv.forward_kinematics(루트 원점)로 만들었다 — CSV Position 은 local 이다.\n"
    f"  tool_pos_rot.pos 는 전완을 이어 {HAND_TOOL_GAP_M} m 앞에 합성했다(FK 공간과 장면 좌표가 다르다).\n"
    "  따라서 tool_not_held · reach · 분사각은 이 입력으로 검증되지 않는다."
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
    joints = {name.lower(): list(pos) for name, pos in fk.items()}
    hand = joints.get("righthand")
    forearm = joints.get("rightforearm")
    if hand is None or forearm is None:
        return None
    return {
        "index": index,
        "time": stamp.strftime("%Y-%m-%dT%H:%M:%S.") + f"{stamp.microsecond // 1000:03d}Z",
        # pos 는 합성(전완 연장), rot 은 CSV 건 오브젝트의 실제 회전이다.
        "tool_pos_rot": {"pos": _tool_from_hand(hand, forearm), "rot": _vec(row, GUN_COLUMN, "Rotation")},
        "skeleton_joints": joints,
        "gun_metrics": {
            "speed": float(row["Speed (Real)"]),
            "distance": float(row["Distance (Real)"]),
            "raypoint": [float(row[column]) for column in RAY_COLUMNS],
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
        "[한계] 이 입력으로는 점수가 나오지 않는다 — reach 가 FK 공간과 장면 좌표를 섞어\n"
        "  레퍼런스 버킷이 선택되지 않는다(reference_quality: unavailable). 검증되는 것은\n"
        "  엔드포인트 배선·모니터·중복 식별·골격 게이트까지다."
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
