#!/usr/bin/env python3
"""Spark 운영 작업 잠금, 버전 번호 예약, 점검 알림(화면 배너용 파일과 Slack).

호스트 python3(3.12) 표준 라이브러리만 쓴다. 운영 폴더(OPS_DIR)에 이 파일과 health_sample.sh 를 같이 둔다.

  ops_lock.py take --what vllm --by KURE --reason "vLLM 교체" --until 10:30
  ops_lock.py release --by KURE                        # 잡은 사람만 푼다
  ops_lock.py release --by 정성오 --force --reason "담당자 연락 두절, 작업 끝난 것 확인"
  ops_lock.py status
  ops_lock.py check --by KURE --what vllm              # 교체 스크립트가 시작 전에 부른다. 자기 잠금이 아니면 1
  ops_lock.py version next --by KURE                   # 다음 패치 버전을 예약하고 출력한다
  ops_lock.py version reserve 3.9.0 --by KURE
  ops_lock.py watch --down "컨테이너 ..."              # health_sample.sh --check 가 부른다

--until 은 한국 시각이다. "HH:MM"(오늘) 또는 ISO 시각. 시간대가 없는 ISO 는 한국 시각으로 읽는다.
기한이 지나도 잠금은 저절로 풀리지 않는다. "만료"로 보이기만 한다.

경로는 환경변수로 바꿀 수 있다(시험용): OPS_DIR, OPS_MAINTENANCE_FILE, OPS_READY_URL, OPS_MODELS_URL,
OPS_AGENT_CONTAINER. Slack 웹훅 주소는 OPS_DIR/slack_webhook(권한 600)에서만 읽고, 어디에도 출력하지 않는다.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

KST = timezone(timedelta(hours=9))  # 한국은 서머타임이 없다
WHAT = ("agent-api", "vllm", "test", "other")
WHAT_LABEL = {"agent-api": "agent-api 교체", "vllm": "vLLM 교체", "test": "시험", "other": "운영 작업"}
IMPACT = {
    "agent-api": "API 잠시 끊김",
    "vllm": "상세 분석(27B) 일시 중단",
    "test": "시험 중(응답이 느릴 수 있음)",
    "other": "일부 기능이 잠시 불안정할 수 있음",
}
_VERSION = re.compile(r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)")


def ops_dir() -> Path:
    return Path(os.environ.get("OPS_DIR", "/home/keti/shipcoat/ops"))


def maintenance_file() -> Path:
    # agent-api 컨테이너에서는 /output/eval_results/ops/maintenance.json 으로 보인다.
    return Path(os.environ.get("OPS_MAINTENANCE_FILE", "/home/keti/shipcoat/results/ops/maintenance.json"))


def lock_file() -> Path:
    return ops_dir() / "lock.json"


def webhook_file() -> Path:
    return ops_dir() / "slack_webhook"


def _utc_iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_iso(text: str) -> datetime:
    moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
    return moment if moment.tzinfo else moment.replace(tzinfo=KST)


def _kst(text: str, now: datetime) -> str:
    moment = _parse_iso(text).astimezone(KST)
    same_day = moment.date() == now.astimezone(KST).date()
    return moment.strftime("%H:%M" if same_day else "%m-%d %H:%M")


def parse_until(text: str, now: datetime) -> datetime:
    """한국 시각 "HH:MM"(오늘) 또는 ISO. 이미 지난 시각은 받지 않는다."""
    match = re.fullmatch(r"([01]?[0-9]|2[0-3]):([0-5][0-9])", text.strip())
    if match:
        moment = now.astimezone(KST).replace(hour=int(match[1]), minute=int(match[2]), second=0, microsecond=0)
    else:
        try:
            moment = _parse_iso(text.strip())
        except ValueError:
            raise ValueError(f"--until 을 읽지 못했다: {text!r} (HH:MM 또는 ISO 시각, 한국 시각)") from None
    if moment <= now:
        raise ValueError(f"--until {text} 은 이미 지난 시각이다(지금 {now.astimezone(KST):%H:%M} 한국 시각)")
    return moment.astimezone(timezone.utc)


def warn(message: str) -> None:
    print(f"경고: {message}", file=sys.stderr)


def record(event: str, **fields: Any) -> None:
    """상태 변화 한 줄. O_APPEND 한 번의 write 라 동시에 써도 줄이 섞이지 않는다."""
    line = json.dumps({"at": _utc_iso(datetime.now(timezone.utc)), "event": event, **fields}, ensure_ascii=False)
    with open(ops_dir() / "lock-history.jsonl", "a", encoding="utf-8") as out:
        out.write(line + "\n")


def _write_json(path: Path, data: Any, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    with os.fdopen(fd, "w", encoding="utf-8") as out:
        json.dump(data, out, ensure_ascii=False, indent=2)
        out.write("\n")
    os.chmod(tmp, mode)
    os.replace(tmp, path)


def read_lock() -> dict[str, Any] | None:
    try:
        data = json.loads(lock_file().read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        return {"broken": True}
    return data if isinstance(data, dict) else {"broken": True}


def expired(lock: dict[str, Any], now: datetime) -> bool:
    try:
        return _parse_iso(str(lock["until"])) <= now
    except (KeyError, ValueError):
        return False


def describe(lock: dict[str, Any], now: datetime) -> str:
    if lock.get("broken"):
        return f"잠금 파일을 읽지 못했다({lock_file()}). 내용을 확인하고 지워야 한다."
    text = (f"{lock.get('what')} · 담당 {lock.get('by')} · ~{_kst(str(lock.get('until')), now)} KST"
            f" · {lock.get('reason')}")
    return text + (" · 만료(기한 지남, 자동으로 풀리지 않는다)" if expired(lock, now) else "")


def _note_expired(lock: dict[str, Any], now: datetime) -> None:
    """만료를 처음 본 때만 이력에 남긴다(cron 이 10분마다 보므로)."""
    if lock.get("broken") or not expired(lock, now):
        return
    try:
        lines = (ops_dir() / "lock-history.jsonl").read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        lines = []
    for line in reversed(lines):
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if entry.get("started_at") == lock.get("started_at"):
            if entry.get("event") == "expired":
                return
            break
    record("expired", what=lock.get("what"), by=lock.get("by"), until=lock.get("until"),
           started_at=lock.get("started_at"))


def write_maintenance(lock: dict[str, Any] | None) -> None:
    """화면 배너가 읽는 파일. 화면에 보이므로 담당 이름(by)은 넣지 않는다. 못 써도 잠금 동작은 이어 간다."""
    data: dict[str, Any] = {"active": False} if lock is None else {
        "active": True, "what": lock["what"], "reason": lock["reason"], "until": lock["until"],
        "started_at": lock["started_at"], "impact": IMPACT[lock["what"]],
    }
    try:
        _write_json(maintenance_file(), data)
    except OSError as error:
        warn(f"점검 배너 파일을 쓰지 못했다({maintenance_file()}, {type(error).__name__})")


def send_slack(text: str) -> bool:
    """실패해도 예외를 던지지 않는다. 주소는 출력하지 않는다."""
    path = webhook_file()
    try:
        url = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        warn(f"Slack 웹훅 파일이 없어 알림을 보내지 않았다({path})")
        return False
    except OSError as error:
        warn(f"Slack 웹훅 파일을 읽지 못했다({type(error).__name__})")
        return False
    if path.stat().st_mode & 0o077:
        warn(f"{path} 권한이 600 이 아니다. chmod 600 으로 바꾼다")
    try:
        request = urllib.request.Request(url, data=json.dumps({"text": text}).encode("utf-8"),
                                         headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(request, timeout=10) as response:
            response.read()
    except urllib.error.HTTPError as error:
        warn(f"Slack 전송 실패(HTTP {error.code})")
        return False
    except (OSError, ValueError) as error:  # URLError·시간 초과 포함. 메시지에 주소가 섞일 수 있어 종류만 쓴다
        warn(f"Slack 전송 실패({type(error).__name__})")
        return False
    return True


def take(what: str, by: str, reason: str, until: datetime, now: datetime) -> tuple[bool, dict[str, Any]]:
    lock = {"what": what, "by": by, "reason": reason, "until": _utc_iso(until), "started_at": _utc_iso(now),
            "host": socket.gethostname()}
    ops_dir().mkdir(parents=True, exist_ok=True)
    # 내용을 다 쓴 임시 파일을 link 로 건다. O_CREAT|O_EXCL 과 같이 원자적이고, 읽는 쪽이 빈 파일을 볼 틈이 없다.
    fd, tmp = tempfile.mkstemp(dir=ops_dir(), prefix=".lock.")
    with os.fdopen(fd, "w", encoding="utf-8") as out:
        json.dump(lock, out, ensure_ascii=False, indent=2)
    os.chmod(tmp, 0o644)
    try:
        os.link(tmp, lock_file())
    except FileExistsError:
        current = read_lock() or {"broken": True}
        _note_expired(current, now)
        return False, current
    finally:
        os.unlink(tmp)
    record("take", **{k: lock[k] for k in ("what", "by", "reason", "until", "started_at")})
    write_maintenance(lock)
    send_slack(f"[점검 시작] {WHAT_LABEL[what]} · {IMPACT[what]} · ~{_kst(lock['until'], now)} KST · 담당 {by}"
               f" · {reason}")
    return True, lock


def release(by: str, now: datetime, force: bool = False, reason: str = "") -> tuple[bool, str]:
    lock = read_lock()
    if lock is None:
        return False, "잡힌 잠금이 없다"
    if not force and (lock.get("broken") or lock.get("by") != by):
        return False, f"남의 잠금이라 풀지 않았다: {describe(lock, now)}. 꼭 풀어야 하면 --force --reason"
    lock_file().unlink(missing_ok=True)
    event = "force_release" if force else "release"
    record(event, what=lock.get("what"), by=lock.get("by"), released_by=by, reason=reason,
           started_at=lock.get("started_at"))
    write_maintenance(None)
    label = WHAT_LABEL.get(str(lock.get("what")), str(lock.get("what")))
    if force:
        send_slack(f"[잠금 강제 해제] {label} · 담당 {lock.get('by')} · 해제 {by} · 사유 {reason}")
    else:
        send_slack(f"[점검 끝] {label} · 담당 {by}")
    return True, f"잠금을 풀었다({event})"


def check(by: str, what: str, now: datetime) -> tuple[bool, str]:
    lock = read_lock()
    if lock is None:
        return False, f"잠금이 없다. 먼저: ops_lock.py take --what {what} --by {by} --reason ... --until HH:MM"
    _note_expired(lock, now)
    if lock.get("broken") or lock.get("by") != by or (what and lock.get("what") != what):
        return False, f"자기 잠금이 아니다: {describe(lock, now)}"
    if expired(lock, now):
        return False, f"자기 잠금이지만 기한이 지났다: {describe(lock, now)}. release 후 다시 take"
    return True, f"잠금 확인: {describe(lock, now)}"


# --- 버전 예약

def _vkey(version: str) -> tuple[int, ...]:
    if not _VERSION.fullmatch(version):
        raise ValueError(f"버전은 X.Y.Z 여야 한다: {version!r}")
    return tuple(int(part) for part in version.split("."))


def deployed_version() -> str | None:
    """운영 agent-api 이미지 라벨. docker 를 못 쓰면 None(장부만으로 계산한다)."""
    container = os.environ.get("OPS_AGENT_CONTAINER", "shipcoat-spark-agent-api-1")
    try:
        out = subprocess.run(
            ["docker", "inspect", "-f", '{{index .Config.Labels "org.opencontainers.image.version"}}', container],
            capture_output=True, text=True, timeout=30, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        warn(f"{container} 이미지 라벨을 읽지 못해 장부만으로 계산한다")
        return None
    return out if _VERSION.fullmatch(out) else None


def _with_ledger(change):
    """장부를 flock 아래에서 읽고 고쳐 쓴다. change(ledger) 가 돌려준 값을 그대로 돌려준다."""
    ops_dir().mkdir(parents=True, exist_ok=True)
    path = ops_dir() / "versions.json"
    with open(ops_dir() / "versions.lock", "a") as guard:
        fcntl.flock(guard, fcntl.LOCK_EX)
        ledger = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
        result = change(ledger)
        _write_json(path, ledger)
        return result


def next_version(by: str, now: datetime) -> str:
    deployed = deployed_version()

    def change(ledger: list[dict[str, Any]]) -> str:
        known = [entry["version"] for entry in ledger] + ([deployed] if deployed else [])
        if not known:
            raise ValueError("장부도 운영 이미지 라벨도 없어 다음 버전을 정할 수 없다. version reserve X.Y.Z 로 시작한다")
        major, minor, patch = max(map(_vkey, known))
        version = f"{major}.{minor}.{patch + 1}"
        ledger.append({"version": version, "by": by, "at": _utc_iso(now)})
        return version

    return _with_ledger(change)


def reserve_version(version: str, by: str, now: datetime) -> str:
    _vkey(version)
    deployed = deployed_version()

    def change(ledger: list[dict[str, Any]]) -> str:
        for entry in ledger:
            if entry["version"] == version:
                raise ValueError(f"{version} 은 이미 {entry['by']} 가 {entry['at']} 에 예약했다")
        if deployed == version:
            raise ValueError(f"{version} 은 지금 운영 중인 버전이다")
        ledger.append({"version": version, "by": by, "at": _utc_iso(now)})
        return version

    return _with_ledger(change)


# --- 운영 상태 감시 (health_sample.sh --check)

def _probe() -> dict[str, Any]:
    ready_url = os.environ.get("OPS_READY_URL", "http://127.0.0.1:58000/api/ready")
    models_url = os.environ.get("OPS_MODELS_URL", "http://127.0.0.1:18000/v1/models")
    try:
        with urllib.request.urlopen(ready_url, timeout=10) as response:
            ready_code = response.status
    except urllib.error.HTTPError as error:
        ready_code = error.code
    except (OSError, ValueError):
        ready_code = None
    try:
        with urllib.request.urlopen(models_url, timeout=10) as response:
            models = [item["id"] for item in json.load(response)["data"]]
        model = ",".join(sorted(models)) or None
    except (OSError, ValueError, KeyError, TypeError):
        model = None
    return {"ready": ready_code == 200, "ready_code": ready_code, "model": model, "containers": _containers()}


def _containers() -> dict[str, str]:
    """운영 컨테이너 이름 → "이미지(ID 앞 12자)". 잠금 없이 바뀐 것을 잡는 데 쓴다. 감싼 docker 를 거치지 않는다."""
    names = os.environ.get("OPS_PROD_CONTAINERS", "shipcoat-spark-agent-api-1 shipcoat-spark-mcp-1 "
                           "shipcoat-spark-vllm-1 shipcoat-spark-quick-vllm-1 shipcoat-spark-web-1").split()
    try:
        out = subprocess.run(["/usr/bin/docker", "ps", "-a", "--no-trunc", "--format", "{{.Names}}\t{{.Image}}\t{{.ID}}"],
                             capture_output=True, text=True, timeout=30, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    found = {}
    for line in out.splitlines():
        name, image, cid = (line.split("\t") + ["", ""])[:3]
        if name in names:
            found[name] = f"{image}({cid[:12]})"
    return found


def changes(previous: dict[str, Any], current: dict[str, Any]) -> list[str]:
    """직전과 달라진 것만 문장으로. 직전 기록이 없으면 비교하지 않는다."""
    if not previous:
        return []
    found = []
    if previous.get("ready") and not current["ready"]:
        found.append(f"agent-api 준비 안 됨(/api/ready {current.get('ready_code') or '응답 없음'})")
    elif not previous.get("ready") and current["ready"]:
        found.append("agent-api 준비 복구(/api/ready 200)")
    before, after = previous.get("model"), current["model"]
    if before != after:
        if before and after:
            found.append(f"모델 바뀜: {before} → {after}")
        elif before:
            found.append(f"vLLM 모델 응답 없음(직전 {before})")
        else:
            found.append(f"vLLM 모델 응답 복구: {after}")
    before_ids, after_ids = previous.get("containers") or {}, current.get("containers") or {}
    for name in sorted(set(before_ids) & set(after_ids)):
        if before_ids[name] != after_ids[name]:
            found.append(f"컨테이너 교체: {name} {before_ids[name]} → {after_ids[name]}")
    was, now_down = set(previous.get("down", [])), set(current["down"])
    if now_down - was:
        found.append("운영 컨테이너 중단: " + ", ".join(sorted(now_down - was)))
    if was - now_down:
        found.append("운영 컨테이너 복귀: " + ", ".join(sorted(was - now_down)))
    return found


def watch(down: list[str], now: datetime, probe=_probe) -> str | None:
    current = {**probe(), "down": sorted(down)}
    state_path = ops_dir() / "watch-state.json"
    try:
        previous = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        previous = {}
    found = changes(previous, current)
    message = None
    sent = True
    if found:
        lock = read_lock()
        unlocked = lock is None and any(text.startswith(("컨테이너 교체", "모델 바뀜")) for text in found)
        message = ("[경고] 잠금 없이 운영 변경 · " if unlocked else "[운영 상태] ") + " · ".join(found)
        if lock:
            _note_expired(lock, now)
            message += f" · 예정된 작업 중(잠금: {describe(lock, now)})"
        sent = send_slack(message)
    # 웹훅이 있는데 전송만 실패했으면 상태를 넘기지 않아 다음 회차에 다시 보낸다.
    if sent or not webhook_file().exists():
        _write_json(state_path, current)
    return message


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("take")
    p.add_argument("--what", choices=WHAT, required=True)
    p.add_argument("--by", required=True)
    p.add_argument("--reason", required=True)
    p.add_argument("--until", required=True, help="한국 시각 HH:MM(오늘) 또는 ISO")
    p = sub.add_parser("release")
    p.add_argument("--by", required=True)
    p.add_argument("--force", action="store_true")
    p.add_argument("--reason", default="")
    sub.add_parser("status")
    p = sub.add_parser("check")
    p.add_argument("--by", required=True)
    p.add_argument("--what", choices=WHAT, help="생략하면 종류는 보지 않는다(감싼 docker 명령이 쓴다)")
    p = sub.add_parser("version")
    vsub = p.add_subparsers(dest="action", required=True)
    vsub.add_parser("next").add_argument("--by", required=True)
    v = vsub.add_parser("reserve")
    v.add_argument("version")
    v.add_argument("--by", required=True)
    p = sub.add_parser("watch")
    p.add_argument("--down", default="", help="꺼져 있는 운영 컨테이너 이름(공백 구분)")
    args = parser.parse_args(argv)
    now = datetime.now(timezone.utc)

    try:
        if args.command == "take":
            ok, lock = take(args.what, args.by.strip(), args.reason.strip(), parse_until(args.until, now), now)
            print(("잠금을 잡았다: " if ok else "이미 잡혀 있다: ") + describe(lock, now))
            if ok:
                print(f'운영 컨테이너를 바꾸는 docker 명령 전에: export OPS_LOCK_BY="{lock["by"]}"')
            return 0 if ok else 1
        if args.command == "release":
            if args.force and not args.reason.strip():
                parser.error("--force 에는 --reason 이 필요하다")
            ok, text = release(args.by.strip(), now, args.force, args.reason.strip())
            print(text)
            return 0 if ok else 1
        if args.command == "status":
            lock = read_lock()
            if lock:
                _note_expired(lock, now)
            print(describe(lock, now) if lock else "잠금 없음")
            return 0
        if args.command == "check":
            ok, text = check(args.by.strip(), args.what, now)
            print(text, file=sys.stdout if ok else sys.stderr)
            return 0 if ok else 1
        if args.command == "version":
            if args.action == "next":
                print(next_version(args.by.strip(), now))
            else:
                print(reserve_version(args.version, args.by.strip(), now))
            return 0
        message = watch(args.down.split(), now)
        if message:
            print(message)
        return 0
    except ValueError as error:
        print(f"오류: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
