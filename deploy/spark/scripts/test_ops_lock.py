from __future__ import annotations

import json
import subprocess
import threading
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from deploy.spark.scripts import ops_lock
from deploy.spark.scripts.ops_lock_guard import require_lock

_REAL_SEND_SLACK = ops_lock.send_slack

NOW = datetime(2026, 11, 17, 0, 0, tzinfo=timezone.utc)  # 한국 09:00
UNTIL = NOW + timedelta(hours=1, minutes=30)  # 한국 10:30


@pytest.fixture(autouse=True)
def ops(tmp_path, monkeypatch):
    monkeypatch.setenv("OPS_DIR", str(tmp_path / "ops"))
    monkeypatch.setenv("OPS_MAINTENANCE_FILE", str(tmp_path / "results" / "ops" / "maintenance.json"))
    monkeypatch.setattr(ops_lock, "deployed_version", lambda: "3.8.8")
    sent: list[str] = []
    monkeypatch.setattr(ops_lock, "send_slack", lambda text: sent.append(text) or True)
    return sent


def _history() -> list[dict]:
    path = ops_lock.ops_dir() / "lock-history.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _maintenance() -> dict:
    return json.loads(ops_lock.maintenance_file().read_text(encoding="utf-8"))


def test_parse_until_reads_kst_and_rejects_past():
    assert ops_lock.parse_until("10:30", NOW) == UNTIL
    assert ops_lock.parse_until("2026-11-17T10:30", NOW) == UNTIL
    assert ops_lock.parse_until("2026-11-17T01:30:00Z", NOW) == UNTIL
    with pytest.raises(ValueError, match="지난"):
        ops_lock.parse_until("08:59", NOW)
    with pytest.raises(ValueError, match="읽지 못했다"):
        ops_lock.parse_until("내일 아침", NOW)


def test_concurrent_take_has_exactly_one_winner(ops):
    barrier = threading.Barrier(8)
    results: list[bool] = []

    def contender(index: int) -> None:
        barrier.wait()
        ok, _ = ops_lock.take("vllm", f"dev{index}", "교체", UNTIL, NOW)
        results.append(ok)

    threads = [threading.Thread(target=contender, args=(i,)) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(results) == [False] * 7 + [True]
    winner = ops_lock.read_lock()
    assert [e["by"] for e in _history() if e["event"] == "take"] == [winner["by"]]
    assert len(ops) == 1 and f"담당 {winner['by']}" in ops[0]
    assert not [p for p in ops_lock.ops_dir().iterdir() if p.name.startswith(".lock.")]


def test_take_writes_banner_file_without_the_name_and_slack_with_it(ops):
    ok, _ = ops_lock.take("vllm", "KURE", "vLLM 교체", UNTIL, NOW)
    assert ok
    assert _maintenance() == {"active": True, "what": "vllm", "reason": "vLLM 교체", "until": "2026-11-17T01:30:00Z",
                              "started_at": "2026-11-17T00:00:00Z", "impact": "상세 분석(27B) 일시 중단"}
    assert ops == ["[점검 시작] vLLM 교체 · 상세 분석(27B) 일시 중단 · ~10:30 KST · 담당 KURE · vLLM 교체"]


def test_second_take_shows_holder_and_cli_exits_1(capsys):
    ops_lock.take("agent-api", "KURE", "이미지 교체", UNTIL, NOW)
    code = ops_lock.main(["take", "--what", "test", "--by", "정성오", "--reason", "시험", "--until", "23:59"])
    out = capsys.readouterr().out
    assert code == 1 and "agent-api · 담당 KURE" in out and "이미 잡혀 있다" in out
    assert ops_lock.read_lock()["by"] == "KURE"


def test_release_by_someone_else_is_refused_and_force_needs_reason(ops):
    ops_lock.take("vllm", "KURE", "교체", UNTIL, NOW)
    ok, text = ops_lock.release("정성오", NOW)
    assert not ok and "남의 잠금" in text and ops_lock.read_lock()["by"] == "KURE"
    with pytest.raises(SystemExit):
        ops_lock.main(["release", "--by", "정성오", "--force"])
    assert ops_lock.read_lock() is not None
    ok, _ = ops_lock.release("정성오", NOW, force=True, reason="연락 두절, 작업 끝남 확인")
    assert ok and ops_lock.read_lock() is None
    last = _history()[-1]
    assert last["event"] == "force_release" and last["by"] == "KURE" and last["released_by"] == "정성오"
    assert _maintenance() == {"active": False}
    assert ops[-1].startswith("[잠금 강제 해제]") and "사유 연락 두절" in ops[-1]


def test_owner_release_clears_lock_and_banner(ops):
    ops_lock.take("test", "KURE", "부하 시험", UNTIL, NOW)
    ok, _ = ops_lock.release("KURE", NOW)
    assert ok and ops_lock.read_lock() is None and _maintenance() == {"active": False}
    assert [e["event"] for e in _history()] == ["take", "release"]
    assert ops[-1] == "[점검 끝] 시험 · 담당 KURE"


def test_expired_lock_is_shown_but_kept_and_logged_once():
    ops_lock.take("vllm", "KURE", "교체", UNTIL, NOW)
    later = UNTIL + timedelta(minutes=5)
    ok, current = ops_lock.take("test", "정성오", "시험", later + timedelta(hours=1), later)
    assert not ok and "만료" in ops_lock.describe(current, later)
    ok, text = ops_lock.check("KURE", "vllm", later)
    assert not ok and "기한이 지났다" in text
    assert ops_lock.read_lock()["by"] == "KURE"
    assert [e["event"] for e in _history()] == ["take", "expired"]


def test_check_passes_only_for_own_unexpired_lock_of_that_kind():
    assert not ops_lock.check("KURE", "vllm", NOW)[0]
    ops_lock.take("vllm", "KURE", "교체", UNTIL, NOW)
    assert ops_lock.check("KURE", "vllm", NOW)[0]
    assert not ops_lock.check("KURE", "agent-api", NOW)[0]
    assert not ops_lock.check("정성오", "vllm", NOW)[0]


def test_version_next_is_monotonic_over_ledger_and_deployed_label(monkeypatch):
    assert ops_lock.next_version("KURE", NOW) == "3.8.9"
    assert ops_lock.next_version("정성오", NOW) == "3.8.10"
    monkeypatch.setattr(ops_lock, "deployed_version", lambda: "3.9.4")  # 장부 밖에서 올라간 운영 이미지
    assert ops_lock.next_version("KURE", NOW) == "3.9.5"
    monkeypatch.setattr(ops_lock, "deployed_version", lambda: None)  # docker 를 못 읽어도 장부로
    assert ops_lock.next_version("KURE", NOW) == "3.9.6"
    ledger = json.loads((ops_lock.ops_dir() / "versions.json").read_text(encoding="utf-8"))
    assert [e["version"] for e in ledger] == ["3.8.9", "3.8.10", "3.9.5", "3.9.6"]


def test_version_next_under_contention_never_repeats():
    got: list[str] = []
    threads = [threading.Thread(target=lambda: got.append(ops_lock.next_version("dev", NOW))) for _ in range(12)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(got, key=ops_lock._vkey) == [f"3.8.{n}" for n in range(9, 21)]


def test_version_reserve_rejects_duplicates_deployed_and_malformed(capsys):
    assert ops_lock.reserve_version("3.10.0", "KURE", NOW) == "3.10.0"
    assert ops_lock.main(["version", "reserve", "3.10.0", "--by", "정성오"]) == 2
    assert "이미 KURE" in capsys.readouterr().err
    with pytest.raises(ValueError, match="운영 중"):
        ops_lock.reserve_version("3.8.8", "KURE", NOW)
    with pytest.raises(ValueError, match="X.Y.Z"):
        ops_lock.reserve_version("v3.10.1", "KURE", NOW)
    assert ops_lock.next_version("KURE", NOW) == "3.10.1"


def test_changes_report_only_transitions():
    up = {"ready": True, "ready_code": 200, "model": "A", "down": []}
    assert ops_lock.changes({}, up) == []
    assert ops_lock.changes(up, dict(up)) == []
    broken = {"ready": False, "ready_code": 503, "model": "B", "down": ["shipcoat-spark-vllm-1"]}
    assert ops_lock.changes(up, broken) == [
        "agent-api 준비 안 됨(/api/ready 503)", "모델 바뀜: A → B", "운영 컨테이너 중단: shipcoat-spark-vllm-1"]
    assert ops_lock.changes(broken, {**up, "model": None}) == [
        "agent-api 준비 복구(/api/ready 200)", "vLLM 모델 응답 없음(직전 B)", "운영 컨테이너 복귀: shipcoat-spark-vllm-1"]


def test_watch_sends_once_per_change_and_notes_the_lock(ops):
    state = {"ready": True, "ready_code": 200, "model": "A"}
    probe = lambda: dict(state)  # noqa: E731
    assert ops_lock.watch([], NOW, probe) is None  # 첫 회는 기준만 남긴다
    state.update(ready=False, ready_code=503)
    ops_lock.take("agent-api", "KURE", "이미지 교체", UNTIL, NOW)
    message = ops_lock.watch([], NOW, probe)
    assert message == ("[운영 상태] agent-api 준비 안 됨(/api/ready 503)"
                       " · 예정된 작업 중(잠금: agent-api · 담당 KURE · ~10:30 KST · 이미지 교체)")
    assert ops[-1] == message
    assert ops_lock.watch([], NOW, probe) is None and ops[-1] == message  # 같은 상태면 다시 보내지 않는다


def test_watch_retries_when_webhook_exists_but_sending_failed(monkeypatch):
    state = {"ready": True, "ready_code": 200, "model": "A"}
    ops_lock.watch([], NOW, lambda: dict(state))
    ops_lock.webhook_file().write_text("https://hooks.slack.com/services/x\n")
    monkeypatch.setattr(ops_lock, "send_slack", lambda text: False)
    assert ops_lock.watch(["shipcoat-spark-web-1"], NOW, lambda: dict(state))
    monkeypatch.setattr(ops_lock, "send_slack", lambda text: True)
    assert "운영 컨테이너 중단" in ops_lock.watch(["shipcoat-spark-web-1"], NOW, lambda: dict(state))


# --- 실제 send_slack (가짜 서버). 진짜 Slack 으로는 보내지 않는다.

class _Hook(BaseHTTPRequestHandler):
    bodies: list[dict] = []
    status = 200

    def do_POST(self):  # noqa: N802
        _Hook.bodies.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
        self.send_response(_Hook.status)
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, *args):
        pass


def test_missing_webhook_only_warns_and_lock_still_succeeds(monkeypatch, capsys):
    monkeypatch.setattr(ops_lock, "send_slack", _REAL_SEND_SLACK)
    ok, _ = ops_lock.take("vllm", "KURE", "교체", UNTIL, NOW)
    assert ok and ops_lock.read_lock()["by"] == "KURE" and _maintenance()["active"] is True
    assert "웹훅 파일이 없어" in capsys.readouterr().err


def test_unwritable_banner_file_only_warns(tmp_path, monkeypatch, capsys):
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("")
    monkeypatch.setenv("OPS_MAINTENANCE_FILE", str(blocker / "maintenance.json"))
    ok, _ = ops_lock.take("vllm", "KURE", "교체", UNTIL, NOW)
    assert ok and ops_lock.read_lock()["by"] == "KURE"
    assert "점검 배너 파일을 쓰지 못했다" in capsys.readouterr().err


def test_send_slack_posts_json_and_never_prints_the_url(monkeypatch, capsys):
    monkeypatch.setattr(ops_lock, "send_slack", _REAL_SEND_SLACK)
    server = HTTPServer(("127.0.0.1", 0), _Hook)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    ops_lock.ops_dir().mkdir(parents=True)
    hook = ops_lock.webhook_file()
    hook.write_text(f"http://127.0.0.1:{server.server_port}/services/SECRET\n")
    hook.chmod(0o600)
    try:
        assert ops_lock.send_slack("[점검 시작] 시험")
        assert _Hook.bodies[-1] == {"text": "[점검 시작] 시험"}
        _Hook.status = 500
        assert not ops_lock.send_slack("다시")
        hook.chmod(0o644)
        _Hook.status = 200
        assert ops_lock.send_slack("권한 경고")
    finally:
        _Hook.status = 200
        server.shutdown()
    err = capsys.readouterr().err
    assert "HTTP 500" in err and "권한이 600" in err
    assert "SECRET" not in err and str(server.server_port) not in err


def test_guard_stops_unless_own_lock(monkeypatch):
    monkeypatch.setenv("OPS_LOCK", str(Path(ops_lock.__file__)))
    with pytest.raises(SystemExit, match="교체를 시작하지 않는다"):
        require_lock("KURE", "vllm")
    now = datetime.now(timezone.utc)
    ops_lock.take("vllm", "KURE", "교체", now + timedelta(hours=1), now)
    require_lock("KURE", "vllm")
    with pytest.raises(SystemExit):
        require_lock("정성오", "vllm")


def test_health_sample_script_parses():
    subprocess.run(["bash", "-n", str(Path(ops_lock.__file__).with_name("health_sample.sh"))], check=True)


def test_container_replaced_without_lock_is_a_warning(ops):
    before = {"ready": True, "ready_code": 200, "model": "A", "containers": {"shipcoat-spark-web-1": "web:1(aaa)"}}
    after = {**before, "containers": {"shipcoat-spark-web-1": "web:2(bbb)"}}
    state = dict(before)
    probe = lambda: dict(state)  # noqa: E731
    ops_lock.watch([], NOW, probe)
    state.update(after)
    message = ops_lock.watch([], NOW, probe)
    assert message == "[경고] 잠금 없이 운영 변경 · 컨테이너 교체: shipcoat-spark-web-1 web:1(aaa) → web:2(bbb)"
    ops_lock.take("other", "KURE", "웹 교체", UNTIL, NOW)
    state.update(before)
    assert ops_lock.watch([], NOW, probe).startswith("[운영 상태] 컨테이너 교체")


def test_check_without_kind_accepts_any_own_lock():
    ops_lock.take("vllm", "KURE", "교체", UNTIL, NOW)
    assert ops_lock.check("KURE", None, NOW)[0]
    assert not ops_lock.check("V3", None, NOW)[0]


def _guard(tmp_path, *args, by=None):
    import os
    fake = tmp_path / "fake-docker"
    fake.write_text('#!/bin/sh\necho "REAL $*"\n')
    fake.chmod(0o755)
    env = {**os.environ, "DOCKER_REAL": str(fake), "OPS_LOCK": str(Path(ops_lock.__file__))}
    env.pop("OPS_LOCK_BY", None)
    if by:
        env["OPS_LOCK_BY"] = by
    script = Path(ops_lock.__file__).with_name("docker_guard.sh")
    return subprocess.run(["bash", str(script), *args], capture_output=True, text=True, env=env)


def test_docker_guard_blocks_production_changes_without_own_lock(tmp_path):
    assert _guard(tmp_path, "ps", "-a").stdout == "REAL ps -a\n"
    blocked = _guard(tmp_path, "stop", "shipcoat-spark-web-1")
    assert blocked.returncode == 3 and "운영 잠금 없이" in blocked.stderr and "REAL" not in blocked.stdout
    assert _guard(tmp_path, "stop", "my-test-box").returncode == 0  # 운영 이름이 아니면 통과
    now = datetime.now(timezone.utc)
    ops_lock.take("other", "KURE", "웹 재시작", now + timedelta(hours=1), now)
    assert _guard(tmp_path, "restart", "shipcoat-spark-web-1", by="V3").returncode == 3  # 남의 잠금
    assert _guard(tmp_path, "restart", "shipcoat-spark-web-1", by="KURE").stdout == "REAL restart shipcoat-spark-web-1\n"
    assert _guard(tmp_path, "compose", "-p", "shipcoat-spark", "up", "-d").returncode == 3
