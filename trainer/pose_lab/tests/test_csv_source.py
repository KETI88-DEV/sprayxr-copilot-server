#!/usr/bin/env python3
# pyright: reportAny=false, reportUnusedCallResult=false

from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from trainer.pose_lab import csv_source, service
from trainer.pose_lab.app import create_app

PARKED = (-694.61, 701.29, 698.25)
STILL = (-2.99, 4.66, -0.27)

# 벤더 export의 철자 그대로다. 앞 공백이 붙은 두 컬럼이 이 도구의 상시 지뢰였다.
RAY_COLUMNS = ("Ray point X", " Ray Point Y", " Ray Point Z")
OBJECT_GROUPS = ("카메라", "플래쉬", "한길숏건", "한길폴건", "반도숏건", "반도폴건")


def _header() -> list[str]:
    columns = ["인덱스", "시간(초)", "분사", *RAY_COLUMNS, "Ray Texture UV X", "Ray Texture UV Y"]
    for group in OBJECT_GROUPS:
        columns.extend(f"{group} Position {axis}" for axis in "XYZ")
    return columns


def _row(index: int, hit: tuple[float, float, float] | None) -> dict[str, str]:
    """착탄점을 주면 분사 중인 프레임, 주지 않으면 분사가 끊긴 프레임이다."""
    row = {"인덱스": str(index), "시간(초)": f"{index * 0.0111:.4f}"}
    if hit is None:
        row["분사"] = "Off"
        row.update({column: "" for column in RAY_COLUMNS})
        row["Ray Texture UV X"] = ""
        row["Ray Texture UV Y"] = ""
        anchor = (1.2, 4.0, 0.8)
    else:
        row["분사"] = "On"
        for column, value in zip(RAY_COLUMNS, hit):
            row[column] = f"{value:.4f}"
        row["Ray Texture UV X"] = "1024"
        row["Ray Texture UV Y"] = "512"
        anchor = hit

    places = {
        # 세 실측점은 착탄점을 따라간다. 머리-손 거리는 마스킹 임계 안이다.
        "카메라": (anchor[0] - 0.7, anchor[1], anchor[2]),
        "한길숏건": (anchor[0] - 0.35, anchor[1], anchor[2] - 0.15),
        "플래쉬": (anchor[0] - 0.53, anchor[1], anchor[2] - 0.01),
        "한길폴건": PARKED,
        "반도폴건": PARKED,
        # 정상으로 보이는 위치에 std=0으로 고정된 건. 분산으로만 걸러진다.
        "반도숏건": STILL,
    }
    for group, point in places.items():
        for axis, value in zip("XYZ", point):
            row[f"{group} Position {axis}"] = f"{value:.4f}"
    return row


def _session_rows(shift: float = 0.0) -> list[dict[str, str]]:
    """왕복 수평 버스트 하나 + 쉬는 구간 + 수직 버스트 하나.

    왕복은 버스트 하나 안에서 방향이 뒤집힌다. 버스트를 그대로 패스로 쓰면 왕복 전체에
    직선을 맞추게 되므로 착탄점 궤적에서 잘려야 한다.
    """
    rows: list[dict[str, str]] = []
    index = 0
    for step in range(30):
        rows.append(_row(index, (1.2, 4.0 + step * 0.02, 0.8 + shift)))
        index += 1
    for step in range(30):
        rows.append(_row(index, (1.2, 4.58 - step * 0.02, 0.8 + shift)))
        index += 1
    for _ in range(5):
        rows.append(_row(index, None))
        index += 1
    for step in range(30):
        rows.append(_row(index, (1.2, 4.3, 0.6 + step * 0.01 + shift)))
        index += 1
    return rows


def _write(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_header())
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """실제 배치를 그대로 흉내 낸다 — 평평한 5인과 한 단계 깊은 전형기."""
    flat = tmp_path / "김경호_20250831" / "김경호_20250831(Sequences)"
    _write(flat / "10_1000399_1.csv", _session_rows())
    # 실전모드다. 레벨 토큰이 숫자가 아니라 범위 밖이다.
    _write(flat / "1-A_1000400_1.csv", _session_rows())
    nested = tmp_path / "전형기" / "전형기_20250901" / "전형기_20250901(Sequences)"
    _write(nested / "10_1000451_1.csv", _session_rows(shift=0.3))
    _write(nested / "3_1000444_1.csv", _session_rows())
    return tmp_path


def test_workers_and_levels_come_from_filenames(root: Path) -> None:
    """목록은 CSV를 열지 않고 만든다. 세션 하나가 수만 행이라 목록에 여는 순간 못 쓴다."""
    source = csv_source.CsvPassSource(root)
    assert source.workers() == ["김경호", "전형기"]
    assert source.levels("전형기") == ["3", "10"]
    assert source._rows == {}


def test_live_mode_sessions_are_out_of_scope(root: Path) -> None:
    """`1-A`/`1-B`/`1-C`는 실전모드다. 학습모드 레벨 1..10만 훑는다."""
    source = csv_source.CsvPassSource(root)
    assert [item.level for item in source.sessions() if item.worker == "김경호"] == ["10"]


def test_bursts_are_split_on_the_hit_path(root: Path) -> None:
    """버스트가 아니라 패스가 단위다. 왕복 버스트는 두 패스로 갈려야 한다."""
    source = csv_source.CsvPassSource(root)
    session = next(item for item in source.sessions() if item.worker == "김경호")
    entries = source.index_of(session)
    assert len(entries) >= 3
    spans = [(entry.row_start, entry.row_end) for entry in entries]
    # 왕복 버스트(0..59) 안에서 경계가 생겨야 한다. 하나면 잘리지 않은 것이다.
    inside = [span for span in spans if span[1] <= 59]
    assert len(inside) >= 2
    assert all(span[0] < span[1] for span in spans)


def test_pass_index_carries_a_motion_key_for_the_requested_params(root: Path) -> None:
    """키 문자열을 색인에 굳히지 않는다. 판정 경계가 요청마다 달라진다."""
    import motion_key

    source = csv_source.CsvPassSource(root)
    session = next(item for item in source.sessions() if item.worker == "김경호")
    entries = source.index_of(session)
    coarse = {source.key_of(entry, motion_key.KeyParams(uv_grid=1)) for entry in entries}
    fine = {source.key_of(entry, motion_key.KeyParams(uv_grid=8)) for entry in entries}
    assert all(key.startswith("m1:") for key in coarse | fine)
    assert all(":0x0:" in key for key in coarse)
    # 수직 버스트가 vertical_ 대역으로 들어간다. 부호가 살아 있어야 한다.
    assert any(":vertical_up:" in key for key in fine)


def test_load_returns_only_the_rows_of_that_pass(root: Path) -> None:
    source = csv_source.CsvPassSource(root)
    session = next(item for item in source.sessions() if item.worker == "김경호")
    entries = source.index_of(session)
    ref = service.PassRef(
        worker=session.worker,
        date=session.date,
        level=session.level,
        session=session.session,
        pass_index=1,
    )
    raw = source.load(ref)
    span = entries[1]
    assert len(raw.rows) == span.row_end - span.row_start + 1
    assert int(raw.rows[0]["인덱스"]) == span.row_start
    # 잘라낸 행만으로 다시 프레임을 만들 수 있어야 한다. 활성 건 판별이 여기서 깨지면
    # 목록에는 있는데 비교하면 "재지 못했다"가 나오는 패스가 된다.
    assert len(service.pass_frames(raw)) >= 2


def test_missing_pass_is_a_lookup_error_not_an_empty_pass(root: Path) -> None:
    """없는 패스를 빈 행으로 내면 "프레임이 모자라다"로 둔갑한다."""
    source = csv_source.CsvPassSource(root)
    session = next(item for item in source.sessions() if item.worker == "김경호")
    ref = service.PassRef(
        worker=session.worker,
        date=session.date,
        level=session.level,
        session=session.session,
        pass_index=9999,
    )
    with pytest.raises(LookupError):
        source.load(ref)
    with pytest.raises(LookupError):
        source.load(service.PassRef("없음", "20250101", "10", "x_1", 0))


def test_rows_are_read_once_per_session(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """요청마다 전수 스캔하면 화면이 못 쓰게 된다. 같은 세션을 다시 읽지 않는다."""
    reads: list[Path] = []
    original = csv_source.simulator_csv.read_rows

    def counted(path: Path) -> list[dict[str, str]]:
        reads.append(path)
        return original(path)

    monkeypatch.setattr(csv_source.simulator_csv, "read_rows", counted)
    source = csv_source.CsvPassSource(root)
    session = next(item for item in source.sessions() if item.worker == "김경호")
    source.index_of(session)
    source.index_of(session)
    ref = service.PassRef(
        worker=session.worker,
        date=session.date,
        level=session.level,
        session=session.session,
        pass_index=0,
    )
    source.load(ref)
    assert len(reads) == 1


def test_rows_cache_does_not_grow_without_bound(root: Path) -> None:
    """세션 하나가 250 MB 규모다. 무제한 캐시는 서버를 먹는다."""
    source = csv_source.CsvPassSource(root, rows_cache=1)
    for item in source.sessions():
        source.index_of(item)
    assert len(source._rows) == 1


def test_compare_runs_end_to_end_over_the_csv_tree(root: Path) -> None:
    """실데이터 경로가 통째로 도는지 본다. 같은 패스끼리면 이탈이 0이어야 한다."""
    source = csv_source.CsvPassSource(root)
    session = next(item for item in source.sessions() if item.worker == "김경호")
    ref = service.PassRef(
        worker=session.worker,
        date=session.date,
        level=session.level,
        session=session.session,
        pass_index=0,
    )
    body = service.compare(source, ref, ref)
    assert body["status"] == "ok"
    measured = body["measured"]
    assert isinstance(measured, dict)
    assert math.isclose(float(measured["score"]), 1.0, abs_tol=1e-9)
    assert measured["corrections"] == []
    assert body["samples"] == service.stroke_frame.DEFAULT_SAMPLES


def test_passes_endpoint_offers_a_choice(root: Path) -> None:
    client = TestClient(create_app(source=csv_source.CsvPassSource(root)))
    body = client.get("/api/passes", params={"worker": "전형기", "level": "10"}).json()
    assert body["status"] == "ok"
    assert body["levels"] == ["3", "10"]
    assert body["sessions_total"] == 1
    assert body["truncated"] is False
    first = body["passes"][0]
    assert first["worker"] == "전형기"
    assert first["date"] == "20250901"
    assert first["level"] == "10"
    assert first["session"] == "1000451_1"
    assert first["pass_index"] == 0
    assert first["motion_key"].startswith("m1:")
    assert first["frames"] > 0


def test_passes_endpoint_says_when_it_stopped_early(root: Path) -> None:
    """훑은 세션 수를 숨기면 "이 작업자에게 이 패스뿐"으로 읽힌다."""
    client = TestClient(create_app(source=csv_source.CsvPassSource(root)))
    body = client.get("/api/passes", params={"worker": "전형기", "limit": 1}).json()
    assert body["sessions_total"] == 2
    assert body["sessions_scanned"] == 1
    assert body["truncated"] is True


def test_missing_pass_becomes_its_own_reason(root: Path) -> None:
    client = TestClient(create_app(source=csv_source.CsvPassSource(root)))
    ref = {
        "worker": "전형기",
        "date": "20250901",
        "level": "10",
        "session": "1000451_1",
        "pass_index": 9999,
    }
    body = client.post("/api/compare", json={"query": ref, "reference": ref}).json()
    assert body["status"] == "unavailable"
    assert body["reason"] == "pass_not_found"


def test_env_root_wires_the_source(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POSE_LAB_DATA_ROOT", str(root))
    client = TestClient(create_app())
    assert client.get("/api/workers").json()["workers"] == ["김경호", "전형기"]


def test_no_env_root_keeps_the_previous_behaviour(monkeypatch: pytest.MonkeyPatch) -> None:
    """루트가 없으면 지금까지와 똑같이 돈다. 기존 테스트가 이 거동 위에 서 있다."""
    monkeypatch.delenv("POSE_LAB_DATA_ROOT", raising=False)
    client = TestClient(create_app())
    assert client.get("/api/workers").json()["workers"] == []
    assert client.get("/api/passes", params={"worker": "A"}).json()["reason"] == "no_source"
