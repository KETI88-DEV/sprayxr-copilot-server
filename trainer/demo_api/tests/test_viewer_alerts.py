"""v2 뷰어 재생 알림 — 공개 이상 알림 SSE와 같은 판정 함수를 재생 기록에 붙였는지 본다."""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from trainer.demo_api import telemetry_router
from trainer.demo_api.anomaly_stream import AnomalyBroadcaster
from trainer.demo_api.motion_viewer import create_motion_viewer_router, viewer_v2_document
from trainer.demo_api.sequence_analysis import SequenceAnalysisEngine
from trainer.demo_api.sequence_store import SequenceStore
from trainer.demo_api.session_analysis import csv_sequences, read_csv
from trainer.demo_api.viewer_alerts import sequence_alerts_for
from trainer.scripts.build_zone_guidance_profiles import build_profiles
from trainer.zone_reference import ZoneReferenceCatalog

_GUN = "한길숏건"
_FIELDS = [
    "인덱스", "시간(초)", "분사",
    *(f"{_GUN} {kind} {axis}" for kind in ("Position", "Rotation") for axis in "XYZ"),
    "Ray point X", " Ray Point Y", " Ray Point Z",
    "Ray Texture Map Number", "Ray Texture UV X", "Ray Texture UV Y", "Zone ID",
    "Speed (Real)", "Distance (Real)", "Distance (Input)",
    *(f"Pelvis Position {axis} (World)" for axis in "XYZ"),
]
_TRAINEE = "trainee_a"


def _write_strokes(path: Path, strokes: list[tuple[str, str]]) -> None:
    """분사마다 4행(0.1초 간격) + 쉼 2행. 방향 h는 UV x로, v는 UV y로 움직인다. 분사 n은 (n-1)*0.6초에 시작한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_FIELDS)
        writer.writeheader()
        index = 0
        for zone, direction in strokes:
            for step in range(6):
                on = step < 4
                u, v = (step * 40, 0) if direction == "h" else (0, step * 40)
                writer.writerow({
                    "인덱스": index, "시간(초)": round(index * 0.1, 3), "분사": "On" if on else "Off",
                    f"{_GUN} Position X": 0.3 + index * 0.01, f"{_GUN} Position Y": 1.2, f"{_GUN} Position Z": 0.2,
                    f"{_GUN} Rotation X": 0, f"{_GUN} Rotation Y": 0, f"{_GUN} Rotation Z": 0,
                    "Ray point X": int(zone) + step * 0.05, " Ray Point Y": 1.4 + (v / 1000), " Ray Point Z": 2.0,
                    "Ray Texture Map Number": 1, "Ray Texture UV X": 200 * int(zone) + u, "Ray Texture UV Y": 300 + v,
                    "Zone ID": zone if on else "-1",
                    "Speed (Real)": 300, "Distance (Real)": 250, "Distance (Input)": 300,
                    "Pelvis Position X (World)": 0, "Pelvis Position Y (World)": 1, "Pelvis Position Z (World)": 0,
                })
                index += 1


def _write_height_map(path: Path, raw_um: float) -> None:
    """Learn04 1번 Zone 안만 원시 두께 raw_um 인 TSB RG16 높이맵(test_motion_viewer 와 같은 부호화)."""
    mask = np.asarray(ZoneReferenceCatalog.default().zone_mask("Learn04", 1, "1", size=(2048, 2048))) > 0
    code = int(np.rint(raw_um / 20000.0 * 65025.0))
    rgba = np.zeros((2048, 2048, 4), dtype=np.uint8)
    rgba[mask] = (code // 255, code % 255, 255, 255)
    stream = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(stream, format="PNG")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(stream.getvalue())


_MODE4 = Path("raw/spray/learning/mode_04")
# 전문가 둘 다 Zone 1 → 2, 수평. 훈련생은 Zone 1 수직(방향) → Zone 2 → Zone 1 다시(순서).
_EXPERT_STROKES = [("1", "h"), ("2", "h")]
_TRAINEE_STROKES = [("1", "v"), ("2", "h"), ("1", "h")]


def _client(tmp_path: Path, *, trainee_um: float | None = None, expert_um: float | None = None) -> TestClient:
    _write_strokes(tmp_path / _MODE4 / "4_1001_1.csv", _EXPERT_STROKES)
    _write_strokes(tmp_path / _MODE4 / "4_1002_1.csv", _EXPERT_STROKES)
    if expert_um is not None:
        _write_height_map(tmp_path / _MODE4 / "Map01_1001_1_1_Height.png", expert_um)
    worker = tmp_path / "workers" / _TRAINEE
    _write_strokes(worker / _MODE4 / "4_2001_1.csv", _TRAINEE_STROKES)
    if trainee_um is not None:
        _write_height_map(worker / _MODE4 / "Map01_2001_1_1_Height.png", trainee_um)
    (worker / "worker.json").write_text(json.dumps({"display_name": "훈련생", "role": "trainee"}), encoding="utf-8")
    build_profiles(tmp_path, process="spray")
    build_profiles(worker, process="spray", legacy_expert_zone_ids=False)
    app = FastAPI()
    app.include_router(create_motion_viewer_router(tmp_path, frame_count=40, mesh_provider=lambda _mode: {"available": False}))
    return TestClient(app)


def _get(client: TestClient, path: str, worker: str = _TRAINEE, session: str = "4_2001_1", mode: int = 4) -> dict[str, Any]:
    params: dict[str, Any] = {"process": "spray", "mode_id": mode, "session_id": session}
    if worker:
        params["worker_id"] = worker
    response = client.get(path, params=params)
    assert response.status_code == 200, response.text
    return response.json()


def test_sequence_alerts_match_the_public_anomaly_stream(tmp_path: Path) -> None:
    client = _client(tmp_path)
    viewer = _get(client, "/motion-viewer/api/alerts")["sequence_alerts"]
    assert viewer["status"] == "available"

    broadcaster = AnomalyBroadcaster()
    app = FastAPI()
    app.include_router(telemetry_router.create_telemetry_router(
        store=SequenceStore(":memory:"), sequence_engine=SequenceAnalysisEngine(tmp_path), broadcaster=broadcaster))
    public = TestClient(app)
    rows = read_csv((tmp_path / "workers" / _TRAINEE / _MODE4 / "4_2001_1.csv").read_bytes())
    for body in csv_sequences(rows, session_id="S-1", user_id="U-1", scenario_id="4"):
        assert public.post("/api/v1/copilot/analyze-sequence", json=body).status_code == 200
    published = {(e.sequence_id, e.anomaly_type, e.payload["reason_code"]) for e in broadcaster.since("S-1", last_event_id=0)}

    shown = {(a["sequence_id"], a["anomaly_type"], a["reason_code"]) for a in viewer["alerts"]}
    assert shown == published == {
        ("SEQ-001", "direction_deviation", "direction_deviation"),
        ("SEQ-003", "sequence_order_deviation", "zone_revisit"),
    }
    by_id = {a["sequence_id"]: a for a in viewer["alerts"]}
    # 게이트 전 판정 문장(공개 SSE는 여기에 도막 진행 문장을 씌운다).
    assert by_id["SEQ-001"]["message"] == "1번 구역은 수평 방향으로 도장하세요."
    # 공개 SSE message 도 같은 한국어 방향 이름이다(4차 결정: 사람이 읽는 문장에 영어 열쇠를 싣지 않는다).
    sse = {e.sequence_id: e.payload["message"] for e in broadcaster.since("S-1", last_event_id=0)
           if e.anomaly_type == "direction_deviation"}
    assert sse["SEQ-001"].startswith("1번 구역은 수평 방향으로 도장하세요.") and "horizontal" not in sse["SEQ-001"]
    assert by_id["SEQ-003"]["message"] == "전문가 권장 Zone 순서를 확인하세요."   # 남은 Zone이 없어 추천이 없다
    assert all(a["basis"] == "sequence" and a["zone_id"] == "1" and a["map_num"] == "1" for a in viewer["alerts"])
    # 뷰어는 방향 문장 대신 판정에 쓴 기대·관측 방향 열쇠를 값으로 받아 한국어로 옮긴다. 열거값은 바꾸지 않는다.
    assert by_id["SEQ-001"]["values"] == {"expected_direction": "horizontal", "observed_direction": "vertical"}
    assert by_id["SEQ-003"]["values"] == {}


def test_alert_times_follow_the_spray_runs_on_the_playback_clock(tmp_path: Path) -> None:
    client = _client(tmp_path, trainee_um=300)
    motion = _get(client, "/motion-viewer/api/motion")
    alerts = sorted([*motion["coating_alerts"], *_get(client, "/motion-viewer/api/alerts")["sequence_alerts"]["alerts"]],
                    key=lambda a: a["at_seconds"])
    timing = {(a["sequence_id"], a["anomaly_type"]): (a["seek_seconds"], a["at_seconds"]) for a in alerts}
    assert timing[("SEQ-001", "direction_deviation")] == pytest.approx((0.0, 0.3))
    assert timing[("SEQ-003", "sequence_order_deviation")] == pytest.approx((1.2, 1.5))
    # 도막 알림은 그 Zone을 마지막으로 칠한 분사(SEQ-003)의 끝이다.
    assert timing[("SEQ-003", "coating_over_app")] == pytest.approx((1.2, 1.5))
    assert all(0 <= a["seek_seconds"] <= a["at_seconds"] <= motion["duration_seconds"] for a in alerts)
    assert [a["at_seconds"] for a in motion["coating_alerts"]] == sorted(a["at_seconds"] for a in motion["coating_alerts"])
    # 재생 시간축(프레임 t)과 같은 기준: 첫 분사 시작이 프레임 0초다.
    assert motion["frames"][0]["t"] == 0.0


@pytest.mark.parametrize(("raw_um", "grade", "expected"), [
    (300, "above", "coating_over_app"), (100, "below", "coating_under_app"), (181, "normal", None)])
def test_coating_alerts_are_the_public_function_applied_to_the_panel_rows(
    tmp_path: Path, raw_um: float, grade: str, expected: str | None,
) -> None:
    motion = _get(_client(tmp_path, trainee_um=raw_um), "/motion-viewer/api/motion")
    target = motion["coating_evaluation"]["target"]
    assert target["analysis_status"] == "available"
    zone_one = next(row for row in target["zones"] if row["zone_id"] == "1")
    assert zone_one["grade"] == grade

    wanted = []
    for row in target["zones"]:
        for alert in telemetry_router.coating_alerts({**row, "left_zone": row}):
            wanted.append((alert["anomaly_type"], alert["payload"]["reason_code"], alert["payload"]["message"],
                           {key: alert["payload"][key] for key in ("grade", "thickness_um", "target_thickness_um", "target_basis", "status")}))
    shown = [(a["anomaly_type"], a["reason_code"], a["message"], a["values"]) for a in motion["coating_alerts"]]
    assert shown == wanted
    zone_one_types = [a["anomaly_type"] for a in motion["coating_alerts"] if a["zone_id"] == "1"]
    assert zone_one_types == ([expected] if expected else [])
    assert all(a["basis"] == "final_height_map" and a["map_num"] == "1" for a in motion["coating_alerts"])


def test_a_record_that_follows_the_expert_has_no_alerts_but_was_judged(tmp_path: Path) -> None:
    client = _client(tmp_path, expert_um=181)
    sequence = _get(client, "/motion-viewer/api/alerts", worker="", session="4_1001_1")["sequence_alerts"]
    motion = _get(client, "/motion-viewer/api/motion", worker="", session="4_1001_1")
    assert sequence == {"status": "available", "reason_code": None, "alerts": []}
    assert motion["coating_evaluation"]["target"]["analysis_status"] == "available"
    assert motion["coating_alerts"] == []


def test_mode_two_record_is_judged_with_canonical_zone_ids(tmp_path: Path) -> None:
    # 원래 번호 3 → 1 은 canonical 1 → 3 이다(학습모드 2 전문가 번호 교정). 원래 번호로 판정하면 순서 이탈이 나온다.
    mode2 = tmp_path / "raw/spray/learning/mode_02"
    _write_strokes(mode2 / "2_1001_1.csv", [("3", "h"), ("1", "h")])
    _write_strokes(mode2 / "2_1002_1.csv", [("3", "h"), ("1", "h")])
    _write_strokes(mode2 / "2_1003_1.csv", [("3", "h"), ("1", "v")])
    build_profiles(tmp_path, process="spray")
    app = FastAPI()
    app.include_router(create_motion_viewer_router(tmp_path, frame_count=40, mesh_provider=lambda _mode: {"available": False}))
    client = TestClient(app)

    alerts = _get(client, "/motion-viewer/api/alerts", worker="", session="2_1003_1", mode=2)["sequence_alerts"]["alerts"]
    motion = _get(client, "/motion-viewer/api/motion", worker="", session="2_1003_1", mode=2)
    assert [(a["sequence_id"], a["anomaly_type"], a["zone_id"]) for a in alerts] == [("SEQ-002", "direction_deviation", "3")]
    painted = {p["zone_id"] for p in motion["paint_trace"] if p["t"] >= alerts[0]["seek_seconds"]}
    assert painted == {"3"}


def test_unavailable_reasons(tmp_path: Path) -> None:
    path = tmp_path / "4_2001_1.csv"
    _write_strokes(path, _TRAINEE_STROKES)
    assert sequence_alerts_for(path, process="blasting", mode_id=4, legacy=False, engine=SequenceAnalysisEngine(tmp_path)) == {
        "status": "unavailable", "reason_code": "process_not_supported", "alerts": []}
    assert sequence_alerts_for(path, process="spray", mode_id=4, legacy=False, engine=SequenceAnalysisEngine(tmp_path)) == {
        "status": "unavailable", "reason_code": "sequence_reference_unavailable", "alerts": []}
    empty = tmp_path / "4_2002_1.csv"
    empty.write_text(",".join(_FIELDS) + "\n", encoding="utf-8")
    assert sequence_alerts_for(empty, process="spray", mode_id=4, legacy=False, engine=SequenceAnalysisEngine(tmp_path))[
        "reason_code"] == "no_sequences"


def test_engine_error_is_reported_and_not_cached(tmp_path: Path) -> None:
    path = tmp_path / "4_2001_1.csv"
    _write_strokes(path, _TRAINEE_STROKES)

    class Broken:
        calls = 0

        def decide(self, *_args: Any) -> Any:
            Broken.calls += 1
            raise RuntimeError("boom")

    engine = Broken()
    for _ in range(2):
        assert sequence_alerts_for(path, process="spray", mode_id=4, legacy=False, engine=engine) == {
            "status": "unavailable", "reason_code": "engine_error", "alerts": []}
    assert Broken.calls == 2


def test_v2_page_has_alert_surfaces_and_loads_the_alert_script_first(tmp_path: Path) -> None:
    client = _client(tmp_path)
    page = viewer_v2_document()
    assert 'id="v2Toast"' in page and 'aria-live="polite"' in page.split('id="v2Toast"', 1)[1].split(">", 1)[0]
    assert 'id="v2AlertMarks"' in page and 'id="v2AlertsButton"' in page and 'id="v2AlertList"' in page
    assert page.index("/motion-viewer/assets/motion_viewer_alerts.js") < page.index("/motion-viewer/assets/motion_viewer_v2.js")
    assert 'aria-atomic="false"' in page and "atEnd" in page   # 새 토스트만 읽히고, 끝 시각을 한 번 그린 뒤 감는다
    script = client.get("/motion-viewer/assets/motion_viewer_alerts.js")
    assert script.status_code == 200 and "motionViewerAlerts" in script.text
    # 화면은 판정하지 않는다: 두께 문턱·등급 규칙 낱말이 알림 스크립트에 없다.
    assert "160" not in script.text and "1.5" not in script.text


def test_alert_times_subtract_a_leading_empty_row(tmp_path: Path) -> None:
    client = _client(tmp_path, trainee_um=300)
    path = tmp_path / "workers" / _TRAINEE / _MODE4 / "4_2001_1.csv"
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["시간(초)"] = str(round(float(row["시간(초)"]) + 0.5, 3))
    # 관절·트래커·착탄이 모두 빈 첫 행: 재생 0초는 둘째 행이다.
    rows.insert(0, {**{key: "" for key in rows[0]}, "인덱스": "0", "시간(초)": "0", "분사": "Off", "Zone ID": "-1"})
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    motion = _get(client, "/motion-viewer/api/motion")
    assert motion["frames"][0]["t"] == 0.0
    alerts = [*motion["coating_alerts"], *_get(client, "/motion-viewer/api/alerts")["sequence_alerts"]["alerts"]]
    timing = {(a["sequence_id"], a["anomaly_type"]): (a["seek_seconds"], a["at_seconds"]) for a in alerts}
    assert timing[("SEQ-001", "direction_deviation")] == pytest.approx((0.0, 0.3))


def test_a_failing_coating_alert_does_not_break_playback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(tmp_path, trainee_um=300)

    def boom(*_args: Any, **_kwargs: Any) -> list[Any]:
        raise OverflowError("cannot convert float infinity to integer")

    monkeypatch.setattr("trainer.demo_api.viewer_alerts.coating_alerts_for", boom)
    motion = _get(client, "/motion-viewer/api/motion")
    assert motion["coating_alerts"] == [] and motion["frames"]
