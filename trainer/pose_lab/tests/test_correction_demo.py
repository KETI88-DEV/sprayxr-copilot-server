"""교정 시연(`correction_demo.py`, `/api/demo`, `static/demo.*`) — 화면의 숫자와 몸이 KPI 측정과 같은 계산인가.

KPI 시험 방법(`docs/year2/`, `docs/year3/`)이 이 화면을 시연 절차로 쓴다. 화면이 다른 숫자를 내거나 몸이 안내와
다르게 움직이면 문서의 시연과 측정이 어긋난다.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from trainer.pose_lab import app as app_module
from trainer.pose_lab import correction_demo as cd
from trainer.pose_lab import service
from trainer.pose_lab.service import PassRef
from trainer.pose_lab.tests import test_scoring_isolation as isolation

STATIC = Path(__file__).resolve().parents[1] / "static"
REAL = [n for n in isolation.CASE_NAMES if not n.startswith("synthetic")]


def _real(name: str) -> tuple[Any, Any, PassRef, PassRef]:
    case = next(c for c in isolation._cases() if c["name"] == name)
    source, kwargs = isolation._source_for(case)
    return source, kwargs["reference_spreads"], PassRef(**case["query"]), PassRef(**case["reference"])


def _measured(source: Any, spreads: Any, query: PassRef, reference: PassRef) -> dict[str, Any]:
    return service.compare(source, query, reference, reference_spreads=spreads, include_view=False)["measured"]


@pytest.mark.parametrize("name", REAL)
def test_default_demo_is_the_kpi_algorithm(name: str) -> None:
    """화면 기본 방식은 교정 알고리즘이 발부한 안내 전부 — `score_after_all`·`improvement_all`(KPI 정의)과 같다.

    깨뜨리는 법: 기본 방식이 1건만 옮기면 `score_after`가 나와 실패한다.
    """
    source, spreads, query, reference = _real(name)
    demo = cd.build_demo(source, query, reference, spreads)
    measured = _measured(source, spreads, query, reference)
    assert demo["mode"] == "issued"
    assert demo["score_before"] == pytest.approx(measured["score"], abs=1e-12)
    assert demo["score_after"] == pytest.approx(measured["score_after_all"], abs=1e-12)
    assert demo["improvement"] == pytest.approx(measured["improvement_all"]["same"], abs=1e-12)
    assert demo["improvement_split"] == pytest.approx(measured["improvement_all"]["split"], abs=1e-12)
    steps = demo["display"]["score_steps"]
    assert steps[0] == demo["display"]["score_before"] and steps[-1] == demo["display"]["score_after"]


@pytest.mark.parametrize("name", REAL)
def test_hidden_delivered_mode_matches_the_main_screen(name: str) -> None:
    """숨김 방식 `delivered`(1건)는 메인 화면의 예상 결과(`score_after`)와 같다."""
    source, spreads, query, reference = _real(name)
    demo = cd.build_demo(source, query, reference, spreads, "delivered")
    measured = _measured(source, spreads, query, reference)
    assert demo["values"] == 1
    assert demo["score_after"] == pytest.approx(measured["score_after"], abs=1e-12)


@pytest.mark.parametrize("name", REAL)
def test_each_arrow_is_the_spoken_amount(name: str) -> None:
    """화살표는 안내 문장의 cm 그대로다. 몸이 팔 길이로 못 미치면 그 거리는 `reach_short_cm`로 따로 나온다."""
    from trainer.year3 import demo_mode

    demo_mode.register()
    source, spreads, query, reference = _real(name)
    for mode in ("issued", "delivered", demo_mode.MODE):
        demo = cd.build_demo(source, query, reference, spreads, mode)
        assert demo["cues"], mode
        for cue in demo["cues"]:
            moved_cm = math.dist(cue["from"], cue["to"]) * 100.0
            assert moved_cm == pytest.approx(cue["cm"], abs=0.05), (mode, cue["text"])


@pytest.mark.parametrize("name", REAL)
def test_no_bone_ever_changes_length(name: str) -> None:
    """**모든** 뼈(가슴-어깨·쇄골·골반-허벅지 포함)의 길이가 모든 단계와 목표 몸에서 원래와 같다 — 사람이 늘어나지 않는다.

    팔 세 마디만 보던 옛 검사는 가슴-왼쪽 어깨가 35.8cm 늘어난 것을 놓쳤다(사용자가 영상에서 찾았다, 2026-10-01).
    깨뜨리는 법: 안내 없는 손을 원래 자리로 되돌리거나 어깨를 손 쪽으로 밀어 옮기면 실패한다.
    """
    from trainer.year3 import demo_mode

    demo_mode.register()
    source, spreads, query, reference = _real(name)
    for mode in ("issued", "delivered", demo_mode.MODE):
        demo = cd.build_demo(source, query, reference, spreads, mode)
        worker = demo["bodies"]["worker"]
        bodies = [("target", demo["bodies"]["target"])] + [(f"step{k}", s) for k, s in enumerate(demo["bodies"]["steps"])]
        for label, body in bodies:
            for parent, child in demo["bones"]:
                if parent in worker and child in worker:
                    before = math.dist(worker[parent], worker[child])
                    assert math.dist(body[parent], body[child]) == pytest.approx(before, abs=1e-3), (mode, label, parent, child)
        assert demo["bodies"]["steps"][0] == worker


def _correction(point: str, unit: tuple[float, float, float], cm: float) -> Any:
    return isolation.measured_score.AxisCorrection(point=point, axis="vertical", magnitude_cm=cm, direction="",
                                                   sigma=1.0, unit=unit)


def _standing_body() -> dict[str, tuple[float, float, float]]:
    return {
        "Pelvis": (0.0, 1.0, 0.0), "Chest": (0.0, 1.4, 0.0), "Neck": (0.0, 1.55, 0.0), "Head": (0.0, 1.65, 0.0),
        "RightShoulder": (0.1, 1.45, 0.0), "RightUpperArm": (0.2, 1.45, 0.0), "RightForeArm": (0.2, 1.17, 0.05),
        "RightHand": (0.2, 0.92, 0.1),
        "LeftShoulder": (-0.1, 1.45, 0.0), "LeftUpperArm": (-0.2, 1.45, 0.0), "LeftForeArm": (-0.2, 1.17, 0.05),
        "LeftHand": (-0.2, 0.92, 0.1),
        "LeftThigh": (-0.1, 0.95, 0.0), "LeftCalf": (-0.1, 0.52, 0.06), "LeftFoot": (-0.1, 0.08, 0.0),
        "LeftToe": (-0.1, 0.0, 0.12),
        "RightThigh": (0.1, 0.95, 0.0), "RightCalf": (0.1, 0.52, 0.06), "RightFoot": (0.1, 0.08, 0.0),
        "RightToe": (0.1, 0.0, 0.12),
    }


def test_lowering_the_stance_keeps_the_feet_planted_and_bends_the_knees() -> None:
    """"자세를 낮추세요"(머리 수직)는 발을 디딘 채 골반 위가 내려가고 무릎이 굽는다 — 몸이 바닥 밑으로 꺼지지 않는다.

    깨뜨리는 법: 수직 이동을 몸 전체에 주면 발이 바닥 아래로 내려가 실패한다.
    """
    body = _standing_body()
    lowered = cd.corrected_body(body, [_correction("head", (0.0, -1.0, 0.0), 10.0)])
    assert lowered["LeftFoot"] == body["LeftFoot"] and lowered["RightToe"] == body["RightToe"]
    assert lowered["Head"][1] == pytest.approx(body["Head"][1] - 0.10)
    assert lowered["Pelvis"][1] == pytest.approx(body["Pelvis"][1] - 0.10)
    assert math.dist(lowered["LeftThigh"], lowered["LeftCalf"]) == pytest.approx(
        math.dist(body["LeftThigh"], body["LeftCalf"]), abs=1e-6)
    assert math.dist(lowered["LeftCalf"], lowered["LeftFoot"]) == pytest.approx(
        math.dist(body["LeftCalf"], body["LeftFoot"]), abs=1e-6)


def test_stepping_moves_the_whole_body_and_a_cued_hand_goes_where_the_cue_says() -> None:
    """"옮겨 서세요"(머리 수평)는 몸 전체가 옮겨 간다. 안내 없는 손은 몸을 따라가고, 안내 있는 손은 안내한 곳에 간다."""
    body = _standing_body()
    stepped = cd.corrected_body(body, [_correction("head", (1.0, 0.0, 0.0), 20.0)])
    assert stepped["LeftFoot"][0] == pytest.approx(body["LeftFoot"][0] + 0.20)
    assert stepped["Head"][0] == pytest.approx(body["Head"][0] + 0.20)
    assert stepped["RightHand"][0] == pytest.approx(body["RightHand"][0] + 0.20)
    cued = cd.corrected_body(body, [_correction("head", (1.0, 0.0, 0.0), 20.0), _correction("right", (0.0, 1.0, 0.0), 5.0)])
    target = (body["RightHand"][0], body["RightHand"][1] + 0.05, body["RightHand"][2])
    assert cued["RightHand"] == pytest.approx(target, abs=1e-6)


def test_an_unreachable_hand_stops_short_instead_of_stretching_the_body() -> None:
    """손을 팔 길이 밖으로 보내라는 안내는 어깨를 돌려 최대한 뻗고, 모자란 거리를 숨기지 않는다."""
    body = _standing_body()
    far = [_correction("right", (0.0, 0.0, 1.0), 120.0)]
    out = cd.corrected_body(body, far)
    for parent, child in (("RightShoulder", "RightUpperArm"), ("RightUpperArm", "RightForeArm"), ("RightForeArm", "RightHand")):
        assert math.dist(out[parent], out[child]) == pytest.approx(math.dist(body[parent], body[child]), abs=1e-6)
    short = cd.point_shortfall_cm(body, out, far, "right")
    assert short > 10.0
    target = (body["RightHand"][0], body["RightHand"][1], body["RightHand"][2] + 1.2)
    assert math.dist(out["RightHand"], target) * 100.0 == pytest.approx(short)


def test_standing_up_never_lifts_the_feet() -> None:
    """다리 길이를 넘게 "자세를 높이라"고 해도 발은 바닥에 남는다 — 다리가 다 펴진 데서 멈추고 못 미친 높이를 낸다.

    사용자가 영상에서 양발이 지면에서 떨어지는 것을 찾았다(2026-10-01). 깨뜨리는 법: 높이 제한을 빼면 발이 떠서 실패한다.
    """
    body = _standing_body()
    up = [_correction("head", (0.0, 1.0, 0.0), 60.0)]
    out = cd.corrected_body(body, up)
    for foot in ("LeftFoot", "LeftToe", "RightFoot", "RightToe"):
        assert out[foot] == body[foot]
    for parent, child in (("LeftThigh", "LeftCalf"), ("LeftCalf", "LeftFoot")):
        assert math.dist(out[parent], out[child]) == pytest.approx(math.dist(body[parent], body[child]), abs=1e-6)
    leg = math.dist(body["LeftThigh"], body["LeftCalf"]) + math.dist(body["LeftCalf"], body["LeftFoot"])
    assert math.dist(out["LeftThigh"], out["LeftFoot"]) <= leg * cd.LEG_REACH_MAX + 1e-6
    short = cd.point_shortfall_cm(body, out, up, "head")
    assert 0.0 < short < 60.0


@pytest.mark.parametrize("name", REAL)
def test_shortfall_is_reported_per_cue_not_accumulated(name: str) -> None:
    """못 미친 거리는 그 안내의 방향으로만 센다 — 앞 안내의 부족분이 뒤 안내에 다시 붙지 않는다.

    옮겨 서기(머리 수평)는 몸 전체가 정확히 옮겨 가므로 부족분이 0이어야 한다.
    """
    from trainer.year3 import demo_mode

    demo_mode.register()
    source, spreads, query, reference = _real(name)
    demo = cd.build_demo(source, query, reference, spreads, demo_mode.MODE)
    for cue in demo["cues"]:
        if cue["point"] == "head" and cue["axis"] != "vertical":
            assert cue["reach_short_cm"] == pytest.approx(0.0, abs=0.05), cue["text"]


@pytest.mark.parametrize("name", REAL)
def test_feet_stay_on_the_floor_in_every_demo_body(name: str) -> None:
    """실데이터 시연의 모든 단계·목표 몸에서 발이 원래 높이보다 뜨지 않는다(옮겨 서기의 수평 이동만 있다)."""
    from trainer.year3 import demo_mode

    demo_mode.register()
    source, spreads, query, reference = _real(name)
    for mode in ("issued", demo_mode.MODE):
        demo = cd.build_demo(source, query, reference, spreads, mode)
        worker = demo["bodies"]["worker"]
        for body in [demo["bodies"]["target"], *demo["bodies"]["steps"]]:
            for foot in ("LeftFoot", "LeftToe", "RightFoot", "RightToe"):
                assert body[foot][1] == pytest.approx(worker[foot][1], abs=1e-4), (mode, foot)


def test_a_hand_without_a_cue_moves_with_the_body() -> None:
    """안내가 없는 손은 몸이 낮아질 때 같이 내려간다(원래 자리로 끌어올리지 않는다)."""
    body = _standing_body()
    lowered = cd.corrected_body(body, [_correction("head", (0.0, -1.0, 0.0), 10.0)])
    assert lowered["LeftHand"][1] == pytest.approx(body["LeftHand"][1] - 0.10)
    assert lowered["RightHand"][1] == pytest.approx(body["RightHand"][1] - 0.10)


def test_unknown_mode_is_refused() -> None:
    source, spreads, query, reference = _real(REAL[0])
    assert cd.build_demo(source, query, reference, spreads, "nope") == {"status": "unavailable", "reason": "unknown_mode"}


def test_demo_api_lists_only_the_visible_mode_and_rule_picked_scenarios() -> None:
    """화면 목록에는 기본 방식만 나온다 — 1건·6개 방식은 숨김(주소로만)이다."""
    client = TestClient(app_module.create_app(source=None))
    body = client.get("/api/demo/scenarios").json()
    assert [m["mode"] for m in body["modes"]] == ["issued"]
    assert {"delivered", "six_values"} <= set(cd.MODES)
    assert len(body["scenarios"]) == 6
    assert all(s["rule"].startswith("안내 종류별") for s in body["scenarios"])
    refused = client.post("/api/demo", json={"query": body["scenarios"][0]["query"],
                                             "reference": body["scenarios"][0]["reference"]}).json()
    assert refused == {"status": "unavailable", "reason": "no_source"}


def test_demo_screen_shows_no_year_and_formats_no_numbers() -> None:
    """시연 화면에는 연차를 쓰지 않는다(사용자 2026-10-01). 숫자 문자열은 서버가 만든다.

    깨뜨리는 법: demo.html이나 demo.js에 '차년도'를 쓰거나 `toFixed`로 유사도를 찍으면 실패한다.
    """
    code = (STATIC / "demo.js").read_text(encoding="utf-8")
    page = (STATIC / "demo.html").read_text(encoding="utf-8")
    assert "차년도" not in code and "차년도" not in page.split("-->", 1)[-1]
    assert not re.search(r"\.toFixed\(|Intl\.|toLocaleString", code)
    assert "display.score_steps" in code and "bodies.steps" in code
    for stage in ("잘못된 자세 입력", "교정 안내", "교정된 결과"):
        assert stage in page
    assert json.loads((Path(app_module.__file__).parent / "demo_scenarios.json").read_text(encoding="utf-8"))["scenarios"]
