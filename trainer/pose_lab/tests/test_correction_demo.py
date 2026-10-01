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
def test_each_arrow_moves_its_body_part_by_the_spoken_amount(name: str) -> None:
    """화살표(안내 전후의 손·머리 관절)는 안내한 cm만큼 움직인다 — 그림과 문장이 같은 것을 말한다.

    손이 팔 길이로 닿지 않을 때도 어깨를 내밀어 정확히 닿는다. 깨뜨리는 법: 어깨 내밀기를 빼면 몸이 크게 물러서는
    쌍에서 손이 모자라 실패한다.
    """
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
def test_corrected_body_keeps_bone_lengths_and_the_last_step_is_the_result(name: str) -> None:
    """교정된 몸의 팔·다리 마디 길이는 원래 몸과 같고(사람이 늘어나지 않는다), 첫 단계는 원래 몸이다."""
    source, spreads, query, reference = _real(name)
    demo = cd.build_demo(source, query, reference, spreads)
    worker = demo["bodies"]["worker"]
    final = demo["bodies"]["steps"][-1]
    for parent, child in (("RightUpperArm", "RightForeArm"), ("RightForeArm", "RightHand"),
                          ("LeftUpperArm", "LeftForeArm"), ("LeftForeArm", "LeftHand"),
                          ("LeftThigh", "LeftCalf"), ("RightCalf", "RightFoot")):
        before = math.dist(worker[parent], worker[child])
        after = math.dist(final[parent], final[child])
        assert after == pytest.approx(before, abs=2e-3), (parent, child)
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


def test_stepping_moves_the_whole_body_but_the_hands_go_where_the_cue_says() -> None:
    """"옮겨 서세요"(머리 수평)는 몸 전체가 옮겨 간다. 손 안내가 없으면 손은 제자리 — 채점이 손을 따로 보기 때문이다."""
    body = _standing_body()
    stepped = cd.corrected_body(body, [_correction("head", (1.0, 0.0, 0.0), 20.0)])
    assert stepped["LeftFoot"][0] == pytest.approx(body["LeftFoot"][0] + 0.20)
    assert stepped["Head"][0] == pytest.approx(body["Head"][0] + 0.20)
    assert stepped["RightHand"] == pytest.approx(body["RightHand"], abs=1e-6)


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
