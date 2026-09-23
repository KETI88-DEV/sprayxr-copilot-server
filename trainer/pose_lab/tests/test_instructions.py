#!/usr/bin/env python3
# pyright: reportAny=false, reportUnusedCallResult=false
"""교정 문장·표시 문자열(`instructions.py`)과 그 배선.

같은 `AxisCorrection`에서 문장(text)과 수치(figure)를 함께 짓는다 — 목록·3D·문장이 한 숫자를
말하는지, 부호가 뒤집히지 않았는지, 표에 없는 조합에서 문장을 지어내지 않는지 고정한다.
"""

from __future__ import annotations

import importlib
import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

measured_score = importlib.import_module("measured_score")

from trainer.pose_lab import instructions, service  # noqa: E402
from trainer.pose_lab.service import PassRef  # noqa: E402
from trainer.pose_lab.tests import test_scoring_isolation as isolation  # noqa: E402
from trainer.pose_lab.tests.test_app import QUERY, REFERENCE, FakeSource  # noqa: E402


def _c(point: str, axis: str, magnitude: float, direction: str, sigma: float = 1.0):  # noqa: ANN202
    return measured_score.AxisCorrection(
        point=point, axis=axis, magnitude_cm=magnitude, direction=direction, sigma=sigma, unit=(1.0, 0.0, 0.0)
    )


def _compare() -> dict:
    body = service.compare(FakeSource(), PassRef(**QUERY), PassRef(**REFERENCE))
    assert body["status"] == "ok"
    return body


def test_corrections_text_matches_axes_by_n() -> None:
    body = _compare()
    axes = body["measured"]["axes"]
    arrows = body["view"]["corrections"]
    assert arrows
    for arrow in arrows:
        entry = axes[arrow["n"] - 1]
        assert entry["n"] == arrow["n"]
        for name in ("text", "text_reason", "text_mode", "figure", "sigma_display", "trace"):
            assert arrow[name] == entry[name], (arrow["n"], name)
        assert arrow["figure"] == instructions.figure(arrow["magnitude_cm"])
        assert entry["figure"] == instructions.figure(entry["magnitude_cm"])


def test_figure_and_text_number_agree() -> None:
    body = _compare()
    for entry in body["measured"]["axes"]:
        assert entry["text"] is not None
        match = re.search(r"(\d+\.\d)cm", entry["text"])
        assert match is not None, entry["text"]
        assert match.group(1) == f"{abs(entry['magnitude_cm']):.1f}"
        assert entry["figure"].lstrip("+−") == match.group(1)


def test_wall_phrases_are_not_swapped() -> None:
    toward, _ = instructions.axis_text(_c("right", "wall", -15.350920007369856, "붙이"))
    away, _ = instructions.axis_text(_c("right", "wall", 9.4, "띄우"))
    assert toward == "오른손(건)을 벽 쪽으로 15.4cm 붙이세요."
    assert away == "오른손(건)을 벽에서 9.4cm 띄우세요."


def test_right_hand_phrase_table() -> None:
    assert instructions.axis_text(_c("right", "vertical", 23.26, "올리"))[0] == "오른손(건)을 23.3cm 올리세요."
    assert instructions.axis_text(_c("right", "vertical", -5.0, "내리"))[0] == "오른손(건)을 5.0cm 내리세요."
    assert (
        instructions.axis_text(_c("right", "lateral", -11.38, "뒤로"))[0]
        == "오른손(건)을 진행 방향 뒤로 11.4cm 옮기세요."
    )
    assert instructions.axis_text(_c("right", "lateral", 7.0, ""))[0] == "오른손(건)을 벽을 따라 7.0cm 옮기세요."


@pytest.mark.parametrize(
    "axis,magnitude,direction",
    [("wall", -15.7, "붙이"), ("wall", 4.4, "띄우"), ("vertical", -3.0, "내리"), ("vertical", 53.4, "올리")],
)
def test_head_text_sign_agrees_with_stance_phrase(axis: str, magnitude: float, direction: str) -> None:
    correction = _c("head", axis, magnitude, direction)
    text, mode = instructions.axis_text(correction)
    stance = measured_score.stance_text(correction)
    assert mode == "directive"
    assert ("떨어져" in text) == ("떨어져" in stance)
    assert ("높여" in text) == ("높여" in stance)
    assert ("붙어" in text) == ("붙어" in stance)
    assert ("낮춰" in text) == ("낮춰" in stance)
    assert f"{abs(magnitude):.1f}cm" in text
    assert "한 걸음" not in text and "한 단계" not in text
    assert text.endswith("세요.")


@pytest.mark.parametrize(
    "correction",
    # head/lateral은 C1에서 표에 들어갔으므로 여기서 뺀다. 남은 둘은 여전히 표에 없다 —
    # 오른손의 "옆으로"는 `_axis_components`가 내지 않는 방향어이고, 왼손은 지시 대상이 아니다.
    [_c("right", "lateral", 5.0, "옆으로"), _c("left", "wall", -3.0, "붙이")],
)
def test_unknown_combination_raises(correction) -> None:  # noqa: ANN001
    with pytest.raises(ValueError):
        instructions.axis_text(correction)


def test_compare_survives_phrase_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_phrase(*_args, **_kwargs):  # noqa: ANN202
        raise ValueError("no phrase")

    monkeypatch.setattr(instructions, "axis_text", no_phrase)
    body = _compare()
    case = next(c for c in isolation._cases() if c["name"] == "synthetic_fake_source")  # noqa: SLF001
    isolation.assert_scoring_matches_baseline(body, case)
    assert body["view"]["display_errors"] == []
    for entry in body["measured"]["axes"]:
        assert entry["text"] is None
        assert entry["text_reason"] == "no_phrase"
        assert entry["figure"] == instructions.figure(entry["magnitude_cm"])
        assert entry["trace"]
    for arrow in body["view"]["corrections"]:
        assert arrow["text"] is None and arrow["text_reason"] == "no_phrase"


def test_wall_text_degrades_when_surface_angle_large() -> None:
    """착탄면과 30° 넘게 어긋난 wall 축은 '벽 쪽으로'라고 쓰지 않되, 수평 법선의 부호가 정확히 말하는
    '착탄 궤적 쪽/에서'로 **방향을 유지**한다. 부호가 뒤집히면(붙이↔띄우, 다가서↔물러서) 실패한다."""
    correction = _c("right", "wall", -15.350920007369856, "붙이")
    text, mode = instructions.axis_text(correction, 45.0)
    assert mode == "stroke_relative"
    assert text == "오른손(건)을 착탄 궤적 쪽으로 수평으로 15.4cm 붙이세요."
    away = instructions.axis_text(_c("right", "wall", 8.8, "띄우"), 83.0)
    assert away == ("오른손(건)을 착탄 궤적에서 수평으로 8.8cm 띄우세요.", "stroke_relative")
    assert instructions.axis_text(_c("head", "wall", -45.4, "붙이"), 83.0) == (
        "착탄 궤적 쪽으로 수평으로 45.4cm 더 다가서세요.", "stroke_relative"
    )
    assert instructions.axis_text(_c("head", "wall", 4.4, "띄우"), 83.0) == (
        "착탄 궤적에서 수평으로 4.4cm 더 물러서세요.", "stroke_relative"
    )
    # 머리 문장의 부호는 착탄면 각과 무관하게 방향 문장(HEAD_PHRASES)과 같다.
    for magnitude in (-3.0, 3.0):
        near = instructions.axis_text(_c("head", "wall", magnitude, "붙이" if magnitude < 0 else "띄우"), 10.0)[0]
        far = instructions.axis_text(_c("head", "wall", magnitude, "붙이" if magnitude < 0 else "띄우"), 80.0)[0]
        assert ("떨어져" in near) == ("물러서" in far)
    assert instructions.axis_text(correction, 29.0)[1] == "directive"
    assert instructions.axis_text(correction, 30.0)[1] == "directive"
    assert instructions.axis_text(correction, None)[1] == "directive"
    # 벽 축이 아니면 착탄면 각과 무관하다.
    assert instructions.axis_text(_c("right", "vertical", 5.0, "올리"), 80.0)[1] == "directive"


def test_axis_display_formats() -> None:
    shown = instructions.axis_display(_c("right", "wall", -15.350920007369856, "붙이", sigma=1.8812), 52.4)
    assert shown["figure"] == "−15.4"
    assert shown["sigma_display"] == "σ1.88"
    assert shown["trace"] == "오른손(건) · 벽(스트로크 기준) 축"
    assert shown["surface_angle_display"] == "착탄면과 52°"
    vertical = instructions.axis_display(_c("head", "vertical", 3.0, "올리"), 52.4)
    assert vertical["surface_angle_deg"] is None and vertical["surface_reason"] is None
    no_mesh = instructions.axis_display(_c("head", "wall", 3.0, "띄우"), None)
    assert no_mesh["surface_reason"] == "no_verified_surface"


def test_measured_display_strings_and_labels() -> None:
    measured = _compare()["measured"]
    assert measured["display"]["score"] == f"{measured['score']:.3f}"
    assert measured["display"]["score_after"] == f"{measured['score_after']:.3f}"
    assert measured["display"]["improvement_same"] == f"{measured['improvement']['same']:.3f}"
    assert measured["display"]["improvement_split"] == "—"  # 이 픽스처는 split이 None이다
    # 원시 차이. G만 보면 천장 가까이에서 속으므로 실제로 몇 점 움직였는지가 같이 나간다.
    # 값을 점수에서 직접 계산해 비교한다 — 문자열만 비교하면 부호·자릿수 규칙이 바뀌어도 통과한다.
    raw_delta = measured["score_after"] - measured["score"]
    assert raw_delta > 0.0, "이 픽스처의 전제: 교정이 유사도를 올린다"
    assert measured["display"]["score_delta"] == f"+{raw_delta:.3f}"
    for shown, delta in zip(measured["deltas_display"], measured["deltas"]):
        assert shown["point"] == delta["point"]
        assert shown["deviation_display"] == f"{delta['deviation_cm']:.1f}"
        assert shown["sigma_display"] == f"σ{delta['sigma']:.2f}"
    labels = measured["improvement_labels"]
    # 두 값 다 "지시를 그대로 따랐을 때"의 계산값이지 관측된 향상이 아니다. 라벨에서 그 조건이
    # 빠지면 화면의 가장 큰 숫자가 관측값처럼 읽힌다 — 이 프로젝트가 반복해서 걸려온 결함이다.
    for key in ("same", "split"):
        assert "따랐을 때의 상한" in labels[key], key
    # 둘의 차이는 자기충족 여부다. 그 구분이 라벨에서 사라지면 정직한 쪽(split)과
    # 상한인 쪽(same)을 읽는 사람이 구별할 수 없다.
    assert "자기충족 포함" in labels["same"]
    assert "자기충족 없음" in labels["split"]


def test_real_level1_texts_match_the_users_screen() -> None:
    """사용자 화면(김경호 L1 1000389_1 #0 대 전형기 1000564_1 #17)의 교정 4건.

    D1(키 m3) 뒤로 이 패스의 산포 버킷이 uv4 층으로 바뀌어 σ로 나눈 크기 순서가 달라졌다.
    네 교정의 크기(cm)는 m1 때와 같고 순서만 바뀌었다(m1: −15.4, +23.3, −15.7, −11.4).

    C1(머리 자세 축에 진행축 추가) 뒤로 **머리 문장이 바뀌었다**. 예전에는 머리의
    wall(−15.7cm, σ가 셋 중 최대였다)이 나갔는데, 이제 후보에 들어온 lateral이
    σ1.19로 더 이상해 그쪽이 나간다(−30.7cm). 오른손 세 축은 그대로다 — C1은 머리만
    건드린다. 이 픽스처가 그 사실을 고정한다.
    """
    case = next(c for c in isolation._cases() if c["name"] == "real_L1_kkh_1000389_1_0")  # noqa: SLF001
    body = isolation._run(case, include_view=True)  # noqa: SLF001
    axes = body["measured"]["axes"]
    assert [a["figure"] for a in axes] == ["+23.3", "−15.4", "−30.7", "−11.4"]
    assert axes[0]["text"] == "오른손(건)을 23.3cm 올리세요."
    wall = next(a for a in axes if a["point"] == "right" and a["axis"] == "wall")
    assert wall["text"] == "오른손(건)을 벽 쪽으로 15.4cm 붙이세요."
    head = next(a for a in axes if a["point"] == "head")
    assert head["axis"] == "lateral"
    assert head["text"] == "진행 방향 뒤로 30.7cm 옮겨 서세요."
    assert "한 걸음" not in head["text"]  # 화면 문장은 수치로 말한다(자세 문장과 다르다)
    blob = json.dumps(body["measured"]["axes"], ensure_ascii=False)
    for word in ("미사용", "사용하지 않음", "unused", "not used"):
        assert word not in blob


CONTRACT_DOC = Path(__file__).resolve().parents[3] / "docs" / "superpowers" / "specs" / "2026-09-15-pose-lab-view-contract.md"


def test_contract_doc_lists_every_text_mode_and_note_key() -> None:
    """응답이 내는 enum·키가 계약 문서에 적혀 있다 — 코드만 바뀌고 문서가 낡는 일(라운드 2 지적)을 막는다.
    합성 비교의 text_mode도 이 enum 안이어야 한다."""
    doc = CONTRACT_DOC.read_text(encoding="utf-8")
    for mode in instructions.TEXT_MODES:
        assert f'"{mode}"' in doc, mode
    for key in service.NOTE_KEYS:
        assert f"`{key}`" in doc, key
    body = service.compare(FakeSource(), PassRef(**QUERY), PassRef(**REFERENCE))
    assert {axis["text_mode"] for axis in body["measured"]["axes"]} <= set(instructions.TEXT_MODES)
    for correction in (_c("right", "wall", -15.4, "붙이"), _c("head", "wall", 45.4, "띄우")):
        for angle in (None, 10.0, 80.0):
            assert instructions.axis_text(correction, angle)[1] in instructions.TEXT_MODES

