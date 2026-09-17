#!/usr/bin/env python3
# pyright: reportAny=false, reportUnusedCallResult=false
"""`docs/vendor_requests_open.md`가 적은 숫자가 **출처가 내는 값과 같은가**(S9).

이 저장소의 반복 결함은 "출처와 다른 라벨이 붙은 숫자"다. 업체 요청 문서는 협상 자리에 그대로
나갈 수치를 담으므로, 문서에서 숫자를 **뽑아** 출처를 다시 읽어 대조한다. 문서만 고치고 출처를
안 고치는(또는 그 반대의) 표류를 잡는 것이 목적이다.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

from trainer.pose_lab import service

ROOT = Path(__file__).resolve().parents[3]
DOC = ROOT / "docs" / "vendor_requests_open.md"
ALIGNMENT = ROOT / "trainer" / "data" / "level_alignment.json"
REQUIREMENTS = ROOT / "docs" / "data_requirements_pose_sequence_mcp.json"


def _doc() -> str:
    return DOC.read_text(encoding="utf-8")


def test_the_document_exists_and_names_all_four_open_requests() -> None:
    text = _doc()
    for heading in ("① 레벨 2 실행 FBX", "② BVH 프레임별 타임스탬프", "③ 나머지 5인 BVH",
                    "④ 아바타 손 IK ↔ 컨트롤러"):
        assert heading in text, heading
    # 항목마다 근거와 그것이 막고 있는 것을 적는다.
    assert text.count("**근거 수치**") == 4
    assert text.count("**막고 있는 것**") == 4


def test_level_two_numbers_match_the_alignment_record() -> None:
    """①의 두 게이트 수치가 `level_alignment.json`이 실제로 담은 값과 같다."""
    entry = next(e for e in json.loads(ALIGNMENT.read_text(encoding="utf-8")) if e.get("level") == 2)
    gates = {g["gate"]: float(g["measured"]) for g in entry["rejected_by"]}
    text = _doc()
    said_mm = float(re.search(r"`uv_degenerate_misattribution_mm` \| \*\*([\d.]+) mm\*\*", text).group(1))
    said_fraction = float(re.search(r"`uv_miss_fraction` \| ([\d.]+) ", text).group(1))
    assert abs(said_mm - gates["uv_degenerate_misattribution_mm"]) <= 0.01, (said_mm, gates)
    assert abs(said_fraction - gates["uv_miss_fraction"]) <= 1e-5, (said_fraction, gates)
    # 후보 전수 탐색이 0개 통과였다는 사실도 레코드에서 읽는다.
    assert entry["fixed_linear_attempt"]["candidates_passed_A"] == 0
    assert "`candidates_passed_A = 0`" in text
    # 레벨 2가 실제로 벽면을 그리지 않는지(막고 있는 것)를 서버에 물어본다.
    assert service.mesh_block("2")["available"] is False


def test_req_17_and_18_numbers_match_the_requirements_file() -> None:
    """②③이 인용한 수치가 요구사항 파일의 `status_2026_08_08`에 실제로 있는 문자열이다."""
    blob = json.dumps(json.loads(REQUIREMENTS.read_text(encoding="utf-8")), ensure_ascii=False)
    text = _doc()
    for quoted in ("1.46~10.30", "14.4mm", "-0.04", "11.9%", "69.7mm", "14mm", "0.13~0.17"):
        assert quoted in blob, f"출처에 없는 수치: {quoted}"
    # 문서는 같은 값을 읽기 좋게 띄어 적는다 — 숫자 자체는 같아야 한다.
    for said in ("1.46 ~ 10.30", "14.4 mm", "−0.04", "11.9%", "69.7 mm", "14 mm", "0.13~0.17"):
        assert said in text, f"문서에 빠진 수치: {said}"
    assert "5인은 **0개**" in text


def test_hand_gun_offset_number_matches_the_fitted_mounts() -> None:
    """④의 0.5672 m가 장착 상수 아티팩트에서 실제로 나오는 값이다(그리고 파지 상한의 2.3배다)."""
    table = service.load_mounts()
    right = table.table["right:한길숏건"]
    norm = math.hypot(*right.offset)
    text = _doc()
    said = float(re.search(r"\*\*0\.(\d{4}) m\*\*", text).group(0).strip("* m").lstrip("0"))
    assert abs(norm - said) < 5e-5, (norm, said)
    for point, quoted in (("head", 0.1098), ("left", 0.1167)):
        assert abs(math.hypot(*table.table[point].offset) - quoted) < 5e-5, point
        assert f"{quoted:.4f} m" in text
    limit = table.grip_limit_m["한길숏건"]
    assert f"숏건 {limit} m" in text
    ratio = norm / limit
    assert abs(ratio - 2.3) < 0.05, ratio
    assert "그 2.3배다" in text


def test_the_document_points_at_the_screen_notice_that_says_the_same_thing() -> None:
    """문서가 가리키는 화면 고지가 실제로 존재하고 같은 사실을 말한다 — 없는 고지를 인용하면
    문서가 거짓이 된다."""
    text = _doc()
    assert "`trainer/pose_lab/service.py::LOW_CONFIDENCE_NOTE`" in text
    note = service.LOW_CONFIDENCE_NOTE.format(gap=0.5672)
    assert "손 IK↔컨트롤러" in note
    assert "0.57m" in note
    # 화면은 이 적합값을 "보정"이라고 부르지 않는다 — 문서도 같은 말을 한다.
    assert "보정" not in note
    assert '"보정"이 아니라' in text
