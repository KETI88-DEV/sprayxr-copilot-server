"""정적 화면(static/)의 문자열 규칙을 고정한다.

브라우저 없이 확인할 수 있는 성질만 여기서 본다(배치·글자 크기·3D 픽셀은 헤드리스
브라우저 검증 몫이다).

- 화면은 서버의 표시 문자열만 쓴다: JS에서 숫자를 `toFixed`/`Intl`로 다시 포맷하지 않는다.
  Python과 JS는 반올림 동률 처리가 달라, 섞이면 같은 교정이 두 숫자로 보일 수 있다.
- 대외 문구 금지어('미사용' 계열)가 화면 소스에 없다.
- ③ 교정 섹션은 ② 유사도 섹션의 바로 다음 형제이고 3D 뷰(#view3d) 밖에 있다.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path

from trainer.pose_lab import service

STATIC_DIR = Path(__file__).resolve().parents[1] / "static"
STATIC_FILES = ("index.html", "app.js", "skeleton_view.js", "style.css")
BANNED_PHRASES = ("미사용", "사용하지 않음", "unused", "not used")


def _read(name: str) -> str:
    return (STATIC_DIR / name).read_text(encoding="utf-8")


def reformat_calls(source: str) -> list[str]:
    """숫자를 브라우저에서 다시 포맷하는 호출을 찾는다."""
    return re.findall(r"\.toFixed\(|\bIntl\.|\.toLocaleString\(|\.toPrecision\(", source)


def banned_phrases(source: str) -> list[str]:
    return [phrase for phrase in BANNED_PHRASES if phrase in source]


class _Tree(HTMLParser):
    """id가 붙은 요소의 부모 id와 형제 순서를 모은다(void 요소는 스택에 올리지 않는다)."""

    VOID = {"meta", "link", "input", "br", "img", "hr", "source", "col"}

    def __init__(self) -> None:
        super().__init__()
        self.stack: list[dict] = [{"tag": "#root", "id": None, "children": []}]
        self.by_id: dict[str, dict] = {}

    def handle_starttag(self, tag, attrs):
        node = {"tag": tag, "id": dict(attrs).get("id"), "children": [], "parent": self.stack[-1]}
        self.stack[-1]["children"].append(node)
        if node["id"]:
            self.by_id[node["id"]] = node
        if tag not in self.VOID:
            self.stack.append(node)

    def handle_endtag(self, tag):
        for index in range(len(self.stack) - 1, 0, -1):
            if self.stack[index]["tag"] == tag:
                del self.stack[index:]
                return


def corrections_placement(html: str) -> dict:
    tree = _Tree()
    tree.feed(html)
    hero = tree.by_id["heroSection"]
    corrections = tree.by_id["corrections"]
    siblings = hero["parent"]["children"]
    next_sibling = siblings[siblings.index(hero) + 1] if siblings.index(hero) + 1 < len(siblings) else None
    ancestor_ids = []
    node = corrections.get("parent")
    while node is not None:
        if node.get("id"):
            ancestor_ids.append(node["id"])
        node = node.get("parent")
    return {
        "next_sibling_of_hero_is_corrections": next_sibling is corrections,
        "inside_view3d": "view3d" in ancestor_ids,
    }


def test_static_sources_do_not_reformat_numbers():
    found = {name: reformat_calls(_read(name)) for name in STATIC_FILES}
    assert found == {name: [] for name in STATIC_FILES}


def test_static_sources_have_no_banned_phrases():
    found = {name: banned_phrases(_read(name)) for name in STATIC_FILES}
    assert found == {name: [] for name in STATIC_FILES}


def test_corrections_section_follows_hero_outside_3d_view():
    placement = corrections_placement(_read("index.html"))
    assert placement == {"next_sibling_of_hero_is_corrections": True, "inside_view3d": False}


def test_checks_can_fail():
    """위 세 검사가 실패할 수 있음을 합성 입력으로 확인한다(실패할 수 없는 단언 방지)."""
    assert reformat_calls("x.toFixed(1)") == [".toFixed("]
    assert banned_phrases("이 값은 미사용") == ["미사용"]
    moved = """<main><div><section id="heroSection"></section><p>끼어듦</p>
      <section id="corrections"></section></div><section id="view3d"></section></main>"""
    assert corrections_placement(moved)["next_sibling_of_hero_is_corrections"] is False
    nested = """<main><div><section id="heroSection"></section>
      <section id="view3d"><section id="corrections"></section></section></div></main>"""
    assert corrections_placement(nested)["inside_view3d"] is True


# --- F1: 범례 진실성 -------------------------------------------------------------------
#
# 실측은 머리(HMD)·건·플래쉬 세 강체의 마커뿐이다. 추정 기하(뼈·관절점·머리·다리)가 범례에서
# "실측"으로 선언한 부호를 그대로 쓰면, 화면을 처음 보는 사람이 가장 큰 구(추정 머리, 반경 0.09m)를
# 실측으로 읽는다. 옛 범례 "구체 = 실측"이 정확히 그랬다.

HEAD_MATERIAL_RE = re.compile(r"const headMaterial = new THREE\.MeshBasicMaterial\(\{([^}]*)\}\)")


def estimate_head_is_outline(js: str) -> bool | None:
    """추정 머리를 채운 구가 아니라 윤곽선으로 그리는가. 재료를 못 찾으면 None."""
    found = HEAD_MATERIAL_RE.search(js)
    if found is None:
        return None
    return "wireframe: true" in found.group(1)


def legend_truth_faults(legend_notes: tuple[str, ...], js: str) -> list[str]:
    """범례가 선언한 '실측' 부호를 추정 기하가 나눠 쓰면 결함 이름을 돌려준다."""
    faults = []
    text = " / ".join(legend_notes)
    if service.LEGEND_MEASURED_SYMBOL not in text:
        faults.append("legend_does_not_name_the_measured_symbol")
    if re.search(r"구체\s*=\s*실측", text) or re.search(r"(?<!색으로 채운 )구체 = ", text):
        faults.append("legend_hands_a_bare_shape_to_measured")
    if service.LEGEND_ESTIMATE_SYMBOL not in text:
        faults.append("legend_does_not_name_the_estimate_symbol")
    if estimate_head_is_outline(js) is not True:
        faults.append("estimate_head_drawn_as_a_filled_sphere")
    return faults


def test_legend_does_not_hand_the_measured_symbol_to_estimated_geometry():
    assert legend_truth_faults(service.LEGEND_NOTES, _read("skeleton_view.js")) == []


def test_f1_checks_can_fail():
    """F1 검사가 실패할 수 있음을 합성 입력으로 확인한다(실패할 수 없는 단언 방지)."""
    filled = "const headMaterial = new THREE.MeshBasicMaterial({ color: BONE_COLOR, opacity: 0.4 })"
    assert estimate_head_is_outline(filled) is False
    assert estimate_head_is_outline("머리 재료가 없다") is None
    old_legend = ("뼈 = 아바타 추정 · 점선 다리 = 자리표시(측정 아님) · 구체 = 실측",)
    faults = legend_truth_faults(old_legend, filled)
    assert "legend_hands_a_bare_shape_to_measured" in faults
    assert "legend_does_not_name_the_measured_symbol" in faults
    assert "estimate_head_drawn_as_a_filled_sphere" in faults
    assert legend_truth_faults(service.LEGEND_NOTES, filled) == ["estimate_head_drawn_as_a_filled_sphere"]


# --- F2: 관절 축 평가가 화면에 실제로 뜬다 ------------------------------------------------


def orientation_layout(html: str) -> dict:
    """관절 축 요약이 ④b 안에 있고, 전체 표(그리고 고지)보다 앞에 오는가."""
    tree = _Tree()
    tree.feed(html)
    summary = tree.by_id.get("jointAxesSummary")
    notes = tree.by_id.get("orientationNotes")
    table = tree.by_id.get("jointAxesTable")
    if summary is None or notes is None or table is None:
        return {"summary_present": False, "summary_before_notes": False, "inside_orientation": False}

    def order(node):
        """문서 순서(깊이 우선) 인덱스."""
        path = []
        while node is not None and node.get("parent") is not None:
            path.append(node["parent"]["children"].index(node))
            node = node["parent"]
        return tuple(reversed(path))

    ancestors = []
    node = summary.get("parent")
    while node is not None:
        if node.get("id"):
            ancestors.append(node["id"])
        node = node.get("parent")
    return {
        "summary_present": True,
        "summary_before_notes": order(summary) < order(notes) < order(table),
        "inside_orientation": "orientationSection" in ancestors,
    }


def joint_angle_note_is_rendered(app_js: str) -> bool:
    """서버의 `joint_angles_note`를 화면이 실제로 그리는가(옛 화면은 필드를 아예 안 봤다)."""
    return bool(re.search(r"orientation\.joint_angles_note", app_js))


def test_joint_axes_summary_precedes_the_full_table_inside_the_angle_panel():
    assert orientation_layout(_read("index.html")) == {
        "summary_present": True,
        "summary_before_notes": True,
        "inside_orientation": True,
    }


def test_joint_angles_note_is_rendered_not_just_served():
    assert joint_angle_note_is_rendered(_read("app.js")) is True


def test_f2_checks_can_fail():
    """F2 검사가 실패할 수 있음을 합성 입력으로 확인한다(실패할 수 없는 단언 방지)."""
    moved = """<main><section id="orientationSection"><ul id="orientationNotes"></ul>
      <div id="jointAxesSummary"></div><table id="jointAxesTable"></table></section></main>"""
    assert orientation_layout(moved)["summary_before_notes"] is False
    outside = """<main><div id="jointAxesSummary"></div><section id="orientationSection">
      <ul id="orientationNotes"></ul><table id="jointAxesTable"></table></section></main>"""
    assert orientation_layout(outside)["inside_orientation"] is False
    assert orientation_layout("<main></main>")["summary_present"] is False
    assert joint_angle_note_is_rendered("for (const t of [o.weight_note, o.excluded_note]) {}") is False
