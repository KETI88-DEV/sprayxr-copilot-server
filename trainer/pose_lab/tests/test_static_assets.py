"""정적 화면(static/)의 문자열 규칙을 고정한다.

브라우저 없이 확인할 수 있는 성질만 여기서 본다(배치·글자 크기·3D 픽셀은 헤드리스
브라우저 검증 몫이다).

- 화면은 서버의 표시 문자열만 쓴다: JS에서 숫자를 `toFixed`/`Intl`로 다시 포맷하지 않는다.
  Python과 JS는 반올림 동률 처리가 달라, 섞이면 같은 교정이 두 숫자로 보일 수 있다.
- 대외 문구 금지어('미사용' 계열)가 화면 소스에 없다.
- ③ 교정 섹션은 ② 유사도 섹션의 바로 다음 형제이고 3D 뷰(#view3d) 밖에 있다.
"""

from __future__ import annotations

import json
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


# 5번째 인자(`measured`)만 본다. 4번째(`labelPointNames`)도 true/false여서 느슨한 정규식은
# 옛 4인자 호출의 4번째 인자를 `measured`로 잘못 읽는다 — 그러면 옛 코드도 통과한다.
LAYER_CALL_RE = re.compile(
    r"buildLayerGroup\(view\.layers\?\.(\w+), LAYER_COLOR\.\w+, [^,()]+, (true|false)\)"
)
MARKER_MATERIAL_RE = re.compile(
    r"const markerMaterial = new THREE\.MeshBasicMaterial\(\{(.*?)\}\);", re.S
)


def layer_measured_flags(js: str) -> dict[str, str]:
    """층별 `buildLayerGroup(..., measured)` 인자. 층 이름 → "true"/"false"."""
    return {name: flag for name, flag in LAYER_CALL_RE.findall(js)}


def computed_layer_faults(js: str) -> list[str]:
    """계산된 층(교정 후·예상)이 범례가 "실측"으로 선언한 부호를 나눠 쓰는가.

    범례 ②는 "실측은 … 층 색으로 채운 구체"라고 말한다. 교정 후 층은 관측이 아니라 발부된 교정을
    따랐을 때의 계산값인데, 옛 화면은 이 층도 층 색(#34c2c2)으로 **채운** 구 60개로 그렸다
    (기본은 꺼져 있지만 사용자가 툴바로 켤 수 있다). 추정 머리와 같은 원칙이 걸린다.
    """
    faults = []
    flags = layer_measured_flags(js)
    if flags.get("corrected") != "false":
        faults.append("corrected_layer_not_declared_computed")
    for measured_layer in ("query", "reference"):
        if flags.get(measured_layer) != "true":
            faults.append(f"{measured_layer}_layer_not_declared_measured")
    material = MARKER_MATERIAL_RE.search(js)
    if material is None:
        faults.append("layer_marker_material_not_found")
    elif "wireframe: measured === false" not in material.group(1):
        faults.append("computed_layer_marker_is_a_filled_sphere")
    if "LineDashedMaterial" not in js.split("function buildLayerGroup(", 1)[-1].split("return group;", 1)[0]:
        faults.append("computed_layer_line_is_solid")
    return faults


def test_the_computed_layer_does_not_borrow_the_measured_symbol():
    assert computed_layer_faults(_read("skeleton_view.js")) == []


def test_computed_layer_checks_can_fail():
    """검사가 실패할 수 있음을 합성 입력으로 확인한다(실패할 수 없는 단언 방지)."""
    old_js = (
        "  function buildLayerGroup(a, b, c, d) {\n"
        "    const markerMaterial = new THREE.MeshBasicMaterial({ color: layerColor, opacity: 1 });\n"
        "    return group;\n"
        "  }\n"
        "  layerGroups.corrected.add(buildLayerGroup(view.layers?.corrected, LAYER_COLOR.corrected, false));\n"
    )
    assert computed_layer_faults(old_js) == [
        "corrected_layer_not_declared_computed",
        "query_layer_not_declared_measured",
        "reference_layer_not_declared_measured",
        "computed_layer_marker_is_a_filled_sphere",
        "computed_layer_line_is_solid",
    ]
    assert layer_measured_flags(
        "buildLayerGroup(view.layers?.corrected, LAYER_COLOR.corrected, false, false))"
    ) == {"corrected": "false"}


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


def joint_angle_quality_is_rendered(app_js: str) -> list[str]:
    """관절 각도 표가 값**과 품질 문구**를 함께 그리는가(D4).

    값만 그리고 품질을 빼면, 오른 팔꿈치처럼 사실상 상수인 각이 상수인 줄 모르게 보인다 — 브리프가
    명시적으로 금지한 것이다. 그리고 값·문구는 서버 문자열이어야 한다(화면이 다시 계산하지 않는다).
    """
    faults = []
    if "renderJointAngles" not in app_js:
        faults.append("no-renderer")
    if not re.search(r"measured\.joint_angles", app_js):
        faults.append("block-not-read")
    if not re.search(r"row\.quality_display", app_js):
        faults.append("quality-not-drawn")
    if not re.search(r"row\.display", app_js):
        faults.append("value-not-drawn")
    return faults


def test_joint_angles_note_is_rendered_not_just_served():
    assert joint_angle_note_is_rendered(_read("app.js")) is True


def test_joint_angle_values_are_drawn_with_their_quality():
    assert joint_angle_quality_is_rendered(_read("app.js")) == []


def test_joint_angle_quality_check_can_fail():
    """품질 문구를 뺀 합성 화면 코드는 잡혀야 한다(실패할 수 없는 단언 방지)."""
    value_only = "function renderJointAngles(b){} payload.measured.joint_angles; row.display.query"
    assert joint_angle_quality_is_rendered(value_only) == ["quality-not-drawn"]
    assert "no-renderer" in joint_angle_quality_is_rendered("nothing here")


def joint_angle_quality_layout_faults(style_css: str, app_js: str, index_html: str) -> list[str]:
    """관절 각도 품질 칸이 실제로 '괄호'가 아니라 값 아래 **별도 줄**임을 CSS·주석에서 고정한다.

    CDP 측정(레벨 2·3): `.joint-angle-quality`는 `display: block`, `::before`/`::after`의
    `content`는 둘 다 `"none"`, 렌더된 `innerText`에 괄호 문자가 없다. 소스 주석이 '괄호'라고
    부르면 이 사실과 어긋난다(리뷰 결함 1 — app.js·index.html도 QUALITY_NOTE와 같은 말을 반복했다).
    """
    faults = []
    if ".joint-angle-quality {" not in style_css:
        faults.append("quality_rule_missing")
    else:
        block = style_css.split(".joint-angle-quality {", 1)[1].split("}", 1)[0]
        if "display: block" not in block:
            faults.append("quality_is_not_its_own_line")
    if "괄호" in app_js:
        faults.append("app_js_claims_parentheses")
    if "괄호" in index_html:
        faults.append("index_html_claims_parentheses")
    return faults


def test_joint_angle_quality_wording_matches_the_block_layout():
    assert joint_angle_quality_layout_faults(_read("style.css"), _read("app.js"), _read("index.html")) == []


def test_joint_angle_quality_layout_check_can_fail():
    """CSS가 인라인으로 바뀌거나 주석이 괄호를 다시 언급하면 잡혀야 한다(실패할 수 없는 단언 방지)."""
    assert joint_angle_quality_layout_faults("", "", "") == ["quality_rule_missing"]
    inline_css = ".joint-angle-quality { display: inline; }"
    assert joint_angle_quality_layout_faults(inline_css, "", "") == ["quality_is_not_its_own_line"]
    assert joint_angle_quality_layout_faults(
        ".joint-angle-quality { display: block; }", "// 값 옆 괄호(quality_display)", "<!-- 괄호가 -->"
    ) == ["app_js_claims_parentheses", "index_html_claims_parentheses"]


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


# --- F3: 라벨이 가리지 말아야 할 것 · 숫자에 붙는 이름 ------------------------------------


def declaration(js: str, name: str) -> str:
    """`const <name> = ... ;` 한 선언을 줄바꿈을 넘어 통째로 꺼낸다(여러 줄 조건식용)."""
    head = f"const {name} = "
    if head not in js:
        return ""
    return js.split(head, 1)[1].split(";", 1)[0]


# 글자 라벨이 덮으면 안 되는 것들. 이름 = 회피 계산 안의 사각형 목록 변수.
LABEL_OBSTACLES = ("torsoRects", "limbRects", "arrowRects", "thinLineRects", "markerRects")


def label_avoidance_faults(js: str) -> list[str]:
    """글자 라벨이 피해야 할 장애물이 실제로 회피 계산에 들어가는가.

    몸통 **줄기**만 장애물로 두면 (1) '보정 오프셋' 띠가 교정 화살표 기둥을 가로질러 화살표가
    끊겨 보이고(L6), (2) 거리 라벨이 팔뼈를 덮고(L3 '10.5cm' × LeftForeArm>LeftHand),
    (3) 오프셋 선 자체가 글자 사이를 지나고(L1), (4) 라벨 모서리가 실측 마커 구를 스친다(L3).
    사각형을 만들기만 하고 회피(`clashes`)·점수(`cost`)에 안 쓰면 화면은 그대로다.
    """
    faults = []
    for maker in ("arrowScreenRects", "limbScreenRects", "thinLineScreenRects", "measuredMarkerScreenRects"):
        if f"function {maker}(" not in js:
            faults.append(f"{maker}_not_defined")
    clashes = declaration(js, "clashes")
    cost = declaration(js, "cost")
    for rects in LABEL_OBSTACLES:
        if rects not in clashes:
            faults.append(f"{rects}_not_in_clash_test")
        if rects not in cost:
            faults.append(f"{rects}_not_in_cost")
    if "debugLive.arrowLabelOverlaps" not in js:
        faults.append("arrow_overlap_not_measured")
    return faults


def badge_avoidance_faults(js: str) -> list[str]:
    """번호 배지가 (1) 서로 사각형으로, (2) **남의** 화살표 기둥을 피하는가.

    옛 배치는 배지끼리를 중심거리 26px로만 봐서 대각선 모서리 겹침(④↔① 20.4px²)을 놓쳤고,
    회피 탐색에서 배지를 통째로 제외해 배지가 남의 화살표 기둥에 앉았다. 두 가지 다 계수기에도
    안 잡혀 '겹침 0'이라는 자기 보고가 배지를 아예 세지 않았다.
    """
    faults = []
    if "function arrowScreenRectsByN(" not in js:
        faults.append("arrow_shafts_not_split_per_arrow")
    if "const badgeRect = " not in js:
        faults.append("badge_rect_not_defined")
    score = declaration(js, "badgeScore")
    if "placedBadgeRects" not in score:
        faults.append("badge_badge_overlap_not_scored")
    if "otherArrowRects" not in score:
        faults.append("other_arrow_shafts_not_scored")
    if not re.search(r"for \(const arrow of arrowsByN\) if \(arrow\.n !== entry\.n\)", js):
        faults.append("own_arrow_not_excluded")
    for counter in ("badgeBadgeOverlaps", "badgeArrowOverlaps", "badgeObstacleOverlaps"):
        if f"debugLive.{counter}" not in js:
            faults.append(f"{counter}_not_measured")
    return faults


def badge_search_scans_full_circle_faults(js: str) -> list[str]:
    """번호 배지 자리 찾기가 자기 화살표 기둥을 낀 가는 띠 안에 갇히면 안 된다.

    옛 탐색은 기둥 방향(along, ×BADGE_MIN_PX)과 그 수직(side, 최대 ±2×BADGE_MIN_PX=52px)만
    훑었다 — 기둥을 낀 얇은 직사각형 띠 밖은 아예 후보에 없었다. 라벨이 촘촘한 화면에서는 그 띠
    안에 0점 자리가 없을 수 있다: 김경호 L2 1000390_1 #2(1875×1000, 상체 확대)에서 배지 ②가
    "6.5cm" 라벨과 226.3px² 겹쳤다(`badgeObstacleOverlaps`=1, `labelOverlaps`=0 — 기본 화면의
    '겹침 0' 불변식을 지키는 계수기가 이 경우를 안 덮었다). 라벨 배치(`layoutLabels` 앞부분)처럼
    반경×회전(30° 단위) 극좌표로 기둥 주변 360°를 훑어야 띠 밖 빈 자리도 후보에 들어간다.
    """
    faults = []
    if "const badgeScore = (r) =>" not in js:
        faults.append("badge_score_not_defined")
        return faults
    search = js.split("const badgeScore = (r) =>", 1)[1]
    search = search.split("const rect = badgeRect(best.x, best.y);", 1)[0] if "const rect = badgeRect(best.x, best.y);" in search else search
    if "Math.cos" not in search or "Math.sin" not in search:
        faults.append("badge_search_does_not_rotate_around_the_shaft")
    radii_literal = re.search(r"radii\s*=\s*\[([^\]]+)\]", search)
    if not radii_literal or len(radii_literal.group(1).split(",")) < 5:
        faults.append("badge_search_radius_steps_too_few")
    return faults


def test_badge_search_can_escape_the_shaft_strip():
    assert badge_search_scans_full_circle_faults(_read("skeleton_view.js")) == []


def test_badge_search_full_circle_check_can_fail():
    """옛 along/side 격자 탐색(회전 없음)은 잡혀야 한다(실패할 수 없는 단언 방지)."""
    old_search = (
        "const badgeScore = (r) => 8 * overlapArea(r, placedBadgeRects) + 4 * overlapArea(r, otherArrowRects) + overlapArea(r, obstacles);\n"
        "for (const along of [0, 1, -1, 2, -2, 3, -3, 4]) {\n"
        "  for (const side of [0, 1, -1, 2, -2]) {\n"
        "    const ux = px.x + dx * along * BADGE_MIN_PX - dy * side * BADGE_MIN_PX;\n"
        "  }\n"
        "}\n"
        "const rect = badgeRect(best.x, best.y);\n"
    )
    assert badge_search_scans_full_circle_faults(old_search) == [
        "badge_search_does_not_rotate_around_the_shaft",
        "badge_search_radius_steps_too_few",
    ]


def counter_naming_faults(js: str) -> list[str]:
    """계수기 이름이 실제로 재는 범위와 같은가.

    `torsoLabelOverlaps`는 줄기 셋(골반-척추-가슴-머리)만 센다 — 그 이름으로 '팔을 포함한 뼈를
    안 덮는다'를 주장하면 출처와 다른 라벨이 된다. 넓은 판정은 `boneLabelOverlaps`가 따로 낸다.
    """
    faults = []
    if not re.search(r"debugLive\.torsoLabelOverlaps = torsoLabelOverlaps;", js):
        faults.append("torso_counter_missing")
    if not re.search(r"debugLive\.boneLabelOverlaps = torsoLabelOverlaps \+ limbLabelOverlaps;", js):
        faults.append("bone_counter_is_not_torso_plus_limb")
    torso_rects = js.split("function torsoScreenRects(", 1)[1].split("\n  }", 1)[0] if "function torsoScreenRects(" in js else ""
    if "TORSO_BONES" not in torso_rects:
        faults.append("torso_counter_does_not_use_the_stem_list")
    return faults


NARROW_NAME = "머리 마커~골반"


def measure_naming_faults(js: str) -> list[str]:
    """화면이 내놓는 두 세로 길이가 각자 이름을 제대로 달고 있는가.

    `upperBody`는 '다리를 뺀 모든 관절 + 모든 마커'의 바운딩 박스다 — 실제 양끝은 손 관절이나
    손 마커일 때가 많다. 이 값을 '머리 마커~골반'이라 부르면 출처와 다른 라벨이 붙는다.
    좁은 정의는 `torsoScreenSpan`이 머리 실측 마커와 골반 관절 **두 점만** 재서 따로 내놓는다.
    """
    faults = []
    wide_doc = js.split("function upperBodyScreen(")[0].rsplit("/**", 1)
    if len(wide_doc) == 2 and NARROW_NAME in wide_doc[1]:
        faults.append("wide_measure_claims_the_narrow_name")
    if "function torsoScreenSpan(" not in js:
        faults.append("narrow_measure_missing")
    else:
        body = js.split("function torsoScreenSpan(", 1)[1].split("\n  }", 1)[0]
        if 'a.joint === "Head"' not in body or "measured_marker" not in body:
            faults.append("narrow_measure_does_not_use_the_head_measured_marker")
        if "joints?.Pelvis" not in body:
            faults.append("narrow_measure_does_not_use_the_pelvis_joint")
        if "debugLive.torsoSpan" not in js:
            faults.append("narrow_measure_not_published")
    return faults


def test_label_avoidance_covers_torso_and_correction_arrows():
    assert label_avoidance_faults(_read("skeleton_view.js")) == []


def test_number_badges_avoid_each_other_and_other_arrows():
    assert badge_avoidance_faults(_read("skeleton_view.js")) == []


def test_overlap_counters_name_exactly_what_they_measure():
    assert counter_naming_faults(_read("skeleton_view.js")) == []


def test_the_two_vertical_measures_carry_their_own_names():
    assert measure_naming_faults(_read("skeleton_view.js")) == []


def test_f3_checks_can_fail():
    """F3 검사가 실패할 수 있음을 합성 입력으로 확인한다(실패할 수 없는 단언 방지)."""
    torso_only = (
        "  const clashes = (r) => hits(r, obstacles) || hits(r, torsoRects);\n"
        "  const cost = (r) => area(r, torsoRects);\n"
    )
    assert label_avoidance_faults(torso_only) == [
        "arrowScreenRects_not_defined",
        "limbScreenRects_not_defined",
        "thinLineScreenRects_not_defined",
        "measuredMarkerScreenRects_not_defined",
        "limbRects_not_in_clash_test",
        "limbRects_not_in_cost",
        "arrowRects_not_in_clash_test",
        "arrowRects_not_in_cost",
        "thinLineRects_not_in_clash_test",
        "thinLineRects_not_in_cost",
        "markerRects_not_in_clash_test",
        "markerRects_not_in_cost",
        "arrow_overlap_not_measured",
    ]
    old_badges = (
        "  const clash = placedBadges.some((p) => Math.hypot(p.x - x, p.y - y) < BADGE_MIN_PX);\n"
        "  const badgeScore2 = (r) => overlapArea(r, obstacles);\n"
    )
    assert badge_avoidance_faults(old_badges) == [
        "arrow_shafts_not_split_per_arrow",
        "badge_rect_not_defined",
        "badge_badge_overlap_not_scored",
        "other_arrow_shafts_not_scored",
        "own_arrow_not_excluded",
        "badgeBadgeOverlaps_not_measured",
        "badgeArrowOverlaps_not_measured",
        "badgeObstacleOverlaps_not_measured",
    ]
    # 옛 이름 붙이기: 줄기 계수기에 팔까지 더해 놓고 이름은 'torso' 그대로 — 이름이 범위보다 좁다.
    wide_name = "debugLive.torsoLabelOverlaps = torsoLabelOverlaps + limbLabelOverlaps;"
    assert counter_naming_faults(wide_name) == [
        "torso_counter_missing",
        "bone_counter_is_not_torso_plus_limb",
        "torso_counter_does_not_use_the_stem_list",
    ]
    mislabelled = f"  /** 다리를 뺀 상체({NARROW_NAME})가 캔버스에서 차지하는 세로 비율. */\n  function upperBodyScreen() {{\n"
    assert measure_naming_faults(mislabelled) == [
        "wide_measure_claims_the_narrow_name",
        "narrow_measure_missing",
    ]


# --- D6: 상체 확대 프리셋 --------------------------------------------------------


def torso_preset_faults(js: str, html: str) -> list[str]:
    """상체 확대 프리셋이 (1) 있고, (2) **실제 점**에 맞추고, (3) 화면 밖으로 내보낸 것의 이름을
    재서 화면에 적는가.

    (2)가 AABB 8꼭짓점이면 몸 앞뒤로 부풀린 상자의 가까운 모서리가 크게 잡혀 카메라가 물러난다 —
    확대 프리셋이 기본보다 **작아진다**. 실측(김경호 L3 1000391_1 #0, 1875×1000): 실제 점 0.6692 ·
    꼭짓점 0.4864 · 기본('자세 맞춤') 0.5827. (3)이 빠지면 화면이 자른 사실을 말하지 않는다.
    """
    faults = []
    if 'id="view3dPresetTorso"' not in js and 'id="view3dPresetTorso"' not in html:
        faults.append("no_button")
    if not re.search(r'frameTo\("torso"\)', js):
        faults.append("button_does_not_frame")
    if not re.search(r"FRAMING_TARGET\s*=\s*{[^}]*torso", js):
        faults.append("no_target")
    if not re.search(r'mode === "pose" \|\| mode === "torso" \? points\.map', js):
        faults.append("fits_aabb_corners")
    if "function offscreenFitTargets(" not in js:
        faults.append("no_offscreen_probe")
    if not re.search(r"function writeFramingNote\(\)[\s\S]{0,400}dom\.framingNote\.textContent\s*=", js):
        faults.append("note_not_shown")
    if not re.search(r"function frameTo\([\s\S]{0,600}writeFramingNote\(\)", js):
        faults.append("preset_does_not_write_the_note")
    return faults


def test_torso_preset_is_wired_and_discloses_what_it_cuts():
    assert torso_preset_faults(_read("skeleton_view.js"), _read("index.html")) == []


def test_torso_preset_checks_can_fail():
    """합성 입력으로 각 분기가 실제로 잡히는지 확인한다(실패할 수 없는 단언 방지)."""
    full = _read("skeleton_view.js")
    html = _read("index.html")
    boxed = full.replace(
        'mode === "pose" || mode === "torso" ? points.map((p) => new THREE.Vector3(...p)) : corners',
        'mode === "pose" ? points.map((p) => new THREE.Vector3(...p)) : corners',
    )
    assert torso_preset_faults(boxed, html) == ["fits_aabb_corners"]
    silent = full.replace("    dom.framingNote.textContent = text;\n", "")
    assert silent != full
    assert torso_preset_faults(silent, html) == ["note_not_shown"]
    unwired = full.replace("    // 자른 사실은 카메라를 적용한 **뒤에** 잰다 — 투영이 새 카메라를 써야 이름이 맞는다.\n    writeFramingNote();\n", "")
    assert unwired != full
    assert torso_preset_faults(unwired, html) == ["preset_does_not_write_the_note"]
    assert "no_button" in torso_preset_faults("", "")


# --- D4: 관절 각도 문구가 화면에서 말이 되는가 -------------------------------------------
#
# 두 번 틀렸던 자리다. (1) `orientation.JOINT_ANGLES_NOTE`가 "관절 각도는 **표시하지 않습니다**"라고
# 말하던 시절의 문구가 되돌아와도 아무 테스트도 잡지 못했다(전체 스위트 통과). (2) 리터럴 마크다운
# 별표가 화면에 글자로 보였고, 관절 축 표의 사유가 "**아래** 관절 각도 표"라고 방향을 거꾸로 말했다 —
# 실제 DOM 순서는 관절 각도 표가 **위**다. 그래서 문구와 **문서 순서**를 같이 본다.

JOINT_ANGLE_BANNED = ("미사용", "사용하지 않", "unused", "not used", "표시하지 않습니다", "괄호")


def document_order(html: str, element_id: str):
    """문서 순서(깊이 우선) 인덱스. 없으면 None."""
    tree = _Tree()
    tree.feed(html)
    node = tree.by_id.get(element_id)
    if node is None:
        return None
    path = []
    while node is not None and node.get("parent") is not None:
        path.append(node["parent"]["children"].index(node))
        node = node["parent"]
    return tuple(reversed(path))


def joint_angle_wording_faults(note: str, axes_blob: str, angles_blob: str, html: str) -> list[str]:
    """관절 각도 문구의 결함 목록.

    `note`는 `orientation.JOINT_ANGLES_NOTE`(건 각도 표 아래 고지), `axes_blob`·`angles_blob`은
    관절 축 평가·관절 각도 블록의 JSON 전문, `html`은 `index.html`이다.
    """
    faults = []
    for phrase in JOINT_ANGLE_BANNED:
        if phrase in note or phrase in axes_blob or phrase in angles_blob:
            faults.append(f"banned:{phrase}")
    if "추정" not in note:
        faults.append("note_does_not_say_estimate")
    for name, text in (("note", note), ("axes", axes_blob), ("angles", angles_blob)):
        if "**" in text:
            faults.append(f"literal_markdown:{name}")
    notes_at = document_order(html, "orientationNotes")
    angles_at = document_order(html, "jointAngles")
    axes_at = document_order(html, "jointAxes")
    if notes_at is None or angles_at is None or axes_at is None:
        return faults + ["missing_element"]
    # 고지가 "아래 관절 각도 표"라고 말하면, 관절 각도 표는 그 고지보다 문서에서 **뒤**에 있어야 한다.
    if "아래 관절 각도 표" in note and not notes_at < angles_at:
        faults.append("note_says_below_but_table_is_above")
    if "위 관절 각도 표" in note and not angles_at < notes_at:
        faults.append("note_says_above_but_table_is_below")
    # 관절 축 평가 표는 관절 각도 표보다 아래에 있다 — 그 표의 사유가 "아래"라고 말하면 거꾸로다.
    if "아래 관절 각도 표" in axes_blob and not axes_at < angles_at:
        faults.append("axes_say_below_but_table_is_above")
    if "위 관절 각도 표" in axes_blob and not angles_at < axes_at:
        faults.append("axes_say_above_but_table_is_below")
    return faults


def _joint_angle_blobs():
    from trainer.pose_lab import joint_angles as joint_angles_module
    from trainer.pose_lab import orientation as orientation_module
    from trainer.pose_lab.tests.test_orientation import _rows  # noqa: PLC0415

    rows = _rows()
    axes = json.dumps(orientation_module.joint_axes_block(rows, rows), ensure_ascii=False)
    angles = json.dumps(joint_angles_module.joint_angles_block(rows, rows), ensure_ascii=False)
    return orientation_module.JOINT_ANGLES_NOTE, axes, angles


def test_joint_angle_wording_matches_the_screen():
    note, axes, angles = _joint_angle_blobs()
    assert joint_angle_wording_faults(note, axes, angles, _read("index.html")) == []


def test_joint_angle_wording_checks_can_fail():
    """합성 입력으로 분기를 하나씩 고정한다 — 이 검사들이 실제로 실패할 수 있어야 한다."""
    html = _read("index.html")
    good_note, good_axes, good_angles = _joint_angle_blobs()
    # (1) 옛 문구가 되돌아온 경우 — 이것이 전체 스위트를 통과하던 구멍이었다.
    old_note = "관절 각도는 표시하지 않습니다 — 관절에는 추적 장치가 없습니다."
    assert joint_angle_wording_faults(old_note, good_axes, good_angles, html) == [
        "banned:표시하지 않습니다",
        "note_does_not_say_estimate",
    ]
    # (2) "추정"이 빠진 고지.
    assert joint_angle_wording_faults("관절 각도를 싣습니다.", good_axes, good_angles, html) == [
        "note_does_not_say_estimate"
    ]
    # (3) 리터럴 마크다운 별표.
    starred = good_note.replace("추정으로", "**추정**으로")
    assert joint_angle_wording_faults(starred, good_axes, good_angles, html) == ["literal_markdown:note"]
    # (4) 관절 축 표가 방향을 거꾸로 말하는 경우(커밋 3253435가 실제로 낸 결함).
    backwards = good_axes.replace("관절 각도 표에 있습니다", "아래 관절 각도 표에 있습니다")
    assert backwards != good_axes
    assert joint_angle_wording_faults(good_note, backwards, good_angles, html) == [
        "axes_say_below_but_table_is_above"
    ]
    # (5) 고지가 "아래"라고 말하는데 DOM에서 관절 각도 표가 고지보다 위인 경우.
    swapped = html.replace('id="jointAngles"', 'id="jointAnglesMoved"').replace(
        'id="orientationNotes"', 'id="jointAngles"'
    ).replace('id="jointAnglesMoved"', 'id="orientationNotes"')
    assert joint_angle_wording_faults(good_note, good_axes, good_angles, swapped) == [
        "note_says_below_but_table_is_above"
    ]
    # (6) 화면에 없는 괄호를 언급하는 품질 문구가 되돌아온 경우(리뷰 결함 1).
    # `.joint-angle-quality`는 값 span 아래 **별도 줄**(display: block)이다 — CDP 측정: 그 요소의
    # ::before/::after content가 둘 다 "none"이고 innerText에 괄호 문자가 없다. "괄호 안은…"은
    # 화면이 실제로 하는 일과 다른 말이다.
    bracketed_angles = good_angles.replace("값 아래 줄은", "괄호 안은")
    assert bracketed_angles != good_angles
    assert joint_angle_wording_faults(good_note, good_axes, bracketed_angles, html) == ["banned:괄호"]


# --- D3: 바닥에 맞춘 자리표시 발이 기본 뷰에서 안 보인다는 사실을 화면이 말하는가 --------------
#
# 실측(헤드리스 CDP, 캔버스 H=600): 기본 뷰('자세 맞춤')에서 자리표시 발의 캔버스 y가 김경호 L1
# 976·1063 / L3 697 / L6 786으로 전부 밖이었다. 다리는 맞춤 대상이 아니므로(브리프대로) 그대로 두되,
# **어느 프리셋에서 보이는지**는 말로 정하지 않고 각 프리셋의 카메라를 실제로 풀어서 투영해 확인한다.


def feet_note_faults(js: str) -> list[str]:
    faults = []
    if "function describeFeetFraming(" not in js:
        faults.append("no_feet_note")
    if not re.search(r"describeFraming\(mode\)\s*\{[\s\S]{0,200}describeFeetFraming\(\)", js):
        faults.append("feet_note_not_in_framing_note")
    # 다리 관절 **이름**은 서버가 준다. 자리표시 다리를 없앤 뒤(S4) 그 출처는 `fit.legs.joints`다.
    if not re.search(r"view\?\.fit\?\.legs\?\.joints", js):
        faults.append("feet_names_not_from_server")
    # 자리표시 다리(B)를 그리는 화면은 **그것이 자리표시라고 말해야 한다**. 재지 않은 다리를 IK와
    # 같은 실선으로 그리면 잰 것처럼 보인다 — 점선이 범례("점선 다리 = 자리표시")와 짝이다.
    # 그리고 어느 관절이 어디에 붙는지는 서버가 준 짝으로만 정한다.
    if "function buildPlaceholderLegGroup(" not in js:
        faults.append("no_placeholder_legs")
    else:
        builder = js.split("function buildPlaceholderLegGroup(", 1)[1].split("\n  }", 1)[0]
        if "LineDashedMaterial" not in builder:
            faults.append("placeholder_legs_are_not_dashed")
        if "placeholder_bones" not in builder:
            faults.append("placeholder_bones_not_from_server")
    if "function solvePreset(" not in js or "PRESET_PROBES" not in js:
        faults.append("no_preset_probe")
    # 프리셋 이름을 말로 정하지 않고 **풀어서** 확인하는가 — solveFraming을 실제로 부르는가.
    if not re.search(r"solvePreset\(probe\)[\s\S]{0,200}insideFrustum", js):
        faults.append("preset_names_are_not_measured")
    # 다리는 이제 리그 제 길이로 내린다(배율 없음) — 발을 "바닥에 맞춘"이라고 부르면 거짓이다.
    if "바닥에 맞춘" in js:
        faults.append("calls_feet_floor_snapped")
    probe_block = js.split("const PRESET_PROBES = [", 1)[-1].split("];", 1)[0]
    labels = set(re.findall(r'label: "([^"]+)"', probe_block))
    buttons = set(re.findall(r'id="view3dPreset\w+" type="button">([^<]+)</button>', js))
    if labels != buttons:
        faults.append("probe_labels_do_not_match_the_buttons")
    return faults


def test_screen_says_where_the_floor_snapped_feet_are_visible():
    assert feet_note_faults(_read("skeleton_view.js")) == []


def test_feet_note_checks_can_fail():
    full = _read("skeleton_view.js")
    assert feet_note_faults("") == [
        "no_feet_note",
        "feet_note_not_in_framing_note",
        "feet_names_not_from_server",
        "no_placeholder_legs",
        "no_preset_probe",
        "preset_names_are_not_measured",
    ]  # 빈 소스는 버튼도 없어 라벨 대조는 걸리지 않는다
    silent = full.replace('[describeTorsoFraming(mode), describeFeetFraming()]', "[describeTorsoFraming(mode)]")
    assert silent != full
    assert feet_note_faults(silent) == ["feet_note_not_in_framing_note"]
    renamed = full.replace('{ label: "궤적 포함", mode: "tracks"', '{ label: "궤적", mode: "tracks"')
    assert renamed != full
    assert feet_note_faults(renamed) == ["probe_labels_do_not_match_the_buttons"]
    always = full.replace(
        'const what = legsShown() ? "IK 발" : "자리표시 발";',
        'const what = legsShown() ? "바닥에 맞춘 IK 발" : "자리표시 발";',
        1,
    )
    assert always != full
    assert feet_note_faults(always) == ["calls_feet_floor_snapped"]
    # 자리표시 다리를 IK와 같은 실선으로 그리면 잡는다(재지 않은 것이 잰 것처럼 보이는 경우).
    solid = full.replace(
        "          opacity,\n          dashSize: COMPUTED_DASH,",
        "          opacity,\n          solidSize: COMPUTED_DASH,",
        1,
    ).replace("new THREE.LineDashedMaterial({\n          color: layerColor,\n          transparent: true,\n          opacity,",
              "new THREE.LineBasicMaterial({\n          color: layerColor,\n          transparent: true,\n          opacity,", 1)
    assert solid != full
    assert feet_note_faults(solid) == ["placeholder_legs_are_not_dashed"]
    # 뼈 짝을 화면이 지어내면 잡는다.
    invented = full.replace("    const bones = fitBlock?.legs?.placeholder_bones;", "    const bones = LEG_CHAIN_GUESS;", 1)
    assert invented != full
    assert feet_note_faults(invented) == ["placeholder_bones_not_from_server"]


def framing_leg_faults(js: str) -> list[str]:
    """기본 프레이밍('자세 맞춤')이 **그리는 다리의 발**을 늘 맞춤 대상에 넣는가(F).

    예전에는 이 자리의 한 줄이 서버 해부학 판정(`legsShown()`)을 따라 갈렸다: 같은 프레이밍 코드가
    레벨마다 다른 점 집합을 받아, L1은 다리를 넣어 상체가 캔버스의 0.240, L3·L6은 다리를 빼
    0.554·0.369가 됐다. 프레이밍 **규칙**이 아니라 **입력**이 레벨마다 달랐던 것이다. 자리표시
    다리는 모든 레벨에 있으므로 그 발을 늘 넣으면 맞춤 상자가 레벨 불문 "머리~바닥"이 된다.
    층마다 자기 판정을 쓴다 — 두 층의 IK 판정은 자주 갈린다.
    """
    if "function framingPoints(" not in js:
        return ["no_framing_points"]
    block = js.split("function framingPoints(", 1)[1].split("\n  }", 1)[0]
    faults: list[str] = []
    if "placeholderJointsOf(view.fit)" not in block:
        faults.append("query_placeholder_feet_not_framed")
    if "placeholderJointsOf(view.reference_fit)" not in block:
        faults.append("reference_placeholder_feet_not_framed")
    return faults


def test_framing_always_includes_the_feet_that_are_drawn():
    assert framing_leg_faults(_read("skeleton_view.js")) == []


def test_framing_leg_checks_can_fail():
    full = _read("skeleton_view.js")
    assert framing_leg_faults("") == ["no_framing_points"]
    dropped = full.replace(
        "      if (!legsShown()) for (const p of Object.values(placeholderJointsOf(view.fit))) push(p);\n", "", 1
    )
    assert dropped != full
    assert framing_leg_faults(dropped) == ["query_placeholder_feet_not_framed"]
    reference = full.replace(
        "      if (!referenceLegsShown()) for (const p of Object.values(placeholderJointsOf(view.reference_fit))) push(p);\n",
        "",
        1,
    )
    assert reference != full
    assert framing_leg_faults(reference) == ["reference_placeholder_feet_not_framed"]


# --- D3/D6: 카메라를 만진 뒤에도 화면 밖 고지가 참인가 ------------------------------------
#
# 검증이 막는 문제로 지목한 자리다. 고지는 `frameTo`에서만 쓰였고, 사람이 드래그·휠로 카메라를 만지면
# 그 자리에 굳었다. 실측(L3 '자세 맞춤', 호스트 1094×600): 캔버스 가운데를 위로 140px 끌고 휠을 12회
# 내리면 두 발이 화면 안(LeftFoot (482.6,323.7) · RightFoot (425.5,340.9))인데 고지는 그대로
# "바닥에 맞춘 자리표시 발은 이 프리셋에서 화면 밖에 있습니다"였다. 프리셋을 바꾼 적이 없으므로
# "이 프리셋에서"라는 표현이 방어해 주지 못한다.
#
# 여기서 고정하는 성질은 두 가지다. (1) 카메라를 바꾸는 **모든** 길(프리셋·드래그·휠)이 고지를 다시
# 쓴다. (2) "이 프리셋에서"는 카메라가 프리셋 그대로일 때만 쓴다(`framing.auto`).


# 두 핸들러 본문은 모두 이 길이 안에 끝난다. 더 멀리 보면 `scheduleFramingNote`의 **정의**까지
# 삼켜서, 호출을 지워도 이름이 남아 검사가 통과해 버린다(실패할 수 없는 단언).
HANDLER_BODY_CHARS = 420


def _handler_body(js: str, event: str) -> str:
    """`dom.canvas.addEventListener("<event>", ...)`의 콜백 본문(다음 addEventListener 전까지)."""
    marker = f'dom.canvas.addEventListener(\n    "{event}",'
    if marker not in js:
        marker = f'dom.canvas.addEventListener("{event}",'
    if marker not in js:
        return ""
    rest = js.split(marker, 1)[1]
    return rest.split("addEventListener", 1)[0][:HANDLER_BODY_CHARS]


def live_framing_note_faults(js: str) -> list[str]:
    faults = []
    if "function writeFramingNote(" not in js:
        faults.append("no_live_writer")
    if "function scheduleFramingNote(" not in js:
        faults.append("no_scheduler")
    for event, fault in (("pointermove", "drag_leaves_the_note_stale"), ("wheel", "wheel_leaves_the_note_stale")):
        body = _handler_body(js, event)
        if "scheduleFramingNote()" not in body:
            faults.append(fault)
    # 고지가 자기를 "이 프리셋에서"라고 부르는 것은 framing.auto일 때뿐이어야 한다.
    if not re.search(r"function nowWord\(\)\s*\{\s*return framing\.auto \? \"이 프리셋에서\" : \"지금\";", js):
        faults.append("preset_wording_not_conditioned_on_camera")
    if '이 프리셋에서 화면 밖에 있습니다' in js:
        faults.append("feet_note_hardcodes_the_preset_wording")
    if not re.search(r'framing\.auto \? "상체 확대" : "지금 화면\(상체 확대에서 카메라를 만졌습니다\)"', js):
        faults.append("torso_note_hardcodes_the_preset_wording")
    return faults


def test_framing_note_is_rewritten_whenever_the_camera_moves():
    assert live_framing_note_faults(_read("skeleton_view.js")) == []


def test_live_framing_note_checks_can_fail():
    """각 분기를 되돌려 실제로 잡히는지 확인한다(실패할 수 없는 단언 방지)."""
    full = _read("skeleton_view.js")
    assert live_framing_note_faults("") == [
        "no_live_writer",
        "no_scheduler",
        "drag_leaves_the_note_stale",
        "wheel_leaves_the_note_stale",
        "preset_wording_not_conditioned_on_camera",
        "torso_note_hardcodes_the_preset_wording",
    ]
    for event, fault in (("pointermove", "drag_leaves_the_note_stale"), ("wheel", "wheel_leaves_the_note_stale")):
        body = _handler_body(full, event)
        assert "scheduleFramingNote()" in body, event
        stale = full.replace(body, body.replace("    scheduleFramingNote();\n", "", 1), 1)
        assert stale != full, event
        assert live_framing_note_faults(stale) == [fault], event
    frozen = full.replace(
        'return framing.auto ? "이 프리셋에서" : "지금";',
        'return "이 프리셋에서";',
    )
    assert frozen != full
    assert live_framing_note_faults(frozen) == ["preset_wording_not_conditioned_on_camera"]
    hard = full.replace(
        'return `${what}은 ${nowWord()} 화면 밖에 있습니다${where}.`;',
        'return `${what}은 이 프리셋에서 화면 밖에 있습니다${where}.`;',
    )
    assert hard != full
    assert live_framing_note_faults(hard) == ["feet_note_hardcodes_the_preset_wording"]
    torso = full.replace(
        'const head = framing.auto ? "상체 확대" : "지금 화면(상체 확대에서 카메라를 만졌습니다)";',
        'const head = "상체 확대";',
    )
    assert torso != full
    assert live_framing_note_faults(torso) == ["torso_note_hardcodes_the_preset_wording"]


# --- 휠 확대 감도(U3) --------------------------------------------------------------------
#
# 옛 핸들러는 `event.deltaY > 0 ? 1.1 : 1/1.1`이라 **부호만** 썼다. 실측(CDP): deltaY가 4든 1000이든
# 배율이 1.1000으로 같았고, 트랙패드 한 번 쓸기(≈50 이벤트)가 1.1^50 = 117배였다. 아래 검사는
# "deltaY 크기를 쓴다 · deltaMode를 픽셀로 정규화한다 · 한 이벤트를 자른다 · 한계가 장면에서 나온다"를
# 고정한다.


def wheel_zoom_faults(js: str) -> list[str]:
    faults: list[str] = []
    body = _handler_body(js, "wheel")
    if "deltaY > 0 ?" in body or "deltaY < 0 ?" in body:
        faults.append("sign_only_zoom")
    if "wheelZoomFactor(" not in body or "wheelPixels(" not in body:
        faults.append("handler_does_not_normalize")
    if not re.search(r"function wheelPixels\(event, viewportHeight\)", js):
        faults.append("no_pixel_normalizer")
    else:
        pixels = js.split("function wheelPixels(", 1)[1].split("\n}", 1)[0]
        if "mode === 1" not in pixels or "WHEEL_LINE_PX" not in pixels:
            faults.append("line_delta_mode_not_handled")
        if "mode === 2" not in pixels or "viewportHeight" not in pixels:
            faults.append("page_delta_mode_not_handled")
    if not re.search(r"function wheelZoomFactor\(pixels\)", js):
        faults.append("no_zoom_factor")
    else:
        factor = js.split("function wheelZoomFactor(", 1)[1].split("\n}", 1)[0]
        if "WHEEL_PIXELS_MAX" not in factor or "clamp(" not in factor:
            faults.append("event_not_capped")
        if "Math.exp(" not in factor or "Math.log(WHEEL_NOTCH_FACTOR)" not in factor:
            faults.append("not_exponential_in_pixels")
    if "radiusLimitsFor(framing.baseRadius)" not in body:
        faults.append("limits_not_from_the_scene")
    if re.search(r"clamp\(camState\.radius \* factor, 0\.05, 60\)", js):
        faults.append("absolute_limits_hardcoded_in_the_handler")
    return faults


def test_wheel_zoom_uses_the_delta_size_and_scene_sized_limits():
    assert wheel_zoom_faults(_read("skeleton_view.js")) == []


def test_wheel_zoom_checks_can_fail():
    """각 성질을 되돌려 실제로 잡히는지 확인한다."""
    full = _read("skeleton_view.js")
    body = _handler_body(full, "wheel")
    assert "wheelZoomFactor(wheelPixels(event, h))" in body
    old = full.replace(
        "const factor = wheelZoomFactor(wheelPixels(event, h));",
        "const factor = event.deltaY > 0 ? 1.1 : 1 / 1.1;",
        1,
    )
    assert old != full
    assert "sign_only_zoom" in wheel_zoom_faults(old)
    assert "handler_does_not_normalize" in wheel_zoom_faults(old)

    flat = full.replace("  if (mode === 1) return delta * WHEEL_LINE_PX;\n", "", 1)
    assert flat != full
    assert wheel_zoom_faults(flat) == ["line_delta_mode_not_handled"]

    uncapped = full.replace("  const px = clamp(pixels, -WHEEL_PIXELS_MAX, WHEEL_PIXELS_MAX);", "  const px = pixels;", 1)
    assert uncapped != full
    assert wheel_zoom_faults(uncapped) == ["event_not_capped"]

    fixed = full.replace(
        "      const limits = radiusLimitsFor(framing.baseRadius);\n      camState.radius = clamp(camState.radius * factor, limits.min, limits.max);",
        "      camState.radius = clamp(camState.radius * factor, 0.05, 60);",
        1,
    )
    assert fixed != full
    assert sorted(wheel_zoom_faults(fixed)) == [
        "absolute_limits_hardcoded_in_the_handler",
        "limits_not_from_the_scene",
    ]


def test_one_wheel_notch_is_gentler_than_the_old_ten_percent():
    """상수가 뜻하는 한 노치 배율을 파이썬에서 다시 계산한다 — 옛 1.1보다 작아야 한다.

    검증 라운드: 핸들러 본문만 옛 `event.deltaY > 0 ? 1.1 : 1 / 1.1`로 되돌려도(상수 선언은 그대로
    둔 채) 이 검사가 **통과했다** — 상수를 소스에서 다시 읽을 뿐, 그 상수가 실제로 휠 핸들러에
    배선돼 있는지는 안 봤기 때문이다(핸들러가 상수를 한 번도 안 써도 구간 재계산은 그대로 값을
    낸다). 그래서 배선 검사(`wheel_zoom_faults`)를 여기서도 요구한다 — 상수와 배선이 갈리면
    이 검사가 재는 "노치가 옛 10%보다 순하다"는 배선된 핸들러의 성질이 아니라 죽은 상수의
    성질이 된다.
    """
    js = _read("skeleton_view.js")
    assert wheel_zoom_faults(js) == []  # 상수가 실제로 휠 핸들러에 배선돼 있는가
    notch_px = float(re.search(r"const WHEEL_NOTCH_PX = ([\d.]+);", js).group(1))
    notch_factor = float(re.search(r"const WHEEL_NOTCH_FACTOR = ([\d.]+);", js).group(1))
    pixels_max = float(re.search(r"const WHEEL_PIXELS_MAX = ([\d.]+);", js).group(1))
    assert notch_px == 100.0  # 관측된 마우스 한 노치(deltaY=100, deltaMode=0)
    assert 1.0 < notch_factor < 1.1, notch_factor
    # 트랙패드가 내는 deltaY=4짜리 이벤트 한 건은 0.5%를 넘지 않아야 한다(옛 값은 10%였다).
    assert notch_factor ** (4.0 / notch_px) < 1.005
    # 한 이벤트가 먹을 수 있는 최대 배율(관성 스크롤 한 방)도 유한해야 한다.
    assert notch_factor ** (pixels_max / notch_px) < 1.2


def test_one_wheel_notch_check_can_fail():
    """핸들러 본문만 옛 삼항식으로 되돌리면(상수 선언은 그대로) 위 검사가 실제로 잡는지 확인한다.

    이 되돌림이 바로 검증 라운드가 재현한 실패 사례다 — 고쳐지기 전에는 `wheel_zoom_faults`를
    묻지 않아 이 자리가 조용히 통과했다.
    """
    full = _read("skeleton_view.js")
    reverted = full.replace(
        "const factor = wheelZoomFactor(wheelPixels(event, h));",
        "const factor = event.deltaY > 0 ? 1.1 : 1 / 1.1;",
        1,
    )
    assert reverted != full
    faults = wheel_zoom_faults(reverted)
    assert "sign_only_zoom" in faults
    assert "handler_does_not_normalize" in faults


# --- 층이 쌓이지 않는가(U2) --------------------------------------------------------------
#
# 사용자가 "'기준'이 세 개가 그려져"라고 했다. 재현해 세어 보니 쌓임은 아니다(기준 층 후손 64개가
# [다른 기준 ▸] 3회·[평가] 3회 뒤에도 64개였다) — 원래 머리·오른손·왼손 **세 줄**이다. 그래도
# 쌓임을 막는 성질은 여기서 고정한다: 다시 그리기 전에 모든 층을 비우고 dispose한다.

LAYER_GROUP_NAMES = ("query", "reference", "corrected")


def layer_reset_faults(js: str) -> list[str]:
    faults: list[str] = []
    if "function resetScene()" not in js:
        return ["no_reset"]
    body = js.split("function resetScene()", 1)[1].split("\n  }", 1)[0]
    if "for (const group of Object.values(layerGroups))" not in body:
        faults.append("layers_not_swept")
    if "disposeObject3D(group)" not in body:
        faults.append("layers_not_disposed")
    if "group.clear()" not in body:
        faults.append("layers_not_cleared")
    if "correctionsGroup.clear()" not in body:
        faults.append("corrections_not_cleared")
    render = js.split("function render(payload)", 1)[1] if "function render(payload)" in js else ""
    if not render:
        faults.append("no_render")
    else:
        head = render.split("layerGroups.query.add(", 1)[0]
        if "resetScene();" not in head:
            faults.append("render_adds_before_reset")
    # 쌓임을 **화면이 스스로 셀 수 있어야** 다시 쌓일 때 잡힌다.
    if "layerObjectCounts: {" not in js or "function countDescendants(group)" not in js:
        faults.append("no_layer_object_counter")
    return faults


def test_layers_are_emptied_and_disposed_before_every_redraw():
    assert layer_reset_faults(_read("skeleton_view.js")) == []


def test_layer_reset_checks_can_fail():
    full = _read("skeleton_view.js")
    assert layer_reset_faults("") == ["no_reset"]
    piled = full.replace("      disposeObject3D(group);\n      group.clear();", "      disposeObject3D(group);", 1)
    assert piled != full
    assert layer_reset_faults(piled) == ["layers_not_cleared"]
    leaked = full.replace("      disposeObject3D(group);\n      group.clear();", "      group.clear();", 1)
    assert leaked != full
    assert layer_reset_faults(leaked) == ["layers_not_disposed"]
    blind = full.replace("      layerObjectCounts: {", "      layerObjectCountsRemoved: {", 1)
    assert blind != full
    assert layer_reset_faults(blind) == ["no_layer_object_counter"]


def test_the_legend_says_how_many_tracks_a_layer_draws():
    """"기준이 세 개"로 읽히지 않게, 층마다 몇 줄인지 서버 문구가 말한다.

    출처는 **그려질 층 자체**다 — 화면이 점이 하나라도 있는 이름마다 폴리라인을 만드므로 같은
    조건으로 센다. 예전에는 가중치에서 뽑고 "진한 차례로"라고 적었는데, 그 진하기 규칙이
    사라졌으므로(궤적은 차분한 한 색, 진해지는 것은 관절) 그 문구를 남기면 범례가 거짓말을 한다.
    """
    three = {"right": [[0, 0, 0]], "head": [[0, 0, 1]], "left": [[0, 1, 0]], "hit": [[1, 0, 0]]}
    note = service.track_legend_note(three)
    assert note is not None
    assert note.startswith("움직임 궤적 3줄")
    assert note.index("오른손") < note.index("머리") < note.index("왼손")
    assert "기준 3개가 아니라" in note
    # 진하기 순서를 더 이상 주장하지 않는다.
    assert "진한 차례" not in note
    # 왼손이 프레임마다 빠지는 층은 두 줄이라고 적는다 — 그려진 줄 수와 같은 출처다.
    two = dict(three, left=[None, None])
    other = service.track_legend_note(two)
    assert other is not None and other.startswith("움직임 궤적 2줄")
    assert "왼손" not in other
    # 그릴 줄이 없으면 적지 않는다.
    assert service.track_legend_note({"hit": [[0, 0, 0]]}) is None


# --- 스트로크는 서버가 자른 연결 구간만 그린다(U1) ---------------------------------------


def stroke_drawing_faults(js: str) -> list[str]:
    faults: list[str] = []
    if "function buildStrokeGroup(strokeBlock, color)" not in js:
        return ["no_stroke_builder"]
    body = js.split("function buildStrokeGroup(strokeBlock, color)", 1)[1].split("\n  }", 1)[0]
    if "strokeBlock?.runs" not in body:
        faults.append("not_from_server_runs")
    if "SphereGeometry" in body:
        faults.append("stroke_draws_loose_dots")
    # 화면이 자기 문턱으로 다시 자르면 서버 수치를 재계산하는 것이다.
    if "STROKE_MAX_LIFT" in body or "lift" in body:
        faults.append("screen_recomputes_the_split")
    if "buildStrokeGroup(view.stroke, STROKE_COLOR.query)" not in js:
        faults.append("render_does_not_draw_the_stroke")
    # 기준 쪽 벽면 스트로크도 그려야 한다(S2) — 사용자가 "기준 스트로크가 보이지 않음"이라고 했다.
    if "buildStrokeGroup(view.reference_stroke, STROKE_COLOR.reference)" not in js:
        faults.append("reference_stroke_not_drawn")
    # 스트로크 색과 움직임 궤적 색이 같으면 사용자가 둘을 혼동한 그 상태로 되돌아간다.
    stroke_colors = set(re.findall(r"const STROKE_COLOR = \{ query: (0x[0-9a-f]+), reference: (0x[0-9a-f]+) \}", js))
    layer_colors = set(re.findall(r"const LAYER_COLOR = \{ query: (0x[0-9a-f]+), reference: (0x[0-9a-f]+)", js))
    if not stroke_colors:
        faults.append("no_stroke_colors")
    elif stroke_colors & layer_colors:
        faults.append("stroke_reuses_the_track_colors")
    # 한 폴리라인으로 이어 붙이던 옛 경로가 남아 있으면 안 된다.
    if "surface-hit-trace" in js:
        faults.append("old_single_polyline_remains")
    if "setFromPoints(surfaceHits)" in js:
        faults.append("old_single_polyline_remains")
    # 낱개 대표 착탄 구체를 다시 그리면 안 된다.
    if "HIT_MARKER_RADIUS" in js:
        faults.append("lone_hit_marker_remains")
    # 층 궤적의 표식은 점마다가 아니라 끝점 하나다.
    if "function buildLayerGroup(" in js:
        layer = js.split("function buildLayerGroup(", 1)[1].split("\n    return group;", 1)[0]
        if re.search(r"for \(const p of points\) \{\s*\n\s*const marker", layer):
            faults.append("marker_on_every_sample")
        if "points[points.length - 1]" not in layer:
            faults.append("no_end_marker")
    return faults


def test_the_stroke_is_drawn_only_from_the_server_split_runs():
    assert stroke_drawing_faults(_read("skeleton_view.js")) == []


def test_stroke_drawing_checks_can_fail():
    full = _read("skeleton_view.js")
    assert stroke_drawing_faults("") == ["no_stroke_builder"]
    ignored = full.replace(
        "    const runs = Array.isArray(strokeBlock?.runs) ? strokeBlock.runs : [];",
        "    const runs = [];",
        1,
    )
    assert ignored != full
    assert stroke_drawing_faults(ignored) == ["not_from_server_runs"]
    dotted = full.replace(
        "        const piece = boneCylinder(points[i], points[i + 1], STROKE_RADIUS, material, \"surface-stroke\", \"\");",
        "        const piece = new THREE.Mesh(new THREE.SphereGeometry(STROKE_RADIUS, 8, 6), material);",
        1,
    )
    assert dotted != full
    assert stroke_drawing_faults(dotted) == ["stroke_draws_loose_dots"]
    every = full.replace(
        "      const marker = new THREE.Mesh(markerGeometry, markerMaterial);\n"
        "      marker.position.copy(points[points.length - 1]);",
        "      for (const p of points) {\n"
        "        const marker = new THREE.Mesh(markerGeometry, markerMaterial);\n"
        "        marker.position.copy(p);\n      }\n"
        "      const marker = new THREE.Mesh(markerGeometry, markerMaterial);",
        1,
    )
    assert every != full
    assert stroke_drawing_faults(every) == ["marker_on_every_sample"]
    lone = full + "\nconst HIT_MARKER_RADIUS = 0.018;\n"
    assert stroke_drawing_faults(lone) == ["lone_hit_marker_remains"]


# --- 머리 구체 지름은 이 리그에서 온다(F2) ------------------------------------------------
#
# 옛 상수 `HEAD_SPHERE_RADIUS = 0.09`(지름 0.18m)는 그려지는 골반→머리 0.409m의 44%였다 — 화면에서
# 풍선으로 보였다. 서버가 이 리그의 골반→머리 길이 × BVH 실측 비례로 지름을 내고(`fit.head_diameter_m`,
# `body_proportions.head_diameter_m`), 화면은 그 값만 쓴다. 폴백 상수는 `fit`이 없는 드문 경로에서만
# 쓰인다 — 정상 경로가 폴백으로 조용히 새면 이 검사가 못 잡으므로, 정상 경로가 폴백을 **안 쓰는지**도 본다.


def head_sphere_faults(js: str) -> list[str]:
    faults: list[str] = []
    if "function buildSkeletonGroup(" not in js:
        return ["no_skeleton_builder"]
    body = js.split("function buildSkeletonGroup(", 1)[1].split("\n  }\n", 1)[0]
    if "headDiameterM" not in js.split("function buildSkeletonGroup(", 1)[1].split(")", 1)[0]:
        faults.append("builder_does_not_take_head_diameter")
    if "headDiameterM / 2" not in body:
        faults.append("radius_not_derived_from_head_diameter")
    if not re.search(r"new THREE\.SphereGeometry\(radius,", body):
        faults.append("sphere_does_not_use_derived_radius")
    call = re.search(r"= buildSkeletonGroup\([^;]*\);", js)
    if call is None or "fit?.head_diameter_m" not in call.group(0):
        faults.append("call_site_does_not_pass_server_value")
    return faults


def test_head_sphere_is_sized_from_the_rigs_own_torso_length():
    assert head_sphere_faults(_read("skeleton_view.js")) == []


def test_head_sphere_checks_can_fail():
    full = _read("skeleton_view.js")
    assert head_sphere_faults("") == ["no_skeleton_builder"]
    fixed = full.replace(
        'const radius = typeof headDiameterM === "number" && headDiameterM > 0 ? headDiameterM / 2 : HEAD_SPHERE_RADIUS_FALLBACK;',
        "const radius = HEAD_SPHERE_RADIUS_FALLBACK;",
        1,
    )
    assert fixed != full
    assert "radius_not_derived_from_head_diameter" in head_sphere_faults(fixed)
    unwired = full.replace(
        "fit?.legs?.joints, fit?.head_diameter_m,",
        "fit?.legs?.joints,",
        1,
    )
    assert unwired != full
    assert "call_site_does_not_pass_server_value" in head_sphere_faults(unwired)


# --- S1/S2/S5/S6: 화면이 사용자가 요구한 그림을 실제로 그리는가 -----------------------------
#
# 사용자가 화면을 보고 말했다: "내가 말한 건 기준 스켈레톤과 질의 스켈레톤을 같이 그리고 유사도가
# 떨어지는 조인트일 수록 색상을 진하게 표시하라는 거였는데 숫자며 수치 무슨 말이인지를 모르겠고
# 보정 오프셋은 왜 표시했는지를 모르겠음." 아래 검사들은 그 그림의 뼈대를 고정한다.


def two_skeleton_faults(js: str) -> list[str]:
    faults: list[str] = []
    if "function buildSkeletonGroup(" not in js:
        return ["no_skeleton_builder"]
    # (1) 두 층에 각각 스켈레톤을 세우고, 각자 자기 층 그룹에 넣는가.
    if "buildSkeletonGroup(\n      joints," not in js:
        faults.append("query_skeleton_not_built")
    if "buildSkeletonGroup(\n      referenceJoints," not in js:
        faults.append("reference_skeleton_not_built")
    if "layerGroups.reference.add(referenceParts.group)" not in js:
        faults.append("reference_skeleton_not_added_to_its_layer")
    # (2) 진하기의 근거는 서버가 잰 벌어짐 하나뿐이다.
    if "view.deviation" not in js:
        faults.append("deviation_not_read_from_the_server")
    if "deviation?.joint_max_m" not in js:
        faults.append("scale_not_from_the_server")
    body = js.split("function buildSkeletonGroup(", 1)[1].split("\n  }\n", 1)[0]
    # 화면이 거리를 다시 재면 서버 수치를 재계산하는 것이다(이 저장소의 규칙).
    if "distanceTo" in body or "Math.hypot" in body:
        faults.append("screen_recomputes_the_gap")
    # (3) 채점 가중치 기반 강조는 버렸다 — 사용자가 본 그 강조가 유사도가 아니라 가중치였다.
    if "function emphasis(" in js:
        faults.append("weight_emphasis_remains")
    if re.search(r"buildSkeletonGroup\([^)]*weights", js):
        faults.append("weights_still_drive_the_skeleton")
    # (4) 추정 관절은 속 빈 표식이다 — 범례가 "채운 구체 = 실측"이라고 선언했다.
    if 'marker.userData.kind = "estimate-joint"' not in body:
        faults.append("joints_are_not_named_estimates")
    if not re.search(r'kind = "estimate-joint"[\s\S]{0,400}wireframe: true', body) and "wireframe: true," not in body:
        faults.append("estimate_joint_is_a_filled_sphere")
    return faults


def test_both_skeletons_are_drawn_and_coloured_by_the_server_gap():
    assert two_skeleton_faults(_read("skeleton_view.js")) == []


def test_two_skeleton_checks_can_fail():
    full = _read("skeleton_view.js")
    assert two_skeleton_faults("") == ["no_skeleton_builder"]
    one_only = full.replace("buildSkeletonGroup(\n      referenceJoints,", "buildSkeletonGroup(\n      joints,", 1)
    assert one_only != full
    assert "reference_skeleton_not_built" in two_skeleton_faults(one_only)
    fixed_scale = full.replace("const maxDeviation = deviation?.joint_max_m ?? null;", "const maxDeviation = 1.0;", 1)
    assert fixed_scale != full
    assert two_skeleton_faults(fixed_scale) == ["scale_not_from_the_server"]
    recomputed = full.replace(
        "    const devOf = (name) => deviationStyle(deviations[name], maxDeviation, layerColor);",
        "    const devOf = (name) => deviationStyle(Math.hypot(1, 2), maxDeviation, layerColor);",
        1,
    )
    assert recomputed != full
    assert two_skeleton_faults(recomputed) == ["screen_recomputes_the_gap"]


def deviation_source_faults(js: str) -> list[str]:
    """`render()`가 진하기 조인트(`deviations`)를 서버 벌어짐(`view.deviation.joints`)에서 뽑는가(S1) —
    채점 가중치(`view.weights`)로 되돌아가면 안 된다.

    `two_skeleton_faults`는 `buildSkeletonGroup(` **호출부** 문자열에 `weights`가 있는지만 본다. 호출부는
    언제나 `deviations,`라고 적혀 있으므로, `render()` 안에서 `const deviations = deviation?.joints ?? {}`를
    `const deviations = weights;`로 바꿔치기해도 호출부 문자열은 한 글자도 바뀌지 않고, 그 검사는 통과한다.
    이 검사는 그 배정문 자체를 본다 — 사용자가 거부한 바로 그 배선("유사도가 떨어지는 조인트"가 아니라
    "채점 가중치"로 진하게 칠하는 것)이 돌아오면 여기서 잡는다.
    """
    faults: list[str] = []
    body = _nested_function_body(js, "render")
    if not body:
        return ["no_render_function"]
    assign = re.search(r"const\s+deviations\s*=([^;]*);", body)
    if assign is None:
        faults.append("no_deviations_assignment")
        return faults
    expression = assign.group(1)
    if "weights" in expression:
        faults.append("deviations_sourced_from_scoring_weight")
    if "deviation" not in expression or "joints" not in expression:
        faults.append("deviations_not_sourced_from_server_deviation_joints")
    return faults


def test_joint_shading_source_is_the_servers_deviation_not_the_scoring_weight():
    assert deviation_source_faults(_read("skeleton_view.js")) == []


def test_deviation_source_checks_can_fail():
    """검증 라운드가 지목한 바로 그 되돌리기: `deviations`를 `weights`에서 뽑도록 바꾼다."""
    full = _read("skeleton_view.js")
    assert deviation_source_faults("") == ["no_render_function"]
    reverted = full.replace(
        "const deviations = deviation?.joints ?? {};",
        "const deviations = weights;",
        1,
    )
    assert reverted != full
    assert sorted(deviation_source_faults(reverted)) == [
        "deviations_not_sourced_from_server_deviation_joints",
        "deviations_sourced_from_scoring_weight",
    ]


def _top_level_function_body(js: str, name: str) -> str:
    """최상위 `function name(`의 본문. 닫는 중괄호는 0열의 `}`다(이 파일의 서식 규약)."""
    head = f"function {name}("
    if head not in js:
        return ""
    after = js.split(head, 1)[1]
    return after.split("\n}\n", 1)[0]


def _nested_function_body(js: str, name: str) -> str:
    """`mount()` 안에 중첩된 `function name(`의 본문. 닫는 중괄호는 2열의 `}`다."""
    head = f"function {name}("
    if head not in js:
        return ""
    return js.split(head, 1)[1].split("\n  }\n", 1)[0]


def without_comments(js: str) -> str:
    """주석을 지운 소스. 배선 검사가 **코드**만 보게 한다 — 주석에 적힌 이름을 배선으로 세면
    주석을 지우는 것만으로 검사가 빨개지고, 주석만 남기고 코드를 지워도 통과한다."""
    js = re.sub(r"/\*[\s\S]*?\*/", "", js)
    return re.sub(r"(?m)^\s*//.*$|\s//.*$", "", js)


def _split_top_level(text: str, separator: str = ",") -> list[str]:
    """괄호·중괄호 깊이 0에서만 자른다."""
    parts: list[str] = []
    depth = 0
    current = ""
    for char in text:
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        if char == separator and depth == 0:
            parts.append(current)
            current = ""
            continue
        current += char
    parts.append(current)
    return parts


def deviation_style_faults(js: str) -> list[str]:
    """`deviationStyle`의 **본문**이 서버가 잰 값에서 진하기를 뽑는가(S1).

    검증 라운드가 구멍을 뚫어 보였다: 이 함수 안의 `t`를 상수로 바꿔 진하기를 서버 편차와 완전히
    끊어도 정적 검사 51건이 전부 통과했다. 그때의 검사(`two_skeleton_faults`)가 **호출부 문자열**만
    보고 함수 본문을 못 봤기 때문이다. 여기서는 본문의 배선을 본다 —
    눈금(`maxMetres`) → `span`, 값(`metres`) ÷ `span` → `t`, `t` → 색·투명도.

    이것은 소스 구조 검사이지 실행 검사가 아니다(컨테이너에 JS 엔진이 없다). 무엇을 잡고 무엇을
    못 잡는지는 `test_deviation_style_checks_can_fail`이 변이로 직접 보인다.
    """
    faults: list[str] = []
    body = _top_level_function_body(js, "deviationStyle")
    if not body:
        return ["no_deviation_style"]
    span = re.search(r"const\s+span\s*=([^;]*);", body)
    if span is None or "maxMetres" not in span.group(1):
        faults.append("span_is_not_the_server_scale")
    ratio = re.search(r"const\s+t\s*=([^;]*);", body)
    if ratio is None:
        faults.append("no_t")
    else:
        expression = ratio.group(1)
        if "metres" not in expression:
            faults.append("t_ignores_the_server_gap")
        if "span" not in expression:
            faults.append("t_ignores_the_scale")
        if "/" not in expression:
            faults.append("t_is_not_a_ratio")
    returned = re.search(r"return\s*\{([^}]*)\}", body)
    if returned is None:
        faults.append("no_style_returned")
    else:
        fields = {}
        for part in _split_top_level(returned.group(1)):
            if ":" not in part:
                continue
            key, _, value = part.partition(":")
            fields[key.strip()] = value
        for key in ("color", "opacity"):
            if key not in fields:
                faults.append(f"style_has_no_{key}")
            elif not re.search(r"\bt\b", fields[key]):
                faults.append(f"{key}_ignores_t")
    # 화면이 거리를 다시 재면 서버 수치를 재계산하는 것이다(이 저장소의 규칙).
    if "distanceTo" in body or "Math.hypot" in body:
        faults.append("style_recomputes_the_gap")
    return faults


def test_joint_shading_is_wired_to_the_server_gap_inside_the_function_body():
    assert deviation_style_faults(_read("skeleton_view.js")) == []


def test_deviation_style_checks_can_fail():
    """검증 라운드가 실제로 넣었던 변이 그대로 — 본문의 `t`를 상수로 만든다."""
    full = _read("skeleton_view.js")
    assert deviation_style_faults("") == ["no_deviation_style"]
    body = _top_level_function_body(full, "deviationStyle")
    ratio = re.search(r"const\s+t\s*=[^;]*;", body).group(0)

    frozen = full.replace(ratio, "const t = 0.5;", 1)
    assert frozen != full
    assert sorted(deviation_style_faults(frozen)) == [
        "t_ignores_the_scale", "t_ignores_the_server_gap", "t_is_not_a_ratio",
    ]
    # 눈금만 끊어도 잡힌다(호출부는 그대로인 채 본문이 서버 눈금을 버리는 경우).
    scaleless = full.replace(ratio, ratio.replace("span", "1.0"), 1)
    assert scaleless != full
    assert "t_ignores_the_scale" in deviation_style_faults(scaleless)
    # 색이 t를 무시하면 진하기가 화면에 닿지 않는다.
    flat = full.replace("color: mixColor(pale, layerColor, t)", "color: layerColor", 1)
    assert flat != full
    assert deviation_style_faults(flat) == ["color_ignores_t"]
    faded = full.replace("opacity: 0.4 + 0.55 * t", "opacity: 0.9", 1)
    assert faded != full
    assert deviation_style_faults(faded) == ["opacity_ignores_t"]
    # 눈금을 서버가 아니라 화면 상수에서 뽑으면 잡힌다.
    own_scale = full.replace(
        'const span = typeof maxMetres === "number" && maxMetres > 0 ? maxMetres : null;',
        "const span = 1.0;", 1,
    )
    assert own_scale != full
    assert deviation_style_faults(own_scale) == ["span_is_not_the_server_scale"]


def deviation_legend_faults(js: str) -> list[str]:
    """진하기 눈금 문구를 **서버가** 짓는가(S1).

    이전 판은 화면이 `관절 진하기 = 기준과 벌어진 거리 (가장 진한 곳 ${max}cm)`를 직접 조립했고,
    그 숫자(`joint_max_display`)가 안 그리는 다리 관절에서 나와 화면 어디에도 없는 값을 적었다.
    """
    faults: list[str] = []
    code = without_comments(js)
    if "view?.deviation?.legend" not in code and "deviation.legend" not in code:
        faults.append("legend_lines_not_read_from_the_server")
    if "관절 진하기" in code:
        faults.append("screen_writes_the_scale_wording_itself")
    if "가장 진한 곳" in code:
        faults.append("old_scale_wording_remains")
    return faults


def test_the_shading_scale_wording_comes_from_the_server():
    assert deviation_legend_faults(_read("skeleton_view.js")) == []


def test_deviation_legend_checks_can_fail():
    full = _read("skeleton_view.js")
    assert deviation_legend_faults("") == ["legend_lines_not_read_from_the_server"]
    old = full.replace(
        "for (const text of view?.deviation?.legend ?? []) {",
        'const max = view?.deviation?.joint_max_display;\n  const lines = [`관절 진하기 = 기준과 벌어진 거리 (가장 진한 곳 ${max}cm)`];\n  for (const text of lines) {',
        1,
    )
    assert old != full
    assert sorted(deviation_legend_faults(old)) == [
        "legend_lines_not_read_from_the_server",
        "old_scale_wording_remains",
        "screen_writes_the_scale_wording_itself",
    ]


def leg_note_faults(js: str) -> list[str]:
    """다리 고지가 **두 층 몫**(`view.leg_notes`)에서 오는가(S4).

    질의 fit의 고지만 싣던 동안에는 기준 층 다리가 그려져 있는데 "다리는 그리지 않았습니다"라고
    적혔다(김경호 L1 1000389_1 #9, 씬에 보이는 기준 다리 뼈 8개). 두 층의 판정은 자동 기준 24쌍
    중 10쌍에서 갈린다.
    """
    faults: list[str] = []
    code = without_comments(js)
    if "view.leg_notes" not in code:
        faults.append("layer_leg_notes_not_read")
    block = code.split("dom.notes.replaceChildren()", 1)
    if len(block) < 2:
        return [*faults, "no_notes_block"]
    rendering = block[1].split("dom.notesMore.hidden", 1)[0]
    if "view.leg_notes" not in rendering:
        faults.append("leg_notes_not_rendered_with_the_other_notes")
    if '=== "legs"' not in rendering:
        faults.append("query_leg_note_is_not_replaced")
    return faults


def test_both_layers_leg_notes_are_shown_not_just_the_query_one():
    assert leg_note_faults(_read("skeleton_view.js")) == []


def test_leg_note_checks_can_fail():
    full = _read("skeleton_view.js")
    assert leg_note_faults("") == ["layer_leg_notes_not_read", "no_notes_block"]
    # 검증 라운드가 잡은 그 상태: 질의 fit의 고지만 싣는다.
    query_only = full.replace('if (keys[i] === "legs") {', "if (false) {", 1)
    assert query_only != full
    assert leg_note_faults(query_only) == ["query_leg_note_is_not_replaced"]
    dropped = full.replace(
        "const legNotes = (view.leg_notes ?? []).filter((text) => typeof text === \"string\");",
        "const legNotes = [];", 1,
    )
    assert dropped != full
    assert sorted(leg_note_faults(dropped)) == [
        "layer_leg_notes_not_read", "leg_notes_not_rendered_with_the_other_notes",
    ]


def default_preset_stroke_faults(js: str) -> list[str]:
    """기본 프리셋('자세 맞춤')이 **그려진 벽면 스트로크를 맞춤 대상에 넣는가**(S2).

    사용자 불만 1번이 "기준 스트로크(벽면 착탄점)이 보이지 않음"이었는데, 이전 판은 스트로크를
    pose 맞춤에서 빼 두어 사용자가 제일 먼저 보는 화면에서 두 줄이 가장자리에 걸리거나 밖으로
    나갔다('벽 포함'·'궤적 포함'에서만 온전히 보였다).
    """
    faults: list[str] = []
    code = without_comments(js)
    points = _nested_function_body(code, "framingPoints")
    if not points:
        return ["no_framing_points"]
    if "strokePoints(view)" not in points:
        faults.append("default_framing_ignores_the_strokes")
    else:
        before_mode_gate = points.split('if (mode === "wall")', 1)[0]
        if "strokePoints(view)" not in before_mode_gate:
            faults.append("strokes_are_only_in_the_wall_preset")
    runs = _nested_function_body(code, "strokePoints")
    if "reference_stroke" not in runs:
        faults.append("reference_stroke_is_not_a_fit_target")
    if "surface_hits" in runs:
        faults.append("framing_uses_points_it_does_not_draw")
    return faults


def test_the_default_preset_frames_both_wall_strokes():
    assert default_preset_stroke_faults(_read("skeleton_view.js")) == []


def test_default_preset_stroke_checks_can_fail():
    full = _read("skeleton_view.js")
    assert default_preset_stroke_faults("") == ["no_framing_points"]
    # 이전 판 그대로: 스트로크를 맞춤 대상에서 뺀다.
    dropped = full.replace("    for (const p of strokePoints(view)) push(p);\n", "", 1)
    assert dropped != full
    assert default_preset_stroke_faults(dropped) == ["default_framing_ignores_the_strokes"]
    # 원본 착탄점 전체(그리지 않는 점 포함)로 카메라를 끌면 잡힌다.
    raw_hits = full.replace(
        "for (const block of [view?.stroke, view?.reference_stroke]) {",
        "for (const block of [{ runs: [view?.surface_hits ?? []] }]) {", 1,
    )
    assert raw_hits != full
    assert sorted(default_preset_stroke_faults(raw_hits)) == [
        "framing_uses_points_it_does_not_draw", "reference_stroke_is_not_a_fit_target",
    ]


def hero_disclosure_faults(app_js: str, html: str) -> list[str]:
    """히어로 패널이 (a) 두 몸의 바닥 기준 머리 높이와 (b) 어느 단에서 맺은 쌍인지를 말하는가.

    (a) 이전 판은 질의 것만 보여 줘 L1 대상 1.51m·기준 1.66m, L3 0.70m·0.95m 중 한 값만 떴다.
    (b) 기준 줄은 "동작 키 완화: …"를 띄우는데 결과 쪽이 그 사실을 말하지 않으면 화면 두 자리가
    다른 말을 한다.
    """
    faults: list[str] = []
    if "view?.head_height?.display" not in app_js:
        faults.append("head_height_not_from_the_server_pair")
    if "view?.fit?.head_above_floor_display" in app_js:
        faults.append("head_height_still_shows_only_the_query")
    if 'id="heroHeadHeightNote"' not in html:
        faults.append("no_head_height_note_slot")
    if 'id="heroMotionMatch"' not in html:
        faults.append("no_key_match_slot")
    if "keyMatch?.label" not in app_js:
        faults.append("key_match_label_not_read_from_the_server")
    if "keyMatch?.relaxed" not in app_js:
        faults.append("relaxed_pairs_are_not_marked")
    if "payload.key_match" not in app_js:
        faults.append("key_match_not_passed_to_the_hero")
    return faults


def test_hero_says_both_head_heights_and_which_rung_matched():
    assert hero_disclosure_faults(_read("app.js"), _read("index.html")) == []


def test_hero_disclosure_checks_can_fail():
    app_js = _read("app.js")
    html = _read("index.html")
    assert sorted(hero_disclosure_faults("", "")) == [
        "head_height_not_from_the_server_pair", "key_match_label_not_read_from_the_server",
        "key_match_not_passed_to_the_hero", "no_head_height_note_slot", "no_key_match_slot",
        "relaxed_pairs_are_not_marked",
    ]
    old = app_js.replace('view?.head_height?.display', "view?.fit?.head_above_floor_display", 1)
    assert old != app_js
    assert sorted(hero_disclosure_faults(old, html)) == [
        "head_height_not_from_the_server_pair", "head_height_still_shows_only_the_query",
    ]
    silent = app_js.replace("keyMatch?.relaxed === true", "false", 1)
    assert silent != app_js
    assert hero_disclosure_faults(silent, html) == ["relaxed_pairs_are_not_marked"]


def collapsed_joint_angle_faults(html: str) -> list[str]:
    """관절 각도 표가 기본 접힘인가.

    사용자 불만 3번("전체적으로 화면이 너무 복잡함 · 텍스트가 너무 많아서")을 실측하니 이 한
    블록이 페이지에서 가장 큰 글자 덩어리였다(1875×1000 김경호 L1: 987자 대 3D 패널 전체 706자).
    점수·교정에 들어가지 않는 참고값이므로 관절 축 평가와 같은 대접을 한다.
    """
    found = re.search(r'<(\w+) class="joint-angles" id="jointAngles"([^>]*)>', html)
    if found is None:
        return ["no_joint_angles_block"]
    tag, attrs = found.group(1), found.group(2)
    faults: list[str] = []
    if tag != "details":
        faults.append("joint_angles_table_is_not_collapsible")
    if "open" in attrs.split():
        faults.append("joint_angles_table_is_open_by_default")
    if tag == "details" and "<summary" not in html.split(found.group(0), 1)[1][:400]:
        faults.append("collapsed_block_has_no_summary")
    return faults


def test_joint_angle_table_is_collapsed_by_default():
    assert collapsed_joint_angle_faults(_read("index.html")) == []


def test_collapsed_joint_angle_checks_can_fail():
    html = _read("index.html")
    assert collapsed_joint_angle_faults("<main></main>") == ["no_joint_angles_block"]
    flat = html.replace('<details class="joint-angles" id="jointAngles" hidden>', '<div class="joint-angles" id="jointAngles" hidden>', 1)
    assert flat != html
    assert collapsed_joint_angle_faults(flat) == ["joint_angles_table_is_not_collapsible"]
    opened = html.replace('<details class="joint-angles" id="jointAngles" hidden>', '<details class="joint-angles" id="jointAngles" hidden open>', 1)
    assert opened != html
    assert collapsed_joint_angle_faults(opened) == ["joint_angles_table_is_open_by_default"]


def three_d_number_faults(js: str) -> list[str]:
    """3D 장면 안에 숫자·보조선이 남아 있는가(S6). 숫자는 왼쪽 패널에만 있어야 한다."""
    faults: list[str] = []
    for token, name in (
        ("offset_display", "offset_label_remains"),
        ("miss_display", "miss_label_remains"),
        ("offset_line_label", "offset_line_legend_remains"),
        ("view3d-label-miss", "miss_label_class_remains"),
        ("view3d-label-offset", "offset_label_class_remains"),
        ("pass-mean-ring", "pass_mean_ring_remains"),
    ):
        if token in js:
            faults.append(name)
    # 교정 화살표 라벨은 번호뿐이다.
    badge = re.search(r'registerLabel\(mid, ([^,]+), "view3d-label-arrow"', js)
    if badge is None:
        faults.append("no_arrow_badge")
    elif "correction.n" not in badge.group(1):
        faults.append("arrow_label_is_not_just_the_number")
    # 앵커 그룹은 실측 구체만 만든다 — 보정 오프셋 선·어긋남 점선은 없다.
    if "function buildAnchorGroup(" in js:
        anchor = js.split("function buildAnchorGroup(", 1)[1].split("\n  }\n", 1)[0]
        if "THREE.Line" in anchor:
            faults.append("anchor_group_still_draws_lines")
    return faults


def test_the_3d_scene_carries_no_numbers_but_the_arrow_badges():
    assert three_d_number_faults(_read("skeleton_view.js")) == []


def test_3d_number_checks_can_fail():
    full = _read("skeleton_view.js")
    relabelled = full.replace(
        'registerLabel(mid, typeof correction.n === "number" ? String(correction.n) : "—", "view3d-label-arrow"',
        'registerLabel(mid, String(correction.magnitude_cm), "view3d-label-arrow"',
        1,
    )
    assert relabelled != full
    assert three_d_number_faults(relabelled) == ["arrow_label_is_not_just_the_number"]
    revived = full + '\n// const label = anchor.offset_display;\n'
    assert three_d_number_faults(revived) == ["offset_label_remains"]


def low_confidence_faults(js: str) -> list[str]:
    """오른 전완·손 저신뢰(S5)를 화면이 **서버가 준 이름**으로만 흐리게 그리는가."""
    faults: list[str] = []
    if "LOW_CONFIDENCE_OPACITY_FACTOR" not in js:
        faults.append("no_low_confidence_fade")
    for field in ("low_confidence_bones", "low_confidence_joints"):
        if f"fit?.{field}" not in js:
            faults.append(f"{field}_not_from_server")
        if f"referenceFit?.{field}" not in js:
            faults.append(f"reference_{field}_not_from_server")
    # 화면이 부위 이름을 지어내면 출처와 다른 라벨이 된다.
    if "RightForeArm" in js or "RightHand" in js:
        faults.append("screen_invents_the_bone_names")
    return faults


def test_low_confidence_segment_is_faded_by_server_named_bones():
    assert low_confidence_faults(_read("skeleton_view.js")) == []


def test_low_confidence_checks_can_fail():
    full = _read("skeleton_view.js")
    hardcoded = full.replace(
        "        lowConfidenceBones: fit?.low_confidence_bones,",
        '        lowConfidenceBones: [["RightForeArm", "RightHand"]],',
        1,
    )
    assert hardcoded != full
    assert low_confidence_faults(hardcoded) == [
        "low_confidence_bones_not_from_server",
        "screen_invents_the_bone_names",
    ]


def per_layer_leg_faults(js: str) -> list[str]:
    """다리는 **층마다 자기 판정**으로만 보인다(S4). 한쪽 판정으로 양쪽을 켜면 범위를 벗어난
    다리를 "기준은 이렇게 섰다"로 그리게 된다."""
    faults: list[str] = []
    if "legsAllowed" not in js:
        faults.append("no_per_layer_leg_gate")
        return faults
    toggle = js.split("function applyLegsToggle()", 1)[1].split("\n  }", 1)[0]
    # **IK 다리를 켜는 줄 자체**를 본다. 자리표시 다리(B)도 같은 함수에서 같은 이름으로 문을 여닫기
    # 때문에, "legsAllowed.reference가 이 함수 어딘가에 있다"로는 IK 쪽 빗장이 풀린 것을 못 잡는다
    # (실제로 그 느슨한 검사는 기준 IK 다리의 빗장을 없앤 변이를 통과시켰다).
    if "skeletonLegs.visible = on && legsAllowed.query" not in toggle:
        faults.append("query_legs_not_gated")
    if "referenceLegs.visible = on && legsAllowed.reference" not in toggle:
        faults.append("reference_legs_not_gated")
    # 기준 층의 IK 토글 가능 여부도 **서버가** 준다(B에서 이름이 legs_toggle_enabled로 바뀌었다) —
    # 화면이 질의 판정으로 기준을 대신하면 범위 밖 다리를 "기준은 이렇게 섰다"로 그리게 된다.
    if "referenceFit?.legs_toggle_enabled" not in js:
        faults.append("reference_gate_not_from_server")
    return faults


def test_each_layer_shows_only_its_own_anatomical_legs():
    assert per_layer_leg_faults(_read("skeleton_view.js")) == []


def test_per_layer_leg_checks_can_fail():
    full = _read("skeleton_view.js")
    assert per_layer_leg_faults("") == ["no_per_layer_leg_gate"]
    shared = full.replace(
        "    if (referenceLegs) referenceLegs.visible = on && legsAllowed.reference;",
        "    if (referenceLegs) referenceLegs.visible = on;",
        1,
    )
    assert shared != full
    assert per_layer_leg_faults(shared) == ["reference_legs_not_gated"]


# --- S7: 관절 축 평가는 접어 둔다 -----------------------------------------------------
#
# 사용자: "관절 축 평가는 접어도 되는 부분으로 보임. 텍스트가 너무 많아서 무슨 말을 하고 싶은지
# 모르겠음." 요약 칩은 그대로 펴 두고 **전체 표만** 접는다.


def collapsed_axes_faults(html: str) -> list[str]:
    faults: list[str] = []
    found = re.search(r'<(\w+) class="joint-axes" id="jointAxes"([^>]*)>', html)
    if found is None:
        return ["no_joint_axes_block"]
    tag, attrs = found.group(1), found.group(2)
    if tag != "details":
        faults.append("joint_axes_table_is_not_collapsible")
    if "open" in attrs.split():
        faults.append("joint_axes_table_is_open_by_default")
    if tag == "details" and "<summary" not in html.split(found.group(0), 1)[1][:400]:
        faults.append("collapsed_block_has_no_summary")
    # 요약 칩은 접지 않는다 — 접힌 것은 표뿐이다.
    summary = re.search(r'<(\w+) class="joint-axes-summary" id="jointAxesSummary"', html)
    if summary is None or summary.group(1) == "details":
        faults.append("summary_chips_were_collapsed_too")
    return faults


def test_joint_axes_table_is_collapsed_by_default():
    assert collapsed_axes_faults(_read("index.html")) == []


def test_collapsed_axes_checks_can_fail():
    full = _read("index.html")
    assert collapsed_axes_faults("") == ["no_joint_axes_block"]
    opened = full.replace('<details class="joint-axes" id="jointAxes" hidden>',
                          '<details class="joint-axes" id="jointAxes" hidden open>', 1)
    assert opened != full
    assert collapsed_axes_faults(opened) == ["joint_axes_table_is_open_by_default"]


# --- 기준 줄이 지나간 패스의 답을 들고 있으면 안 된다 --------------------------------------
#
# 실측: 6번에 1번, 대상 패스를 빠르게 바꾸면 앞서 나간 `/api/reference` 요청이 나중 요청보다
# 늦게 돌아와 `referenceState`를 덮었다. 기준 줄은 옛 패스의 "동작 키 벽면 8×8 칸까지 일치"를
# 그대로 두고, 같은 화면의 `/api/compare`는 이미 새 패스로 나가 "동작 키가 달라 비교하지
# 않습니다"를 냈다 — 한 화면이 서로 다른 패스를 두고 말한 것. 응답을 요청과 묶고, 늦게 온
# 것은 버려야 한다.


def reference_staleness_faults(js: str) -> list[str]:
    faults: list[str] = []
    if "referenceRequestId" not in js:
        return ["no_request_counter"]
    body = _top_level_function_body(js, "fetchReference")
    if not body:
        return ["no_fetch_reference"]
    if not re.search(r"const\s+requestId\s*=\s*\+\+referenceRequestId", body):
        faults.append("no_request_id_stamped_per_call")
    if "await fetch(" not in body:
        faults.append("no_await_fetch")
        return faults
    after_fetch = body.split("await fetch(", 1)[1]
    # 하나는 네트워크 오류(catch) 경로, 하나는 정상 응답 경로 — 둘 다 `referenceState`를
    # 건드리기 전에 이 요청이 여전히 최신인지 본다.
    guard_count = len(re.findall(r"requestId\s*!==\s*referenceRequestId", after_fetch))
    if guard_count < 2:
        faults.append("response_not_checked_against_latest_request")
    return faults


def test_stale_reference_responses_are_dropped():
    assert reference_staleness_faults(_read("app.js")) == []


def test_reference_staleness_checks_can_fail():
    full = _read("app.js")
    assert reference_staleness_faults("") == ["no_request_counter"]
    no_stamp = full.replace("  const requestId = ++referenceRequestId;\n", "", 1)
    assert no_stamp != full
    assert "no_request_id_stamped_per_call" in reference_staleness_faults(no_stamp)
    unguarded = full.replace(
        "    if (requestId !== referenceRequestId) return; // 늦게 온 응답 — 더 새 요청이 이미 나갔다.\n", "", 1,
    ).replace(
        "  if (requestId !== referenceRequestId) return; // 늦게 온 응답 — 버리고 최신 요청의 결과를 기다린다.\n", "", 1,
    )
    assert unguarded != full
    assert reference_staleness_faults(unguarded) == ["response_not_checked_against_latest_request"]
