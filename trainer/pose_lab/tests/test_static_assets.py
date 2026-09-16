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


# 5번째 인자(`measured`)만 본다. 4번째(`labelPointNames`)도 true/false여서 느슨한 정규식은
# 옛 4인자 호출의 4번째 인자를 `measured`로 잘못 읽는다 — 그러면 옛 코드도 통과한다.
LAYER_CALL_RE = re.compile(
    r"buildLayerGroup\(view\.layers\?\.(\w+), weights, LAYER_COLOR\.\w+, [^,()]+, (true|false)\)"
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
        "  layerGroups.corrected.add(buildLayerGroup(view.layers?.corrected, weights, LAYER_COLOR.corrected, false));\n"
    )
    assert computed_layer_faults(old_js) == [
        "corrected_layer_not_declared_computed",
        "query_layer_not_declared_measured",
        "reference_layer_not_declared_measured",
        "computed_layer_marker_is_a_filled_sphere",
        "computed_layer_line_is_solid",
    ]
    assert layer_measured_flags(
        "buildLayerGroup(view.layers?.corrected, weights, LAYER_COLOR.corrected, false, false))"
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
