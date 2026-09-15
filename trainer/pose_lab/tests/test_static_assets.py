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
