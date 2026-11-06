"""Release notes must match the tagged source version and contain one release only."""
from pathlib import Path
import pytest
from trainer.scripts.release_notes import ReleaseNotesError, build_release_notes, extract_release_notes, version_from_tag


def test_extracts_one_release_without_leaking_previous_release():
    text = "# 변경 이력\n\n## [2.7.0] - 2026-11-06\n\n### 질의\n- 다국어 지원\n\n## [2.6.0] - 기준\n\n- 이전 기능\n"
    assert extract_release_notes(text, "2.7.0") == "### 질의\n- 다국어 지원\n"


@pytest.mark.parametrize("tag", ["main", "v2.7", "v02.7.0", "v2.7.0;echo x", "v2.7.0\nextra"])
def test_rejects_invalid_or_injected_tags(tag):
    with pytest.raises(ReleaseNotesError):
        version_from_tag(tag)


@pytest.mark.parametrize("text", ["# 없음", "## [2.7.0] - 날짜\n", "## [2.7.0]\n내용\n## [2.7.0]\n중복"])
def test_rejects_missing_empty_or_duplicate_sections(text):
    with pytest.raises(ReleaseNotesError):
        extract_release_notes(text, "2.7.0")


def test_tag_must_match_the_actual_package_version(tmp_path: Path):
    package = tmp_path / "trainer/demo_api/__init__.py"
    package.parent.mkdir(parents=True)
    package.write_text('__version__ = "2.7.0"\n')
    (tmp_path / "CHANGELOG.md").write_text("## [2.7.0] - 날짜\n\n### 추가\n- 새 기능\n", encoding="utf-8")
    assert "새 기능" in build_release_notes(tmp_path, "v2.7.0")
    with pytest.raises(ReleaseNotesError, match="version"):
        build_release_notes(tmp_path, "v2.8.0")
