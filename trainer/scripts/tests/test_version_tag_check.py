from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from trainer.scripts.version_tag_check import VERSION_FILE, problems


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *args],
                          check=True, capture_output=True, text=True).stdout.strip()


def _commit(repo: Path, version: str | None = None, note: str = "x") -> None:
    if version:
        (repo / VERSION_FILE).write_text(f'"""p."""\n\n__version__ = "{version}"\n', encoding="utf-8")
    (repo / "note.txt").write_text(note, encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", note)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    (tmp_path / VERSION_FILE).parent.mkdir(parents=True)
    _git(tmp_path, "init", "-q", "-b", "main")
    _commit(tmp_path, "3.8.8", "release")
    _git(tmp_path, "tag", "v3.8.8")
    return tmp_path


def test_tagged_release_and_later_plain_commits_pass(repo):
    assert problems(repo) == []
    _commit(repo, note="fix after release")
    assert problems(repo) == []
    _commit(repo, "3.8.9", "next release, not tagged yet")
    assert problems(repo) == []


def test_same_number_from_another_line_fails(repo):
    _git(repo, "checkout", "-q", "-b", "other")
    _commit(repo, "3.8.9", "other dev bumps")
    _git(repo, "tag", "v3.8.9")
    _git(repo, "checkout", "-q", "main")
    _commit(repo, "3.8.9", "we bump to the same number")
    assert any("다른 줄기" in p for p in problems(repo))


def test_retouching_version_after_tag_fails(repo):
    _git(repo, "checkout", "-q", "-b", "other")
    _commit(repo, "3.8.9", "other dev bumps")
    _git(repo, "checkout", "-q", "main")
    _commit(repo, "3.8.9", "we bump too")
    _git(repo, "tag", "v3.8.9")
    _git(repo, "merge", "-q", "--no-edit", "-X", "ours", "other")
    assert any("같은 번호 재사용" in p for p in problems(repo))


def test_version_below_latest_tag_fails(repo):
    _commit(repo, "3.9.0", "minor")
    _git(repo, "tag", "v3.9.0")
    _commit(repo, "3.8.10", "accidental downgrade")
    assert any("마지막 태그 v3.9.0 보다 작다" in p for p in problems(repo))
