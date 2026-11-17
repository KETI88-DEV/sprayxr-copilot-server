from __future__ import annotations

import subprocess
import os
import tempfile
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


def _source_with_manifest(repo: Path, version: str = "3.8.9") -> str:
    release_dir = repo / "deploy/release"
    release_dir.mkdir(parents=True, exist_ok=True)
    (release_dir / "release_files.txt").write_text(
        VERSION_FILE + "\ntrainer/demo_api/service.py\n", encoding="utf-8")
    (release_dir / "README.release.md").write_text("Deployment instructions\n", encoding="utf-8")
    (repo / "trainer/demo_api/service.py").write_text("VALUE = 1\n", encoding="utf-8")
    _commit(repo, version, "source with runtime manifest")
    return _git(repo, "rev-parse", "HEAD")


def _split_release(repo: Path, source: str, *, tag: str = "v3.8.9", extra: str | None = None,
                   omit: str | None = None, changed: str | None = None, annotated: bool = True,
                   declared_source: str | None = None, declared_manifest: str | None = None,
                   executable: bool = False, main_parent: bool = False) -> str:
    """Build a real independent deployment tree without changing main or a remote."""
    manifest = _git(repo, "show", f"{source}:deploy/release/release_files.txt").splitlines()
    wanted = {path: path for path in manifest}
    wanted["README.md"] = "deploy/release/README.release.md"
    if omit:
        wanted.pop(omit)
    if extra:
        wanted[extra] = "deploy/release/README.release.md"
    with tempfile.TemporaryDirectory(dir=repo.parent) as temporary:
        env = {**os.environ, "GIT_INDEX_FILE": str(Path(temporary) / "index")}

        def indexed(*args):
            return subprocess.run(["git", "-C", str(repo), *args], env=env, check=True,
                                  capture_output=True, text=True).stdout.strip()

        indexed("read-tree", "--empty")
        for destination, original in wanted.items():
            blob = _git(repo, "rev-parse", f"{source}:{original}")
            if destination == changed:
                blob = _git(repo, "rev-parse", f"{source}:note.txt")
            mode = "100755" if executable and destination == "trainer/demo_api/service.py" else "100644"
            indexed("update-index", "--add", "--cacheinfo", f"{mode},{blob},{destination}")
        tree = indexed("write-tree")
    manifest_blob = declared_manifest or _git(repo, "rev-parse", f"{source}:deploy/release/release_files.txt")
    message = (f"release: {tag}\n\nSource-Main-Commit: {declared_source or source}\n"
               f"Release-Manifest-Blob: {manifest_blob}\n")
    parents = ["-p", source] if main_parent else []
    commit = _git(repo, "commit-tree", tree, *parents, "-m", message)
    if annotated:
        _git(repo, "tag", "-a", tag, commit, "-m", f"{tag} deployment")
    else:
        _git(repo, "tag", tag, commit)
    return commit


def test_separate_release_tag_with_exact_source_tree_passes(repo):
    source = _source_with_manifest(repo)
    release = _split_release(repo, source)
    assert subprocess.run(["git", "-C", str(repo), "merge-base", "--is-ancestor", release, source],
                          capture_output=True).returncode == 1
    assert problems(repo) == []
    _commit(repo, note="plain main documentation change after publishing")
    assert problems(repo) == []


@pytest.mark.parametrize("change", [
    {"extra": "docs/unwanted.md"},
    {"omit": "trainer/demo_api/service.py"},
    {"changed": "trainer/demo_api/service.py"},
    {"changed": "README.md"},
    {"executable": True},
    {"main_parent": True},
    {"annotated": False},
    {"declared_source": "0" * 40},
    {"declared_manifest": "0" * 40},
])
def test_separate_release_tag_requires_verified_source_and_exact_tree(repo, change):
    source = _source_with_manifest(repo)
    _split_release(repo, source, **change)
    assert any("다른 줄기" in item for item in problems(repo))


def test_separate_release_source_must_be_main_ancestor(repo):
    _git(repo, "checkout", "-q", "-b", "unrelated")
    source = _source_with_manifest(repo)
    _split_release(repo, source)
    _git(repo, "checkout", "-q", "main")
    _commit(repo, "3.8.9", "same version from another source")
    assert any("다른 줄기" in item for item in problems(repo))


def test_separate_release_retouching_version_still_fails(repo):
    source = _source_with_manifest(repo)
    _split_release(repo, source)
    _commit(repo, "3.8.10", "temporary next version")
    _commit(repo, "3.8.9", "reuse existing release number")
    assert any("같은 번호 재사용" in item for item in problems(repo))


def test_separate_release_manifest_cannot_allow_development_docs(repo):
    source = _source_with_manifest(repo)
    manifest = repo / "deploy/release/release_files.txt"
    manifest.write_text(manifest.read_text() + "note.txt\nREADME.md\n", encoding="utf-8")
    (repo / "README.md").write_text("Development README\n", encoding="utf-8")
    _commit(repo, note="source manifest accidentally includes development README")
    source = _git(repo, "rev-parse", "HEAD")
    _split_release(repo, source)
    assert any("다른 줄기" in item for item in problems(repo))


def test_make_release_reads_manifest_from_pinned_source_and_records_full_sha(repo):
    source = _source_with_manifest(repo)
    # An uncommitted local list must not change the pinned source's deployment tree.
    (repo / "deploy/release/release_files.txt").write_text("docs/do-not-publish.md\n", encoding="utf-8")
    script = Path(__file__).resolve().parents[3] / "deploy/release/make_release.sh"
    env = {**os.environ, "GIT_AUTHOR_NAME": "Release test", "GIT_AUTHOR_EMAIL": "test@local.invalid",
           "GIT_COMMITTER_NAME": "Release test", "GIT_COMMITTER_EMAIL": "test@local.invalid"}
    result = subprocess.run(["bash", str(script), source, "v3.8.9", "test"], cwd=repo,
                            env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert _git(repo, "cat-file", "-t", "refs/tags/v3.8.9") == "tag"
    release = _git(repo, "rev-parse", "v3.8.9^{commit}")
    message = _git(repo, "show", "-s", "--format=%B", release)
    assert f"Source-Main-Commit: {source}" in message
    assert f"Release-Manifest-Blob: {_git(repo, 'rev-parse', source + ':deploy/release/release_files.txt')}" in message
    assert set(_git(repo, "ls-tree", "-r", "--name-only", release).splitlines()) == {
        VERSION_FILE, "trainer/demo_api/service.py", "README.md"}
    assert problems(repo) == []


def test_make_release_only_parents_the_previous_deployment_commit(repo):
    source = _source_with_manifest(repo)
    previous = _split_release(repo, source)
    _git(repo, "update-ref", "refs/remotes/test/release", previous)
    source = _source_with_manifest(repo, "3.9.0")
    script = Path(__file__).resolve().parents[3] / "deploy/release/make_release.sh"
    env = {**os.environ, "GIT_AUTHOR_NAME": "Release test", "GIT_AUTHOR_EMAIL": "test@local.invalid",
           "GIT_COMMITTER_NAME": "Release test", "GIT_COMMITTER_EMAIL": "test@local.invalid"}
    result = subprocess.run(["bash", str(script), source, "v3.9.0", "test"], cwd=repo,
                            env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    release = _git(repo, "rev-parse", "v3.9.0^{commit}")
    assert _git(repo, "show", "-s", "--format=%P", release) == previous
    assert problems(repo) == []


def test_make_release_rejects_a_release_parent_with_main_history(repo):
    source = _source_with_manifest(repo)
    _git(repo, "update-ref", "refs/remotes/test/release", source)
    script = Path(__file__).resolve().parents[3] / "deploy/release/make_release.sh"
    env = {**os.environ, "GIT_AUTHOR_NAME": "Release test", "GIT_AUTHOR_EMAIL": "test@local.invalid",
           "GIT_COMMITTER_NAME": "Release test", "GIT_COMMITTER_EMAIL": "test@local.invalid"}
    result = subprocess.run(["bash", str(script), source, "v3.8.9", "test"], cwd=repo,
                            env=env, capture_output=True, text=True)
    assert result.returncode != 0
    assert "main 이력" in result.stderr
    assert subprocess.run(["git", "-C", str(repo), "rev-parse", "--verify", "refs/tags/v3.8.9"],
                          capture_output=True).returncode != 0
