"""main 에 올라온 패키지 버전(trainer/demo_api/__init__.py)이 이미 다른 코드에 쓰인 번호인지 본다.

실패로 보는 경우:
- 버전이 마지막 태그보다 작다.
- 같은 이름의 태그가 있는데 지금 커밋이 아니고, 아래 둘 중 하나다.
  - 태그 커밋이 지금 커밋의 조상이 아니고, 검증된 별도 배포본도 아니다.
  - 태그 뒤에 버전 파일을 고친 커밋이 있다(같은 번호를 다시 붙였다).
  릴리스 뒤에 버전 파일을 건드리지 않은 평범한 커밋은 통과한다. 그렇지 않으면 릴리스 사이의 모든 push 가 실패한다.

별도 release 이력은 주석 태그와 배포 커밋의 Source-Main-Commit·Release-Manifest-Blob
기록으로 확인한다. 원본 main이 HEAD의 조상이고, 버전과 manifest blob이 맞으며,
배포 트리가 원본 allowlist + 배포 README와 mode/blob까지 일치할 때만 인정한다.

표준 라이브러리만 쓴다(.github/workflows/version-check.yml 이 러너의 python3 로 바로 부른다).
"""
from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath

VERSION_FILE = "trainer/demo_api/__init__.py"
MANIFEST_FILE = "deploy/release/release_files.txt"
RELEASE_README = "deploy/release/README.release.md"
_TAG = re.compile(r"v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)")
_DEVELOPMENT_PATH = re.compile(r"\.(?:md|pdf)$|(^|/)tests?/|^docs/|^\.github/|^demo-ui/", re.IGNORECASE)


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)


def _key(tag: str) -> tuple[int, ...]:
    return tuple(int(part) for part in tag[1:].split("."))


def package_version(repo: Path) -> str:
    tree = ast.parse((repo / VERSION_FILE).read_text(encoding="utf-8"))
    versions = [node.value.value for node in tree.body
                if isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "__version__" for t in node.targets)
                and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)]
    if len(versions) != 1 or not _TAG.fullmatch("v" + versions[0]):
        raise SystemExit(f"{VERSION_FILE} 에 X.Y.Z 형식 __version__ 이 하나 있어야 한다")
    return versions[0]


def _tree_entries(repo: Path, revision: str) -> dict[str, tuple[str, str, str]] | None:
    result = _git(repo, "ls-tree", "-r", "-z", revision)
    if result.returncode:
        return None
    entries = {}
    for record in result.stdout.split("\0"):
        if not record:
            continue
        metadata, path = record.split("\t", 1)
        mode, kind, blob = metadata.split()
        entries[path] = (mode, kind, blob)
    return entries


def _release_source(repo: Path, tag: str, tagged: str, head: str) -> str | None:
    """Verify a separate deployment tree instead of trusting a message alone."""
    if _git(repo, "cat-file", "-t", f"refs/tags/{tag}").stdout.strip() != "tag":
        return None
    message = _git(repo, "show", "-s", "--format=%B", tagged).stdout
    sources = re.findall(r"^Source-Main-Commit: ([0-9a-f]{40})$", message, re.MULTILINE)
    manifests = re.findall(r"^Release-Manifest-Blob: ([0-9a-f]{40})$", message, re.MULTILINE)
    if len(sources) != 1 or len(manifests) != 1:
        return None
    source = sources[0]
    if (_git(repo, "cat-file", "-t", source).stdout.strip() != "commit"
            or _git(repo, "merge-base", "--is-ancestor", source, head).returncode != 0):
        return None
    # A filtered tree parented to main still brings all development history into release clones.
    if _git(repo, "merge-base", tagged, head).returncode != 1:
        return None
    manifest_blob = _git(repo, "rev-parse", "--verify", f"{source}:{MANIFEST_FILE}")
    if manifest_blob.returncode or manifest_blob.stdout.strip() != manifests[0]:
        return None
    manifest = _git(repo, "show", f"{source}:{MANIFEST_FILE}")
    version = _git(repo, "show", f"{source}:{VERSION_FILE}")
    if manifest.returncode or version.returncode:
        return None
    try:
        version_tree = ast.parse(version.stdout)
    except SyntaxError:
        return None
    versions = [node.value.value for node in version_tree.body
                if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "__version__" for target in node.targets)
                and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)]
    if versions != [tag[1:]]:
        return None
    wanted: set[str] = set()
    for path in manifest.stdout.splitlines():
        if not path:
            continue
        relative = PurePosixPath(path)
        if (path != path.strip() or path != relative.as_posix() or relative.is_absolute()
                or ".." in relative.parts or "\\" in path or _DEVELOPMENT_PATH.search(path)
                or (relative.name.startswith(".env") and not relative.name.endswith(".example"))
                or path in wanted):
            return None
        wanted.add(path)
    if VERSION_FILE not in wanted:
        return None
    original = _tree_entries(repo, source)
    actual = _tree_entries(repo, tagged)
    if original is None or actual is None:
        return None
    expected = {}
    for path in wanted | {RELEASE_README}:
        entry = original.get(path)
        if entry is None or entry[0] not in {"100644", "100755"} or entry[1] != "blob":
            return None
        if path != RELEASE_README:
            expected[path] = entry
    expected["README.md"] = ("100644", "blob", original[RELEASE_README][2])
    return source if actual == expected else None


def problems(repo: Path) -> list[str]:
    tag = "v" + package_version(repo)
    tags = [t for t in _git(repo, "tag", "--list").stdout.split() if _TAG.fullmatch(t)]
    found = []
    if tags:
        latest = max(tags, key=_key)
        if _key(tag) < _key(latest):
            found.append(f"버전 {tag[1:]} 이 마지막 태그 {latest} 보다 작다")
    if tag in tags:
        tagged = _git(repo, "rev-parse", f"refs/tags/{tag}^{{commit}}").stdout.strip()
        head = _git(repo, "rev-parse", "HEAD").stdout.strip()
        if tagged != head:
            source = tagged
            if _git(repo, "merge-base", "--is-ancestor", tagged, head).returncode != 0:
                source = _release_source(repo, tag, tagged, head)
            if source is None:
                found.append(f"태그 {tag} 는 다른 줄기의 커밋 {tagged[:12]} 에 있다. 이 코드는 새 버전 번호가 필요하다")
            else:
                touched = _git(repo, "log", "--full-history", "--format=%h %s", f"{source}..{head}", "--", VERSION_FILE).stdout.strip()
                if touched:
                    found.append(f"태그 {tag} 뒤에 버전 파일을 고친 커밋이 있다(같은 번호 재사용):\n{touched}")
    return found


def main() -> int:
    found = problems(Path.cwd())
    for problem in found:
        print(f"::error file={VERSION_FILE}::{problem}" if "\n" not in problem else problem)
    if not found:
        print(f"버전 {package_version(Path.cwd())} 확인")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
