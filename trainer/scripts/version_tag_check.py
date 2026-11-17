"""main 에 올라온 패키지 버전(trainer/demo_api/__init__.py)이 이미 다른 코드에 쓰인 번호인지 본다.

실패로 보는 경우:
- 버전이 마지막 태그보다 작다.
- 같은 이름의 태그가 있는데 지금 커밋이 아니고, 아래 둘 중 하나다.
  - 태그 커밋이 지금 커밋의 조상이 아니다(다른 줄기에서 같은 번호를 썼다).
  - 태그 뒤에 버전 파일을 고친 커밋이 있다(같은 번호를 다시 붙였다).
  릴리스 뒤에 버전 파일을 건드리지 않은 평범한 커밋은 통과한다. 그렇지 않으면 릴리스 사이의 모든 push 가 실패한다.

표준 라이브러리만 쓴다(.github/workflows/version-check.yml 이 러너의 python3 로 바로 부른다).
"""
from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path

VERSION_FILE = "trainer/demo_api/__init__.py"
_TAG = re.compile(r"v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)")


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
            if _git(repo, "merge-base", "--is-ancestor", tagged, head).returncode != 0:
                found.append(f"태그 {tag} 는 다른 줄기의 커밋 {tagged[:12]} 에 있다. 이 코드는 새 버전 번호가 필요하다")
            else:
                touched = _git(repo, "log", "--full-history", "--format=%h %s", f"{tagged}..{head}", "--", VERSION_FILE).stdout.strip()
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
