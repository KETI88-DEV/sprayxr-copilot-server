#!/usr/bin/env bash
# release 브랜치용 커밋을 만든다: 원본 커밋에서 API 서버 운영 파일(release_files.txt)만 골라 담는다.
#
# release 브랜치는 main 이력과 끊긴 별도 이력이다. 각 릴리스 커밋의 부모는 직전 release 커밋뿐이라
# `git clone --single-branch --branch release` 로 받으면 운영 코드만 내려간다.
# 문서·테스트·학습/평가 스크립트·웹 UI(demo-ui)는 넣지 않는다. 문서는 배포 안내 README 하나뿐이다
# (main의 deploy/release/README.release.md → release의 README.md).
#
# 사용(개발 서버, 저장소 안에서):
#   deploy/release/make_release.sh <원본 커밋> <vX.Y.Z> [원격 이름, 기본 keti]
# 결과: 로컬에 릴리스 커밋과 주석 태그를 만들고, 확인 방법과 push·게시 명령을 출력한다(push는 하지 않는다).
#
# 운영 코드에 새 모듈·데이터 파일이 생기면 release_files.txt에 더한다.
# 출력되는 import 확인이 실패하면 빠진 모듈이 있다는 뜻이다.
set -euo pipefail

src="${1:?원본 커밋을 준다(예: main)}"
version="${2:?버전을 준다(예: v3.0.3)}"
remote="${3:-keti}"
[[ "$version" =~ ^v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$ ]] || { echo "버전 형식은 vX.Y.Z" >&2; exit 2; }

top="$(git rev-parse --show-toplevel)"
list="$top/deploy/release/release_files.txt"
src_sha="$(git rev-parse --verify "$src^{commit}")"

code_version="$(git show "$src_sha:trainer/demo_api/__init__.py" | sed -n 's/^__version__ = "\(.*\)"$/\1/p')"
[ "v$code_version" = "$version" ] || { echo "원본의 __version__(${code_version})과 ${version}이 다르다" >&2; exit 1; }
git rev-parse -q --verify "refs/tags/$version" >/dev/null && { echo "태그 ${version}이 이미 있다" >&2; exit 1; }

# 문서·테스트가 목록에 섞이지 않았는지 먼저 본다(test_api_docs.py는 앱이 import하는 운영 모듈이다).
bad="$(grep -E '\.md$|\.pdf$|(^|/)tests?/|^docs/|^\.github/|^demo-ui/' "$list" || true)"
[ -z "$bad" ] || { echo "목록에 문서·테스트가 있다:" >&2; echo "$bad" >&2; exit 1; }

index="$(mktemp)"
trap 'rm -f "$index"' EXIT
export GIT_INDEX_FILE="$index"
git read-tree --empty
python3 - "$list" "$src_sha" <<'PY' | git update-index --index-info
import subprocess, sys
want = {line for line in open(sys.argv[1], encoding='utf-8').read().splitlines() if line}
raw = subprocess.run(['git', '-c', 'core.quotepath=false', 'ls-tree', '-r', '-z', sys.argv[2]],
                     capture_output=True, check=True).stdout.decode('utf-8')
found = set()
for rec in raw.split('\0'):
    if not rec:
        continue
    meta, path = rec.split('\t', 1)
    if path in want:
        mode, _type, sha = meta.split()
        sys.stdout.write(f'{mode} {sha}\t{path}\n')
        found.add(path)
missing = sorted(want - found)
if missing:
    sys.exit('원본 커밋에 없는 파일: ' + ', '.join(missing))
PY
# 배포 안내(git에 없는 모델·데이터를 둘 위치)는 release 루트의 README.md로 넣는다.
readme_blob="$(git rev-parse -q --verify "$src_sha:deploy/release/README.release.md" || true)"
[ -n "$readme_blob" ] || { echo "원본 커밋에 deploy/release/README.release.md가 없다" >&2; exit 1; }
git update-index --add --cacheinfo "100644,$readme_blob,README.md"
tree="$(git write-tree)"
unset GIT_INDEX_FILE

parent_args=()
if prev="$(git rev-parse -q --verify "refs/remotes/$remote/release")"; then
  parent_args=(-p "$prev")
fi
short="$(git rev-parse --short "$src_sha")"
commit="$(git commit-tree "$tree" "${parent_args[@]}" -m "release: ${version} — API 서버 운영 코드만 담은 배포본" -m "main ${short} 기준. agent-api와 MCP 실행에 필요한 $(wc -l < "$list")개 파일만 담는다. 문서, 테스트, 학습·평가 스크립트, 웹 UI(demo-ui)는 넣지 않는다.")"
git tag -a "$version" "$commit" -m "${version}: API 서버 운영 코드만 담은 배포본 (main ${short})"

echo "릴리스 커밋 ${commit} (파일 $(git ls-tree -r --name-only "$commit" | wc -l)개), 태그 ${version}"
echo
echo "1) import 확인(개발 서버 컨테이너):"
echo "   d=\$PWD/output/release-check-${version}; mkdir -p \$d && git archive ${commit} | tar -x -C \$d"
echo "   docker exec -w /workspace/output/release-check-${version} -e PYTHONPATH=/workspace/output/release-check-${version} ship-paint-trainer python3 -c 'import trainer.demo_api.app, mcp_servers.main'"
echo "2) 올리기:"
echo "   git push --atomic ${remote} ${commit}:refs/heads/release refs/tags/${version}"
echo "3) GitHub 릴리스 게시(노트는 원본 커밋의 CHANGELOG에서):"
echo "   docker exec -w /workspace ship-paint-trainer python3 -m trainer.scripts.release_notes --tag ${version} --output /workspace/output/notes-${version}.md"
echo "   gh release create ${version} --verify-tag --latest --title '${version} · SprayXR AI Co-Pilot' --notes-file output/notes-${version}.md"
