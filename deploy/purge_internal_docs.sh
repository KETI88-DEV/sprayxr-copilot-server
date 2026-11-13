#!/usr/bin/env bash
# 배포 PC에 남은 내부 개발 문서를 찾아 지운다.
#
# 2026-11-13부터 내부 개발 문서는 이 저장소에서 빠지고 .gitignore의 "내부 개발 문서" 절에 묶였다.
# 이 커밋을 pull하면 git이 해당 파일을 작업 트리에서 지운다. 이 스크립트는 그 뒤에도 남은 사본
# (pull 전에 손으로 고쳤거나 따로 복사해 둔 파일)을 찾는다. 지우는 대상은 .gitignore가 무시하는
# 경로뿐이다. 저장소가 추적하는 파일은 건드리지 않는다.
#
# 사용(저장소 안에서):
#   git pull --ff-only
#   deploy/purge_internal_docs.sh            # 남은 파일 목록만 출력
#   deploy/purge_internal_docs.sh --execute  # 목록의 파일을 지운다
set -euo pipefail

execute=0
case "${1:-}" in
  "") ;;
  --execute) execute=1 ;;
  *) echo "사용: $0 [--execute]" >&2; exit 2 ;;
esac

root="$(git rev-parse --show-toplevel)"
cd "$root"

if ! grep -q '^# 내부 개발 문서' .gitignore; then
  echo "이 사본에는 2026-11-13 정리 커밋이 없다. 먼저 git pull --ff-only 를 실행한다." >&2
  exit 1
fi

# .gitignore의 "내부 개발 문서" 절(빈 줄까지)에서 / 로 시작하는 규칙만 읽는다.
mapfile -t patterns < <(sed -n '/^# 내부 개발 문서/,/^$/p' .gitignore | grep '^/' | sed 's#^/##; s#/$##')

found=()
for pattern in "${patterns[@]}"; do
  # compgen -G 는 일치하는 경로를 한 줄에 하나씩 낸다(공백이 든 이름도 안전).
  while IFS= read -r path; do
    [ -e "$path" ] || continue
    if git ls-files --error-unmatch -- "$path" >/dev/null 2>&1; then
      echo "건너뜀(저장소가 추적 중): $path" >&2
      continue
    fi
    git check-ignore -q -- "$path" || { echo "건너뜀(무시 규칙 밖): $path" >&2; continue; }
    found+=("$path")
  done < <(compgen -G "$pattern" || true)
done

if [ "${#found[@]}" -eq 0 ]; then
  echo "남은 내부 개발 문서가 없다."
  exit 0
fi

printf '남은 내부 개발 문서 %d개:\n' "${#found[@]}"
printf '  %s\n' "${found[@]}"
if [ "$execute" -eq 1 ]; then
  rm -rf -- "${found[@]}"
  echo "지웠다."
else
  echo "지우려면 --execute 를 붙여 다시 실행한다."
fi
