#!/usr/bin/env bash
# 8130 비교·교정 화면(pose_lab)을 컨테이너 안에 (재)기동한다. 호스트에서 저장소 루트 기준으로 실행한다.
#
#   deploy/start_pose_lab.sh            # 기본 설정(내부 문서 저장소 docs/year2/KPI_시험방법.md §6)
#   CONTAINER=다른이름 deploy/start_pose_lab.sh
#
# 컨테이너 명령은 `sleep infinity`라 `docker compose up -d` 뒤에 8130은 따로 띄워야 한다.
# 2026-10-14에 README의 옛 예시(산포 `_yup` = 옛 동작 키 m1)를 붙여 넣어 띄운 탓에 서버가 산포를 버리고
# 임시 산포(5 cm)로 돌아 기존 CSV 비교 유사도가 거의 0이 됐다. 그래서 설정을 여기 고정하고,
# 산포 파일의 키 접두어가 현행 동작 키와 다르면 띄우지 않는다.
set -euo pipefail

CONTAINER="${CONTAINER:-ship-paint-trainer}"
PORT="${PORT:-8130}"
DATA_ROOT="${POSE_LAB_DATA_ROOT:-/data/sources/professionals}"
SPREAD="${POSE_LAB_SPREAD_REFERENCE:-/output/spread_reference_expert_m3.json}"
ARM="${ARM_REFERENCE_PATH:-/output/expert_reference/expert_arm_reference.json}"
LOG="${POSE_LAB_LOG:-/logs/pose_lab_${PORT}.log}"

in_container() { docker exec -w /workspace "$CONTAINER" "$@"; }

# 1) 입력 확인 — 없는 파일이나 옛 키 산포로 띄우면 서버는 뜨지만 점수가 조용히 무너진다.
in_container test -d "$DATA_ROOT" || { echo "데이터 폴더 없음: $DATA_ROOT" >&2; exit 1; }
in_container test -f "$ARM" || { echo "팔 기준 없음: $ARM" >&2; exit 1; }
# -i가 없으면 heredoc이 전달되지 않아 검사가 빈 입력으로 "통과"한다.
docker exec -i -w /workspace "$CONTAINER" env PYTHONPATH=trainer/scripts:. python3 - "$SPREAD" <<'PY'
import sys
from pathlib import Path
import motion_key
from trainer.pose_lab.spread_reference import SpreadReference
path = Path(sys.argv[1])
if not path.is_file():
    sys.exit(f"산포 파일 없음: {path}")
prefixes = SpreadReference.load(path).key_prefixes()
if prefixes != {motion_key.KEY_PREFIX}:
    sys.exit(f"산포 키 접두어 {sorted(prefixes)} ≠ 현행 {motion_key.KEY_PREFIX!r}: {path} — 임시 산포로 돌게 된다")
print(f"산포 확인: {path} (키 {motion_key.KEY_PREFIX})")
PY

# 2) 떠 있는 서버를 PID로 내린다. `pkill -f`는 bash -c 인자 자신과 겹치면 셸이 죽는다(README 참고).
old_pids=$(docker exec "$CONTAINER" ps -eo pid=,args= \
  | awk -v port="$PORT" '$2=="python3" && /trainer\.pose_lab\.app:create_app/ && $0 ~ ("--port " port "( |$)") {print $1}')
if [ -n "$old_pids" ]; then
  echo "기존 서버 종료: $old_pids"
  docker exec "$CONTAINER" kill $old_pids
  sleep 2
fi

# 3) 기동. 로그는 파일로 — `docker exec -d`의 표준출력은 아무도 읽지 않는 파이프라 경고가 사라진다.
docker exec -d -w /workspace \
  -e POSE_LAB_DATA_ROOT="$DATA_ROOT" \
  -e POSE_LAB_SPREAD_REFERENCE="$SPREAD" \
  -e ARM_REFERENCE_PATH="$ARM" \
  -e PYTHONPATH=trainer/scripts:. \
  "$CONTAINER" sh -c "exec python3 -m uvicorn trainer.pose_lab.app:create_app --factory --host 0.0.0.0 --port $PORT > $LOG 2>&1"

# 4) 떴는지, 산포를 버렸다는 경고가 없는지 본다.
for _ in $(seq 60); do
  if in_container python3 -c "import urllib.request as u; u.urlopen('http://127.0.0.1:$PORT/api/health', timeout=3)" 2>/dev/null; then
    if in_container grep -q "잠정 산포" "$LOG"; then
      echo "경고: 서버가 임시 산포로 돈다 — $LOG 확인" >&2
      in_container grep "잠정 산포" "$LOG" >&2
      exit 1
    fi
    echo "8130 기동 완료 (컨테이너 $CONTAINER, 로그 $LOG)"
    exit 0
  fi
  sleep 2
done
echo "120초 안에 /api/health 응답이 없다 — $LOG 확인" >&2
in_container tail -20 "$LOG" >&2
exit 1
