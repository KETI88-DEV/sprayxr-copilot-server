#!/usr/bin/env bash
# keti 계정의 `docker` 앞에 서서, 운영 잠금 없이 운영 컨테이너를 바꾸는 명령을 거부한다.
#
# 설치(Spark): ~/bin/docker 로 복사하고 ~/.bashrc·~/.profile 맨 위에서 PATH 앞에 ~/bin 을 둔다.
#   잠금:  python3 ~/shipcoat/ops/ops_lock.py take --what agent-api --by 이름 --reason ... --until HH:MM
#          export OPS_LOCK_BY=이름
# 읽기 명령(ps·inspect·logs·exec·images·stats 등)은 그대로 통과한다.
# 바꾸는 명령(run·create·start·stop·restart·kill·rm·rename·update·pause·unpause·compose up/down/…)이
# 운영 이름(shipcoat-spark-*)이나 shipcoat compose 를 건드리면 OPS_LOCK_BY 의 잠금을 확인한다.
# 우회(/usr/bin/docker 직접 호출, docker.sock API)는 막지 못한다. 그 변화는 10분 점검이 Slack 으로 경고한다.
set -u
REAL="${DOCKER_REAL:-/usr/bin/docker}"  # 시험용으로만 바꾼다
LOCK="${OPS_LOCK:-/home/keti/shipcoat/ops/ops_lock.py}"

mutating=0
case "${1:-}" in
  run|create|start|stop|restart|kill|rm|rename|update|pause|unpause) mutating=1 ;;
  container)
    case "${2:-}" in run|create|start|stop|restart|kill|rm|rename|update|pause|unpause|prune) mutating=1 ;; esac ;;
  compose)
    for arg in "$@"; do
      case "$arg" in up|down|start|stop|restart|kill|rm|create|run|pause|unpause) mutating=1; break ;; esac
    done ;;
esac

if (( mutating )); then
  touches=0
  for arg in "$@" "$PWD"; do
    case "$arg" in *shipcoat-spark*|*/shipcoat/*|*/shipcoat) touches=1; break ;; esac
  done
  if (( touches )); then
    if [[ -z "${OPS_LOCK_BY:-}" ]] || ! python3 "$LOCK" check --by "$OPS_LOCK_BY" >/dev/null 2>&1; then
      {
        echo "운영 잠금 없이 운영 컨테이너를 바꾸려 해서 멈췄다: docker $*"
        python3 "$LOCK" status 2>/dev/null | sed 's/^/  지금 잠금: /'
        echo "  먼저: python3 $LOCK take --what agent-api|vllm|test|other --by 이름 --reason 이유 --until HH:MM"
        echo "        export OPS_LOCK_BY=이름"
      } >&2
      exit 3
    fi
  fi
fi
exec "$REAL" "$@"
