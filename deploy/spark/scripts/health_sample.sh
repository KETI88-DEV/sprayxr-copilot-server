#!/usr/bin/env bash
# Spark 호스트의 메모리·디스크·컨테이너·GPU 표본을 CSV 에 덧붙인다.
#
#   health_sample.sh           표본만 기록한다. 종료 코드 0.
#   health_sample.sh --check   기록한 뒤 기준을 넘은 항목을 한 줄씩 쓴다. 하나라도 있으면 종료 코드 1.
#
# cron (keti 계정):
#   */10 * * * * .../health_sample.sh --check >> .../cron.log 2>&1
#   @reboot sleep 900 && .../health_sample.sh --check >> .../boot-check.log 2>&1
#
# cron 도 --check 로 돌린다. 재시작 횟수는 직전 표본과만 비교하므로, 표본마다 점검하지 않으면
# 표본 사이에 난 재시작이 기록만 되고 드러나지 않는다. 문제 줄만 보려면: grep -v ' ok$' cron.log
#
# -e 는 쓰지 않는다. 한 항목을 못 읽어도 나머지는 기록해야 한다.
# shortcut: health.csv 에 회전이 없다(하루 약 200KB). 100MB 를 넘으면 월별 파일로 나눈다.
set -u

OPS_DIR="${OPS_DIR:-/home/keti/shipcoat/ops}"
PROD_CONTAINERS="${PROD_CONTAINERS:-shipcoat-spark-agent-api-1 shipcoat-spark-mcp-1 shipcoat-spark-vllm-1 shipcoat-spark-quick-vllm-1 shipcoat-spark-web-1}"
MIN_AVAILABLE_MB="${MIN_AVAILABLE_MB:-6144}"
MAX_SWAP_USED_MB="${MAX_SWAP_USED_MB:-8192}"
MAX_DISK_PCT="${MAX_DISK_PCT:-80}"
MAX_MEM_LIMIT_PCT="${MAX_MEM_LIMIT_PCT:-80}"

CHECK=0
[[ "${1:-}" == "--check" ]] && CHECK=1

CSV="${OPS_DIR}/health.csv"
mkdir -p "${OPS_DIR}"
[[ -s "${CSV}" ]] || echo 'ts,metric,name,value' > "${CSV}"

ts="$(date -u +%FT%TZ)"
problems=()
row() { echo "${ts},$1,$2,$3" >> "${CSV}"; }
# 부팅 직후 docker·nvidia-smi 가 응답하지 않아도 멈추지 않는다.
run() { timeout 30 "$@" 2>/dev/null; }

# 직전 표본의 재시작 횟수. 이번 표본을 쓰기 전에 읽는다.
declare -A prev_restarts=()
while read -r name value; do
  prev_restarts["${name}"]="${value}"
done < <(tail -n 400 "${CSV}" | awk -F, '$2 == "container_restarts" { print $3, $4 }')

# --- 호스트
available_mb="$(awk '/^MemAvailable:/ { print int($2 / 1024) }' /proc/meminfo)"
swap_used_mb="$(awk '/^SwapTotal:/ { total = $2 } /^SwapFree:/ { free = $2 } END { print int((total - free) / 1024) }' /proc/meminfo)"
disk_pct="$(df --output=pcent / | tail -n 1 | tr -dc '0-9')"
row mem_available_mb host "${available_mb}"
row swap_used_mb host "${swap_used_mb}"
row disk_used_pct / "${disk_pct}"
(( available_mb < MIN_AVAILABLE_MB )) && problems+=("가용 메모리 ${available_mb}MB < ${MIN_AVAILABLE_MB}MB")
(( swap_used_mb > MAX_SWAP_USED_MB )) && problems+=("스왑 사용 ${swap_used_mb}MB > ${MAX_SWAP_USED_MB}MB")
(( disk_pct > MAX_DISK_PCT )) && problems+=("디스크 사용률 ${disk_pct}% > ${MAX_DISK_PCT}%")

# --- 컨테이너 (실행 중인 것 전부 기록, 점검은 PROD_CONTAINERS 만)
declare -A mem_mb=() limit_mb=() restarts=() health=() name_of_id=()
while read -r name usage _; do
  [[ -n "${name}" ]] || continue
  mem_mb["${name}"]="$(awk -v v="${usage}" 'BEGIN { n = v + 0; u = v; sub(/^[0-9.]+/, "", u);
    print int(u == "GiB" ? n * 1024 : u == "MiB" ? n : u == "KiB" ? n / 1024 : n / 1048576) }')"
done < <(run docker stats --no-stream --format '{{.Name}} {{.MemUsage}}')
while read -r id name; do
  [[ -n "${name}" ]] || continue
  name_of_id["${id}"]="${name}"
  limit="" count="" state=""
  read -r limit count state < <(run docker inspect -f \
    '{{.HostConfig.Memory}} {{.RestartCount}} {{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}' "${name}")
  limit_mb["${name}"]="$(( ${limit:-0} / 1048576 ))"
  restarts["${name}"]="${count:-0}"
  health["${name}"]="${state:-unknown}"
  row container_mem_mb "${name}" "${mem_mb[${name}]:-0}"
  row container_mem_limit_mb "${name}" "${limit_mb[${name}]}"
  row container_restarts "${name}" "${restarts[${name}]}"
  row container_health "${name}" "${health[${name}]}"
done < <(run docker ps --no-trunc --format '{{.ID}} {{.Names}}')

for name in ${PROD_CONTAINERS}; do
  if [[ -z "${health[${name}]:-}" ]]; then
    row container_health "${name}" missing
    problems+=("${name} 컨테이너가 없거나 꺼져 있다")
    continue
  fi
  [[ "${health[${name}]}" == healthy ]] || problems+=("${name} 상태가 ${health[${name}]}")
  now="${restarts[${name}]}" previous="${prev_restarts[${name}]:-}"
  if [[ -n "${previous}" ]] && (( now > previous )); then
    problems+=("${name} 재시작 횟수 ${previous} → ${now}")
  fi
  used="${mem_mb[${name}]:-0}" limit="${limit_mb[${name}]}"
  if (( limit > 0 )) && (( used * 100 > limit * MAX_MEM_LIMIT_PCT )); then
    problems+=("${name} 메모리 ${used}MB 가 한도 ${limit}MB 의 ${MAX_MEM_LIMIT_PCT}% 를 넘었다")
  fi
done

# --- GPU (docker stats 에는 GPU 할당이 들어가지 않는다)
while IFS=', ' read -r pid used; do
  [[ "${pid}" =~ ^[0-9]+$ ]] || continue
  id="$(grep -o -E '[0-9a-f]{64}' "/proc/${pid}/cgroup" 2>/dev/null | head -n 1)"
  row gpu_mem_mb "${name_of_id[${id:-none}]:-pid:${pid}}" "${used}"
done < <(run nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits)

if (( CHECK )); then
  if (( ${#problems[@]} )); then
    for problem in "${problems[@]}"; do echo "${ts} ${problem}"; done
    exit 1
  fi
  echo "${ts} ok"
fi
exit 0
