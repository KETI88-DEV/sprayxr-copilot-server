#!/usr/bin/env bash
# 장시간 소크 (Spark 호스트에서 실행): copilot/query knowledge_search를 반복 호출하며
# 응답 코드·지연, 컨테이너 재시작, vLLM/agent-api 로그의 크래시 흔적을 기록한다(README §16).
#
#   DURATION_SEC=28800 API_URL=http://127.0.0.1:58000 deploy/spark/scripts/soak.sh
#
# 종료 코드: 0 합격(비-200 0건, 재시작 0회, 크래시 패턴 0건), 1 불합격.
set -euo pipefail

SPARK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dc() { docker compose -f "${SPARK_DIR}/compose.yaml" "$@"; }

API_URL="${API_URL:-http://127.0.0.1:58000}"
DURATION_SEC="${DURATION_SEC:-28800}"
PAUSE_SEC="${PAUSE_SEC:-1}"
RESOURCE_EVERY="${RESOURCE_EVERY:-50}"
OUT_DIR="${OUT_DIR:-./soak_$(date -u +%Y%m%dT%H%M%SZ)}"
CRASH_PATTERN='CUDA error|illegal memory access|Xid|EngineDeadError|EngineCore.*(died|failed|exception)|Segmentation fault|core dumped|Traceback'

mkdir -p "${OUT_DIR}"
START_ISO="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

container_state() { docker inspect -f '{{.RestartCount}} {{.State.StartedAt}}' "$(dc ps -q "$1")"; }
declare -A before
for svc in vllm agent-api mcp; do before[$svc]="$(container_state "$svc")"; done

prompts=(
  "무기질 아연 프라이머의 도장 조건을 알려줘"
  "상대습도가 높을 때 도장을 중지해야 하는 기준은?"
  "DFT 90/10 규칙이 무엇인지 설명해줘"
  "블라스팅 후 표면 처리 등급 Sa 2.5의 의미는?"
  "에폭시 도료의 재도장 간격에 영향을 주는 요인은?"
)

printf 'ts\tseq\thttp_code\tseconds\tstatus\n' > "${OUT_DIR}/requests.tsv"
printf 'ts\tseq\tmem_used_mb\tmem_available_mb\tgpu_apps\n' > "${OUT_DIR}/resources.tsv"

end=$((SECONDS + DURATION_SEC))
seq=0
while (( SECONDS < end )); do
  seq=$((seq + 1))
  prompt="${prompts[$((seq % ${#prompts[@]}))]}"
  body="$(python3 -c 'import json,sys; print(json.dumps({"user_id":"soak","session_id":"soak","request_id":"soak-%06d" % int(sys.argv[2]),"prompt":sys.argv[1],"task_type":"knowledge_search","use_rag":"Y"}, ensure_ascii=False))' "${prompt}" "${seq}")"
  result="$(curl -sS -o "${OUT_DIR}/last_body.json" -w '%{http_code}\t%{time_total}' --max-time 600 \
    -H 'Content-Type: application/json' -d "${body}" "${API_URL}/api/v1/copilot/query" 2>/dev/null || printf '000\t0')"
  status="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("status",""))' "${OUT_DIR}/last_body.json" 2>/dev/null || true)"
  printf '%s\t%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "${seq}" "${result}" "${status}" >> "${OUT_DIR}/requests.tsv"
  if (( seq % RESOURCE_EVERY == 1 )); then
    mem="$(free -m | awk '/^Mem:/ {print $3"\t"$7}')"
    gpu="$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | tr '\n' ';' || true)"
    printf '%s\t%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "${seq}" "${mem}" "${gpu}" >> "${OUT_DIR}/resources.tsv"
  fi
  sleep "${PAUSE_SEC}"
done

restarts=0
for svc in vllm agent-api mcp; do
  after="$(container_state "$svc" || echo 'gone gone')"
  echo "${svc}: before=${before[$svc]} after=${after}" >> "${OUT_DIR}/containers.txt"
  [[ "${after}" == "${before[$svc]}" ]] || restarts=$((restarts + 1))
done
dc logs --no-color --since "${START_ISO}" vllm agent-api > "${OUT_DIR}/logs.txt" 2>&1 || true
grep -Ei "${CRASH_PATTERN}" "${OUT_DIR}/logs.txt" > "${OUT_DIR}/crash_hits.txt" || true
crash_hits="$(wc -l < "${OUT_DIR}/crash_hits.txt")"

python3 - "${OUT_DIR}" "${restarts}" "${crash_hits}" <<'PY'
import csv, json, statistics, sys
out, restarts, crash_hits = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
rows = list(csv.DictReader(open(f"{out}/requests.tsv"), delimiter="\t"))
ok = [float(r["seconds"]) for r in rows if r["http_code"] == "200" and r["status"] == "SUCCESS"]
bad = [r for r in rows if not (r["http_code"] == "200" and r["status"] == "SUCCESS")]
def pct(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(round(q * (len(values) - 1))))] if values else None
summary = {
    "requests": len(rows), "success": len(ok), "failures": len(bad),
    "latency_s": {"p50": pct(ok, 0.5), "p95": pct(ok, 0.95), "max": max(ok) if ok else None},
    "first_100_p50_s": statistics.median(ok[:100]) if ok else None,
    "last_100_p50_s": statistics.median(ok[-100:]) if ok else None,
    "container_restarts_or_recreations": restarts, "crash_log_lines": crash_hits,
}
summary["pass"] = bool(rows) and not bad and restarts == 0 and crash_hits == 0
json.dump(summary, open(f"{out}/summary.json", "w"), indent=2)
print(json.dumps(summary, indent=2))
sys.exit(0 if summary["pass"] else 1)
PY
