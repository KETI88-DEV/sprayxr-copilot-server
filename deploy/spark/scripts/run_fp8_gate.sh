#!/usr/bin/env bash
# FP8 품질 게이트 후보 측정 (Spark 호스트에서 실행, 스택이 떠 있어야 한다).
# agent-api 컨테이너 안에서 agentic 30문항을 SEEDS 회 생성하고 run_agentic_eval로 채점한다.
# 판정은 scripts/quality_gate.py (README §17).
#
# 한계(README §17에 같은 내용):
# - generate_agentic_responses의 --seed는 torch 시드만 고정한다. openai 백엔드는 vLLM 요청에 seed를 보내지 않으므로
#   후보의 seed_<n>은 재현 가능한 시드가 아니라 독립 반복 번호다.
# - 기준선은 HF NF4(A40), 후보는 vLLM FP8(Spark)이다. 게이트는 양자화만이 아니라 서빙 스택 전체 차이를 잰다.
set -euo pipefail

SPARK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dc() { docker compose -f "${SPARK_DIR}/compose.yaml" "$@"; }

SEEDS="${SEEDS:-42 43 44 45 46 47 48 49}"
CONFIG="${GATE_CONFIG_PATH:-/workspace/trainer/configs/rag_config_spark_vllm.yaml}"
COMMIT="$(dc exec -T agent-api printenv SHIPCOAT_CODE_COMMIT || echo unknown)"
RUN_NAME="${RUN_NAME:-qwen36_27b_v6_1_fp8_spark_${COMMIT:0:7}}"
RUN_DIR="/output/eval_results/${RUN_NAME}"
EVAL_SET=/data/eval/eval_set_agentic.jsonl

dc exec -T agent-api test -f "${EVAL_SET}" || { echo "missing ${EVAL_SET}: build the bundle with --with-eval-set" >&2; exit 2; }
dc exec -T agent-api python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8120/api/ready', timeout=10)" \
  || { echo "agent-api is not ready (GET /api/ready)" >&2; exit 2; }

dc exec -T agent-api mkdir -p "${RUN_DIR}"
dc exec -T agent-api sh -c "{
  echo 'commit: ${COMMIT}'
  echo 'started_at: '\$(date -u +%Y-%m-%dT%H:%M:%SZ)
  echo 'env: TOOL_ADAPTER_NATIVE_TOOLS='\${TOOL_ADAPTER_NATIVE_TOOLS}' TOOL_ADAPTER_FORCE_SYNTHESIS='\${TOOL_ADAPTER_FORCE_SYNTHESIS}
  echo 'seeds: ${SEEDS}'
  python3 -c \"import hashlib,sys; [print(hashlib.sha256(open(p,'rb').read()).hexdigest(), p) for p in sys.argv[1:]]\" ${EVAL_SET} ${CONFIG} /output/indexes/faiss.index /output/indexes/metadata.json
} > ${RUN_DIR}/RUN_INFO.txt"

for seed in ${SEEDS}; do
  out="${RUN_DIR}/seed_${seed}"
  echo "== seed ${seed} -> ${out}"
  dc exec -T agent-api mkdir -p "${out}/eval"
  # --qlora-config 에 없는 경로: 어댑터 기본값을 찾지 않는다(openai 백엔드는 어차피 무시).
  dc exec -T agent-api python3 trainer/scripts/generate_agentic_responses.py \
    --config "${CONFIG}" \
    --mcp-url http://mcp:8100 \
    --qlora-config /nonexistent/qlora_config.yaml \
    --eval-set "${EVAL_SET}" \
    --seed "${seed}" \
    --output "${out}/agentic_responses.jsonl"
  dc exec -T agent-api python3 trainer/scripts/run_agentic_eval.py \
    --eval-set "${EVAL_SET}" \
    --responses "${out}/agentic_responses.jsonl" \
    --output-dir "${out}/eval"
done

echo "done: ${RUN_DIR} (host: \${RESULTS_DIR}/${RUN_NAME})"
