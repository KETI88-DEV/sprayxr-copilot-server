#!/usr/bin/env bash
# vLLM 컨테이너 엔트리포인트: 환경 변수로 `vllm serve` 인자를 조립한다.
#
# - D6: 서버 tool parser·reasoning parser를 붙이지 않는다. VLLM_EXTRA_ARGS로 들어와도 거부한다.
# - VLLM_API_KEY는 인자로 넘기지 않는다(ps에 노출). vLLM이 같은 이름의 환경 변수를 읽는다.
# - VLLM_DRY_RUN=1 이면 argv를 한 줄에 하나씩 출력하고 끝낸다(정합성 테스트용).
set -euo pipefail

: "${SERVED_MODEL_NAME:?SERVED_MODEL_NAME is required}"
: "${MAX_MODEL_LEN:?MAX_MODEL_LEN is required}"

args=(
  serve "${VLLM_MODEL_PATH:-/models/shipcoat}"
  --host 0.0.0.0
  --port 8000
  --served-model-name "${SERVED_MODEL_NAME}"
  --max-model-len "${MAX_MODEL_LEN}"
  --max-num-seqs "${MAX_NUM_SEQS:-4}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.72}"
  --enable-prefix-caching
  --generation-config "${GENERATION_CONFIG:-vllm}"
)

if [[ -n "${OVERRIDE_GENERATION_CONFIG:-}" ]]; then
  args+=(--override-generation-config "${OVERRIDE_GENERATION_CONFIG}")
fi

case "${ENABLE_MTP:-0}" in
  1)
    tokens="${MTP_NUM_SPECULATIVE_TOKENS:-1}"
    if [[ ! "${tokens}" =~ ^[1-9][0-9]*$ ]]; then
      echo "MTP_NUM_SPECULATIVE_TOKENS must be a positive integer, got '${tokens}'" >&2
      exit 2
    fi
    args+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${tokens}}")
    ;;
  0) ;;
  *)
    echo "ENABLE_MTP must be 0 or 1, got '${ENABLE_MTP}'" >&2
    exit 2
    ;;
esac

case "${LANGUAGE_MODEL_ONLY:-0}" in
  1) args+=(--language-model-only) ;;
  0) ;;
  *)
    echo "LANGUAGE_MODEL_ONLY must be 0 or 1, got '${LANGUAGE_MODEL_ONLY}'" >&2
    exit 2
    ;;
esac

if [[ -n "${VLLM_EXTRA_ARGS:-}" ]]; then
  read -r -a extra <<< "${VLLM_EXTRA_ARGS}"
  args+=("${extra[@]}")
fi

for arg in "${args[@]}"; do
  case "${arg}" in
    --enable-auto-tool-choice* | --tool-call-parser* | --reasoning-parser* | --api-key*)
      echo "refusing ${arg%%=*}: tool/reasoning parsers are off by plan D6; pass the API key via VLLM_API_KEY" >&2
      exit 2
      ;;
  esac
done

if [[ "${VLLM_DRY_RUN:-0}" == "1" ]]; then
  printf '%s\n' vllm "${args[@]}"
  exit 0
fi

exec vllm "${args[@]}"
