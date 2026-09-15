# Spark 완성품 패키지: vllm / mcp / agent-api / web

DGX Spark(aarch64, GB10 128GB 통합 메모리, CUDA 13)에 올리는 배포 정의와 절차다. 상위 계획은 `docs/superpowers/plans/2026-09-15-spark-rag-first-release.md`(D1~D7, T8)다.

- **이미지**에는 실행 코드와 고정 의존성만 넣는다(`images/`, `requirements/`).
- **번들**(`BUNDLE_DIR`)은 모델, 임베딩, 리랭커, RAG 인덱스, 기준 데이터다. sha256 매니페스트로 반입을 검증한다(`scripts/`).
- 모든 단계에 **합격 기준**이 있다. 이 문서를 쓸 때 확인하지 못한 것은 **미확인**으로 표시하고 [§19](#19-미확인-항목)에 모았다.
- 이 저장소의 x86_64 개발 서버에서는 이미지를 빌드하지 않는다.

## 0. 구성

| 서비스 | 이미지 | 컨테이너 포트 | 호스트 노출(기본) | GPU | 역할 |
|---|---|---|---|---|---|
| `vllm` | `VLLM_IMAGE` (1순위 `vllm/vllm-openai:v0.29.0`, 대안 `nvcr.io/nvidia/vllm:26.08-py3`) | 8000 | `127.0.0.1:18000` | O | FP8 병합본 서빙. OpenAI 호환 API |
| `mcp` | `images/agent.Dockerfile` target `mcp` | 8100 | 없음 | X | 도구 7종. 단일 replica |
| `agent-api` | `images/agent.Dockerfile` target `agent-api` | 8120 | `127.0.0.1:58000` (`COPILOT_BIND`/`COPILOT_PORT`) | O (bge-m3, reranker) | `trainer.demo_api.app` live 모드, 단일 워커 |
| `web` | `images/web.Dockerfile` | 80 | `0.0.0.0:8080` | X | demo-ui 정적 파일. `/api` same-origin 프록시 |

기동 순서는 `vllm`(healthy) → `mcp`(healthy) → `agent-api` → `web`이다. 통합 메모리 환경이라 vLLM이 먼저 메모리를 잡아야 `--gpu-memory-utilization` 계산이 흔들리지 않는다.

```
deploy/spark/
├── compose.yaml                  # 네 서비스, 헬스체크, 볼륨, GPU
├── .env.example                  # 모든 가변 값(태그/digest, 경로, 포트, MTP 토글)
├── vllm/entrypoint.sh            # vllm serve 인자 조립(D6 금지 플래그 차단, VLLM_DRY_RUN)
├── images/
│   ├── agent.Dockerfile(+.dockerignore)   # mcp / agent-api 공용, linux/arm64
│   └── web.Dockerfile(+.dockerignore)     # node 빌드 → nginx
├── web/nginx.conf                # SSE 버퍼링 off, /api → agent-api:8120
├── requirements/
│   ├── agent-api.lock.txt        # 개발 컨테이너 설치본에서 뽑은 고정 버전(근거는 파일 머리말)
│   └── mcp.lock.txt
└── scripts/
    ├── prepare_bundle.py         # 번들 디렉터리 생성(A40)
    ├── bundle_manifest.py        # create / verify (표준 라이브러리만, 호스트 python3)
    ├── check_tokenizer_parity.py # HF 토큰 ID vs vLLM /tokenize
    ├── render_rollback_config.py # BF16 롤백 설정
    ├── run_fp8_gate.sh           # 30문항 × RUNS회(기본 8) 생성·채점, 기존 run에 이어 붙이기
    ├── quality_gate.py           # NF4 기준선 대비 단계적 95% Welch CI 판정(exit 0/1/2/3)
    └── soak.sh                   # 장시간 소크
```

`trainer/scripts/tests/test_spark_deploy_contract.py`는 compose와 entrypoint의 값을 `rag_config_spark_vllm.yaml`, 번들 배치, 코드와 대조한다(served name, max-model-len, 샘플링 override, 마운트 경로, nginx 타임아웃, lock). `test_spark_bundle_scripts.py`는 스크립트를 합성 디렉터리로 검증한다.

### 코드의 고정 경로

- `agent_pipeline.py`와 `tool_calling_adapter.py`는 `/workspace/trainer/configs/{tool_schemas,tool_calling_protocol.json}`를 코드에 고정해 쓴다. 그래서 이미지의 `WORKDIR`을 `/workspace`로 두고 코드를 그 아래에 복사한다.
- 설정의 자산 경로(`/models/bge-m3`, `/models/bge-reranker-v2-m3`, `/output/indexes/*`, `/output/merged/qwen36_27b_shipcoat_v6_1_fp8`)는 번들 안의 상대 경로와 1:1이다. compose 볼륨이 이 경로를 그대로 연결하므로 설정 파일을 고치지 않는다.

## 1. 번들 만들기 (A40 서버, 개발 컨테이너 안)

```bash
# 계획 확인(쓰기 없음). BF16 롤백을 함께 보낼 거면 --with-bf16
docker exec -w /workspace ship-paint-trainer nice -n 19 python3 deploy/spark/scripts/prepare_bundle.py \
  --bundle-dir /output/bundle/spark-v6.1-fp8 --with-eval-set --dry-run

# 생성. --link: 같은 파일시스템이면 하드링크(모델을 복사하지 않음)
docker exec -w /workspace ship-paint-trainer nice -n 19 ionice -c3 python3 deploy/spark/scripts/prepare_bundle.py \
  --bundle-dir /output/bundle/spark-v6.1-fp8 --with-eval-set --link

# 매니페스트. --code-commit은 이미지에 굽는 코드 커밋. 이미 아는 이미지 digest는 --image 로 기록
docker exec -w /workspace ship-paint-trainer nice -n 19 ionice -c3 python3 deploy/spark/scripts/bundle_manifest.py create \
  --bundle-dir /output/bundle/spark-v6.1-fp8 --code-commit "$(git rev-parse HEAD)" \
  --image vllm=vllm/vllm-openai:v0.29.0@sha256:<digest>
```

번들에 들어가는 것(자산 경로 기준):

- **bge-m3**
  - HF 캐시 `refs/main` 스냅숏에는 `pytorch_model.bin`만 있다.
  - `search_index`는 `use_safetensors=True`로 읽는데, 오프라인에서는 safetensors 자동 변환이 네트워크를 요구한다.
  - 그래서 `refs/pr/130` 스냅숏(`9a0624b8`)의 `model.safetensors`를 main 스냅숏 파일과 합친 로컬 디렉터리를 만든다. `pytorch_model.bin`은 기본으로 제외한다.
  - 2026-09-15에 `HF_HUB_OFFLINE=1`, CPU로 확인했다. 생성된 디렉터리를 `search_index.load_embedder`와 `load_reranker`로 로드하고 encode/predict까지 성공했다.
- **FP8 모델**: `fp8_manifest.json`의 `files_sha256`(78개)과 번들 해시를 대조한다. 하나라도 다르면 매니페스트를 쓰지 않는다. BF16은 `merge_manifest.json`의 `files`와 대조한다.
- **RAG 인덱스**: `faiss.index`와 `metadata.json`만 넣는다(`*.bak_prerepair` 등은 제외). 2026-09-15 기준 `ntotal`은 10829다.
- **MCP 기준 데이터**
  - `mcp_servers`가 읽는 파일은 `pose_estimation`의 `POSE_REFERENCE_PATH`(기본 `/output/expert_reference/expert_pose_reference.json`) 하나다.
  - **현재 A40 기본 경로에 이 파일이 없다.** 번들에서도 빠지고 `bundle_sources.json`의 `absent_optional`에 기록된다.
  - 파일이 없으면 pose 비교는 기준을 추정하지 않고 None을 낸다. knowledge_search와는 무관하다.
- **평가셋**: `--with-eval-set`을 주면 `data/eval/eval_set_agentic.jsonl`이 들어간다. §17 게이트에 필요하다.

2026-09-15 실측 크기(`bundle_manifest.py create`, 101파일):

| 묶음 | 파일 | 바이트 |
|---|---:|---:|
| model_fp8 | 79 | 30,890,018,611 |
| embedding (bge-m3, bin 제외) | 11 | 2,293,250,249 |
| reranker | 7 | 2,293,259,337 |
| rag_index | 2 | 59,070,331 |
| eval_set | 1 | 33,899 |
| 합계 (FP8 번들) | 101 | **35,535,633,534 (약 33.1 GiB)** |
| model_bf16 (`--with-bf16` 추가 시) | 28 | +55,586,084,951 → 합계 약 91.1 GB |

해시 시간은 페이지 캐시가 따뜻할 때 `--jobs 4`로 약 17초였다. 캐시가 차가울 때는 측정하지 않았다(미측정).

**합격 기준**

- `create`가 exit 0으로 끝나고, `provenance.model_fp8.mismatched`와 `missing`이 모두 `[]`이다.
- `absent_optional`에 적힌 항목은 모두 의도한 것이다.
- `manifest_sha256`을 반입 경로와 **다른 채널**(메신저, 메일)로 Spark 담당자에게 보낸다.

> `--link`를 쓰면 번들 파일이 `/output/merged`와 inode를 공유한다. 원본을 제자리에서 고치지 않는다. rsync나 외장 디스크는 내용을 복사하므로 반입본에는 영향이 없다.

## 2. 이미지

### A. Spark에서 네이티브 빌드 (권장, 인터넷 필요)

```bash
# Spark: 릴리스 커밋의 저장소 사본에서
cd deploy/spark && cp .env.example .env    # CODE_COMMIT=$(git rev-parse HEAD) 등 채움(§4)
docker compose build mcp agent-api web     # platforms: linux/arm64
docker pull "$(grep ^VLLM_IMAGE= .env | cut -d= -f2)"
docker image inspect --format '{{index .RepoDigests 0}}' vllm/vllm-openai:v0.29.0   # → .env VLLM_IMAGE 에 digest 고정
docker image inspect --format '{{.Id}}' shipcoat/agent-api:v6.1 shipcoat/mcp:v6.1 shipcoat/web:v6.1  # 릴리스 기록에 남김
```

- agent-api에는 PyTorch cu130 인덱스의 `torch==2.9.1+cu130`(cp311 manylinux_2_28_aarch64 휠)이 들어간다. 이 휠이 인덱스에 있는 것은 2026-09-15에 확인했다.
- 그 위에 `requirements/agent-api.lock.txt`를 `--no-deps`로 설치하고 `pip check`를 돌린다.
- 빌드 마지막에 import 스모크를 돌린다(`trainer.scripts.agent_pipeline`, `search_index`, `trainer.demo_api.app`, 그리고 `torch.version.cuda`가 13.x인지).
- web은 `npm ci && npm run build:web`이다. `VITE_API_BASE_URL`을 주지 않으므로 same-origin으로 호출한다(빌드 산출물 `index.html`에 `%VITE_API_BASE_URL%`이 남는 것을 확인했다).
- 태그가 linux/arm64 매니페스트를 갖고 있는지는 2026-09-15 `docker manifest inspect`로 확인했다: `python:3.11.9-slim-bookworm`, `node:24.18.0-bookworm-slim`, `nginx:1.29.4-alpine`, `vllm/vllm-openai:v0.29.0`, `nvcr.io/nvidia/vllm:26.08-py3`. 베이스 이미지 digest 고정이 필요하면 `PYTHON_IMAGE`, `NODE_IMAGE`, `NGINX_IMAGE` build arg에 `이름@sha256:...`을 준다.

**합격 기준**

- 세 이미지의 빌드가 모두 성공한다. `pip check`와 import 스모크가 실패하면 빌드가 중단된다.
- `docker image inspect -f '{{.Architecture}}'`가 `arm64`다.
- `docker run --rm shipcoat/agent-api:v6.1 printenv SHIPCOAT_CODE_COMMIT`가 매니페스트의 `code_commit`과 같다.

### B. 다른 x86 호스트에서 buildx로 빌드 후 반입 (Spark가 오프라인일 때)

```bash
docker run --privileged --rm tonistiigi/binfmt --install arm64
docker buildx create --use
CODE_COMMIT=$(git rev-parse HEAD) docker buildx bake -f deploy/spark/compose.yaml --env-file deploy/spark/.env mcp agent-api web --load
docker pull --platform linux/arm64 vllm/vllm-openai:v0.29.0
docker save shipcoat/agent-api:v6.1 shipcoat/mcp:v6.1 shipcoat/web:v6.1 vllm/vllm-openai:v0.29.0 | zstd -T0 -o images-v6.1.tar.zst
sha256sum images-v6.1.tar.zst > images-v6.1.tar.zst.sha256
# Spark
sha256sum -c images-v6.1.tar.zst.sha256 && zstd -dc images-v6.1.tar.zst | docker load
```

QEMU 에뮬레이션에서의 빌드 시간은 측정하지 않았다(미측정). 이 개발 서버에는 buildx가 없고, 이 서버에서 빌드하는 것도 금지다.

## 3. 반입과 체크섬 검증

```bash
# rsync (-H: 하드링크 보존, 대상은 ext4/xfs. exFAT는 권한·하드링크 불가)
rsync -aH --info=progress2 a40:/data/ship-paint-ai/output/bundle/spark-v6.1-fp8/ /srv/shipcoat/bundle/spark-v6.1-fp8/

# Spark 호스트 python3(표준 라이브러리만)로 검증
python3 deploy/spark/scripts/bundle_manifest.py verify \
  --bundle-dir /srv/shipcoat/bundle/spark-v6.1-fp8 --model fp8 \
  --code-commit <릴리스 커밋> --image-digest vllm=sha256:<digest>
sha256sum /srv/shipcoat/bundle/spark-v6.1-fp8/BUNDLE_MANIFEST.json   # §1에서 따로 받은 값과 비교
```

**합격 기준**

- exit 0, `"ok": true`, `"mode": "full"`, `"problems": []`, `files_checked`가 매니페스트의 `file_count`와 같다.
- `BUNDLE_MANIFEST.json`의 sha256이 따로 받은 값과 같다.
- `warnings`(매니페스트에 없는 파일)는 전부 설명할 수 있어야 한다. `config/`는 검사 대상이 아니다.
- `--quick`(크기만 확인)은 재기동 전 빠른 재확인용이다. 같은 크기의 손상은 잡지 못하므로 반입 직후에는 반드시 full로 돌린다.

## 4. `.env`와 compose 검증

```bash
cd deploy/spark && cp .env.example .env    # .env는 커밋하지 않는다
# 채울 값: BUNDLE_DIR, RESULTS_DIR, APP_UID/APP_GID, CODE_COMMIT, VLLM_IMAGE(@digest), VLLM_API_KEY
mkdir -p "$RESULTS_DIR" && sudo chown "$APP_UID:$APP_GID" "$RESULTS_DIR"
# 없는 마운트 원본을 docker가 root 소유로 만들지 않게 미리 만든다
mkdir -p "$BUNDLE_DIR/output/expert_reference" "$BUNDLE_DIR/data/eval" "$BUNDLE_DIR/config"
docker compose config -q && echo COMPOSE_OK
```

`COPILOT_BIND`나 `VLLM_BIND`를 `0.0.0.0`으로 열 경우 `VLLM_API_KEY`를 설정한다. 이 키는 vLLM이 환경 변수로 읽고, agent-api는 `llm.api_key_env`로 읽는다.

**합격 기준**

- `COMPOSE_OK`가 출력된다.
- 개발 서버에서도 2026-09-15에 `docker compose -f deploy/spark/compose.yaml --env-file deploy/spark/.env.example config`가 exit 0으로 끝났다. `OVERRIDE_GENERATION_CONFIG` JSON이 그대로 전달되는 것도 확인했다.

## 5. 호스트·이미지 GPU 점검

```bash
nvidia-smi
free -g
docker run --rm --gpus all shipcoat/agent-api:v6.1 python3 -c "
import torch, faiss
print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0), torch.cuda.get_arch_list())
x = torch.randn(1024, 1024, device='cuda'); print(float((x @ x).sum()))"
```

**합격 기준**

- `is_available() == True`, capability `(12, 1)`이다.
- 행렬곱이 `no kernel image is available` 같은 오류 없이 끝난다.
- torch cu130 aarch64 휠이 GB10(SM121) 커널을 지원하는지는 **미확인**이다. 실패하면 `TORCH_VERSION` build arg를 더 새로운 cu130 휠(2.10~2.14 aarch64 존재 확인)로 올려 다시 빌드한다.

## 6. vLLM 단독 기동

```bash
docker compose up -d vllm
docker compose logs -f vllm            # 로드 경과 확인, Ctrl-C
curl -fsS http://127.0.0.1:18000/health && echo HEALTH_OK
curl -sS http://127.0.0.1:18000/v1/models ${VLLM_API_KEY:+-H "Authorization: Bearer $VLLM_API_KEY"} | python3 -m json.tool
docker compose logs vllm | grep -Ei "quantiz|fp8|KV cache|maximum concurrency|override|generation config|took"
```

**합격 기준**

- 컨테이너 상태가 `healthy`이고 `/health`가 200이다.
- `/v1/models`의 `data[0].id`가 `qwen36-27b-shipcoat-v6-1-fp8`, `max_model_len`이 `8192`다.
- 로그에서 FP8(block 128) 양자화 로드를 확인할 수 있다.
- 기동 시간(컨테이너 시작부터 healthy까지), KV 캐시 크기, 최대 동시성을 **기록**한다. `start_period`(1800s)가 충분한지 여기서 판단한다.
- `OVERRIDE_GENERATION_CONFIG`가 적용됐다는 로그가 있어야 한다. `--generation-config vllm`과 함께 쓸 때 override가 반영되는지는 **미확인**이며, 로그로 확인한다.

확인 항목(**미확인**): `--language-model-only`

1. `LANGUAGE_MODEL_ONLY=1`, `ENABLE_MTP=0`으로 `docker compose up -d --force-recreate vllm` 한다. 기동 성공 여부와 KV 캐시 크기 변화를 기록한다.
2. 1이 성공하면 `ENABLE_MTP=1`을 함께 켜서 기동한다.
3. 둘을 동시에 쓸 수 있는지(MTP 헤드 로드 포함) 결과를 기록한다.
4. 결론이 날 때까지 기본값은 `LANGUAGE_MODEL_ONLY=0`이다.

플래그 이름은 로컬 vLLM v0.22.1 소스(`engine/arg_utils.py`)에서 `--language-model-only`와 `--override-generation-config`를 확인했다. v0.29.0에서는 확인하지 않았다.

## 7. 단일 응답

```bash
curl -sS http://127.0.0.1:18000/v1/chat/completions -H 'Content-Type: application/json' \
  ${VLLM_API_KEY:+-H "Authorization: Bearer $VLLM_API_KEY"} -d '{
  "model": "qwen36-27b-shipcoat-v6-1-fp8",
  "messages": [
    {"role": "system", "content": "너는 선박 도장 분야 전문 AI 어시스턴트이다."},
    {"role": "user", "content": "무기질 아연 프라이머를 도장할 때 확인할 환경 조건을 두 문장으로 설명해줘."}
  ],
  "max_tokens": 256, "temperature": 0.3, "top_p": 0.8, "top_k": 20, "repetition_penalty": 1.15,
  "chat_template_kwargs": {"enable_thinking": false}, "skip_special_tokens": true, "stream": false
}' | python3 -m json.tool
```

**합격 기준**

- HTTP 200, `choices[0].finish_reason == "stop"`이다.
- `message.content`는 한국어 답변이고, `<think>`, `</think>`, `<tool_call>` 문자열이 없다.
- `usage.prompt_tokens`와 `completion_tokens`가 0보다 크다.
- 요청 시간(curl `-w '%{time_total}'`)과 `completion_tokens`로 대략의 디코딩 속도(tok/s)를 **기록**한다.

## 8. 스트리밍

```bash
curl -N -sS http://127.0.0.1:18000/v1/chat/completions -H 'Content-Type: application/json' \
  ${VLLM_API_KEY:+-H "Authorization: Bearer $VLLM_API_KEY"} -d '{
  "model": "qwen36-27b-shipcoat-v6-1-fp8",
  "messages": [{"role": "user", "content": "도막 두께 90/10 규칙을 설명해줘."}],
  "max_tokens": 256, "temperature": 0.3, "top_p": 0.8, "top_k": 20, "repetition_penalty": 1.15,
  "chat_template_kwargs": {"enable_thinking": false}, "stream": true, "stream_options": {"include_usage": true}
}' | while IFS= read -r line; do printf '%s %s\n' "$(date +%T.%3N)" "${line:0:120}"; done
```

**합격 기준**

- `data:` 줄이 시간차를 두고 여러 개 도착한다. 첫 청크 도착 시각을 TTFT로 기록한다.
- 청크 하나에 `finish_reason`이 있고, 마지막 줄이 `data: [DONE]`이다.
- 중간에 끊겨 `finish_reason` 없이 끝나면 불합격이다. agent-api의 openai 백엔드는 이 경우를 `OpenAIBackendUnavailableError`로 처리한다.

## 9. HF 토큰 ID vs vLLM `/tokenize`

```bash
docker compose run --rm --no-deps agent-api python3 deploy/spark/scripts/check_tokenizer_parity.py
```

- 사례는 다섯 가지다. 채팅 3종(fast 경로 RAG, 도구 시스템 프롬프트와 환경 JSON, `<tool_call>`/`tool` 응답 재생)은 `enable_thinking=false`와 `add_generation_prompt`로 만들고, 원문 2종(한국어 단위 기호, 프로토콜 태그)은 `add_special_tokens=false`로 만든다.
- 비교 규약은 `prompt_budget.TokenizerTokenCounter`와 같다.
- HF 쪽 렌더링은 개발 서버에서 FP8 토크나이저로 확인했다(tool 응답이 `<tool_response>`로 렌더링된다). vLLM `/tokenize`의 chat 요청 스키마는 v0.29.0에서 **미확인**이다.

**합격 기준**: exit 0, 모든 사례에서 `"equal": true`. 불일치하면 `first_difference`와 앞뒤 토큰을 기록하고 FP8 게이트(§17) 전에 원인을 해결한다.

## 10. agent-api·mcp 기동과 `/api/ready`

```bash
docker compose up -d mcp agent-api
docker compose logs -f agent-api      # "eager live pipeline initialization finished"
curl -sS -o /dev/stderr -w '%{http_code}\n' http://127.0.0.1:58000/api/ready | tail -1
curl -sS http://127.0.0.1:58000/api/ready | python3 -m json.tool
```

**합격 기준**

- HTTP 200, `ready: true`, `mode: "live"`이다.
- `checks.pipeline`: `{ok: true, loaded: true, rag_vectors: 10829}`
- `checks.llm`: `{ok: true, backend: "openai", health: true, model_served: true}`
- `checks.mcp.ok`가 `true`다(정보용, `required: false`).
- 컨테이너 상태가 `healthy`다.
- `rag_vectors`가 10829가 아니면 번들 인덱스가 바뀐 것이므로 §3으로 돌아간다.

## 11. copilot/query (v2.5)

```bash
Q=http://127.0.0.1:58000/api/v1/copilot/query
# knowledge_search, RAG 사용
curl -sS -w '\ntime_total=%{time_total}\n' $Q -H 'Content-Type: application/json' -d '{
  "user_id": "keti-test", "scenario_id": "TR-L01", "session_id": "sess-001", "request_id": "req-0001",
  "prompt": "무기질 아연 프라이머의 도장 조건은?", "task_type": "knowledge_search", "use_rag": "Y"}'
# RAG 미사용
curl -sS $Q -H 'Content-Type: application/json' -d '{"user_id":"keti-test","request_id":"req-0002","prompt":"도막 두께 90/10 규칙은?","task_type":"knowledge_search","use_rag":"N"}'
# not_available 3종
for t in trajectory_recomm pose_recomm error_analysis; do
  curl -sS -w '  [%{http_code} %{time_total}s]\n' $Q -H 'Content-Type: application/json' \
    -d "{\"user_id\":\"keti-test\",\"request_id\":\"req-$t\",\"prompt\":\"확인 요청\",\"task_type\":\"$t\",\"use_rag\":\"Y\"}"
done
# 검증 오류(v2.5 ERROR 래퍼)
curl -sS -w '  [%{http_code}]\n' $Q -H 'Content-Type: application/json' -d '{"user_id":"u","request_id":"r","prompt":"p","task_type":"knowledge_search","use_rag":true}'
```

**합격 기준**

| 요청 | 기대 |
|---|---|
| knowledge_search `Y` | HTTP 200, `status: "SUCCESS"`, `code: 200`, `data.request_id`와 `session_id`가 요청값 그대로, `resolved_intents: ["knowledge_search"]`, `retrieval_used: true`, `citations`가 비어 있지 않음(`DOC-…-C…`), `context_status: "not_available"`, `answer`가 비어 있지 않음, `latency_ms`가 curl `time_total`×1000과 비슷함(차이는 네트워크·직렬화 수준) |
| knowledge_search `N` | 위와 같되 `retrieval_used: false`, `citations: []` |
| trajectory_recomm / pose_recomm / error_analysis | HTTP 200, `SUCCESS`, `answer`는 "…기능은 준비 중입니다…" 고정 문구, `retrieval_used: false`, `citations: []`, 응답이 즉시 옴(생성 없음). 같은 시각 `docker compose logs mcp`에 요청 로그 없음 |
| `use_rag: true`(boolean) | HTTP 422, `status: "ERROR"`, `code: 422`, `data: null` |

MCP 장애 회귀(T7) 절차:

1. `docker compose stop mcp`
2. knowledge_search `Y`가 여전히 200이고, `/api/ready`도 200이며 `checks.mcp.ok`만 `false`인지 확인한다.
3. `docker compose start mcp`

## 12. web UI와 SSE 프록시

```bash
docker compose up -d web
# 직접 경로와 nginx 경로의 SSE 프레임 도착 시각 비교
for url in http://127.0.0.1:58000 http://127.0.0.1:8080; do
  echo "== $url"
  curl -N -sS "$url/api/chat" -H 'Content-Type: application/json' \
    -d '{"message":"무기질 아연 프라이머의 도장 조건은?","session_id":"web-check","mode":"live","level_id":"TR-L01"}' \
  | while IFS= read -r line; do [[ "$line" == event:* ]] && printf '%s %s\n' "$(date +%T.%3N)" "$line"; done
done
```

브라우저에서 `http://<spark>:8080`에 접속해 질문을 보낸다.

**합격 기준**

- nginx 경로(8080)의 `event:` 줄이 직접 경로와 같은 순서로 온다.
- 각 이벤트가 생성 완료 뒤 한꺼번에 몰리지 않고 직접 경로와 비슷한 간격(±1초)으로 도착한다. `proxy_buffering off`가 적용됐다는 뜻이다.
- UI에서 답변이 표시되고, 브라우저 네트워크 탭에서 `/api/chat`이 same-origin(8080) 요청이다.
- `/api/monitor/events` 연결이 유지된다.
- 참고: 지금 `run_streaming()`은 완성된 답변을 다시 청크로 나눈다(실제 토큰 스트리밍은 아직 없음, AGENTS.md). 체감 지연은 §13 수치로 판단한다.

2026-09-15 개발 서버에서 `nginx:latest`(1.29.4)로 `nginx -t` 문법 검사를 통과했다. 배포 이미지는 `nginx:1.29.4-alpine`이다.

## 13. 지연 측정

```bash
# 직접(agent-api 컨테이너 안에서 자기 자신) — --api-url을 반드시 명시한다(스크립트 기본값 8130은 개발용)
docker compose exec agent-api python3 trainer/scripts/bench_latency.py --api-url http://127.0.0.1:8120 \
  --profile rag --repeat 3 --output-root /output/eval_results/latency_bench/direct_mtp${ENABLE_MTP:-0}
docker compose exec agent-api python3 trainer/scripts/bench_latency.py --api-url http://127.0.0.1:8120 \
  --profile tools --repeat 1 --output-root /output/eval_results/latency_bench/direct_tools_mtp${ENABLE_MTP:-0}
# nginx 경유(compose 네트워크 안의 web:80)
docker compose exec agent-api python3 trainer/scripts/bench_latency.py --api-url http://web \
  --profile rag --repeat 3 --output-root /output/eval_results/latency_bench/web_mtp${ENABLE_MTP:-0}
# copilot/query 지연: 10분 짧은 소크(§16 스크립트)
DURATION_SEC=600 OUT_DIR="$RESULTS_DIR/soak_short_mtp${ENABLE_MTP:-0}" scripts/soak.sh
```

**합격 기준**

- 벤치 오류 0건. `bench_report` 요약(첫 이벤트와 완료 시간의 p50/p95)을 `RESULTS_DIR`에 **기록**한다.
- web 경유 p50과 직접 경로 p50의 차이가 0.5초 이내다.
- 목표 지연값은 KETI와 재합의 대상이다(계획 §4). 이 단계는 합격/불합격 대신 MTP off와 on의 수치 비교표를 남긴다. D1 추정치(FP8 단일 스트림 약 10 tok/s)는 Spark 실측 전 값이다.

## 14. 재기동

| 시험 | 명령 | 합격 기준 |
|---|---|---|
| agent-api 재시작 | `docker compose restart agent-api` | `/api/ready`가 다시 200(소요 시간 기록). `SimSessionStore` 상태는 초기화되는 것이 정상(인메모리) |
| vLLM 재시작 | `docker compose restart vllm` | 재시작 중 knowledge_search는 HTTP 503 `"Model server unavailable"`, `/api/ready`는 503. vLLM이 healthy가 되면 agent-api 재시작 없이 200으로 회복 |
| 전체 재생성 | `docker compose down && docker compose up -d` | 모든 서비스 healthy, §10·§11 기준 재통과. 콜드 스타트 전체 시간 기록 |
| 호스트 재부팅 | `sudo reboot` | docker 서비스 자동 시작, `restart: unless-stopped`로 네 서비스 복귀, §10 기준 재통과 |

## 15. MTP 수락률

```bash
sed -i 's/^ENABLE_MTP=.*/ENABLE_MTP=1/' .env && docker compose up -d --force-recreate vllm
curl -sS http://127.0.0.1:18000/metrics | grep -E '^vllm:spec_decode_num_(accepted|draft)_tokens_total|^vllm:generation_tokens_total' > /tmp/mtp_before.txt
# §13 rag 벤치 1회 실행
curl -sS http://127.0.0.1:18000/metrics | grep -E '^vllm:spec_decode_num_(accepted|draft)_tokens_total|^vllm:generation_tokens_total' > /tmp/mtp_after.txt
```

- 수락률은 Δaccepted_tokens / Δdraft_tokens로 계산한다.
- 메트릭 이름은 로컬 vLLM v0.22.1 소스(`v1/spec_decode/metrics.py`)에서 확인했다. v0.29.0에서는 **미확인**이며, 이름이 다르면 `grep spec_decode`로 찾는다.

**합격 기준**

- MTP를 켠 상태에서 기동하고 §7·§8·§9·§11을 재통과한다.
- 수락률과 디코딩 tok/s를 MTP off와 비교해 **기록**한다.
- MTP를 켜서 §13 p50이 개선되고 §16 소크와 §17 게이트를 MTP on으로 통과하면 `ENABLE_MTP=1`을 릴리스 값으로 확정한다. 그렇지 않으면 0으로 둔다.

## 16. 장시간 소크

```bash
DURATION_SEC=28800 OUT_DIR="$RESULTS_DIR/soak_8h_mtp${ENABLE_MTP:-0}" scripts/soak.sh; echo EXIT=$?
```

vLLM #37431 계열 크래시(SM121에서 장시간 운용 중 엔진이 종료되는 문제로 계획에 기록됨)를 확인하는 단계다. 이슈 본문은 이 문서를 쓸 때 확인하지 않았다(**미확인**). 스크립트는 로그에서 `CUDA error|illegal memory access|Xid|EngineDeadError|Traceback` 등을 찾는다.

**합격 기준**

- `EXIT=0`: 비-200과 `status != SUCCESS` 응답 0건, vllm·agent-api·mcp 재시작 0회, 크래시 패턴 0줄.
- `summary.json`의 `last_100_p50_s`가 `first_100_p50_s`의 1.2배 이하다. 이 기준은 제안값이다.
- `resources.tsv`의 사용 메모리가 단조 증가하지 않는다(누수 없음).
- 릴리스 MTP 설정으로 최소 8시간을 돌린다.

## 17. FP8 품질 게이트 (30문항, NF4 기준선 대비 Δ=0.2, 단계적 95% CI 규칙)

기준선은 `/output/eval_results/qwen36_27b_v6_1_nf4_baseline_965549e`다(A40, HF NF4 + v6.1, 스냅숏 965549e). seed 42–49 측정은 2026-09-15 11:45Z~18:05Z에 끝났고 total 6.704 ± 0.215(실행 평균들의 SD)다.

### 판정 규칙 (사전등록, 2026-09-15 사용자 결정 "8회로 먼저 보고 결정")

결과를 보기 전에 고정한 규칙이다. 결과를 본 뒤 Δ, 반복 수, 규칙을 바꾸지 않는다.

- 실행 점수: seed 하나의 30문항 `total` 평균(누락 응답 0점). 차이 d = 후보 평균 − 기준선 평균.
- Welch 표준오차 se = √(s_c²/n_c + s_b²/n_b), Welch–Satterthwaite 자유도, 양측 95% 신뢰구간 d ± t₀.₉₇₅(df)·se. t 분위수는 표준 라이브러리로 계산한다(불완전 베타 함수 + 이분법, 표값과 소수 3자리 일치를 테스트로 고정).
- **pass**: CI 하한 > −Δ → exit 0.
- **fail**: CI 상한 < −Δ → exit 1.
- **inconclusive**: 그 외 → exit 3. 보고서 `decision`에 확장 권고와 현재 SD로 계산한 필요 반복 수 n ≈ 2·(1.96+0.84)²·s_pooled²/Δ²(`required_runs_estimate`, 올림)가 나온다.
- **incomplete**: 입력 문제(각 측 실행 수 < `--min-runs`, 문항 수 ≠ 30, 문항 집합 불일치, seed 디렉터리의 CSV 개수 ≠ 1) → exit 2. 양측 실행 수는 달라도 된다.

단계:

1. **1단계**: 양측 각 8회(`--min-runs 8`). pass나 fail이면 그 결과로 확정한다.
2. **2단계**(1단계가 inconclusive일 때만): 양측을 각 18회로 늘린다. 1단계 실행을 포함하고 추가분만 돌린다. 같은 규칙을 `--min-runs 18`로 한 번 더 적용하고 그 결과로 확정한다. 보고서 `stage`는 양측 min(n) ≥ `--target-runs`(18)이면 2다.
3. 2단계에서도 inconclusive면 **판정 불가 — 사용자 결정**으로 기록한다(exit 3 그대로).

한계: 같은 규칙을 최대 두 번 확인하고 2단계 표본은 1단계 표본을 포함하므로 1종 오류가 명목 5%보다 조금 커진다. 확인이 두 번뿐이라 작다고 보고 **보정하지 않는다**.

### 1단계

```bash
# 1) 기준선 CSV를 Spark로 복사(seed_<n>/…/agentic_eval_results.csv 가 seed마다 하나)
rsync -a --include '*/' --include 'agentic_eval_results.csv' --include 'RUN_INFO.txt' --exclude '*' \
  a40:/data/ship-paint-ai/output/eval_results/qwen36_27b_v6_1_nf4_baseline_965549e/ "$RESULTS_DIR/baseline_nf4_965549e/"
# 2) 후보 생성·채점(트래픽 없는 시간에. agent-api 안에서 파이프라인을 하나 더 만든다). RUNS 기본 8 → seed_42..49
scripts/run_fp8_gate.sh
# 3) 판정
python3 scripts/quality_gate.py --baseline-root "$RESULTS_DIR/baseline_nf4_965549e" \
  --candidate-root "$RESULTS_DIR/qwen36_27b_v6_1_fp8_spark_<commit7>" --delta 0.2 --min-runs 8 --target-runs 18 \
  --output "$RESULTS_DIR/fp8_gate_report_stage1.json"; echo EXIT=$?
```

### 2단계 (1단계 EXIT=3일 때만)

**NF4 기준선 seed 50–59 추가(A40).** 라이브 API를 멈추는 창이 필요하다. 1단계 기준선 8 seed가 6시간 20분 걸렸으므로 10 seed는 약 8시간, 기동·확인을 더해 약 9시간을 잡는다. 1단계 기준선을 만든 `.claude/worktrees/nf4-baseline-965549e/run_nf4_baseline.sh`는 레포에 없는 로컬 파일이고, **seed 인자만 바꿔 다시 쓰면 안 된다**. 그 스크립트는 `RUN_INFO.txt`를 `>`로 덮어쓴다. `run_multiseed_eval.py`도 `multiseed_report.json`을 이번에 돌린 seed만으로 다시 쓴다(게이트는 CSV를 읽으므로 판정에는 영향이 없다). 컨테이너 안 git이 워크트리를 해석하지 못해 `commit:` 줄도 비어 있었다. 그래서 아래 명령을 직접 쓴다. 생성 부분(`PYTHONPATH`, 환경 변수, `run_multiseed_eval.py` 인자)은 1단계와 같고 seed만 다르다.

```bash
# A40 ship-paint-trainer 컨테이너 안. 스냅숏이 없으면 호스트 레포 루트에서:
#   git worktree add --detach .claude/worktrees/nf4-baseline-965549e 965549e
WT=/workspace/.claude/worktrees/nf4-baseline-965549e
OUT=/output/eval_results/qwen36_27b_v6_1_nf4_baseline_965549e   # 1단계와 같은 결과 루트
LOG=/logs/nf4_baseline_965549e
mkdir -p "$LOG"
pkill -f 'trainer.demo_api.app --host 0.0.0.0 --port 8120' || true   # 라이브 API 중단 창 시작
pkill -f 'uvicorn mcp_servers.main:app --host 0.0.0.0 --port 8100' || true
sleep 5
cd "$WT"
nohup uvicorn mcp_servers.main:app --host 0.0.0.0 --port 8100 > "$LOG/mcp_stage2.log" 2>&1 &
until curl -sf http://127.0.0.1:8100/health | grep -q '"servers":7'; do sleep 2; done
cp "$OUT/multiseed_report.json" "$OUT/multiseed_report_seed42-49.json"
{
  echo "--- stage 2: seeds 50-59"
  echo "commit: 965549eb9ebe299b49378bc4eec8d6f0be343d1b"
  echo "started_at: $(date -u +%FT%TZ)"
  echo "env: TOOL_ADAPTER_NATIVE_TOOLS=0 TOOL_ADAPTER_FORCE_SYNTHESIS=1"
  echo "pythonpath: $WT"
  sha256sum /data/eval/eval_set_agentic.jsonl "$WT/trainer/configs/rag_config_fewshot_27b.yaml" \
    /output/adapters/qwen36_27b_shipcoat_lora_v6_1/adapter_model.safetensors \
    /output/indexes/faiss.index /output/indexes/metadata.json
} >> "$OUT/RUN_INFO.txt"   # 덧붙인다. 해시가 1단계 블록과 같아야 한다
PYTHONPATH="$WT" TOOL_ADAPTER_NATIVE_TOOLS=0 TOOL_ADAPTER_FORCE_SYNTHESIS=1 \
python3 trainer/scripts/run_multiseed_eval.py \
  --eval-set /data/eval/eval_set_agentic.jsonl \
  --config "$WT/trainer/configs/rag_config_fewshot_27b.yaml" \
  --adapter /output/adapters/qwen36_27b_shipcoat_lora_v6_1 \
  --mcp-url http://127.0.0.1:8100 \
  --seeds 50 51 52 53 54 55 56 57 58 59 \
  --output-root "$OUT" \
  --tool-schemas-dir "$WT/trainer/configs/tool_schemas" > "$LOG/multiseed_stage2.log" 2>&1
echo "finished_at: $(date -u +%FT%TZ)" >> "$OUT/RUN_INFO.txt"
# 이후 스냅숏 MCP를 내리고 라이브 MCP·API를 평소 방식으로 다시 올린다.
```

**후보 seed 50–59 추가(Spark)와 재판정.**

```bash
rsync -a --include '*/' --include 'agentic_eval_results.csv' --include 'RUN_INFO.txt' --exclude '*' \
  a40:/data/ship-paint-ai/output/eval_results/qwen36_27b_v6_1_nf4_baseline_965549e/ "$RESULTS_DIR/baseline_nf4_965549e/"
# 채점 CSV가 있는 seed_42..49는 건너뛰고 seed_50..59만 돈다. 커밋이 1단계와 다르면 RUN_NAME을 1단계 이름으로 준다.
RUNS=18 RUN_NAME=qwen36_27b_v6_1_fp8_spark_<commit7> scripts/run_fp8_gate.sh
python3 scripts/quality_gate.py --baseline-root "$RESULTS_DIR/baseline_nf4_965549e" \
  --candidate-root "$RESULTS_DIR/qwen36_27b_v6_1_fp8_spark_<commit7>" --delta 0.2 --min-runs 18 --target-runs 18 \
  --output "$RESULTS_DIR/fp8_gate_report_stage2.json"; echo EXIT=$?
```

평가 하네스를 openai 백엔드 설정으로 돌리는 방식:

- `generate_agentic_responses.py --config /workspace/trainer/configs/rag_config_spark_vllm.yaml --mcp-url http://mcp:8100`
- **`--no-adapter`는 필요 없다.** openai 백엔드는 `adapter_path`를 무시한다(`agent_pipeline._load_openai_backend`). 러너는 `--qlora-config`에 없는 경로를 줘서 어댑터 경로를 찾지도 않게 한다.
- `TOOL_ADAPTER_NATIVE_TOOLS=0`과 `TOOL_ADAPTER_FORCE_SYNTHESIS=1`은 compose 환경에서 들어온다. 기준선 RUN_INFO와 같은 값이다.

**명시적 한계**

1. `--seed`는 torch 시드만 고정한다. openai 백엔드는 vLLM 요청에 `seed`를 보내지 않는다. 그래서 후보의 `seed_<n>`은 재현 가능한 시드가 아니라 **seed가 요청에 전달되지 않는 독립 반복** 번호다. 게이트는 실행 평균들을 독립 표본으로 비교(Welch)하므로 판정은 가능하지만, 문항별로 seed끼리 짝지어 비교하는 것은 의미가 없다.
2. 기준선은 HF NF4(in-process), 후보는 vLLM FP8(HTTP)이다. 게이트는 양자화만의 차이가 아니라 서빙 스택 전체의 차이를 잰다.
3. 문항별 thinking 사용 여부(`_should_enable_thinking`)와 MCP pose 기준 데이터 부재는 양쪽이 같다.
4. 단계 규칙의 반복 확인(최대 2회)에 따른 1종 오류 증가는 보정하지 않았다(위 판정 규칙).

**합격 기준**

- 확정 단계의 `quality_gate.py` exit 0: `verdict: "pass"`, CI 하한(`ci_low`) > −0.2.
- exit 1(fail)은 불합격 확정이다. 1단계 exit 3이면 2단계로 간다. 2단계 exit 3은 판정 불가로 기록하고 사용자가 결정한다.
- exit 2는 입력 문제다. 각 측 `--min-runs` 이상, 각 30문항, 문항 집합 동일, seed 디렉터리마다 CSV 1개여야 한다. 고친 뒤 같은 단계를 다시 판정한다.
- 보고서 JSON의 `stage`, `n_baseline`, `n_candidate`, `difference`, `se`, `df`, `t_crit`, `ci_low`, `ci_high`, `delta`, `verdict`, `decision`, `required_runs_estimate`를 기록하고, `per_question` 하위 항목(가장 크게 떨어진 문항)을 검토해 기록한다.

## 18. 롤백 (BF16)

1. 번들에 BF16을 넣는다. §1에서 `--with-bf16`으로 만든다(`prepare_bundle.py`는 비어 있지 않은 디렉터리를 거부하므로 처음부터 포함한다). 또는 `--skip-fp8 --with-bf16`으로 별도 번들을 만든다.
2. 번들을 검증한다: `bundle_manifest.py verify --model bf16`. 합격 기준은 §3과 같다.
3. 롤백 설정을 만든다. served name과 tokenizer_path만 바꾼다.
   ```bash
   sudo chown "$APP_UID:$APP_GID" "$BUNDLE_DIR/config"
   docker compose run --rm --no-deps -v "$BUNDLE_DIR/config:/out" agent-api \
     python3 deploy/spark/scripts/render_rollback_config.py --output /out/rag_config_spark_vllm_bf16.yaml
   ```
4. `.env`를 바꾼다.
   ```
   MODEL_SUBDIR=qwen36_27b_shipcoat_v6_1_bf16
   SERVED_MODEL_NAME=qwen36-27b-shipcoat-v6-1-bf16
   AGENT_CONFIG_PATH=/bundle-config/rag_config_spark_vllm_bf16.yaml
   ENABLE_MTP=0
   ```
   BF16 가중치는 약 52GB다. 로그의 KV 캐시 크기를 보고 `GPU_MEMORY_UTILIZATION`을 조정한다(**미확인**).
5. 재생성: `docker compose up -d --force-recreate vllm agent-api`
6. §6, §7, §9, §10, §11을 반복한다. 합격 기준은 각 절과 같고, served name만 `-bf16`이다.

FP8로 돌아갈 때는 `.env`를 되돌리고 5를 반복한다. 이미지 롤백은 이전 digest를 `.env`에 넣거나 보관한 `docker save` 묶음을 `docker load`한다.

## 19. 미확인 항목

1. vLLM v0.29.0과 NGC 26.08의 GB10(SM121)에서 Qwen3.6(`Qwen3_5ForConditionalGeneration`) FP8 block-128 로드와 커널 동작(§6).
2. `--language-model-only`와 MTP의 동시 사용(§6). 플래그 존재는 v0.22.1 소스에서만 확인했다.
3. `--generation-config vllm`과 `--override-generation-config` 조합에서 override가 적용되는지(§6 로그).
4. vLLM v0.29.0의 MTP 메트릭 이름(§15), `/tokenize` chat 요청 스키마(§9), `VLLM_API_KEY` 환경 변수 인증.
5. torch 2.9.1+cu130 aarch64 휠의 GB10 커널 지원(§5). faiss-cpu와 sentence-transformers의 aarch64 실행(휠 존재만 확인).
6. 이미지 빌드 자체(§2): x86 서버 빌드 금지로 실행하지 않았다. lock의 `pip check`, import 스모크, node 이미지에서의 `npm ci`, Dockerfile별 `.dockerignore` 적용이 미확인이다. lock은 개발 컨테이너 설치본에서 의존 폐포를 계산해 만들었다.
7. 통합 메모리 예산: `GPU_MEMORY_UTILIZATION=0.72`와 agent-api의 bge-m3·reranker(cuda), OS 사이의 여유. 헬스체크 `start_period`(vllm 1800s, agent-api 900s)가 충분한지.
8. 모든 지연·처리량 수치(§7, §13, §15). 계획 D1의 값은 추정치다.
9. vLLM #37431 이슈 본문(§16).
10. MCP pose 기준 데이터 `expert_pose_reference.json`: A40 기본 경로에도 없다. 필요한 경우 생성 주체와 경로를 따로 정해야 한다.
11. arm64 vLLM 이미지 크기와 `docker save` 반입 시간.
12. 배포 nginx 이미지(`1.29.4-alpine`)에서의 `nginx -t`. 같은 버전의 debian 이미지로만 검사했다.
