# SprayXR AI Co-Pilot 서버 배포본

API 서버(agent-api, MCP)를 띄우는 데 필요한 코드와 배포 설정만 들어 있습니다. 모델과 데이터는 git에 없으므로 아래 표대로 받아 둡니다.

## 1. 코드 받기

```bash
git clone --single-branch --branch release <저장소 주소> sprayxr-copilot-server
cd sprayxr-copilot-server
git pull --ff-only      # 새 버전을 받을 때
```

## 2. 모델·데이터 받기

넣을 위치는 `deploy/spark/.env`의 경로 변수입니다(`cp deploy/spark/.env.example deploy/spark/.env` 후 채움).

| 무엇 | 받는 곳 | 넣을 위치 |
|---|---|---|
| 27B 상세분석 모델(약 52GB) | 공개 배포 없음. 개발 서버의 병합본을 폴더째 복사 | `${BUNDLE_DIR}/output/merged/qwen36_27b_shipcoat_v6_1_bf16/` |
| RAG 색인 | 개발 서버에서 복사 | `${BUNDLE_DIR}/output/indexes/kure3-20261113/` |
| 자세 기준(JSON 2개) | 개발 서버에서 복사 | `${BUNDLE_DIR}/output/expert_reference/` |
| 서빙 설정 | 개발 서버에서 복사 | `${BUNDLE_DIR}/config/` |
| 임베딩 모델 | Hugging Face `nlpai-lab/KURE-v3` | `${BUNDLE_DIR}/models/KURE-v3/` |
| 재정렬 모델 | Hugging Face `nlpai-lab/KURE-Reranker-nano` | `${BUNDLE_DIR}/models/KURE-Reranker-nano/` |
| 빠른 안내 8B 모델 | Hugging Face `Qwen/Qwen3-8B-AWQ` | `${QUICK_MODEL_DIR}` |
| 음성 인식 모델 | Hugging Face `openai/whisper-large-v3-turbo` | `${STT_MODEL_DIR}` |
| 전문가·작업자 기록, 학습모드 FBX | 개발 서버에서 복사 | `${EXPERT_REFERENCE_DIR}`, `${POSE_MODELINGS_DIR}` |
| vLLM 이미지 | Docker Hub `vllm/vllm-openai:v0.29.0` | `.env`의 `VLLM_IMAGE` |
| 운영 DB | 받지 않음(서버가 만듦) | `${RESULTS_DIR}`(빈 폴더, 쓰기 가능) |

- **가장 쉬운 방법:** 위의 `${BUNDLE_DIR}` 항목은 검증된 번들 폴더(`bundle/spark-v6.1-bf16`)를 통째로 복사하면 한 번에 갖춰집니다.
- **Hugging Face 모델**은 운영과 같은 리비전으로 받습니다.

```bash
hf download nlpai-lab/KURE-v3 --revision 73d5e710bc0b1f324a60746922cbad6d38f480d4 --local-dir "$BUNDLE_DIR/models/KURE-v3"
hf download nlpai-lab/KURE-Reranker-nano --revision 8514b9e113f3b27c441ed9c2b50f81d80388b2fd --local-dir "$BUNDLE_DIR/models/KURE-Reranker-nano"
hf download Qwen/Qwen3-8B-AWQ --revision 4da05a8edb55c6046cce958586c33b61da07bb79 --local-dir "$QUICK_MODEL_DIR"
hf download openai/whisper-large-v3-turbo --revision 41f01f3fe87f28c78e2fbf8b568835947dd65ed9 --local-dir "$STT_MODEL_DIR"
```

- **받은 뒤 확인:** `python3 deploy/spark/scripts/bundle_manifest.py verify --bundle-dir "$BUNDLE_DIR" --model bf16`
- **`.env`에서 모델을 맞춥니다:** `MODEL_SUBDIR=qwen36_27b_shipcoat_v6_1_bf16`, `SERVED_MODEL_NAME=qwen36-27b-shipcoat-v6-1-bf16`, `AGENT_CONFIG_PATH=/bundle-config/<서빙 설정 파일>`. 예시 파일의 기본값은 FP8 모델입니다.
- 학습 체크포인트(LoRA 어댑터, 학습 중간 저장)는 필요 없습니다. 배포에는 병합본만 씁니다.

## 3. 실행

```bash
cd deploy/spark
CODE_COMMIT=$(git rev-parse HEAD) docker buildx bake -f compose.yaml --env-file .env mcp agent-api --load
docker compose -f compose.yaml -f compose.quick.yaml --env-file .env up -d vllm quick-vllm mcp agent-api
curl -s http://127.0.0.1:58000/api/ready      # "ready": true 이면 준비 완료
```

- 27B 모델을 처음 올리는 데 몇 분 걸립니다.
- 모션뷰어(`/motion-viewer`), DB뷰어(`/admin`), API 문서(`/docs`)는 agent-api가 직접 냅니다. 웹 채팅 화면(demo-ui)은 이 배포본에 없습니다.
- `${RESULTS_DIR}`에는 훈련 기록과 관리자 계정 파일이 쌓이므로 외부에 공유하지 않습니다.
