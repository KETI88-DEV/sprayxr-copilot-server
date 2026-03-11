# Task ID: 13

**Title:** 재현 가능성 문서화 및 실행 스크립트 정리

**Status:** pending

**Dependencies:** 9 ✓, 12, 19 ⧖

**Priority:** medium

**Description:** /workspace/README.md(설치/실행 명령어), 설정 파일(/workspace/trainer/configs/), 주요 스크립트(/workspace/trainer/scripts/) 완성, 폴더 구조 및 버전 명시

**Details:**

1. README.md 구성:
   - 프로젝트 개요
   - 필수 조건(Docker, NVIDIA A40, 필수 라이브러리)
   - Docker 실행 단계(docker-compose up)
   - 모델 다운로드(/hf-cache/ 사전 다운로드)
   - 주요 스크립트 사용법
   - 문제 해결(OOM, import 오류)
   - 기본 명령어 모음(부록 A 참고)
2. 스크립트 체크리스트:
   - /workspace/trainer/scripts/ingest_docs.py: raw → parsed
   - /workspace/trainer/scripts/clean_docs.py: 정규화, dedup
   - /workspace/trainer/scripts/chunk_docs.py: chunk 생성
   - /workspace/trainer/scripts/build_index.py: FAISS index 생성
   - /workspace/trainer/scripts/run_rag_eval.py: 50문항 평가
   - /workspace/trainer/scripts/generate_sft.py: SFT 데이터 생성
   - /workspace/trainer/scripts/train_qlora.py: QLoRA 학습
   - /workspace/trainer/scripts/serve_demo_api.py: FastAPI/vLLM 서빙
3. 설정 파일:
   - /workspace/trainer/configs/rag_config.yaml: top_k, rerank on/off, temperature, max_tokens
   - /workspace/trainer/configs/qlora_config.yaml: batch, lr, epochs, seq_len, rank, alpha
   - /workspace/trainer/configs/env.example: HF_HOME, CUDA 설정
4. 폴더 구조 다이어그램(Docker 볼륨 기반)
5. 버전 고정: requirements.txt 또는 uv.lock
6. 실행 체크리스트: smoke test 확인 순서
7. 예상 소요 시간: 각 단계별
8. 지원 연락처 및 이슈 추적 방식

**Test Strategy:**

README 지시대로 새로운 Docker 환경에서 설치 가능 여부 확인(dry run), 모든 스크립트 entry point 확인(python -m <script> --help), 설정 파일 YAML/JSON 유효성 검증, requirements.txt 모든 패키지 설치 가능 확인

## Subtasks

### 13.1. README.md 기본 구조 및 필수 섹션 작성

**Status:** pending  
**Dependencies:** None  

프로젝트 개요, 필수 조건(Docker, NVIDIA A40, 필수 라이브러리), Docker 실행 단계, 모델 다운로드 절차를 포함한 README.md의 기본 골격 완성

**Details:**

README.md 파일 생성/수정하여 다음 섹션 포함: 1) 프로젝트 개요(선박 도장 RAG 시스템), 2) Prerequisites 섹션(Docker, nvidia-docker, NVIDIA A40 48GB 요구사항), 3) Installation 섹션(docker-compose 설치 및 실행 단계), 4) Model Download 섹션(/hf-cache/ 사전 다운로드 지침, HuggingFace 토큰 설정), 5) Quick Start 섹션. 마크다운 형식 준수, 코드 블록 포함.

### 13.2. 주요 스크립트 사용법 및 문제 해결 가이드 작성

**Status:** pending  
**Dependencies:** 13.1  

8개 주요 스크립트(ingest, clean, chunk, build_index, run_rag_eval, generate_sft, train_qlora, serve_demo_api)의 사용법, 파라미터, 예제 및 일반적인 문제 해결(OOM, import 오류) 섹션 작성

**Details:**

README.md에 'Scripts' 섹션 추가: 1) 각 스크립트별 설명(ingest_docs.py: raw→parsed, clean_docs.py: 정규화/dedup 등), 2) 실행 명령어(python -m trainer.scripts.<script> --help 형식), 3) 주요 파라미터 설명, 4) 예상 입출력, 5) Troubleshooting 섹션(OOM 메모리 부족 해결, import 오류, CUDA 관련 오류), 6) 부록 A에 기본 명령어 모음(copy-paste 가능한 형식). 각 스크립트는 /workspace/trainer/scripts/ 경로 참조.

### 13.3. 설정 파일 완성 및 env.example 작성

**Status:** pending  
**Dependencies:** 13.1  

/workspace/trainer/configs/ 디렉터리의 rag_config.yaml, qlora_config.yaml 완성, env.example 파일 작성으로 환경 변수 설정 가이드 제공

**Details:**

1) /workspace/trainer/configs/rag_config.yaml 확인/완성: top_k, rerank on/off, temperature, max_tokens 파라미터, 각 파라미터 설명(주석) 포함. 2) /workspace/trainer/configs/qlora_config.yaml 확인/완성: batch_size, learning_rate, num_epochs, max_seq_length, lora_rank, lora_alpha 파라미터, 주석 추가. 3) /workspace/trainer/configs/env.example 신규 생성: HF_HOME, HF_TOKEN, CUDA_VISIBLE_DEVICES, WANDB_PROJECT, LOG_DIR 등 필수 환경 변수 나열 및 설명. YAML/JSON 파일 형식 유효성 검증.

### 13.4. 프로젝트 폴더 구조 다이어그램 및 버전 관리 문서화

**Status:** pending  
**Dependencies:** 13.1, 13.3  

Docker 볼륨 기반 폴더 구조 다이어그램 작성, requirements.txt 또는 uv.lock으로 버전 고정, 폴더 구조 설명 추가

**Details:**

1) README.md 또는 별도 ARCHITECTURE.md에 폴더 구조 다이어그램 추가(tree 명령 또는 마크다운 테이블 형식): /workspace(code), /data(raw/chunks/eval/manifest.csv), /output(indexes/reports/eval_results), /models, /hf-cache, /logs 등. 각 디렉터리 용도 설명. 2) Docker 볼륨 마운트 정보 표시(볼륨명→컨테이너 경로). 3) /workspace/requirements.txt 확인/생성: 모든 의존 라이브러리 버전 고정(예: torch==2.4.0, transformers==5.1.0), pip freeze 결과 활용 또는 uv.lock 파일 생성. 4) 버전 호환성 노트 추가(Python 3.9+, CUDA 12.4, PyTorch 2.4.0 등).

### 13.5. 실행 체크리스트 및 smoke test 문서화

**Status:** pending  
**Dependencies:** 13.1, 13.2, 13.3, 13.4  

새로운 Docker 환경에서 순차적으로 설치/실행 가능 여부 검증하는 smoke test 체크리스트 작성, 예상 소요 시간 및 지원 연락처 문서화

**Details:**

1) README.md에 'Installation Checklist' 섹션 추가: 단계별 실행 순서(docker-compose up → 모델 다운로드 → ingest_docs → clean_docs → chunk_docs → build_index → run_rag_eval), 각 단계 완료 기준(예: 'FAISS index 생성 완료, /output/indexes/faiss.index >100MB'). 2) Smoke Test 섹션: 5개 샘플 쿼리로 retrieval 동작 확인, 모든 스크립트 --help 실행(약 2분), 설정 파일 YAML 파싱(약 1분), requirements.txt 설치(약 10분). 3) Expected Timeline: 각 단계별 소요 시간(초 단위 또는 분 단위) 명시. 4) Support 섹션: 이슈 추적 방식(GitHub Issues 링크), 연락처, FAQ 추가.
