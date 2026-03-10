# Task ID: 1

**Title:** 개발 환경 구축 및 GPU 검증

**Status:** pending

**Dependencies:** None

**Priority:** high

**Description:** Docker 컨테이너(pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime) 환경에서 NVIDIA A40 GPU 검증, /workspace 프로젝트 루트 구조 확인, 필수 라이브러리 설치 확인 및 smoke test 수행

**Details:**

1. GPU 확인: nvidia-smi, A40 48GB GDDR6 ECC 검증
2. Docker 볼륨 마운트 확인: /workspace(코드), /data(데이터), /output(산출물), /models(모델), /hf-cache(HF 캐시), /logs(로그)
3. 설치 완료 라이브러리 확인: torch cu124, transformers 5.1.0, accelerate 1.13.0, bitsandbytes 0.49.1, peft, trl, unsloth 2026.2.1, llama-index-core, sentence-transformers 5.2.3, FlagEmbedding 1.3.5, faiss-cpu, vllm 0.16.0, fastapi, uvicorn, pymupdf, beautifulsoup4
4. 각 라이브러리 import 및 CUDA 가용성 확인(torch.cuda.is_available())
5. 프로젝트 디렉터리 구조 확인:
   - /workspace/trainer/scripts/ (Python 스크립트)
   - /workspace/trainer/configs/ (설정 파일)
   - /workspace/trainer/notebooks/ (Jupyter 노트북)
   - /workspace/rag_api/ (RAG API 모듈)
   - /workspace/deploy/ (Docker 배포)
   - /data/{raw,parsed,cleaned,chunked,deduplicated,sft,eval,exports}
   - /output/{indexes,adapters,checkpoints,eval_results,embeddings}
   - /qdrant_storage/ (향후 Qdrant 벡터 DB 확장용)
   주의: 아직 존재하지 않는 디렉터리(parsed, deduplicated, exports, indexes, embeddings)는 Task 1에서 생성 필요
6. 오프라인 모델 사전 다운로드 확인: Qwen2.5-7B-Instruct, BAAI/bge-m3, BAAI/bge-reranker-v2-m3
7. HF_HOME=/hf-cache, HF_HUB_CACHE=/hf-cache/hub 환경변수 설정 확인

**Test Strategy:**

torch.cuda.is_available() 반환값 확인, python -c 'import unsloth; import peft; import vllm' 성공, 각 라이브러리 버전 정보 출력(pip show), /data, /output, /models, /hf-cache 디렉터리 존재 및 쓰기 권한 확인, 모델 다운로드 완료 여부 확인(/hf-cache/hub 크기 >20GB)

## Subtasks

### 1.1. NVIDIA A40 GPU 및 Docker 환경 검증

**Status:** pending  
**Dependencies:** None  

nvidia-smi 명령어로 A40 48GB GDDR6 ECC GPU 확인, Docker 컨테이너의 CUDA 12.4 및 cuDNN9 설치 상태 검증

**Details:**

Docker 컨테이너(pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime) 내에서 nvidia-smi 실행하여 A40 GPU 48GB 메모리 확인, CUDA_VISIBLE_DEVICES 설정 확인, GPU 드라이버 버전 12.4 이상 검증, 초기 메모리 할당 상태 기록

### 1.2. Docker 볼륨 마운트 지점 및 디렉터리 구조 검증

**Status:** pending  
**Dependencies:** None  

/workspace, /data, /output, /models, /hf-cache, /logs 마운트 지점 존재 확인 및 쓰기 권한 검증, 프로젝트 루트 디렉터리 구조 확인

**Details:**

ls -la /workspace, /data, /output, /models, /hf-cache, /logs 명령어로 마운트 확인, 각 디렉터리 쓰기 권한 테스트(touch 파일 생성), /workspace 하위 trainer/scripts/, trainer/configs/, trainer/notebooks/, rag_api/, deploy/ 디렉터리 존재 확인, /data 하위 raw/, parsed/, cleaned/, chunked/, deduplicated/, sft/, eval/, exports/ 디렉터리 확인(존재하지 않는 디렉터리는 mkdir -p로 생성: parsed, deduplicated, exports). /output 하위 indexes/, adapters/, checkpoints/, eval_results/, embeddings/ 디렉터리 확인(존재하지 않는 디렉터리는 mkdir -p로 생성: indexes, embeddings)

### 1.3. PyTorch 및 CUDA 라이브러리 버전 검증

**Status:** pending  
**Dependencies:** 1.1  

torch, transformers, accelerate 등 핵심 머신러닝 라이브러리 설치 확인 및 CUDA 가용성 검증

**Details:**

python -c 'import torch; print(torch.__version__)' 실행하여 torch cu124 버전 확인, torch.cuda.is_available() 반환값 확인(True), pip show torch | grep Version, pip show transformers | grep Version(5.1.0), pip show accelerate | grep Version(1.13.0) 실행, torch.cuda.get_device_name(0) 확인

### 1.4. 양자화 및 최적화 라이브러리(bitsandbytes, peft, unsloth) 검증

**Status:** pending  
**Dependencies:** 1.3  

bitsandbytes 0.49.1, peft, unsloth 2026.2.1, trl 라이브러리 import 및 CUDA 호환성 확인

**Details:**

python -c 'import bitsandbytes as bnb; print(bnb.__version__)' 실행(0.49.1 확인), python -c 'import peft; print(peft.__version__)', python -c 'from unsloth import FastLanguageModel; print("unsloth imported")' 실행, python -c 'import trl; print(trl.__version__)', bnb CUDA 초기화 확인(bitsandbytes가 CUDA 감지)

### 1.5. RAG 및 API 관련 라이브러리 검증 및 HF_HOME 환경변수 설정 확인

**Status:** pending  
**Dependencies:** 1.3  

transformers 5.1.0, sentence-transformers 5.2.3, FlagEmbedding 1.3.5, faiss-cpu, vllm 0.16.0, fastapi, uvicorn 라이브러리 검증 및 HF_HOME, HF_HUB_CACHE 환경변수 설정 확인

**Details:**

python -c 'import sentence_transformers; print(sentence_transformers.__version__)' 실행(5.2.3 확인), python -c 'from FlagEmbedding import FlagModel' 실행(1.3.5 확인), python -c 'import faiss; print(faiss.__version__)', python -c 'import vllm; print(vllm.__version__)' 실행(0.16.0 확인), python -c 'import fastapi, uvicorn', env | grep HF_HOME 및 env | grep HF_HUB_CACHE 확인(/hf-cache 포함), python -c 'import os; print(os.environ.get("HF_HOME"))'
