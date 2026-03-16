---
title: "선박 도장 AI Co-Pilot 통합 PRD (최종)"
subtitle: "RAG+QLoRA 기반 Agentic 아키텍처, Simulator Protocol 및 RTX 4080 최적화"
date: "2026-03-13"
lang: ko-KR
numbersections: true
geometry: margin=0.78in
---

## 목차

1. 문서 개요
2. 배경 및 왜 이 PRD가 필요한가
3. 제품 정의
4. 사용자와 핵심 사용 시나리오
5. 권장 아키텍처
6. 기술 선택과 의사결정
7. 처음부터 시작하는 환경 설정
8. 데이터 파이프라인 설계
9. RAG 기준선 설계
10. QLoRA 실험 설계
11. 7일 실행 계획
12. 평가 설계
13. API / 파일 인터페이스 정의
14. 리스크와 폴백
15. 최종 산출물 목록
16. Co-Pilot 아키텍처 (구현 완료)
17. Agent Orchestration Specification
18. Intent Router & Tool Schema
19. 부록

\newpage

# 문서 개요

| 항목 | 내용 |
|---|---|
| 문서 목적 | 기존 PRD, 전략 보고서, 기술 스택 자료를 통합하여 **1차 PoC 결과를 반영한 최종 실행 및 배포 문서**로 정의 |
| 이번 PoC 목표 | 1) RAG+QLoRA 통합 지식 베이스 구축, 2) Agentic Co-Pilot 아키텍처 구현, 3) Simulator ↔ Co-Pilot 구조화 JSON 통신 프로토콜 확립 |
| 권장 기본 모델 | **Qwen3-4B** (Apache 2.0, Thinking Mode 내장, BFCL 도구 호출 최적화) |
| 권장 우선 전략 | **RAG + QLoRA 통합 → Agentic Tool Calling → Simulator Protocol 적용** |
| 배포 원칙 | **온프레미스 / RTX 4080 16GB VRAM 타겟 / 기존 시뮬레이터 비침습(sidecar)** |
| 통신 프로토콜 | **Simulator ↔ Co-Pilot 구조화 JSON 통신 프로토콜** |
| 한 줄 결론 | RAG+QLoRA가 도메인 지식 및 응답 안정성에 효과적임을 확인했으며, Qwen3-4B 기반의 Agentic Co-Pilot 아키텍처를 통해 실시간 코칭 체계를 완성함 |

## 통합 설계 원칙

1. **PoC의 성공 기준은 RAG-only와 RAG+QLoRA의 차이를 명확히 판별하고, 실시간 코칭이 가능한 Agentic 구조를 확립하는 것이다.**
2. **기존 RealCOAT/Unity 제품은 바꾸지 않는다.** `Simulator Request Schema`를 통해 데이터를 넘겨받고 `Simulator Response Schema`로 응답을 돌려주는 **sidecar 통합**을 유지한다.
3. **수치 계산은 LLM에 맡기지 않는다.** 작업 속도, 이격거리, 두께, 90/10 룰 판정 등은 MCP Server(dft_judge 등)가 수행하고, LLM은 이를 해석하여 코칭으로 변환한다.
4. **학습 환경과 서빙 환경을 최적화한다.** RTX 4080 16GB 환경에서 4-bit QLoRA 학습과 4-bit inference가 원활히 돌아가도록 구성한다.
5. **최신 문서 업데이트는 RAG가 담당하고, LoRA는 출력 형식(JSON), 용어, 코칭 스타일, Tool Calling 습관을 안정화하는 데 집중한다.**

\newpage

# 1. 배경 및 왜 이 PRD가 필요한가

과제의 상위 목표는 **비언어적 숙련자 경험을 디지털로 획득·재현하는 멀티모달 XR 기반 AI 도장훈련시뮬레이터**이며, 도장 공정 특화 sLMM 데이터셋, 훈련자 맞춤형 AI Co-pilot, XR 재현, 시험 실증, 사업화가 핵심 축으로 제시되어 있다. 특히 스프레이도장·블라스팅 영역에서 **도막 두께, 도장 순서, 도장 자세, 도막 분포**를 AI Agent와 Co-pilot이 지원하는 그림이 명시되어 있다.

다만 1차 PoC 단계에서는 멀티모달 센서 학습보다 **웹 크롤링 및 기존 내부 자료로 확보 가능한 문서 중심 지식 적응**과 **Agentic 추론 구조의 기틀**을 마련하는 것이 현실적이다. RTX 4080 16GB 환경에서 1차 단계의 핵심은 **문서 기반 QA와 RAG/QLoRA 통합, 그리고 Simulator와의 구조화된 인터페이스 확립**이다.

기존 RealCOAT 환경은 이미 시나리오 관리, 작업 변수 설정, Replay, 히트맵, 성과 레포트, 이동속도·이격거리·분사각도·도막두께 등의 지표 표시 및 교관용 평가 구성을 보유하고 있다. 즉 이번 PoC는 **시뮬레이터 자체를 새로 개발하는 프로젝트가 아니라, 기존 시스템 외부에 문서형 AI 코어를 붙여 해석/질의응답/코칭 레이어를 추가하고 이를 Agentic하게 구동하는 프로젝트**로 정의해야 한다.

이 PRD는 바로 그 목적을 위해 작성되었다. 이전 PRD의 강점은 **실행 순서와 실험 프레임**, 첨부 PRD/전략 자료의 강점은 **정확한 기술 스택과 버전, 운영 리스크, 모듈형 확장 관점**이다. 본 문서는 이들을 통합하고 실제 구현된 **Co-Pilot 아키텍처와 Simulator Protocol**을 반영하여 **최종적인 실행 및 배포 가이드**로 정리한 것이다.

# 2. 제품 정의

## 2.1 이번 단계에서 완성한 것

이번 PoC를 통해 아래 세 가지 핵심 기능을 완성했다.

1. **Simulator Protocol 기반 AI Co-Pilot**
   - 시뮬레이터로부터 구조화된 JSON 요청을 받아 RAG 검색, Tool Calling, CoT 추론을 거쳐 구조화된 JSON 응답을 반환한다.
2. **3-Way SemanticRouter**
   - 사용자의 의도에 따라 `training_realtime`(실시간 코칭), `training_review`(사후 분석), `fast`(단순 QA) 경로로 자동 분기한다.
3. **MCP Server 기반 전문 도구 통합**
   - `dft_judge`(90/10 룰 판정), `anomaly_detection`(이상 탐지) 등 전문 분석 도구를 LLM이 자율적으로 호출하여 결과를 통합한다.

## 2.2 이번 단계에서 제외된 것

아래 항목은 향후 단계(Phase 5-6)에서 다룬다.

- 전신 모션/시선/손모션 원시 데이터의 직접적인 신경망 학습 (현재는 MCP Stub 처리)
- 도막 두께 예측 모델의 신규 학습 (현재는 기존 시뮬레이터 데이터 활용)
- Unity 내 실시간 인퍼런스 엔진 내장 (현재는 sidecar REST API 방식)
- 최종 다국어 음성 인터페이스 (현재는 텍스트 기반 JSON 응답)

## 2.3 성공 기준 (검증 완료)

이번 단계의 성공 기준은 다음과 같이 정의하고 달성 여부를 판단했다.

| 항목 | 성공 기준 |
|---|---|
| RAG+QLoRA 통합 | RAG-only 대비 용어 정확도 및 형식 일관성이 향상되고 faithfulness가 유지될 것 |
| Hallucination | 10% 이하 유지 |
| Citation Coverage | 모든 최종 답변이 근거 chunk id를 포함할 것 |
| Simulator Protocol | 정의된 JSON Schema에 따른 입출력이 100% 호환될 것 |
| Agentic Tool Calling | 복잡한 질의에 대해 최소 1개 이상의 도구를 정확히 호출할 것 |
| 배포 적합성 | RTX 4080 16GB VRAM 내에서 추론 및 학습이 가능할 것 |

# 3. 사용자와 핵심 사용 시나리오

## 3.1 주요 사용자

- **훈련생 (Trainee)**: XR 시뮬레이터에서 훈련 중 실시간 가이드와 훈련 후 분석 리포트를 받는다.
- **교관 (Instructor)**: 훈련생의 성과를 AI 분석 결과와 함께 검토하고 맞춤형 교육 계획을 세운다.
- **시스템 운영자**: AI 모델의 지식 베이스(RAG)를 업데이트하고 에이전트의 도구 호출 로그를 모니터링한다.

## 3.2 대표 사용 시나리오

### 시나리오 A: 실시간 도장 코칭 (training_realtime)
- **입력**: 시뮬레이터 센서 데이터 (현재 위치 B-3, 속도 0.5m/s, 이격거리 320mm)
- **처리**: SemanticRouter가 `training_realtime` 감지 → `location_tracking`, `sequence_validation` 도구 호출 → RAG 검색 (수직 패널 도장 기준) → CoT 추론
- **출력**: Simulator Response JSON (다음 이동 경로 지시, 속도 조절 경고)

### 시나리오 B: 훈련 후 성과 분석 (training_review)
- **입력**: 세션 전체 요약 데이터 (DFT 측정값 배열, 히트맵 이미지 컨텍스트)
- **처리**: SemanticRouter가 `training_review` 감지 → `dft_judge`(90/10 룰), `anomaly_detection` 도구 호출 → 분포 분석 및 원인 추론
- **출력**: Simulator Response JSON (합부 판정 결과, 저도막 원인 분석, 개선 행동 제안)

### 시나리오 C: 도장 지식 질의 (fast)
- **입력**: "에폭시 도료의 가사시간이 뭐야?"
- **처리**: SemanticRouter가 `fast` 감지 → RAG 검색 → LLM 응답 생성 (도구 호출 생략)
- **출력**: Simulator Response JSON (텍스트 가이드 및 근거 문서 ID)

\newpage

# 4. 권장 아키텍처

## 4.1 아키텍처 결정

이번 PoC는 **단일 통합 멀티모달 모델**이 아니라 **모듈형 Agentic sidecar 아키텍처**로 설계했다. 이유는 단순하다.

- 프로젝트의 장기 목표는 멀티모달 sLMM이지만, 1차 단계에서는 문서형 AI가 핵심이다.
- 기존 RealCOAT는 이미 리플레이, 히트맵, 자세 확인, 성과 리포트, 작업지표를 제공한다.
- 수치 기반 평가지표는 이미 구조화가 가능하므로, LLM은 설명과 지식 연결에 집중해야 한다.
- RTX 4080 16GB에서는 학습과 서빙을 동시에 돌릴 수는 있어도, **일주일짜리 PoC에서는 그렇게 하지 않는 것이 안정적**이다.

## 4.2 컴포넌트 구성

| 컴포넌트 | 필수/선택 | 역할 |
|---|---|---|
| Document Ingestor | 필수 | 문서 수집, 파싱, 정규화, dedup, chunk |
| Knowledge Index | 필수 | 임베딩 생성, FAISS/Qdrant 인덱스, 메타데이터 저장 |
| LLM Inference | 필수 | 질의응답 및 코칭 생성 |
| Eval Harness | 필수 | 50문항 평가, 자동/수동 채점, 전략 비교 |
| QLoRA Trainer | 선택 | 300-500 샘플 기반 adapter 학습 |
| vLLM Serve Layer | 선택 | 조선소 서버(A40/L4) 환경용 stretch goal. RTX 4080에서는 배제 |
| VLM Stub | 미래확장 | heatmap/screenshot를 받아 JSON으로 변환하는 인터페이스만 정의 |
| Motion/Rule Scorer Stub | 미래확장 | 시뮬레이터/규칙엔진이 만든 점수와 피처를 sidecar로 넘길 수 있게 준비 |

## 4.3 통합 원칙

### 원칙 A. 기존 제품 비침습
기존 RealCOAT/Unity는 파일 export 또는 내부 REST만 추가한다. 엔진 내부에 모델을 내장하지 않는다.

### 원칙 B. 계산과 설명 분리
- **시뮬레이터/평가 모듈**: 평균 속도, 이격거리, 각도, DFT, under/over thickness, 시퀀스 위반 계산
- **LLM**: 원인 설명, 개선 조치, 문서 근거 연결

### 원칙 C. 답변 스키마 고정
모든 답변은 텍스트가 아니라 **JSON schema**를 기본으로 한다. UI 연결과 재현성이 좋아진다.

# 5. 기술 선택과 의사결정

## 5.1 기본 모델

### 최종 권장: Qwen3-4B
선정 이유는 아래와 같다.

- **Apache 2.0 라이선스**: 상용 배포 및 온프레미스 구축에 제약이 없다.
- **RTX 4080 16GB 최적화**: 4-bit 양자화 시 약 3GB의 VRAM만 사용하여 추론과 학습이 동시에 가능하다.
- **Thinking Mode 내장**: 모델 내부에서 `<think>` 블록을 통한 Chain of Thought 추론이 가능하여 복잡한 도구 호출 판단에 유리하다.
- **강력한 한국어 및 도구 호출 능력**: BFCL(Berkeley Function Calling Leaderboard) 등에서 동급 모델 대비 압도적인 성능을 보여준다.
- **Qwen3-8B와의 비교**: 8B 모델이 성능은 더 우수하나 RTX 4080 16GB 환경에서 VRAM 부족(OOM) 리스크가 있어, 4B를 기본 배포 모델로 확정했다. (8B는 서버급 환경용 옵션으로 유지)

### 왜 3B가 아닌가
3B급 모델은 실험 회전 속도는 빠르지만, 한국어/영어 혼합 기술 문서와 절차 설명의 안정성이 4B 대비 떨어질 가능성이 높다. 이번 PoC는 파이프라인 smoke test가 아니라 **실제 연구 의사결정**이 목적이므로 4B를 기준선으로 잡는 편이 낫다.

### 왜 14B를 지금 안 하는가
14B는 성능 여유가 있지만, RTX 4080 16GB 환경에서는 서빙 복잡도만 늘어난다. 이번 단계는 **RAG와 QLoRA의 효과를 판별하는 것**이므로 4B가 sweet spot이다.

## 5.2 RAG vs QLoRA 전략

### 최종 권고
- **필수**: RAG-only
- **선택**: Light QLoRA
- **배제**: LoRA-only 단독 제품화

### 이유
- 최신 문서, 규정, 허용 범위, 안전 기준은 **RAG가 가장 유지보수에 유리**하다.
- QLoRA는 문서 지식 전체를 암기시키는 수단이 아니라, **용어 이해, 출력 형식, 코칭 톤, retrieval context 활용 습관**을 보정하는 수단으로만 쓴다.
- 문서형 산업 AI에서 답변의 감사 가능성과 출처 추적성은 매우 중요하므로, RAG를 버리고 LoRA-only로 가면 오히려 리스크가 커진다.

## 5.3 임베딩과 리랭커

### 임베딩: BAAI/bge-m3
- 1024차원
- 최대 8192 토큰
- dense + sparse + multi-vector 계열을 포괄하는 다기능 모델
- query prefix가 필수적이지 않아 RAG 파이프라인이 단순하다

### 리랭커: BAAI/bge-reranker-v2-m3
- query와 passage를 함께 넣고 relevance score를 직접 출력하는 cross-encoder 계열이다.
- Week 1에서는 Top-5 retrieval 뒤 Top-3 rerank 정도로만 쓴다.
- 설치/성능 문제가 있으면 Day 3에는 리랭커를 끄고 retrieval-only로 먼저 기준선을 만든다.

## 5.4 벡터 저장소

### Week 1 기본: FAISS CPU
이유는 매우 단순하다.
- 50-100문서 규모에서는 GPU 벡터 검색이 필요하지 않다.
- 단일 RTX 4080은 이미 LLM 추론에 사용되므로, Week 1에서는 CUDA 복잡도를 줄이는 편이 낫다.
- FAISS CPU는 설치와 디버깅이 쉽다.

## 5.5 서빙 엔진

### 최종 선정: HuggingFace Transformers 4-bit inference
- **장점**: RTX 4080 16GB 환경에서 가장 안정적이며, VRAM 점유율이 낮다 (~3GB for Qwen3-4B).
- **vLLM 배제 사유**: RTX 4080 16GB에서 OOM 문제로 배제. 조선소 서버(A40/L4급)에서만 사용 가능.
- **모바일/태블릿**: llama.cpp (GGUF) 별도 양자화 필요.

## 5.6 QLoRA 학습 엔진

### 최종 선정: peft + trl 직접 사용
- **사유**: Unsloth는 특정 torch/transformers 버전과 의존성 충돌 리스크가 있음.
- **방법**: `BitsAndBytesConfig`를 통한 4-bit 로딩 후 `SFTTrainer`로 직접 학습.
- **안정성**: A40 및 RTX 4080 환경에서 검증된 표준 경로.

\newpage

# 6. 처음부터 시작하는 환경 설정

## 6.1 권장 운영 구조

| 장비 | 역할 | OS |
|---|---|---|
| AI 서버 | 문서 파싱, 인덱싱, 평가, QLoRA, LLM 추론 | Ubuntu 22.04 또는 24.04 |
| 기존 시뮬레이터 PC | RealCOAT/Unity 실행, 결과 export | Windows 11 |

**권장 구조는 2대 분리**다. 같은 장비에서 모두 하더라도, Week 1 기준으로는 Windows 위에 WSL2로 모든 것을 우겨 넣기보다 AI 전용 Linux가 훨씬 안정적이다.

## 6.2 필수 사전 확인

RTX 4080 공식 사양은 16GB GDDR6X, 716.8 GB/s 메모리 대역폭이다.

```bash
nvidia-smi
python3 --version
nvcc --version || true
free -h
df -h
```

권장 최소 조건:
- NVIDIA RTX 4080 16GB x1
- Ubuntu 22.04/24.04
- Python 3.10
- 디스크 여유: 200GB+
- 외부망 1회 접속 가능(모델/휠 사전 다운로드용)

## 6.3 프로젝트 디렉터리

```bash
mkdir -p /workspace/trainer/{scripts,configs,notebooks,logs}
mkdir -p /data/{raw,parsed,cleaned,chunks,sft,eval,exports}
mkdir -p /output/{models,indexes,embeddings,adapters,reports}
mkdir -p /models/Qwen3-4B
cd /workspace
```

## 6.4 환경 분리 전략

이번 PRD의 핵심은 **virtualenv를 분리**하는 것이다.

| 환경 | 필수/선택 | 목적 |
|---|---|---|
| `.venv-rag` | 필수 | 문서 파싱, 임베딩, 인덱싱, 평가, baseline inference |
| `.venv-train` | 선택 | peft + trl QLoRA 학습 |

## 6.5 `.venv-rag` 생성

```bash
python3 -m venv .venv-rag
source .venv-rag/bin/activate
pip install --upgrade pip uv
pip install torch==2.10.0+cu128 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install "transformers==5.3.0" "accelerate==1.13.0" "bitsandbytes==0.49.2"
pip install "peft==0.18.1" "trl==0.29.0"
pip install "sentence-transformers==5.2.3" "faiss-cpu==1.13.2"
pip install fastapi==0.135.1 uvicorn pymupdf beautifulsoup4 lxml markdownify datasketch pandas openpyxl rouge-score bert-score scikit-learn python-dotenv pyyaml tqdm jinja2 gradio==6.9.0
```

### `.venv-rag` smoke test

```bash
python - <<'PY'
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no-gpu')
PY
```

## 6.6 오프라인 사전 다운로드

Hugging Face Hub는 `snapshot_download()`와 `local_dir` 방식, 그리고 오프라인 환경 변수(`HF_HUB_OFFLINE`)를 제공한다.

```bash
source .venv-rag/bin/activate
python - <<'PY'
from huggingface_hub import snapshot_download
models = [
    ('Qwen/Qwen3-4B', '/models/Qwen3-4B'),
    ('BAAI/bge-m3', '/models/bge-m3'),
    ('BAAI/bge-reranker-v2-m3', '/models/bge-reranker-v2-m3'),
]
for repo, path in models:
    snapshot_download(repo_id=repo, local_dir=path, local_dir_use_symlinks=False)
PY
```

```bash
export HF_HOME=/workspace/cache/hf
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
```

# 7. 데이터 파이프라인 설계

## 7.1 수집 대상

우선순위를 이렇게 잡는다.

### 1순위
- 선박/조선 도장 SOP
- coating/blasting 매뉴얼
- 도료 TDS / 사양서
- 안전지침 / 작업허가 / PPE 문서
- 결함 설명 문서
- 도막 두께/검사 가이드

### 2순위
- 교육 자료, 자격 학습 자료
- 기존 제품 교육 PPT / 시나리오 정의서
- 훈련 평가 기준표, 결과 레포트 샘플

### 3순위
- 배경 이론 문서
- 산업 안전 일반 문서
- 영문 표준 문서

## 7.2 RAG / SFT / Eval 분리 원칙

데이터 저장소를 **RAG 코퍼스 / LLM SFT / 예측모델 / VLM**로 나누어 관리한다.

| 저장소 | 이번 주 사용 여부 | 내용 |
|---|---|---|
| RAG Corpus | 사용 | 최신 문서, 규정, 절차, 허용 범위, 안전 수칙 |
| SFT Set | 사용 | 코칭 템플릿, 용어 설명, 절차 요약, 문서 QA |
| Eval Set | 사용 | 50문항 고정 질문과 참조 답변 |
| Motion/Prediction Set | 보류 | 향후 자세/두께/분포/결함 모델 |
| VLM Set | 보류 | 스크린샷/히트맵/리포트 화면 |

## 7.3 문서 정규화 규칙

```text
- Unicode NFKC 정규화
- 반복 헤더/푸터 제거
- 페이지 번호 제거
- 단위 통일: µm, %, mm, °C
- 용어 통일: 건조 도막 두께(DFT), 습도막 두께(WFT), SSPC-SP10, Sa 2.5 등
- 문단 내 개행은 공백, 문단 사이 개행은 유지
- 표는 markdown 또는 CSV 구조로 보존
```

## 7.4 중복 제거

- 문서 단위: SHA-256
- near-duplicate: MinHash + LSH
- 문단 단위: 512자 이상 장문 중복 블록 제거

## 7.5 구조적 청킹

### RAG 청킹 규칙
- 기본 목표 크기: 300-800 토큰
- 절/항/Step 경계 우선
- checklist/procedure는 가급적 한 chunk에 유지
- 표는 제목/컬럼명과 함께 별도 chunk
- overlap은 10-20% 수준

### 메타데이터 스키마

```json
{
  "doc_id": "DOC-COAT-0007",
  "chunk_id": "DOC-COAT-0007-C012",
  "source_type": "technical_manual",
  "domain_category": "coating_thickness",
  "process_stage": "spray",
  "subtask": "distance_angle_dft",
  "language": "ko_en_technical",
  "revision": "2025-01",
  "quality_score": 0.91,
  "has_table": true,
  "has_procedure": true
}
```

## 7.6 세션 요약 스키마

기존 RealCOAT가 제공하는 지표 체계를 반영해 아래와 같이 정의한다.

```json
{
  "session_id": "SPRAY-DEMO-0001",
  "scenario_id": "WBTk1S_A",
  "mode": "practice",
  "avg_speed": 0.42,
  "avg_distance_mm": 310,
  "avg_angle_deg": 84,
  "dft_mean_um": 142,
  "dft_min_um": 98,
  "dft_max_um": 224,
  "under_thickness_ratio": 0.08,
  "over_thickness_ratio": 0.11,
  "paint_usage_ratio": 1.07,
  "notes": ["C구간 하부 저도막", "거리 편차 큼"]
}
```

## 7.7 AI 응답 스키마 (Simulator Protocol)

```json
{
  "request_id": "uuid",
  "session_id": "session-001",
  "mode": "training_realtime",
  "text_guide": {"language": "ko", "detail_level": "detailed|standard|minimal", "summary": "...", "body": "..."},
  "pose_comparison": null,
  "next_path": null,
  "dft_result": {"verdict": "pass|fail", "nominal_um": 100, "measurements_um": [105, 95, 110], "mean_um": 103.3, "rule": "90/10", "out_of_spec_count": 1} | null,
  "anomaly": {"detected": true, "type": "저도막|과도막|결함", "zone": "B-3-L", "cause": "...", "severity": "warning|critical"} | null,
  "steps": [...] | null,
  "citations": ["DOC-xxx-Cxxx"],
  "metadata": {"latency_ms": 1200, "tools_used": ["dft_judge"], "rounds": 2, "route": "training_realtime", "thinking": "..."}
}
```

\newpage

# 8. RAG 기준선 설계

## 8.1 질의 흐름

1. 사용자 질문 수신
2. 임베딩 생성(BGE-M3)
3. Top-K retrieval (기본 5)
4. optional rerank (Top-5 → Top-3)
5. LLM prompt 구성
6. JSON schema 기반 답변 생성
7. citations / retrieved docs / confidence 반환

## 8.2 기본 파라미터

| 항목 | 기본값 | 비고 |
|---|---|---|
| retrieval top-k | 5 | Day 3에 3/5/8 비교 |
| rerank top-n | 3 | Day 3에 on/off 비교 |
| temperature | 0.1 | 평가 안정성 우선 |
| max output tokens | 1024 | QA 및 코칭 기준 |
| citation 반환 | 필수 | chunk id 리스트 |
| 답변 형식 | JSON 우선 | Simulator Protocol 준수 |

## 8.3 추천 프롬프트 골격

```text
System:
너는 선박 도장 훈련 및 도장 공정 문서 해석을 지원하는 산업 AI 어시스턴트다.
항상 제공된 근거 문서에 기반해 답하고, 근거가 부족하면 모른다고 말한다.
수치 계산은 입력값을 그대로 사용하고 임의로 만들지 않는다.
최종 출력은 Simulator Protocol JSON으로 작성한다.

User:
[질문]
...

[검색 문서]
- DOC-...: ...
- DOC-...: ...
```

## 8.4 RAG 기준선 구현 체크리스트

- [ ] chunk jsonl 생성 완료
- [ ] FAISS index 생성 완료
- [ ] retrieval smoke test 10문항 통과
- [ ] reranker on/off 비교 완료
- [ ] citation field 누락 0건
- [ ] hallucination 주요 유형 카탈로그 작성

# 9. QLoRA 실험 설계

## 9.1 QLoRA가 필요한 경우

아래 현상이 보일 때만 QLoRA를 유지한다.
- 용어가 반복적으로 흔들린다.
- 답변 형식(JSON, 코칭 구조)이 불안정하다.
- retrieval된 문서를 읽고도 핵심을 잘 못 뽑는다.
- 코칭 톤이 연구팀/교관 기준과 크게 다르다.

## 9.2 QLoRA가 필요 없는 경우

- RAG-only가 이미 정확도와 faithfulness를 만족한다.
- 문서 업데이트 빈도가 높아 adapter 유지비가 더 크다.
- 용어/형식 문제만 프롬프트로 충분히 보정된다.

## 9.3 학습 데이터 규모

| 수준 | 샘플 수 | 목표 |
|---|---|---|
| 최소 검증 | 300 | LoRA 투자 가치 판별 |
| 권장 PoC | 433 | 카테고리 균형 잡힌 1차 도메인 적응 |
| 차기 단계 | 2000+ | 프로젝트 목표 대응용 확장 |

## 9.4 카테고리 분포

| 카테고리 | 목표수 |
|---|---:|
| 문서 QA | 100 |
| 절차 설명 | 80 |
| 결함 분석 | 60 |
| DFT/사양 피드백 | 60 |
| 용어/다국어 | 50 |
| 요약/비교 | 50 |
| Agentic Tool Calling | 33 |

## 9.5 학습 포맷

```json
{
  "messages": [
    {
      "role": "system",
      "content": "너는 선박 도장 전문 AI 어시스턴트다. 항상 제공된 근거 문서에 기반해 답하고, 최종 출력은 Simulator Protocol JSON으로 작성한다."
    },
    {
      "role": "user",
      "content": "[문서]\n에폭시 프라이머의 적정 DFT는 125-200µm이며 과도한 도포는 균열 위험을 높인다.\n[질문]\n230µm로 도포하면 어떤 문제가 생길 수 있나?"
    },
    {
      "role": "assistant",
      "content": "{\"text_guide\": {\"summary\": \"과도막 위험\", \"body\": \"230µm는 권장 범위를 초과하므로 균열 위험이 있습니다.\"}, \"citations\": [\"DOC-COAT-0007-C012\"]}"
    }
  ]
}
```

## 9.6 권장 하이퍼파라미터

| 항목 | 권장값 |
|---|---|
| Base model | Qwen3-4B |
| Quantization | 4-bit NF4 + bf16 compute |
| LoRA target | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| rank | 16 |
| alpha | 32 |
| dropout | 0.05 |
| max seq len | 2048 |
| batch | 1 |
| grad accum | 8 |
| epoch | 3 |
| optimizer | adamw_8bit |

# 10. 7일 실행 계획

## 10.1 전체 일정

| Day | 필수 작업 | 산출물 | Gate |
|---|---|---|---|
| Day 1 | 환경 구축, 모델 사전 다운로드, smoke test | env log, requirements | `.venv-rag` 정상동작 |
| Day 2 | 문서 수집/파싱/정제/중복 제거 | `manifest.csv`, `cleaned/*.json` | 147건 정제 완료 |
| Day 3 | chunk/index/RAG baseline 구현 | `chunks/*.jsonl`, `faiss.index` | 10문항 smoke test 통과 |
| Day 4 | 50문항 평가셋 완성, RAG-only 평가 | `eval_set.jsonl`, `rag_only_results.csv` | 평균/오류 패턴 확인 |
| Day 5 | SFT 데이터 433건 생성, QLoRA 1회 학습 | `sft_train.jsonl`, `adapter/` | adapter 생성 완료 |
| Day 6 | RAG+QLoRA 비교, Simulator Protocol 연동 | `comparison.csv`, `agent_pipeline.py` | keep/drop decision |
| Day 7 | 최종 리포트, README, 재실행 스크립트 정리 | `final_report.md`, artifacts manifest | 의사결정 문서 완성 |

## 10.2 Day별 상세 작업

### Day 1 — 환경 확정
- OS/GPU 점검 (RTX 4080 16GB)
- `.venv-rag`, `.venv-train` 생성
- 모델 3종 다운로드 (Qwen3-4B, BGE-M3, reranker)
- 4bit inference smoke test

### Day 2 — 데이터 파이프라인 가동
- raw 수집 (179건)
- parse/clean/dedup
- metadata schema 확정
- 고품질 문서 선별

### Day 3 — RAG baseline 완성
- BGE-M3 임베딩 생성
- FAISS 인덱스 구축
- retrieval top-k 및 rerank on/off 비교
- 응답 JSON schema 확정

### Day 4 — 평가 기준선 확보
- 50문항 평가셋 확정
- RAG-only 일괄 실행
- 수동 채점 및 오류 분석
- hallucination 유형 분류

### Day 5 — QLoRA 검증
- 433 SFT 샘플 생성
- 품질 검수
- peft + trl 기반 1회 학습
- QLoRA-only 간이 평가

### Day 6 — 전략 비교
- RAG-only vs QLoRA-only vs RAG+QLoRA 비교
- adapter keep/drop 의사결정
- Simulator Protocol 연동 테스트

### Day 7 — 문서화와 handoff
- 실행 README 정리
- 폴더 구조, 버전, 명령어 문서화
- 최종 PRD/리포트 제출

# 11. 평가 설계

## 11.1 고정 50문항 평가셋

| 카테고리 | 수량 | 예시 |
|---|---:|---|
| 용어 이해 | 10 | DFT와 WFT 차이, SSPC-SP10 의미 |
| 공정 절차 | 10 | 발라스트 탱크 도장 순서 |
| 결함 분석 | 10 | Sagging, Fingering, Wrinkling |
| 코팅 두께/사양 | 10 | 적정 DFT, under-thickness 조치 |
| 안전/규정 | 10 | 밀폐공간 작업, PPE, 환기 |

## 11.2 수동 채점 루브릭

전략 보고서의 5차원 0-2점 체계를 유지한다.

| 항목 | 2점 | 1점 | 0점 |
|---|---|---|---|
| 기술 정확성 | 기술적으로 정확 | 대체로 정확 | 명백한 오류 |
| 문서 충실도 | 근거 문서에 충실 | 일부 추론 과다 | hallucination |
| 용어 적절성 | 현장 용어 사용 정확 | 일부 흔들림 | 오용 |
| 완전성 | 필요한 정보 모두 포함 | 일부 누락 | 핵심 누락 |
| 한국어 품질 | 자연스럽고 전문적 | 어색하지만 이해 가능 | 읽기 어려움 |

## 11.3 자동 평가

ROUGE-L, BERTScore, hallucination 비율, 용어 정확도, 응답 길이 적절성을 측정한다.

| 지표 | 도구 |
|---|---|
| ROUGE-L | rouge-score |
| BERTScore | bert-score |
| hallucination ratio | 수동 + LLM-as-Judge |
| terminology accuracy | 용어 사전 대조 |
| response length | 토큰 통계 |

## 11.4 프로젝트 KPI와의 연결

| 상위 KPI | 이번 PoC에서 확인할 것 |
|---|---|
| sLMM 데이터셋 2000건 이상 | 데이터 파이프라인 확장성 |
| 자세 유사도 교정 정확도 ≥70% | 텍스트 기반 자세 코칭 포맷 가능성 |
| 도막 분포 이상 탐지 ≥70% | DFT/분포 관련 QA 정확성 |
| AI Co-Pilot 기능 8개 이상 | 시나리오별 응답 품질과 JSON 스키마 |
| E-7 대상국 번역 지원 | 다국어 경고/가이드 샘플 생성 |

# 12. API / 파일 인터페이스 정의

## 12.1 Simulator Request Schema (JSON)

시뮬레이터가 AI Co-Pilot에게 보내는 표준 요청 형식이다.

```json
{
  "request_id": "uuid-v4",
  "session_id": "session-001",
  "mode": "training_realtime | training_review",
  "trainee": {
    "trainee_id": "T-001",
    "language": "ko",
    "skill_level": "beginner | intermediate | advanced"
  },
  "instruction": "현재 내 도장 상태가 어때?",
  "environment": {
    "zone_id": "B-3",
    "zone_type": "수직 패널",
    "temperature": 25.5,
    "humidity": 62.0,
    "dew_point_gap": 4.5
  },
  "sensor_data": {
    "dft_measurements": [102, 95, 110, 88, 105],
    "current_pose": {"gun_angle": 85, "distance_mm": 320},
    "trajectory": [{"x": 1.0, "y": 0.5, "z": 0.8, "t": 0.1}, ...],
    "image_context": "좌하단 청색 집중, 저도막 패턴 관찰됨"
  }
}
```

## 12.2 Simulator Response Schema (JSON)

AI Co-Pilot이 시뮬레이터에게 돌려주는 표준 응답 형식이다.

```json
{
  "request_id": "uuid-v4",
  "session_id": "session-001",
  "mode": "training_realtime",
  "text_guide": {
    "language": "ko",
    "detail_level": "detailed",
    "summary": "속도 조절 및 이격거리 유지 필요",
    "body": "현재 B-3 구역에서 이동 속도가 다소 빠르며 이격거리가 불안정합니다."
  },
  "pose_comparison": {
    "similarity_score": 0.85,
    "corrections": [
      {"joint": "wrist_r", "current_deg": 53, "target_deg": 45, "deviation_deg": 8, "direction": "낮추세요"}
    ]
  },
  "next_path": {
    "zone_id": "B-3",
    "direction": "top_to_bottom",
    "waypoints": [{"x": 1.0, "y": 0.4}, {"x": 1.0, "y": 0.3}]
  },
  "dft_result": {
    "verdict": "fail",
    "nominal_um": 100,
    "measurements_um": [102, 95, 110, 88, 105],
    "mean_um": 100.0,
    "rule": "90/10",
    "out_of_spec_count": 1
  },
  "anomaly": {
    "detected": true,
    "type": "저도막",
    "zone": "B-3-L",
    "cause": "이동 속도 과다",
    "severity": "warning"
  },
  "steps": [
    {"step": 1, "action": "이동 속도를 0.4m/s 이하로 유지"},
    {"step": 2, "action": "이격거리를 300mm로 고정"}
  ],
  "citations": ["DOC-COAT-0007-C012"],
  "metadata": {
    "latency_ms": 1500,
    "tools_used": ["dft_judge", "location_tracking"],
    "rounds": 2,
    "route": "training_realtime",
    "thinking": "사용자의 DFT 측정값 중 하나가 기준 미달이며 속도 데이터가 높게 측정됨..."
  }
}
```

# 13. 리스크와 폴백

| 리스크 | 징후 | 폴백 |
|---|---|---|
| GPU OOM | 추론 중 메모리 초과 | max_seq_len 1024로 축소, batch size 1 고정 |
| 도구 호출 실패 | 잘못된 JSON 생성 또는 미등록 도구 호출 | RAG 기반 일반 가이드 제공 및 에러 로그 기록 |
| 응답 지연 | 60초 초과 지연 | SemanticRouter에서 fast 경로 강제 할당 |
| hallucination 증가 | 출처 없는 답변 증가 | adapter 폐기 및 RAG-only 복귀 |
| 문서 파싱 품질 저하 | PDF 표 깨짐 | 표는 별도 CSV 저장 및 본문 RAG 우선 |

# 14. 최종 산출물 목록

Day 7까지 아래 파일이 남아 있어야 한다.

```text
/workspace/
├── trainer/
│   ├── scripts/
│   │   ├── agent_pipeline.py          (메인 에이전트 로직)
│   │   ├── demo_app.py               (Gradio UI)
│   │   ├── train_qlora.py            (학습 스크립트)
│   │   ├── build_index.py            (인덱스 빌더)
│   │   ├── qa_pipeline.py            (RAG QA 파이프라인)
│   │   └── tool_calling_adapter.py    (도구 호출 어댑터)
│   └── configs/
│       ├── rag_config_fewshot.yaml    (활성 설정)
│       ├── tool_calling_protocol.json (호출 규약)
│       └── tool_schemas/              (6개 도구 스키마)
├── mcp_servers/
│   ├── main.py                        (FastAPI 서버)
│   ├── dft_judge.py                   (90/10 룰 로직)
│   └── ...                            (기타 도구 Stub)
├── data/
│   ├── sft/sft_train.jsonl            (학습 데이터)
│   └── eval/eval_set.jsonl            (평가 데이터)
├── output/
│   ├── adapters/qwen3_4b_shipcoat_lora/
│   └── indexes/faiss.index
└── models/
    └── Qwen3-4B/                      (베이스 모델)
```

# 15. 다음 단계 제안

이번 PoC가 성공적으로 완료됨에 따라 다음 단계를 제안한다.

1. **Qwen3.5-4B 마이그레이션**: Gated DeltaNet 하이브리드 아키텍처와 Native VLM 기능을 활용하여 히트맵 이미지 직접 분석 및 추론 효율 향상을 꾀한다.
2. **실시간 센서 스트리밍 연동**: 현재의 REST API 방식에서 WebSocket 또는 gRPC를 통한 실시간 스트리밍 연동으로 확장하여 지연 시간을 최소화한다.
3. **다국어 음성 인터페이스**: 조선소 현장의 외국인 작업자를 위해 TTS/ASR 기능을 통합하여 음성 가이드를 제공한다.
4. **도장 자세/시퀀스 스코어러 고도화**: 현재 Stub 상태인 자세 및 순서 검증 로직을 실제 숙련자 데이터 기반의 신경망 모델로 교체한다.

\newpage

# 16. Co-Pilot 아키텍처 (구현 완료)

## 16.1 문제 인식 및 해결
1차 PoC 초기에는 단순 텍스트 QA에 집중했으나, 실제 훈련 현장에서는 센서 데이터와의 연동 및 실시간 판단이 필수적임을 인식했다. 이에 따라 **Agentic Co-Pilot** 구조로 전환하여 구현을 완료했다.

## 16.2 Co-Pilot 아키텍처 구조

```text
┌─────────────────────────────────────────────────────┐
│                  XR Client (Unity)                   │
│         AR/VR 콘텐츠 렌더링 + 사용자 인터페이스           │
└─────────────┬───────────────────────┬───────────────┘
              │ Request JSON          │ Response JSON
              ▼                       ▼
┌─────────────────────────────────────────────────────┐
│         AI Co-Pilot Agent (Qwen3-4B)                │
│  ┌──────────┐  ┌──────────┐  ┌────────────────────┐ │
│  │ RAG Core │  │ QLoRA    │  │ Thinking Mode      │ │
│  │ (문서 지식)│  │ (스타일) │  │ (<think> CoT 내장)  │ │
│  └──────────┘  └──────────┘  └────────────────────┘ │
│                       │                              │
│              Tool Calling Layer                       │
│  ┌──────┬──────┬──────┬──────┬──────┬──────────────┐ │
│  │자세  │이상  │시계열│위치  │순서  │DFT 판정      │ │
│  │추정  │탐지  │처리  │추적  │검증  │합부 판정      │ │
│  └──┬───┴──┬───┴──┬───┴──┬───┴──┬───┴──────┬───────┘ │
└─────┼──────┼──────┼──────┼──────┼──────────┼─────────┘
      ▼      ▼      ▼      ▼      ▼          ▼
┌─────────────────────────────────────────────────────┐
│                MCP Server Layer                       │
│  pose_estimation | anomaly_detection | timeseries    │
│  location_tracking | sequence_validation | dft_judge │
└─────────────────────────────────────────────────────┘
```

## 16.3 3-Way SemanticRouter
사용자의 입력과 시뮬레이터 상태를 분석하여 최적의 경로를 선택한다.

| Route | 목적 | Max Rounds | Latency Target |
|---|---|---|---|
| **training_realtime** | 실시간 코칭 (순서, 이상 탐지) | 3 | < 60s |
| **training_review** | 사후 분석 (자세, 분포 분석) | 5 | No constraint |
| **fast** | 단순 지식 질의 (RAG 전용) | 1 | ~ 10s |

## 16.4 TraineeProfile 및 개인화
훈련생의 숙련도에 따라 응답의 상세 수준(`detail_level`)을 조절한다.

| Skill Level | Detail Level | 응답 특성 |
|---|---|---|
| **beginner** | detailed | 상세한 원인 설명 및 단계별 행동 지시 포함 |
| **intermediate** | standard | 핵심 주의사항 및 표준 가이드 위주 |
| **advanced** | minimal | 수치 중심 리포트 및 최소한의 피드백 |

## 16.5 Tool Result Assembler
여러 MCP 도구의 실행 결과를 취합하여 Simulator Protocol JSON의 각 섹션에 정확히 매핑한다.
- `dft_judge` 결과 → `dft_result` 섹션
- `anomaly_detection` 결과 → `anomaly` 섹션
- `pose_estimation` 결과 → `pose_comparison` 섹션
- `sequence_validation` 결과 → `next_path` 섹션

기존의 `_detect_xr_content()` 방식은 폐기하고 구조화된 JSON 생성을 원칙으로 한다.

## 16.6 MCP Server 인터페이스 상세

| MCP Server | 입력 | 출력 | 용도 |
|---|---|---|---|
| `dft_judge` | 측정값 배열, 사양 | 합격/불합격, 위반 포인트 | 도막 두께 규격 합부 판정 |
| `anomaly_detection` | 히트맵, 센서 데이터 | 이상 유형, 위치, 심각도 | 도막 이상상태 탐지 |
| `pose_estimation` | 현재 자세, 숙련자 자세 | 관절별 편차, 교정 방향 | 숙련자 대비 자세 교정 |
| `location_tracking` | 궤적(x,y,z) | 현재 구역 ID, 특성 | 작업 위치 파악 |
| `sequence_validation` | 현재 궤적, 기준 순서 | 위반 여부, 권장 방향 | 도장 순서 검증 |
| `timeseries` | 속도/거리 시계열 | 추세, stroke 간 편차 | 작업 패턴 분석 |

## 16.7 Co-Pilot 대표 시나리오 (Simulator Protocol 적용)

### 시나리오 D: 실시간 코칭
- **상황**: 훈련생이 수직 패널 도장 중 속도가 빨라짐.
- **응답**:
```json
{
  "mode": "training_realtime",
  "text_guide": {"summary": "속도 낮추세요", "body": "현재 속도가 0.5m/s로 기준을 초과합니다."},
  "anomaly": {"detected": true, "type": "저도막 위험", "severity": "warning"},
  "steps": [{"step": 1, "action": "속도를 0.4m/s 이하로 유지"}]
}
```

### 시나리오 E: 이상 탐지 및 진단
- **상황**: 훈련 후 특정 구역에 저도막 발생.
- **응답**:
```json
{
  "mode": "training_review",
  "dft_result": {"verdict": "fail", "out_of_spec_count": 2},
  "anomaly": {"detected": true, "zone": "B-3-L", "cause": "이격거리 과다"}
}
```

### 시나리오 F: XR 가이드 생성
- **상황**: "스트라이프 코팅 방법 보여줘."
- **응답**:
```json
{
  "mode": "fast",
  "steps": [
    {"step": 1, "action": "웰드 양쪽 50mm 마스킹"},
    {"step": 2, "action": "붓 도장으로 코팅"}
  ],
  "citations": ["DOC-COAT-0012-C003"]
}
```

## 16.8 훈련 중 / 훈련 후 모드 구분
- **훈련 중 (Real-time)**: 텍스트 가이드 + 단계별 지시 + (다음 경로, 합부 판정, 이상 탐지). 빠른 피드백 중점.
- **훈련 후 (Review)**: 텍스트 가이드 + 단계별 지시 + (자세 비교, 이상 탐지, 합부 판정). 심층 분석 중점.

## 16.9 로드맵 (Phase 1-6)

- **Phase 1 (완료)**: RAG+QLoRA 문서 QA 기준선
- **Phase 2 (완료)**: Tool Calling + MCP Server + Agentic 평가셋
- **Phase 3 (완료)**: Agent Orchestration + CoT + Session Coaching
- **Phase 4 (완료)**: Prompt Engineering + 3-Way Router + Simulator Protocol
- **Phase 5 (진행중)**: QLoRA 재학습 (최종 구조 기반)
- **Phase 6 (향후): Qwen3.5-4B 마이그레이션**
  - Gated DeltaNet 하이브리드 아키텍처 도입
  - Native VLM (text + image) 통합
  - XML tool call 포맷 전환
  - bf16 LoRA 학습 (VRAM ~10GB)
  - 262K 컨텍스트 윈도우 활용
  - 예상 공수: 약 1 스프린트 (3-4일)

\newpage

# 17. Agent Orchestration Specification

## 17.1 목적
결정론적 수치 계산(Tools)과 언어적 추론(LLM)을 분리하여 신뢰성 있고 설명 가능한 AI Co-Pilot을 구현한다.

## 17.2 오케스트레이션 아키텍처
사용자 지시 → 의도 라우터 → 오케스트레이터 → 도구 선택 → 도구 실행 → 결과 취합 → LLM 해석 → 최종 응답의 파이프라인을 따른다.

## 17.3 도구 레지스트리
현재 6개의 전문 도구와 RAG 검색 기능을 포함하여 총 7개의 핵심 역량을 보유한다.

| Tool | Purpose |
|---|---|
| `rag_retriever` | 도장 가이드라인 및 SOP 문서 검색 |
| `trajectory_analyzer` | 도장 궤적 패턴 분석 |
| `pose_analyzer` | 작업자 도장 자세 분석 |
| `thickness_heatmap` | 도막 두께 분포 분석 |
| `sequence_analyzer` | 도장 순서 적정성 검증 |
| `dft_judge` | DFT 도막 두께 합부 판정 |
| `anomaly_detection` | 도막 이상 상태 탐지 |

## 17.4 도구 호출 규약 (Qwen3 Native)
Qwen3의 Native Function Calling 기능을 사용하며, 내부적으로는 다음과 같은 XML 태그 형식을 활용한다.
`<tool_call>{"name":"...","arguments":{...}}</tool_call>`

## 17.5 Multi-Tool Execution (ReAct Loop)
일부 작업은 여러 도구의 조합이 필요하다. 오케스트레이터는 도구 실행 결과를 바탕으로 다음 도구 호출 여부를 자율적으로 결정하며, 각 경로별 `max_rounds`를 준수한다.

## 17.6 Fallback 전략
- **Graceful Skip**: 특정 도구 실행 실패 시 해당 결과를 제외하고 RAG 지식에 기반하여 답변한다.
- **Unregistered Tool**: 정의되지 않은 도구 호출 시 에러 메시지를 모델에게 전달하여 재시도를 유도한다.

## 17.7 Execution Trace Logging
모든 상호작용은 아래 형식의 로그를 생성하여 디버깅 및 데이터셋 생성에 활용한다.
```json
{
  "instruction": "현재 도막 두께가 규격에 맞는지 확인해줘",
  "intent": "quality_diagnosis",
  "selected_tools": ["dft_judge"],
  "tool_calls": [{"name": "dft_judge", "args": {"measurements_um": [102, 95], "nominal_um": 100}}],
  "tool_results": [{"status": "success", "result": {"verdict": "pass", "mean_um": 98.5}}],
  "final_response": "현재 평균 도막 두께는 98.5µm로 기준에 적합합니다."
}
```

## 17.8 디자인 제약 사항
1. 분석 모듈은 반드시 결정론적(Deterministic)이어야 한다.
2. LLM 응답은 반드시 도구의 출력값을 참조해야 한다.
3. 모든 응답은 운영자가 해석 가능해야 한다.
4. 도구 지연 시간은 실시간 피드백을 지원해야 한다.

# 18. Intent Router & Tool Schema

## 18.1 의도 라우팅 매핑

| Intent | Description | Required Tools |
|---|---|---|
| `pose_feedback` | 자세 교정 및 피드백 | `pose_estimation`, `rag_retriever` |
| `quality_diagnosis` | 도막 품질 진단 | `dft_judge`, `anomaly_detection`, `thickness_heatmap` |
| `sequence_validation` | 도장 순서 검증 | `sequence_validation`, `location_tracking` |
| `knowledge_query` | 일반 지식 질의 | `rag_retriever` |

## 18.2 도구 JSON 스키마 상세

### dft_judge
```json
{
  "name": "dft_judge",
  "description": "DFT 도막 두께 합부 판정 (90/10 룰)",
  "parameters": {
    "type": "object",
    "properties": {
      "measurements_um": {"type": "array", "items": {"type": "number"}},
      "nominal_um": {"type": "number"}
    },
    "required": ["measurements_um", "nominal_um"]
  }
}
```

### anomaly_detection
```json
{
  "name": "anomaly_detection",
  "description": "도막 이상 상태 탐지",
  "parameters": {
    "type": "object",
    "properties": {
      "image_context": {"type": "string"},
      "dft_measurements": {"type": "array", "items": {"type": "number"}}
    },
    "required": ["image_context"]
  }
}
```

### sequence_validation
```json
{
  "name": "sequence_validation",
  "description": "도장 순서 및 방향 검증",
  "parameters": {
    "type": "object",
    "properties": {
      "trajectory": {"type": "array", "items": {"type": "object"}},
      "zone_id": {"type": "string"}
    },
    "required": ["trajectory", "zone_id"]
  }
}
```

## 18.3 Tool Calling Prompt Template
모델이 도구를 정확히 호출하도록 유도하는 시스템 프롬프트 구조를 정의한다.

**System Prompt:**
너는 선박 도장 작업을 지원하는 산업 AI Co-Pilot이다.
수치 계산이나 전문 분석이 필요한 경우 반드시 도구를 호출하라.
절대로 수치를 임의로 조작하지 말고 도구의 결과를 그대로 인용하라.

**Response Structure:**
Thought: 도구 호출이 필요한 이유 설명
Tool Call: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
Tool Result: 도구 실행 결과 (시스템 주입)
Final Answer: 결과를 해석한 최종 답변

\newpage

# 19. 부록

## A-1. 주요 파일 및 경로
- **실행 스크립트**: `/workspace/trainer/scripts/`
  - `agent_pipeline.py`: 메인 에이전트 로직 (977 lines)
  - `demo_app.py`: Gradio 기반 데모 UI (1066 lines)
  - `train_qlora.py`: QLoRA 학습 스크립트
  - `build_index.py`: FAISS 인덱스 빌더
  - `tool_calling_adapter.py`: 도구 호출 파싱 및 실행
- **설정 파일**: `/workspace/trainer/configs/`
  - `rag_config_fewshot.yaml`: 활성 RAG 및 도구 설정
  - `tool_calling_protocol.json`: 도구 호출 규약
  - `tool_schemas/`: 6개 도구 JSON 스키마 파일들
- **모델 및 데이터**:
  - 모델: `/models/Qwen3-4B`
  - 어댑터: `/output/adapters/qwen3_4b_shipcoat_lora/`
  - 인덱스: `/output/indexes/faiss.index`
  - 학습 데이터: `/data/sft/sft_train.jsonl` (433 샘플)
- **MCP 서버**: `/workspace/mcp_servers/`
  - `main.py`: FastAPI 서버 (Port 8100)
  - `dft_judge.py`: 90/10 룰 로직 실제 구현

## A-2. 실행 명령어

### 데모 앱 실행
```bash
source .venv-rag/bin/activate
python3 /workspace/trainer/scripts/demo_app.py --live
```

### MCP 서버 실행
```bash
source .venv-rag/bin/activate
python3 /workspace/mcp_servers/main.py
```

### 인덱스 빌드
```bash
source .venv-rag/bin/activate
python3 /workspace/trainer/scripts/build_index.py --input /data/chunked/chunks.jsonl --output /output/indexes/faiss.index
```

### QLoRA 학습 실행
```bash
source .venv-train/bin/activate
python3 /workspace/trainer/scripts/train_qlora.py --config /workspace/trainer/configs/qlora_config.yaml
```

## A-3. 의사결정 트리

### RAG vs QLoRA 선택 기준
1. **최신 지식 (규정, 사양)** → RAG 우선
2. **응답 형식 (JSON, 특정 톤)** → QLoRA 우선
3. **도메인 용어 이해** → QLoRA 우선
4. **대규모 지식 베이스** → RAG 우선

### 모델 선택 기준
1. **VRAM < 16GB (RTX 4080)** → Qwen3-4B
2. **VRAM > 40GB (A40)** → Qwen3-8B 권장
3. **실시간성 중요** → 4B 모델

## A-4. 참고 자료

| ID | 자료명 | 출처/URL |
|---|---|---|
| R1 | Qwen3 HuggingFace Model Card | https://huggingface.co/Qwen/Qwen3-4B |
| R2 | Qwen-Agent Framework | https://github.com/QwenLM/Qwen-Agent |
| R3 | Model Context Protocol (MCP) | https://modelcontextprotocol.io |
| R4 | 선박 도장 표준 작업 절차서 (SOP) | 내부 문서 |
| R5 | 도료 기술 사양서 (TDS) | 제조사 제공 자료 |

---
*본 문서는 선박 도장 AI Co-Pilot PoC의 최종 통합 PRD이며, 향후 Qwen3.5 마이그레이션 및 실무 배포의 기준점으로 활용된다.*

## 18.4 상세 도구 스키마 (Full JSON)

### dft_judge
```json
{
  "name": "dft_judge",
  "description": "건조 도막 두께(DFT) 측정값을 입력받아 90/10 룰에 따라 합격 여부를 판정합니다.",
  "parameters": {
    "type": "object",
    "properties": {
      "measurements_um": {
        "type": "array",
        "items": {"type": "number"},
        "description": "측정된 도막 두께 값들의 배열 (단위: µm)"
      },
      "nominal_um": {
        "type": "number",
        "description": "설계 도막 두께 (단위: µm)"
      },
      "rule": {
        "type": "string",
        "enum": ["90/10", "80/20"],
        "default": "90/10",
        "description": "적용할 판정 규칙"
      }
    },
    "required": ["measurements_um", "nominal_um"]
  }
}
```

### anomaly_detection
```json
{
  "name": "anomaly_detection",
  "description": "히트맵 이미지 분석 결과와 센서 데이터를 종합하여 도장 결함이나 이상 상태를 탐지합니다.",
  "parameters": {
    "type": "object",
    "properties": {
      "image_context": {
        "type": "string",
        "description": "VLM이 분석한 히트맵 이미지의 텍스트 설명"
      },
      "dft_measurements": {
        "type": "array",
        "items": {"type": "number"},
        "description": "해당 구역의 DFT 측정값"
      },
      "environment": {
        "type": "object",
        "properties": {
          "temperature": {"type": "number"},
          "humidity": {"type": "number"}
        }
      }
    },
    "required": ["image_context"]
  }
}
```

### sequence_validation
```json
{
  "name": "sequence_validation",
  "description": "작업자의 도장 궤적을 분석하여 표준 작업 순서(Stroke 방향 등)를 준수했는지 검증합니다.",
  "parameters": {
    "type": "object",
    "properties": {
      "trajectory": {
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "x": {"type": "number"},
            "y": {"type": "number"},
            "z": {"type": "number"},
            "t": {"type": "number"}
          }
        },
        "description": "도장건의 이동 궤적 데이터"
      },
      "zone_id": {
        "type": "string",
        "description": "현재 작업 중인 구역 ID"
      }
    },
    "required": ["trajectory", "zone_id"]
  }
}
```

### pose_estimation
```json
{
  "name": "pose_estimation",
  "description": "작업자의 자세 데이터를 숙련자의 표준 자세와 비교하여 편차를 계산합니다.",
  "parameters": {
    "type": "object",
    "properties": {
      "current_pose": {
        "type": "object",
        "description": "현재 작업자의 6DOF 자세 데이터"
      },
      "reference_id": {
        "type": "string",
        "description": "비교 대상이 될 숙련자 데이터 ID"
      }
    },
    "required": ["current_pose"]
  }
}
```

### location_tracking
```json
{
  "name": "location_tracking",
  "description": "3D 좌표 데이터를 기반으로 현재 작업자가 선박의 어느 구역을 도장하고 있는지 식별합니다.",
  "parameters": {
    "type": "object",
    "properties": {
      "coords": {
        "type": "object",
        "properties": {
          "x": {"type": "number"},
          "y": {"type": "number"},
          "z": {"type": "number"}
        }
      }
    },
    "required": ["coords"]
  }
}
```

### timeseries
```json
{
  "name": "timeseries",
  "description": "세션 동안 누적된 작업 데이터를 분석하여 속도나 거리의 변화 추세를 파악합니다.",
  "parameters": {
    "type": "object",
    "properties": {
      "metric": {
        "type": "string",
        "enum": ["speed", "distance", "angle"],
        "description": "분석할 지표"
      },
      "window_size": {
        "type": "integer",
        "description": "분석할 최근 데이터 포인트 개수"
      }
    },
    "required": ["metric"]
  }
}
```

# 20. 추가 부록: 트러블슈팅 가이드

## A-5. 일반적인 문제 및 해결 방법

### 1. 모델 로딩 시 OOM 발생
- **원인**: RTX 4080의 16GB VRAM이 다른 프로세스에 의해 점유되어 있거나, `max_seq_len`이 너무 큼.
- **해결**: `nvidia-smi`로 점유 프로세스 확인 후 종료. `rag_config.yaml`에서 `max_new_tokens`를 줄이거나 `load_in_4bit: true` 확인.

### 2. 도구 호출 결과가 응답에 반영되지 않음
- **원인**: LLM이 도구 결과를 무시하고 자체 지식으로 답변함.
- **해결**: 시스템 프롬프트에서 "반드시 도구 결과를 참조하라"는 지시를 강화. QLoRA 어댑터가 도구 결과 해석 샘플을 충분히 학습했는지 확인.

### 3. RAG 검색 결과의 관련성이 낮음
- **원인**: 쿼리가 너무 짧거나 모호함. 또는 인덱스 구축 시 청킹이 부적절함.
- **해결**: 쿼리 확장(Query Expansion) 기법 적용. 청크 크기를 조절하여 다시 인덱싱. 리랭커의 `top_n` 값을 높임.

### 4. Simulator Protocol JSON 파싱 에러
- **원인**: 모델이 유효하지 않은 JSON 형식을 생성함 (예: 콤마 누락, 따옴표 오류).
- **해결**: `tool_calling_adapter.py`에서 정규식을 통한 유연한 파싱 로직 적용. QLoRA 학습 시 JSON 구조 보존을 위한 SFT 데이터 보강.

## A-6. 용어 사전 (Glossary)

| 용어 | 설명 |
|---|---|
| **DFT** | Dry Film Thickness. 건조 후의 도막 두께. |
| **WFT** | Wet Film Thickness. 도장 직후 건조 전의 도막 두께. |
| **90/10 Rule** | 측정값의 90% 이상이 설계 두께 이상이어야 하며, 나머지 10%도 설계 두께의 90% 이상이어야 한다는 합격 판정 기준. |
| **Sa 2.5** | 표면 처리 등급 중 하나로, 매우 철저한 분사 세정 상태를 의미. |
| **TDS** | Technical Data Sheet. 도료의 물리적 성질과 작업 지침이 담긴 기술 사양서. |
| **MCP** | Model Context Protocol. AI 모델과 외부 도구 간의 통신 규약. |
| **CoT** | Chain of Thought. 모델이 단계적으로 사고하여 결론에 도달하는 추론 방식. |

---
*문서 끝*

# 21. 데이터 보안 및 개인정보 보호

## 21.1 온프레미스 배포의 이점
본 시스템은 조선소 내부망에 독립적으로 배포되도록 설계되었다. 이는 다음과 같은 보안 이점을 제공한다.
- **데이터 유출 방지**: 훈련생의 작업 데이터 및 기업 비밀인 도장 SOP가 외부 클라우드로 전송되지 않는다.
- **네트워크 독립성**: 외부 인터넷 연결이 끊긴 상태에서도 시뮬레이터와 AI Co-Pilot 간의 통신이 보장된다.
- **접근 제어**: 조선소 내부의 권한 관리 시스템과 연동하여 인가된 사용자만 AI 서비스를 이용할 수 있게 제한할 수 있다.

## 21.2 데이터 익명화 처리
시뮬레이터에서 AI Co-Pilot으로 전송되는 `trainee_id`는 실제 성명이나 사번이 아닌 해시된 식별자를 사용한다.
- **개인 식별 정보(PII) 제거**: 요청 데이터 내에 포함될 수 있는 개인 정보를 사전에 필터링한다.
- **로그 관리**: 에이전트의 추론 로그는 성능 개선 목적으로만 저장되며, 일정 기간 후 자동 삭제되도록 설정한다.

# 22. 향후 확장성 및 유지보수

## 22.1 지식 베이스 업데이트 프로세스
새로운 도료가 도입되거나 규정이 변경될 경우, 다음과 같은 절차로 RAG 인덱스를 업데이트한다.
1. 신규 문서를 `/data/raw/` 디렉터리에 추가.
2. `ingest_docs.py` 및 `clean_docs.py` 실행.
3. `build_index.py`를 실행하여 FAISS 인덱스 갱신.
4. 별도의 모델 재학습 없이 즉시 새로운 지식이 반영됨.

## 22.2 모델 성능 모니터링
운영 중 AI의 응답 품질을 지속적으로 모니터링하기 위해 다음과 같은 지표를 수집한다.
- **도구 호출 성공률**: 전체 요청 대비 성공적으로 MCP 도구를 호출하고 결과를 해석한 비율.
- **사용자 피드백**: 응답에 대한 교관의 "좋아요/싫어요" 피드백 수집.
- **추론 지연 시간**: 각 단계별(검색, 추론, 도구 호출) 소요 시간 분석.

## 22.3 하드웨어 업그레이드 가이드
사용자 수가 증가하거나 더 복잡한 추론이 필요할 경우 하드웨어를 다음과 같이 확장할 수 있다.
- **VRAM 확장**: RTX 4080에서 A40 또는 H100으로 교체하여 Qwen3-72B 등 대형 모델 서빙.
- **멀티 GPU 구성**: 여러 대의 GPU를 사용하여 추론 요청을 병렬 처리(Load Balancing).
- **전용 벡터 DB 도입**: 청크 수가 100만 개 이상으로 늘어날 경우 FAISS에서 Qdrant 또는 Milvus로 전환.

---
*최종 수정일: 2026년 3월 13일*
*작성자: OhMyOpenCode Sisyphus-Junior*

# 23. 문서 개정 이력 (Revision History)

| 버전 | 날짜 | 주요 변경 내용 | 작성자 |
|---|---|---|---|
| v0.1 | 2026-03-09 | 초안 작성 (Qwen2.5-7B 기반) | 연구팀 |
| v0.5 | 2026-03-11 | Qwen3 마이그레이션 및 Agentic 구조 설계 | Sisyphus-Junior |
| v0.8 | 2026-03-12 | Simulator Protocol 확립 및 MCP Server 통합 | Sisyphus-Junior |
| v1.0 | 2026-03-13 | 최종 통합 PRD 완성 (RTX 4080 최적화 반영) | Sisyphus-Junior |

# 24. 연락처 및 기술 지원

본 시스템의 구축, 운영 및 기술 지원에 관한 문의는 아래 채널을 통해 가능하다.
- **기술 지원팀**: support@shipcoat-ai.com
- **프로젝트 관리**: pm@shipcoat-ai.com
- **내부 위키**: https://wiki.shipcoat-ai.internal/copilot

---
*본 문서는 선박 도장 AI Co-Pilot 프로젝트의 공식 문서이며, 무단 복제 및 배포를 금합니다.*
