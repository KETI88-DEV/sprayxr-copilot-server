---
title: "선박 도장 문서형 AI 1차 PoC 통합 PRD"
subtitle: "RAG-first 아키텍처, 선택적 QLoRA 검증, A40 x1 온프레미스 기준 실행 문서"
date: "2026-03-09"
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
16. 다음 단계 제안
17. 부록

\newpage


# 문서 개요

| 항목 | 내용 |
|---|---|
| 문서 목적 | 기존 PRD, 첨부 PRD, 전략 보고서, 기술 스택 참고문서, 기존 제품 자료를 통합하여 **일주일 안에 1차 PoC 결과를 만드는 실행 문서**로 재정의 |
| 이번 PoC 목표 | 1) 문서형 RAG 기준선 구축, 2) QLoRA가 실제로 필요한지 검증, 3) 기존 RealCOAT/Unity 제품과 충돌 없는 sidecar 구조 정의 |
| 권장 기본 모델 | **Qwen2.5-7B-Instruct** [I4][W1] |
| 권장 우선 전략 | **RAG 먼저 → 필요 시 QLoRA 추가 → 안정화 후 vLLM 패키징** [I4][I5] |
| 배포 원칙 | **온프레미스 / 오프라인 우선 / 단일 NVIDIA A40 x1 / 기존 시뮬레이터 비침습(sidecar)** [I1][I3][W3] |
| Week 1 필수 산출물 | 환경 스크립트, 문서 인덱스, 50문항 평가셋, RAG 결과표, QLoRA 비교표, 실행 리포트 |

## 한 줄 결론

이번 1차 PoC는 **최종 멀티모달 시스템을 만드는 프로젝트가 아니라, 문서 기반 로컬 AI 코어가 실제로 유의미한지 검증하는 프로젝트**로 정의해야 한다. 따라서 가장 낮은 리스크의 경로는 **Qwen2.5-7B-Instruct + BGE-M3 + FAISS 기반 RAG 기준선**을 먼저 완성하고, 그 다음 **300-500건 규모의 QLoRA를 선택적으로 붙여 투자 가치만 판단**하는 것이다. [I4][I5][W1][W3]

## 통합 설계 원칙

1. **PoC의 성공 기준은 “모델을 학습했다”가 아니라 “RAG-only와 RAG+QLoRA의 차이를 의사결정 가능 수준으로 판별했다”이다.** [I4]
2. **기존 RealCOAT/Unity 제품은 바꾸지 않는다.** 이번 주에는 `session_summary.json`, `training_report.json`, `heatmap.png` 같은 산출물만 넘겨받는 **sidecar 통합**으로 제한한다. [I3]
3. **수치 계산은 LLM에 맡기지 않는다.** 작업 속도, 이격거리, 두께, under/over thickness, 순서 위반 등은 시뮬레이터/규칙엔진/평가 모듈이 계산하고, LLM은 이를 설명과 코칭으로 변환한다. [I6]
4. **학습 환경과 서빙 환경을 분리한다.** 단일 A40에서 학습과 서빙을 동시에 돌리지 않으며, Python 환경도 `rag`, `train`, `serve-vllm`로 분리한다. [I5]
5. **최신 문서 업데이트는 RAG가 담당하고, LoRA는 출력 형식·용어·코칭 스타일을 안정화하는 데만 쓴다.** [I4][I6]

\newpage

# 1. 배경 및 왜 이 PRD가 필요한가

과제의 상위 목표는 **비언어적 숙련자 경험을 디지털로 획득·재현하는 멀티모달 XR 기반 AI 도장훈련시뮬레이터**이며, 도장 공정 특화 sLMM 데이터셋, 훈련자 맞춤형 AI Co-pilot, XR 재현, 시험 실증, 사업화가 핵심 축으로 제시되어 있다. 특히 스프레이도장·블라스팅 영역에서 **도막 두께, 도장 순서, 도장 자세, 도막 분포**를 AI Agent와 Co-pilot이 지원하는 그림이 명시되어 있다. [I1][I2]

다만 1차 PoC 단계에서는 멀티모달 센서 학습보다 **웹 크롤링 및 기존 내부 자료로 확보 가능한 문서 중심 지식 적응**이 훨씬 현실적이다. 첨부된 전략 보고서 역시 A40 x1 환경에서 1차 단계의 핵심을 **문서 기반 QA와 RAG/QLoRA 검증**으로 정의하고 있다. [I4]

기존 RealCOAT 환경은 이미 시나리오 관리, 작업 변수 설정, Replay, 히트맵, 성과 레포트, 이동속도·이격거리·분사각도·도막두께 등의 지표 표시 및 교관용 평가 구성을 보유하고 있다. 즉 이번 PoC는 **시뮬레이터 자체를 새로 개발하는 프로젝트가 아니라, 기존 시스템 외부에 문서형 AI 코어를 붙여 해석/질의응답/코칭 레이어를 추가하는 프로젝트**로 정의해야 한다. [I3]

이 PRD는 바로 그 목적을 위해 작성되었다. 이전 PRD의 강점은 **실행 순서와 실험 프레임**, 첨부 PRD/전략 자료의 강점은 **정확한 기술 스택과 버전, 운영 리스크, 모듈형 확장 관점**이다. 본 문서는 둘을 합쳐 **실제로 팀이 Day 1부터 착수할 수 있는 수준의 실행 문서**로 정리한 것이다. [I4][I5][I6]

# 2. 제품 정의

## 2.1 이번 주에 만드는 것

이번 주에 만드는 제품은 아래 세 가지로 제한한다.

1. **문서형 질의응답 AI**
   - 웹/내부 문서에서 선박 도장 관련 절차, 기준, 결함, 용어, 안전 수칙을 근거 기반으로 답변한다.
2. **세션 요약 기반 코칭 AI**
   - 기존 시뮬레이터가 내보낸 작업지표 요약(JSON/CSV)을 받아 훈련 피드백을 생성한다.
3. **평가 하네스**
   - RAG-only, QLoRA-only, RAG+QLoRA를 비교해서 다음 스프린트 의사결정을 할 수 있게 한다.

## 2.2 이번 주에 만들지 않는 것

아래 항목은 범위에서 제외한다.

- 전신 모션/시선/손모션 원시 데이터 학습
- 도막 두께 예측 모델의 신규 학습
- VLM 파인튜닝
- Unity 내 실시간 인퍼런스 임베딩
- 운영용 대시보드/권한 체계/감사 로그 완성
- 최종 다국어 음성 인터페이스

## 2.3 성공 기준

이번 주의 성공 기준은 다음과 같이 고정한다.

| 항목 | 성공 기준 |
|---|---|
| RAG 기준선 | 50문항 기준 평균 **1.5/2.0 이상** 또는 팀이 납득할 수준의 근거성 향상 [I4] |
| hallucination | **10% 이하** [I4] |
| citation coverage | 모든 최종 답변이 **근거 chunk id**를 반환 |
| QLoRA 검증 | 최소 **300개 SFT 샘플**로 1회 이상 학습 후, RAG-only 대비 개선 여부를 수치로 판단 [I4] |
| 재현 가능성 | 다른 연구원이 README/스크립트만 보고 재실행 가능 |
| 일정 준수 | Day 7에 비교 리포트와 실행 산출물이 남을 것 |

추가 판단 규칙은 아래와 같이 둔다.

- **RAG+QLoRA가 RAG-only 대비 평균 점수 0.15 이상 향상**하거나, 용어 정확도/형식 일관성이 눈에 띄게 좋아지고 faithfulness가 악화되지 않으면 adapter를 유지한다.
- 반대로 **faithfulness가 떨어지거나 hallucination이 늘면 adapter는 즉시 폐기**하고 RAG-only를 다음 단계의 기준선으로 채택한다.

# 3. 사용자와 핵심 사용 시나리오

## 3.1 1차 사용자

- **연구원/개발자**: 모델, 데이터, RAG, LoRA 실험을 수행하고 비교 결과를 본다.
- **교관/도메인 검토자**: 50문항 평가셋 채점, 용어/절차 품질 검수, 세션 코칭 문장 검증을 수행한다.
- **기존 시뮬레이터 운영자**: 결과 파일을 AI sidecar에 전달하고 응답 JSON을 확인한다.

## 3.2 대표 사용 시나리오

### 시나리오 A: 문서 QA
- 입력: “Sa 2.5와 SSPC-SP10의 관계를 설명해줘.”
- 처리: RAG 검색 → 상위 chunk 3개 → LLM 응답 → citation 반환
- 출력: 정의, 현장 해석, 근거 chunk id

### 시나리오 B: 세션 코칭
- 입력: `session_summary.json` (평균 이격거리 320mm, under-thickness 8%, C구간 하부 저도막)
- 처리: 규칙/지표는 외부에서 계산됨 → LLM은 원인과 개선 행동으로 변환
- 출력: 즉시 개선 행동 3개, 위험 플래그, 근거 문서 ID

### 시나리오 C: 실험 비교
- 입력: 50문항 평가셋
- 처리: Base zero-shot / RAG-only / QLoRA-only / RAG+QLoRA 비교
- 출력: CSV/Excel/Markdown 결과표, 평균 점수, hallucination 비율, 용어 정확도

\newpage

# 4. 권장 아키텍처

## 4.1 아키텍처 결정

이번 PoC는 **단일 통합 멀티모달 모델**이 아니라 **모듈형 sidecar 아키텍처**로 설계한다. 이유는 단순하다.

- 프로젝트의 장기 목표는 멀티모달 sLMM이지만, 1차 단계에서는 문서형 AI가 핵심이다. [I1][I4]
- 기존 RealCOAT는 이미 리플레이, 히트맵, 자세 확인, 성과 리포트, 작업지표를 제공한다. [I3]
- 수치 기반 평가지표는 이미 구조화가 가능하므로, LLM은 설명과 지식 연결에 집중해야 한다. [I6]
- A40 x1에서는 학습과 서빙을 동시에 돌릴 수는 있어도, **일주일짜리 PoC에서는 그렇게 하지 않는 것이 안정적**이다. [I5][W3]

![권장 아키텍처](ship_painting_optimal_prd_architecture.png)

## 4.2 컴포넌트 구성

| 컴포넌트 | 필수/선택 | 역할 |
|---|---|---|
| Document Ingestor | 필수 | 문서 수집, 파싱, 정규화, dedup, chunk |
| Knowledge Index | 필수 | 임베딩 생성, FAISS/Qdrant 인덱스, 메타데이터 저장 |
| LLM Inference | 필수 | 질의응답 및 코칭 생성 |
| Eval Harness | 필수 | 50문항 평가, 자동/수동 채점, 전략 비교 |
| QLoRA Trainer | 선택 | 300-500 샘플 기반 adapter 학습 |
| vLLM Serve Layer | 선택 | Day 6-7 stretch goal. OpenAI-compatible API 패키징 [I5][W6] |
| VLM Stub | 미래확장 | heatmap/screenshot를 받아 JSON으로 변환하는 인터페이스만 정의 [I6][W2] |
| Motion/Rule Scorer Stub | 미래확장 | 시뮬레이터/규칙엔진이 만든 점수와 피처를 sidecar로 넘길 수 있게 준비 |

## 4.3 통합 원칙

### 원칙 A. 기존 제품 비침습
기존 RealCOAT/Unity는 파일 export 또는 내부 REST만 추가한다. 엔진 내부에 모델을 내장하지 않는다. [I3]

### 원칙 B. 계산과 설명 분리
- **시뮬레이터/평가 모듈**: 평균 속도, 이격거리, 각도, DFT, under/over thickness, 시퀀스 위반 계산
- **LLM**: 원인 설명, 개선 조치, 문서 근거 연결

### 원칙 C. 답변 스키마 고정
모든 답변은 텍스트가 아니라 **JSON schema**를 기본으로 한다. UI 연결과 재현성이 좋아진다.

# 5. 기술 선택과 의사결정

## 5.1 기본 모델

### 최종 권장: Qwen2.5-7B-Instruct
선정 이유는 아래와 같다.

- Apache-2.0 라이선스로 공개되어 있다. [W1]
- 7.61B 파라미터, 최대 131,072 토큰 컨텍스트, 29개 이상 언어 지원, JSON 등 구조화 출력 및 테이블/긴 텍스트 처리 강점을 모델 카드가 직접 강조한다. [W1]
- 첨부 전략 보고서도 1차 PoC 기본 모델로 Qwen2.5-7B-Instruct를 1순위로 제안한다. [I4]
- A40 48GB에서 4bit inference와 QLoRA가 현실적이다. [I4][W3][W4]

### 왜 3B가 아닌가
3B급 모델은 실험 회전 속도는 빠르지만, 한국어/영어 혼합 기술 문서와 절차 설명의 안정성이 7B 대비 떨어질 가능성이 높다. 이번 PoC는 파이프라인 smoke test가 아니라 **실제 연구 의사결정**이 목적이므로 7B를 기준선으로 잡는 편이 낫다. [I4]

### 왜 14B를 지금 안 하는가
14B는 성능 여유가 있지만, Week 1에서는 학습/서빙 복잡도만 늘어난다. 이번 단계는 **RAG와 QLoRA의 효과를 판별하는 것**이므로 7B가 sweet spot이다. [I4]

## 5.2 RAG vs QLoRA 전략

### 최종 권고
- **필수**: RAG-only
- **선택**: Light QLoRA
- **배제**: LoRA-only 단독 제품화

### 이유
- 최신 문서, 규정, 허용 범위, 안전 기준은 **RAG가 가장 유지보수에 유리**하다. [I4][I6]
- QLoRA는 문서 지식 전체를 암기시키는 수단이 아니라, **용어 이해, 출력 형식, 코칭 톤, retrieval context 활용 습관**을 보정하는 수단으로만 쓴다. [I4][I6]
- 문서형 산업 AI에서 답변의 감사 가능성과 출처 추적성은 매우 중요하므로, RAG를 버리고 LoRA-only로 가면 오히려 리스크가 커진다. [I6]

## 5.3 임베딩과 리랭커

### 임베딩: BAAI/bge-m3
- 1024차원
- 최대 8192 토큰
- dense + sparse + multi-vector 계열을 포괄하는 다기능 모델
- query prefix가 필수적이지 않아 RAG 파이프라인이 단순하다 [W10]

### 대안: multilingual-e5-large
- 1024차원
- 최대 512 토큰
- 안정적인 대안이지만 긴 한국어 문서 처리에서는 BGE-M3보다 불리하다 [W10]

### 리랭커: BAAI/bge-reranker-v2-m3
- query와 passage를 함께 넣고 relevance score를 직접 출력하는 cross-encoder 계열이다. [W11]
- Week 1에서는 Top-5 retrieval 뒤 Top-3 rerank 정도로만 쓴다.
- 설치/성능 문제가 있으면 Day 3에는 리랭커를 끄고 retrieval-only로 먼저 기준선을 만든다.

## 5.4 벡터 저장소

### Week 1 기본: FAISS CPU
이유는 매우 단순하다.

- 50-100문서 규모에서는 GPU 벡터 검색이 필요하지 않다.
- 단일 A40은 이미 LLM과 임베딩/리랭킹에 사용되므로, Week 1에서는 CUDA 복잡도를 줄이는 편이 낫다.
- FAISS CPU는 설치와 디버깅이 쉽다.

### 옵션 A: FAISS GPU
필요 시 `faiss-gpu-cu12`를 사용할 수 있다. PyPI 기준 최신 릴리스는 1.13.2이고, Ampere(Compute Capability 8.6)는 지원 범위 안에 있다. [W12][W3]

### 옵션 B: Qdrant
메타데이터 payload, 필터링, hybrid retrieval를 빨리 붙여야 하면 Qdrant를 쓴다. Qdrant는 payload 저장과 필터링, dense/sparse/hybrid 검색을 지원하며, `qdrant-client`는 local mode도 제공한다. [W15]

## 5.5 서빙 엔진

### Week 1 기본: Transformers 직접 로딩
- 장점: 디버깅이 쉽고 환경이 단순하다.
- 적합 시점: Day 1-5의 실험/평가 단계

### Stretch Goal: vLLM 0.16.0
- OpenAI-compatible server 제공 [W6]
- `vllm serve`로 빠르게 API 패키징 가능 [W6]
- 단, Unsloth 학습 환경과는 분리하는 것이 안전하다. [I5]

## 5.6 QLoRA 학습 엔진

### Fast path: Unsloth
- PyPI 기준 최신 릴리스는 2026.2.1이다. [W7]
- 메모리 절감과 빠른 학습이 장점이며, Qwen 계열 튜닝 사례가 풍부하다. [I4][W7]

### Conservative path: Week 1은 RAG-only로 마감
- Day 5 오전까지 Unsloth smoke test가 실패하면 QLoRA는 이번 스프린트 핵심이 아니다.
- 이 경우 “LoRA 효과 미검증”으로 기록하고 RAG-only 결과를 1차 결론으로 낸다.

\newpage

# 6. 처음부터 시작하는 환경 설정

## 6.1 권장 운영 구조

| 장비 | 역할 | OS |
|---|---|---|
| AI 서버 | 문서 파싱, 인덱싱, 평가, QLoRA, LLM 추론 | Ubuntu 22.04 또는 24.04 |
| 기존 시뮬레이터 PC | RealCOAT/Unity 실행, 결과 export | Windows 11 |

**권장 구조는 2대 분리**다. 같은 장비에서 모두 하더라도, Week 1 기준으로는 Windows 위에 WSL2로 모든 것을 우겨 넣기보다 AI 전용 Linux가 훨씬 안정적이다.

## 6.2 필수 사전 확인

A40 공식 사양은 48GB GDDR6 ECC, 696 GB/s 메모리 대역폭, 300W 전력이다. [W3]

```bash
nvidia-smi
python3 --version
nvcc --version || true
free -h
df -h
```

권장 최소 조건:

- NVIDIA A40 x1
- Ubuntu 22.04/24.04
- Python 3.10
- 디스크 여유: 200GB+
- 외부망 1회 접속 가능(모델/휠 사전 다운로드용)

## 6.3 프로젝트 디렉터리

```bash
mkdir -p ~/shipcoat-poc/{configs,scripts,notebooks,logs}
mkdir -p ~/shipcoat-poc/data/{raw,parsed,cleaned,chunks,sft,eval,exports}
mkdir -p ~/shipcoat-poc/artifacts/{models,indexes,embeddings,adapters,reports}
mkdir -p ~/shipcoat-poc/cache/{hf,wheels}
cd ~/shipcoat-poc
```

## 6.4 환경 분리 전략

이번 PRD의 핵심은 **세 개의 virtualenv를 분리**하는 것이다.

| 환경 | 필수/선택 | 목적 |
|---|---|---|
| `.venv-rag` | 필수 | 문서 파싱, 임베딩, 인덱싱, 평가, baseline inference |
| `.venv-train` | 선택 | Unsloth QLoRA 학습 |
| `.venv-vllm` | 선택 | vLLM API 서빙 패키징 |

## 6.5 `.venv-rag` 생성

```bash
python3 -m venv .venv-rag
source .venv-rag/bin/activate
pip install --upgrade pip uv
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install "transformers==5.1.0" "accelerate==1.13.0" "bitsandbytes==0.49.1"
pip install "llama-index-core==0.14.14"
pip install "llama-index-embeddings-huggingface==0.6.1"
pip install "llama-index-vector-stores-faiss==0.5.3"
pip install "llama-index-postprocessor-flag-embedding-reranker==0.4.1"
pip install "sentence-transformers==5.2.3" "FlagEmbedding==1.3.5"
pip install faiss-cpu fastapi uvicorn pymupdf beautifulsoup4 lxml markdownify datasketch pandas openpyxl rouge-score bert-score scikit-learn python-dotenv pyyaml tqdm jinja2
```

위 버전은 2026-03-09 기준 PyPI/공식 문서 기준으로 검증해 반영했다. [W6][W8][W9][W10][W11][W12][W13][W14]

### `.venv-rag` smoke test

```bash
python - <<'PY'
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no-gpu')
PY
```

## 6.6 `.venv-train` 생성 (선택)

```bash
python3 -m venv .venv-train
source .venv-train/bin/activate
pip install --upgrade pip uv
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install "unsloth==2026.2.1"
pip install datasets pandas pymupdf beautifulsoup4 datasketch openpyxl
```

### `.venv-train` smoke test

```bash
python - <<'PY'
from unsloth import FastLanguageModel
print('unsloth import ok')
PY
```

> 주의: Unsloth는 별도 PyTorch/CUDA 조합과 얽힐 수 있으므로, `venv-rag`와 절대 합치지 않는다. `pad_token` 관련 문제를 피하려면 가능하면 Unsloth가 제공하는 사전 4bit 모델을 쓰고, 문제가 생기면 즉시 이번 주 LoRA 범위를 축소한다. [I5]

## 6.7 `.venv-vllm` 생성 (선택)

```bash
python3 -m venv .venv-vllm
source .venv-vllm/bin/activate
pip install --upgrade pip uv
pip install "vllm==0.16.0"
pip install fastapi uvicorn openai
```

### `.venv-vllm` smoke test

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --dtype auto --api-key test-key
```

vLLM은 OpenAI-compatible server를 제공하므로, 나중에 `OpenAI(base_url=..., api_key=...)` 형태로 바로 붙일 수 있다. [W6]

## 6.8 오프라인 사전 다운로드

Hugging Face Hub는 `snapshot_download()`와 `local_dir` 방식, 그리고 오프라인 환경 변수(`HF_HUB_OFFLINE`)를 제공한다. [W5]

```bash
source .venv-rag/bin/activate
python - <<'PY'
from huggingface_hub import snapshot_download
models = [
    ('Qwen/Qwen2.5-7B-Instruct', './cache/hf/Qwen2.5-7B-Instruct'),
    ('BAAI/bge-m3', './cache/hf/bge-m3'),
    ('BAAI/bge-reranker-v2-m3', './cache/hf/bge-reranker-v2-m3'),
]
for repo, path in models:
    snapshot_download(repo_id=repo, local_dir=path)
PY
```

```bash
export HF_HOME=$PWD/cache/hf
export HF_HUB_CACHE=$PWD/cache/hf
export TRANSFORMERS_CACHE=$PWD/cache/hf
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
```

## 6.9 선택적 GPU FAISS

Week 1에서 retrieval 규모가 작다면 `faiss-cpu`를 기본으로 유지한다. GPU 버전이 꼭 필요하면 아래를 추가한다.

```bash
source .venv-rag/bin/activate
pip uninstall -y faiss-cpu || true
pip install "faiss-gpu-cu12==1.13.2"
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

첨부 자료의 핵심 주장은 데이터 저장소를 **RAG 코퍼스 / LLM SFT / 예측모델 / VLM**로 나누라는 것이다. 이번 주는 그중 세 가지만 쓴다. [I6]

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

기존 RealCOAT가 제공하는 지표 체계를 반영해 아래와 같이 정의한다. RealCOAT 자료에는 Replay, 히트맵, 도막두께, 평균 이격거리/속도/각도, 도료 사용률, 작업 속도, 레포트 형식 출력이 명시되어 있다. [I3]

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

## 7.7 AI 응답 스키마

```json
{
  "session_id": "SPRAY-DEMO-0001",
  "summary": "C구간 하부 저도막과 이격거리 편차가 관찰됨",
  "action_items": [
    "이격거리를 250-300mm 범위로 더 일정하게 유지",
    "하부 구간은 이동 속도를 약간 낮추고 50% 오버랩 유지",
    "재도장 전 해당 구간 DFT 재측정"
  ],
  "risk_flags": ["under_thickness"],
  "citations": ["DOC-COAT-0007-C012", "DOC-TRAIN-0003-C004"],
  "confidence": "medium"
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
| temperature | 0.1 | 평가 안정성 우선 [I4] |
| max output tokens | 768 | QA 기준 |
| citation 반환 | 필수 | chunk id 리스트 |
| 답변 형식 | JSON 우선 | UI 연동 대비 |

## 8.3 추천 프롬프트 골격

```text
System:
너는 선박 도장 훈련 및 도장 공정 문서 해석을 지원하는 산업 AI 어시스턴트다.
항상 제공된 근거 문서에 기반해 답하고, 근거가 부족하면 모른다고 말한다.
수치 계산은 입력값을 그대로 사용하고 임의로 만들지 않는다.
최종 출력은 JSON으로 작성한다.

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
| 권장 PoC | 500-800 | 카테고리 균형 잡힌 1차 도메인 적응 [I4] |
| 차기 단계 | 2000+ | 프로젝트 목표 대응용 확장 [I4] |

## 9.4 카테고리 분포

| 카테고리 | 목표수 |
|---|---:|
| 문서 QA | 100 |
| 절차 설명 | 80 |
| 결함 분석 | 60 |
| DFT/사양 피드백 | 60 |
| 용어/다국어 | 50 |
| 요약/비교 | 50 |

## 9.5 학습 포맷

```json
{
  "messages": [
    {
      "role": "system",
      "content": "너는 선박 도장 훈련 코치다. 항상 안전과 표준 절차를 우선하고, 마지막에 근거 문서 ID를 적는다."
    },
    {
      "role": "user",
      "content": "[문서]\n에폭시 프라이머의 적정 DFT는 125-200µm이며 과도한 도포는 균열 위험을 높인다.\n[질문]\n230µm로 도포하면 어떤 문제가 생길 수 있나?"
    },
    {
      "role": "assistant",
      "content": "230µm는 권장 범위를 초과하므로 과도막 위험이 있다. 균열과 건조 지연, 재작업 가능성을 설명하고 추가 검사 필요성을 안내한다.\n근거: DOC-COAT-0007-C012"
    }
  ],
  "category": "coating_thickness"
}
```

## 9.6 권장 하이퍼파라미터

전략 보고서와 첨부 기술 스택 자료를 종합하면 A40 x1에서 현실적인 기본값은 아래다. [I4][I5]

| 항목 | 권장값 |
|---|---|
| Base model | Qwen2.5-7B-Instruct |
| Quantization | 4-bit NF4 + bf16 compute [W4] |
| LoRA target | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj [I4] |
| rank | 16 |
| alpha | 32 |
| dropout | 0.05 |
| max seq len | 2048 또는 4096 |
| batch | 1-4 |
| grad accum | 4-16 |
| epoch | 3 |
| optimizer | adamw_8bit |

## 9.7 운영상 최적값

기술 스택 참고문서는 A40에서 `batch 4 x seq 4096 x grad_accum 4`를 sweet spot으로 제안한다. 다만 Week 1에서는 안정성이 더 중요하므로 아래 순서로 올린다. [I5]

1. **안전 시작값**: batch 1, seq 2048, grad accum 8
2. **안정 시 상향**: batch 2, seq 2048, grad accum 8
3. **충분히 안정 시**: batch 4, seq 4096, grad accum 4

# 10. 7일 실행 계획

## 10.1 전체 일정

| Day | 필수 작업 | 산출물 | Gate |
|---|---|---|---|
| Day 1 | 환경 구축, 모델 사전 다운로드, smoke test | env log, requirements, download manifest | `.venv-rag` 정상동작 |
| Day 2 | 문서 수집/파싱/정제/중복 제거 | `manifest.csv`, `cleaned/*.json` | 50-100건 정제 완료 |
| Day 3 | chunk/index/RAG baseline 구현 | `chunks/*.jsonl`, `faiss.index` | 10문항 smoke test 통과 |
| Day 4 | 50문항 평가셋 완성, RAG-only 평가 | `eval_set.jsonl`, `rag_only_results.csv` | 평균/오류 패턴 확인 |
| Day 5 | SFT 데이터 300-500건 생성, Unsloth smoke test, 1회 학습 | `sft_train.jsonl`, `adapter/` | adapter 또는 실패 원인 기록 |
| Day 6 | RAG+QLoRA 비교, stretch로 vLLM API 패키징 | `comparison.csv`, `demo_api.py` | keep/drop decision |
| Day 7 | 최종 리포트, README, 재실행 스크립트 정리 | `final_report.md`, artifacts manifest | 의사결정 문서 완성 |

## 10.2 Day별 상세 작업

### Day 1 — 환경 확정
- OS/GPU 점검
- `.venv-rag`, `.venv-train` 생성
- 모델 3종 다운로드(Qwen, BGE-M3, reranker)
- 4bit inference smoke test
- 실패 시 즉시 `RAG-only + CPU FAISS` 경로 확정

### Day 2 — 데이터 파이프라인 가동
- raw 수집
- parse/clean/dedup
- metadata schema 확정
- 고품질 문서 20건 선별

### Day 3 — RAG baseline 완성
- BGE-M3 임베딩
- FAISS 인덱스 구축
- retrieval top-k 및 rerank on/off 비교
- 응답 JSON schema 확정

### Day 4 — 평가 기준선 확보
- 50문항 평가셋 확정
- RAG-only 일괄 실행
- 수동 채점 시작
- hallucination 유형 분류

### Day 5 — QLoRA 검증
- 300-500 SFT 샘플 생성
- 품질 검수(최소 20-30%)
- 1회 학습
- QLoRA-only 간이 평가

### Day 6 — 전략 비교
- RAG-only vs QLoRA-only vs RAG+QLoRA 비교
- adapter keep/drop 의사결정
- 시간이 남으면 vLLM API 포장

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

전략 보고서의 5차원 0-2점 체계를 그대로 유지한다. [I4]

| 항목 | 2점 | 1점 | 0점 |
|---|---|---|---|
| 기술 정확성 | 기술적으로 정확 | 대체로 정확 | 명백한 오류 |
| 문서 충실도 | 근거 문서에 충실 | 일부 추론 과다 | hallucination |
| 용어 적절성 | 현장 용어 사용 정확 | 일부 흔들림 | 오용 |
| 완전성 | 필요한 정보 모두 포함 | 일부 누락 | 핵심 누락 |
| 한국어 품질 | 자연스럽고 전문적 | 어색하지만 이해 가능 | 읽기 어려움 |

## 11.3 자동 평가

첨부 보고서는 ROUGE-L, BERTScore, hallucination 비율, 용어 정확도, 응답 길이 적절성을 권장한다. [I4]

| 지표 | 도구 |
|---|---|
| ROUGE-L | rouge-score |
| BERTScore | bert-score |
| hallucination ratio | 수동 + LLM-as-Judge |
| terminology accuracy | 용어 사전 대조 |
| response length | 토큰 통계 |

## 11.4 프로젝트 KPI와의 연결

상위 프로젝트 KPI는 2-3차년도 멀티모달 확장과 연결되므로, 이번 주에는 아래 형태로만 대응한다. [I4]

| 상위 KPI | 이번 PoC에서 확인할 것 |
|---|---|
| sLMM 데이터셋 2000건 이상 | 데이터 파이프라인 확장성 |
| 자세 유사도 교정 정확도 ≥70% | 텍스트 기반 자세 코칭 포맷 가능성 |
| 도막 분포 이상 탐지 ≥70% | DFT/분포 관련 QA 정확성 |
| AI Co-Pilot 기능 8개 이상 | 시나리오별 응답 품질과 JSON 스키마 |
| E-7 대상국 번역 지원 | 다국어 경고/가이드 샘플 생성 |

# 12. API / 파일 인터페이스 정의

## 12.1 파일 기반 경로(권장)

```text
exports/
  session_summary/
    SPRAY-DEMO-0001.json
  training_report/
    SPRAY-DEMO-0001.json
  heatmap/
    SPRAY-DEMO-0001.png
  questions/
    ask_0001.json
  answers/
    ask_0001.answer.json
```

## 12.2 내부 REST 경로(선택)

| Method | Path | 설명 |
|---|---|---|
| POST | `/qa` | 문서 기반 QA |
| POST | `/coach/session` | 세션 요약 기반 코칭 |
| POST | `/eval/run` | 평가셋 일괄 실행 |
| GET | `/health` | 상태 확인 |

## 12.3 응답 규칙

- 모든 응답은 `answer`, `citations`, `confidence`, `model_name`, `timestamp`를 포함한다.
- 코칭 응답은 `action_items`, `risk_flags`를 추가한다.
- 문서 근거가 약하면 `confidence="low"`로 반환한다.

# 13. 리스크와 폴백

| 리스크 | 징후 | 폴백 |
|---|---|---|
| Unsloth 설치 실패 | import 오류, CUDA mismatch | 이번 주 LoRA 생략, RAG-only로 결론 |
| GPU OOM | 4bit 로딩/학습 중 메모리 초과 | seq 2048, batch 1로 축소 |
| 문서 파싱 품질 저하 | PDF 표 깨짐, 내용 손실 | 표는 별도 CSV 저장, 본문 RAG 우선 |
| reranker 속도 저하 | Day 3 retrieval 느림 | reranker 끄고 retrieval-only baseline 확보 |
| vLLM 환경 충돌 | serve env import 실패 | Day 6 stretch goal 취소, FastAPI+Transformers 유지 |
| 평가셋 편향 | 특정 카테고리 과다 | 5카테고리 균형 유지 |
| hallucination 증가 | RAG+QLoRA에서 출처 없는 답변 증가 | adapter 폐기 |

# 14. 최종 산출물 목록

Day 7까지 아래 파일이 남아 있어야 한다.

```text
shipcoat-poc/
  README.md
  configs/
    rag_config.yaml
    qlora_config.yaml
  scripts/
    ingest_docs.py
    build_index.py
    run_rag_eval.py
    train_qlora.py
    serve_demo_api.py
  data/
    manifest.csv
    eval/eval_set.jsonl
    sft/sft_train.jsonl
  artifacts/
    indexes/faiss.index
    reports/rag_only_results.csv
    reports/comparison.csv
    adapters/<optional>
  docs/
    final_prd.docx
    final_report.md
```

# 15. 다음 단계 제안

이번 PoC가 성공하면 다음 단계는 아래 순서가 가장 자연스럽다.

1. **세션 요약 자동 export 고도화**
2. **heatmap/screenshot 기반 VLM stub 연결**
3. **도장 자세/시퀀스/두께 분포 스코어러 분리**
4. **RAG + scoring + LLM orchestration 통합**
5. **다국어 음성/UI 확장**

Qwen2-VL-2B-Instruct는 향후 스크린샷/리포트 패널 해석용 후보로 적합하다. 모델 카드는 이미지 내 다국어 텍스트 이해, dynamic resolution, 문서·OCR 성능을 강조한다. 다만 Week 1 범위에는 넣지 않는다. [W2]

# 16. Co-Pilot 아키텍처 방향 (2차 PoC 로드맵)

## 16.1 문제 인식

1차 PoC는 **문서 기반 QA + QLoRA 검증**에 성공했으나, 최종 제품 목표인 **VLM 기반 AI Co-Pilot**과 현재 구현 사이에 구조적 괴리가 존재한다.

| 항목 | 1차 PoC (현재) | 최종 목표 (Co-Pilot) |
|---|---|---|
| 입력 | 텍스트 질문 | 센서 데이터, 자세, DFT 측정값, 시뮬레이터 상태, 이미지 |
| 추론 | 문서 검색 → 답변 생성 | Tool Calling → MCP Server → Chain of Thought → 의사결정 |
| 출력 | JSON 답변 + citation | XR 콘텐츠용 텍스트, 행동 지시, 시각화 데이터, 경고 |
| 평가 | 용어/규정 조회 (50문항) | 이상 탐지, 순서 판단, 조건부 의사결정, 도구 사용 |
| 아키텍처 | 단일 RAG 파이프라인 | Agent + Tool Calling + MCP orchestration |

## 16.2 Co-Pilot 목표 아키텍처

### LLM 모델 선정 (2026-03 결정)

| 후보 | 결정 | 사유 |
|---|---|---|
| Qwen2.5-7B-Instruct | 1차 PoC 완료, 마이그레이션 대상 | Tool Calling/CoT는 프롬프트 방식만 지원 |
| **Qwen3-4B** | **1차 마이그레이션 대상** | Apache 2.0, Thinking Mode 내장, VRAM 효율 (A40에서 36GB 여유), 파인튜닝 응답성 #1, Qwen2.5-7B를 13/15 벤치마크에서 초과 |
| **Qwen3-8B** | **A/B 비교 대상** | Tool Calling BFCL ~85%, Thinking Mode, 한국어 더 강력 |
| EXAONE 3.5 7.8B | ❌ 제외 | NC 라이선스 (조선소 상용 배포 불가), Native Tool Calling 미지원, trust_remote_code 필요 |
| Gemma 3 12B | 차순위 | Gemma ToS 라이선스 제한, Tool Calling 벤치마크 미공개 |

**Qwen3 핵심 이점**: Native Thinking Mode (`<think>` 블록)로 CoT 추론이 모델 내장. 복잡한 요청 시 `/think`, 단순 조회 시 `/no_think`로 턴별 전환 가능. MCP 공식 지원 (Qwen-Agent 프레임워크).

```text
┌─────────────────────────────────────────────────────┐
│                  XR Client (Unity)                   │
│         AR/VR 콘텐츠 렌더링 + 사용자 인터페이스           │
└─────────────┬───────────────────────┬───────────────┘
              │ instruction           │ sensor data
              ▼                       ▼
┌─────────────────────────────────────────────────────┐
│         AI Co-Pilot Agent (Qwen3-4B/8B)              │
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
      ▲      ▲      ▲      ▲      ▲          ▲
      │      │      │      │      │          │
┌─────────────────────────────────────────────────────┐
│         시뮬레이터 / 센서 / 측정 장비                    │
│  RealCOAT | DFT gauge | IMU | Camera | 환경 센서     │
└─────────────────────────────────────────────────────┘
```

## 16.3 MCP Server 인터페이스 (설계 대상)

학습 데이터셋 구조(`episode_id`, `worker_pose(BVH)`, `trajectory(x,y,z,roll,pitch,yaw)`, `heatmap`, `lang_instruction` 등)를 기반으로 각 MCP 서버의 역할을 정의한다.

| MCP Server | 입력 | 출력 | 용도 | 데이터 출처 |
|---|---|---|---|---|
| `pose_estimation` | current_pose(6DOF, BVH), reference_pose(숙련자 BVH) | 관절별 편차 벡터, 교정 방향 지시 | 숙련자 자세 대비 현재 작업자 자세 차이 분석 및 교정 | dataset `작업행동(worker_pose)` 숙련자 reference |
| `anomaly_detection` | heatmap 이미지, DFT 측정값, 환경(온도/습도) | 이상 유형(저도막/과도막/결함), 위치, 심각도 | 도막 이상상태 탐지 — LLaSA 등 멀티모달 이미지+정량 데이터 복합 분석 | dataset `결과(heatmap)`, `장비상태` |
| `timeseries` | 세션 내 속도/거리/각도 시계열 누적 | 추세, stroke 간 편차, 피로도 패턴 | Agent 루프의 순환 구조용 세션 state 갱신 — 매 턴 업데이트되어 context 축적 | dataset `작업시간`, `장비궤적` 시계열 |
| `location_tracking` | 장비궤적(x,y,z) + 선박 3D 모델 메타데이터 | 현재 구역 ID/특성, 미도장 구역, 커버리지 % | 선박 3D 공간 내 현재 칠하는 구역 인식 — pose/sequence MCP에 공간 context 공급 | dataset `장비궤적`, PaintNet 기반 trajectory simulation |
| `sequence_validation` | current_trajectory, location(구역 ID), reference_sequence(숙련자) | stroke 방향 위반 여부, 권장 다음 stroke 방향 | 현재 구역 내 도장 순서 검증 (위→아래, 아래→위 등 stroke 단위) — 공정 단계 순서가 아닌 구역 내 붓질 순서 | dataset `작업시퀀스(stroke 단위)`, `언어명령` |
| `dft_judge` | DFT 측정값 배열, 사양 기준 | 합격/불합격, 90/10 룰 위반 포인트 목록 | 도막 두께 규격 합부 판정 — anomaly_detection과 분리하여 규제/리포트용 명시적 pass/fail 출력 | dataset `작업결과(도막상태, 스코어)` |

> **anomaly_detection 구현 방향**: 두 가지 파이프라인 후보가 존재한다.
>
> - **옵션 A (VL 앞단 처리)**: Qwen VL 모델이 히트맵 이미지를 먼저 분석하여 `image_context` JSON(결함 위치, 색상 분포 설명 등)을 생성 → `anomaly_detection` MCP는 이 텍스트 context + DFT 수치만 입력받아 판정. 이 경우 anomaly_detection은 순수 텍스트+수치 서버가 되어 구현이 단순해진다.
> - **옵션 B (단일 멀티모달)**: LLaSA 계열 등 멀티모달 이상탐지 모델이 이미지+수치를 직접 처리. 모델 선정 및 fine-tuning 필요.
>
> **현재 방향**: 옵션 A 우선 검토 (Qwen VL이 이미지 context 분석 담당, anomaly_detection은 결과를 받아 판정). Qwen VL 웨이트는 미다운로드 상태 — Phase 4 시점에 `Qwen2.5-VL` 또는 `Qwen3-VL` 계열 다운로드 예정. Phase 2 Stub은 image_context를 fixture 텍스트로 대체하여 인터페이스만 확정.
>
> **timeseries 역할**: 단발성 호출이 아니라 ReAct Agent 루프가 돌면서 매 턴 누적 업데이트되는 session state 역할. "3번째 stroke부터 속도가 점점 빨라지고 있다" 같은 트렌드를 잡아 다음 Tool Call 판단에 영향.
>
> **location_tracking 구현 의존성**: RealCOAT/Unity의 3D 좌표 export 인터페이스 및 PaintNet trajectory simulation 데이터에 의존. Phase 2 Stub은 fixture 반환으로 구현하고, 실제 연동은 Phase 4~5에서 수행.

## 16.4 Co-Pilot 대표 시나리오

### 시나리오 D: 실시간 코칭 (Tool Calling + CoT)
```
[사용자] "지금 내 도장 상태 어때?"
[Agent]
  1. location_tracking(trajectory) → {zone: "B-3", type: "수직 패널", coverage: 0.62}
  2. pose_estimation(current_pose, reference_pose=숙련자_BVH)
     → {deviation: {wrist: 8°, elbow: 5°}, correction: "스프레이건 8° 아래로 조정"}
  3. sequence_validation(trajectory, zone="B-3", reference=숙련자_stroke_순서)
     → {violation: true, current: "아래→위", recommended: "위→아래", reason: "수직 패널 중력 흘림 방지"}
  4. timeseries.update(speed_history) → {trend: "속도 18% 상승", last_5min: [0.38, 0.41, 0.45, 0.48, 0.52]}
  5. RAG 검색: 수직 패널 도장 순서 및 속도 기준
  6. CoT: 자세 편차 + stroke 순서 위반 + 속도 상승 → 저도막 인과관계 추론
[응답] "B-3 수직 패널에서 복합 문제가 감지됩니다.
        ① 스프레이건 각도 8° 과다 → 건 수직 유지
        ② 아래→위 stroke 비권장 → 위→아래로 전환 (흘림 방지)
        ③ 속도 점진 상승 중 → 0.4m/s 이하 유지
        이 세 가지가 복합되면 좌하단 저도막으로 이어질 수 있습니다."
```

### 시나리오 E: 이상 탐지 + 의사결정 (멀티모달 ReAct)
```
[사용자] "도막 상태에 이상이 있나?"
[Agent]
  1. [VL 앞단] Qwen VL(heatmap_image) → image_context: "좌하단 B-3-L 구역 청색 집중, 전체 대비 저도막 패턴"
  2. anomaly_detection(image_context=위결과, dft=[150,85,92,110,78], env={temp:28, humidity:65})
     — VL이 이미지 분석 담당, anomaly_detection은 텍스트 context + 수치로 판정
     → {anomaly: true, type: "under_thickness", location: "좌하단 B-3-L", severity: "high",
        image_evidence: "히트맵 청색 집중 구역 확인", likely_cause: "excessive_speed+angle"}
  2. dft_judge.evaluate(measurements=[150,85,92,110,78], spec={nominal:100, tolerance:"90/10"})
     → {pass: false, violations: [{point:2, value:85}, {point:5, value:78}]}
  3. timeseries.get_context() → {speed_at_violation_zone: 0.58, avg_session: 0.41}
  4. RAG 검색: 저도막 원인 및 재도장 절차
  5. CoT: 이미지 근거 + 수치 위반 + 속도 상관관계 → 교정 행동 결정
[응답] "히트맵 이미지와 DFT 측정값 모두에서 B-3 좌하단 저도막 확인.
        포인트 2(85µm), 5(78µm) — 90/10 룰 위반.
        원인: 해당 구간 통과 속도 0.58m/s (세션 평균 대비 41% 초과).
        교정: ① 해당 구역 재도장 ② 이동 속도 0.35m/s 이하로 조절"
```

### 시나리오 F: XR 콘텐츠 생성
```
[사용자] "웰드 부위 스트라이프 코팅 방법 보여줘"
[Agent]
  1. RAG 검색: 스트라이프 코팅 절차, 웰드 부위 도장 규정
  2. CoT: 문서 절차 → XR 표시 가능 형태로 변환
[응답] {
  "xr_type": "ar_instruction",
  "steps": [
    {"step": 1, "action": "웰드 양쪽 50mm 범위 마스킹", "highlight_zone": "weld±50mm"},
    {"step": 2, "action": "붓 도장으로 스트라이프 코팅", "tool_angle": 45, "direction": "along_weld"},
    {"step": 3, "action": "DFT 확인 (기준: 150µm 이상)", "measurement_points": 3}
  ],
  "citations": ["DOC-COAT-0012-C003"]
}
```

## 16.5 단계별 로드맵

| Phase | 기간 | 핵심 산출물 | 의존성 |
|---|---|---|---|
| **Phase 1** (완료) | Week 1 | RAG+QLoRA 문서 QA 기준선, 3-way 비교, 평가 하네스 | - |
| **Phase 2** | Week 2-3 | Tool Calling 프레임워크, MCP Server stub, Agentic 평가셋 | Phase 1 |
| **Phase 3** | Week 3-4 | Session Coaching + CoT, DFT 판정 도구 통합, 실시간 코칭 시연 | Phase 2 |
| **Phase 4** | Week 4-6 | VLM stub (heatmap/screenshot 해석), XR 콘텐츠 생성 포맷 | Phase 3 |
| **Phase 5** | Week 6-8 | 전체 Agent orchestration, 시뮬레이터 연동, 다국어 확장 | Phase 4 |

### Phase 2 상세 (Week 2-3, 즉시 착수)

| Task | 설명 | 산출물 |
|---|---|---|
| Tool Calling 인터페이스 설계 | JSON Schema 기반 tool 정의, 호출 규약 | `tool_schemas/*.json` |
| MCP Server stub 구현 | 6개 서버의 mock 구현, 입출력 검증 | `mcp_servers/` |
| Agentic 평가셋 설계 | Co-Pilot형 30문항 (도구 사용, 이상 탐지, 순서 판단, CoT) | `eval_set_agentic.jsonl` |
| Session Coaching 재설계 | session_summary.json → Tool Calling → CoT → 코칭 | `session_coach.py` |
| CoT 평가 루브릭 | 추론 과정 품질 평가 기준 | `eval_rubric_agentic.md` |

## 16.6 평가셋 확장 방향

기존 50문항(문서 QA)에 추가로 **Agentic 평가셋 30문항**을 설계한다.

| 카테고리 | 수량 | 테스트 능력 | 예시 |
|---|---|---|---|
| Tool Calling | 6 | 올바른 도구 선택 + 파라미터 구성 | "현재 DFT가 [120, 85, 95, 110]이고 기준이 100µm일 때 합부 판정해줘" |
| 이상 탐지 해석 | 6 | 센서 데이터 → 원인 분석 → 조치 | "작업 속도가 급격히 변하는 구간의 도막 품질 문제를 분석해줘" |
| 순서/의사결정 | 6 | 조건부 분기, 우선순위 판단 | "습도 92%, 표면온도가 이슬점 이하일 때 다음 행동은?" |
| CoT 추론 | 6 | 다단계 추론 + 근거 연결 | "블라스팅→프라이머→상도 과정에서 각 단계별 대기 시간 조건은?" |
| XR 콘텐츠 생성 | 6 | 구조화된 지시문 생성 | "밀폐 공간 환기 절차를 AR 가이드 형식으로 생성해줘" |

## 16.7 아키텍처 문서(AgenticCopilot_Architecture.md) 정합성 매핑

본 절은 `AgenticCopilot_Architecture.md`의 설계 개념과 실제 구현 간 매핑을 정리한다.

### MCP 모듈 매핑

| 아키텍처 원본 | PRD 구현 | 실제 역할 (재정의) |
|---|---|---|
| MCP 자세 추정 | `pose_estimation` | 숙련자 BVH reference 대비 현재 작업자 6DOF 자세 편차 분석 및 교정 벡터 출력 |
| MCP 이상탐지 | `anomaly_detection` | Qwen VL이 히트맵 이미지를 앞단에서 분석 → image_context JSON 생성 → anomaly_detection은 텍스트 context + DFT 수치로 판정. 단순 임계값 탐지가 아닌 이미지 근거 포함 |
| MCP 시계열 처리 | `timeseries` | ReAct 루프 내 세션 state 갱신용. 매 턴 누적되어 속도/거리 트렌드를 다음 Tool Call 판단에 제공 |
| MCP 위치 | `location_tracking` | 선박 3D 공간 내 현재 도장 구역 식별. pose/sequence MCP에 공간 context 공급. PaintNet trajectory 기반 |
| MCP 순서 | `sequence_validation` | 공정 단계 순서가 아닌 **구역 내 stroke 순서** 검증 (위→아래 등). 숙련자 stroke 패턴이 reference |
| MCP 작업 매뉴얼 | RAG Core (별도 MCP 불필요) | RAG 파이프라인이 문서 검색·제공 기능을 이미 수행하므로 별도 MCP 서버로 분리하지 않음 |
| ••• (확장) | `dft_judge` | 규제/리포트용 명시적 pass/fail. anomaly_detection과 분리하여 90/10 룰 결과를 독립 출력 |

### Tool Calling 메커니즘 매핑

아키텍처에서는 CoT → Thinking 단계에서 `</ToolPos>`, `</ToolSeq>` 등 특수 종료 태그를 생성하여 Tool Calling 계층이 파싱·호출하는 방식을 정의하였다. 실제 구현에서는 **Qwen3(4B/8B)의 native function calling + Thinking Mode**를 사용한다.

| 아키텍처 개념 | Qwen3 구현 방식 |
|---|---|
| `</ToolPos>` 태그 → Tool Calling 계층 파싱 | Qwen3 `tools` 파라미터로 도구 목록 전달, `<think>` 블록에서 도구 필요성 추론 후 `tool_calls` JSON 생성 |
| Thinking 내 도구 식별 | Qwen3 Thinking Mode (`enable_thinking=True`): 모델이 `<think>...</think>` 내에서 자율적으로 도구 필요성 판단 → tool_call 결정. 아키텍처의 "Thinking" 컴포넌트와 1:1 대응 |
| 빨간 점선 (Thinking → Tool Calling) | tool_calls 응답 파싱 → MCP Server HTTP 호출 → 결과를 context에 삽입 |
| 턴별 모드 전환 | `/think` (복잡한 추론) / `/no_think` (단순 조회) 소프트 스위치로 턴별 thinking 토글 |

> **모델 마이그레이션 결정 (2026-03)**: 1차 PoC의 Qwen2.5-7B-Instruct에서 **Qwen3-4B**(우선) / **Qwen3-8B**(A/B 비교)로 마이그레이션. EXAONE 3.5는 NC 라이선스 및 Native Tool Calling 미지원으로 제외.

### Encoder/Decoder 및 입출력 변환 계층

아키텍처의 `Conversion Request Format → Encoder → ... → Decoder → Conversion Output Format` 파이프라인은 Agent Orchestration (Task 18)에서 다음과 같이 구현한다.

| 아키텍처 컴포넌트 | 구현 위치 |
|---|---|
| Conversion Request Format | `agent_pipeline.py` — instruction + context를 LLM 프롬프트 포맷으로 정규화 |
| Encoder | Qwen3 tokenizer + embedding (모델 내부) |
| LLM Backbone + CoT + Thinking | Qwen3-4B/8B 추론 (Thinking Mode `<think>` 블록 = 아키텍처의 CoT+Thinking 컴포넌트) |
| Tool Calling → MCP Server | HTTP client → FastAPI MCP 서버 호출 |
| Decoder | Qwen2.5 토큰 생성 (모델 내부) |
| Conversion Output Format | 응답 후처리 — JSON 구조화 (XR 콘텐츠) 또는 텍스트 포맷 변환 |

### Context 순환 갱신

아키텍처 §6.3의 "응답 후 Context 업데이트 → 다음 질문 대비 순환 구조"는 다음과 같이 구현한다.

- **Session Context Manager**: 각 세션의 대화 이력, MCP 호출 결과, 추론 과정을 `session_context` 딕셔너리에 누적
- **Vector DB 갱신**: 새로운 MCP 결과가 생성될 때 Vector DB에 임시 인덱싱하여 후속 질문에서 검색 가능
- **Context Window 관리**: 최근 N턴 대화 + 관련 MCP 결과를 LLM 입력에 포함 (max_context_turns 설정)

\newpage

# 부록 A. 필수 명령어 모음

## A-1. 문서 인제스트 실행 순서

```bash
source .venv-rag/bin/activate
python scripts/ingest_docs.py --input ./data/raw --output ./data/parsed
python scripts/clean_docs.py --input ./data/parsed --output ./data/cleaned
python scripts/dedup_docs.py --input ./data/cleaned --report ./artifacts/reports/dedup.csv
python scripts/chunk_docs.py --input ./data/cleaned --output ./data/chunks/chunks.jsonl
```

## A-2. 인덱스 생성

```bash
source .venv-rag/bin/activate
python scripts/build_index.py \
  --chunks ./data/chunks/chunks.jsonl \
  --embed-model ./cache/hf/bge-m3 \
  --index-out ./artifacts/indexes/faiss.index
```

## A-3. RAG 평가

```bash
source .venv-rag/bin/activate
python scripts/run_rag_eval.py \
  --eval-set ./data/eval/eval_set.jsonl \
  --index ./artifacts/indexes/faiss.index \
  --model ./cache/hf/Qwen2.5-7B-Instruct \
  --output ./artifacts/reports/rag_only_results.csv
```

## A-4. QLoRA 학습

```bash
source .venv-train/bin/activate
python scripts/train_qlora.py \
  --model unsloth/Qwen2.5-7B-Instruct-bnb-4bit \
  --train ./data/sft/sft_train.jsonl \
  --out ./artifacts/adapters/qwen25_shipcoat_lora \
  --seq-len 2048 \
  --r 16 \
  --alpha 32 \
  --epochs 3
```

## A-5. vLLM 패키징(선택)

```bash
source .venv-vllm/bin/activate
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype auto \
  --served-model-name qwen2.5-7b-ship \
  --api-key shipcoat-demo-key
```

# 부록 B. 실행 의사결정 트리

```text
RAG baseline 완성?
 ├─ 아니오 → 문서 파이프라인/인덱스부터 수정. LoRA 금지
 └─ 예
    ├─ RAG-only가 기준 만족? (>=1.5/2.0, hallucination <=10%)
    │   ├─ 예 → LoRA는 선택 검증만 수행
    │   └─ 아니오 → retrieval/chunk/prompt를 먼저 수정
    └─ QLoRA 결과가 RAG-only보다 나은가?
        ├─ 예 → adapter 유지
        └─ 아니오 → adapter 폐기, RAG-only 채택
```

# 부록 C. 참고 자료

## 내부 자료

| ID | 자료 |
|---|---|
| I1 | `6. 2025년도 신규지원 대상과제 RFP_방송·콘텐츠(72번).pdf` |
| I2 | `멀티모달 AI 도장훈련시뮬레이터(검토용)_rev1.5.pptx` |
| I3 | `RealCOAT Indroduction(2025) 20250107.pdf` |
| I4 | `선박도장 도메인 AI PoC 전략 보고서.pdf` |
| I5 | `Ship-painting domain AI: 1-week PoC technical stack reference` |
| I6 | `deep-research-report2.md` |

## 외부 공식 자료 (2026-03-09 확인)

| ID | 자료 |
|---|---|
| W1 | Hugging Face - `Qwen/Qwen2.5-7B-Instruct` model card |
| W2 | Hugging Face - `Qwen/Qwen3-VL-2B-Instruct` model card (Phase 4 이미지 context 분석용, `/models/Qwen3-VL-2B-Instruct` 다운로드 완료) |
| W3 | NVIDIA - A40 공식 사양 |
| W4 | Hugging Face Transformers / bitsandbytes 문서 - QLoRA, NF4, CUDA 지원 |
| W5 | Hugging Face Hub 문서 - `snapshot_download`, `local_dir`, offline mode |
| W6 | vLLM 공식 문서 + PyPI `vllm==0.16.0` |
| W7 | PyPI `unsloth==2026.2.1` |
| W8 | PyPI `llama-index-core==0.14.14` |
| W9 | PyPI `llama-index-vector-stores-faiss==0.5.3` |
| W10 | Hugging Face `BAAI/bge-m3`, PyPI `sentence-transformers==5.2.3` |
| W11 | Hugging Face `BAAI/bge-reranker-v2-m3`, PyPI `FlagEmbedding==1.3.5` |
| W12 | PyPI `faiss-gpu-cu12==1.13.2` |
| W13 | PyPI `accelerate==1.13.0` |
| W14 | PyPI `transformers==5.1.0`, PyPI `bitsandbytes==0.49.1` |
| W15 | Qdrant 공식 문서 / PyPI `qdrant-client` |

