---
title: "선박 도장 AI Co-Pilot — Qwen3.5 마이그레이션 PRD"
subtitle: "Qwen3-4B → Qwen3.5-4B 전환 계획, 영향 분석, 실행 로드맵"
date: "2026-03-12"
lang: ko-KR
numbersections: true
geometry: margin=0.78in
---

## 목차

1. 문서 개요
2. 배경 및 왜 이 마이그레이션이 필요한가
3. 현재 상태 (As-Is)
4. 목표 상태 (To-Be)
5. Qwen3.5 기술 분석
6. Breaking Changes 영향 분석
7. 마이그레이션 범위 및 제외 항목
8. 의존성 업그레이드 계획
9. SFT 데이터 마이그레이션
10. Tool Calling 포맷 마이그레이션
11. 학습 파이프라인 변경
12. 추론 파이프라인 변경
13. MCP Server 변경
14. 평가 체계 변경
15. 리스크 및 폴백 전략
16. 실행 계획 (Phase별)
17. 성공 기준
18. 부록

\newpage

# 1. 문서 개요

| 항목 | 내용 |
|---|---|
| 문서 목적 | 현재 Qwen3-4B 기반 선박 도장 AI Co-Pilot을 **Qwen3.5-4B**로 마이그레이션하기 위한 실행 문서 |
| 마이그레이션 대상 | 학습(QLoRA/LoRA), 추론(agent_pipeline), MCP 연동(tool_calling_adapter), 평가(agentic eval) 전 파이프라인 |
| 마이그레이션 사유 | 네이티브 멀티모달(VLM 통합), Tool Calling 벤치마크 2배 향상, 262K 컨텍스트, 201개 언어 지원 |
| 핵심 제약 | 기존 LoRA adapter **전량 재학습** 필수 (vocab 152K→248K 호환 불가) |
| 선행 조건 | 현재 PoC 모든 Task 완료 (Tasks 1-19 done), FC-aware SFT v2 평가 완료 (2.97/10) |
| 배포 환경 | 온프레미스 / NVIDIA A40 x1 (46GB VRAM) / 오프라인 우선 |

## 한 줄 결론

Qwen3.5-4B는 네이티브 VLM(early fusion)으로 별도 Qwen3-VL 모델이 불필요해지고, Tool Calling 벤치마크에서 GPT-5-Nano를 2배 초과하며, 262K 컨텍스트로 RAG 파이프라인 제약이 해소된다. 다만 vocab 크기 변경(152K→248K)과 아키텍처 변경(GQA→Gated DeltaNet)으로 **모든 adapter와 SFT 데이터를 재구축**해야 하므로, 현재 PoC 안정화 이후 별도 Sprint로 진행한다.

\newpage

# 2. 배경 및 왜 이 마이그레이션이 필요한가

## 2.1 현재 PoC 성과 요약

| 지표 | 값 | 비고 |
|---|---|---|
| Agentic Score (30문항) | 2.97/10 | FC-aware SFT v2, Iter7 |
| Tool Calls Made | 60 | Baseline 0 → 60 |
| RAG+QLoRA Score (50문항) | 1.368/2.0 | Iter4 기준 |
| Hallucination Rate | 10% | 목표 ≤10% 달성 |
| 모델 | Qwen3-4B | VRAM ~8.5GB (QLoRA 학습 시) |

## 2.2 마이그레이션 동기

### 성능 한계

현재 Qwen3-4B 기반 파이프라인의 주요 병목:

1. **Tool Calling 정확도**: 30문항 중 14문항만 tool_call 생성 (46.7%). Malformed JSON 4건, 미등록 도구 호출 1건
2. **VLM 분리 구조**: Qwen3-VL-2B를 별도 인스턴스로 운영해야 함 → 복잡성 증가, VRAM 분할 필요
3. **컨텍스트 제한**: 32K 토큰 → 긴 세션 히스토리나 다중 MCP 결과 통합 시 truncation 발생
4. **한국어+다국어**: 119개 언어 → 조선소 다국어 환경(한/영/중/일) 지원에 한계

### Qwen3.5가 해결하는 문제

| 병목 | Qwen3 (현재) | Qwen3.5 (목표) | 개선 |
|---|---|---|---|
| Tool Calling | BFCL ~85%, TIR-Bench N/A | TIR-Bench **38.9** (GPT-5-Nano: 18.5) | **2x+** |
| VLM | 별도 Qwen3-VL-2B 필요 | 네이티브 통합 (early fusion) | 모델 1개로 통합 |
| 컨텍스트 | 32K | **262K** (YaRN 확장 ~1M) | **8x** |
| 다국어 | 119개 언어 | **201개** 언어+방언 | +82 |
| GPQA Diamond | ~59 (est.) | **76.2** | +17.2 |
| IFEval | ~85 (est.) | **89.8** | +4.8 |
| MMMLU (다국어) | ~72 (est.) | **76.1** | +4.1 |

### 비용-편익 분석

| 비용 | 편익 |
|---|---|
| SFT 데이터 재토큰화 (425건) | 네이티브 VLM → Phase 4 VLM 별도 구현 불필요 |
| LoRA adapter 전량 재학습 (~20min) | Tool Calling 정확도 대폭 향상 → SFT 데이터 감소 가능 |
| Tool Calling 포맷 코드 변경 | 262K 컨텍스트 → RAG 증거 제한 해소 |
| transformers v5 업그레이드 | 향후 Qwen3.5 생태계 (vLLM, SGLang) 안정 지원 |
| 약 1 Sprint (3-5일) | Phase 4-5 로드맵 단순화, 프로덕션 준비도 상승 |

\newpage

# 3. 현재 상태 (As-Is)

## 3.1 모델 스택

```
Qwen3-4B (text-only, 152K vocab)
├── QLoRA adapter: qwen3_4b_shipcoat_lora (FC-aware SFT v2)
│   ├── 425 train / 30 val (380 docQA + 45 FC)
│   ├── rank=16, alpha=32, dropout=0
│   └── targets: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
├── RAG: FAISS + BGE-M3 (10,829 chunks)
└── Thinking Mode: enable_thinking=False, /nothink soft switch
```

## 3.2 Tool Calling 포맷 (Qwen3)

```xml
<tool_call>{"name": "dft_judge", "arguments": {"measurements_um": [120, 135, 128]}}</tool_call>
```

파싱: `tool_calling_adapter.py` → regex `<tool_call>(.*?)</tool_call>` → JSON.loads

## 3.3 VLM 구조

```
Qwen3-4B (text LLM) ──── agent_pipeline.py
Qwen3-VL-2B-Instruct ─── Phase 4 예정 (별도 인스턴스)
```

## 3.4 추론 파이프라인

```
qa_pipeline.py / agent_pipeline.py
├── model.generate(max_new_tokens=1024)
├── tool_calling_adapter.py → parse_tool_calls() → dispatch_tool_call()
├── MCP Server (FastAPI, port 8100)
└── generate_agentic_responses.py → run_agentic_eval.py
```

## 3.5 학습 파이프라인

```
train_qlora.py
├── bitsandbytes NF4 quantization
├── peft LoraConfig (r=16, alpha=32)
├── trl SFTTrainer
├── transformers v4.x/v5.x (현재 5.1.0)
└── max_seq_length=2048
```

## 3.6 주요 파일 경로

| 파일 | 역할 |
|---|---|
| `/workspace/trainer/scripts/tool_calling_adapter.py` | FC 파싱, MCP 디스패치, 에이전트 루프 |
| `/workspace/trainer/scripts/agent_pipeline.py` | AgentPipeline, SessionContext |
| `/workspace/trainer/scripts/qa_pipeline.py` | RAG QA 파이프라인 |
| `/workspace/trainer/scripts/train_qlora.py` | QLoRA/LoRA 학습 |
| `/workspace/trainer/scripts/generate_sft.py` | SFT 데이터 생성 |
| `/workspace/trainer/scripts/generate_fc_sft.py` | FC-aware SFT 데이터 생성 |
| `/workspace/trainer/configs/rag_config.yaml` | RAG 설정 (모델명, 프롬프트) |
| `/workspace/trainer/configs/rag_config_fewshot.yaml` | Few-shot tool call 예시 포함 |
| `/workspace/trainer/configs/qlora_config.yaml` | QLoRA 학습 설정 |
| `/workspace/trainer/configs/tool_calling_protocol.json` | Tool calling 규약 |
| `/workspace/trainer/configs/tool_schemas/*.json` | 6개 도구 JSON 스키마 |
| `/data/sft/sft_train.jsonl` | 425건 SFT 학습 데이터 |
| `/data/sft/sft_val.jsonl` | 30건 SFT 검증 데이터 |
| `/models/Qwen3-4B/` | 현재 base 모델 (7.6GB) |
| `/models/Qwen3-VL-2B-Instruct/` | Phase 4 VLM (별도) |
| `/output/adapters/qwen3_4b_shipcoat_lora/` | 현재 active adapter |

\newpage

# 4. 목표 상태 (To-Be)

## 4.1 모델 스택

```
Qwen3.5-4B (unified VLM, 248K vocab, Gated DeltaNet hybrid)
├── bf16 LoRA adapter: qwen35_4b_shipcoat_lora
│   ├── ~455+ train / ~30 val (docQA + FC + 신규 XML FC 포맷)
│   ├── rank=16, alpha=16, dropout=0
│   └── targets: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
├── RAG: FAISS + BGE-M3 (10,829 chunks, 변경 없음)
├── Thinking Mode: enable_thinking=False via chat_template_kwargs
└── Native Vision: heatmap/screenshot 분석 가능 (Phase 4 통합)
```

## 4.2 Tool Calling 포맷 (Qwen3.5)

```xml
<tool_call>
<function=dft_judge>
<parameter=measurements_um>
[120, 135, 128]
</parameter>
<parameter=spec_min_um>
100
</parameter>
<parameter=spec_max_um>
200
</parameter>
</function>
</tool_call>
```

파싱: 신규 XML 파서 필요 (`<function=NAME>` + `<parameter=KEY>VALUE</parameter>`)

## 4.3 VLM 구조 (통합)

```
Qwen3.5-4B (unified VLM) ──── agent_pipeline.py
└── text + image 모두 처리 (Phase 4 heatmap 분석 포함)
```

> Qwen3-VL-2B-Instruct 별도 인스턴스 **불필요**. `/models/Qwen3-VL-2B-Instruct/` 아카이브 처리.

## 4.4 핵심 개선 지표 (예상)

| 지표 | 현재 (Qwen3-4B) | 목표 (Qwen3.5-4B) | 근거 |
|---|---|---|---|
| Agentic Score | 2.97/10 | ≥4.0/10 | TIR-Bench 2x 향상, 네이티브 FC 강화 |
| Tool Calls / 30문항 | 60 | ≥80 | XML 포맷 정확도 + 컨텍스트 확장 |
| Malformed JSON | 4건 | 0건 | XML 포맷은 truncation에 더 robust |
| VRAM (학습) | 8.5GB (QLoRA 4bit) | ~10GB (bf16 LoRA) | A40 46GB 내 충분 |
| 컨텍스트 | 32K | 262K | RAG 증거 제한 해소 |
| VLM 인스턴스 | 2개 (text + VL) | 1개 (통합) | VRAM 절약, 복잡성 감소 |

\newpage

# 5. Qwen3.5 기술 분석

## 5.1 릴리스 현황

| 항목 | 상세 |
|---|---|
| 초기 릴리스 | 2026-02-16 (flagship 397B-A17B MoE) |
| 소형 모델 오픈소스 | 2026-03-02 (0.8B, 2B, **4B**, 9B) |
| 라이선스 | Apache 2.0 |
| GitHub | [QwenLM/Qwen3.5](https://github.com/QwenLM/Qwen3.5) |
| HuggingFace | [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) |

## 5.2 아키텍처: Gated DeltaNet Hybrid

Qwen3.5는 순수 Transformer가 아닌 **Gated DeltaNet 하이브리드** 아키텍처를 사용한다.

```
32 layers = 24 × Gated DeltaNet (linear attention) + 8 × Gated Attention (full GQA)
비율: 3:1 (linear : full)
```

### Gated DeltaNet 레이어 (24개)
- Linear attention heads: 32 (V), 16 (QK)
- Head dimension: 128
- Projections: `in_proj_qkv`, `in_proj_z`, `in_proj_b`, `in_proj_a`
- **Custom Mamba Triton 커널** 필요 (초기 컴파일 시간 발생)

### Gated Attention 레이어 (8개)
- GQA: 16 Q heads, 4 KV heads
- Head dimension: 256
- RoPE dimension: 64
- Standard `q_proj`, `k_proj`, `v_proj`, `o_proj`

### LoRA 적용 범위

표준 LoRA target (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`)은 **Gated Attention 레이어(8개)와 전체 FFN 레이어(32개)**에만 적용된다. DeltaNet 레이어의 어텐션 파라미터(`in_proj_*`)에는 적용되지 않지만, Unsloth/DataCamp 튜토리얼에서 이 설정으로 정상 동작이 확인되었다.

## 5.3 Vocabulary (248,320 tokens)

| | Qwen3 | Qwen3.5 | 변화 |
|--|-------|---------|------|
| Vocab size | 152,064 (padded) | **248,320** | +63.3% |
| `<\|endoftext\|>` ID | 151,643 | **248,044** | shifted |
| `<\|im_start\|>` ID | 151,644 | **248,045** | shifted |
| `<\|im_end\|>` ID | 151,645 | **248,046** | shifted |
| 신규 특수 토큰 | — | `<\|audio_start\|>`, `<\|audio_end\|>`, `<tts_pad>` | 오디오 지원 |

> ⚠️ **Hard Breaking Change**: 모든 특수 토큰 ID가 변경됨. 하드코딩된 토큰 ID 참조가 있으면 반드시 업데이트 필요.

## 5.4 네이티브 멀티모달 (Early Fusion)

Qwen3.5는 **통합 VLM**이다. 모든 체크포인트(0.8B~397B)에 Vision Encoder가 내장되어 있다.

| 특성 | Qwen3 | Qwen3.5 |
|---|---|---|
| 모델 클래스 | `Qwen3ForCausalLM` | `Qwen3_5ForConditionalGeneration` |
| Vision | 별도 Qwen3-VL 모델 | **내장** (early fusion) |
| 지원 모달리티 | text only | text + image + video |
| heatmap 분석 | Qwen3-VL-2B 별도 필요 | 기본 모델에서 직접 가능 |

## 5.5 Thinking Mode 변경

| | Qwen3 | Qwen3.5 |
|--|-------|---------|
| 기본 동작 | Non-thinking | **Thinking (기본 ON)** |
| 활성화/비활성화 | `/think`, `/nothink` soft switch | `chat_template_kwargs: {"enable_thinking": False}` |
| SFT 데이터 권장 | — | 75% 이상 `<think>...</think>` 포함 (reasoning 보존 시) |

> ⚠️ `/think`, `/nothink` soft switch가 **제거**되었다. 기존 프롬프트 템플릿에서 이를 사용하는 코드는 반드시 변경해야 한다.

## 5.6 컨텍스트 윈도우

| | Qwen3-4B | Qwen3.5-4B |
|--|----------|------------|
| 네이티브 | 32,768 | **262,144** |
| YaRN 확장 | ~128K | ~1,010,000 |

## 5.7 Multi-Token Prediction (MTP)

Qwen3.5는 MTP로 학습되어 추론 시 Speculative Decoding을 지원한다:

```bash
vllm serve Qwen/Qwen3.5-4B \
  --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}'
```

## 5.8 Sampling Parameters (변경)

| Mode | temperature | top_p | top_k | presence_penalty |
|------|-----------|-------|-------|-----------------|
| Thinking / general | 1.0 | 0.95 | 20 | 1.5 |
| Non-thinking / general | 0.7 | 0.8 | 20 | 1.5 |
| Non-thinking / reasoning | 1.0 | 0.95 | 40 | 2.0 |

\newpage

# 6. Breaking Changes 영향 분석

## 6.1 영향도 매트릭스

| # | Breaking Change | 영향 범위 | 심각도 | 수정 난이도 |
|---|---|---|---|---|
| **BC-1** | Vocab 152K→248K | adapter 전량 재학습, SFT 재토큰화 | 🔴 Critical | Medium (자동화 가능) |
| **BC-2** | 아키텍처 GQA→DeltaNet hybrid | model 로딩 코드, transformers v5 필수 | 🔴 Critical | Low (버전 업그레이드) |
| **BC-3** | Tool Call 포맷 JSON→XML | parse_tool_calls, SFT FC 데이터, few-shot 프롬프트 | 🔴 Critical | Medium |
| **BC-4** | `/think`/`/nothink` 제거 | 프롬프트 템플릿, inference 코드 | 🟡 High | Low |
| **BC-5** | 모델 클래스명 변경 | `ForCausalLM` → `ForConditionalGeneration` | 🟡 High | Low |
| **BC-6** | Thinking 기본 ON | SFT 데이터 포맷, 생성 후처리 | 🟡 High | Medium |
| **BC-7** | QLoRA 품질 저하 | 학습 방식 QLoRA→bf16 LoRA | 🟢 Medium | Low (VRAM 충분) |
| **BC-8** | vLLM 0.16.x 미지원 | 서빙 인프라 (현재 미사용) | 🟢 Low | Low |
| **BC-9** | Sampling params 변경 | generation config | 🟢 Low | Low |

## 6.2 파일별 영향 분석

| 파일 | 영향받는 BC | 변경 유형 | 예상 공수 |
|---|---|---|---|
| `tool_calling_adapter.py` | BC-3, BC-4, BC-5, BC-6 | **대규모 수정**: parse_tool_calls XML 파서, thinking 처리 | 4h |
| `agent_pipeline.py` | BC-4, BC-5 | 중규모: 모델 로딩, thinking 모드 설정 | 2h |
| `qa_pipeline.py` | BC-4, BC-5, BC-9 | 중규모: 모델 로딩, generation config | 2h |
| `train_qlora.py` | BC-1, BC-2, BC-7 | 중규모: bf16 LoRA 전환, model_id 변경 | 2h |
| `generate_fc_sft.py` | BC-3, BC-6 | **대규모 수정**: XML tool call 포맷 SFT 생성 | 3h |
| `generate_sft.py` | BC-6 | 소규모: thinking 블록 포맷 조정 | 1h |
| `rag_config.yaml` | BC-4 | 소규모: 모델명, 프롬프트 수정 | 0.5h |
| `rag_config_fewshot.yaml` | BC-3, BC-4 | 중규모: few-shot 예시 XML 변환 | 1h |
| `qlora_config.yaml` | BC-1, BC-7 | 소규모: model_name, 학습 설정 | 0.5h |
| `tool_calling_protocol.json` | BC-3 | 소규모: 포맷 설명 업데이트 | 0.5h |
| `run_agentic_eval.py` | BC-3 | 소규모: tool call 추출 로직 | 1h |
| `generate_agentic_responses.py` | BC-5 | 소규모: 모델 로딩 | 0.5h |

**총 예상 공수**: ~18시간 (코드 변경) + ~20분 (재학습) + ~1시간 (재평가)

\newpage

# 7. 마이그레이션 범위 및 제외 항목

## 7.1 이번 마이그레이션에 포함 (Scope)

1. **모델 다운로드**: `Qwen/Qwen3.5-4B` → `/models/Qwen3.5-4B/`
2. **의존성 업그레이드**: transformers v5+, peft, trl, unsloth
3. **SFT 데이터 변환**: Qwen3 chat template → Qwen3.5 chat template, FC XML 포맷
4. **LoRA 재학습**: bf16 LoRA (QLoRA 4-bit 대신)
5. **Tool Calling 파서**: JSON → XML (`<function=...><parameter=...>`)
6. **프롬프트 템플릿**: `/nothink` 제거, `enable_thinking=False` 적용
7. **추론 코드**: 모델 클래스, generation config 업데이트
8. **평가 재실행**: 50문항 docQA + 30문항 agentic
9. **PATCH_NOTE 업데이트**: Iter8 기록

## 7.2 이번 마이그레이션에서 제외 (Out of Scope)

1. **Vision 파인튜닝**: Phase 4에서 별도 진행. 이번에는 text-only 파인튜닝만 수행
2. **vLLM/SGLang 서빙 전환**: 프로덕션 서빙은 별도 Sprint
3. **Intent Router 구현**: 별도 Task로 관리
4. **추가 MCP 도구 구현**: paint_usage_analyzer, report_generator 등
5. **Qwen3.5-9B 비교 실험**: 4B 안정화 이후 검토
6. **오디오 모달리티**: Qwen3.5의 TTS/ASR 기능은 현재 미사용

\newpage

# 8. 의존성 업그레이드 계획

## 8.1 Python 패키지

| 패키지 | 현재 버전 | 필요 버전 | 비고 |
|---|---|---|---|
| `transformers` | 5.1.0 | **≥5.0.0** (이미 충족) | `Qwen3_5ForConditionalGeneration` 클래스 필요 |
| `peft` | 현재 설치 | 최신 안정 | LoRA target_modules 호환 확인 |
| `trl` | 현재 설치 | ≥0.28.x | SFTTrainer `processing_class` 파라미터 |
| `bitsandbytes` | 0.49.1 | 유지 | bf16 LoRA 시 quantization 미사용 가능 |
| `unsloth` | 2026.2.1 | **최신 (force-reinstall)** | Gated DeltaNet 커널 지원 |
| `torch` | 2.4.0 | 유지 | CUDA 12.4 호환 |
| `vllm` | 미설치 | 0.17.1+ (선택) | MTP speculative decoding 시 필요 |

## 8.2 업그레이드 명령어

```bash
# transformers v5 확인 (이미 충족 예상)
pip show transformers | grep Version

# unsloth 강제 업데이트 (Gated DeltaNet 커널)
pip install --upgrade --force-reinstall --no-cache-dir unsloth unsloth_zoo

# peft, trl 최신
pip install --upgrade peft trl
```

## 8.3 모델 다운로드

```bash
# HuggingFace Hub에서 다운로드 (오프라인 환경 대비)
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='Qwen/Qwen3.5-4B',
    local_dir='/models/Qwen3.5-4B',
    local_dir_use_symlinks=False
)
"
```

예상 크기: ~8-9 GB (248K vocab으로 embedding 확장)

\newpage

# 9. SFT 데이터 마이그레이션

## 9.1 현재 SFT 데이터 구조

```
/data/sft/sft_train.jsonl (425건)
├── 380건 docQA (126건 FC system prompt 업그레이드)
├── 45건 FC tool-calling samples
│   ├── 15 dft_judge
│   ├── 10 anomaly_detection
│   ├── 5 location_tracking
│   ├── 5 pose_estimation
│   ├── 5 sequence_validation
│   ├── 5 timeseries
│   └── 5 multi-tool (미포함 가능)
└── 5건 no-tool QA

/data/sft/sft_val.jsonl (30건)
├── 20건 docQA
└── 10건 FC
```

## 9.2 변환 필요 항목

### 9.2.1 Tool Call 포맷 변환 (45건 FC + 5건 no-tool)

**Before (Qwen3)**:
```json
{"role": "assistant", "content": "<tool_call>{\"name\": \"dft_judge\", \"arguments\": {\"measurements_um\": [120, 135, 128], \"spec_min_um\": 100, \"spec_max_um\": 200}}</tool_call>"}
```

**After (Qwen3.5)**:
```json
{"role": "assistant", "content": "<tool_call>\n<function=dft_judge>\n<parameter=measurements_um>\n[120, 135, 128]\n</parameter>\n<parameter=spec_min_um>\n100\n</parameter>\n<parameter=spec_max_um>\n200\n</parameter>\n</function>\n</tool_call>"}
```

### 9.2.2 Thinking Mode 처리

Qwen3.5는 기본적으로 thinking이 활성화된다. SFT 데이터에서:

- **FC 데이터 (45건)**: `enable_thinking=False`로 생성 → thinking 블록 미포함
- **docQA 데이터 (380건)**: thinking 블록 포함 여부 결정 필요
  - **권장**: 75% 이상 `<think>...</think>` 포함 (reasoning 보존)
  - **대안**: 전량 non-thinking (현재와 동일, 더 간단)

### 9.2.3 Chat Template 변환

```python
# Qwen3.5 tokenizer의 apply_chat_template 사용
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")
formatted = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    chat_template_kwargs={"enable_thinking": False}
)
```

## 9.3 데이터 변환 스크립트 (신규)

`/workspace/trainer/scripts/migrate_sft_qwen35.py` 신규 생성:

```
입력: /data/sft/sft_train.jsonl, /data/sft/sft_val.jsonl
출력: /data/sft/sft_train_qwen35.jsonl, /data/sft/sft_val_qwen35.jsonl
기능:
  1. Qwen3 JSON tool_call → Qwen3.5 XML tool_call 변환
  2. Qwen3.5 chat template 적용
  3. /nothink, /think 참조 제거
  4. 검증: 변환 전후 건수 일치, 필수 필드 존재
```

## 9.4 추가 SFT 데이터 생성 (선택)

Qwen3.5의 네이티브 FC 능력이 Qwen3보다 강하므로, FC SFT 데이터를 줄이거나 더 복잡한 multi-tool 시나리오에 집중할 수 있다:

| 전략 | FC 데이터량 | 기대 효과 |
|---|---|---|
| A: 기존 유지 | 45건 (XML 변환) | 안전, 최소 변경 |
| B: 복잡도 증강 | 45건 + 20건 multi-tool | 복합 추론 강화 |
| C: 축소 | 20건 (대표 사례만) | 네이티브 FC에 의존 |

**권장**: 전략 A (기존 유지, XML 변환)로 시작 → 평가 결과에 따라 B/C 선택

\newpage

# 10. Tool Calling 포맷 마이그레이션

## 10.1 포맷 비교

### Qwen3 (현재)
```
호출: <tool_call>{"name": "func", "arguments": {"k": "v"}}</tool_call>
응답: <tool_response>{"result": ...}</tool_response>
```

### Qwen3.5 (목표)
```
호출:
<tool_call>
<function=func>
<parameter=k>
v
</parameter>
</function>
</tool_call>

응답: <tool_response>{"result": ...}</tool_response>  (동일)
```

## 10.2 파서 변경 (`tool_calling_adapter.py`)

### 현재 파서 (JSON 기반)

```python
TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)

def parse_tool_calls(text):
    for match in TOOL_CALL_RE.finditer(text):
        payload = json.loads(match.group(1))
        tool_name = payload["name"]
        arguments = payload["arguments"]
```

### 신규 파서 (XML 기반)

```python
TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
FUNCTION_RE = re.compile(r"<function=([^>]+)>(.*?)</function>", re.DOTALL)
PARAMETER_RE = re.compile(r"<parameter=([^>]+)>\n?(.*?)\n?</parameter>", re.DOTALL)

def parse_tool_calls_qwen35(text):
    for tc_match in TOOL_CALL_RE.finditer(text):
        tc_block = tc_match.group(1)
        func_match = FUNCTION_RE.search(tc_block)
        if not func_match:
            continue
        tool_name = func_match.group(1).strip()
        params_block = func_match.group(2)
        arguments = {}
        for p_match in PARAMETER_RE.finditer(params_block):
            key = p_match.group(1).strip()
            value = _parse_param_value(p_match.group(2).strip())
            arguments[key] = value
        yield {"name": tool_name, "arguments": arguments}
```

### 하위 호환성

마이그레이션 과도기에 두 포맷을 모두 지원하는 것을 권장:

```python
def parse_tool_calls(text):
    """Qwen3 JSON 및 Qwen3.5 XML 포맷 모두 파싱"""
    calls = list(parse_tool_calls_qwen35_xml(text))
    if not calls:
        calls = list(parse_tool_calls_qwen3_json(text))
    return calls
```

## 10.3 Few-shot 프롬프트 변경 (`rag_config_fewshot.yaml`)

기존 few-shot 예시의 tool_call 포맷을 XML로 변환:

```yaml
# Before (Qwen3)
tool_call_example: |
  <tool_call>{"name": "dft_judge", "arguments": {"measurements_um": [120]}}</tool_call>

# After (Qwen3.5)
tool_call_example: |
  <tool_call>
  <function=dft_judge>
  <parameter=measurements_um>
  [120]
  </parameter>
  </function>
  </tool_call>
```

## 10.4 Tool Schema 전달 방식

Qwen3.5의 system prompt에서 도구 정의 방식:

```
# Tools

You have access to the following functions:

<tools>
[
  {
    "type": "function",
    "function": {
      "name": "dft_judge",
      "description": "DFT 도막 두께 합부 판정 (90/10 룰)",
      "parameters": { ... JSON Schema ... }
    }
  }
]
</tools>
```

> 도구 스키마 JSON 자체는 변경 불필요. 전달하는 system prompt 포맷만 업데이트.

\newpage

# 11. 학습 파이프라인 변경

## 11.1 QLoRA → bf16 LoRA 전환

### 사유

Unsloth 공식: *"Qwen3.5 모델에 QLoRA(4-bit) 학습은 권장하지 않음 — MoE, Dense 모두 양자화 차이가 정상보다 큼"*

### VRAM 비교

| 방식 | Qwen3-4B | Qwen3.5-4B |
|---|---|---|
| QLoRA (4-bit) | 8.5 GB | ~9 GB (비권장) |
| **bf16 LoRA** | ~14 GB | **~10 GB** |
| A40 여유 | 37.5 GB | **36 GB** |

A40(46GB)에서 bf16 LoRA는 여유 있게 수행 가능.

## 11.2 `train_qlora.py` 변경 사항

```python
# Before (QLoRA 4-bit)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config)

# After (bf16 LoRA)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
# quantization_config 제거
```

### LoRA 설정 변경

```python
# Before
LoraConfig(r=16, lora_alpha=32, ...)

# After (Unsloth/DataCamp 권장)
LoraConfig(
    r=16,
    lora_alpha=16,       # alpha=rank (1:1 비율)
    lora_dropout=0,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)
```

### 학습 안정성 설정

```python
SFTConfig(
    bf16=True,
    fp16=False,              # 반드시 False
    packing=False,
    optim="adamw_torch",     # 안정성 위해 8bit 아닌 표준
    max_grad_norm=0.5,       # gradient clipping 강화
    warmup_ratio=0.05,
    learning_rate=1e-4,      # 2e-4 → 1e-4 (안정성)
)
```

## 11.3 예상 학습 시간

| 항목 | Qwen3-4B QLoRA | Qwen3.5-4B bf16 LoRA |
|---|---|---|
| 데이터 | 425건 | ~455건 |
| 시간 | 19분 | **~25-30분** (bf16 메모리 증가) |
| Peak VRAM | 8.53 GB | **~10 GB** |
| loss 수렴 | 2.53→0.57 | TBD |

\newpage

# 12. 추론 파이프라인 변경

## 12.1 모델 로딩 (`qa_pipeline.py`, `agent_pipeline.py`)

```python
# Before
model = AutoModelForCausalLM.from_pretrained("/models/Qwen3-4B", ...)

# After
model = AutoModelForCausalLM.from_pretrained("/models/Qwen3.5-4B", ...)
# 또는 명시적으로:
from transformers import Qwen3_5ForConditionalGeneration
model = Qwen3_5ForConditionalGeneration.from_pretrained("/models/Qwen3.5-4B", ...)
```

## 12.2 Thinking Mode 처리

```python
# Before (Qwen3)
prompt += "\n/nothink"  # soft switch

# After (Qwen3.5)
# 프롬프트에 /nothink 추가하지 않음
# 대신 tokenizer.apply_chat_template에서 설정
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    chat_template_kwargs={"enable_thinking": False}
)
```

## 12.3 Generation Config

```python
# Before
generation_config = {
    "max_new_tokens": 1024,
    "temperature": 0.7,
    "top_p": 0.9,
    "pad_token_id": tokenizer.eos_token_id,
}

# After (Qwen3.5 권장)
generation_config = {
    "max_new_tokens": 1024,
    "temperature": 0.7,
    "top_p": 0.8,          # 0.9 → 0.8
    "top_k": 20,            # 신규 추가
    "presence_penalty": 1.5, # 신규 추가 (반복 방지)
    "pad_token_id": tokenizer.eos_token_id,
}
```

## 12.4 `<think>` 블록 후처리

Qwen3.5는 `enable_thinking=False`여도 간헐적으로 `<think>` 블록을 생성할 수 있다. 기존 `_strip_think_blocks()` 함수 유지.

```python
def _strip_think_blocks(text: str) -> str:
    """Remove <think>...</think> blocks from model output."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
```

\newpage

# 13. MCP Server 변경

## 13.1 변경 불필요 항목

MCP Server (FastAPI, port 8100)는 **변경 불필요**:

- 엔드포인트 URL 동일
- 요청/응답 JSON 스키마 동일
- `dft_judge`, `anomaly_detection` 등 6개 서버 로직 동일

## 13.2 변경 필요 항목

`tool_calling_adapter.py`의 `dispatch_tool_call()` — 이미 raw arguments 전송으로 수정 완료. Qwen3.5 XML 파서에서 추출한 `arguments` dict를 동일하게 전달하므로 추가 변경 불필요.

## 13.3 Tool Response 포맷

Tool response는 Qwen3과 동일하게 `<tool_response>` 태그 사용:

```xml
<tool_response>
{"status": "success", "result": {"verdict": "PASS", ...}}
</tool_response>
```

변경 불필요.

\newpage

# 14. 평가 체계 변경

## 14.1 평가셋 변경 불필요

- `/data/eval/eval_set.jsonl` (50문항 docQA) — 변경 없음
- `/data/eval/eval_set_agentic.jsonl` (30문항 agentic) — 변경 없음

## 14.2 평가 스크립트 변경

`run_agentic_eval.py`: tool_call 추출 로직에서 XML 파싱 지원 추가 필요.

## 14.3 평가 기준선 재설정

| 지표 | Qwen3-4B (현재 최선) | Qwen3.5-4B 목표 |
|---|---|---|
| Agentic Score (30문항) | 2.97/10 | **≥4.0/10** |
| Tool Calls Made | 60 | **≥80** |
| Questions with tools | 14/30 | **≥20/30** |
| Malformed output | 4건 | **≤1건** |
| RAG+QLoRA (50문항) | 1.368/2.0 | **≥1.3/2.0** (유지) |
| Hallucination | 10% | **≤10%** |

## 14.4 A/B 비교 프레임

Qwen3-4B (FC-aware v2)를 baseline으로, Qwen3.5-4B를 treatment로 동일 30문항 평가:

```
Baseline: agentic_responses_fc_v2b.jsonl (Qwen3-4B, 현재 진행 중)
Treatment: agentic_responses_qwen35.jsonl (Qwen3.5-4B, 마이그레이션 후)
```

\newpage

# 15. 리스크 및 폴백 전략

## 15.1 리스크 매트릭스

| 리스크 | 확률 | 영향 | 대응 |
|---|---|---|---|
| **R1**: QLoRA/LoRA 학습 loss 폭발 | Medium | High | `max_grad_norm=0.5`, `lr=1e-4`, batch=1. 실패 시 Unsloth FastLanguageModel 사용 |
| **R2**: Tool Call XML 파싱 정확도 저하 | Low | High | 하위호환 파서(JSON+XML 모두 지원). 실패 시 Qwen3 유지 |
| **R3**: docQA 성능 회귀 | Medium | Medium | 50문항 재평가. 1.2 미만 시 SFT 데이터 증강 또는 Qwen3 유지 |
| **R4**: transformers v5 호환 문제 | Low | High | 이미 v5.1.0 설치됨. 호환 확인 후 진행 |
| **R5**: Gated DeltaNet 커널 컴파일 실패 | Low | Medium | Unsloth force-reinstall. 실패 시 순수 transformers 사용 |
| **R6**: VRAM 부족 (bf16 LoRA) | Very Low | Medium | A40 46GB에서 10GB 필요. gradient_checkpointing 추가 가능 |
| **R7**: 모델 다운로드 실패 (오프라인) | Low | High | 사전 다운로드 후 `/models/Qwen3.5-4B/` 로컬 저장 |

## 15.2 폴백 전략

```
Qwen3.5-4B 마이그레이션 시도
├── 성공 (Agentic ≥4.0, docQA ≥1.3) → Qwen3.5 채택
├── 부분 성공 (한 지표만 미달) → SFT 데이터 증강 후 재시도 (1회)
└── 실패 (두 지표 모두 미달 또는 학습 불안정) → Qwen3-4B 유지, Qwen3.5-9B 검토
```

**Rollback 보장**: 기존 Qwen3-4B 모델, adapter, SFT 데이터를 모두 보존하므로 언제든 복원 가능.

\newpage

# 16. 실행 계획 (Phase별)

## Phase M-0: 준비 (0.5일)

| # | 작업 | 산출물 | 의존성 |
|---|---|---|---|
| M-0.1 | 현재 PoC 전체 완료 확인 | 모든 Task done 상태 | — |
| M-0.2 | Qwen3.5-4B 모델 다운로드 | `/models/Qwen3.5-4B/` | 인터넷 접근 |
| M-0.3 | 의존성 업그레이드 | transformers≥5, unsloth 최신 | M-0.2 |
| M-0.4 | 기존 자산 백업 | adapter, SFT 데이터 백업 | — |

## Phase M-1: 데이터 마이그레이션 (0.5일)

| # | 작업 | 산출물 | 의존성 |
|---|---|---|---|
| M-1.1 | SFT 데이터 변환 스크립트 작성 | `migrate_sft_qwen35.py` | M-0.3 |
| M-1.2 | FC SFT 데이터 XML 변환 | `sft_train_qwen35.jsonl` (455건) | M-1.1 |
| M-1.3 | 변환 데이터 검증 | 건수 일치, 포맷 확인 | M-1.2 |

## Phase M-2: 코드 마이그레이션 (1.5일)

| # | 작업 | 산출물 | 의존성 |
|---|---|---|---|
| M-2.1 | `tool_calling_adapter.py` XML 파서 구현 | parse_tool_calls_qwen35() | M-0.3 |
| M-2.2 | `agent_pipeline.py` 모델 로딩/thinking 변경 | Qwen3.5 호환 | M-0.3 |
| M-2.3 | `qa_pipeline.py` 모델 로딩/generation config | Qwen3.5 호환 | M-0.3 |
| M-2.4 | `train_qlora.py` bf16 LoRA 전환 | bf16 학습 지원 | M-0.3 |
| M-2.5 | 설정 파일 업데이트 | yaml, json config 수정 | M-0.3 |
| M-2.6 | `generate_fc_sft.py` XML 포맷 대응 | Qwen3.5 FC SFT 생성기 | M-2.1 |
| M-2.7 | `rag_config_fewshot.yaml` XML few-shot | XML 예시 포함 | M-2.1 |

## Phase M-3: 학습 및 평가 (1일)

| # | 작업 | 산출물 | 의존성 |
|---|---|---|---|
| M-3.1 | bf16 LoRA 학습 | `qwen35_4b_shipcoat_lora/` adapter | M-1.3, M-2.4 |
| M-3.2 | 30문항 Agentic 평가 | `agentic_responses_qwen35.jsonl` | M-3.1, M-2.1 |
| M-3.3 | 50문항 docQA 평가 | `ragqlora_qwen35_results.csv` | M-3.1, M-2.3 |
| M-3.4 | A/B 비교 리포트 | Qwen3 vs Qwen3.5 비교표 | M-3.2, M-3.3 |
| M-3.5 | PATCH_NOTE Iter8 기록 | PATCH_NOTE.md 업데이트 | M-3.4 |

## Phase M-4: 검증 및 안정화 (0.5일)

| # | 작업 | 산출물 | 의존성 |
|---|---|---|---|
| M-4.1 | 성능 기준 판정 (Go/No-Go) | 의사결정 기록 | M-3.4 |
| M-4.2 | (Go 시) 설정 파일 기본값 Qwen3.5로 전환 | config 업데이트 | M-4.1 |
| M-4.3 | (Go 시) tasks.json 마이그레이션 Task 추가/완료 | Task 상태 반영 | M-4.2 |
| M-4.4 | (No-Go 시) Rollback 및 원인 분석 | 분석 보고서 | M-4.1 |

## 총 예상 기간: **3-4일 (1 Sprint)**

```
Day 1 (AM): Phase M-0 + M-1
Day 1 (PM): Phase M-2 시작
Day 2 (Full): Phase M-2 완료
Day 3 (AM): Phase M-3 (학습 + 평가)
Day 3 (PM): Phase M-4 (검증 + 안정화)
Day 4 (Buffer): 이슈 대응 / SFT 데이터 증강 (필요 시)
```

\newpage

# 17. 성공 기준

## 17.1 필수 기준 (Must Have)

| 기준 | 목표값 | 측정 방법 |
|---|---|---|
| Agentic Score (30문항) | ≥4.0/10 | `run_agentic_eval.py` |
| Tool Calls Made | ≥80 | 응답 내 tool_call 파싱 카운트 |
| Malformed Output | ≤1건 | 파싱 실패 카운트 |
| RAG+QLoRA (50문항) | ≥1.3/2.0 | `run_qlora_eval.py` |
| Hallucination Rate | ≤10% | 평가 결과 분석 |
| 학습 Loss 수렴 | final_loss < 1.0 | train_qlora.py 로그 |

## 17.2 추가 기준 (Nice to Have)

| 기준 | 목표값 | 비고 |
|---|---|---|
| Agentic Score | ≥5.0/10 | Qwen3.5 네이티브 FC 활용 시 |
| Questions with tools | ≥22/30 | 73%+ tool 사용률 |
| 학습 시간 | ≤30분 | bf16 LoRA, A40 |
| VRAM Peak | ≤15GB | bf16 LoRA |

## 17.3 Go/No-Go 의사결정 기준

```
Agentic ≥4.0 AND docQA ≥1.3 AND halluc ≤10%
├── 모두 충족 → GO (Qwen3.5 채택)
├── Agentic만 미달 → FC SFT 증강 후 1회 재시도
├── docQA만 미달 → docQA SFT 증강 후 1회 재시도
└── 2개 이상 미달 또는 학습 불안정 → NO-GO (Qwen3 유지)
```

\newpage

# 18. 부록

## 부록 A. Qwen3.5 모델 라인업 전체

| 모델 | 타입 | 전체 파라미터 | 활성 파라미터 | bf16 LoRA VRAM |
|---|---|---|---|---|
| Qwen3.5-0.8B | Dense VLM | 0.8B | 0.8B | 3 GB |
| Qwen3.5-2B | Dense VLM | 2B | 2B | 5 GB |
| **Qwen3.5-4B** | **Dense VLM** | **4B** | **4B** | **10 GB** |
| Qwen3.5-9B | Dense VLM | 9B | 9B | 22 GB |
| Qwen3.5-27B | Dense VLM | 27B | 27B | 56 GB |
| Qwen3.5-35B-A3B | MoE VLM | 35B | 3B | 74 GB |
| Qwen3.5-122B-A10B | MoE VLM | 122B | 10B | 256 GB |
| Qwen3.5-397B-A17B | MoE VLM | 397B | 17B | Flagship |

## 부록 B. 벤치마크 상세

### Qwen3.5-4B vs 경쟁 모델

| 벤치마크 | Qwen3.5-4B | GPT-5-Nano | Gemini-2.5-Flash-Lite |
|---|---|---|---|
| TIR-Bench (Tool Use) | **38.9** | 18.5 | 21.5 |
| V* (Vision) | **84.3** | 68.1 | 69.6 |
| GPQA Diamond | **76.2** | — | — |
| IFEval | **89.8** | — | — |
| MMMLU (다국어) | **76.1** | — | — |

### Qwen3.5-4B vs Qwen3-30B-A3B

| 벤치마크 | Qwen3-30B-A3B | Qwen3.5-4B | 결과 |
|---|---|---|---|
| GPQA Diamond | 73.4 | **76.2** | 4B > 30B |
| IFEval | 88.9 | **89.8** | 4B > 30B |
| LongBench v2 | 44.8 | **50.0** | 4B > 30B |

## 부록 C. 참고 자료

| ID | 자료 | URL |
|---|---|---|
| M1 | Qwen3.5 공식 GitHub | https://github.com/QwenLM/Qwen3.5 |
| M2 | Qwen3.5 공식 블로그 | https://qwen.ai/blog?id=qwen3.5 |
| M3 | Qwen3.5-4B HuggingFace 모델 카드 | https://huggingface.co/Qwen/Qwen3.5-4B |
| M4 | Unsloth Qwen3.5 Fine-tuning 가이드 | https://unsloth.ai/docs/models/qwen3.5/fine-tune |
| M5 | DataCamp QLoRA 튜토리얼 | https://www.datacamp.com/tutorial/fine-tuning-qwen3-5-small |
| M6 | vLLM Qwen3.5 Recipe | https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html |
| M7 | HF Forums: loss 폭발 스레드 | https://discuss.huggingface.co/t/qwen3-5-4b-loss-exploding/174057 |
| M8 | vLLM Issue #36217 (transformers v5 필요) | https://github.com/vllm-project/vllm/issues/36217 |
| M9 | QwenLM/Qwen3.5 Issue #33 (vocab 문제) | https://github.com/QwenLM/Qwen3.5/issues/33 |

## 부록 D. Iteration History (Visualization-Ready JSON, 확장)

```json
[
  {"iter": "Iter0", "model": "Qwen2.5-7B", "sft": "v0_boilerplate", "rag_only": 1.380, "qlora_only": 0.980, "rag_qlora": 1.148, "halluc_pct": 10.0, "agentic": null, "tc_made": null, "decision": "DROP"},
  {"iter": "Iter1", "model": "Qwen2.5-7B", "sft": "v1_diverse", "rag_only": 1.380, "qlora_only": 0.928, "rag_qlora": 1.376, "halluc_pct": 2.0, "agentic": null, "tc_made": null, "decision": "KEEP"},
  {"iter": "Iter3", "model": "Qwen3-4B", "sft": "v1_diverse", "rag_only": 1.288, "qlora_only": 0.864, "rag_qlora": 1.348, "halluc_pct": 18.0, "agentic": null, "tc_made": null, "decision": "REGEN"},
  {"iter": "Iter4", "model": "Qwen3-4B", "sft": "v2_qwen3_rag", "rag_only": 1.308, "qlora_only": 0.816, "rag_qlora": 1.368, "halluc_pct": 10.0, "agentic": 1.90, "tc_made": 0, "decision": "FC_NEEDED"},
  {"iter": "Iter5", "model": "Qwen3-8B", "sft": "v2_qwen3_rag", "rag_only": 1.328, "qlora_only": 0.872, "rag_qlora": 1.408, "halluc_pct": 8.0, "agentic": null, "tc_made": null, "decision": "ARCHIVED"},
  {"iter": "Iter6", "model": "Qwen3-4B", "sft": "v3_fc_aware", "rag_only": null, "qlora_only": null, "rag_qlora": null, "halluc_pct": null, "agentic": 2.60, "tc_made": 53, "decision": "KEEP_IMPROVE"},
  {"iter": "Iter7", "model": "Qwen3-4B", "sft": "v3_fc_aware_v2", "rag_only": null, "qlora_only": null, "rag_qlora": null, "halluc_pct": null, "agentic": 2.97, "tc_made": 60, "decision": "KEEP_IMPROVE"},
  {"iter": "Iter8", "model": "Qwen3.5-4B", "sft": "v4_qwen35_xml", "rag_only": null, "qlora_only": null, "rag_qlora": null, "halluc_pct": null, "agentic": null, "tc_made": null, "decision": "PENDING"}
]
```

## 부록 E. 마이그레이션 체크리스트

```
□ M-0.1: 현재 PoC 모든 Task 완료 확인
□ M-0.2: Qwen3.5-4B 모델 다운로드 → /models/Qwen3.5-4B/
□ M-0.3: transformers≥5 확인, unsloth 최신 설치
□ M-0.4: 기존 adapter/SFT 백업

□ M-1.1: migrate_sft_qwen35.py 작성
□ M-1.2: SFT 데이터 XML 변환 실행
□ M-1.3: 변환 결과 검증 (건수, 포맷)

□ M-2.1: tool_calling_adapter.py XML 파서
□ M-2.2: agent_pipeline.py Qwen3.5 호환
□ M-2.3: qa_pipeline.py Qwen3.5 호환
□ M-2.4: train_qlora.py bf16 LoRA 전환
□ M-2.5: config 파일 업데이트 (yaml, json)
□ M-2.6: generate_fc_sft.py XML 포맷
□ M-2.7: rag_config_fewshot.yaml XML few-shot

□ M-3.1: bf16 LoRA 학습 완료 (loss < 1.0)
□ M-3.2: 30문항 Agentic 평가 (≥4.0/10)
□ M-3.3: 50문항 docQA 평가 (≥1.3/2.0)
□ M-3.4: A/B 비교 리포트 작성
□ M-3.5: PATCH_NOTE Iter8 기록

□ M-4.1: Go/No-Go 의사결정
□ M-4.2: (Go) 기본 설정 Qwen3.5 전환
□ M-4.3: (Go) tasks.json 업데이트
□ M-4.4: (No-Go) Rollback + 원인 분석
```

---

*End of Document*
