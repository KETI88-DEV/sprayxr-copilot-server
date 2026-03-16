---
title: "선박 도장 AI Co-Pilot Phase 2 MVP PRD"
subtitle: "Agentic 점수 6.0+ 달성을 위한 추론 강화, 도구 고도화, VLM/Rule Engine 통합"
date: "2026-03-16"
lang: ko-KR
numbersections: true
geometry: margin=0.78in
---

## 목차

1. 문서 개요
2. 배경 및 Phase 1 PoC 성과 요약
3. Phase 2 목표 및 성공 기준
4. 상용화 목표 테이블
5. 우선순위별 작업 정의
6. P0 — 즉시 개선 (Week 1-2, 코드 전용)
7. P1 — Phase 2 핵심 (Week 3-4, 코드 + 데이터)
8. P2 — Phase 3 미래 (Week 5+, 코드 + 데이터)
9. 데이터 인벤토리 및 Gap 분석
10. 기술 아키텍처 변경 사항
11. 평가 체계 확장
12. TDD 전략 및 테스트 설계
13. 커밋 전략
14. 일정 및 공수 추정
15. 리스크 및 폴백 전략
16. 인터페이스 사양 참조
17. 문서 개정 이력

\newpage

# 1. 문서 개요

| 항목 | 내용 |
|---|---|
| 문서 목적 | Phase 1 PoC 완료 결과를 반영하여 **Phase 2 MVP (Minimum Viable Product)** 목표와 실행 계획을 정의 |
| 선행 조건 | Phase 1 PoC 완료 (18/19 Tasks done, Agentic 4.97/10, DocQA RAG+QLoRA 1.368/2.0, halluc 2%) |
| Phase 2 목표 | 1) Agentic 점수 6.0+/10 달성, 2) VLM 실제 추론 파이프라인 구축, 3) Rule Engine 연동, 4) MCP Stub 실구현 전환 |
| 기본 모델 | **Qwen3-4B** (Apache 2.0, QLoRA rank 16, 4-bit NF4 유지) |
| VLM 모델 | **Qwen2-VL-2B-Instruct** (히트맵/스크린샷 분석 전용, sidecar 배포) |
| 배포 환경 | 온프레미스 / RTX 4080 16GB (추론) / A40 48GB (학습) |
| 우선순위 체계 | P0 (코드 전용, 즉시 착수) → P1 (코드+데이터, 데이터 확보 후) → P2 (향후 확장) |
| 한 줄 결론 | P0 코드 최적화로 Agentic 6.0+을 달성하고, P1에서 VLM과 Rule Engine을 연결하여 실제 시뮬레이터 데이터 기반의 분석 역량을 확보한다 |

## 설계 원칙

1. **Phase 1 안정성 보존**: 기존 파이프라인(RAG+QLoRA+3-Way Router+6 MCP)을 깨지 않으면서 점진적으로 개선한다.
2. **데이터 의존성 분리**: P0 작업은 원시 데이터(raw data) 없이 즉시 착수 가능하며, P1 작업은 데이터 확보를 명시적으로 기다린다.
3. **TDD 우선**: 모든 모듈은 테스트를 먼저 작성하고, 테스트가 통과할 때까지 구현을 반복한다.
4. **Atomic Commit**: 각 기능 단위로 독립적인 커밋을 만들어 롤백 가능성을 보장한다.
5. **측정 가능한 개선**: 모든 작업의 성공 기준은 정량적 지표로 정의한다.

\newpage

# 2. 배경 및 Phase 1 PoC 성과 요약

## 2.1 Phase 1 최종 성과

| 항목 | 목표 | 달성 | 비고 |
|---|---|---|---|
| Agentic 평가 점수 | 4.0/10 | **4.97/10** | 목표 초과 달성 |
| DocQA RAG+QLoRA | — | **1.368/2.0** | Iter4 어댑터 기준, Iter14 재평가 진행중 |
| Hallucination | <10% | **2.0%** | 목표 대비 5배 양호 |
| Tool Calling | 최소 1개 호출 | **86회/30문항** | 평균 2.87 tools/question |
| MCP 도구 | 6개 구현 | **6개** (1 real + 5 stub) | dft_judge만 실제 로직 |
| RTX 4080 VRAM | <16GB | **~4GB (추론), ~8.5GB (학습)** | 여유 충분 |
| SFT 데이터 | 433건 | **567건** (dedup 후) | DocQA 380 + FC 187 |

## 2.2 Phase 1 차원별 분석 (Agentic 5차원)

| 차원 | 평균 점수 (0-2) | 분석 |
|---|---|---|
| tool_selection_accuracy | **0.67** | 8/30 질문(26.7%)에서 도구 호출 0건, 가장 큰 개선 여지 |
| parameter_correctness | **1.17** | 파라미터 타입 오류 간헐적 발생, 필수 필드 누락 사례 있음 |
| reasoning_depth | **0.77** | 최저 차원, AQ024 4-tool 순차 추론 완전 실패 (1/10) |
| context_integration | **1.60** | 비교적 양호, MCP 결과와 RAG chunk 통합 안정적 |
| response_structure | **0.77** | JSON 스키마 준수율 불안정, citation 누락 빈번 |

## 2.3 난이도별 분석

| 난이도 | 평균 점수 (/10) | 문항 수 | 분석 |
|---|---|---|---|
| Easy (단일 도구) | **5.67** | 6 | 비교적 안정 |
| Medium (2-3 도구) | **4.09** | 11 | 파라미터 조합 및 순차 호출 약점 |
| Hard (4+ 도구) | **5.38** | 13 | 의외로 medium보다 높음, 변동폭 큼 |

## 2.4 잔여 작업

| 항목 | 상태 | 비고 |
|---|---|---|
| Task 19: vLLM 서빙 계층 | Deferred | RTX 4080 OOM, A40/L4급 전용 |
| Iter14 DocQA 3-Way 재평가 | **완료** | 결과 아래 참조 |

### 2.4.1 Iter14 DocQA 3-Way 재평가 결과 (Critical Finding)

| Config | Iter4 Score | Iter14 Score | Delta | Iter4 Halluc | Iter14 Halluc |
|---|---|---|---|---|---|
| RAG-Only | 1.380 | 1.288 | -0.092 | 10% | 4% |
| QLoRA-Only | 0.980 | 0.536 | -0.444 | — | 20% |
| **RAG+QLoRA** | **1.368** | **0.956** | **-0.412** | **2%** | **38%** |

**결정: DROP adapter (DocQA 기준)**

**원인 분석**:
1. **SFT 데이터 비율 불균형**: FC(Function Calling) 데이터 187건/567건(33%)으로 도구 호출 학습에 편향, DocQA 응답 능력 퇴화
2. **assistant_only_loss 부작용**: 어시스턴트 발화만 loss를 적용하여 Agentic 성능(4.97/10)은 향상되었으나, DocQA 스타일의 자연어 답변 품질이 급락
3. **Catastrophic Forgetting**: FC 특화 학습이 기존 DocQA 지식을 덮어씀

**영향**: Phase 2에서 Agentic 성능 개선과 DocQA 품질 유지를 동시에 달성하는 **균형 학습 전략**이 필수 (T26에서 해결)

\newpage

# 3. Phase 2 목표 및 성공 기준

## 3.1 Phase 2 핵심 목표

| 목표 | 현재 값 | Phase 2 목표 | 측정 방법 |
|---|---|---|---|
| Agentic 전체 점수 | 4.97/10 | **6.0+/10** | eval_set_agentic.jsonl 30문항 |
| **DocQA RAG+QLoRA 점수** | **0.956/2.0** | **≥1.3/2.0** | eval_set.jsonl 50문항 (Iter14 기준 회복) |
| **DocQA Hallucination** | **38%** | **≤5%** | RAG+QLoRA halluc rate (Iter14 기준 회복) |
| tool_selection_accuracy | 0.67/2.0 | **1.2+/2.0** | 도구 호출 정확도 |
| reasoning_depth | 0.77/2.0 | **1.2+/2.0** | CoT 단계 수 및 논리 연결 |
| response_structure | 0.77/2.0 | **1.3+/2.0** | JSON 스키마 준수율 |
| parameter_correctness | 1.17/2.0 | **1.5+/2.0** | 파라미터 타입/값 정확도 |
| Medium 난이도 점수 | 4.09/10 | **5.5+/10** | Medium 문항 11개 |
| VLM 추론 (신규) | N/A (Stub) | **실제 추론 작동** | 히트맵 분석 성공률 |
| Rule Engine 연동 (신규) | N/A | **점수 변환 정확도 95%+** | RealCOAT 대비 일관성 |
| MCP Stub 전환 | 1/6 real | **4/6+ real** | 실제 로직 구현 완료 수 |
| SFT 데이터 규모 | 567건 | **800+건** | DocQA 비율 ≥60% 유지, CoT/Tool Selection 보강분 포함 |
| Hallucination (Agentic) | 2% | **<2%** | 유지 또는 개선 |

## 3.2 Phase 2 비목표 (Explicit Non-Goals)

아래 항목은 Phase 2 범위에 포함하지 않는다.

- Qwen3.5 마이그레이션 (별도 PRD: `qwen35_migration_prd.md`)
- RLHF/DPO 학습 (Phase 3, P2 우선순위)
- 음성 인터페이스 (Phase 3, P2 우선순위)
- Kubernetes 프로덕션 배포 (Phase 3, P2 우선순위)
- vLLM 서빙 계층 구축 (RTX 4080 OOM, 서버급 전용)

\newpage

# 4. 상용화 목표 테이블

| 단계 | Agentic 점수 | 주요 기능 | 상태 |
|---|---|---|---|
| **PoC** | 4.0/10 | RAG+QLoRA, 3-Way Router, 6 MCP (1 real + 5 stub) | **✅ 달성 (4.97)** |
| **MVP** | **6.0/10** | CoT 강화, Tool Selection 최적화, VLM 추론, Rule Engine, 4+ real MCP | **← Phase 2 목표** |
| **Production** | **7.5/10** | RLHF/DPO, Qdrant, 음성 인터페이스, K8s 배포, 모니터링 | Phase 3 (향후) |
| **Advanced** | **8.5+/10** | 멀티모달 sLMM 통합, 실시간 센서 학습, 자동 교육 과정 생성 | Phase 4+ (장기) |

\newpage

# 5. 우선순위별 작업 정의

## 5.1 우선순위 분류 기준

| 우선순위 | 기준 | 데이터 의존성 | 착수 시점 |
|---|---|---|---|
| **P0** | 기존 코드/데이터만으로 즉시 개선 가능, Agentic 점수 직접 영향 | 없음 (코드 전용) | Week 1-2 즉시 |
| **P1** | Phase 2 핵심 목표 달성에 필수, 외부 데이터 확보 필요 | 있음 (코드+데이터) | Week 3-4 (데이터 확보 후) |
| **P2** | Phase 3 이후 확장 기능, 장기 로드맵 | 있음/없음 혼합 | Week 5+ |

## 5.2 작업 매핑 요약

| ID | 작업명 | 우선순위 | 데이터 필요 | 영향 차원 | 예상 공수 |
|---|---|---|---|---|---|
| T20 | DocQA 3-Way 재평가 (Iter14) | P0 | ❌ | DocQA 기준선 | 0.5일 |
| T21 | CoT 추론 강화 | P0 | ❌ | reasoning_depth | 2일 |
| T22 | 도구 선택 정확도 개선 | P0 | ❌ | tool_selection_accuracy | 2일 |
| T23 | 응답 구조 표준화 | P0 | ❌ | response_structure | 1.5일 |
| T24 | 파라미터 정확도 개선 | P0 | ❌ | parameter_correctness | 1일 |
| T25 | Medium 난이도 갭 해소 | P0 | ❌ | Medium 점수 | 1.5일 |
| T26 | P0 통합 회귀 평가 및 QLoRA 재학습 (DocQA 균형 학습) | P0 | ❌ | 전체 Agentic + DocQA 복구 | 2.5일 |
| T27 | VLM 통합 (Qwen2-VL-2B) | P1 | ✅ | xr_content, anomaly | 3일 |
| T28 | Rule Engine 통합 | P1 | ✅ | sequence_decision | 2일 |
| T29 | Session Export 자동화 | P1 | ✅ | 운영 효율 | 2일 |
| T30 | MCP Stub 실구현 전환 | P1 | ✅ | 도구 정확도 전체 | 3일 |
| T31 | RLHF/DPO 학습 준비 | P2 | ✅ | hallucination | 5일+ |
| T32 | Qdrant 벡터 DB 마이그레이션 | P2 | ❌ | 확장성 | 2일 |
| T33 | 음성 인터페이스 (Whisper STT/TTS) | P2 | ✅ | UX, 접근성 | 5일+ |
| T34 | 프로덕션 배포 (K8s) | P2 | ❌ | 운영 | 5일+ |

\newpage

# 6. P0 — 즉시 개선 (Week 1-2, 코드 전용)

## 6.1 T20: DocQA 3-Way 재평가 (Iter14 어댑터) — **완료**

| 항목 | 내용 |
|---|---|
| 목표 | Iter14 어댑터 기준으로 RAG-Only / QLoRA-Only / RAG+QLoRA 3-Way 비교 완료 |
| 현재 상태 | **완료** — 심각한 DocQA 성능 저하 확인됨 |
| 결과 요약 | RAG+QLoRA 점수 1.368→0.956 (-30%), halluc 2%→38% (19배 증가). Decision: DROP adapter |
| 결과 위치 | `/output/eval_results/docqa_eval_iter14/comparison_analysis.md` |
| 후속 조치 | **T26에서 균형 학습 전략으로 DocQA 성능 복구 필수** |

### T20 실행 결과

| Config | Avg Score | ROUGE-L | BERTScore F1 | Halluc Rate |
|---|---|---|---|---|
| RAG-Only | 1.288 | 0.1051 | 0.7017 | 4.0% |
| QLoRA-Only | 0.536 | 0.0240 | 0.5798 | 20.0% |
| RAG+QLoRA | 0.956 | 0.0455 | 0.5912 | **38.0%** |

**핵심 발견**: Iter14 어댑터는 Agentic 성능(4.97/10)에 특화되어 있으나, FC 데이터 편향(33%)과 assistant_only_loss로 인해 DocQA 답변 품질이 Catastrophic Forgetting을 일으켰다. T26에서 반드시 해결해야 한다.

## 6.2 T21: CoT 추론 강화

| 항목 | 내용 |
|---|---|
| 목표 | reasoning_depth 차원 0.77/2.0 → **1.2+/2.0** |
| 문제 분석 | AQ024 (4-tool 순차 추론) 1/10 완전 실패, 단계적 사고 부재, 결론만 제시하는 패턴 반복 |
| 작업 내용 | 1) SFT 데이터에 step-by-step CoT 예시 80건+ 추가 생성 (**DocQA 스타일 CoT 30건 포함**), 2) few-shot 프롬프트에 `<think>` 블록 활용 CoT 템플릿 추가, 3) multi-tool sequential reasoning 전용 SFT 샘플 20건 추가, 4) CoT 강제 생성 규칙 (thinking_budget) 조정 |
| 필요 데이터 | ❌ 없음 — 기존 chunks.jsonl 기반 synthetic 생성 |
| 성공 기준 | reasoning_depth 평균 1.2+/2.0, AQ024 유형 4+/10 |
| DocQA 균형 | CoT SFT 80건 중 **DocQA 스타일(도구 없이 문서 근거만으로 추론) 30건 이상** 포함하여 DocQA 능력 유지 |
| 예상 공수 | 2일 |
| 선행 작업 | T20 (기준선 확보) |

### 작업 세부 사항

#### 6.2.1 CoT SFT 데이터 생성 스크립트

`/workspace/trainer/scripts/generate_cot_sft.py` 신규 생성:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "너는 선박 도장 전문 AI Co-Pilot이다. 복잡한 질문에는 반드시 단계별로 사고하고, 각 단계의 근거를 명시하라."
    },
    {
      "role": "user",
      "content": "[문서]\n에폭시 프라이머 DFT 125-200µm...\n[질문]\n작업자의 DFT가 [102, 95, 110, 88, 105]이고 기준이 100µm일 때, 합부 판정 및 원인 분석을 해줘."
    },
    {
      "role": "assistant",
      "content": "<think>\n단계 1: DFT 측정값 분석 - 5개 측정값 중 88µm가 기준 100µm의 90%(90µm) 미만\n단계 2: 90/10 룰 적용 - dft_judge 도구 호출 필요\n단계 3: 원인 추론 - 88µm 저도막 발생 → 이동 속도 과다 또는 이격거리 과다 가능성\n단계 4: 문서 근거 확인 - DOC-COAT-0007에서 권장 이격거리 250-350mm\n</think>\n{\"text_guide\": {\"summary\": \"90/10 룰 불합격, 저도막 1건\", \"body\": \"단계별 분석 결과...\"}, \"dft_result\": {\"verdict\": \"fail\", \"out_of_spec_count\": 1}, \"citations\": [\"DOC-COAT-0007-C012\"]}"
    }
  ]
}
```

#### 6.2.2 CoT 카테고리별 목표 분포

| 카테고리 | 생성 목표 | 설명 |
|---|---|---|
| **DocQA 스타일 CoT (도구 없음)** | **30건** | **문서 근거만으로 단계별 추론 — DocQA 능력 유지 목적** |
| 단일 도구 + CoT | 15건 | 도구 호출 전 사고 과정 명시 |
| 2-3 도구 순차 CoT | 15건 | 도구 간 의존성 추론 |
| 4+ 도구 복합 CoT | 20건 | AQ024 유형 대응 |
| 합계 | **80건** | DocQA:FC = 30:50 (37.5% DocQA) |

### 테스트 전략

```python
# test_t21_cot_enhancement.py
def test_cot_sft_data_count():
    """CoT SFT 데이터가 80건 이상 생성되었는지 확인"""
    data = load_jsonl("/data/sft/sft_cot.jsonl")
    assert len(data) >= 80

def test_cot_sft_has_think_block():
    """모든 CoT SFT 샘플에 <think> 블록이 포함되어 있는지 확인"""
    data = load_jsonl("/data/sft/sft_cot.jsonl")
    for item in data:
        assistant_msg = item["messages"][-1]["content"]
        assert "<think>" in assistant_msg and "</think>" in assistant_msg

def test_cot_multitool_samples():
    """4+ 도구 CoT 샘플이 20건 이상인지 확인"""
    data = load_jsonl("/data/sft/sft_cot.jsonl")
    multitool = [d for d in data if d.get("tool_count", 0) >= 4]
    assert len(multitool) >= 20

def test_reasoning_depth_improvement():
    """CoT 강화 후 reasoning_depth가 1.2 이상인지 확인"""
    results = run_agentic_eval()
    assert results["dimensions"]["reasoning_depth"]["mean"] >= 1.2
```

## 6.3 T22: 도구 선택 정확도 개선

| 항목 | 내용 |
|---|---|
| 목표 | tool_selection_accuracy 차원 0.67/2.0 → **1.2+/2.0** |
| 문제 분석 | 30문항 중 8건(26.7%)에서 도구 호출 0건, LLM이 도구 없이 직접 답변 시도 |
| 작업 내용 | 1) `rag_config_fewshot.yaml`의 few-shot 프롬프트에 명시적 도구 선택 규칙 추가, 2) 도구 선택 전용 SFT 데이터 40건 생성 (의도→도구 매핑 학습), 3) 도구 선택 신뢰도 임계값(confidence threshold) 구현, 4) intent-to-tool 매핑 테이블 강화 |
| 필요 데이터 | ❌ 없음 — 프롬프트 엔지니어링 + SFT 증강 |
| 성공 기준 | tool_selection_accuracy 평균 1.2+/2.0, 도구 호출 0건 비율 <10% |
| 예상 공수 | 2일 |
| 선행 작업 | T20 (기준선 확보) |

### 작업 세부 사항

#### 6.3.1 도구 선택 규칙 프롬프트 추가

`rag_config_fewshot.yaml`에 추가할 규칙:

```text
## 도구 선택 필수 규칙
1. DFT/두께 측정값이 언급되면 → 반드시 dft_judge 호출
2. 히트맵/분포/이상 키워드 → anomaly_detection 호출
3. 자세/관절/포즈 키워드 → pose_estimation 호출
4. 구역/위치/좌표 키워드 → location_tracking 호출
5. 순서/방향/시퀀스 키워드 → sequence_validation 호출
6. 속도/거리/추세 키워드 → timeseries 호출
7. 수치 판단이 필요한 질문에는 도구 없이 답하지 마라
```

#### 6.3.2 도구 선택 SFT 데이터 구조

| 유형 | 건수 | 설명 |
|---|---|---|
| 단일 의도 → 단일 도구 | 15건 | 명확한 1:1 매핑 학습 |
| 복합 의도 → 다중 도구 | 15건 | 2-3 도구 조합 학습 |
| 부정 예시 (도구 불필요) | 10건 | fast 경로 학습 (도구 호출 생략) |
| 합계 | **40건** | |

### 테스트 전략

```python
# test_t22_tool_selection.py
def test_tool_selection_accuracy_improved():
    """tool_selection_accuracy 평균이 1.2 이상인지 확인"""
    results = run_agentic_eval()
    assert results["dimensions"]["tool_selection_accuracy"]["mean"] >= 1.2

def test_zero_tool_call_rate():
    """도구 호출 0건 비율이 10% 미만인지 확인"""
    results = run_agentic_eval()
    zero_count = sum(1 for r in results["items"] if r["tool_calls"] == 0)
    assert zero_count / len(results["items"]) < 0.10

def test_tool_selection_sft_count():
    """도구 선택 SFT 데이터가 40건 이상인지 확인"""
    data = load_jsonl("/data/sft/sft_tool_selection.jsonl")
    assert len(data) >= 40

def test_tool_selection_confidence_threshold():
    """confidence threshold 미만 시 fallback 동작 확인"""
    result = agent_pipeline.process(query="일반 질문", confidence_threshold=0.5)
    # 도구 호출이 없어야 하는 케이스에서 도구를 호출하지 않는지 확인
    assert result["metadata"]["route"] == "fast"
```

## 6.4 T23: 응답 구조 표준화

| 항목 | 내용 |
|---|---|
| 목표 | response_structure 차원 0.77/2.0 → **1.3+/2.0** |
| 문제 분석 | JSON 스키마 준수 불안정, citation 누락 빈번, 카테고리별 응답 형식 불일치 |
| 작업 내용 | 1) 도구 카테고리별 구조화된 응답 템플릿 구현, 2) `response_formatter.py`에 응답 형식 검증 로직 추가, 3) JSON 스키마 유효성 검사 post-processing 단계 추가, 4) citation 필드 필수 강제 규칙 추가 |
| 필요 데이터 | ❌ 없음 — 코드 개선 |
| 성공 기준 | response_structure 평균 1.3+/2.0, citation 누락률 <5% |
| 예상 공수 | 1.5일 |
| 선행 작업 | 없음 (T20과 병렬 가능) |

### 작업 세부 사항

#### 6.4.1 응답 템플릿 JSON Schema

```json
{
  "training_realtime_template": {
    "required": ["text_guide", "steps", "citations", "metadata"],
    "optional": ["dft_result", "anomaly", "next_path"],
    "text_guide": {
      "required": ["language", "detail_level", "summary", "body"]
    }
  },
  "training_review_template": {
    "required": ["text_guide", "dft_result", "citations", "metadata"],
    "optional": ["pose_comparison", "anomaly", "steps"]
  },
  "fast_template": {
    "required": ["text_guide", "citations", "metadata"],
    "optional": ["steps"]
  }
}
```

#### 6.4.2 response_formatter.py 확장

```python
# 추가할 검증 함수 시그니처
def validate_response_structure(response: dict, route: str) -> ValidationResult:
    """응답이 해당 route의 템플릿을 준수하는지 검증"""
    pass

def ensure_citations(response: dict) -> dict:
    """citations 필드가 비어있으면 RAG 검색 결과에서 자동 채움"""
    pass

def repair_json_structure(raw_text: str) -> dict:
    """불완전한 JSON을 수복하여 유효한 구조로 변환"""
    pass
```

### 테스트 전략

```python
# test_t23_response_structure.py
def test_response_structure_score():
    """response_structure 점수가 1.3 이상인지 확인"""
    results = run_agentic_eval()
    assert results["dimensions"]["response_structure"]["mean"] >= 1.3

def test_citation_coverage():
    """citation 누락률이 5% 미만인지 확인"""
    results = run_agentic_eval()
    missing = sum(1 for r in results["items"] if not r.get("citations"))
    assert missing / len(results["items"]) < 0.05

def test_json_schema_validation():
    """모든 route별 응답이 해당 JSON 스키마를 통과하는지 확인"""
    for route in ["training_realtime", "training_review", "fast"]:
        response = generate_sample_response(route)
        result = validate_response_structure(response, route)
        assert result.is_valid, f"Route {route} failed: {result.errors}"

def test_json_repair():
    """불완전한 JSON이 정상 수복되는지 확인"""
    broken = '{"text_guide": {"summary": "test"'
    repaired = repair_json_structure(broken)
    assert isinstance(repaired, dict)
```

## 6.5 T24: 파라미터 정확도 개선

| 항목 | 내용 |
|---|---|
| 목표 | parameter_correctness 차원 1.17/2.0 → **1.5+/2.0** |
| 문제 분석 | 간헐적 타입 오류 (string vs array), 필수 필드 누락, 값 범위 초과 |
| 작업 내용 | 1) `tool_calling_adapter.py`에 파라미터 타입 검증 로직 추가, 2) few-shot에 파라미터 추출 예시 강화, 3) 도구별 필수 파라미터 체크 및 디폴트값 자동 채움 |
| 필요 데이터 | ❌ 없음 — 코드 개선 |
| 성공 기준 | parameter_correctness 평균 1.5+/2.0, 타입 오류 0건 |
| 예상 공수 | 1일 |
| 선행 작업 | 없음 (T20과 병렬 가능) |

### 작업 세부 사항

#### 6.5.1 tool_calling_adapter.py 확장

```python
# 추가할 검증 함수 시그니처
def validate_tool_parameters(tool_name: str, params: dict) -> ValidatedParams:
    """도구 스키마 대비 파라미터 타입/필수 필드 검증 및 자동 수정"""
    pass

def coerce_parameter_types(tool_name: str, params: dict) -> dict:
    """잘못된 타입을 자동 변환 (예: string '102' → number 102)"""
    pass

def fill_default_parameters(tool_name: str, params: dict) -> dict:
    """누락된 선택 파라미터에 디폴트값 채움"""
    pass
```

### 테스트 전략

```python
# test_t24_parameter_correctness.py
def test_parameter_type_validation():
    """잘못된 타입이 감지되고 자동 수정되는지 확인"""
    bad_params = {"measurements_um": "102, 95, 110", "nominal_um": "100"}
    result = validate_tool_parameters("dft_judge", bad_params)
    assert isinstance(result.params["measurements_um"], list)
    assert isinstance(result.params["nominal_um"], (int, float))

def test_required_parameter_check():
    """필수 파라미터 누락 시 에러 반환 확인"""
    incomplete = {"nominal_um": 100}  # measurements_um 누락
    result = validate_tool_parameters("dft_judge", incomplete)
    assert not result.is_valid
    assert "measurements_um" in result.missing_fields

def test_parameter_correctness_score():
    """parameter_correctness 점수가 1.5 이상인지 확인"""
    results = run_agentic_eval()
    assert results["dimensions"]["parameter_correctness"]["mean"] >= 1.5
```

## 6.6 T25: Medium 난이도 갭 해소

| 항목 | 내용 |
|---|---|
| 목표 | Medium 난이도 점수 4.09/10 → **5.5+/10** |
| 문제 분석 | Medium 문항은 2-3 도구 호출 + 파라미터 조합이 필요, 도구 간 의존성 추론 약점 |
| 작업 내용 | 1) Medium 복잡도 멀티 도구 SFT 예시 30건 생성, 2) 도구 순차 호출 체인(chain) 패턴 학습, 3) 파라미터 전달 (이전 도구 결과 → 다음 도구 입력) SFT 추가 |
| 필요 데이터 | ❌ 없음 — synthetic 생성 |
| 성공 기준 | Medium 평균 5.5+/10 |
| 예상 공수 | 1.5일 |
| 선행 작업 | T21, T22 (CoT 및 도구 선택 개선 후) |

### 테스트 전략

```python
# test_t25_medium_gap.py
def test_medium_difficulty_score():
    """Medium 난이도 평균 점수가 5.5 이상인지 확인"""
    results = run_agentic_eval()
    medium_scores = [r["score"] for r in results["items"] if r["difficulty"] == "medium"]
    assert sum(medium_scores) / len(medium_scores) >= 5.5

def test_medium_sft_data_count():
    """Medium 복잡도 SFT 데이터가 30건 이상인지 확인"""
    data = load_jsonl("/data/sft/sft_medium.jsonl")
    assert len(data) >= 30

def test_tool_chain_pattern():
    """도구 순차 호출 패턴이 정상 동작하는지 확인"""
    result = agent_pipeline.process("B-3 구역의 DFT를 확인하고 이상이 있으면 원인을 분석해줘")
    tools_used = result["metadata"]["tools_used"]
    assert len(tools_used) >= 2
    assert "dft_judge" in tools_used
```

## 6.7 T26: P0 통합 회귀 평가 및 QLoRA 재학습 (DocQA 균형 학습 포함)

| 항목 | 내용 |
|---|---|
| 목표 | P0 모든 개선 사항 통합 후 QLoRA 재학습. **Agentic 6.0+/10 달성과 DocQA 성능 복구를 동시에 확인** |
| 핵심 문제 | Iter14에서 FC 데이터 편향(33%) + assistant_only_loss로 DocQA halluc 2%→38% 급증. Agentic과 DocQA 간 Catastrophic Forgetting 발생 |
| 작업 내용 | 아래 6.7.1~6.7.5 참조 |
| 필요 데이터 | ❌ 없음 — P0 산출물 사용 |
| 성공 기준 | **Dual Gate**: (1) Agentic 6.0+/10 AND (2) DocQA RAG+QLoRA ≥1.3/2.0, halluc ≤5%. **둘 다 충족해야 PASS** |
| 예상 공수 | 2.5일 |
| 선행 작업 | T20, T21, T22, T23, T24, T25 전부 완료 |

### 작업 세부 사항

#### 6.7.1 SFT 데이터 비율 균형 조정

Iter14 실패 원인: FC 데이터 33%(187/567)로 DocQA 학습 능력 상실.
Phase 2에서는 DocQA 비율을 **최소 60%** 이상으로 유지한다.

| 카테고리 | Iter14 (현재) | Iter15 목표 | 비율 |
|---|---|---|---|
| DocQA 기반 | 380건 | 380건+ | ≥60% |
| CoT 추론 (T21) | — | 80건 | ~10% |
| Tool Selection (T22) | — | 40건 | ~5% |
| Medium 복합 (T25) | — | 30건 | ~4% |
| FC 도구 호출 | 187건 | 160건 (중복 제거+필터) | ≤20% |
| **합계** | 567건 | **~690건+** | 100% |

**FC 데이터 비율 상한**: 전체 SFT 데이터의 **20% 이하**로 제한. 초과 시 품질 기준으로 필터링.

#### 6.7.2 Loss 전략 분기

| SFT 데이터 유형 | assistant_only_loss | 근거 |
|---|---|---|
| DocQA (문서 기반 QA) | **False** (전체 토큰 loss) | DocQA는 시스템 프롬프트+문서 컨텍스트도 학습 필요 |
| FC (Tool Calling) | **True** (어시스턴트만 loss) | FC는 도구 호출 형식 학습에 집중 |
| CoT / Medium | **True** (어시스턴트만 loss) | 추론 체인 학습에 집중 |

구현: `train_qlora.py`에 `loss_strategy` 파라미터 추가하거나, 데이터 유형별 `{% generation %}` 마커를 분기 적용.

#### 6.7.3 SFT 데이터 통합 스크립트

```bash
# SFT 데이터 통합 (비율 검증 + 중복 제거 포함)
python3 /workspace/trainer/scripts/merge_sft.py \
  --base /data/sft/sft_train_docqa_only.jsonl \
  --additions /data/sft/sft_cot.jsonl \
              /data/sft/sft_tool_selection.jsonl \
              /data/sft/sft_medium.jsonl \
              /data/sft/sft_fc_filtered.jsonl \
  --output /data/sft/sft_train_v5.jsonl \
  --dedup \
  --max-fc-ratio 0.20 \
  --min-docqa-ratio 0.60
```

#### 6.7.4 QLoRA Iter15 학습

```bash
python3 /workspace/trainer/scripts/train_qlora.py \
  --config /workspace/trainer/configs/qlora_config.yaml \
  --data /data/sft/sft_train_v5.jsonl \
  --output /output/adapters/qwen3_4b_shipcoat_lora_iter15/
```

#### 6.7.5 Dual Gate 평가 (Agentic + DocQA 동시 검증)

Iter15 학습 후 **반드시** 아래 두 평가를 모두 수행하고, 둘 다 기준을 충족해야 PASS로 판정한다.

**Gate 1: Agentic 평가**
```bash
python3 trainer/scripts/generate_agentic_responses.py \
  --config trainer/configs/rag_config_fewshot.yaml \
  --adapter /output/adapters/qwen3_4b_shipcoat_lora_iter15 \
  --output /output/eval_results/agentic_responses_iter15.jsonl

python3 trainer/scripts/run_agentic_eval.py \
  --responses /output/eval_results/agentic_responses_iter15.jsonl \
  --output-dir /output/eval_results/agentic_eval_iter15
```

**Gate 2: DocQA 3-Way 재평가**
```bash
python3 trainer/scripts/run_qlora_eval.py \
  --config trainer/configs/rag_config.yaml \
  --mode rag-qlora \
  --adapter-dir /output/adapters/qwen3_4b_shipcoat_lora_iter15 \
  --output-dir /output/eval_results/docqa_eval_iter15
```

| Gate | 지표 | 기준 | FAIL 시 조치 |
|---|---|---|---|
| Gate 1 | Agentic 점수 | ≥6.0/10 | CoT/Tool Selection SFT 추가 생성 후 재학습 |
| Gate 1 | 모든 차원 | ≥1.0/2.0 | 최저 차원 집중 보강 |
| Gate 2 | DocQA manual_score | ≥1.3/2.0 | DocQA SFT 비율 70%로 상향 후 재학습 |
| Gate 2 | DocQA halluc rate | ≤5% | FC 데이터 비율 15%로 하향 또는 FC 품질 필터링 강화 |
| Gate 2 | DocQA ROUGE-L | ≥0.08 | 응답 길이/형식 프롬프트 조정 |

**FAIL 시 롤백**: 둘 중 하나라도 FAIL이면 Iter14 어댑터로 롤백 후 SFT 비율 재조정하여 재학습.

### 테스트 전략

```python
# test_t26_integration.py
def test_agentic_score_target():
    """[Gate 1] Agentic 점수가 6.0 이상인지 확인"""
    results = run_agentic_eval(adapter="iter15")
    assert results["overall_mean"] >= 6.0

def test_all_dimensions_above_threshold():
    """[Gate 1] 모든 차원이 1.0 이상인지 확인"""
    results = run_agentic_eval(adapter="iter15")
    for dim, score in results["dimensions"].items():
        assert score["mean"] >= 1.0, f"{dim} scored {score['mean']} < 1.0"

def test_docqa_score_recovery():
    """[Gate 2] DocQA RAG+QLoRA 점수가 1.3 이상으로 복구되었는지 확인"""
    results = run_qlora_eval(mode="rag-qlora", adapter="iter15")
    assert results["avg_manual_score"] >= 1.3

def test_docqa_hallucination_recovery():
    """[Gate 2] DocQA halluc이 5% 이하로 복구되었는지 확인"""
    results = run_qlora_eval(mode="rag-qlora", adapter="iter15")
    assert results["hallucination_rate"] <= 0.05

def test_docqa_rouge_l():
    """[Gate 2] DocQA ROUGE-L이 0.08 이상인지 확인"""
    results = run_qlora_eval(mode="rag-qlora", adapter="iter15")
    assert results["avg_rouge_l"] >= 0.08

def test_sft_data_ratio():
    """SFT 데이터에서 DocQA 비율이 60% 이상인지 확인"""
    data = load_jsonl("/data/sft/sft_train_v5.jsonl")
    docqa_count = sum(1 for d in data if d.get("category_type") != "fc")
    assert docqa_count / len(data) >= 0.60

def test_sft_fc_ratio_cap():
    """SFT 데이터에서 FC 비율이 20% 이하인지 확인"""
    data = load_jsonl("/data/sft/sft_train_v5.jsonl")
    fc_count = sum(1 for d in data if d.get("category_type") == "fc")
    assert fc_count / len(data) <= 0.20

def test_sft_data_total_count():
    """통합 SFT 데이터가 690건 이상인지 확인"""
    data = load_jsonl("/data/sft/sft_train_v5.jsonl")
    assert len(data) >= 690
```

\newpage

# 7. P1 — Phase 2 핵심 (Week 3-4, 코드 + 데이터)

## 7.1 T27: VLM 통합 (Qwen2-VL-2B-Instruct)

| 항목 | 내용 |
|---|---|
| 목표 | `vlm_analysis` 스텁을 실제 Qwen2-VL-2B-Instruct 추론 파이프라인으로 교체 |
| 모델 | Qwen2-VL-2B-Instruct (약 4.5GB, 4-bit 시 ~2GB VRAM) |
| 작업 내용 | 1) VLM 추론 파이프라인 구현 (`vlm_pipeline.py`), 2) 히트맵 이미지 → ROI 추출 → 텍스트 설명 변환, 3) MCP `anomaly_detection` 도구에 VLM 결과 연계, 4) VRAM 예산 관리 (LLM 4GB + VLM 2GB = 6GB < 16GB), 5) 추론 지연 시간 벤치마크 |
| 성공 기준 | 히트맵 분석 성공률 80%+, 추론 지연 <5초/이미지, VRAM 총합 <10GB |
| 예상 공수 | 3일 |
| 선행 작업 | T26 (P0 완료) |

### 필요 데이터

| 데이터 유형 | 포맷 | 예상 규모 | 출처 | 차단 여부 |
|---|---|---|---|---|
| 히트맵 이미지 | PNG, 512x512+ | 50-100장 | RealCOAT 시뮬레이터 | 부분 차단 (기존 71장으로 시작 가능) |
| XR 시뮬레이터 스크린샷 | PNG/JPG | 20-30장 | RealCOAT | 차단 (확보 필요) |
| 도막 분포 Ground Truth | JSON (zone별 thickness) | 50세트 | 도메인 전문가 라벨링 | 차단 (확보 필요) |

### 아키텍처

```text
[히트맵 이미지] → [Qwen2-VL-2B-Instruct]
                         │
                    VLM 분석 결과 (JSON)
                    ├── roi_regions: [{zone_id, bbox, severity}]
                    ├── distribution_summary: "좌하단 저도막 패턴"
                    └── anomaly_flags: [{type, confidence}]
                         │
                    [anomaly_detection MCP Tool]
                         │
                    [Agent Pipeline 통합]
```

### VLM 출력 스키마

```json
{
  "vlm_analysis": {
    "image_id": "heatmap_session_001",
    "resolution": [512, 512],
    "roi_regions": [
      {
        "zone_id": "B-3-L",
        "bbox": [100, 200, 250, 350],
        "mean_intensity": 0.35,
        "severity": "warning",
        "description": "좌하단 영역에 저도막 패턴 감지"
      }
    ],
    "distribution_summary": "도막 두께 분포가 좌측 하단에 집중되어 있으며, 우측 상단은 과도막 경향",
    "overall_quality": "fair",
    "confidence": 0.82
  }
}
```

### 테스트 전략

```python
# test_t27_vlm_integration.py
def test_vlm_model_loads():
    """Qwen2-VL-2B-Instruct가 정상 로딩되는지 확인"""
    model = load_vlm_model("Qwen/Qwen2-VL-2B-Instruct")
    assert model is not None

def test_vlm_heatmap_analysis():
    """히트맵 이미지 분석 결과가 유효한 JSON인지 확인"""
    result = vlm_analyze(image_path="/data/raw/html/sample_heatmap.png")
    assert "roi_regions" in result
    assert "distribution_summary" in result

def test_vlm_vram_budget():
    """VLM + LLM 동시 로드 시 VRAM이 10GB 미만인지 확인"""
    vram_usage = measure_vram_with_both_models()
    assert vram_usage < 10_000  # MB

def test_vlm_latency():
    """단일 이미지 VLM 추론이 5초 이내인지 확인"""
    start = time.time()
    vlm_analyze(image_path="/data/raw/html/sample_heatmap.png")
    assert time.time() - start < 5.0
```

## 7.2 T28: Rule Engine 통합

| 항목 | 내용 |
|---|---|
| 목표 | RealCOAT Rule Engine 점수를 에이전트 파이프라인에 연결 |
| 인터페이스 사양 | `/workspace/trainer/docs/rule_engine_interface.md` |
| 작업 내용 | 1) Rule Engine 입력/출력 어댑터 구현 (`rule_engine_adapter.py`), 2) 표준 점수 스키마(0-100) 변환, 3) 에이전트 파이프라인에 점수 기반 코칭 생성 연동, 4) 점수 임계값 기반 자동 경고/조치 생성 |
| 성공 기준 | 점수 변환 정확도 95%+, 코칭 생성 응답 시간 <2초 |
| 예상 공수 | 2일 |
| 선행 작업 | T26 (P0 완료) |

### 필요 데이터

| 데이터 유형 | 포맷 | 예상 규모 | 출처 | 차단 여부 |
|---|---|---|---|---|
| 모션 데이터 | JSON (timestamp, speed, distance, angle) | 10세션+ | RealCOAT | 차단 (확보 필요) |
| 두께 포인트 | JSON (zone_id, x, y, dft_um) | 10세션+ | RealCOAT | 차단 (확보 필요) |
| 시퀀스 로그 | JSON (timestamp, zone_id, action, order_index) | 10세션+ | RealCOAT | 차단 (확보 필요) |

> **합성 데이터로 시작 가능**: 인터페이스 사양 기반으로 합성 테스트 데이터를 먼저 생성하여 개발을 시작하고, 실제 RealCOAT 데이터가 확보되면 검증한다.

### Rule Engine 어댑터 아키텍처

```text
[RealCOAT Session Data] → [rule_engine_adapter.py]
                                │
                         ┌──────┴──────┐
                         │ 점수 변환    │
                         │ (0-100 표준) │
                         └──────┬──────┘
                                │
                         ┌──────┴──────┐
                         │ 임계값 판단  │
                         │ speed < 60  │
                         │ = warning   │
                         └──────┬──────┘
                                │
                         [Agent Pipeline]
                         코칭 메시지 생성
```

### 테스트 전략

```python
# test_t28_rule_engine.py
def test_rule_engine_input_validation():
    """Rule Engine 입력 스키마 유효성 검증"""
    valid_input = load_json("/data/eval/coaching_samples/synthetic_rule_input.json")
    result = validate_rule_engine_input(valid_input)
    assert result.is_valid

def test_score_transformation():
    """점수 변환(0-100)이 정확한지 확인"""
    raw_scores = {"speed_score": 85, "distance_score": 72, "angle_score": 90}
    transformed = transform_scores(raw_scores)
    assert all(0 <= v <= 100 for v in transformed.values())

def test_coaching_generation_from_scores():
    """낮은 점수에 대한 코칭 메시지가 생성되는지 확인"""
    low_scores = {"speed_score": 45, "distance_score": 30}
    coaching = generate_coaching_from_scores(low_scores)
    assert "speed" in coaching["warnings"] or "distance" in coaching["warnings"]

def test_rule_engine_latency():
    """Rule Engine 어댑터 처리 시간이 2초 이내인지 확인"""
    start = time.time()
    process_rule_engine_session(synthetic_session_data)
    assert time.time() - start < 2.0
```

## 7.3 T29: Session Export 자동화

| 항목 | 내용 |
|---|---|
| 목표 | RealCOAT 세션 데이터 자동 감지 및 인덱싱 |
| 인터페이스 사양 | `/workspace/trainer/docs/export_automation.md` |
| 작업 내용 | 1) `/exports/realcoat/` 디렉터리 감시 (watchdog), 2) `session_summary.json` 스키마 검증, 3) 검증 통과 시 분석 큐 등록, 4) 실패 시 재시도 정책 및 로깅 |
| 성공 기준 | 파일 감지 → 검증 → 큐 등록 전체 파이프라인 자동화, 스키마 검증 실패 시 명확한 에러 리포트 |
| 예상 공수 | 2일 |
| 선행 작업 | T28 (Rule Engine 연동) |

### 필요 데이터

| 데이터 유형 | 포맷 | 예상 규모 | 출처 | 차단 여부 |
|---|---|---|---|---|
| session_summary.json 샘플 | JSON | 5-10건 | RealCOAT | 부분 차단 (합성 데이터로 시작) |

### Export 자동화 아키텍처

```text
/exports/realcoat/{session_id}/
├── session_summary.json    (필수)
├── heatmap.png             (선택)
├── screenshot.png          (선택)
├── motion_data.json        (선택)
└── sequence_log.json       (선택)

[File Watcher (watchdog)] → [Schema Validator] → [Analysis Queue]
         │                        │                      │
    신규 파일 감지          스키마 검증 실패        Rule Engine +
                           → 에러 로깅           Agent Pipeline
                           → 재시도 큐            분석 실행
```

### 테스트 전략

```python
# test_t29_session_export.py
def test_session_summary_schema_validation():
    """유효한 session_summary.json이 검증을 통과하는지 확인"""
    valid = load_json("tests/fixtures/valid_session_summary.json")
    assert validate_session_summary(valid).is_valid

def test_invalid_session_summary_rejected():
    """필수 필드 누락 시 검증 실패하는지 확인"""
    invalid = {"session_id": "test"}  # 필수 필드 대부분 누락
    result = validate_session_summary(invalid)
    assert not result.is_valid
    assert len(result.errors) > 0

def test_file_watcher_detects_new_session():
    """새 세션 디렉터리 생성 시 자동 감지되는지 확인"""
    create_test_session("/exports/realcoat/TEST-001/")
    detected = wait_for_detection(timeout=10)
    assert detected["session_id"] == "TEST-001"

def test_failed_validation_logged():
    """스키마 검증 실패 시 에러 로그에 기록되는지 확인"""
    create_invalid_session("/exports/realcoat/BAD-001/")
    time.sleep(5)
    logs = read_error_log()
    assert any("BAD-001" in log for log in logs)
```

## 7.4 T30: MCP Stub 실구현 전환

| 항목 | 내용 |
|---|---|
| 목표 | 5개 Stub MCP 도구 중 최소 3개를 실제 로직으로 전환 |
| 현재 상태 | 1/6 real (dft_judge만 실제), 5개 Stub |
| 작업 내용 | 1) `anomaly_detection.py`: 히트맵 + DFT 데이터 기반 이상 탐지 로직, 2) `sequence_validation.py`: 궤적 데이터 기반 순서 검증 로직, 3) `location_tracking.py`: 3D 좌표 → Zone ID 매핑 로직, 4) `pose_estimation.py`: 관절 각도 비교 로직 (우선순위 낮음), 5) `timeseries.py`: 시계열 추세 분석 로직 (우선순위 낮음) |
| 성공 기준 | 최소 3/5 Stub → Real 전환, 단위 테스트 100% 통과 |
| 예상 공수 | 3일 |
| 선행 작업 | T27 (VLM 결과가 anomaly_detection에 필요), T28 (Rule Engine 데이터가 sequence에 필요) |

### 필요 데이터

| 도구 | 필요 데이터 | 출처 | 차단 여부 |
|---|---|---|---|
| anomaly_detection | 히트맵 + DFT 기준값 + VLM 분석 결과 | T27 VLM 산출물 | 부분 (합성으로 시작) |
| sequence_validation | 궤적 + 기준 순서 정의 | 도메인 전문가 | 부분 (합성으로 시작) |
| location_tracking | Zone 정의 (3D 바운딩 박스) | RealCOAT 설정 | 부분 (합성으로 시작) |
| pose_estimation | 숙련자 BVH 데이터 | RealCOAT 녹화 | 차단 |
| timeseries | 세션 시계열 데이터 | RealCOAT | 부분 |

### 구현 우선순위

| 순위 | 도구 | 이유 |
|---|---|---|
| 1 | anomaly_detection | VLM 결과 활용, Agentic 점수 직접 영향 |
| 2 | sequence_validation | Rule Engine 데이터 활용, 순서 검증 핵심 |
| 3 | location_tracking | 좌표 → Zone 매핑 비교적 단순 |
| 4 | timeseries | 통계 분석 로직 단순, 데이터 의존 낮음 |
| 5 | pose_estimation | BVH 데이터 확보 필요, Phase 3 이전 |

### 테스트 전략

```python
# test_t30_mcp_real_implementation.py
def test_anomaly_detection_real():
    """anomaly_detection이 실제 로직으로 이상 탐지하는지 확인"""
    result = anomaly_detection.detect({
        "image_context": "좌하단 저도막 패턴",
        "dft_measurements": [102, 95, 65, 110, 88]
    })
    assert result["detected"] == True
    assert result["type"] in ["저도막", "과도막", "결함"]

def test_sequence_validation_real():
    """sequence_validation이 순서 위반을 감지하는지 확인"""
    trajectory = [{"x": 1, "y": 0, "z": 0, "t": 0.1}, {"x": 1, "y": 1, "z": 0, "t": 0.2}]
    result = sequence_validation.validate({"trajectory": trajectory, "zone_id": "B-3"})
    assert "violation" in result or "compliant" in result

def test_location_tracking_real():
    """location_tracking이 좌표를 Zone ID로 매핑하는지 확인"""
    result = location_tracking.identify({"coords": {"x": 1.0, "y": 0.5, "z": 0.8}})
    assert "zone_id" in result
    assert result["zone_id"] != ""

def test_stub_count_reduced():
    """Stub 도구가 2개 이하인지 확인"""
    tools = get_mcp_tool_status()
    stubs = [t for t in tools if t["implementation"] == "stub"]
    assert len(stubs) <= 2
```

\newpage

# 8. P2 — Phase 3 미래 (Week 5+, 코드 + 데이터)

## 8.1 T31: RLHF/DPO 학습 준비

| 항목 | 내용 |
|---|---|
| 목표 | Hallucination 2% → <1%, 응답 품질 일관성 향상 |
| 작업 내용 | 1) DPO 학습 파이프라인 구축, 2) Human preference annotation 프로토콜 설계, 3) chosen/rejected 쌍 500건 수집 |
| 필요 데이터 | ✅ 필수 — 도메인 전문가의 preference annotation (chosen/rejected 쌍 500-1000건) |
| 예상 공수 | 5일+ |

### 필요 데이터

| 데이터 유형 | 포맷 | 예상 규모 | 출처 | 차단 여부 |
|---|---|---|---|---|
| Preference 쌍 (chosen/rejected) | JSON | 500-1000쌍 | 도메인 전문가 + 교관 | 완전 차단 |

## 8.2 T32: Qdrant 벡터 DB 마이그레이션

| 항목 | 내용 |
|---|---|
| 목표 | FAISS (파일 기반, 단일 클라이언트) → Qdrant (서버 기반, 멀티 클라이언트) |
| 작업 내용 | 1) Qdrant 서버 Docker 배포, 2) `build_index.py` 마이그레이션, 3) `search_index.py` Qdrant 클라이언트 전환, 4) 메타데이터 필터링 활용 |
| 필요 데이터 | ❌ 없음 — 코드 마이그레이션 |
| 예상 공수 | 2일 |

## 8.3 T33: 음성 인터페이스 (Whisper STT + TTS)

| 항목 | 내용 |
|---|---|
| 목표 | 조선소 현장 핸즈프리 코칭을 위한 음성 입출력 |
| 작업 내용 | 1) Whisper STT 통합, 2) TTS 엔진 선정 및 연동, 3) 도메인 용어 인식률 튜닝 |
| 필요 데이터 | ✅ 필수 — 도장 작업 환경 오디오 샘플, 도메인 용어 발화 데이터 |
| 예상 공수 | 5일+ |

### 필요 데이터

| 데이터 유형 | 포맷 | 예상 규모 | 출처 | 차단 여부 |
|---|---|---|---|---|
| 작업 환경 오디오 | WAV/MP3, 16kHz+ | 100건+ | 현장 녹음 | 완전 차단 |
| 도메인 용어 발화 | WAV + 전사 텍스트 | 200쌍+ | 교관/훈련생 | 완전 차단 |

## 8.4 T34: 프로덕션 배포 (Kubernetes)

| 항목 | 내용 |
|---|---|
| 목표 | 컨테이너 오케스트레이션, 모니터링, 오토스케일링 |
| 작업 내용 | 1) K8s Deployment/Service 매니페스트, 2) Prometheus + Grafana 모니터링, 3) GPU 노드 오토스케일 설정 |
| 필요 데이터 | ❌ 없음 — 인프라 |
| 예상 공수 | 5일+ |

\newpage

# 9. 데이터 인벤토리 및 Gap 분석

## 9.1 현재 보유 데이터

| 데이터 | 규모 | 위치 | 상태 |
|---|---|---|---|
| SFT 학습 데이터 | 567건 (DocQA 380 + FC 187) | `/data/sft/sft_train.jsonl` | 활성 |
| SFT 검증 데이터 | 30건 | `/data/sft/sft_val.jsonl` | 활성 |
| DocQA 평가셋 | 50문항 | `/data/eval/eval_set.jsonl` | 활성 |
| Agentic 평가셋 | 30문항 | `/data/eval/eval_set_agentic.jsonl` | 활성 |
| RAG 청크 | 10,829건 (147 문서) | `/data/chunked/chunks.jsonl` | 활성 |
| 이미지 (비주석) | 71장 | `/data/raw/html/` | 미활용 |

## 9.2 필요 데이터 Gap

| 데이터 | 필요 규모 | 확보 상태 | 확보 방법 | 우선순위 |
|---|---|---|---|---|
| 히트맵 이미지 (주석) | 50-100장 | 71장 미주석 보유 | 도메인 전문가 라벨링 | P1 |
| XR 스크린샷 | 20-30장 | 0건 | RealCOAT 캡처 | P1 |
| 모션 데이터 (JSON) | 10세션+ | 0건 | RealCOAT Export | P1 |
| 시퀀스 로그 (JSON) | 10세션+ | 0건 | RealCOAT Export | P1 |
| session_summary.json | 5-10건 | 0건 | RealCOAT Export | P1 |
| Zone 3D 바운딩 박스 | 1 설정파일 | 0건 | RealCOAT 설정 | P1 |
| RLHF Preference 쌍 | 500-1000쌍 | 0건 | 전문가 annotation | P2 |
| 음성 오디오 샘플 | 100건+ | 0건 | 현장 녹음 | P2 |

## 9.3 합성 데이터 전략

P1 작업의 개발 차단을 방지하기 위해 아래 합성 데이터를 먼저 생성한다.

| 합성 데이터 | 생성 방법 | 용도 |
|---|---|---|
| 합성 모션 데이터 | 정규분포 기반 시뮬레이션 | Rule Engine 어댑터 개발 |
| 합성 시퀀스 로그 | 템플릿 기반 랜덤 생성 | Sequence Validation 개발 |
| 합성 session_summary | 스키마 기반 Faker 생성 | Export 자동화 개발 |
| 합성 Zone 정의 | 가상 선박 블록 좌표 | Location Tracking 개발 |

\newpage

# 10. 기술 아키텍처 변경 사항

## 10.1 Phase 2 아키텍처 확장

```text
┌─────────────────────────────────────────────────────┐
│                  XR Client (Unity)                   │
│         AR/VR 콘텐츠 렌더링 + 사용자 인터페이스           │
└─────────────┬───────────────────────┬───────────────┘
              │ Request JSON          │ Response JSON
              ▼                       ▼
┌─────────────────────────────────────────────────────┐
│         AI Co-Pilot Agent (Qwen3-4B + QLoRA)        │
│  ┌──────────┐  ┌──────────┐  ┌────────────────────┐ │
│  │ RAG Core │  │ QLoRA    │  │ Enhanced CoT       │ │  ← T21 강화
│  │ (문서 지식)│  │ (Iter15) │  │ (<think> 다단계)    │ │
│  └──────────┘  └──────────┘  └────────────────────┘ │
│                       │                              │
│         Enhanced Tool Calling Layer                   │  ← T22 강화
│  ┌──────┬──────┬──────┬──────┬──────┬──────────────┐ │
│  │자세  │이상  │시계열│위치  │순서  │DFT 판정      │ │
│  │추정  │탐지  │분석  │추적  │검증  │합부 판정      │ │
│  │(Stub)│(Real)│(Real)│(Real)│(Real)│(Real)        │ │  ← T30 실구현
│  └──┬───┴──┬───┴──┬───┴──┬───┴──┬───┴──────┬───────┘ │
│     │      │      │      │      │          │         │
│  ┌──┴──────┴──────┴──────┴──────┴──────────┴───────┐ │
│  │          Response Formatter + Validator          │ │  ← T23 표준화
│  │         Parameter Validator + Type Coercion      │ │  ← T24 개선
│  └─────────────────────────────────────────────────┘ │
└─────┬──────┬──────┬──────┬──────┬──────────┬─────────┘
      ▼      ▼      ▼      ▼      ▼          ▼
┌─────────────────────────────────────────────────────┐
│                MCP Server Layer (FastAPI)             │
│  pose_estimation | anomaly_detection | timeseries    │
│  location_tracking | sequence_validation | dft_judge │
└──────────────────┬──────────────────────────────────┘
                   │
    ┌──────────────┼──────────────┐
    ▼              ▼              ▼
┌────────┐  ┌──────────┐  ┌────────────┐
│ VLM    │  │ Rule     │  │ Session    │  ← P1 신규
│ Qwen2  │  │ Engine   │  │ Export     │
│ -VL-2B │  │ Adapter  │  │ Watcher    │
└────────┘  └──────────┘  └────────────┘
```

## 10.2 VRAM 예산 (RTX 4080 16GB)

| 구성 요소 | VRAM | Phase |
|---|---|---|
| LLM (Qwen3-4B, 4-bit NF4) + QLoRA | ~4.5 GB | Phase 1 유지 |
| BGE-M3 임베딩 + FAISS | ~2.0 GB | Phase 1 유지 |
| VLM (Qwen2-VL-2B, 4-bit) | ~2.0 GB | **Phase 2 신규** |
| MCP Server (FastAPI) | ~0.5 GB | Phase 1 유지 (CPU) |
| 여유 | ~7.0 GB | XR 시뮬레이터용 |
| **합계** | **~9.0 GB** | **16GB 내 안전** |

## 10.3 신규 파일 목록

| 파일 | 위치 | 용도 | Task |
|---|---|---|---|
| `generate_cot_sft.py` | `/workspace/trainer/scripts/` | CoT SFT 데이터 생성 | T21 |
| `generate_tool_sft.py` | `/workspace/trainer/scripts/` | 도구 선택 SFT 데이터 생성 | T22 |
| `merge_sft.py` | `/workspace/trainer/scripts/` | SFT 데이터 통합 (중복 제거 포함) | T26 |
| `vlm_pipeline.py` | `/workspace/trainer/scripts/` | VLM 추론 파이프라인 | T27 |
| `rule_engine_adapter.py` | `/workspace/trainer/scripts/` | Rule Engine 입출력 어댑터 | T28 |
| `session_watcher.py` | `/workspace/trainer/scripts/` | Export 파일 자동 감지 | T29 |

\newpage

# 11. 평가 체계 확장

## 11.1 기존 평가셋 유지

| 평가셋 | 문항 수 | 용도 | 변경 |
|---|---|---|---|
| DocQA (`eval_set.jsonl`) | 50 | 문서 QA 기준선, hallucination 모니터링 | 변경 없음 |
| Agentic (`eval_set_agentic.jsonl`) | 30 | 5차원 Agentic 점수 | 변경 없음 |

## 11.2 신규 평가 항목

| 평가 | 문항/케이스 수 | 기준 | Task |
|---|---|---|---|
| VLM 히트맵 분석 정확도 | 20 이미지 | ROI 일치율 80%+ | T27 |
| Rule Engine 점수 일관성 | 10 세션 | RealCOAT 대비 차이 <5% | T28 |
| Export 자동화 성공률 | 10 세션 | 감지 → 검증 → 큐 100% | T29 |
| MCP Real 도구 정확도 | 20 케이스 | 단위 테스트 100% 통과 | T30 |

\newpage

# 12. TDD 전략 및 테스트 설계

## 12.1 테스트 계층

| 계층 | 범위 | 도구 | 위치 |
|---|---|---|---|
| Unit Test | 개별 함수/클래스 | pytest | `/workspace/tests/unit/` |
| Integration Test | 모듈 간 연동 | pytest + MCP mock | `/workspace/tests/integration/` |
| E2E Test | 전체 파이프라인 | pytest + live MCP | `/workspace/tests/e2e/` |
| Eval Test | Agentic/DocQA 점수 | 기존 eval 스크립트 | `/workspace/tests/eval/` |

## 12.2 TDD 실행 규칙

1. **Red**: 테스트를 먼저 작성한다. 실패하는 것을 확인한다.
2. **Green**: 최소한의 코드를 작성하여 테스트를 통과시킨다.
3. **Refactor**: 코드를 정리하고 중복을 제거한다. 테스트가 여전히 통과하는지 확인한다.

## 12.3 테스트 파일 구조

```text
/workspace/tests/
├── unit/
│   ├── test_cot_sft_generation.py          # T21
│   ├── test_tool_selection_rules.py        # T22
│   ├── test_response_formatter.py          # T23
│   ├── test_parameter_validator.py         # T24
│   ├── test_vlm_pipeline.py               # T27
│   ├── test_rule_engine_adapter.py         # T28
│   ├── test_session_schema_validator.py    # T29
│   └── test_mcp_real_tools.py             # T30
├── integration/
│   ├── test_agent_pipeline_cot.py          # T21+T22+T23
│   ├── test_vlm_mcp_integration.py         # T27+T30
│   └── test_rule_engine_pipeline.py        # T28+T29
├── e2e/
│   ├── test_full_agentic_eval.py           # T26
│   └── test_full_pipeline.py              # All
└── eval/
    ├── test_agentic_score.py               # 6.0+ 확인
    └── test_docqa_regression.py            # halluc ≤2% 확인
```

\newpage

# 13. 커밋 전략

## 13.1 Atomic Commit 원칙

각 커밋은 하나의 논리적 단위에 대응하며, 단독으로 테스트를 통과해야 한다.

## 13.2 커밋 메시지 형식

```text
<type>(<scope>): <description>

- feat: 새 기능
- fix: 버그 수정
- refactor: 리팩토링
- test: 테스트 추가/수정
- data: SFT/평가 데이터 변경
- docs: 문서 변경
- chore: 빌드/설정 변경
```

## 13.3 Task별 커밋 계획

| Task | 커밋 수 | 커밋 내용 |
|---|---|---|
| T20 | 2 | `data(eval): run Iter14 3-way reeval`, `docs(report): update final report with Iter14` |
| T21 | 3 | `test(cot): add CoT SFT generation tests`, `feat(cot): implement CoT SFT generator`, `data(sft): generate 80 CoT training samples` |
| T22 | 3 | `test(tool): add tool selection tests`, `feat(tool): enhance tool selection rules in prompt`, `data(sft): generate 40 tool selection samples` |
| T23 | 3 | `test(response): add response structure tests`, `feat(response): implement response templates and validator`, `refactor(formatter): integrate response validation pipeline` |
| T24 | 2 | `test(param): add parameter validation tests`, `feat(param): implement type coercion and default filling` |
| T25 | 2 | `test(medium): add medium difficulty tests`, `data(sft): generate 30 medium complexity samples` |
| T26 | 3 | `data(sft): merge and deduplicate SFT v5 (800+ samples)`, `feat(qlora): train Iter15 adapter`, `docs(eval): generate P0 integration evaluation report` |
| T27 | 4 | `test(vlm): add VLM pipeline tests`, `feat(vlm): implement Qwen2-VL inference pipeline`, `feat(mcp): connect VLM to anomaly_detection`, `docs(vlm): add VRAM benchmark results` |
| T28 | 3 | `test(rule): add Rule Engine adapter tests`, `feat(rule): implement score transformation adapter`, `feat(coaching): integrate rule scores into coaching generation` |
| T29 | 3 | `test(export): add session export tests`, `feat(export): implement file watcher and schema validator`, `feat(export): add retry policy and error logging` |
| T30 | 4 | `test(mcp): add real MCP tool tests`, `feat(mcp): implement anomaly_detection real logic`, `feat(mcp): implement sequence_validation real logic`, `feat(mcp): implement location_tracking real logic` |

\newpage

# 14. 일정 및 공수 추정

## 14.1 Phase 2 일정표

| 주차 | 기간 | 작업 | 마일스톤 |
|---|---|---|---|
| **Week 1** | Day 1-5 | T20, T21, T22, T23, T24 | P0 개별 개선 완료 |
| **Week 2** | Day 6-10 | T25, T26 | P0 통합 평가, **Agentic 6.0+ 확인** |
| **Week 3** | Day 11-15 | T27, T28 | VLM 추론 + Rule Engine 연동 |
| **Week 4** | Day 16-20 | T29, T30 | Export 자동화 + MCP 실구현 |
| **Week 5+** | — | T31-T34 | P2 Phase 3 계획 |

## 14.2 공수 요약

| 우선순위 | 작업 수 | 총 공수 | 비고 |
|---|---|---|---|
| P0 (코드 전용) | 7 | **10일** | 외부 의존성 없음 |
| P1 (코드+데이터) | 4 | **10일** | 데이터 확보 일정에 따라 변동 |
| P2 (향후) | 4 | **17일+** | Phase 3에서 상세 계획 |
| **Phase 2 합계** | **11** | **~20일** | P0+P1 기준 |

\newpage

# 15. 리스크 및 폴백 전략

| 리스크 | 징후 | 영향 | 폴백 |
|---|---|---|---|
| **DocQA Catastrophic Forgetting 재발** | **DocQA halluc >5% 또는 score <1.3** | **DocQA 품질 훼손** | **FC 비율 15%로 하향, DocQA 비율 70%로 상향, loss 전략 분기 적용 후 재학습** |
| **Dual Gate 동시 충족 실패** | **Agentic ≥6.0이지만 DocQA <1.3 (또는 반대)** | **Phase 2 진행 차단** | **데이터 비율 미세 조정 (5% 단위), multi-task loss weight 도입 검토** |
| CoT 강화 후 Agentic 점수 미달 | 6.0 미도달 | MVP 지연 | SFT 데이터 추가 생성 (100건+), 프롬프트 반복 튜닝 |
| VLM VRAM 초과 | RTX 4080에서 OOM | VLM 통합 불가 | CPU offloading 또는 해상도 축소 (256x256) |
| P1 원시 데이터 미확보 | RealCOAT 팀 지연 | Week 3-4 차단 | 합성 데이터로 개발 진행, 실데이터 확보 시 검증 |
| QLoRA 재학습 성능 저하 | halluc 증가 | 안정성 훼손 | Iter14 어댑터로 롤백, 데이터 품질 재검수 |
| MCP 실구현 지연 | 3개 미만 전환 | Agentic 점수 제한 | 가장 영향력 큰 2개만 우선 전환 (anomaly, sequence) |
| medium 점수 개선 미흡 | 5.0 미만 | 전체 점수 영향 | Medium 전용 SFT 추가 20건, few-shot 튜닝 |

\newpage

# 16. 인터페이스 사양 참조

| 문서 | 위치 | 관련 Task |
|---|---|---|
| Rule Engine 인터페이스 계약서 | `/workspace/trainer/docs/rule_engine_interface.md` | T28 |
| Export 자동화 프로토콜 | `/workspace/trainer/docs/export_automation.md` | T29 |
| 향후 로드맵 | `/workspace/trainer/docs/future_roadmap.md` | T27, T28, T33 |
| Phase 1 통합 PRD | `/workspace/.taskmaster/docs/ship_painting_optimal_prd_final.md` | 전체 참조 |
| Qwen3.5 마이그레이션 PRD | `/workspace/.taskmaster/docs/qwen35_migration_prd.md` | Phase 3 참조 |
| Agentic 평가 루브릭 | `/workspace/trainer/configs/eval_rubric_agentic.md` | T20, T26 |

\newpage

# 17. 문서 개정 이력

| 버전 | 날짜 | 주요 변경 내용 | 작성자 |
|---|---|---|---|
| v0.1 | 2026-03-16 | Phase 2 MVP PRD 초안 작성, P0/P1/P2 분류, TDD 전략 포함 | Sisyphus-Junior |
| v0.2 | 2026-03-16 | Iter14 DocQA 3-Way 재평가 결과 반영: T20 완료 처리, T26에 DocQA 균형 학습 전략 추가 (Dual Gate, SFT 비율 상한, Loss 전략 분기), T21 CoT에 DocQA 스타일 30건 포함, 리스크 테이블에 Catastrophic Forgetting 항목 추가 | Sisyphus |

---
*본 문서는 선박 도장 AI Co-Pilot Phase 2 MVP의 실행 계획서이며, `task-master parse-prd --append`로 파싱하여 작업 목록을 생성한다.*
