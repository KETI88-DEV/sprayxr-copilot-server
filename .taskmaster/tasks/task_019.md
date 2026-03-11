# Task ID: 19

**Title:** Qwen3 파이프라인 마이그레이션 및 A/B 비교

**Status:** in-progress

**Dependencies:** 9 ✓

**Priority:** high

**Description:** Qwen2.5-7B-Instruct에서 Qwen3-4B로 기본 마이그레이션 후, Qwen3-8B와 A/B 비교 평가. 학습/추론 파이프라인 전체 업데이트. Thinking Mode 통합, 샘플링 파라미터 변경, chat template 대응.

**Details:**

1. 의존성 업그레이드: transformers>=4.51.0, trl/peft/bitsandbytes 최신
2. 모델 다운로드: Qwen/Qwen3-4B (우선), Qwen/Qwen3-8B (비교용)
3. train_qlora.py 업데이트:
   - model_name 변경
   - enable_thinking=False로 SFT 학습 (non-reasoning task)
   - 샘플링 파라미터: temp=0.7, top_p=0.8, top_k=20 (non-thinking mode)
4. run_qlora_eval.py 업데이트:
   - Qwen3 chat template 적용 (enable_thinking 파라미터)
   - <think> 블록 후처리 (응답에서 strip)
   - 추론 모드 선택: 평가 시 thinking/non-thinking 양쪽 테스트
5. generate_sft.py: SYSTEM_PROMPT + chat template 호환성 확인
6. qlora_config.yaml: model_name_or_path, sampling params 업데이트
7. Qwen3-4B QLoRA 학습 → 3-way 평가 (Iter3)
8. Qwen3-8B QLoRA 학습 → 3-way 평가 → 4B와 비교
9. 평가 기준: 기존 50문항 + 한국어 품질 + 추론 속도 + VRAM 사용량
10. 최종 모델 선정 및 기존 adapter 경로 업데이트
11. 산출물: 마이그레이션 비교 리포트, 업데이트된 파이프라인 스크립트

**Test Strategy:**

Qwen3-4B QLoRA 학습 정상 완료 (loss 수렴), 3-way 평가 RAG+QLoRA avg score >= 1.2, Qwen3-4B vs Qwen3-8B 비교표 작성, Qwen2.5-7B 대비 성능 비교 (regression 없음 확인)

## Subtasks

### 19.1. 의존성 업그레이드 및 Qwen3 모델 다운로드

**Status:** pending  
**Dependencies:** None  

transformers>=4.51.0, trl, peft, bitsandbytes 최신 버전 업그레이드. Qwen/Qwen3-4B 및 Qwen/Qwen3-8B 모델을 /models/에 다운로드. 기존 torch 2.4.0 호환성 확인.

**Details:**

pip install -U 'transformers>=4.51.0' trl peft bitsandbytes 실행. Qwen3-4B: huggingface-cli download Qwen/Qwen3-4B --local-dir /models/Qwen3-4B. Qwen3-8B: huggingface-cli download Qwen/Qwen3-8B --local-dir /models/Qwen3-8B. torch 2.4.0 + transformers 4.51+ 호환성 smoke test (import + model load). set_submodule monkey-patch 필요 여부 확인.

### 19.2. train_qlora.py Qwen3 대응 업데이트

**Status:** pending  
**Dependencies:** 19.1  

QLoRA 학습 스크립트를 Qwen3 모델에 대응. model_name 변경, chat template에 enable_thinking=False 적용, 샘플링 파라미터 변경.

**Details:**

qlora_config.yaml: model_name_or_path를 /models/Qwen3-4B로 변경. train_qlora.py: tokenizer.apply_chat_template 호출 시 enable_thinking=False 추가 (SFT는 non-reasoning task). 샘플링: temperature=0.7, top_p=0.8, top_k=20. LoRA target_modules는 동일 (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj). BitsAndBytesConfig NF4+bfloat16 그대로 유지. generate_sft.py의 SYSTEM_PROMPT + chat template 호환성 확인.

### 19.3. run_qlora_eval.py Qwen3 대응 업데이트

**Status:** pending  
**Dependencies:** 19.1  

평가 스크립트를 Qwen3에 대응. Thinking Mode 지원(enable_thinking 파라미터), <think> 블록 후처리, 샘플링 파라미터 변경.

**Details:**

run_qlora_eval.py: (1) Qwen3 모델 로딩 (Qwen2→Qwen3 model_type 대응). (2) enable_thinking 파라미터 추가 (기본 False, 선택적 True). (3) <think>...</think> 블록을 응답에서 strip하는 extract_answer() 함수 추가 (re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)). (4) thinking mode 평가 시 temperature=0.6, top_p=0.95, top_k=20 적용. (5) 디버그 로그에 <think> 내용 기록 (평가 분석용).

### 19.4. Qwen3-4B QLoRA 학습 및 3-way 평가 (Iter3)

**Status:** pending  
**Dependencies:** 19.2, 19.3  

Qwen3-4B 기반 QLoRA 학습 실행 → RAG-Only, QLoRA-Only, RAG+QLoRA 3-way 평가. 기존 Qwen2.5-7B Iter1 결과와 비교.

**Details:**

train_qlora.py 실행 → /output/adapters/qwen3_4b_shipcoat_lora/ 저장. 3-way 평가: run_qlora_eval.py --mode qlora-only, --mode rag-qlora, run_rag_eval.py. 목표: RAG+QLoRA avg score >= 1.2. Qwen2.5-7B Iter1 결과 (RAG+QLoRA 1.376)와 비교. VRAM 사용량(nvidia-smi), 추론 속도(tokens/sec), 학습 시간 기록.

### 19.5. Qwen3-8B QLoRA 학습 및 4B vs 8B A/B 비교

**Status:** pending  
**Dependencies:** 19.4  

Qwen3-8B 기반 QLoRA 학습 → 3-way 평가 → Qwen3-4B 결과와 A/B 비교표 작성. 최종 모델 선정.

**Details:**

qlora_config.yaml model_name_or_path를 /models/Qwen3-8B로 변경 후 학습. 동일 SFT 데이터, 동일 평가셋으로 3-way 평가. A/B 비교 항목: (1) avg score, (2) 카테고리별 점수, (3) hallucination rate, (4) ROUGE-L, BERTScore, (5) VRAM 사용량, (6) 추론 tokens/sec, (7) 학습 시간. 비교 리포트: /output/eval_results/qwen3_migration_report.md.
