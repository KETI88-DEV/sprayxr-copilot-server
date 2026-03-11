# Task ID: 9

**Title:** QLoRA-Only 및 RAG+QLoRA 평가 실행

**Status:** done

**Dependencies:** 6 ✓, 8 ✓

**Priority:** high

**Description:** 학습된 adapter(/output/adapters/qwen25_shipcoat_lora/)를 로드하여 QLoRA-Only 구성으로 50문항 평가 실행, RAG+QLoRA 구성도 평가, RAG-Only 결과와 비교 분석

**Details:**

1. QLoRA-Only 실행: Qwen2.5-7B + adapter(/output/adapters/qwen25_shipcoat_lora/)만 로드, RAG 검색 미사용, 50문항 일괄 실행
2. RAG+QLoRA 실행: Qwen2.5-7B + adapter + RAG 검색, 동일 프롬프트, 50문항 실행
3. 자동 평가: 각 구성별 ROUGE-L, BERTScore, hallucination ratio, 응답 길이
4. 수동 채점: 5차원 루브릭 재적용(동일 검수자 또는 검증 인원)
5. 비교 분석: 3가지 구성(RAG-Only vs QLoRA-Only vs RAG+QLoRA)
   - 평균 점수 비교
   - hallucination 비율 비교
   - 용어 정확도, 형식 일관성 비교
   - 카테고리별 성능 차이
6. Keep/Drop 의사결정:
   - RAG+QLoRA가 RAG-Only 대비 평균 0.15점 이상 향상 && hallucination 동일/감소 → adapter 유지
   - 그 외 → adapter 폐기
7. 산출물: /output/eval_results/qlora_only_results.csv, /output/eval_results/ragqlora_results.csv, /output/eval_results/comparison.csv, /output/eval_results/comparison_analysis.md
8. 시각화: 카테고리별 점수 히트맵, 오류 유형 분포도

**Test Strategy:**

세 가지 구성 모두 50문항 실행 완료, 결과 CSV 모두 /output/eval_results/에 저장, 비교 분석 문서 작성, keep/drop 의사결정 명확히 기록, 평균 점수/hallucination 수치 정량적 비교

## Subtasks

### 9.1. QLoRA-Only 구성으로 50문항 평가 실행

**Status:** done  
**Dependencies:** None  

학습된 adapter를 로드하여 RAG 검색 없이 QLoRA-Only 구성으로 50문항 일괄 평가 실행

**Details:**

Qwen2.5-7B 모델에 /output/adapters/qwen25_shipcoat_lora/ adapter만 로드하여 RAG 검색을 미사용한 상태에서 /data/eval/eval_set.jsonl의 50문항을 순차 실행. /workspace/trainer/scripts/run_rag_eval.py 스크립트 실행, temperature=0.1, max_tokens=768 설정. 각 문항별 q_id, answer, timestamp를 기록하고 /output/eval_results/qlora_only_results.csv에 저장. adapter 로드 시간, 각 문항 추론 시간 기록.

### 9.2. RAG+QLoRA 구성으로 50문항 평가 실행

**Status:** done  
**Dependencies:** 9.1  

Qwen2.5-7B + adapter + RAG 검색을 통합한 RAG+QLoRA 구성으로 동일 50문항 평가 실행

**Details:**

QLoRA-Only 평가와 동일한 50문항을 RAG+QLoRA 구성으로 실행. Qwen2.5-7B + /output/adapters/qwen25_shipcoat_lora/ adapter + RAG 검색(top-k=5) 활성화. 동일한 프롬프트 템플릿 사용하되 retrieved_docs를 포함. temperature=0.1, max_tokens=768. 각 문항별 q_id, answer, citations, timestamp, rag_context 기록하고 /output/eval_results/ragqlora_results.csv에 저장. 검색 시간, 추론 시간 분리 측정.

### 9.3. QLoRA-Only 및 RAG+QLoRA 자동 평가 지표 계산

**Status:** done  
**Dependencies:** 9.2  

두 구성의 평가 결과에 대해 ROUGE-L, BERTScore, hallucination ratio, 응답 길이 등 자동 평가 지표 계산

**Details:**

QLoRA-Only과 RAG+QLoRA 결과 CSV에 대해 자동 평가 실행. /data/eval/eval_set.jsonl의 expected_answer를 reference로 사용하여 ROUGE-L(rouge-score 라이브러리), BERTScore F1(bert-score 라이브러리) 계산. 응답 길이(토큰 수), citation 개수 기록. hallucination ratio 계산: 출처 없는 사실 주장, 사실 오류, 과장/추론 과다, 모순 등을 pattern matching 및 수동 샘플링으로 분류. 각 구성별 지표를 qlora_only_metrics.csv, ragqlora_metrics.csv로 저장.

### 9.4. QLoRA-Only 및 RAG+QLoRA 수동 채점 실행

**Status:** done  
**Dependencies:** 9.2  

두 구성의 평가 결과에 대해 5차원 루브릭을 적용하여 수동 채점 실행

**Details:**

QLoRA-Only 50문항과 RAG+QLoRA 50문항 결과에 대해 동일 검수자 또는 검증 인원이 5차원 루브릭(기술정확성, 문서충실도, 용어적절성, 완전성, 한국어품질) × 0-2점 = 최대 10점을 적용하여 채점. 두 구성 각각을 블라인드 처리하여 편향 제거. 검수자별 점수를 기록하고 평균값 계산. qlora_only_manual_scores.csv, ragqlora_manual_scores.csv로 저장(q_id, manual_score, reviewer_name, 각 차원별 점수).

### 9.5. 3가지 구성 비교 분석 및 Keep/Drop 의사결정

**Status:** done  
**Dependencies:** 9.3, 9.4  

RAG-Only vs QLoRA-Only vs RAG+QLoRA 3가지 구성을 종합적으로 비교 분석하고 adapter 유지 여부 의사결정

**Details:**

RAG-Only(/output/eval_results/rag_only_results.csv), QLoRA-Only, RAG+QLoRA 결과를 종합하여 비교 분석. 1) 평균 점수 비교: 세 구성별 수동 채점 평균값, 자동 평가 지표 평균값(ROUGE-L, BERTScore) 계산 및 비교표 작성. 2) hallucination 비율 비교: 세 구성의 hallucination 건수, 비율(%) 비교. 3) 용어 정확도, 형식 일관성: 오류 유형별 분포, 카테고리별 성능 분석. 4) 카테고리별 성능: 5개 카테고리(용어, 공정절차, 결함분석, 코팅두께/사양, 안전/규정)별 평균 점수 비교. Keep/Drop 의사결정: RAG+QLoRA 평균 점수가 RAG-Only 대비 0.15점 이상 향상 AND hallucination 동일/감소 → adapter 유지, 그 외 → adapter 폐기. 결과를 /output/eval_results/comparison.csv(수치), /output/eval_results/comparison_analysis.md(상세 분석)로 저장.

### 9.6. SFT system prompt 통일

**Status:** done  
**Dependencies:** None  

SFT 학습 system prompt에서 '항상 안전과 표준 절차를 우선하고' 제거. RAG 추론 system prompt('너는 선박 도장 분야 전문 AI 어시스턴트이다. 제공된 문서를 근거로 정확한 기술 정보를 제공하고...')와 동일하게 통일. SFT 훈련 코치 역할 프롬프트를 RAG 전문 AI 어시스턴트 프롬프트로 교체하여 추론 시 role/style 충돌 방지. generate_sft.py의 SYSTEM_PROMPT 상수 수정, qlora_config.yaml의 system_prompt 필드 업데이트.

### 9.7. SFT 답변 다양성 필터 구현

**Status:** done  
**Dependencies:** None  

SFT 데이터 생성 후 답변 다양성을 검증하는 필터 구현. (1) 동일 보일러플레이트로 시작하는 답변 비율을 20% 미만으로 제한 (현재 66.3%). (2) 답변 시작 패턴 분석: 첫 50자 기준 클러스터링으로 반복 패턴 탐지. (3) 기준 초과 시 해당 답변 자동 재생성 또는 제거. (4) generate_sft.py에 diversity_check() 함수 추가, review_sft.py에 보일러플레이트 비율 메트릭 추가. (5) 검증 기준: unique_start_ratio >= 0.8, max_single_pattern_ratio <= 0.15.

### 9.8. SFT 데이터 증량 1,000건+

**Status:** cancelled  
**Dependencies:** None  

현재 380건(effective unique ~130건)에서 최소 1,000건 이상으로 SFT 학습 데이터 증량. (1) effective unique 샘플 500건+ 확보 목표. (2) 청크 소스 다양화: 현재 10,829 chunks 중 미사용 청크 활용 비율 확대. (3) 카테고리별 증량 목표: 문서QA 250, 절차설명 200, 결함분석 150, DFT피드백 150, 용어/다국어 125, 요약/비교 125. (4) 생성 후 다양성 필터(9.7) 적용 필수. (5) generate_sft.py의 카테고리별 목표 수량 파라미터 업데이트.

### 9.9. Quality gate 엄격화 (SFT 품질 >= 1.5)

**Status:** done  
**Dependencies:** None  

SFT 품질 검수 기준 강화. (1) review_sft.py의 quality gate 기준을 1.5에서 통과가 아닌, 개별 샘플 레벨에서 점수 < 1.5 미달 샘플을 자동 제거하도록 변경. (2) 현재는 전체 평균만 확인(1.23으로도 PoC 통과)했으나 개별 샘플 필터링 필요. (3) 5차원 루브릭 중 기술정확성, 문서충실도 차원에서 0점인 샘플은 무조건 제거. (4) 필터링 후 남은 샘플 수 < 목표의 80%면 재생성 트리거. (5) quality_report.json에 제거된 샘플 수, 제거 사유 통계 추가.

### 9.10. 결함분석 카테고리 데이터 증강

**Status:** cancelled  
**Dependencies:** None  

결함분석(defects) 카테고리의 청크가 전체 대비 1.2%(128/10,829)로 매우 부족. (1) defects 도메인 청크 128개를 모두 활용한 집중 SFT 데이터 생성. (2) 결함 유형별(핀홀, 크레이터링, 오렌지필, 흘러내림, 박리 등) 균형 잡힌 QA 생성. (3) 결함 원인-현상-조치 3단계 답변 구조 표준화. (4) 결함분석 전용 프롬프트 템플릿 추가. (5) 카테고리 비율이 전체 SFT 데이터의 최소 12% 이상 되도록 증강.

### 9.11. 학습 하이퍼파라미터 최적화

**Status:** cancelled  
**Dependencies:** None  

현재 학습 파라미터(lr=2e-4, r=16, alpha=32) 재조정. (1) learning rate 5e-5로 하향(현재 2e-4는 과적합 유발 가능). (2) LoRA rank r=32 또는 r=64로 상향(표현력 증가). (3) lora_alpha=2×rank 유지 규칙 적용. (4) epochs 3→5로 증가(데이터 증량 시). (5) warmup_ratio 0.03→0.1로 조정. (6) qlora_config.yaml 업데이트 및 train_qlora.py에서 config 로드 확인. (7) 변경 전후 eval loss 및 token accuracy 비교.

### 9.12. RLHF/DPO 기반 hallucination 억제 설계

**Status:** cancelled  
**Dependencies:** None  

QLoRA 후속으로 hallucination 억제를 위한 RLHF/DPO 파이프라인 설계 문서 작성. (1) DPO(Direct Preference Optimization) 학습 데이터 구조 정의: chosen(올바른 답변)/rejected(hallucination 답변) 쌍. (2) Task 9 평가에서 발견된 fabricated KMS 번호 응답을 rejected 샘플로 활용. (3) trl DPOTrainer 사용 시 필요한 데이터 포맷 및 config 설계. (4) 예상 학습 비용(VRAM, 시간) 추정. (5) 설계 문서를 /workspace/trainer/configs/dpo_design.md로 저장. 구현은 2차 PoC 범위.
