# Task ID: 9

**Title:** QLoRA-Only 및 RAG+QLoRA 평가 실행

**Status:** pending

**Dependencies:** 6, 8

**Priority:** medium

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

**Status:** pending  
**Dependencies:** None  

학습된 adapter를 로드하여 RAG 검색 없이 QLoRA-Only 구성으로 50문항 일괄 평가 실행

**Details:**

Qwen2.5-7B 모델에 /output/adapters/qwen25_shipcoat_lora/ adapter만 로드하여 RAG 검색을 미사용한 상태에서 /data/eval/eval_set.jsonl의 50문항을 순차 실행. /workspace/trainer/scripts/run_rag_eval.py 스크립트 실행, temperature=0.1, max_tokens=768 설정. 각 문항별 q_id, answer, timestamp를 기록하고 /output/eval_results/qlora_only_results.csv에 저장. adapter 로드 시간, 각 문항 추론 시간 기록.

### 9.2. RAG+QLoRA 구성으로 50문항 평가 실행

**Status:** pending  
**Dependencies:** 9.1  

Qwen2.5-7B + adapter + RAG 검색을 통합한 RAG+QLoRA 구성으로 동일 50문항 평가 실행

**Details:**

QLoRA-Only 평가와 동일한 50문항을 RAG+QLoRA 구성으로 실행. Qwen2.5-7B + /output/adapters/qwen25_shipcoat_lora/ adapter + RAG 검색(top-k=5) 활성화. 동일한 프롬프트 템플릿 사용하되 retrieved_docs를 포함. temperature=0.1, max_tokens=768. 각 문항별 q_id, answer, citations, timestamp, rag_context 기록하고 /output/eval_results/ragqlora_results.csv에 저장. 검색 시간, 추론 시간 분리 측정.

### 9.3. QLoRA-Only 및 RAG+QLoRA 자동 평가 지표 계산

**Status:** pending  
**Dependencies:** 9.2  

두 구성의 평가 결과에 대해 ROUGE-L, BERTScore, hallucination ratio, 응답 길이 등 자동 평가 지표 계산

**Details:**

QLoRA-Only과 RAG+QLoRA 결과 CSV에 대해 자동 평가 실행. /data/eval/eval_set.jsonl의 expected_answer를 reference로 사용하여 ROUGE-L(rouge-score 라이브러리), BERTScore F1(bert-score 라이브러리) 계산. 응답 길이(토큰 수), citation 개수 기록. hallucination ratio 계산: 출처 없는 사실 주장, 사실 오류, 과장/추론 과다, 모순 등을 pattern matching 및 수동 샘플링으로 분류. 각 구성별 지표를 qlora_only_metrics.csv, ragqlora_metrics.csv로 저장.

### 9.4. QLoRA-Only 및 RAG+QLoRA 수동 채점 실행

**Status:** pending  
**Dependencies:** 9.2  

두 구성의 평가 결과에 대해 5차원 루브릭을 적용하여 수동 채점 실행

**Details:**

QLoRA-Only 50문항과 RAG+QLoRA 50문항 결과에 대해 동일 검수자 또는 검증 인원이 5차원 루브릭(기술정확성, 문서충실도, 용어적절성, 완전성, 한국어품질) × 0-2점 = 최대 10점을 적용하여 채점. 두 구성 각각을 블라인드 처리하여 편향 제거. 검수자별 점수를 기록하고 평균값 계산. qlora_only_manual_scores.csv, ragqlora_manual_scores.csv로 저장(q_id, manual_score, reviewer_name, 각 차원별 점수).

### 9.5. 3가지 구성 비교 분석 및 Keep/Drop 의사결정

**Status:** pending  
**Dependencies:** 9.3, 9.4  

RAG-Only vs QLoRA-Only vs RAG+QLoRA 3가지 구성을 종합적으로 비교 분석하고 adapter 유지 여부 의사결정

**Details:**

RAG-Only(/output/eval_results/rag_only_results.csv), QLoRA-Only, RAG+QLoRA 결과를 종합하여 비교 분석. 1) 평균 점수 비교: 세 구성별 수동 채점 평균값, 자동 평가 지표 평균값(ROUGE-L, BERTScore) 계산 및 비교표 작성. 2) hallucination 비율 비교: 세 구성의 hallucination 건수, 비율(%) 비교. 3) 용어 정확도, 형식 일관성: 오류 유형별 분포, 카테고리별 성능 분석. 4) 카테고리별 성능: 5개 카테고리(용어, 공정절차, 결함분석, 코팅두께/사양, 안전/규정)별 평균 점수 비교. Keep/Drop 의사결정: RAG+QLoRA 평균 점수가 RAG-Only 대비 0.15점 이상 향상 AND hallucination 동일/감소 → adapter 유지, 그 외 → adapter 폐기. 결과를 /output/eval_results/comparison.csv(수치), /output/eval_results/comparison_analysis.md(상세 분석)로 저장.
