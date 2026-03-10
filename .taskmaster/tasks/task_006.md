# Task ID: 6

**Title:** RAG-Only 평가 실행 및 분석

**Status:** pending

**Dependencies:** 4, 5

**Priority:** high

**Description:** 50문항 평가셋을 RAG-only 구성으로 일괄 실행, ROUGE-L/BERTScore 자동 평가, 수동 채점, hallucination 분류, 성공 기준(평균 1.5/2.0 이상, hallucination ≤10%) 확인, 결과 /output/eval_results/에 저장

**Details:**

1. 배치 실행: /workspace/trainer/scripts/run_rag_eval.py로 50문항 순차 실행, 응답 저장(q_id, answer, citations, timestamp)
2. 자동 평가: ROUGE-L(reference vs predicted), BERTScore F1, 응답 길이, citation 개수
3. hallucination 분류: 출처 없는 사실 주장, 사실 오류, 과장/추론 과다, 모순 등
4. 수동 채점: 5차원 루브릭 적용, 2명 검토자 평균 점수
5. 오류 분석: 저점수 문항 원인(검색 실패/LLM 오류/프롬프트 부적절) 분류
6. 결과 CSV: /output/eval_results/rag_only_results.csv에 q_id, category, manual_score, rouge_l, bertscore_f1, hallucination_flag, error_type, confidence 포함
7. 집계: 카테고리별 평균, 전체 평균, 표준편차, hallucination rate, citation coverage
8. 산출물: /output/eval_results/rag_only_results.csv, /output/eval_results/rag_only_analysis.md, /output/eval_results/error_samples.jsonl
9. Gate: 평균 점수 ≥1.5/2.0 또는 hallucination ≤10% 달성 시 다음 단계 진행

**Test Strategy:**

50개 결과 모두 /output/eval_results/rag_only_results.csv에 저장 확인, 자동 평가 지표 계산 완료, 수동 채점 점수 분포 통계(최소값/최대값/평균/중앙값), hallucination 발생 건수 카운트, 기준 충족 여부 확인

## Subtasks

### 6.1. 배치 실행 스크립트 검증 및 50문항 평가 일괄 실행

**Status:** pending  
**Dependencies:** None  

run_rag_eval.py 스크립트 동작 확인, 50개 평가셋 순차 실행, q_id/answer/citations/timestamp 응답 저장

**Details:**

/workspace/trainer/scripts/run_rag_eval.py 스크립트로 /data/eval/eval_set.jsonl의 50문항을 순차 실행합니다. 각 문항마다 질의→RAG 검색(/output/indexes/faiss.index)→LLM 응답 생성 프로세스를 거쳐 q_id, answer, citations(chunk_id 리스트), timestamp를 JSON 형식으로 저장합니다. 실행 시간, 메모리 사용량, API 에러 발생 여부를 로깅하고 실패한 문항이 있을 경우 재시도 메커니즘을 적용합니다. 최종 응답 50개를 /output/eval_results/rag_only_raw_responses.jsonl에 저장합니다.

### 6.2. 자동 평가 지표 계산(ROUGE-L, BERTScore, hallucination)

**Status:** pending  
**Dependencies:** 6.1  

50개 응답에 대해 ROUGE-L, BERTScore F1, 응답 길이, citation 개수 자동 계산 및 hallucination 분류

**Details:**

각 응답에 대해 평가셋의 reference 답변과 비교하여 ROUGE-L 점수(0~1)를 계산합니다. BERTScore F1 점수도 동일하게 계산합니다. 응답 길이(문자 수), citation 개수(citations 배열 크기)를 추출합니다. hallucination 분류: (1)출처 없는 사실 주장, (2)사실 오류, (3)과장/추론 과다, (4)모순을 manual review 또는 자동 휴리스틱으로 판단하여 hallucination_flag(true/false) 및 hallucination_type을 기록합니다. 결과를 /output/eval_results/rag_only_auto_metrics.csv에 저장(q_id, rouge_l, bertscore_f1, answer_length, citation_count, hallucination_flag, hallucination_type)합니다.

### 6.3. 수동 채점 수행 및 5차원 루브릭 적용

**Status:** pending  
**Dependencies:** 6.1  

2명 검토자가 50개 응답 각각에 대해 5차원 루브릭을 적용하여 수동 점수 부여 및 평균 산출

**Details:**

50개 응답에 대해 정의된 5차원 루브릭(예: 정확성, 완전성, 형식, 근거 제시, 명확성 각 0~2점)을 적용하여 2명의 검토자가 독립적으로 채점합니다. 각 검토자의 점수를 기록하고, 점수 불일치 시 협의를 통해 최종 점수를 결정하거나 평균값(최대 2점)을 산출합니다. 결과를 /output/eval_results/rag_only_manual_scores.csv에 저장(q_id, reviewer1_score, reviewer2_score, final_manual_score, disagreement_note)합니다. 루브릭 정의 및 채점 가이드를 /output/eval_results/rubric_definition.md에 기록합니다.

### 6.4. 오류 분석 및 저점수 문항 원인 분류

**Status:** pending  
**Dependencies:** 6.2, 6.3  

manual_score가 1.5 미만인 문항들의 오류 원인을 검색 실패/LLM 오류/프롬프트 부적절로 분류

**Details:**

final_manual_score < 1.5인 모든 문항을 대상으로 오류 원인을 분류합니다. (1)검색 실패: Top-K 검색 결과 관련성 낮음, citation 부재, (2)LLM 오류: 생성 오류, 할루시네이션, 문법 오류, (3)프롬프트 부적절: 프롬프트 길이 부족, 문맥 부족 등의 카테고리로 분류합니다. 각 저점수 문항의 original question, retrieved_docs, predicted_answer, reference_answer를 함께 기록하여 원인 파악을 용이하게 합니다. 결과를 /output/eval_results/rag_only_error_analysis.csv(q_id, manual_score, error_type, error_description, root_cause)와 /output/eval_results/error_samples.jsonl(상세 정보)에 저장합니다.

### 6.5. 결과 집계 및 성공 기준 검증

**Status:** pending  
**Dependencies:** 6.2, 6.3, 6.4  

모든 평가 지표를 종합하여 최종 결과 CSV 및 분석 리포트 작성, 성공 기준(평균 1.5/2.0 이상, hallucination ≤10%) 달성 여부 확인

**Details:**

/output/eval_results/rag_only_auto_metrics.csv, /output/eval_results/rag_only_manual_scores.csv, /output/eval_results/rag_only_error_analysis.csv를 병합하여 /output/eval_results/rag_only_results.csv 생성(q_id, category, manual_score, rouge_l, bertscore_f1, hallucination_flag, error_type, confidence). 카테고리별 평균 점수, 전체 평균, 표준편차, hallucination 발생 건수 및 비율(%), citation coverage 등을 계산합니다. 성공 기준 검증: (1)평균 manual_score ≥1.5/2.0 확인, (2)hallucination_flag=true인 건수 / 전체 50 ≤ 10% 확인. 검증 결과를 포함한 최종 분석 리포트를 /output/eval_results/rag_only_analysis.md(마크다운 형식)에 작성하고, 기준 달성 여부를 명확히 기록합니다.
