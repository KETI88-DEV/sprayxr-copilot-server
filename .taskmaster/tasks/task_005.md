# Task ID: 5

**Title:** 50문항 평가셋 구성 및 검수

**Status:** done

**Dependencies:** 2 ✓

**Priority:** high

**Description:** 용어(10문), 공정절차(10문), 결함분석(10문), 코팅두께/사양(10문), 안전/규정(10문) 균형잡힌 평가셋 작성, /data/eval/에 저장, 참조 답변 및 채점 루브릭 확정

**Details:**

1. 5개 카테고리별 10문항씩 설계
2. 각 문항: question, expected_answer, difficulty(easy/medium/hard), category, required_chunks(참조 chunk_id 리스트)
3. 참조 답변: 도메인 전문가 검수, 기술 정확성/완전성/한국어 품질 확인
4. JSON 포맷: {q_id, question, category, expected_answer, required_chunks, difficulty}
5. 채점 루브릭: 5차원(기술정확성, 문서충실도, 용어적절성, 완전성, 한국어품질) × 0-2점 = 최대 10점
6. 자동 평가 지표: ROUGE-L(rouge-score), BERTScore(bert-score), hallucination ratio, 용어 정확도
7. 검수 프로세스: 2명 이상 도메인 검토자 채점, 의견 불일치 시 합의
8. 파일: /data/eval/eval_set.jsonl, /output/eval_results/eval_rubric.md, /output/eval_results/eval_answer_key.json
9. 통계: 카테고리별 난이도 분포, 필요 문서 coverage 확인

**Test Strategy:**

50문항 모두 JSON 유효성 확인, 카테고리별 10문항 정확히 배분 검증, 참조 답변 평균 길이 100-300 토큰 범위 확인, required_chunks 모두 실제 /data/chunked/chunks.jsonl에 존재 확인

## Subtasks

### 5.1. 5개 카테고리별 50문항 설계 및 JSON 구조화

**Status:** done  
**Dependencies:** None  

용어(10문), 공정절차(10문), 결함분석(10문), 코팅두께/사양(10문), 안전/규정(10문) 총 50문항을 설계하고 JSON 포맷으로 구조화합니다.

**Details:**

각 문항에 q_id, question, category, expected_answer, required_chunks(참조 chunk_id 리스트), difficulty(easy/medium/hard) 필드를 포함하여 설계합니다. 카테고리별 난이도 분포를 균형있게 배분하고(easy:4, medium:4, hard:2 기준), 각 문항이 /data/chunked/chunks.jsonl의 실제 chunk_id를 참조하도록 합니다. /data/eval/eval_set.jsonl에 JSONL 포맷으로 저장합니다.

### 5.2. 도메인 전문가 검수를 위한 참조 답변 작성 및 검증

**Status:** done  
**Dependencies:** 5.1  

50문항 각각에 대해 도메인 전문가가 참조 답변을 작성하고, 기술 정확성/완전성/한국어 품질을 검증합니다.

**Details:**

각 문항의 expected_answer를 100-300 토큰 범위의 상세 답변으로 작성합니다. 2명 이상의 도메인 전문가가 각 답변을 검수하고, 기술 정확성(관련 기준/규격 준수), 문서 충실도(chunks 내용 반영), 완전성(필수 정보 포함)을 평가합니다. 전문가 의견 불일치 시 합의하여 최종 답변을 확정합니다. /data/eval/eval_answer_key.json에 저장합니다.

### 5.3. 5차원 채점 루브릭 설계 및 확정

**Status:** done  
**Dependencies:** 5.2  

기술정확성, 문서충실도, 용어적절성, 완전성, 한국어품질 5차원에 대해 0-2점 척도의 채점 루브릭을 설계하고 확정합니다.

**Details:**

각 차원별로 0점(미달), 1점(부분충족), 2점(완전충족)의 명확한 기준을 정의합니다. 기술정확성: 기준/규격 정확도, 문서충실도: 제공 chunks 충실도, 용어적절성: 산업 표준 용어 사용, 완전성: 필수 정보 포함도, 한국어품질: 명확성/자연스러움. 최대 10점 만점 채점이 가능하도록 설계합니다. /workspace/trainer/configs/eval_rubric.md에 상세 설명과 함께 저장합니다.

### 5.4. 자동 평가 지표 구현 (ROUGE-L, BERTScore, hallucination 감지)

**Status:** done  
**Dependencies:** 5.2, 5.3  

ROUGE-L, BERTScore, hallucination ratio, 용어 정확도를 계산하는 자동 평가 지표를 구현합니다.

**Details:**

rouge-score 라이브러리로 ROUGE-L 점수 계산, bert-score 라이브러리로 semantic 유사도 평가. hallucination 감지는 generated answer와 required_chunks의 정보 일치도를 측정합니다. 용어 정확도는 산업 표준 용어 사용 비율 계산. 각 지표를 0-1 범위의 점수로 정규화합니다. /workspace/trainer/scripts/eval_metrics.py에 구현하고, 50문항 모두에 대해 자동 평가 점수를 생성합니다.

### 5.5. 최종 평가셋 통계 분석 및 품질 검수

**Status:** done  
**Dependencies:** 5.4  

완성된 50문항 평가셋의 카테고리별 난이도 분포, 문서 coverage, 최종 통계를 작성하고 전체 품질을 검증합니다.

**Details:**

카테고리별 난이도 분포 히스토그램 생성(5개 카테고리 × 3개 난이도), 필요 문서 coverage 확인(모든 required_chunks가 실제 chunks에 포함), 참조 답변 길이/토큰 수 분포, 자동 평가 지표 분포(ROUGE-L, BERTScore, hallucination ratio). 최종 평가셋의 품질 점수(참조 답변 전문가 검수 점수 평균)를 계산합니다. /output/eval_results/eval_statistics.json에 모든 통계를 저장하고 요약 리포트를 /output/eval_results/eval_quality_report.md에 작성합니다.
