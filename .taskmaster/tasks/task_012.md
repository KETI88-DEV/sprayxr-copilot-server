# Task ID: 12

**Title:** 최종 평가 리포트 및 의사결정 문서 작성

**Status:** pending

**Dependencies:** 9 ✓, 19 ⧖

**Priority:** high

**Description:** RAG-Only, QLoRA-Only, RAG+QLoRA 3가지 구성 비교표, 성공 기준 달성 여부, keep/drop 의사결정, 다음 단계 제안 포함한 최종 리포트 작성, /output/eval_results/에 저장

**Details:**

1. 리포트 구성:
   - 실행 요약(Executive Summary): 목표, 구성, 주요 결과, 의사결정
   - 기준선 비교표: 모델별 평균 점수, hallucination율, 응답 시간, 메모리 사용, 추론 비용
   - 상세 결과: 카테고리별 성능, 오류 분석, hallucination 유형 분포
   - 성공 기준 검증표: 각 기준별 달성 여부
   - 기술 의사결정: adapter keep/drop 근거, 문제점 및 해결방안
   - 다음 단계: VLM stub 정의, 세션 export 자동화, 다국어 확장
2. 포맷: /output/eval_results/final_report.md (마크다운), /output/eval_results/final_report.xlsx(표)
3. 수치 데이터: CSV로 완전히 제공 가능해야 함
4. 시각화: 카테고리별 성능 그래프, 모델 비교 차트
5. 근거 자료: 평가 결과 상세 파일 첨부(/data/eval/eval_set.jsonl, 결과 CSV)
6. 결론: 1차 PoC의 주요 학습, 기술적 검증, 다음 단계 우선순위

**Test Strategy:**

/output/eval_results/final_report.md, /output/eval_results/final_report.xlsx 파일 생성 확인, 모든 수치 근거 파일과 연결 확인, 표 데이터 정확성 검증(sum, avg 재계산), 의사결정 근거 명확성 검토

## Subtasks

### 12.1. 평가 결과 CSV 데이터 수집 및 통합

**Status:** pending  
**Dependencies:** None  

RAG-Only, QLoRA-Only, RAG+QLoRA 3가지 구성의 평가 결과 CSV 파일들을 /output/eval_results/에서 수집하여 통합 데이터셋 생성

**Details:**

Task 6과 Task 9에서 생성된 rag_only_results.csv, qlora_only_results.csv, rag_qlora_results.csv를 읽어들여 통합 비교 테이블 생성. 각 파일에서 q_id, category, manual_score, rouge_l, bertscore_f1, hallucination_flag, error_type, confidence 컬럼 추출 및 정규화. 결측값 처리 및 데이터 타입 검증. comparison.csv 생성으로 3가지 모델 구성 나란히 비교 가능하도록 정렬.

### 12.2. 성공 기준 달성 여부 검증표 작성

**Status:** pending  
**Dependencies:** 12.1  

정의된 성공 기준(평균 1.5/2.0 이상, hallucination ≤10%, 응답시간, 메모리 사용, 추론 비용)에 대해 3가지 구성별 달성 여부를 정량적으로 검증

**Details:**

성공 기준별로 RAG-Only, QLoRA-Only, RAG+QLoRA의 실제 수치 비교. 예: 평균 점수 기준(1.5/2.0 이상) - 각 구성별 평균값 계산 및 기준 충족 여부(O/X), hallucination 기준(≤10%) - 각 구성별 hallucination 발생률 계산, 응답시간/메모리/추론비용은 평가 실행 로그에서 추출. 검증표를 마크다운 테이블 형식으로 작성하여 명확한 pass/fail 표시. 기준 미충족 항목에 대한 편차값(예: 기준 1.5 vs 실제 1.42) 명시.

### 12.3. 기술 의사결정 문서 작성 (Adapter Keep/Drop)

**Status:** pending  
**Dependencies:** 12.2  

QLoRA adapter 유지/제거 의사결정을 위해 성능 향상 근거, 비용-효과 분석, 문제점 및 해결방안을 상세히 기술

**Details:**

Keep 조건: RAG+QLoRA가 RAG-Only 대비 평균 0.15점 이상 향상 && hallucination 동일/감소. 이 조건의 충족 여부를 정량적 수치로 입증(예: RAG-Only 평균 1.65점 vs RAG+QLoRA 1.82점, 향상도 +0.17). QLoRA 미사용 시 비용 절감(메모리, 추론 시간, 저장소) vs 사용 시 성능 향상의 trade-off 분석. 발견된 문제점(hallucination 증가 양상, 특정 카테고리 성능 저하 등) 상세 기술 및 향후 개선방안 제시. 최종 의사결정(Keep/Drop) 명확히 결론 도출.

### 12.4. 시각화 및 비교 차트 생성

**Status:** pending  
**Dependencies:** 12.1  

카테고리별 성능 그래프, 3가지 모델 구성 비교 차트 등 마크다운 리포트에 포함될 시각화 자산 생성

**Details:**

Python matplotlib/plotly를 사용하여 다음 차트 생성: (1) 카테고리별 성능 비교(막대/선 그래프, 5개 카테고리×3가지 구성), (2) 모델별 평균 점수 비교(막대 차트), (3) hallucination 비율 비교(막대 또는 원형 차트), (4) 각 평가 지표(ROUGE-L, BERTScore) 분포(박스플롯), (5) 오류 유형 분포(파이 또는 막대 차트). 차트 이미지를 /output/eval_results/images/ 디렉토리에 PNG 형식으로 저장. 차트 제목, 축 레이블, 범례는 명확하고 한국어로 표시.

### 12.5. 최종 리포트 작성 및 결과 문서 패키징

**Status:** pending  
**Dependencies:** 12.2, 12.3, 12.4  

실행 요약, 기준선 비교표, 상세 결과, 성공 기준 검증, 의사결정, 다음 단계를 포함한 최종 마크다운 리포트 작성 및 Excel 비교표 생성

**Details:**

마크다운 리포트 구조: (1) 실행 요약(Executive Summary): 1차 PoC 목표/구성 요약, 주요 결과(3가지 구성 평균 점수, hallucination율), 의사결정 요약(keep/drop), (2) 기준선 비교표: 모델별 평균 점수/hallucination율/응답시간/메모리/추론비용을 테이블로 정렬, (3) 상세 결과: 카테고리별 성능(표 + 그래프), 오류 분석(오류 유형 분포, 저점수 문항 원인), hallucination 유형 분석, (4) 성공 기준 검증표(Task 2 산출물), (5) 기술 의사결정(Task 3 산출물), (6) 다음 단계: VLM stub 정의, 세션 export 자동화, 다국어 확장 우선순위 설정. Excel 파일은 comparison.csv 데이터로 피벗 테이블 및 기본 통계 시트 포함. /output/eval_results/final_report.md, /output/eval_results/final_report.xlsx 생성. 부록으로 원본 평가 CSV, 평가셋(eval_set.jsonl) 참조 정보 포함.
