# Task ID: 4

**Title:** RAG 기준선 응답 시스템 구현

**Status:** pending

**Dependencies:** 3

**Priority:** high

**Description:** 질의응답 엔드-투-엔드 파이프라인 구현: 쿼리 임베딩→Top-K 검색→선택적 리랭킹→LLM 응답 생성→JSON 스키마 기반 답변 및 인용 반환, /workspace/rag_api/에 서버 코드 배치

**Details:**

1. 응답 스키마: {answer, citations(chunk_id 리스트), confidence(high/medium/low), model_name, timestamp, source_docs}
2. 프롬프트 템플릿: system role(산업 AI 어시스턴트, 근거 기반 답변, 모른다고 표현) + user 질문 + retrieved_docs
3. LLM 설정: Qwen2.5-7B-Instruct, 4bit 로딩(bitsandbytes NF4), temperature 0.1, max_tokens 768
4. retrieval 기본값: top-k=5, rerank on/off 비교
5. hallucination 방지: 근거 문서 부족 시 confidence='low', 답변 거부 패턴 포함
6. 응답 생성: retrieved chunk context를 프롬프트에 삽입, LLM 추론, JSON 파싱
7. 스크립트: /workspace/trainer/scripts/qa_pipeline.py(retrieval→prompt→inference), response_formatter.py
8. 구현 체크리스트: 10문항 smoke test 통과, citation 누락 0건, hallucination 유형 카탈로그(최소 5가지)
9. 설정: /workspace/trainer/configs/rag_config.yaml에서 temperature, max_tokens, top_k, rerank_enabled 조정 가능

**Test Strategy:**

10개 수동 테스트 문항 실행, 응답 JSON 스키마 유효성(citation field 필수), 응답 시간 <10초, hallucination 주요 유형 5가지 기록, confidence 분포 확인

## Subtasks

### 4.1. LLM 모델 로딩 및 토크나이저 설정

**Status:** pending  
**Dependencies:** None  

Qwen2.5-7B-Instruct 모델을 4bit 양자화(bitsandbytes NF4)로 로딩하고, 추론 파라미터(temperature=0.1, max_tokens=768) 설정

**Details:**

1. /hf-cache/에서 Qwen2.5-7B-Instruct 모델 로드
2. bitsandbytes NF4 4bit 양자화 적용
3. 토크나이저 초기화 및 패딩/트림 설정
4. 추론 파라미터 설정: temperature=0.1(낮은 창의성), max_tokens=768
5. VRAM 사용량 검증(A40 48GB 이하)
6. /workspace/trainer/configs/rag_config.yaml에서 temperature, max_tokens 조정 가능하도록 구현

### 4.2. 검색 및 리랭킹 파이프라인 구현

**Status:** pending  
**Dependencies:** 4.1  

쿼리 임베딩 생성 → /output/indexes/faiss.index에서 Top-K(기본값 5) 검색 → 선택적 리랭킹(on/off 비교) 구현

**Details:**

1. 쿼리 임베딩 생성: sentence-transformers 또는 FlagEmbedding 모델 사용
2. FAISS 인덱스 로드 및 검색(cosine similarity, top_k=5)
3. 리랭킹 옵션: rerank_enabled 플래그로 on/off 제어
4. 리랭킹 모델(선택): bge-reranker-base 또는 cross-encoder 사용
5. 검색 결과: chunk_id, content, score, source_doc 포함
6. /workspace/trainer/scripts/qa_pipeline.py에서 검색 로직 구현
7. /workspace/trainer/configs/rag_config.yaml: top_k, rerank_enabled 설정 변수화

### 4.3. 프롬프트 템플릿 및 문맥 삽입 구현

**Status:** pending  
**Dependencies:** 4.2  

산업 AI 어시스턴트 역할의 system prompt + 사용자 질문 + retrieved_docs 문맥 구성, 근거 기반 답변 및 거부 패턴 포함

**Details:**

1. System prompt 작성: '산업 AI 어시스턴트, 정확한 기술 정보 제공, 근거 부족 시 모른다고 표현'
2. 검색된 문서 문맥 삽입: "Retrieved Documents:\n{chunk_content}\n" 형식
3. 사용자 쿼리 삽입: "User Question: {query}\n\nPlease answer based on the documents above."
4. 거부 패턴: 신뢰도 낮을 시 "해당 정보가 제공된 문서에 없습니다" 응답 명시
5. 프롬프트 길이 관리: 최대 2000 토큰 이내(모델 입력 토큰 제한)
6. /workspace/trainer/scripts/qa_pipeline.py에서 프롬프트 빌더 클래스 구현

### 4.4. 응답 생성 및 JSON 스키마 파싱

**Status:** pending  
**Dependencies:** 4.3  

LLM 추론 실행 → 출력 파싱 → JSON 스키마 기반 응답(answer, citations, confidence, metadata) 반환

**Details:**

1. LLM 추론: 준비된 프롬프트로 Qwen2.5-7B-Instruct 실행
2. 출력 파싱: JSON 형식 강제(structured output 또는 regex 기반)
3. 응답 스키마: answer(생성된 답변 텍스트), citations(chunk_id 리스트), confidence('high'/'medium'/'low'), model_name('Qwen2.5-7B-Instruct'), timestamp(ISO 8601), source_docs(메타데이터)
4. 파싱 오류 처리: 유효하지 않은 JSON 시 기본값으로 대체
5. /workspace/trainer/scripts/response_formatter.py에서 formatter 클래스 구현
6. citation validation: 모든 citations이 검색 결과에 존재하는지 확인

### 4.5. 할루시네이션 방지 및 신뢰도 검증

**Status:** pending  
**Dependencies:** 4.4  

근거 부족 시 confidence='low' 설정, 주요 할루시네이션 5가지 유형 카탈로그 작성, 방지 패턴 구현

**Details:**

1. 할루시네이션 감지 규칙: 검색 결과 0건→confidence='low', 점수 합 <0.5→confidence='low', semantic 유사도 <0.6→confidence='low'
2. 할루시네이션 유형 카탈로그(최소 5가지): (a)비존재 제품/규격명, (b)숫자 왜곡, (c)공정 단계 오류, (d)안전 기준 임의 수정, (e)없는 근거 문서 인용
3. 방지 패턴: "위 문서에 명시되지 않은 정보입니다" 응답 패턴, 신뢰도 점수=(검색점수합+semantic유사도)/2
4. /workspace/trainer/scripts/qa_pipeline.py에서 confidence 산출 로직 구현
5. 할루시네이션 로그: 탐지된 사례 기록(/output/hallucination_log.jsonl)
