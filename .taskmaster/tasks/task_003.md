# Task ID: 3

**Title:** 문서 청킹 및 RAG 색인 구축

**Status:** done

**Dependencies:** 1 ✓, 2 ✓

**Priority:** high

**Description:** 정제된 문서를 300-800 토큰 기반으로 구조적 청킹, BGE-M3 임베딩 생성, FAISS 색인 구축 및 /output/indexes/에 저장, 리랭커 통합

**Details:**

1. 청킹 규칙: 절/항/Step 경계 우선, checklist/procedure는 한 chunk 유지, 표는 제목/컬럼명과 함께 별도 chunk, 10-20% overlap
2. 토큰 카운팅: transformers.AutoTokenizer로 Qwen2.5 토크나이저 기준
3. 임베딩: sentence-transformers로 BGE-M3 로드, batch_size 32, device 'cuda'
4. FAISS 인덱스: IVFFlat/HNSW 비교, base model 선택, metric L2
5. chunk_id 생성: DOC-{doc_id}-C{seq} 형식
6. chunks.jsonl: /data/chunked/chunks.jsonl에 각 행 {chunk_id, text, doc_id, tokens, embedding} 포함
7. FAISS 저장: /output/indexes/faiss.index, /output/indexes/faiss.index.pkl(metadata)
8. 선택적: BAAI/bge-reranker-v2-m3 통합, retrieval top-5 후 rerank top-3
9. 스크립트: /workspace/trainer/scripts/chunk_docs.py, build_index.py, 설정 /workspace/trainer/configs/rag_config.yaml

**Test Strategy:**

/data/chunked/chunks.jsonl 행 수 확인(최소 200), 임베딩 차원 1024 확인, /output/indexes/faiss.index 저장 성공 및 파일 크기 >100MB, smoke test 5개 쿼리 retrieval 수행(응답 <5초), citation chunk_id 누락률 0%

## Subtasks

### 3.1. 구조적 청킹 규칙 구현 및 토큰 카운팅

**Status:** done  
**Dependencies:** None  

절/항/Step 경계 우선 청킹 규칙 정의 및 Qwen2.5 토크나이저 기반 토큰 카운팅 로직 구현, 300-800 토큰 범위 유지 및 10-20% overlap 처리

**Details:**

1. transformers.AutoTokenizer로 Qwen2.5 토크나이저 로드
2. 청킹 규칙 구현: 절/항/Step 경계 우선, checklist/procedure는 한 chunk 유지, 표는 제목/컬럼명과 함께 별도 chunk
3. 오버랩 로직: 10-20% 범위에서 토큰 기반 오버랩 구현
4. /workspace/trainer/scripts/chunk_docs.py 작성
5. 테스트: 샘플 문서 5건으로 청킹 결과 검증(토큰 범위 확인, 오버랩 비율 확인)

### 3.2. BGE-M3 임베딩 생성 및 JSONL 저장

**Status:** done  
**Dependencies:** 3.1  

/hf-cache/에서 BGE-M3 모델 로드, 청킹된 문서에 대해 배치 임베딩 생성, chunk_id(DOC-{doc_id}-C{seq}) 형식으로 /data/chunked/chunks.jsonl 생성

**Details:**

1. sentence-transformers로 BGE-M3 모델 로드(/hf-cache/에서)
2. 배치 임베딩 생성: batch_size 32, device 'cuda' 사용
3. chunk_id 생성 로직: DOC-{doc_id}-C{seq} 형식 구현
4. /data/chunked/chunks.jsonl 생성: 각 행 {chunk_id, text, doc_id, tokens, embedding} 포함
5. chunk_docs.py 스크립트에 임베딩 처리 로직 추가
6. GPU 메모리 관리: batch_size 32에서 out-of-memory 방지

### 3.3. FAISS 인덱스 구축 및 메타데이터 저장

**Status:** done  
**Dependencies:** 3.2  

IVFFlat/HNSW 비교를 통해 최적 인덱스 선택, /data/chunked/chunks.jsonl의 임베딩으로 FAISS 인덱스 구축, metric L2 기준으로 /output/indexes/에 저장

**Details:**

1. /workspace/trainer/scripts/build_index.py 작성
2. IVFFlat 및 HNSW 인덱스 구현 및 성능 비교(인덱싱 시간, 검색 속도, 메모리)
3. 최적 인덱스 선택 및 구축: metric L2 기반
4. FAISS 인덱스 저장: /output/indexes/faiss.index
5. 메타데이터 저장: /output/indexes/faiss.index.pkl (chunk_id, doc_id, text 매핑 정보)
6. 설정 파일: /workspace/trainer/configs/rag_config.yaml에 인덱스 타입, 파라미터 기록

### 3.4. 리랭커 통합 및 RAG 파이프라인 검증

**Status:** done  
**Dependencies:** 3.3  

BAAI/bge-reranker-v2-m3 모델 통합, retrieval top-5 후 rerank top-3 로직 구현, smoke test 5개 쿼리로 응답 시간 및 품질 검증

**Details:**

1. BAAI/bge-reranker-v2-m3 모델 로드 및 통합
2. RAG 파이프라인: FAISS retrieval (top-5) → rerank (top-3)
3. /workspace/trainer/scripts/build_index.py에 reranker 로직 추가
4. 프롬프트 포맷: '[문서] ... [질문] ...'
5. Smoke test: 5개 샘플 쿼리로 end-to-end 검증
6. 응답 시간 측정: 검색+리랭킹 총 시간 <5초 목표

### 3.5. RAG 인덱싱 전체 통합 테스트 및 보고서 작성

**Status:** done  
**Dependencies:** 3.4  

전체 청킹→임베딩→인덱싱→리랭킹 파이프라인 통합 테스트, 성능 메트릭 수집, 최종 검증 보고서 작성

**Details:**

1. 전체 파이프라인 end-to-end 테스트: /data/raw/ 전체 문서 처리
2. 성능 메트릭 수집: 처리 시간, chunks 수량, 임베딩 차원, 인덱스 크기, 검색 시간, rerank 효과
3. 데이터 품질 검증: chunks.jsonl 행 수, embedding 유효성, chunk_id 형식
4. FAISS 인덱스 검증: 로드 성공, 쿼리 수행 정상
5. 최종 보고서 작성: /output/eval_results/rag_indexing_report.md
6. /output/indexes/ 최종 산출물 확인: faiss.index, faiss.index.pkl 존재
