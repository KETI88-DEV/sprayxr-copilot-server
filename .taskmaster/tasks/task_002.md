# Task ID: 2

**Title:** 문서 수집 및 파이프라인 설계

**Status:** pending

**Dependencies:** None

**Priority:** high

**Description:** 선박 도장 관련 문서(SOP, 매뉴얼, TDS, 안전지침, 결함 가이드, 도막 두께 기준) 50-100건 수집, /data/raw/에 저장, 파싱 규칙 정의, 정규화 및 중복 제거 전략 수립

**Details:**

1. 1순위 문서 수집: coating/blasting 매뉴얼, 도료 TDS, 안전지침, 결함 설명, 도막 두께/검사 가이드 → /data/raw/
2. 2순위 교육 자료, RealCOAT 시나리오 정의서, 훈련 평가 기준표
3. 문서 정규화 규칙: Unicode NFKC, 헤더/푸터 제거, 페이지 번호 제거, 단위 통일(µm, %, mm, °C), 용어 통일(DFT, WFT, SSPC-SP10, Sa 2.5)
4. 중복 제거: 문서 단위 SHA-256, near-duplicate MinHash+LSH, 문단 512자 이상 장문 중복 제거
5. 파싱 포맷: PDF→pymupdf, HTML→beautifulsoup4, 표는 markdown/CSV 보존
6. metadata schema 설계: doc_id, chunk_id, source_type, domain_category, process_stage, subtask, language, revision, quality_score, has_table, has_procedure
7. 프롬프트/스크립트: /workspace/trainer/scripts/ingest_docs.py, clean_docs.py, dedup_docs.py 스켈레톤 생성

**Test Strategy:**

수집된 문서 목록 /data/manifest.csv 생성(doc_id, source, pages, bytes), 정규화 전후 샘플 비교(5건), dedup 결과 검증(중복 제거율 ≥95%), 메타데이터 스키마 JSON 유효성 확인, /data/raw/ 문서 저장 확인

## Subtasks

### 2.1. 선박 도장 문서 50-100건 수집 및 /data/raw/ 저장

**Status:** pending  
**Dependencies:** None  

1순위 문서(코팅/블라스팅 매뉴얼, 도료 TDS, 안전지침, 결함 설명, 도막 두께/검사 가이드)를 50-100건 수집하여 /data/raw/에 저장하고, /data/manifest.csv에 문서 메타데이터(doc_id, source, pages, bytes) 기록

**Details:**

1순위 문서 카테고리별로 수집: coating 매뉴얼, blasting 매뉴얼, 도료 TDS, 안전지침, 결함 가이드, 도막 두께 기준. PDF/HTML 등 다양한 포맷 지원. /data/raw/ 디렉터리 구조: /raw/manuals/, /raw/tds/, /raw/safety/, /raw/defects/, /raw/thickness/. 각 문서마다 doc_id 부여(doc_001, doc_002 등), 원본 파일명 기록. 2순위 문서(교육 자료, RealCOAT 시나리오)는 선택사항. 최종 manifest.csv: doc_id, source, pages, bytes, format, upload_date 컬럼 포함.

### 2.2. 문서 정규화 규칙 정의 및 clean_docs.py 스켈레톤 작성

**Status:** pending  
**Dependencies:** 2.1  

Unicode NFKC 정규화, 헤더/푸터/페이지 번호 제거, 단위 통일(µm→µm, %→%, mm→mm, °C→°C), 용어 통일(DFT/WFT/SSPC-SP10/Sa 2.5) 규칙을 정의하고 /workspace/trainer/scripts/clean_docs.py 스켈레톤 작성

**Details:**

정규화 규칙 문서화: /workspace/trainer/configs/normalization_rules.yaml에 규칙 정의 (단위 매핑 테이블, 용어 치환 딕셔너리). clean_docs.py 구조: 입력(/data/raw/), 처리(정규화 함수들), 출력(/data/cleaned/). 정규화 단계: (1) Unicode NFKC, (2) 특수문자 정리, (3) 헤더/푸터 제거(정규식), (4) 페이지 번호 제거, (5) 단위 통일, (6) 용어 대체. 샘플 문서 5건에 대해 정규화 전후 비교 가능하도록 로깅 포함. argparse로 입출력 경로 지정 가능.

### 2.3. 중복 제거 전략 구현 및 dedup_docs.py 스켈레톤 작성

**Status:** pending  
**Dependencies:** 2.2  

문서 단위 SHA-256, near-duplicate MinHash+LSH, 문단 512자 이상 장문 중복 제거 전략을 구현하고 /workspace/trainer/scripts/dedup_docs.py 스켈레톤 작성, 중복 제거율 ≥95% 달성

**Details:**

중복 제거 3단계: (1) 문서 단위 SHA-256 해시 기반 정확 중복 제거, (2) near-duplicate MinHash+LSH 기반 유사 문서 감지(Jaccard similarity threshold 0.8 이상), (3) 문단 512자 이상 장문 중복 제거(슬라이딩 윈도우). dedup_docs.py: 입력(/data/cleaned/), 처리(3단계 중복 제거), 출력(/data/deduplicated/). 중복 제거 결과 레포트: 원본 문서 수, 정확 중복 수, near-duplicate 수, 장문 중복 수, 최종 문서 수, 중복 제거율. datasketch 라이브러리 활용 예정.

### 2.4. 메타데이터 스키마 설계 및 chunk_docs.py 스켈레톤 작성

**Status:** pending  
**Dependencies:** 2.3  

메타데이터 스키마(doc_id, chunk_id, source_type, domain_category, process_stage, subtask, language, revision, quality_score, has_table, has_procedure) 정의하고 /workspace/trainer/scripts/chunk_docs.py 스켈레톤 작성, 각 청크에 메타데이터 부여

**Details:**

메타데이터 스키마 JSON 정의: /workspace/trainer/configs/metadata_schema.json. 각 청크 구조: {chunk_id, doc_id, text, source_type(PDF/HTML/etc), domain_category(coating/blasting/safety/defects/thickness), process_stage(preparation/application/inspection/QC), subtask(paint_mixing/thickness_measurement), language(ko), revision(v1.0), quality_score(0.0-1.0), has_table(bool), has_procedure(bool), page_num, offset}. chunk_docs.py: 입력(/data/deduplicated/), 처리(텍스트 청킹 512-1024 토큰 크기), 메타데이터 자동 추출, 출력(/data/chunked/). 표는 markdown/CSV 형식으로 별도 보존.

### 2.5. 파싱 포맷 정의 및 ingest_docs.py 스켈레톤 작성

**Status:** pending  
**Dependencies:** 2.1  

PDF→pymupdf, HTML→beautifulsoup4 파싱 포맷 정의하고 /workspace/trainer/scripts/ingest_docs.py 스켈레톤 작성, /data/raw/→/data/parsed/ 파이프라인 구축

**Details:**

ingest_docs.py 파이프라인: 입력(/data/raw/), 포맷 감지(확장자 기반), 파싱(pymupdf for PDF, BeautifulSoup for HTML), 메타데이터 추출, 출력(/data/parsed/). pymupdf 옵션: 페이지 단위 텍스트 추출, 표 감지(table_finder). BeautifulSoup 옵션: 헤더/푸터 제거, 스크립트/스타일 태그 제거, 마크다운 변환. 파싱 결과: JSON 형식 저장 {doc_id, format, pages, content, tables[], extracted_at}. 에러 처리: 파싱 실패 문서 로깅, 재시도 메커니즘. argparse로 입출력 경로, 포맷 옵션 지정 가능.
