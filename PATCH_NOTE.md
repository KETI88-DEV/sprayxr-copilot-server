# PATCH NOTE

---

## [Task 1] 개발 환경 구축 및 GPU 검증

**Status**: ✅ DONE  
**Date**: 2026-03-10  
**Duration**: ~15min

### Summary

Docker 컨테이너(pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime) 환경에서 NVIDIA A40 GPU 검증, 
프로젝트 디렉터리 구조 확인, 필수 라이브러리 설치/검증, 누락 디렉터리 생성 완료.

---

### Subtask 결과

#### 1.1 NVIDIA A40 GPU 및 Docker 환경 검증 ✅

| 항목 | 결과 |
|---|---|
| GPU | NVIDIA A40 |
| VRAM | 46,068 MiB (47.6 GB) |
| Driver | 550.54.15 |
| CUDA Version | 12.4 |
| GPU Utilization | 5% (idle) |
| Memory Used | 0 MiB |
| ECC Errors | 0 |

#### 1.2 Docker 볼륨 마운트 및 디렉터리 구조 ✅

**마운트 지점** (전체 WRITABLE 확인):
- `/workspace` — 코드
- `/data` — 데이터
- `/output` — 산출물
- `/models` — 모델 (Qwen2.5-7B-Instruct 14.2GB 다운로드 완료)
- `/hf-cache` — HuggingFace 캐시
- `/logs` — 로그
- `/qdrant_storage` — Qdrant 벡터 DB (향후 확장용)

**신규 생성 디렉터리**:
```
mkdir -p /data/{parsed,deduplicated,exports}
mkdir -p /output/{indexes,embeddings}
mkdir -p /workspace/trainer/{configs,notebooks}
```

**최종 디렉터리 구조**:
```
/data/          raw/ parsed/ cleaned/ chunked/ deduplicated/ sft/ eval/ exports/
/output/        indexes/ adapters/ checkpoints/ eval_results/ embeddings/
/workspace/     trainer/{scripts,configs,notebooks,Dockerfile} rag_api/ deploy/
```

#### 1.3 PyTorch 및 CUDA 라이브러리 ✅

| 패키지 | 버전 | PRD 기준 | 비고 |
|---|---|---|---|
| torch | 2.4.0 | cu124 | CUDA 12.4 |
| transformers | 5.3.0 | 5.1.0 | 상위호환 |
| accelerate | 1.13.0 | 1.13.0 | 일치 |
| torch.cuda.is_available() | True | - | |
| torch.cuda.get_device_name(0) | NVIDIA A40 | - | |

#### 1.4 양자화 및 최적화 라이브러리 ⚠️ (부분)

| 패키지 | 버전 | PRD 기준 | 상태 |
|---|---|---|---|
| bitsandbytes | 0.49.2 | 0.49.1 | ✅ 패치 업 |
| peft | 0.18.1 | - | ✅ |
| trl | 0.29.0 | - | ✅ |
| unsloth | ❌ 미설치 | 2026.2.1 | ⚠️ 비호환 |

**⚠️ unsloth 비호환 이슈**:
- `unsloth 2026.3.4`는 `transformers<=5.2.0` 요구
- 현재 환경 `transformers==5.3.0`과 충돌
- `unsloth_zoo` 설치 시 `torchao 0.16.0`이 `torch 2.4.0`과 충돌 (`torch.int1` 미지원)
- **해결 방안** (Task 8에서 진행):
  - Option A: transformers 5.2.0 다운그레이드 후 unsloth 사용
  - Option B: unsloth 없이 peft + trl 직접 QLoRA 학습
  - Option C: torch 업그레이드 (Docker 이미지 재빌드 필요)

#### 1.5 RAG/API 라이브러리 및 환경변수 ⚠️ (부분)

| 패키지 | 버전 | 상태 |
|---|---|---|
| sentence-transformers | 5.2.3 | ✅ |
| faiss-cpu | 1.13.2 | ✅ (신규 설치) |
| PyMuPDF | 1.27.1 | ✅ (신규 설치) |
| fastapi | 0.135.1 | ✅ |
| uvicorn | 0.41.0 | ✅ |
| beautifulsoup4 | - | ✅ |
| llama-index | - | ✅ |
| FlagEmbedding | 1.3.5 | ⚠️ import 실패 |
| vllm | ❌ 미설치 | ⚠️ Task 11 stretch |

| 환경변수 | 값 | 상태 |
|---|---|---|
| HF_HOME | /hf-cache | ✅ |
| HF_HUB_CACHE | /hf-cache/hub | ✅ |

**⚠️ FlagEmbedding 비호환 이슈**:
- `FlagEmbedding 1.3.5`의 reranker 모듈이 `transformers.utils.import_utils.is_torch_fx_available` 참조
- `transformers 5.3.0`에서 해당 함수 제거됨
- **우회 방안**: `sentence-transformers`로 BGE-M3 모델 직접 로드 가능
- Reranker는 `bge-reranker-v2-m3`를 cross-encoder로 별도 로드

**⚠️ vllm 미설치**:
- Task 11 (Stretch Goal)에서 별도 설치 예정
- 설치 시간 5분+ 소요, 현재 불필요

---

### 사전 다운로드 모델

| 모델 | 경로 | 크기 | 상태 |
|---|---|---|---|
| Qwen2.5-7B-Instruct | /models/Qwen2.5-7B-Instruct/ | 14.2 GB (4 shards) | ✅ |
| Qwen2.5-7B-Instruct (cache) | /hf-cache/hub/models--Qwen--Qwen2.5-7B-Instruct/ | - | ✅ |
| BAAI/bge-m3 | - | - | ❌ 미확인 (Task 3에서 다운로드) |
| BAAI/bge-reranker-v2-m3 | - | - | ❌ 미확인 (Task 3에서 다운로드) |

---

### 후속 Task 영향도

| 영향 받는 Task | 이슈 | 조치 필요 |
|---|---|---|
| Task 3 (RAG 인덱스) | FlagEmbedding 비호환 | sentence-transformers로 BGE-M3 로드 |
| Task 8 (QLoRA 학습) | unsloth 비호환 | transformers 다운그레이드 or peft+trl 직접 사용 |
| Task 11 (vLLM 서빙) | vllm 미설치 | 해당 Task에서 설치 |

---

### 신규 설치 패키지

```
pip install faiss-cpu    # 1.13.2
pip install PyMuPDF      # 1.27.1
```

### 롤백한 패키지

```
pip uninstall torchao    # 0.16.0 (torch 2.4.0 비호환)
pip install transformers==5.3.0  # unsloth_zoo가 5.2.0으로 다운그레이드한 것 복원
pip install trl==0.29.0          # unsloth_zoo가 0.24.0으로 다운그레이드한 것 복원
```

---

## [Task 2] 문서 수집 및 파이프라인 설계

**Status**: ✅ DONE  
**Date**: 2026-03-10  
**Duration**: ~35min

### Summary

179건 선박 도장 문서(PDF 110, HTML 69)를 수집·분류하고, 4단계 문서 처리 파이프라인(ingest → clean → dedup → chunk) 전체를 구현 및 실행 완료. 최종 147건 문서에서 8,013개 청크 생성.

---

### Pipeline 결과

```
/data/raw/ (179건) → [ingest_docs.py] → /data/parsed/ (179건)
    → [clean_docs.py] → /data/cleaned/ (179건)
    → [dedup_docs.py] → /data/deduplicated/ (147건, 32건 중복 제거)
    → [chunk_docs.py] → /data/chunked/ (8,013 chunks)
```

---

### Subtask 결과

#### 2.1 문서 수집 및 manifest.csv 생성 ✅

| 항목 | 결과 |
|---|---|
| 총 문서 수 | 179건 |
| PDF | 110건 |
| HTML | 69건 |
| 총 용량 | ~685 MB |
| P1 (한국 법령) | 20건 (doc_0001~0020) |
| P2 (NCS·표준) | 30건 (doc_0021~0050) |
| P3 (기타) | 129건 (doc_0051~0179) |

- `/data/manifest.csv`: doc_id, filename, title, format, pages, bytes, priority, domain_category, source_url, file_path, upload_date

#### 2.5 ingest_docs.py 파싱 파이프라인 ✅

- **파일**: `/workspace/trainer/scripts/ingest_docs.py` (365 lines)
- PDF → pymupdf (페이지별 텍스트 + 표 감지 → markdown 변환)
- HTML → BeautifulSoup (script/style 제거, 표 → markdown)
- TXT → plain text
- HTML/TXT 중복 시 TXT 우선 (57건 자동 선택)
- **결과**: 179/179 성공, 0 실패

#### 2.2 clean_docs.py 정규화 파이프라인 ✅

- **파일**: `/workspace/trainer/scripts/clean_docs.py`
- **규칙 파일**: `/workspace/trainer/configs/normalization_rules.yaml` (105 lines)
- 10단계 정규화: NFKC → 제어문자 → 공백 → 줄바꿈 → 헤더/푸터 → 페이지번호 → 단위 → 용어 → 공백정리 → 빈줄축소

| 정규화 단계 | 적용 횟수 |
|---|---|
| Unicode NFKC | 3,618,485 |
| 헤더/푸터 제거 | 11,540 |
| 용어 대체 (DFT, WFT, Sa 등) | 11,690 |
| 단위 통일 (µm, °C, mm) | 2,200 |
| 제어문자 제거 | 692 |
| 페이지번호 제거 | 190 |
| 빈줄 축소 | 1,222 |

- **결과**: 179/179 성공, 0 실패

#### 2.3 dedup_docs.py 중복 제거 파이프라인 ✅

- **파일**: `/workspace/trainer/scripts/dedup_docs.py`
- 3단계 중복 제거:
  - Stage 1 (SHA-256 정확 중복): 32건 제거
  - Stage 2 (MinHash+LSH 유사 문서): 0건 (데이터 다양성 양호)
  - Stage 3 (문단 수준 512자+ 중복): 87개 문단 제거
- **의존성**: `datasketch 1.9.0` 자동 설치

| 항목 | 수치 |
|---|---|
| 입력 문서 | 179 |
| 정확 중복 제거 | 32 |
| 유사 문서 제거 | 0 |
| 문단 수준 중복 | 87 paragraphs |
| 최종 출력 | 147 |
| 중복 제거율 | 17.9% |

- **리포트**: `/data/deduplicated/dedup_report.json`

#### 2.4 metadata_schema.json + chunk_docs.py 청킹 파이프라인 ✅

- **스키마**: `/workspace/trainer/configs/metadata_schema.json` (JSON Schema draft-07)
- **파일**: `/workspace/trainer/scripts/chunk_docs.py`
- 청킹 전략: 시맨틱 경계 → 문단 → 문장 순차 분할
- 타겟 크기: 512-1024자, 50자 오버랩

| 항목 | 수치 |
|---|---|
| 입력 문서 | 147 |
| 총 청크 수 | 8,013 |
| 평균 청크 크기 | 847.91자 |
| 표 포함 청크 | 1,926 (24.0%) |
| 절차 포함 청크 | 2,733 (34.1%) |

**도메인 분포**:
| 도메인 | 청크 수 | 비율 |
|---|---|---|
| coating (도장) | 4,151 | 51.8% |
| general (일반) | 2,090 | 26.1% |
| safety (안전) | 912 | 11.4% |
| blasting (블라스팅) | 436 | 5.4% |
| thickness (도막두께) | 357 | 4.5% |
| defects (결함) | 67 | 0.8% |

- **요약 리포트**: `/data/chunked/chunk_summary.json`

---

### 생성된 파일

| 파일 | 경로 | 설명 |
|---|---|---|
| ingest_docs.py | /workspace/trainer/scripts/ | PDF/HTML/TXT 파싱 |
| clean_docs.py | /workspace/trainer/scripts/ | 10단계 텍스트 정규화 |
| dedup_docs.py | /workspace/trainer/scripts/ | 3단계 중복 제거 |
| chunk_docs.py | /workspace/trainer/scripts/ | 시맨틱 청킹 + 메타데이터 |
| normalization_rules.yaml | /workspace/trainer/configs/ | 정규화 규칙 (단위, 용어, 헤더) |
| metadata_schema.json | /workspace/trainer/configs/ | 청크 메타데이터 JSON Schema |
| manifest.csv | /data/ | 179건 문서 카탈로그 |
| dedup_report.json | /data/deduplicated/ | 중복 제거 통계 |
| chunk_summary.json | /data/chunked/ | 청크 통계 요약 |

### 신규 설치 패키지

```
pip install datasketch    # 1.9.0 (MinHash+LSH 중복 제거)
```

### 알려진 이슈

- `dedup_report.json`이 `/data/deduplicated/`에 같이 저장되어 chunk_docs.py가 문서로 인식 시도 → 1건 에러 로그 (실제 처리에 영향 없음, 147건 정상 청킹)
- `doc_0020`은 content가 비어 있어 0 chunks 생성 (원본 문서 이슈)

### 후속 Task 영향도

| 영향 받는 Task | 데이터 | 비고 |
|---|---|---|
| Task 3 (RAG 인덱스) | /data/chunked/ 8,013 chunks | 바로 임베딩 가능 |
| Task 5 (평가 셋) | /data/chunked/ | 도메인별 QA 생성 소스 |
| Task 7 (QLoRA 데이터) | /data/chunked/ | SFT 데이터 생성 소스 |
