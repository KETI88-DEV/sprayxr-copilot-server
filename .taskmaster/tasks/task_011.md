# Task ID: 11

**Title:** vLLM API 서빙 패키징(선택 Stretch Goal)

**Status:** pending

**Dependencies:** 3 ✓, 9 ✓

**Priority:** low

**Description:** Docker 컨테이너 환경에서 vLLM 0.16.0을 이용해 OpenAI-compatible REST API 패키징, 모델 선택지(base/RAG-only/RAG+QLoRA) 제공

**Details:**

1. vLLM serve 명령어: vllm serve Qwen/Qwen2.5-7B-Instruct --dtype auto --api-key shipcoat-demo-key --port 8000 --host 0.0.0.0
   - 모델은 /hf-cache/에서 자동 로드
2. 선택적 adapter 로드: vLLM의 lora_modules 파라미터 활용(지원 확인 필요)
3. OpenAI-compatible endpoints: /v1/chat/completions, /v1/models, /health
4. FastAPI wrapper 추가(선택): /workspace/rag_api/에서 custom `/qa`, `/coach/session` endpoint 구현
5. 요청 스키마: {model, messages, temperature, max_tokens} 호환
6. 응답: OpenAI format {choices[{message}], usage} 유지
7. API key 검증: Authorization header 확인
8. 스크립트: /workspace/trainer/scripts/serve_vllm.py(서버 시작), /workspace/trainer/scripts/demo_api_client.py(테스트)
9. 설정: port 8000, host 0.0.0.0, dtype auto
10. 문서: /workspace/README.md에서 API 사용 예시 curl/python 코드 포함

**Test Strategy:**

vLLM 서버 시작 성공(listening on 0.0.0.0:8000), curl 또는 python-openai로 chat completion 요청 성공, 응답 시간 <15초, API key 검증 동작

## Subtasks

### 11.1. vLLM 서버 기본 구성 및 모델 로딩 검증

**Status:** pending  
**Dependencies:** None  

vLLM 0.16.0을 이용해 Qwen2.5-7B-Instruct 모델을 포트 8000에서 기본 구성으로 서빙 시작, OpenAI-compatible API 엔드포인트 정상 작동 확인

**Details:**

vLLM serve 명령어 실행(vllm serve Qwen/Qwen2.5-7B-Instruct --dtype auto --api-key shipcoat-demo-key --port 8000 --host 0.0.0.0), /hf-cache/에서 모델 자동 로드, health check 엔드포인트(/health) 응답 확인, /v1/models 엔드포인트에서 모델 정보 반환 검증, 서버 메모리 사용량 모니터링(<40GB)

### 11.2. OpenAI-compatible /v1/chat/completions 엔드포인트 테스트

**Status:** pending  
**Dependencies:** 11.1  

vLLM의 OpenAI-compatible /v1/chat/completions 엔드포인트가 표준 요청 스키마(model, messages, temperature, max_tokens)를 수용하고 올바른 응답 형식 반환 확인

**Details:**

테스트 스크립트(/workspace/trainer/scripts/demo_api_client.py)에서 python-openai 클라이언트 또는 직접 HTTP POST로 요청 송신, 요청 바디: {"model": "Qwen2.5-7B-Instruct", "messages": [{"role": "user", "content": "테스트"}], "temperature": 0.7, "max_tokens": 256}, 응답 검증: choices[0].message.content 존재, usage 필드(prompt_tokens, completion_tokens, total_tokens) 포함, 응답 시간 <15초 확인

### 11.3. API 키 검증 및 인증 미들웨어 구현

**Status:** pending  
**Dependencies:** 11.1  

Authorization header에서 API 키 검증 로직 구현, 올바른 키(shipcoat-demo-key)는 요청 허용, 잘못된 키는 401 응답 반환 확인

**Details:**

vLLM serve 시작 시 --api-key 파라미터로 shipcoat-demo-key 설정, 요청 헤더 Authorization: Bearer shipcoat-demo-key 형식으로 전송, 잘못된 키 또는 누락된 헤더에 대해 401 Unauthorized 응답 검증, 다양한 invalid key 시나리오 테스트(누락, 틀린값, 만료된값 등)

### 11.4. 선택적 adapter 로드 기능 구현 및 검증(vLLM lora_modules 파라미터)

**Status:** pending  
**Dependencies:** 11.1  

vLLM의 lora_modules 파라미터를 활용해 /output/adapters/에서 QLoRA adapter 동적 로드 기능 구현, base 모델과 adapter 로드된 모델 두 가지 구성 지원 확인

**Details:**

vLLM 공식 문서에서 lora_modules 파라미터 지원 확인, /output/adapters/qwen25_shipcoat_lora/ 경로의 adapter 감지, vllm serve 시작 시 --lora-modules parameter 추가(예: --lora-modules qwen-shipcoat=/output/adapters/qwen25_shipcoat_lora/), /v1/models 엔드포인트에서 adapter 정보 포함 여부 확인, model 선택 시 base 또는 adapter 적용 모델 선택 가능 확인

### 11.5. FastAPI wrapper 및 커스텀 /qa, /coach/session 엔드포인트 구현(선택)

**Status:** pending  
**Dependencies:** 11.2  

/workspace/rag_api/에서 FastAPI wrapper 구현, vLLM의 OpenAI-compatible API를 기반으로 커스텀 /qa, /coach/session 엔드포인트 제공, RAG 모듈과 통합

**Details:**

FastAPI 애플리케이션 구축(serve_vllm.py 또는 별도 rag_api_server.py), vLLM 서버를 reverse proxy로 활용, /qa 엔드포인트: 질문 입력 → RAG 검색 → vLLM LLM 호출 → 응답 반환, /coach/session 엔드포인트: 세션 관리, 기존 RAG 모듈과 호환성 유지, 요청/응답 스키마: OpenAI format 유지, API 키 검증 미들웨어 적용
