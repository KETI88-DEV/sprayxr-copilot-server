# 로컬 다국어 1차 평가 배포 rc.1

한국어·영어·중국어(간체)의 음성 전사와 지식 질의 번역을 제공한다. 베트남어는 실제 번역 실패, 우즈벡어·미얀마어·네팔어는 용어/품질 근거 미완료로 비활성이다. 모든 활성 언어는 동일 Whisper large-v3-turbo와 동일 Qwen 27B 서버를 사용한다. 국가별 모델은 없다. 전체 단일 모델 Gemma 전환 검토는 [PRD §0](../../../docs/local_voice_multilingual_copilot_prd.md)에 있다.

## 위치와 사용

Spark의 설치 디렉터리는 `/home/keti/shipcoat/releases/multilingual-rc1`, 컨테이너는 `shipcoat-multilingual-gateway-1`, 포트는 **Spark 루프백 58010**이다. 기존 8080 UI에는 아직 통합하지 않았으며 별도 Swagger에서 평가한다.

사용자 PC에서 다음 명령을 한 줄로 실행한 뒤 `http://localhost:58010/docs`에 접속한다.

```bash
ssh -N -L 58010:127.0.0.1:58010 keti@spark-8bb9
```

- `GET /languages`: 활성·보류 언어, 모델, 용어집 버전.
- `GET /ready`: Whisper 로드, 지정된 LLM 모델, 기존 Co-Pilot readiness.
- `POST /transcriptions`: multipart `audio` 파일과 `language`. WAV PCM16·16kHz·mono·15초 이하, 최대 1 MiB. 인식문을 반환하며 질의를 자동 실행하지 않는다.
- `POST /query`: 아래 JSON. `query`는 기존 v2.5 질의 형식, 현재 `knowledge_search`만 허용한다. 인식문을 확인/수정한 뒤 `prompt`에 넣는다.

```json
{"query":{"user_id":"USER-001","prompt":"저도막이 뭐야?","task_type":"knowledge_search","use_rag":"Y"},"input_language":"ko","output_language":"en"}
```

언어 코드는 `ko/en/zh`다. 이 평가 API는 별도 포트에 있으며 기존 `/api/v1/copilot/query` 계약에 언어 필드를 추가하지 않는다. `data.copilot`은 한국어 정본과 기존 분석·근거, `data.localization.answer`는 번역문이다. 실패 시 번역문은 null이고 `fallback_original`을 반환한다. 원문을 외국어 번역처럼 표시하지 않는다. `data.timings`에서 입력 번역·RAG·출력 번역을 분리해 볼 수 있다.

## 빌드·시작·롤백

**Spark의 설치 디렉터리에서** 실행한다. x86 개발 호스트에서 이미지를 빌드하지 않는다. 모델 경로는 기존 검증된 로컬 Whisper 디렉터리다. 실행 중 모델 다운로드는 비활성이다.

```bash
MULTILINGUAL_STT_DIR=/home/keti/shipcoat/models/whisper-large-v3-turbo docker compose --env-file /dev/null -f deploy/spark/multilingual/compose.yaml up -d --build
```

기반 이미지는 현재 Spark의 `shipcoat/agent-api:swagger-ux`다. 첫 빌드에서 확인한 이미지 ID는 `sha256:96ed8e8a58cce877207ab257543eb03ebd4e1af4d0daa4594ba72e33833a5243`이다. 이 이미지는 공개 레지스트리에 배포됐다고 가정하지 않는다. 새 장비에서는 기반 이미지도 함께 export/import하고 image ID·의존성 목록을 기록해야 재현할 수 있다. 태그만으로 재현됐다고 판단하지 않는다. 본 패키지는 기존 이미지의 torch/transformers/fastapi/httpx/python-multipart를 사용하며 의존성을 설치·업그레이드하지 않는다.

기존 LLM에 인증키가 설정됐다면 비밀 관리 경로를 통해 `VLLM_API_KEY`를 프로세스 환경으로 제공한다. 값은 명령 기록이나 문서에 넣지 않는다. 기존 모델 이름이 다르면 `MULTILINGUAL_MODEL`을 실제 `/v1/models` 값으로 지정한다.

중지는 다음 한 줄로 수행한다. 기존 vLLM/MCP/agent-api/web 서비스는 중지하지 않는다.

```bash
MULTILINGUAL_STT_DIR=/home/keti/shipcoat/models/whisper-large-v3-turbo docker compose --env-file /dev/null -f deploy/spark/multilingual/compose.yaml down
```

## 평가 범위와 알려진 한계

이 버전은 지식 질의의 **평가용 API 배포**다. ASR과 번역의 대상 언어별 현장 화자 정확도, 안전 문장 검수, 30분 부하 게이트를 통과한 정식 제품이라는 뜻은 아니다. 현재 서비스는 요청을 하나씩 처리하고 초과 요청은 429로 거부한다. 이 잠금은 평가 서비스 내부이며 기존 운영 API와의 전역 생성 잠금을 공유하지 않는다. 혼합 부하의 간섭을 별도로 측정해야 한다.

언어 스크립트·숫자·단위·ID·비교기호·생성 중단을 검사하지만, 부정 의미와 영어/베트남어 언어 판별을 완전히 검증하는 장치는 아니다. 원어민 평가를 대체하지 않는다. 원음은 메모리에서만 처리하고 저장·로그하지 않는다. 공개 인터넷 배포용 인증 계층은 추가하지 않았으므로 루프백 바인딩과 SSH 접근을 유지한다.

근거 용어와 보류 항목은 [현지 문서 대조표](../../../docs/local_voice_multilingual_terminology.md), 실제 확인 결과는 [검증 기록](../../../docs/local_voice_multilingual_release_evidence.md)을 참조한다.
