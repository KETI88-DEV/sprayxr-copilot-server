"""모델이 쓰는 도장·기록 답에 싣는 시뮬레이터 기본 설정(v3.5.1). 모델이 현장 조언을 지어내지 않게 이 훈련이 무엇인지 알려 준다.

지침이 아니라 사실이다. 유출 검사 지문에는 넣지 않고(dialogue._instruction_texts는 이 블록을 싣기 전 지침만 본다),
블록의 수치는 모델에게 알려 주기만 하고 근거로 쓰지 않는다(수치 근거 검사는 v3.5.0 그대로).
"""
from __future__ import annotations

import re
from decimal import Decimal
from typing import Final

from trainer.demo_api.coating_assessment import GATE_COMPLETION_PERCENT
from trainer.demo_api.copilot_intent import is_domain_question
from trainer.demo_api.query_runtime import explicit_work_process
from trainer.demo_api.record_query import has_record_marker

SIMULATOR_CONTEXT: Final = f"""[시뮬레이터 기본 설정]
- 이 훈련은 실물 스프레이건 모형을 들고 VR 속 가상 피도면에 분사하는 시뮬레이터입니다. 실제 도료는 나가지 않습니다.
- 이 시뮬레이터의 훈련모드(학습모드)는 1~10번, 총 10개입니다.
- 훈련생은 건, 노즐 코드(단건 519·523·527·531·535, 폴건 323·325), 압력(108~504 bar, 36 bar 단위), 신너 희석률(0~24%, 1.6% 단위), 월(그 달 기온 적용)을 고릅니다.
- 도막 두께는 점도·노즐·압력·이격거리·건 속도·분사 시간 등으로 예측 모델이 계산합니다. 분사 각도는 분사 형상에 영향을 줍니다. 습도는 설정 항목에 없습니다.
- 기본 정상 범위는 160~200µm이고 세션마다 다르게 정할 수 있습니다. Zone 진행률이 {GATE_COMPLETION_PERCENT:g}% 이상이면 완료입니다. 진행률은 공식 Zone 전체에서 화소별 목표 두께 도달 비율을 최대 100%로 제한해 평균한 값입니다. 이미 두껍게 칠한 곳은 추가 분사를 멈춥니다. 평균 도막 등급은 품질 안내에 사용합니다. 90/10 규칙 같은 측정점 규격으로 판정하지 않습니다.
- 이격거리·속도의 기준은 Zone별 전문가 측정 중앙값입니다. 전문가 속도는 이전 장비 기록이라 속도를 맞추라고 하지 않습니다. 거리나 속도를 몇 % 바꾸라는 고정 규칙은 없습니다.
- 단위: 위치 m, 이격거리 mm, 속도 mm/s, 각도 °, 도막 µm, 압력 bar, 온도 ℃."""

_TRAINING_MODE: Final = re.compile(r"(?:훈련|학습)\s*모드")
_MODE_CATALOG: Final = re.compile(
    r"(?:훈련|학습)\s*모드(?:는|가|의)?\s*(?:총\s*)?"
    r"(?:몇\s*(?:개|가지|종류)|개수|갯수|종류(?!별)|범위|목록|전체\s*(?:목록|종류)|번호\s*(?:범위|목록))")


_RECORD_DEFINITIONS: Final = (
    (
        (r"raytexuv", r"PNG|이미지|배열|행", r"(?<![A-Za-z])Y(?![A-Za-z])|원점|방향"),
        "raytexuv는 좌하단 원점의 픽셀 좌표이며 Y는 위로 증가합니다. 위에서 시작하는 PNG 배열은 첫 행이 가장 큰 Y에 해당하므로, 배열 행 번호는 아래로 갈수록 커집니다.",
    ),
    (
        (r"크롭|crop", r"PNG|img_width|img_height", r"폭|면적|배치|해상도|width"),
        "width·height는 원본 텍스처에서 차지하는 크롭 영역이고, img_width·img_height는 전송된 PNG의 해상도입니다. 서버는 원본 영역에 맞춰 배치하고 도막을 판정합니다.",
    ),
    (
        (r"리포트|보고서|replay", r"pending", r"접수|자료|결과|평가|UI"),
        "완료 요청 응답의 report_id로 접수 여부를 확인하세요. report_status가 pending이면 분석 중입니다. 상태를 조회해 complete가 된 뒤 실제 결과를 표시하세요.",
    ),
    (
        (r"analysis_status", r"unavailable|null", r"유사도|점수|0\s*%|0으로"),
        "unavailable과 null은 평가할 근거가 없다는 의미입니다. 0%는 평가해서 얻은 점수이므로, 화면에는 ‘분석 불가’를 표시하세요.",
    ),
    (
        (r"use_rag", r"retrieval_used", r"false|오류|실패|미사용"),
        "use_rag=Y는 문서 검색 허용입니다. retrieval_used=false는 실제 답변에 문서 근거가 쓰이지 않았다는 뜻입니다. 이 값만으로 요청 오류를 판단하지 않습니다.",
    ),
    (
        (r"response_mode", r"summarize", r"분량|길이|차이|둘|독립"),
        "response_mode는 빠른 안내와 상세 분석 중 답변 방식·모델을 고릅니다. summarize는 답변 분량을 별개로 고르는 값입니다. detailed와 false를 함께 보내면 상세 분석 방식으로 전체 답변을 요청합니다.",
    ),
    (
        (r"도막|두께", r"전문가\s*(?:비교|기준)", r"미등록|없|확인\s*불가", r"세션\s*목표|목표\s*범위"),
        "공식 구역과 유효한 높이맵이 있으면 세션 목표 범위로 도막을 판정할 수 있습니다. 전문가 기준이 없으면 전문가 비교만 확인할 수 없습니다.",
    ),
    (
        (r"방아쇠|트리거|분사\s*버튼", r"건만|도구만|이동|궤적", r"기록|프레임|시퀀스"),
        "도구를 이동시키기만 해도 이동 프레임은 기록될 수 있습니다. 도막과 도포 구역에는 방아쇠를 눌러 피도면에 유효하게 착탄한 분사가 반영됩니다.",
    ),
    (
        (r"피도면\s*(?:밖|외부)|착탄(?:이|을)?\s*없|(?:표면|레이)\s*(?:에)?\s*(?:안\s*맞|미충돌)",
         r"구역|Zone|어느\s*면", r"분사|방아쇠|도장|도포"),
        "피도면에 유효한 착탄이 없으면 도포 위치나 맵 번호를 확인할 수 없습니다. 구역은 미판정으로 남고 도막에는 반영되지 않습니다.",
    ),
    (
        (r"(?:최종|마지막).{0,25}(?:높이맵|히트맵)", r"몇\s*초|시점|도포\s*시간|순서", r"부터|알\s*수|찾아|추적"),
        "최종 높이맵에는 종료 시점의 누적 두께만 있습니다. 구역별 도포 시간이나 시작 시점은 시간 정보가 있는 프레임·시퀀스 또는 중간 높이맵이 필요합니다.",
    ),
    (
        (r"HMD|카메라|헤드셋", r"위치.{0,20}(?:올라|이동|위로|움직)", r"시선|눈|(?:위를|위에)\s*본"),
        "카메라 위치로는 머리가 향한 방향을 알 수 없습니다. 머리 방향에는 HMD의 회전 또는 전방 벡터가 필요하며, 눈의 실제 시선은 별도 시선 추적 데이터가 필요합니다.",
    ),
    (
        (r"앞면", r"뒷면", r"구역|Zone", r"합산|합치|같은\s*번호|묶어"),
        "구역 번호가 같아도 앞면과 뒷면의 맵 번호가 다르면 별개의 구역입니다. 맵 번호와 Zone 번호를 함께 사용해 식별하고 각 면의 도막을 따로 표시합니다.",
    ),
    (
        (r"ray\s*point|착탄\s*좌표", r"픽셀|텍스처|\bUV\b", r"같|차이|좌표"),
        "Ray Point의 XYZ는 3차원 피도면의 착탄 위치입니다. 높이맵의 픽셀 XY는 각 맵의 2차원 텍스처 좌표이며, 두 값을 연결하려면 맵 번호와 UV 대응이 필요합니다.",
    ),
    (
        (r"높이맵|히트맵", r"RGB|픽셀|PNG", r"빨갛|빨간|빨강|red", r"과도막|경고|판정"),
        "원본 높이맵은 R·G 채널에 두께를 저장합니다. 원본 픽셀의 빨간색과 화면의 과도막 경고 색은 별개입니다. 해독한 두께를 적용 기준과 비교해 판정합니다.",
    ),
    (
        (r"칠한\s*(?:곳|부분|면적)만|도포(?:한|된)\s*(?:곳|부분|면적)만",
         r"전체.{0,15}(?:평균|구역)|(?:구역|피도면)\s*평균", r"평균", r"다[르른]|오류|차이"),
        "칠한 곳의 평균은 도료가 있는 부분만 포함합니다. 전체 구역 평균은 미도장을 포함한 구역 전체를 분모로 삼으므로, 두 평균은 다를 수 있습니다.",
    ),
    (
        (r"히트맵|높이맵", r"크롭|일부만|잘라낸", r"전체\s*(?:면|피도면|구역)", r"판단|평가|적정"),
        "히트맵의 일부 영역으로는 전체 피도면의 도막을 판단할 수 없습니다. 전체 면 높이맵이나 각 공식 구역을 덮는 누적 높이 자료가 필요합니다.",
    ),
    (
        (r"손잡이|컨트롤러", r"분사축|(?:도료|분사).{0,15}(?:판|표면)", r"각도", r"같|차이"),
        "손이나 손잡이의 각도는 작업자의 자세와 도구 방향을 설명합니다. 도료가 닿는 각도는 분사축과 피도면의 방향으로 구하는 별개의 수치입니다.",
    ),
    (
        (r"녹음|음성\s*기록", r"분사.{0,14}(?:횟수|패스)", r"만으로|확정|확인"),
        "녹음 여부만으로는 분사 패스의 횟수를 확인할 수 없습니다. 분사 시작·종료 이벤트나 시퀀스 기록을 확인해야 합니다.",
    ),
    (
        (r"음성\s*질문|음성\s*인식", r"도장\s*기록|훈련\s*기록", r"복구|복원|되살"),
        "Query는 음성 인식으로 나온 문장을 사용합니다. 도장 기록에는 음성 원문이 없으므로 복구할 수 없습니다. 인식 문장을 수정해 다시 보내거나 다시 발화해 주세요.",
    ),
    (
        (r"(?<![A-Za-z])RAG(?![A-Za-z])|retrieval_used", r"표시|배지|붙으면|true", r"훈련|세션|기록|의미|뜻"),
        "RAG 표시는 관련 문서를 실제 답변의 근거로 사용했다는 뜻입니다. 개인 훈련을 분석했는지는 별개이며, 해당 세션의 시퀀스나 리포트에 근거했는지 확인해야 합니다.",
    ),
)

_MODE_IN_QUESTION = re.compile(r"(?:학습|훈련)\s*모드\s*(\d+)")
_TWO_MODES = re.compile(r"(?:학습|훈련)\s*모드\s*(\d+)\s*(?:와|과|및|,)\s*(\d+)")
_CUMULATIVE_READING = re.compile(
    r"(?P<when>이전|직전|바로\s*전|현재|이번|최신)\s+"
    r"누적.{0,26}?(?<![\d.+-])(?P<value>\d{1,6}(?:\.\d{1,4})?)"
    r"(?![\d.])\s*(?=[µμu]m|마이크론|마이크로미터|이(?:고|며|면|라|다))",
    re.IGNORECASE,
)


def _cumulative_record_answer(prompt: str) -> str | None:
    """Compare explicitly supplied readings at one location, outside session lookup."""
    if not re.search(r"같은\s*(?:위치|픽셀|지점|좌표)", prompt):
        return None
    if not re.search(r"합산|합해서|더하|증분|차이", prompt):
        return None
    if not re.search(r"[µμu]m|마이크론|마이크로미터", prompt, re.I):
        return None
    readings = list(_CUMULATIVE_READING.finditer(prompt))
    if len(readings) != 2:
        return None
    timelines = {"현재", "이번", "최신"}
    first, second = readings
    first_is_latest = first["when"] in timelines
    if first_is_latest == (second["when"] in timelines):
        return None
    previous_reading, latest_reading = (second, first) if first_is_latest else (first, second)
    prior = Decimal(previous_reading["value"])
    latest = Decimal(latest_reading["value"])
    return (
        f"같은 위치의 누적 높이맵은 최신 값인 {latest}µm로 교체합니다. "
        f"두 관측값의 차이는 {latest - prior}µm입니다. 누적값을 서로 더하면 중복 집계됩니다."
    )


def simulator_record_answer(prompt: str) -> str | None:
    """Explain documented record formats without drawing conclusions about a session."""
    modes = set(_MODE_IN_QUESTION.findall(prompt))
    modes.update(mode for pair in _TWO_MODES.findall(prompt) for mode in pair)
    if (len(modes) > 1
            and re.search(r"구역|zone", prompt, re.I) and re.search(r"같|일괄", prompt)
            and re.search(r"비교|같은\s*부위|합쳐", prompt)):
        return "같은 번호라도 시나리오가 다르면 별개의 구역입니다. 비교하려면 시나리오·모델·면의 대응을 확인하세요."
    if cumulative := _cumulative_record_answer(prompt):
        return cumulative
    for patterns, answer in _RECORD_DEFINITIONS:
        if all(re.search(pattern, prompt, re.IGNORECASE) for pattern in patterns):
            return answer
    return None


def simulator_configuration_facts(prompt: str) -> dict[str, object] | None:
    """Server configuration, separate from NCS evidence or session measures.

    A numbered mode's coating technique is not a request for the mode catalog.
    Only the confirmed catalog is supplied; names and per-mode properties are
    not inferred from the mode numbers.
    """
    if _MODE_CATALOG.search(prompt) is None:
        return None
    return {"simulator_configuration": {"training_mode_ids": list(range(1, 11))}}


def spray_simulator_context(prompt: str) -> str:
    """스프레이건·도막 목표 설정은 스프레이 공정에만 적용한다."""
    return "" if explicit_work_process(prompt) in {"blasting", "grinding"} else SIMULATOR_CONTEXT


def wants_simulator_context(prompt: str) -> bool:
    """대화 경로에서는 도장 낱말이나 기록 표지가 있는 스프레이 질문에 설정을 전달한다."""
    return bool(spray_simulator_context(prompt)) and (
        is_domain_question(prompt) or has_record_marker(prompt) or bool(_TRAINING_MODE.search(prompt))
    )
