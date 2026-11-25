import pytest

from trainer.demo_api import query_technical_facts
from trainer.demo_api.query_technical_facts import technical_fact_answer


@pytest.mark.parametrize(
    ("question", "required"),
    [
        ("칠한 게 자꾸 눈물처럼 아래로 흘러. 이건 도막이 부족한 거야?", ("처짐", "측정")),
        ("도료가 벽 아래로 흘러내리는데 저도막인가요?", ("처짐", "측정")),
        ("표면이 가루 묻은 것처럼 거칠거칠해. 건조 분무가 그 말인가?", ("닿기 전", "외관만")),
        ("칠한 면이 모래처럼 거칠어요. 드라이 스프레이일까요?", ("닿기 전", "외관만")),
        ("방금 칠하고 잰 160마이크로랑 마른 뒤 160마이크로랑 같은 판정이야?", ("습도막", "건도막", "휘발", "체적 고형분")),
        ("습도막 두께랑 건도막 두께 차이를, 언제 재는지도 같이 알려 줘.", ("도장 직후", "건조된 뒤")),
        ("도막 얇다고 무조건 희석제 많이 넣어서 그런 거 맞지?", ("이동 속도", "이격거리", "중첩")),
        ("굵게 왕창 칠한 데가 있으면 옆에 안 칠한 자리 점수까지 메워주는 건가? VR 얘기.", ("화소", "100%", "0%")),
        ("시뮬레이터에서 과도막이면 미도장 부분도 진행률에 보상돼?", ("화소", "100%", "0%")),
        ("분사 건을 한곳에서 자꾸 멈추면 거긴 보통 어떻게 돼?", ("분사가 계속", "과도막", "처짐")),
        ("스프레이 건이 같은 자리에 머물면 어떤 문제가 생겨?", ("분사가 계속", "과도막")),
        ("아니, 스프레이 말고 그라인더 디스크를 한곳에 멈추고 누르는 얘기야.", ("회전", "국소", "열")),
        ("그라인더를 한 자리에 계속 대면 어떻게 돼?", ("회전", "국소", "열")),
        ("블라스트 얘긴데 '앵커 패턴'이 작업 순서 외우라는 뜻이야?", ("표면 조도", "접착", "작업 순서")),
        ("블라스팅에서 쇼트하고 그릿, 이름만 다른 똑같은 알갱이야?", ("둥근", "각진", "연마재")),
        ("녹 걷어내는 블라스팅이랑 용접 비드 다듬는 그라인딩은 뭐가 다른 거야?", ("연마재를 분사", "회전 디스크", "접촉")),
        ("저도마악이라는 게 두껍게 발랐다는 말이야? 현장 용어 맞아?", ("저도막", "못 미친", "과도막")),
        ("저도막은 어떤 상태인가요?", ("건도막", "못 미친")),
        ("동료가 제트 코치 컷 77 기준이라는데 우리 조선소 표준 수치가 정확히 얼마야?", ("표준서", "기준명")),
        ("우리 현장 작업표준의 허용 도막 두께 수치는 몇 µm인가요?", ("표준서", "기준명")),
        ("블라스팅 할 때 도막이 너무 얇다고 자꾸 신경 쓰이는데, 지금도 160까지 도료를 더 올려야 하나?", ("표면 준비", "160µm", "적용하지")),
        ("그라인딩 작업 기록도 도료 분사 시퀀스처럼 과도막 경고가 뜨는 건가요?", ("스프레이", "그라인딩", "생성하지")),
    ],
)
def test_grounded_technical_distinctions(question, required):
    answer = technical_fact_answer(question)
    assert answer and all(word in answer for word in required), answer


def test_simulator_example_uses_app_band_and_distinguishes_explicit_session_limits():
    assert "정상" in technical_fact_answer("VR에서 목표가 160이라는데 190이면 이제 무조건 빨간 과도막이야?")
    assert "160~200µm" in technical_fact_answer("VR에서 목표가 160이라는데 190이면 이제 무조건 빨간 과도막이야?")
    answer = technical_fact_answer(
        "시뮬레이터에서 200µm는 과도막인가요?",
        target_thickness_um=170,
        target_min_thickness_um=180,
        target_max_thickness_um=195,
    )
    assert "180~195µm" in answer and "200µm는 과도막" in answer
    assert "200µm는 정상" in technical_fact_answer("시뮬레이터에서 200µm는 과도막인가요?")


@pytest.mark.parametrize(
    "question",
    [
        "3구역 도막이 부족한 거야?",
        "분사 건이 멈췄는데 펌프가 고장 난 거야?",
        "드라이 스프레이를 방지하려면 분사 거리를 몇 cm로 해야 할까?",
        "사내 표준서에서 저도막 처리 절차를 찾아 줘.",
        "우리 VR의 도막 두께 기준이 얼마야?",
        "블라스팅 전에 도장이 마르는 데 걸리는 시간은?",
        "저도막이 생기는 원인은 무엇인가요?",
        "저도막이 뭐야?",
        "오늘 미술시간에 물감을 쏟았어.",
        "도장 품질은 어때?",
    ],
)
def test_other_subjects_keep_the_existing_query_route(question):
    assert technical_fact_answer(question) is None


def test_non_summarized_measurement_definition_keeps_the_document_route():
    question = "습도막 두께랑 건도막 두께 차이를, 언제 재는지도 같이 알려 줘."
    assert technical_fact_answer(question, summarize=False) is None


def test_missing_term_evidence_leaves_the_existing_route_available(monkeypatch):
    monkeypatch.setattr(query_technical_facts, "matched_terms", lambda _: [])
    assert technical_fact_answer("저도막은 어떤 상태인가요?") is None


def test_correction_switches_to_the_named_tool():
    answer = technical_fact_answer("그라인더 말고 스프레이 건을 한 곳에 계속 대면?")
    assert answer and "과도막" in answer and "과삭" not in answer


@pytest.mark.parametrize(
    ("question", "required"),
    [
        ("접착력 검사표의 격자 절개 등급을 풀오프 MPa 값으로 바꿔 적어도 되나요?", ("격자", "인장 응력", "대체")),
        ("크로스컷과 pull-off의 결과를 검사표에 같은 항목으로 적을 수 있나요?", ("박리", "인장 응력", "대체")),
        ("시방에서 요구한 조도 Rz 대신 조도계의 Ra를 써도 되나요?", ("평균", "높낮이", "대체")),
        ("블라스팅 표면 검사 때 Ra랑 Rz가 둘 다 마이크로미터면 환산해도 돼요?", ("평균", "높낮이", "대체")),
        ("먼지 테이프에서 먼지 양하고 입자 크기를 하나만 기록하면 안 돼요?", ("잔류 먼지", "입자", "시방")),
        ("도막 두께 시방을 통과했으니 홀리데이 검사는 생략해도 돼요?", ("표본", "핀홀", "대체")),
        ("강판 잔존 두께 mm 값을 도막 두께 µm로 환산해 항목을 대신 적으래요.", ("남은 강재", "도료층", "대체")),
        ("도장표에서 색상 번호가 같다고 유광/반광을 같은 항목으로 검사해도 되나요?", ("반사", "광택 요구사항", "별도")),
        ("두께계 값이 모서리 가까이 가면 튀는데 저도막이라고 써요?", ("프로브", "확정할 수 없으니", "시방")),
    ],
)
def test_inspection_methods_keep_their_distinct_result_types(question, required):
    answer = technical_fact_answer(question)
    assert answer and all(word in answer for word in required)


@pytest.mark.parametrize("question", [
    "풀오프 시험의 장비 설치 방법을 문서에서 찾아 주세요.",
    "Ra와 Rz라고 이름 붙인 두 작업자는 같은 팀인가요?",
    "NCS 문서에서 격자 절개와 풀오프가 어떤 차이인지 근거를 찾아 주세요.",
    "이번 2번 구역의 도막 두께 합격 기준을 확인해 주세요.",
    "홀리데이 검사표를 보내줄 테니 실제 결과를 보고 판정해 주세요.",
])
def test_inspection_comparisons_do_not_replace_document_or_off_topic_queries(question):
    assert technical_fact_answer(question) is None
