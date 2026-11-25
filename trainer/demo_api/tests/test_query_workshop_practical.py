import pytest

from trainer.demo_api import safety_guard
from trainer.demo_api.query_technical_facts import technical_fact_answer


@pytest.mark.parametrize("question", [
    "에어리스 도료 줄기가 손에 닿아서 바늘 자국 비슷하게 났어요. 피가 별로 안 나는데 물로 헹구고 작업 계속해도 돼요?",
    "고압 분무기에서 새는 페인트에 손가락을 찔렸는데 작은 상처뿐이에요.",
    "에어리스로 뿌리다 페인트가 피부 안으로 들어갔어요. 소독약만 바르면 될까요?",
    "An airless spray gun punctured my finger and injected paint. Can I carry on?",
])
def test_possible_injection_requires_emergency_care(question):
    kind = safety_guard.hazard_class((question,))
    assert kind == "injection_injury"
    answer, citations = safety_guard.fixed_safety_answer(kind)
    assert "응급" in answer and "즉시" in answer and "작업을 중단" in answer
    assert "진료" in answer and citations == []
    assert safety_guard.guard_answer(kind, "물로만 씻고 다시 작업해도 됩니다.", []) == (answer, [])


def test_injection_class_wins_even_if_translated_words_only_describe_contact():
    assert safety_guard.hazard_class((
        "피부에 도료가 묻었는데 일을 해도 될까요?",
        "An airless spray gun punctured my finger and injected paint.",
    )) == "injection_injury"


def test_generic_contact_keeps_the_existing_safety_route():
    assert safety_guard.classify("붓으로 칠하다 손 피부에 페인트가 묻었어요.") == "contact"
    assert safety_guard.classify("고압 분사로 피부에 주입되면 왜 위험한가요?") == "high_pressure"
    assert safety_guard.classify("장갑을 끼고 스프레이건 손잡이를 잡았어요.") is None


def test_disconnected_grounding_is_a_hazard():
    prompt = "분사 설비 접지선을 뺐다고 하는데, 페인트가 잘 나오는지만 시험하고 접지를 나중에 확인해도 돼요?"
    assert safety_guard.classify(prompt) == "grounding"
    answer, _ = safety_guard.guard_answer("grounding", "", [])
    assert "접지" in answer and "정전기" in answer and "하지 마세요" in answer


def test_airless_tip_removal_explains_safe_preconditions_without_inventing_controls():
    kind = safety_guard.classify("에어리스 팁을 분리하기 전에 먼저 확인할 점이 뭐예요?")
    assert kind == "airless_tip_removal"
    answer, _ = safety_guard.fixed_safety_answer(kind)
    assert "잔압 해소" in answer and "방아쇠 잠금" in answer and "장비 설명서" in answer


@pytest.mark.parametrize(
    ("prompt", "required"),
    [
        ("에어리스 도료에 손을 찔려 작은 상처가 났어요. 작업해도 돼요?", ("즉시", "응급실")),
        ("에어리스 팁을 빼기 전에 무엇을 확인해야 해요?", ("잔압 해소", "방아쇠 잠금")),
    ],
)
def test_query_endpoint_sends_fixed_safety_answer_before_generation(prompt, required):
    from trainer.demo_api.tests.test_v342_safety import Pipeline, _ask

    pipeline = Pipeline()
    response, headers = _ask(pipeline, prompt, use_rag="Y")
    assert all(text in response["answer"] for text in required)
    assert response["retrieval_used"] is False
    assert headers["x-copilot-answer-source"] == "analysis"
    assert pipeline.calls == []


@pytest.mark.parametrize(("question", "kind", "required"), [
    ("그라인더에 얇은 절단용 디스크가 있는데, 옆면으로 용접 비드를 갈면 안 돼요?", "cutting_wheel_side", ("파손", "갈지")),
    ("컷오프 휠의 측면으로 연삭도 가능한가요?", "cutting_wheel_side", ("파손", "갈지")),
    ("디스크가 그라인더 축에 맞는데 허용 RPM은 뭘 확인하면 되나요?", "grinding_wheel_speed", ("최대 무부하", "낮으면 안")),
])
def test_wheel_safety_response_does_not_need_generation(question, kind, required):
    assert safety_guard.classify(question) == kind
    answer, citations = safety_guard.fixed_safety_answer(kind)
    assert all(text in answer for text in required) and citations == []


@pytest.mark.parametrize(
    ("question", "required"),
    [
        ("공기 온도랑 습도는 이미 적었는데, 도장 직전에 강판 온도를 또 재는 이유가 뭐예요?", ("이슬점", "결로", "강판")),
        ("도장하기 전에 기온·습도랑 철판 표면 온도를 모두 확인하는 이유가 있나요?", ("이슬점", "결로")),
        ("하도랑 상도를 전부 마친 뒤 일반 도막 게이지로 한 번만 찍었어요. 그 값에서 상도만 따로 알 수 있어요?", ("전체", "같은", "전후")),
        ("프라이머랑 마감칠을 끝내고 두께계를 찍었어요. 여기서 상도만 따로 알 수 있나요?", ("전체", "전후")),
        ("매끈한 철판에서 두께계를 영점 맞췄는데, 블라스팅한 판에서도 그 설정으로 바로 검사해도 될까요?", ("요철", "규격", "보정")),
        ("두께 게이지는 전부 합격이에요. 검사표의 접착력 항목에도 합격이라고 적어도 돼요?", ("접착력", "별도 시험")),
        ("손대도 안 묻는 상태가 됐는데, 이제 바로 덧칠해도 된다는 뜻인가요?", ("지촉건조", "최소·최대")),
        ("2액형 도료의 가사시간이 넘었지만 아직 묽어요. 희석하면 계속 써도 되나요?", ("사용하지", "되돌리지")),
        ("배합표에 부피비라고 쓰여 있는데 전자저울로 똑같이 달아도 될까요?", ("질량비", "밀도")),
        ("상도를 테이프에서 떼어 보니까 위에 칠한 층만 들리고 아래 칠은 철판에 그대로 있네요. 철판 접착이 나빴다고 보면 돼요?", ("층간 박리", "확정할 수 없")),
        ("시험 분사 자국 양 끝에 진한 줄 두 개가 생겨요. 압력을 끝까지 올리면 되나요?", ("테일링", "점도", "팁")),
        ("공기분무 건과 에어리스 건이 각각 어떻게 무화하나요?", ("압축공기", "가압된 도료", "팁 구멍")),
        ("강판 염분이 눈에 안 보이면 검사를 생략할 수 있나요?", ("수용성 염분", "전도도")),
        ("강판에 기름 자국이 있는데 블라스팅하면 다 날아갈 거라고 먼저 쏘자네요. 그 순서로 작업해도 괜찮아요?", ("먼저 제거", "오염")),
        ("전날 블라스트를 끝낸 판이 다시 녹슬었는데 곧바로 도장해도 될까요?", ("표면 상태", "시방")),
        ("회수한 블라스팅 연마재를 다시 장비에 넣어도 되나요?", ("재사용 가능한 종류", "분리", "입도")),
        ("블라스팅에 쓰는 압축공기는 압력계로 청정도 검사하나요?", ("오일", "물", "따로")),
        ("절단부 날카로운 모서리를 블라스팅만 해서 둥글게 만들 수 있나요?", ("라운딩", "연삭", "따로")),
        ("도장 전에 용접 비드를 다듬으라는데, 그라인더로 주변 철판 높이까지 무조건 싹 깎아야 해요?", ("용접·도장 시방", "단면")),
        ("탄소강을 갈던 디스크로 옆에 있는 스테인리스 판을 바로 다듬으면 뭐가 문제예요?", ("철 입자", "녹", "지정")),
        ("녹을 막는 하도랑 마감 상도는 보통 어떤 일을 하나요?", ("바탕면", "외부 환경", "외관")),
    ],
)
def test_grounded_surface_and_layer_measurement_distinctions(question, required):
    answer = technical_fact_answer(question)
    assert answer and all(text in answer for text in required)


@pytest.mark.parametrize("question", [
    "실내 습도 표시가 어디에 있어요?",
    "VR 내부 습도가 바뀌면 실제 강판 온도도 바뀌나요?",
    "상도랑 중도 사이에는 얼마나 기다리나요?",
    "NCS에서 두께 게이지 교정 순서를 찾아 줘.",
])
def test_nonmatching_requests_keep_existing_routes(question):
    assert technical_fact_answer(question) is None
