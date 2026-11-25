import pytest

from trainer.demo_api.query_unit_math import supplied_thickness_answer


@pytest.mark.parametrize(
    ("question", "values"),
    [
        ("도막 0.18 mm면 마이크론으로 얼마고, 160~200이라고 써 있는 범위에는 들어? 내 로그는 보지 말고 이 숫자로만.",
         ("180 µm", "160~200 µm", "범위 안")),
        ("정상 범위는 160~200이라고 하셨죠? 그럼 210은 어느 쪽인지랑 두께를 밀리미터로 바꾼 값을 같이요.",
         ("0.21 mm", "160~200 µm", "상한")),
        ("160–200 μm가 목표 범위입니다. 0.16mm를 마이크론으로 바꾸면 해당 범위 안인가요?",
         ("160 µm", "160~200 µm", "범위 안")),
        ("두께가 140µm예요. 기준 범위 0.16~0.2 mm와 비교해 밀리미터로 변환하고 아래인지 알려 주세요.",
         ("0.14 mm", "0.16~0.2 mm", "하한")),
    ],
)
def test_compares_supplied_thickness_without_session(question, values):
    answer = supplied_thickness_answer(question)
    assert answer is not None
    assert all(value in answer for value in values)


@pytest.mark.parametrize("question", ["지금 도막은 정상 범위인가요?", "속도 0.18 mm/s는 160~200마이크론 범위?", "도막 160~200의 범위를 알려줘."])
def test_leaves_measurement_inspection_to_existing_handlers(question):
    assert supplied_thickness_answer(question) is None
