import pytest

from trainer.demo_api.simulator_context import simulator_record_answer


@pytest.mark.parametrize(
    ("prompt", "facts"),
    [
        ("건만 움직인 프레임과 방아쇠를 누른 프레임은 기록이 같나요?", ("이동 프레임", "착탄")),
        ("방아쇠를 눌렀지만 피도면 밖만 향했습니다. 도포 구역이 있나요?", ("유효", "미판정")),
        ("최종 높이맵으로 구역마다 몇 초부터 칠했는지 알 수 있나요?", ("시간", "시퀀스")),
        ("HMD 카메라 위치가 위로 움직인 게 위를 본 건가요?", ("회전", "실제 시선")),
        ("같은 번호를 가진 앞면과 뒷면 구역의 도막을 합산해도 돼?", ("면", "맵 번호")),
        ("Ray Point의 XYZ와 높이맵 픽셀의 XY는 같은 좌표인가요?", ("3차원", "UV")),
        ("PNG 높이맵 빨간 픽셀을 바로 과도막 경고로 봐도 됩니까?", ("채널", "두께")),
        ("칠한 곳만 잰 평균과 전체 구역 평균이 다른 이유는?", ("미도장", "분모")),
        ("크롭 히트맵의 일부로 전체 피도면 도막을 판단할 수 있나요?", ("전체", "누적")),
        ("손잡이 자세의 각도랑 분사축과 판 사이 각도는 같은 수치야?", ("자세", "피도면")),
        ("음성 질문이 잘못 받아 적히면 도장 기록으로 원문을 복구하나요?", ("음성 인식", "다시")),
        ("소리를 녹음했는지만 보고 분사 횟수를 확인할 수 있어요?", ("시퀀스", "확인할 수 없습니다")),
        ("RAG가 붙으면 제 훈련을 직접 분석했다는 의미인가요?", ("문서", "별개")),
    ],
)
def test_explains_record_and_interface_semantics(prompt, facts):
    answer = simulator_record_answer(prompt)
    assert answer is not None
    assert all(fact in answer for fact in facts)


@pytest.mark.parametrize(
    "prompt", ["지금 내 Zone 1 과도막이야?", "실제 구역 평균을 알려줘.", "현재 내 분사 각도는 몇 도야?"],
)
def test_does_not_treat_measurement_lookups_as_simulator_definitions(prompt):
    assert simulator_record_answer(prompt) is None
