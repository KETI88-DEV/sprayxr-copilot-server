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
        ("raytexuv의 Y가 커질수록 PNG에서 아래쪽인가요?", ("좌하단", "위", "아래")),
        ("히트맵 크롭의 width와 PNG img_width가 다른데 배치에는 어느 쪽을 써요?", ("원본", "해상도")),
        ("같은 위치에서 이전 누적 히트맵이 170µm이고 현재 누적값이 190µm면 합산하나요?", ("190µm", "20µm", "교체")),
        ("학습모드 3과 학습모드 8의 번호가 같은 Zone을 한꺼번에 비교해도 돼요?", ("시나리오", "별개")),
        ("Replay 리포트가 pending인데 UI에 평가값부터 그리면 되나요?", ("report_id", "complete")),
        ("analysis_status가 unavailable이면 유사도를 0으로 표시해야 하나요?", ("null", "평가")),
        ("use_rag=Y인데 retrieval_used=false면 API에 오류가 난 건가요?", ("허용", "근거", "오류")),
        ("두 옵션 response_mode와 summarize는 둘 다 답변 길이를 바꿔요?", ("모델", "분량", "별개")),
        ("최종 도막 리포트에서 전문가 기준이 미등록이면 세션 목표로 판정할 수 없어요?", ("세션 목표", "높이맵", "전문가")),
    ],
)
def test_explains_record_and_interface_semantics(prompt, facts):
    answer = simulator_record_answer(prompt)
    assert answer is not None
    assert all(fact in answer for fact in facts)


@pytest.mark.parametrize(
    "prompt",
    [
        "지금 내 Zone 1 과도막이야?", "실제 구역 평균을 알려줘.", "현재 내 분사 각도는 몇 도야?",
        "이전 누적에서 2번 구역은 170µm였는데 현재 누적의 3번 구역과 두께 차이 계산 가능해?",
    ],
)
def test_does_not_treat_measurement_lookups_as_simulator_definitions(prompt):
    assert simulator_record_answer(prompt) is None
