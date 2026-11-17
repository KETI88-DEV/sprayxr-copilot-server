import pytest

from trainer.demo_api.record_query import Premise, classify_record_question

COATING_8 = {"domain": "coating", "zone_ids": ["8"], "operator": "argmin"}
DISTANCE_8 = {"domain": "distance", "zone_ids": ["8"], "operator": "compare"}
DISTANCE_1 = {"domain": "distance", "zone_ids": ["1"], "operator": "compare"}
CAUSE_8 = {"domain": "coating", "zone_ids": ["8"], "operator": "cause"}

# 공개 질문표 T01~T19(output/trainee-qa-20261114/questions.json). 출력 폴더는 저장소에 없어 문자열로 옮겼다.
PUBLIC_QUESTIONS = [
    "에어리스 스프레이로 칠할 때 스프레이 건과 피도면 사이 거리는 얼마로 유지해야 합니까?",
    "그럼 건 이동 속도는요?",
    "중첩은 몇 퍼센트로 해야 돼요?",
    "도장 작업을 해도 되는 기후 조건이 뭐예요? 상대습도랑 강판 온도·이슬점 기준을 알려 주세요.",
    "IMO PSPC 1차 표면처리에서 블라스팅 청정도와 표면조도 기준은 뭐예요?",
    "90/10 규칙이 뭔지 설명해 주세요. PSPC 에폭시 도장에는 어떻게 적용돼요?",
    "고형분 용적비가 60%인 도료를 칠하고 습도막(WFT)을 재 보니 200µm였어요. 건조도막(DFT)은 얼마쯤 나와요?",
    "스트라이프 코트가 뭐예요?",
    "PSPC 검사에서 평평한 면 DFT는 몇 m²마다 한 군데씩 재요?",
    "도료가 흘러내리는(새깅) 건 왜 생기고 어떻게 막아요?",
    "오렌지 필은 왜 생겨요?",
    "가사 시간이 지난 도료 아까운데 그냥 써도 돼요?",
    "스프레이를 좀 멀리서 뿌리면 어떻게 돼요? 그게 더 고르게 나오지 않아요?",
    "페인트 뿌릴 때 마스크는 그냥 아무거나 써도 돼요?",
    "에어리스 건이 막힌 것 같은데 건 끝을 손가락으로 막아서 확인해도 되나요?",
    "VR로 연습하다가 어지러우면 몇 분 쉬어야 해요?",
    "금속 도장 기능사 실기는 몇 점 이상이면 합격이에요?",
    "고형분 53% 도료를 시너 10% 희석하면 실제 고형분 용적비는 얼마가 돼요? 계산식도요.",
    "에어리스 팁 517은 무슨 뜻이에요?"
]

# 질문시험_v340.csv 40번 원문
LONG_V340_40 = "안녕하세요 저는 이번에 처음 도장 훈련을 받는 사람인데요, 오늘 시뮬레이터에서 스프레이로 칠해 봤는데 화면에 저도막이라고 빨갛게 많이 뜨더라고요. 선배는 건이 너무 가까우면 오히려 흘러내린다고 하고 다른 선배는 멀리 떨어지면 건조 분무가 생긴다고 해서 헷갈립니다. 손목을 꺾지 말라는 얘기도 들었고 패스를 50% 겹치라는 얘기도 들었는데 그게 무슨 뜻인지 잘 모르겠어요. 그리고 속도는 빨리 해야 하는지 천천히 해야 하는지도 모르겠고요. 결국 제가 지금 제일 먼저 신경 써야 하는 게 거리인지 속도인지 겹침인지, 그리고 그 이유가 뭔지 하나만 콕 집어서 알려주시면 좋겠습니다. 너무 길게 말고 핵심만요."

CASES = [  # (번호, 질문, task_type, previous_subject, playback_zone, kind, [(code, zones, operator)])
    ("T20", "제 자세 괜찮아요? 잘하고 있는 거예요?", "pose_recomm", None, "8", "judgement", [("pose_judgement", (), None)]),
    ("T21", "숙련자도 70% 정도라던데, 그럼 저는 나쁜 거예요?", "pose_recomm", None, "8", "judgement", [("pose_judgement", (), None)]),
    ("T27", "화면에 87% 떠 있던데, 내 자세 유사도가 87%면 숙련자(70%)보다 잘한 거 맞지?", "pose_recomm", None, "8", "judgement", [("pose_judgement", (), None)]),
    ("T28", "건 각도는 괜찮았어?", "pose_recomm", None, "8", "fact", [("work_metric", (), None)]),
    ("T29", "다음엔 어디 칠해야 돼?", "trajectory_recomm", None, "8", "fact", [("progress_next", (), None)]),
    ("T30", "빼먹은 데는 없어?", "trajectory_recomm", None, "8", "fact", [("progress_status", (), None)]),
    ("T31", "칠하는 순서가 정해져 있어요? 아무 데서나 시작하면 안 돼요?", "trajectory_recomm", None, "8", "fact",
     [("order_shape", (), "allowed"), ("order_shape", (), "allowed")]),
    # T32 앞 절(계산 방식)은 지식 조각이다(2026-11-17 사례 3). 기록 수치는 그대로 답하고 문서가 계산 방식을 답한다.
    ("T32", "순서 준수율 45%는 어떻게 나온 거고, 숙련자들끼리는 순서가 얼마나 일치해?", "trajectory_recomm", None, "8", "mixed",
     [("order_shape", (), "adherence"), ("order_shape", (), "agreement")]),
    ("T33", "Zone 10이랑 11은 왜 안 칠했다고 나와요?", "trajectory_recomm", None, "8", "fact", [("progress_status", ("10", "11"), None)]),
    ("T34", "다 칠했으니까 이제 끝난 거죠?", "trajectory_recomm", None, "9", "judgement", [("completion", (), None)]),
    ("T35", "숙련자는 보통 어느 Zone에서 시작해서 어디서 끝내요?", "trajectory_recomm", None, "8", "fact", [("order_shape", (), "start")]),
    ("T36", "칠 다 했는데 어디가 제일 얇게 나왔어?", "knowledge_search", None, "9", "fact", [("coating_extreme", (), "argmin")]),
    ("T37", "왜 그렇게 얇게 나온 거야?", "knowledge_search", COATING_8, "9", "cause", [("cause", ("8",), None)]),
    ("T38", "그럼 거기부터 다시 칠하면 돼? 어떻게 칠해?", "knowledge_search", CAUSE_8, "9", "mixed", [("completion", ("8",), None), (None, (), None)]),
    ("T39", "이 Zone 속도랑 이격거리 숙련자랑 비교해줘.", "knowledge_search", None, "8", "fact", [("speed", ("8",), None), ("distance_compare", ("8",), None)]),
    ("T40", "거리가 멀어서 얇게 나온 거지?", "knowledge_search", DISTANCE_8, "8", "cause", [("cause", ("8",), None)]),
    ("T41", "가까이서 천천히 뿌리면 원래 두꺼워지는 거 아냐? 그럼 진짜 원인이 뭐야?", "knowledge_search", CAUSE_8, "8", "mixed", [(None, (), None), ("cause", ("8",), None)]),
    ("T42", "두껍게 너무 많이 뿌린 데도 있어요?", "knowledge_search", None, "1", "fact", [("coating_over", (), None)]),
    ("T43", "Zone 1은 정상이라고 나왔으니까 다 끝난 거지? 거긴 다시 안 가도 되지?", "knowledge_search", None, "1", "judgement", [("completion", ("1",), None), ("completion", ("1",), None)]),
    ("T44", "나 숙련자보다 느리던데, 속도를 숙련자만큼 올려야 해?", "knowledge_search", None, "8", "judgement", [("speed", ("8",), None)]),
    ("T45", "문서엔 적정 이격거리가 30cm라던데 왜 나한테는 멀리 떨어지라고 해?", "knowledge_search", None, "1", "mixed", [("distance_compare", ("1",), None)]),
    ("T46", "그럼 숙련자보다 멀리서 뿌린 Zone은 있었어?", "knowledge_search", DISTANCE_1, "1", "fact", [("distance_filter", (), "higher")]),
    ("T49", "뭐부터 고쳐야 돼? 딱 하나만 콕 집어줘.", "knowledge_search", None, "8", "fact", [("one_priority", (), None), ("one_priority", (), None)]),
    ("T50", "이 기록 도막을 PSPC 90/10 규칙으로 판정하면 합격이에요?", "knowledge_search", None, "9", "mixed", [("external_standard", (), None)]),
    ("v19", "왜 이렇게 얇게 나왔어?", "knowledge_search", None, "8", "cause", [("cause", ("8",), None)]),
    ("v22", "Zone 6 속도랑 거리 어땠어? 뭐부터 고치면 돼?", "knowledge_search", None, "8", "fact",
     [("speed", ("6",), None), ("distance_compare", ("6",), None), ("one_priority", (), None)]),
    ("v33", "과도막은 왜 생겨? 내 기록에서도 과도막 난 데 있어?", "knowledge_search", None, "1", "mixed", [(None, (), None), ("coating_over", (), None)]),
]


def _classify(prompt, task="knowledge_search", subject=None, playback="8"):
    return classify_record_question(prompt, task_type=task, previous_subject=subject, playback_zone=playback)


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_record_questions_are_classified(case) -> None:
    _, prompt, task, subject, playback, kind, parts = case
    result = _classify(prompt, task, subject, playback)
    assert result.kind == kind
    assert [(p.code, p.zones, p.operator) for p in result.parts] == parts


def _text(case_id):
    return next(case[1] for case in CASES if case[0] == case_id)


def _premises(prompt, task, subject=None, playback="8"):
    return {premise for part in _classify(prompt, task, subject, playback).parts for premise in part.premises}


def test_premises_are_extracted() -> None:
    assert Premise("named_number", "pose_similarity", 87.0) in _premises(_text("T27"), "pose_recomm")
    assert Premise("cause", "far>thin", None) in _premises(_text("T40"), "knowledge_search", DISTANCE_8)
    assert Premise("conclusion", "visited_all", None) in _premises(_text("T34"), "trajectory_recomm", None, "9")
    assert Premise("conclusion", "grade_normal", None) in _premises(_text("T43"), "knowledge_search", None, "1")
    assert Premise("comparison", "speed_slower", None) in _premises(_text("T44"), "knowledge_search")
    assert [p.cardinality for p in _classify(_text("T49")).parts] == ["one", "one"]
    assert [p.cardinality for p in _classify(_text("v22")).parts] == ["any", "any", "any"]


@pytest.mark.parametrize("task,prompt", [
    ("pose_recomm", "그럼 뭐부터 고치면 돼요?"),
    ("pose_recomm", "왼팔은?"),
    ("pose_recomm", "그럼 오른쪽 팔꿈치는 몇 도 더 펴야 돼?"),
    ("pose_recomm", "자세가 제일 문제였던 구역이 어디야?"),
    ("pose_recomm", "지금 재생 중인 분사(SEQ-002) 팔 교정 내용이랑 근거 알려줘."),
    ("trajectory_recomm", "도포 방향은?"),
    ("trajectory_recomm", "권장 Zone 순서는?"),
])
def test_existing_paths_keep_their_questions(task, prompt) -> None:
    assert _classify(prompt, task).kind == "none"


@pytest.mark.parametrize("prompt", ["적정 이격거리는?", "에어리스 스프레이 분사 속도는 얼마가 적당해?", "저도막이 뭐야?", "적정 분사 속도는?"])
def test_knowledge_questions_in_the_viewer_are_not_record_questions(prompt) -> None:
    assert _classify(prompt).kind == "none"


@pytest.mark.parametrize("prompt", PUBLIC_QUESTIONS)
def test_public_questions_are_not_record_questions(prompt) -> None:
    assert _classify(prompt, playback=None).kind == "none"


def test_bare_first_person_does_not_make_a_record_question() -> None:
    assert _classify("제가 VR 하다 어지러운데 몇 분 쉬어야 해요?").kind == "none"


def test_long_question_v340_40_is_not_a_record_question() -> None:
    # 예전에는 섞인 질문이었다(기록 우선순위 + 문서). 2026-11-17 사례 1: 일반 요인 중 고르기라 문서로만 답한다.
    assert _classify(LONG_V340_40).kind == "none"


def test_pronoun_without_a_previous_subject_uses_the_playback_zone() -> None:
    first = _classify("그럼 거기부터 다시 칠하면 돼?", subject=None, playback="9").parts[0]
    assert (first.zones, first.zone_source) == (("9",), "playback")


def test_subject_from_another_family_is_not_used() -> None:
    subject = {"domain": "pose", "zone_ids": ["9"], "operator": "argmin"}
    assert _classify("왜 그렇게 얇게 나온 거야?", subject=subject, playback="8").parts[0].zones == ("8",)


@pytest.mark.parametrize("prompt", [
    "도장 작업 시작 전에 뭘 점검해야 해요?",
    "스프레이 건을 어디서 시작해서 어디서 끝내요?",
    "도장 작업 뭐부터 해야 돼요?",
    "몇 코트까지 칠해야 끝난 거예요?",
])
def test_bare_words_without_a_record_signal_are_not_record_questions(prompt) -> None:
    assert _classify(prompt).kind == "none"


def test_a_part_splits_at_go_only_when_both_halves_are_record_questions() -> None:
    both = _classify("내 거리 비교하고 다음 Zone 알려줘")
    assert [part.code for part in both.parts] == ["distance_compare", "progress_next"]
    assert [part.code for part in _classify("제 자세 괜찮아요? 잘하고 있는 거예요?", "pose_recomm").parts] == ["pose_judgement"]
    assert len(_classify("제 자세 잘하고 있는 거예요?", "pose_recomm").parts) == 1
