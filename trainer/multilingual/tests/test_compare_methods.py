from trainer.multilingual.compare_methods import SAMPLING, grounded_request, successful_content, translation_request


def test_grounding_changes_only_user_context_and_does_not_mutate_baseline():
    baseline = {"question": "What is DFT?", "request": {"model": "test", "messages": [{"role": "system", "content": "Answer in English"}, {"role": "user", "content": "What is DFT?"}], **SAMPLING}}
    evidence = {"chunks": [{"chunk_id": "DOC-1", "text": "건조도막두께"}]}
    result = grounded_request(baseline, evidence)
    assert result["messages"][0] == baseline["request"]["messages"][0]
    assert "[DOC-1]\n건조도막두께" in result["messages"][-1]["content"]
    assert result["messages"][-1]["content"].endswith("What is DFT?")
    assert baseline["request"]["messages"][-1]["content"] == "What is DFT?"


def test_translation_uses_generated_korean_answer_not_reference_answer():
    result = translation_request("test", "실제 모델 답변", "en", "English")
    assert result["messages"][-1] == {"role": "user", "content": "실제 모델 답변"}
    assert result["max_tokens"] == SAMPLING["max_tokens"]
    assert result["messages"][2]["role"] == "assistant"


def test_low_resource_translation_has_no_fabricated_fewshot():
    result = translation_request("test", "실제 모델 답변", "my", "Burmese")
    assert len(result["messages"]) == 2
    assert "Burmese" in result["messages"][0]["content"]


def test_incomplete_korean_generation_is_not_forwarded_to_translation():
    assert successful_content({"http_status": 200, "response": {"choices": [{"finish_reason": "length", "message": {"content": "부분 답변"}}]}}) is None
    assert successful_content({"http_status": 500}) is None
    assert successful_content({"http_status": 200, "response": {"choices": [{"finish_reason": "stop", "message": {"content": "완료"}}]}}) == "완료"
