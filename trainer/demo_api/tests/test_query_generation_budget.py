from trainer.scripts.agent_pipeline import fast_response_budget


def test_detailed_queries_have_a_larger_real_generation_budget_than_quick():
    config = {"generation": {"max_new_tokens": 1024}}
    assert fast_response_budget(config, 128) == 128
    assert fast_response_budget(config, 512) == 512
    assert fast_response_budget(config, None) == 256
    assert fast_response_budget(config, 100000) == 1024
    assert fast_response_budget({"generation": {"max_new_tokens": 400}}, 512) == 400
