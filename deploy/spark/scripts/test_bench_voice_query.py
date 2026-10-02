from deploy.spark.scripts.bench_voice_query import percentile, summarize


def test_percentile_and_summary_are_stable_for_small_samples() -> None:
    values = [40.0, 10.0, 30.0, 20.0]

    assert percentile(values, 0.50) == 30.0
    assert percentile(values, 0.95) == 40.0
    assert summarize(values) == {
        "count": 4,
        "mean_ms": 25.0,
        "p50_ms": 30.0,
        "p95_ms": 40.0,
        "max_ms": 40.0,
    }


def test_empty_summary_is_zeroed() -> None:
    assert summarize([]) == {
        "count": 0,
        "mean_ms": 0.0,
        "p50_ms": 0.0,
        "p95_ms": 0.0,
        "max_ms": 0.0,
    }
