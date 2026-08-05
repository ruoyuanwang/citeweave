from citeweave.scale import run_duckdb_benchmark


def test_scale_benchmark_full_counts_and_bounds_candidates() -> None:
    result = run_duckdb_benchmark(documents=10_000, terms_per_document=4)
    assert result["passed"]
    assert result["relationship_rows"] == 40_000
    assert result["selected_candidates"] == 800
