from pathlib import Path

import pandas as pd
import pytest

from citeweave.graph_scope_audit import audit_workspace_graph_scope
from citeweave.io import write_json


def test_scope_uses_full_membership_denominator_without_building_all_pairs(tmp_path: Path) -> None:
    workspace = tmp_path / "topic"
    canonical = workspace / "canonical"
    graph_root = canonical / "visualization"
    graph_root.mkdir(parents=True)
    pd.DataFrame(
        {"work_id": ["w1", "w1", "w1", "w2", "w2", "w2"], "keyword": ["A", "B", "C", "A", "B", "B"]}
    ).to_parquet(canonical / "keywords.parquet")
    pd.DataFrame({"keyword": ["A", "B", "C"], "occurrences": [2, 2, 1]}).to_parquet(
        graph_root / "keyword_occurrences.parquet"
    )
    pd.DataFrame({"source_id": ["A"], "target_id": ["B"], "weight": [2]}).to_parquet(
        graph_root / "keyword_cooccurrence_edges.parquet"
    )
    write_json(
        workspace / "audit" / "processing_manifest.json",
        {"candidate_pool_size": 2, "edge_row_limit": 100},
    )
    result = audit_workspace_graph_scope(workspace, {("keyword_cooccurrence", "large")})
    row = result["records"][0]
    assert row["raw_distinct_entities"] == 3
    assert row["retained_graph_nodes"] == 2
    assert row["raw_unique_work_entity_memberships"] == 5
    assert row["retained_entity_fraction"] == pytest.approx(2 / 3)
    assert row["retained_membership_fraction"] == pytest.approx(4 / 5)
    assert row["raw_pair_incidence_mass"] == 4
    assert row["retained_pair_incidence_fraction"] == 0.5
    assert row["work_coverage_any_retained_entity"] == 1
    assert row["is_uncapped_corpus_graph"] is False
    assert row["accounting_issues"] == []
