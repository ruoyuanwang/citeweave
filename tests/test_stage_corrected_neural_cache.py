import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from citeweave.io import read_json, write_json

spec = importlib.util.spec_from_file_location(
    "cache_stage", Path(__file__).parents[1] / "scripts" / "stage_corrected_neural_cache.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_partial_cache_copy_keeps_prefix_and_refuses_author_or_changed_graph(tmp_path):
    old, new = tmp_path / "old", tmp_path / "new"
    for root in (old, new):
        graph = root / "canonical" / "visualization"
        graph.mkdir(parents=True)
        for name in module.GRAPH_TABLES["keyword_cooccurrence"]:
            pd.DataFrame({"x": [1, 2]}).to_parquet(graph / name)
    source = tmp_path / "cache"
    source.mkdir()
    np.save(
        source / "embeddings.building.npy",
        np.array([[1.0, 0.0], [0.0, 1.0], [np.nan, np.nan]], dtype=np.float32),
    )
    write_json(
        source / "manifest.json",
        {
            "status": "in_progress",
            "document_serialization": "semantic_entity_relation_json_v2",
            "next_start": 2,
            "row_count": 3,
            "embedding_dimension": 2,
        },
    )
    dest = tmp_path / "copy"
    receipt = module.copy_unchanged_cache(source, dest, old, new, "keyword_cooccurrence")
    assert receipt["committed_rows"] == 2
    assert read_json(dest / "manifest.json") == read_json(source / "manifest.json")
    with pytest.raises(ValueError, match="Only unchanged"):
        module.copy_unchanged_cache(source, tmp_path / "author", old, new, "coauthorship")
    pd.DataFrame({"x": [9]}).to_parquet(
        new / "canonical" / "visualization" / "keyword_occurrences.parquet"
    )
    with pytest.raises(ValueError, match="content/order changed"):
        module.copy_unchanged_cache(source, tmp_path / "bad", old, new, "keyword_cooccurrence")
