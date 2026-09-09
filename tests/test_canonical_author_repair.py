from pathlib import Path

import pandas as pd
import pytest

from citeweave.canonical_author_repair import repair_author_layer
from citeweave.canonical_identity_audit import is_placeholder_identity
from citeweave.io import sha256_file, write_json, write_parquet


def test_sentinel_detection_does_not_flag_scoped_occurrences():
    assert all(
        is_placeholder_identity(v)
        for v in (None, "openalex-author:None", "openalex-author:", "nan")
    )
    assert not is_placeholder_identity("openalex-author-occurrence:123")
    assert not is_placeholder_identity("openalex-author:A123")


def test_repair_removes_false_cross_work_bridge_and_preserves_other_fields(tmp_path: Path):
    source = tmp_path / "source"
    canonical = source / "canonical"
    write_parquet(
        canonical / "authors.parquet",
        pd.DataFrame(
            [
                {
                    "author_id": "openalex-author:None",
                    "name": "Incorrect first person",
                    "orcid": None,
                },
                {"author_id": "known:A", "name": "A", "orcid": None},
                {"author_id": "known:B", "name": "B", "orcid": None},
            ]
        ),
    )
    memberships = pd.DataFrame(
        {
            "work_id": ["w1", "w1", "w2", "w2"],
            "author_id": ["known:A", "openalex-author:None", "known:B", "openalex-author:None"],
            "position": [1, 2, 1, 2],
            "institution_id": ["i1", "i1", "i2", "i2"],
        }
    )
    write_parquet(canonical / "authorships.parquet", memberships)
    write_parquet(
        canonical / "works.parquet",
        pd.DataFrame({"work_id": ["w1", "w2"], "cited_by_count": [0, 0]}),
    )
    write_json(source / "audit" / "processing_manifest.json", {"edge_row_limit": 100})
    before = sha256_file(canonical / "authorships.parquet")
    output = tmp_path / "corrected"
    receipt = repair_author_layer(source, output, candidate_pool=10)
    assert receipt["unresolved_occurrence_nodes"] == 2
    assert receipt["graph_statistics"]["all_edges"] == {
        "nonisolated_nodes": 4,
        "edges": 2,
        "pair_incidence_mass": 2,
    }
    fixed = pd.read_parquet(output / "canonical" / "authorships.parquet")
    assert fixed.loc[1, "author_id"] != fixed.loc[3, "author_id"]
    pd.testing.assert_frame_equal(
        fixed.drop(columns="author_id"), memberships.drop(columns="author_id")
    )
    assert sha256_file(canonical / "authorships.parquet") == before
    assert receipt["status"] == "isolated_correction_not_promoted"
    with pytest.raises(ValueError, match="overwrite"):
        repair_author_layer(source, output)
