from __future__ import annotations

from pathlib import Path

import pandas as pd

from citeweave.article_evidence_pack import _representative_sources


def test_representative_sources_bind_each_phenomenon(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    pd.DataFrame(
        {
            "work_id": [f"w{i}" for i in range(10)],
            "title": [f"paper {i}" for i in range(10)],
            "abstract": [f"abstract {i}" for i in range(10)],
            "year": [2020 + i % 3 for i in range(10)],
            "doi": [f"10.x/{i}" for i in range(10)],
            "cited_by_count": list(range(10)),
        }
    ).to_parquet(canonical / "works.parquet", index=False)
    pd.DataFrame(
        {
            "work_id": [f"w{i}" for i in range(10)],
            "keyword": ["alpha"] * 5 + ["beta"] * 5,
        }
    ).to_parquet(canonical / "keywords.parquet", index=False)
    tasks = [
        {"item_id": "a", "answer": {"node": "alpha"}},
        {"item_id": "b", "answer": {"node": "beta"}},
    ]
    sources, dependencies = _representative_sources(
        tmp_path, tasks, per_phenomenon=3
    )
    assert len(sources) == 6
    assert len(dependencies["a"]) == 3
    assert len(dependencies["b"]) == 3
    assert set(dependencies["a"]).isdisjoint(dependencies["b"])


def test_topic_task_bm25_beats_generic_high_citation_match(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    pd.DataFrame(
        {
            "work_id": ["generic", "relevant", "relevant2"],
            "title": [
                "General chemistry review",
                "Perovskite solar cell stability under moisture",
                "Degradation pathways in perovskite photovoltaics",
            ],
            "abstract": [
                "Materials science and chemistry across many fields.",
                "Perovskite solar cells degrade under moisture and heat.",
                "Stability testing identifies degradation in perovskite devices.",
            ],
            "year": [2025, 2024, 2023],
            "doi": ["10.x/generic", "10.x/relevant", "10.x/relevant2"],
            "cited_by_count": [100_000, 20, 10],
        }
    ).to_parquet(canonical / "works.parquet", index=False)
    pd.DataFrame(
        {
            "work_id": ["generic", "relevant", "relevant2"],
            "keyword": ["Chemistry", "Perovskite", "Stability"],
        }
    ).to_parquet(canonical / "keywords.parquet", index=False)
    tasks = [
        {
            "item_id": "stability",
            "task_type": "temporal_structural_shift",
            "question": "How does the stability literature change over time?",
            "answer": {"node": "Chemistry"},
        }
    ]
    sources, _ = _representative_sources(
        tmp_path,
        tasks,
        per_phenomenon=2,
        topic_hint="perovskite_solar_cell_stability_2012_2025",
    )
    selected_titles = {row["title"] for row in sources}
    assert "General chemistry review" not in selected_titles
    assert all(row["selection_basis"].startswith("topic_task_bm25") for row in sources)


def test_representative_sources_deduplicate_title_versions(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    pd.DataFrame(
        {
            "work_id": ["v1", "v2", "unique"],
            "title": [
                "Federated Learning in Healthcare!",
                "Federated learning in healthcare",
                "Privacy in federated clinical learning",
            ],
            "abstract": ["federated healthcare"] * 3,
            "year": [2024, 2025, 2023],
            "doi": ["10.x/v1", "10.x/v2", "10.x/unique"],
            "cited_by_count": [10, 20, 5],
        }
    ).to_parquet(canonical / "works.parquet", index=False)
    pd.DataFrame(
        {
            "work_id": ["v1", "v2", "unique"],
            "keyword": ["Federated learning", "Healthcare", "Clinical"],
        }
    ).to_parquet(canonical / "keywords.parquet", index=False)
    sources, dependencies = _representative_sources(
        tmp_path,
        [
            {
                "item_id": "federated",
                "task_type": "multi_hop_connector",
                "question": "How is federated learning connected in healthcare?",
                "answer": {"node": "Healthcare"},
            }
        ],
        per_phenomenon=2,
        topic_hint="federated_learning_healthcare_2016_2025",
    )
    assert len(sources) == 2
    assert len(dependencies["federated"]) == 2
    normalized = {
        " ".join(character.lower() for character in row["title"] if character.isalnum())
        for row in sources
    }
    assert len(normalized) == 2
