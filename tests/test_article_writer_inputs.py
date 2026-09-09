from __future__ import annotations

from pathlib import Path

from citeweave.article_writer_inputs import build_text_only_writer_inputs
from citeweave.io import read_json, sha256_file, write_json


def test_builds_text_only_inputs_and_retains_evaluator_figure(tmp_path: Path) -> None:
    records = []
    for index in range(8):
        topic = f"topic_{index}"
        topic_dir = tmp_path / "source" / topic
        topic_dir.mkdir(parents=True)
        figure_path = topic_dir / "graph.png"
        figure_path.write_bytes(b"frozen-image")
        pack_path = topic_dir / "evidence_pack.json"
        write_json(
            pack_path,
            {
                "status": "same_evidence_pack_frozen_before_articles",
                "writing_brief": {"word_target": 3000},
                "condition_contract": {"shared_inputs": ["graph_overview_figure"]},
                "graph_phenomena": [{"phenomenon_id": "PH-1"}],
                "representative_sources": [{"reference_id": "REF-1"}],
                "figures": [
                    {
                        "path": str(figure_path.resolve()),
                        "sha256": sha256_file(figure_path),
                        "displayed_nodes": 90,
                    }
                ],
            },
        )
        records.append(
            {
                "dataset_id": topic,
                "source_panel": "primary",
                "pack": str(pack_path.resolve()),
                "pack_sha256": sha256_file(pack_path),
            }
        )
    source_manifest = tmp_path / "source" / "manifest.json"
    write_json(
        source_manifest,
        {"status": "same_evidence_packs_ready", "records": records},
    )

    manifest = build_text_only_writer_inputs(
        source_manifest,
        output_dir=tmp_path / "text_only",
    )
    assert manifest["status"] == "text_only_same_evidence_writer_inputs_ready"
    assert manifest["datasets"] == 8
    for record in manifest["records"]:
        writer_input = read_json(Path(record["pack"]))
        serialized = Path(record["pack"]).read_text(encoding="utf-8")
        assert "graph.png" not in serialized
        assert "figures" not in writer_input
        assert writer_input["posthoc_figure_commitment"]["figure_sha256"]
        assert record["rendered_figure_exposed_to_writer"] is False
        assert Path(record["evaluation_figure"]).is_file()
