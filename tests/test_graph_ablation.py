import json
from types import SimpleNamespace

import pandas as pd
from test_graph_explanation import FakeClient, sample_network

from citeweave.graph_ablation import (
    _freeze_case,
    rescore_graph_ablation,
    run_graph_ablation,
)
from citeweave.models import GraphExplanationPolicy
from citeweave.visualization import FigureArtifact


def test_ablation_preserves_all_three_groups(tmp_path):
    image = tmp_path / "network_keyword_cooccurrence.png"
    image.write_bytes(b"fake-png-for-test-client")
    figure = FigureArtifact(
        "network_keyword_cooccurrence",
        image,
        tmp_path / "network_keyword_cooccurrence.svg",
        {},
        {},
    )
    analyses = SimpleNamespace(networks={"keyword_cooccurrence": sample_network()})
    policy = GraphExplanationPolicy(mode="graph_rag", temperature=0.0)
    output = tmp_path / "ablation"

    result = run_graph_ablation(
        analyses,
        [figure],
        policy,
        max_nodes=10,
        repeats=1,
        output=output,
        client=FakeClient(),
    )

    assert result["runs"] == 3
    assert result["successful_runs"] == 3
    assert result["failed_runs"] == 0
    assert (output / "records.jsonl").exists()
    assert (output / "summary.md").exists()
    assert (output / "cases" / "keyword_cooccurrence.json").exists()
    records = [
        json.loads(line)
        for line in (output / "records.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len({record["case_id"] for record in records}) == 1
    summary = pd.read_csv(output / "summary.csv")
    assert summary["mode"].tolist() == ["vlm", "flat_kg", "graph_rag"]
    assert summary["successful_runs"].tolist() == [1, 1, 1]
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["execution_order_strategy"] == "deterministic_case_rotated_latin_square"
    assert len(manifest["execution_order"]) == 3
    assert manifest["audit_hashes"]["output_schema_sha256"]
    assert manifest["audit_hashes"]["system_prompt_sha256"]
    assert manifest["protocol_version"] == "2.1-copy-ready-path-focus-normalization"
    assert len(manifest["prompt_hashes"]) == 3
    assert sorted(record["execution_position"] for record in records) == [1, 2, 3]
    for mode in ("vlm", "flat_kg", "graph_rag"):
        assert (
            output
            / "runs"
            / mode
            / "keyword_cooccurrence"
            / "repeat-01.json"
        ).exists()

    rescored = rescore_graph_ablation(output, output=tmp_path / "rescored")

    assert rescored["runs"] == 3
    assert rescored["successful_runs"] == 3
    assert (tmp_path / "rescored" / "summary.csv").exists()
    assert "没有重新调用模型" in (
        tmp_path / "rescored" / "summary.md"
    ).read_text(encoding="utf-8")


def test_case_identity_ignores_absolute_figure_path(tmp_path):
    first = tmp_path / "copy-a" / "network_keyword_cooccurrence.png"
    second = tmp_path / "copy-b" / "network_keyword_cooccurrence.png"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_bytes(b"identical-image")
    second.write_bytes(b"identical-image")
    network = sample_network()
    first_figure = FigureArtifact(
        "network_keyword_cooccurrence", first, first.with_suffix(".svg"), {}, {}
    )
    second_figure = FigureArtifact(
        "network_keyword_cooccurrence", second, second.with_suffix(".svg"), {}, {}
    )

    first_case = _freeze_case(network, first_figure, 10)
    second_case = _freeze_case(network, second_figure, 10)

    assert first_case["figure"] != second_case["figure"]
    assert first_case["case_id"] == second_case["case_id"]
