from __future__ import annotations

from pathlib import Path

from citeweave.io import read_json, write_json
from scripts.audit_complementary_oversight_packets import audit_packets
from scripts.build_complementary_oversight_packets import build_packets


def test_builds_blinded_packets_from_frozen_evidence_pack(tmp_path: Path) -> None:
    source = tmp_path / "source" / "evidence_pack.json"
    write_json(
        source,
        {
            "passed": True,
            "dataset_id": "d1",
            "representative_sources": [
                {
                    "reference_id": "REF-1",
                    "title": "Paper",
                    "year": 2024,
                    "doi": "10/x",
                    "abstract_excerpt": "Evidence.",
                }
            ],
            "graph_phenomena": [
                {
                    "phenomenon_id": "PH-1",
                    "task_type": "bridge_counterfactual",
                    "question": "Is the edge redundant?",
                    "verified_answer": {"alternate_hops": 2},
                    "operator_trace": [{"operator": "delete_edge"}],
                    "interpretation_contract": {
                        "forbidden": ["causality"],
                        "required_limitation": "Corpus-dependent.",
                    },
                    "graph_evidence_ids": ["edge-1"],
                    "reference_ids": ["REF-1"],
                }
            ],
        },
    )
    output = tmp_path / "output"
    manifest = build_packets(pack_paths=[source], output_root=output)
    assert manifest["cases"] == 1
    public = read_json(output / "public" / "PH-1.json")
    internal = read_json(output / "internal" / "PH-1.json")
    assert "role_map" not in public
    assert set(internal["role_map"].values()) == {"support", "challenge"}
    assert manifest["records"][0]["source_pack_sha256"]
    audit = audit_packets(packet_root=output)
    assert audit["status"] == "passed"
