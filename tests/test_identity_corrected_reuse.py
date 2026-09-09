import importlib.util
from pathlib import Path

from citeweave.io import write_json

spec = importlib.util.spec_from_file_location(
    "reuse_audit", Path(__file__).parents[1] / "scripts" / "audit_identity_corrected_reuse.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_reuse_requires_exact_payload_context_and_messages():
    identity = {
        "task_payload_sha256": "task",
        "context_sha256": "context",
        "request_messages_sha256": "message",
    }
    assert module.identity_difference(identity, identity) == []
    for field in identity:
        assert module.identity_difference({**identity, field: "drift"}, identity) == [field]


def test_full_360_reuse_audit_with_real_readiness_schema(tmp_path, monkeypatch):
    class TestTokenizer:
        def __init__(self, *args, **kwargs):
            pass

        def count(self, value):
            return len(value)

    monkeypatch.setattr(module, "CommandTokenizer", TestTokenizer)
    roots = {p: tmp_path / p for p in ("primary", "replication", "extension")}
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer = {"context_token_budget": 1000}
    write_json(tokenizer_path, tokenizer)
    neural_path = tmp_path / "neural.json"
    artifacts, records, cells = [], [], []
    for number in range(8):
        topic = f"topic_{number}"
        panel = "primary" if number < 4 else "replication"
        tasks = [
            {
                "item_id": f"{topic}_{i}",
                "task_type": "test",
                "question": "test?",
                "complexity": 2 if i < 15 else 1,
                "answer": {"n": 1},
                "contexts": {
                    condition: {"records": [{"id": "e", "n": 1}]} for condition in module.CONDITIONS
                },
            }
            for i in range(21)
        ]
        source = roots[panel] / topic / "benchmark.json"
        target = roots["extension"] / topic / "benchmark.json"
        write_json(source, {"tasks": tasks[:15]})
        write_json(target, {"tasks": tasks})
        records.append(
            {
                "dataset_id": topic,
                "source_panel": panel,
                "benchmark_sha256": module.sha256_file(target),
            }
        )
        if panel == "primary":
            sidecar = tmp_path / f"{topic}_sidecar.json"
            write_json(
                sidecar,
                {
                    "source_benchmark_sha256": module.sha256_file(source),
                    "contexts": {
                        t["item_id"]: t["contexts"]["flat_neural_dense"] for t in tasks[:15]
                    },
                },
            )
            artifacts.append(
                {
                    "artifact_type": "neural_context_sidecar",
                    "dataset_id": topic,
                    "path": sidecar.name,
                    "sha256": module.sha256_file(sidecar),
                }
            )
        for task in tasks:
            for condition in module.CONDITIONS:
                messages, audit = module.build_messages(
                    task, condition, tokenizer=TestTokenizer(), token_budget=1000
                )
                cells.append(
                    {
                        "dataset_id": topic,
                        "item_id": task["item_id"],
                        "condition": condition,
                        "task_payload_sha256": module.task_payload_sha256(task),
                        "context_sha256": audit["context_sha256"],
                        "request_messages_sha256": module.canonical_sha256(messages),
                    }
                )
    write_json(roots["extension"] / "construction_manifest.json", {"records": records})
    write_json(neural_path, {"artifacts": artifacts})
    readiness = {}
    for panel in roots:
        payload = {
            "status": "ready",
            "tokenizer_manifest_path": str(tokenizer_path),
            "tokenizer_manifest": tokenizer,
        }
        if panel == "primary":
            payload.update(
                {
                    "neural_dense_manifest_path": str(neural_path),
                    "neural_dense_manifest": {"artifacts": artifacts},
                }
            )
        if panel == "extension":
            payload.update(
                {
                    "cell_identity_manifest": cells,
                    "cell_identity_manifest_sha256": module.canonical_sha256(cells),
                    "tokenizer_manifest_sha256": module.sha256_file(tokenizer_path),
                }
            )
        readiness[panel] = tmp_path / f"{panel}_readiness.json"
        write_json(readiness[panel], payload)
    args = {
        **{f"{k}_root": v for k, v in roots.items()},
        **{f"{k}_readiness": v for k, v in readiness.items()},
    }
    result = module.audit_reuse(**args)
    assert result["status"] == "all_360_reuse_identities_equal"
    assert result["source_counts"] == {"primary": 240, "replication": 120}
    drift = module.read_json(roots["primary"] / "topic_0" / "benchmark.json")
    drift["tasks"][0]["contexts"]["flat_hybrid"]["records"][0]["n"] = 2
    # Drift in evidence is caught even if its sidecar source binding is updated.
    write_json(roots["primary"] / "topic_0" / "benchmark.json", drift)
    sidecar = tmp_path / "topic_0_sidecar.json"
    payload = module.read_json(sidecar)
    payload["source_benchmark_sha256"] = module.sha256_file(
        roots["primary"] / "topic_0" / "benchmark.json"
    )
    write_json(sidecar, payload)
    primary = module.read_json(readiness["primary"])
    primary["neural_dense_manifest"]["artifacts"][0]["sha256"] = module.sha256_file(sidecar)
    write_json(readiness["primary"], primary)
    assert module.audit_reuse(**args)["status"] == "reuse_identity_mismatch"
