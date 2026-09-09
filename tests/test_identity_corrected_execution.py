import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "identity_execution",
    Path(__file__).parents[1] / "scripts/prepare_identity_corrected_execution.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_binding_rejects_modified_and_missing_inputs(tmp_path):
    path = tmp_path / "input.json"
    path.write_text("original")
    bindings = module.bind([path])
    module.verify(bindings)
    path.write_text("changed")
    with pytest.raises(ValueError, match="drift"):
        module.verify(bindings)
    path.unlink()
    with pytest.raises(ValueError, match="drift"):
        module.verify(bindings)


def test_credentials_cannot_be_frozen(tmp_path):
    with pytest.raises(ValueError, match="Credential"):
        module.bind([tmp_path / "apikey.md"])


def test_existing_outcomes_block_prospective_promotion(tmp_path):
    module.require_no_outcomes([tmp_path])
    (tmp_path / "provider_result.json").write_text("{}")
    with pytest.raises(ValueError, match="empty outcomes"):
        module.require_no_outcomes([tmp_path])


def test_preparation_can_never_request_provider_execution():
    module.validate_command(["scripts/run_formal_v3_panel.py", "--output-plan", "plan.json"])
    for command in (
        ["scripts/run_formal_v3_panel.py", "--execute"],
        ["scripts/run_formal_v3_panel.py", "--api-key-file", "private"],
        ["scripts/run_graph_discovery_experiment.py"],
    ):
        with pytest.raises(ValueError):
            module.validate_command(command)


def test_artifact_validation_detects_nested_drift(tmp_path):
    data = tmp_path / "sidecar.json"
    data.write_text("{}")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {"passed": True, "artifacts": [{"path": data.name, "sha256": module.sha256_file(data)}]}
        )
    )
    module.artifact_bindings(manifest)
    data.write_text("changed")
    with pytest.raises(ValueError, match="drift"):
        module.artifact_bindings(manifest)


def test_immutable_receipts_cannot_be_overwritten(tmp_path):
    path = tmp_path / "receipt.json"
    module.immutable_json(path, {"first": True})
    with pytest.raises(FileExistsError):
        module.immutable_json(path, {"first": False})
    assert json.loads(path.read_text()) == {"first": True}


def test_promotion_requires_exact_explicit_review(tmp_path, monkeypatch):
    path = tmp_path / "qualification.json"
    module.immutable_json(path, {"status": "qualified"})
    monkeypatch.setattr(module, "QUALIFICATION", path)
    with pytest.raises(ValueError, match="Explicit reviewer"):
        module.promote("wrong", "execution auditor")
    with pytest.raises(ValueError, match="Explicit reviewer"):
        module.promote(module.sha256_file(path), "")


def test_resume_verifies_qualification_without_reauditing_outcomes(tmp_path, monkeypatch):
    path = tmp_path / "qualification.json"
    path.write_text("{}")
    monkeypatch.setattr(module, "QUALIFICATION", path)
    calls = []
    monkeypatch.setattr(module, "verify_qualification", lambda: calls.append("verify"))
    monkeypatch.setattr(module, "run", lambda *args: pytest.fail("Must not rerun preparation"))
    module.prepare()
    assert calls == ["verify"]


def test_frozen_configuration_uses_corrected_primary_and_v2_statistics():
    config = module.read_json(module.CONFIG)
    assert config["automatic_api_resume"] is False
    assert len(config["readiness_commands"]) == len(config["plan_commands"]) == 3
    for command in config["readiness_commands"] + config["plan_commands"]:
        module.validate_command(command)
    primary = config["readiness_commands"][0]
    assert (
        module.option(primary, "--benchmark-root").name == "formal_v3_identity_corrected_benchmarks"
    )
    assert (
        module.option(primary, "--statistics-amendment").parent.name
        == "formal_v3_identity_corrected_statistics_v2"
    )
