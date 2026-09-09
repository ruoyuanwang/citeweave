import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "binding", Path(__file__).parents[1] / "scripts" / "rebind_identity_corrected_statistics.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_rebinding_preserves_original_statistical_decisions():
    original = {
        "amendment_id": "old",
        "scale_pair_artifact": {"path": "old.json", "sha256": "old", "groups": 14},
        "prior_amendments": ["a", "b"],
        "reason": ["initial"],
        "primary_hypotheses": {"H1": {"test": "exact_mcnemar"}},
        "statistics": {"seed": 42, "multiplicity": "holm"},
    }
    result = module.rebound_statistics(original, pair_sha256="new", correction_hash="correction")
    assert result["primary_hypotheses"] == original["primary_hypotheses"]
    assert result["statistics"] == original["statistics"]
    assert result["scale_pair_artifact"]["groups"] == 14
    assert original["scale_pair_artifact"]["sha256"] == "old"
    assert result["scale_pair_artifact"]["sha256"] == "new"
    assert result["prior_amendments"] == original["prior_amendments"] == ["a", "b"]
    assert result["identity_correction_amendment_sha256"] == "correction"
