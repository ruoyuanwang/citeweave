import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "robust_implications",
    Path(__file__).parents[1] / "scripts/derive_graph_robustness_implications.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_thresholding_loss_is_not_misattributed_to_deleting_the_hub():
    def row(before, after):
        return {
            "graph": {"largest_component_fraction": before},
            "measurements": {
                "hub_removal_resilience": {"largest_component_fraction_of_original_nodes": after}
            },
        }

    result = module.deletion_decomposition(row(1.0, 0.9995), row(0.5675, 0.555))
    assert result["total_post_deletion_change"] == pytest.approx(-0.4445)
    assert result["pre_deletion_topology_change"] == pytest.approx(-0.4325)
    assert result["additional_marginal_deletion_loss"] == pytest.approx(0.012)
