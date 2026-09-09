import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "materialize_temporal",
    Path(__file__).parents[1] / "scripts/materialize_temporal_control_requests.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_shared_annex_is_checked_after_actual_message_serialization():
    annex = {"records": [{"keyword": "a", "year_document_counts": [1, 2, 3]}]}
    messages = [
        {"role": "system", "content": "test"},
        {
            "role": "user",
            "content": "QUESTION:test\nCONTEXT:\n"
            + json.dumps({"rows": [], "shared_temporal_evidence": annex}),
        },
    ]
    module.verify_annex_retention(messages, annex)
    with pytest.raises(ValueError, match="changed or truncated"):
        module.verify_annex_retention(messages, {"records": []})
