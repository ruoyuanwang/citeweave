import pytest

from citeweave.formal_protocol import (
    query_payload,
    query_set_sha256,
    sha256_json,
    verify_frozen_query_registry,
)


def _registry():
    dataset = {
        "id": "topic-a",
        "role": "locked",
        "topic": "Topic A",
        "source": "crossref",
        "keywords": ["concept one", "concept two"],
        "query_mode": "all",
        "year_from": 2010,
        "year_to": 2020,
        "document_types": ["journal-article"],
        "language": None,
        "max_records": None,
        "query_status": "frozen",
    }
    dataset["query_sha256"] = sha256_json(query_payload(dataset))
    return {
        "status": "frozen",
        "datasets": [dataset],
        "query_freeze_attestation": {
            "query_set_sha256": query_set_sha256([dataset]),
            "accepted_ids": ["topic-a"],
        },
    }


def test_frozen_query_registry_rejects_keyword_tampering():
    registry = _registry()
    verify_frozen_query_registry(registry)

    registry["datasets"][0]["keywords"].append("silent replacement")

    with pytest.raises(ValueError, match="query-set hash"):
        verify_frozen_query_registry(registry)


def test_frozen_query_registry_rejects_record_caps():
    registry = _registry()
    registry["datasets"][0]["max_records"] = 300
    registry["datasets"][0]["query_sha256"] = sha256_json(
        query_payload(registry["datasets"][0])
    )
    registry["query_freeze_attestation"]["query_set_sha256"] = query_set_sha256(
        registry["datasets"]
    )

    with pytest.raises(ValueError, match="max_records"):
        verify_frozen_query_registry(registry)
