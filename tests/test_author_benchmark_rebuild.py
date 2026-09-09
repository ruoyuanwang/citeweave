import copy

import pytest

from citeweave.author_benchmark_rebuild import replace_author_tasks


def fixture():
    authors = [
        {"item_id": f"a{i}", "network": "coauthorship", "scale": "large", "answer": i}
        for i in range(5)
    ]
    others = [
        {"item_id": f"k{i}", "network": "keyword_cooccurrence", "scale": "large", "answer": i}
        for i in range(20)
    ]
    original = {
        "tasks": authors + others,
        "scales": [{"network": "coauthorship", "name": "large", "nodes": 99}],
    }
    component = {
        "tasks": [{**t, "answer": 100} for t in authors],
        "scales": [{"network": "coauthorship", "name": "large", "nodes": 98}],
    }
    return original, component


def test_author_only_rebuild_keeps_other_content_and_old_object_immutable():
    original, component = fixture()
    before = copy.deepcopy(original)
    rebuilt, receipt = replace_author_tasks(original, component)
    assert original == before
    assert rebuilt["tasks"][5:] == original["tasks"][5:]
    assert receipt["unchanged_tasks"] == 20
    assert receipt["rebuilt_author_tasks"] == 5
    assert all(t["answer"] == 100 for t in rebuilt["tasks"][:5])


def test_missing_author_task_or_remaining_placeholder_is_rejected():
    original, component = fixture()
    broken = copy.deepcopy(component)
    broken["tasks"].pop()
    with pytest.raises(ValueError, match="five"):
        replace_author_tasks(original, broken)
    component["tasks"][0]["operator_trace"] = ["openalex-author:None"]
    with pytest.raises(ValueError, match="placeholder"):
        replace_author_tasks(original, component)
