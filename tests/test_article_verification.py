from __future__ import annotations

from pathlib import Path

from src.citeweave.article_verification import verify_phenomenon_article
from src.citeweave.io import write_json


def test_rejects_unknown_provenance_ids(tmp_path: Path) -> None:
    cards = {
        "cards": [
            {
                "card_id": f"PC-{index}",
                "representative_works": [{"work_id": f"W-{index}"}],
            }
            for index in range(5)
        ]
    }
    cards_path = tmp_path / "cards.json"
    article_path = tmp_path / "article.md"
    write_json(cards_path, cards)
    sections = [
        "Abstract",
        "Introduction",
        "Methods",
        "Results",
        "Discussion",
        "Limitations",
        "Conclusion",
    ]
    body = " ".join(["Network evidence [CARD:PC-0] [WORK:W-0]."] * 300)
    article_path.write_text(
        "# Title\n\n"
        + "\n\n".join(f"## {name}\n\n{body}" for name in sections)
        + "\n\n## Provenance Map\n\nUnknown [CARD:PC-UNKNOWN].",
        encoding="utf-8",
    )
    result = verify_phenomenon_article(
        article_path=article_path,
        cards_path=cards_path,
    )
    assert result["passed"] is False
    assert result["unknown_card_ids"] == ["PC-UNKNOWN"]
