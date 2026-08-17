# Human Review Records

Human decisions are stored as append-only JSON Lines. Each record must contain:

- `decision_id`;
- `timestamp`;
- `reviewer_code`;
- `dataset_id`;
- `stage`;
- `item_id`;
- `issue_signature`;
- `detector_score`;
- `decision` (`accept`, `correct`, `reject`, or `abstain`);
- `original`;
- `correction`;
- `reason`;
- `review_seconds`;
- `feedback_memory_version`.

Corrections are never copied into locked-test gold labels before prediction.

