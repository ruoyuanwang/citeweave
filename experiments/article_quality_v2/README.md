# Article quality v2 expert evaluation

The confirmatory design is frozen in `expert_evaluation_protocol.yml`. Its hash must match
`expert_evaluation_protocol_freeze.json` before any rating file is analyzed.

The analysis input is a JSON object with these top-level fields:

```json
{
  "protocol_sha256": "7e8267bd91b69ac12b563aa1e5af47921cd67d521a4f17a28bb652fb1fd875a5",
  "holistic_ratings": [],
  "claim_ratings": [],
  "forced_pairwise_preferences": []
}
```

`holistic_ratings` use the eight frozen 1--5 dimensions and record `topic_id`, `article_id`,
`condition`, `evaluator_id`, `evaluator_role`, and positive server-timed
`evaluation_seconds`. `claim_ratings` use the frozen five strata and four nullable binary
labels. If `cannot_assess` is true, all four substantive labels must be null. A disagreement
between the two primary ratings requires an `adjudication: true` row from a third evaluator.
`forced_pairwise_preferences` name two distinct conditions and require the evaluator to select
one of them.

The analyzer rejects a panel unless it contains at least eight topics, exactly one article from
each of the three conditions per topic, three holistic raters per article, two domain experts per
topic, all within-topic condition pairs co-rated, exactly 20 stratified claims per article, two
independent primary ratings per claim, and two forced preferences per condition pair and topic.

The frozen same-evidence input is now material rather than nominal. The v4 source-pack manifest
contains eight passed topic packs. Each source pack binds five
verified Large-graph phenomena (multi-hop connection, bridge deletion, community-role contrast,
hub-removal resilience, and temporal structural shift), 15 distinct source records with nonempty
abstract excerpts, and one audited graph overview. This yields 40 registered phenomena, 120
source excerpts, and eight figures for the planned 24 articles.

Amendment 002 separates graph reasoning from image capability. The audited
`same_evidence_text_writer_inputs_v1` packs expose the identical writing brief, structured graph
phenomena, operator traces, source excerpts, and nonvisual metadata to all three writer
conditions, but contain no rendered-image path or bytes. After each draft is frozen, the packet
builder attaches the same immutable topic figure to every condition for evaluator access. The
production validator requires the writer-input hash, a no-image-access assertion, draft hash,
condition-specific request/delivery receipt, and posthoc-insertion receipt. Consequently this
article experiment tests graph-conditioned reasoning and workflow quality; VLM image
interpretation is a separate modality experiment.

`expert_packet_intake` is intentionally blocked until the 24 articles, their production records,
condition-blinded claim inventories, and a conflict-screened evaluator roster exist. The packet
builder enforces the production contract for each condition, normalizes article formatting,
replaces PH/REF provenance with topic-stable opaque evidence IDs, selects exactly four unique
claims from each of the five registered strata, and creates a seeded assignment design. Per topic,
one evaluator reads all three articles and three evaluators read one distinct condition pair. Thus
every article has three holistic raters and every condition pair has two co-raters without exposing
condition labels in evaluator packets.

Prepare or refresh the intake templates and readiness report:

```powershell
.\.venv\Scripts\python.exe scripts/prepare_article_expert_evaluation.py `
  --writer-pack-manifest experiments/article_quality_v2/same_evidence_text_writer_inputs_v1/manifest.json `
  --output-dir experiments/article_quality_v2/expert_packet_intake
```

After the real article paths, production records, claim inventories, and expert roster are frozen,
build the blinded packets with `scripts/build_article_expert_packets.py`. A blocked readiness file
is a truthful recruitment/data-collection state, not an experimental result.

Quality-control amendment 004 is validated by the packet-building CLI before any evaluator packet
is released. It rejects a topic when the longest article is more than 1.10 times the shortest and
rejects a systematic condition-length imbalance when the largest condition mean is more than 1.05
times the smallest. The private packet manifest records evidence, candidate-claim, and duplicate-
claim density without exposing condition labels or diagnostics to evaluators. After ratings are
locked, report the descriptive word-normalized sensitivity metrics with:

```powershell
.\.venv\Scripts\python.exe scripts/analyze_article_quality_controls.py `
  --packet-manifest <expert-packet-manifest.json> `
  --ratings <frozen-rating-export.json> `
  --output <length-normalized-quality.json>
```

These density summaries do not add a hypothesis test and cannot replace or rescue the exact
topic-cluster A1/A2/A3 analysis.

Run:

```powershell
.\.venv\Scripts\python.exe scripts/analyze_article_expert_evaluation.py `
  --protocol experiments/article_quality_v2/expert_evaluation_protocol.yml `
  --freeze experiments/article_quality_v2/expert_evaluation_protocol_freeze.json `
  --input <frozen-rating-export.json> `
  --output <analysis.json>
```

Analysis amendment 003 was frozen before any article or expert outcome. It makes the eight topics
the actual inferential units: claim disagreements resolve through exactly one independent
adjudicator, holistic contrasts pair conditions within evaluator before topic aggregation, and
A1/A2/A3 use fully enumerated one-sided topic sign-flips with Holm correction. A1 reports strict
cannot-assess coding plus condition-adverse and condition-favorable extreme-case sensitivity.
The analyzer permits "exceeds humans" language only when Holm-adjusted research-utility
noninferiority and traceability superiority both pass and the prospective domain-specificity
non-regression guard also passes. Cumulative-link and mixed-logistic fits are diagnostic
sensitivity analyses; they cannot manufacture more than eight independent clusters or overturn
the exact tests.

Machine generation and real claim review are now separately frozen. The 16-cell generation plan
uses one provider response per topic-condition, never resamples a quality failure, and gives both
machine writers identical text-only graph and source evidence. CiteWeave receives only a
deterministic planning blueprint derived from that shared evidence. Its passing output is a
pre-review draft, not a final article.

`article_claim_review_protocol.yml` registers 160 Results/Discussion claims (20 per topic, with at
least two for every one of five graph phenomena) and 320 primary reviews. Two conflict-free domain
reviewers judge support, calibration, alternatives, and evidence sufficiency. Exact disagreements
alone go to one independent third reviewer. The resolved decisions are compiled into an
`evidence -> claim -> paragraph` graph: invalid upstream dependencies propagate to every affected
paragraph, while controlled revision can alter only those frozen paragraph spans and cannot add
unregistered PH/REF evidence. This makes feedback an executable, reversible change set rather
than prompt history.

The current readiness file is intentionally blocked on eight pending CiteWeave drafts and a real
reviewer roster. Once generation is terminal, the deferred supervisor refreshes readiness and
builds the 160 packets only if every topic has three qualified, conflict-free real reviewers.

After real reviewers are entered in `article_claim_review_v1/reviewer_roster.json`, refresh the
outcome-blind readiness audit with:

```powershell
.\.venv\Scripts\python.exe scripts/prepare_article_claim_review.py `
  --machine-plan experiments/article_quality_v2/machine_generation_v1/plan.json `
  --roster experiments/article_quality_v2/article_claim_review_v1/reviewer_roster.json `
  --output-root experiments/article_quality_v2/article_claim_review_v1 `
  --protocol experiments/article_quality_v2/article_claim_review_protocol.yml `
  --protocol-freeze experiments/article_quality_v2/article_claim_review_protocol_freeze.json `
  --audit `
  --readiness-output experiments/article_quality_v2/article_claim_review_v1/readiness.json
```

The completed roster must be hash-frozen before removing `--audit` to build packets. Reviewer
access tokens are generated separately and only their SHA-256 digests are stored. After both
primary returns are frozen, `scripts/finalize_article_claim_review.py` prepares disagreement-only
adjudication, resolves decisions, compiles the dependency worklist, and applies hash-bound
paragraph replacements. Its global `--protocol`, `--protocol-freeze`, and `--machine-plan`
arguments are mandatory for every subcommand.

Integrity amendment 001 supersedes only the listed implementation hashes, leaving the base
protocol unchanged. It corrects an audited omission in the initial worklist: indirect paragraphs,
including paragraphs not selected for claim review, now receive upstream invalidation through
explicit PH/REF and phenomenon dependencies. It also fixes identifier digits counted as numerical
risk, validates frozen primary/adjudication provenance, prevents identity overrides, preserves
original newline bytes, and copies unchanged articles verbatim. Uncited semantic dependencies are
not automatically covered. Both preparation and finalization CLIs validate the amendment by default.

Packet preparation now also freezes `routing_plan.json` before any review outcomes. Use
`run_article_review_experiment.py routing-analysis` after validated review resolution to compare
same-claim-count budgets; time includes both primary reviews plus adjudication. These are offline
defect-capture curves, not causal labor savings or counterfactual article quality.

After controlled revisions exist, `run_article_review_experiment.py revision-packets` creates
independently evaluated, version-blind pre/post paragraph packets. It requires an independent,
domain-qualified, conflict-screened roster and excludes original claim-review participants. These
packets use the existing article review UI and the same adjudication/resolution commands.
`revision-analysis` requires eight topics, resolves each version once, computes strict four-check
adequacy change within topic, enumerates 256 topic sign flips, and reports descriptive topic-bootstrap
uncertainty and harm/abstention counts. Similar phrasing may reveal pairing; perfect blinding and
randomized causal attribution are not claimed.
