# Experiment Data Versioning Policy

CiteWeave keeps implementation code and compact scientific records in Git while
storing bulky or regenerable data outside normal Git history.

## Versioned in Git

- experiment protocols and prospective amendments;
- freeze receipts and SHA-256 manifests;
- compact readiness, audit, verification, and analysis summaries;
- schemas, small fixtures, and tests needed to reproduce the implementation;
- an entry in `STATUS.md` that states whether an artifact is formal,
  development-only, blocked, failed, or superseded.

## Not versioned in normal Git

- canonical Parquet tables, embeddings, indexes, model files, and download parts;
- generated benchmark payloads, reviewer packets, execution workspaces, and logs;
- packaged release candidates and rendered document workspaces;
- access tokens, reviewer identities, private responses, or local absolute paths.

Large immutable source snapshots may be published through an external archive or
an explicit Git LFS rule. Their portable manifest must record the archive
location, content hash, schema, license, and reproduction command. Existing
formal snapshots already tracked by Git LFS remain governed by their registered
manifests.

Frozen artifacts are never silently rewritten to remove local paths. A portable
export must instead record the original artifact hash, replace machine-specific
locations with repository-relative identifiers, and receive its own hash and
status entry.
