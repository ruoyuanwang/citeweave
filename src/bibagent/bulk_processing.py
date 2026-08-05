from __future__ import annotations

import json
import os
import platform
import shutil
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from itertools import islice
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from . import __version__
from .bulk_acquisition import _pid_is_running, iter_staged_records
from .exceptions import ProcessingError
from .io import read_json, sha256_file, write_json, write_parquet
from .models import ProjectConfig, ProjectPaths
from .transform import (
    CANONICAL_COLUMNS,
    Canonicalizer,
    normalize_title,
    stable_id,
)

PART_TABLE_COLUMNS = {
    **CANONICAL_COLUMNS,
    "duplicates": ["kept_work_id", "removed_work_id", "rule"],
}
WORK_HELPER_COLUMNS = [
    "_dedup_key",
    "_completeness_score",
    "_record_order",
]
EXCLUSION_COLUMNS = [
    "work_id",
    "source",
    "source_record_id",
    "source_record_hash",
    "publication_year",
    "rule",
    "detail",
]
STRING_COLUMNS = {
    "works": set(CANONICAL_COLUMNS["works"])
    - {"year", "cited_by_count", "reference_count", "is_retracted"},
    "authors": set(CANONICAL_COLUMNS["authors"]),
    "institutions": set(CANONICAL_COLUMNS["institutions"]),
    "authorships": {"work_id", "author_id", "institution_id"},
    "sources": set(CANONICAL_COLUMNS["sources"]),
    "keywords": {"work_id", "keyword", "keyword_type"},
    "topics": {"work_id", "topic_id", "topic", "field"},
    "references": set(CANONICAL_COLUMNS["references"]) - {"cited_year"},
    "provenance": set(CANONICAL_COLUMNS["provenance"]),
    "duplicates": {"kept_work_id", "removed_work_id", "rule"},
}
INTEGER_COLUMNS = {
    "works": {"year", "cited_by_count", "reference_count"},
    "authorships": {"position"},
    "references": {"cited_year"},
}
FLOAT_COLUMNS = {"keywords": {"score"}, "topics": {"score"}}
BOOLEAN_COLUMNS = {"works": {"is_retracted"}, "authorships": {"is_corresponding"}}
DERIVED_TERM_STOPWORDS = {
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "also",
    "am",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "doing",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "has",
    "have",
    "having",
    "he",
    "her",
    "here",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "itself",
    "just",
    "may",
    "me",
    "might",
    "more",
    "most",
    "my",
    "myself",
    "no",
    "nor",
    "not",
    "now",
    "of",
    "off",
    "on",
    "once",
    "only",
    "or",
    "other",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "same",
    "she",
    "should",
    "so",
    "some",
    "such",
    "than",
    "that",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "very",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whom",
    "why",
    "will",
    "with",
    "would",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
    # Generic scholarly prose is poor science-mapping signal.
    "aim",
    "aimed",
    "article",
    "background",
    "based",
    "conclusion",
    "conclusions",
    "data",
    "findings",
    "included",
    "including",
    "method",
    "methods",
    "objective",
    "paper",
    "purpose",
    "research",
    "result",
    "results",
    "review",
    "study",
    "studies",
    "using",
}


def _utc_iso() -> str:
    return datetime.now(UTC).isoformat()


def _run_metadata(
    config: ProjectConfig,
    *,
    chunk_size: int,
    candidate_pool: int,
    keep_partitions: bool,
) -> dict[str, Any]:
    return {
        "pipeline_version": 1,
        "cleaning_rules_version": 2,
        "protocol": {
            "year_from": config.protocol.year_from,
            "year_to": config.protocol.year_to,
            "source": config.protocol.source.value,
        },
        "parameters": {
            "chunk_size": chunk_size,
            "duckdb_memory_limit": config.processing.duckdb_memory_limit,
            "candidate_pool_size": candidate_pool,
            "edge_row_limit": config.processing.edge_row_limit,
            "keep_partitions": keep_partitions,
        },
        "runtime": {
            "bibagent": __version__,
            "python": platform.python_version(),
            "duckdb": duckdb.__version__,
            "pandas": pd.__version__,
        },
    }


def _sql_string(value: str | Path) -> str:
    return "'" + str(value).replace("\\", "/").replace("'", "''") + "'"


@contextmanager
def processing_lock(paths: ProjectPaths) -> Iterator[None]:
    """Protect checkpoint and Parquet partitions from concurrent writers."""
    lock_path = paths.audit / "processing.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if lock_path.exists():
        try:
            pid = int(json.loads(lock_path.read_text(encoding="utf-8")).get("pid", 0))
        except (OSError, ValueError, json.JSONDecodeError):
            pid = 0
        if _pid_is_running(pid):
            raise ProcessingError(f"Metadata processing is already running with PID {pid}.")
        lock_path.unlink(missing_ok=True)
    payload = json.dumps({"pid": os.getpid(), "started_at": _utc_iso()}).encode("utf-8")
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise ProcessingError("Another process acquired the metadata processing lock.") from exc
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        yield
    finally:
        try:
            current = json.loads(lock_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            current = {}
        if current.get("pid") == os.getpid():
            lock_path.unlink(missing_ok=True)


def _input_fingerprint(paths: ProjectPaths, input_path: Path) -> str:
    harvest_path = paths.audit / "harvest_manifest.json"
    if harvest_path.exists():
        harvest = read_json(harvest_path)
        staged = harvest.get("staged_path")
        if staged and (paths.root / staged).resolve() == input_path.resolve():
            fingerprint = harvest.get("staged_sha256")
            if fingerprint:
                return str(fingerprint)
    return sha256_file(input_path)


def _empty_frame(table: str) -> pd.DataFrame:
    return pd.DataFrame(columns=PART_TABLE_COLUMNS[table])


def _coerce_frame(table: str, frame: pd.DataFrame) -> pd.DataFrame:
    """Keep schemas stable even when a partition contains only null values."""
    columns = list(PART_TABLE_COLUMNS[table])
    if table == "works":
        columns += WORK_HELPER_COLUMNS
    result = frame.reindex(columns=columns).copy()
    for column in STRING_COLUMNS.get(table, set()):
        result[column] = (
            result[column]
            .map(lambda value: pd.NA if pd.isna(value) else str(value))
            .astype("string")
        )
    for column in INTEGER_COLUMNS.get(table, set()):
        result[column] = pd.to_numeric(result[column], errors="coerce").astype("Int64")
    for column in FLOAT_COLUMNS.get(table, set()):
        result[column] = pd.to_numeric(result[column], errors="coerce").astype("Float64")
    for column in BOOLEAN_COLUMNS.get(table, set()):
        result[column] = result[column].astype("boolean")
    if table == "works":
        result["_completeness_score"] = (
            pd.to_numeric(result["_completeness_score"], errors="coerce").fillna(0).astype("Int16")
        )
        result["_record_order"] = (
            pd.to_numeric(result["_record_order"], errors="coerce").fillna(0).astype("Int64")
        )
        result["_dedup_key"] = (
            result["_dedup_key"]
            .map(lambda value: pd.NA if pd.isna(value) else str(value))
            .astype("string")
        )
    return result


def _prepare_works(frame: pd.DataFrame, record_offset: int) -> pd.DataFrame:
    frame = frame.copy()
    if frame.empty:
        for column in WORK_HELPER_COLUMNS:
            frame[column] = pd.Series(dtype="object")
        return frame
    normalized = frame["title"].map(normalize_title)
    frame["_dedup_key"] = [
        (
            f"doi:{doi}"
            if pd.notna(doi) and str(doi).strip()
            else (
                stable_id("dedup", title, year if pd.notna(year) else "")
                if len(title) >= 20
                else f"record:{record_hash}"
            )
        )
        for doi, title, year, record_hash in zip(
            frame["doi"],
            normalized,
            frame["year"],
            frame["source_record_hash"],
            strict=True,
        )
    ]
    completeness_fields = [
        "doi",
        "title",
        "abstract",
        "year",
        "publication_date",
        "document_type",
        "language",
        "source_id",
        "publisher",
        "cited_by_count",
        "reference_count",
    ]
    frame["_completeness_score"] = frame[completeness_fields].notna().sum(axis=1)
    frame["_record_order"] = range(record_offset, record_offset + len(frame))
    return frame


def _write_batch(
    parts_root: Path,
    source: str,
    records: list[dict[str, Any]],
    batch_index: int,
    record_offset: int,
) -> dict[str, int]:
    tables = Canonicalizer(source).canonicalize(records)
    counts: dict[str, int] = {}
    for table, frame in tables.as_dict().items():
        if table == "works":
            frame = _prepare_works(frame, record_offset)
        if frame is None:
            frame = _empty_frame(table)
        frame = _coerce_frame(table, frame)
        destination = parts_root / table / f"part-{batch_index:06d}.parquet"
        write_parquet(destination, frame)
        counts[table] = len(frame)
    return counts


def _create_part_views(connection: duckdb.DuckDBPyConnection, parts_root: Path) -> None:
    for table in PART_TABLE_COLUMNS:
        pattern = parts_root / table / "part-*.parquet"
        connection.execute(
            f"""
            CREATE OR REPLACE TEMP VIEW p_{table} AS
            SELECT * FROM read_parquet({_sql_string(pattern)}, union_by_name=true)
            """
        )


def _copy_query(
    connection: duckdb.DuckDBPyConnection,
    query: str,
    destination: Path,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    connection.execute(
        f"""
        COPY ({query}) TO {_sql_string(temporary)}
        (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
        """
    )
    os.replace(temporary, destination)


def _build_work_mapping(
    connection: duckdb.DuckDBPyConnection,
    *,
    year_from: int,
    year_to: int,
) -> None:
    connection.execute(
        """
        CREATE OR REPLACE TEMP TABLE ranked_works AS
        SELECT
            *,
            first_value(work_id) OVER (
                PARTITION BY _dedup_key
                ORDER BY _completeness_score DESC,
                         source_record_hash ASC NULLS LAST,
                         work_id ASC,
                         _record_order ASC
            ) AS _kept_work_id,
            row_number() OVER (
                PARTITION BY _dedup_key
                ORDER BY _completeness_score DESC,
                         source_record_hash ASC NULLS LAST,
                         work_id ASC,
                         _record_order ASC
            ) AS _dedup_rank
        FROM p_works
        """
    )
    connection.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE included_work_ids AS
        SELECT work_id
        FROM ranked_works
        WHERE _dedup_rank = 1
          AND year BETWEEN {year_from} AND {year_to}
        """
    )
    connection.execute(
        """
        CREATE OR REPLACE TEMP TABLE global_work_map AS
        SELECT work_id AS old_work_id, min(_kept_work_id) AS kept_work_id
        FROM ranked_works
        GROUP BY work_id
        """
    )
    connection.execute(
        """
        CREATE OR REPLACE TEMP TABLE work_id_map AS
        WITH mappings AS (
            SELECT old_work_id, kept_work_id FROM global_work_map
            UNION ALL
            SELECT
                local.removed_work_id AS old_work_id,
                coalesce(global.kept_work_id, local.kept_work_id) AS kept_work_id
            FROM p_duplicates AS local
            LEFT JOIN global_work_map AS global
              ON local.kept_work_id = global.old_work_id
        )
        SELECT old_work_id, min(kept_work_id) AS kept_work_id
        FROM mappings
        WHERE old_work_id IS NOT NULL AND kept_work_id IS NOT NULL
        GROUP BY old_work_id
        """
    )


def _canonical_queries() -> dict[str, str]:
    return {
        "works": f"""
            SELECT {", ".join(CANONICAL_COLUMNS["works"])}
            FROM ranked_works
            WHERE _dedup_rank = 1
              AND work_id IN (SELECT work_id FROM included_work_ids)
            ORDER BY work_id
        """,
        "authors": """
            SELECT author_id, name, given_name, family_name, orcid
            FROM p_authors
            WHERE author_id IS NOT NULL
              AND author_id IN (
                  SELECT DISTINCT rel.author_id
                  FROM p_authorships AS rel
                  LEFT JOIN work_id_map AS mapping
                    ON rel.work_id = mapping.old_work_id
                  JOIN included_work_ids AS included
                    ON coalesce(mapping.kept_work_id, rel.work_id) = included.work_id
              )
            QUALIFY row_number() OVER (
                PARTITION BY author_id
                ORDER BY (
                    (name IS NOT NULL)::INT +
                    (given_name IS NOT NULL)::INT +
                    (family_name IS NOT NULL)::INT +
                    (orcid IS NOT NULL)::INT
                ) DESC, name ASC NULLS LAST
            ) = 1
            ORDER BY author_id
        """,
        "institutions": """
            SELECT institution_id, name, ror, country_code
            FROM p_institutions
            WHERE institution_id IS NOT NULL
              AND institution_id IN (
                  SELECT DISTINCT rel.institution_id
                  FROM p_authorships AS rel
                  LEFT JOIN work_id_map AS mapping
                    ON rel.work_id = mapping.old_work_id
                  JOIN included_work_ids AS included
                    ON coalesce(mapping.kept_work_id, rel.work_id) = included.work_id
                  WHERE rel.institution_id IS NOT NULL
              )
            QUALIFY row_number() OVER (
                PARTITION BY institution_id
                ORDER BY (
                    (name IS NOT NULL)::INT +
                    (ror IS NOT NULL)::INT +
                    (country_code IS NOT NULL)::INT
                ) DESC, name ASC NULLS LAST
            ) = 1
            ORDER BY institution_id
        """,
        "authorships": """
            SELECT DISTINCT
                coalesce(mapping.kept_work_id, rel.work_id) AS work_id,
                rel.author_id,
                rel.institution_id,
                rel.position,
                rel.is_corresponding
            FROM p_authorships AS rel
            LEFT JOIN work_id_map AS mapping ON rel.work_id = mapping.old_work_id
            JOIN included_work_ids AS included
              ON coalesce(mapping.kept_work_id, rel.work_id) = included.work_id
            WHERE rel.author_id IS NOT NULL
            ORDER BY work_id, position, author_id, institution_id
        """,
        "sources": """
            SELECT source_id, name, issn, publisher, source_type
            FROM p_sources
            WHERE source_id IS NOT NULL
              AND source_id IN (
                  SELECT DISTINCT works.source_id
                  FROM ranked_works AS works
                  JOIN included_work_ids AS included USING (work_id)
                  WHERE works._dedup_rank = 1 AND works.source_id IS NOT NULL
              )
            QUALIFY row_number() OVER (
                PARTITION BY source_id
                ORDER BY (
                    (name IS NOT NULL)::INT +
                    (issn IS NOT NULL)::INT +
                    (publisher IS NOT NULL)::INT +
                    (source_type IS NOT NULL)::INT
                ) DESC, name ASC NULLS LAST
            ) = 1
            ORDER BY source_id
        """,
        "keywords": """
            SELECT DISTINCT
                coalesce(mapping.kept_work_id, rel.work_id) AS work_id,
                rel.keyword,
                rel.keyword_type,
                rel.score
            FROM p_keywords AS rel
            LEFT JOIN work_id_map AS mapping ON rel.work_id = mapping.old_work_id
            JOIN included_work_ids AS included
              ON coalesce(mapping.kept_work_id, rel.work_id) = included.work_id
            WHERE rel.keyword IS NOT NULL AND trim(rel.keyword) <> ''
            ORDER BY work_id, keyword, keyword_type
        """,
        "topics": """
            SELECT DISTINCT
                coalesce(mapping.kept_work_id, rel.work_id) AS work_id,
                rel.topic_id,
                rel.topic,
                rel.score,
                rel.field
            FROM p_topics AS rel
            LEFT JOIN work_id_map AS mapping ON rel.work_id = mapping.old_work_id
            JOIN included_work_ids AS included
              ON coalesce(mapping.kept_work_id, rel.work_id) = included.work_id
            WHERE rel.topic_id IS NOT NULL OR rel.topic IS NOT NULL
            ORDER BY work_id, topic_id, topic
        """,
        "references": """
            SELECT
                citing_work_id,
                cited_work_id,
                cited_doi,
                cited_title,
                cited_author,
                cited_year,
                source
            FROM (
                SELECT
                    coalesce(citing_map.kept_work_id, rel.citing_work_id)
                        AS citing_work_id,
                    coalesce(cited_map.kept_work_id, rel.cited_work_id)
                        AS cited_work_id,
                    rel.cited_doi,
                    rel.cited_title,
                    rel.cited_author,
                    rel.cited_year,
                    rel.source,
                    row_number() OVER (
                        PARTITION BY
                            coalesce(citing_map.kept_work_id, rel.citing_work_id),
                            coalesce(cited_map.kept_work_id, rel.cited_work_id)
                        ORDER BY (
                            (rel.cited_doi IS NOT NULL)::INT +
                            (rel.cited_title IS NOT NULL)::INT +
                            (rel.cited_author IS NOT NULL)::INT +
                            (rel.cited_year IS NOT NULL)::INT
                        ) DESC
                    ) AS rank
                FROM p_references AS rel
                LEFT JOIN work_id_map AS citing_map
                  ON rel.citing_work_id = citing_map.old_work_id
                LEFT JOIN work_id_map AS cited_map
                  ON rel.cited_work_id = cited_map.old_work_id
                JOIN included_work_ids AS included
                  ON coalesce(citing_map.kept_work_id, rel.citing_work_id)
                   = included.work_id
                WHERE rel.cited_work_id IS NOT NULL
            )
            WHERE rank = 1
            ORDER BY citing_work_id, cited_work_id
        """,
        "provenance": """
            SELECT DISTINCT
                coalesce(mapping.kept_work_id, rel.work_id) AS work_id,
                rel.source,
                rel.source_record_id,
                rel.source_record_hash
            FROM p_provenance AS rel
            LEFT JOIN work_id_map AS mapping ON rel.work_id = mapping.old_work_id
            JOIN included_work_ids AS included
              ON coalesce(mapping.kept_work_id, rel.work_id) = included.work_id
            ORDER BY work_id, source, source_record_id, source_record_hash
        """,
        "duplicates": """
            SELECT DISTINCT kept_work_id, removed_work_id, rule
            FROM (
                SELECT
                    _kept_work_id AS kept_work_id,
                    work_id AS removed_work_id,
                    'exact_doi_or_normalized_title_year' AS rule
                FROM ranked_works
                WHERE _dedup_rank > 1
                UNION ALL
                SELECT
                    coalesce(mapping.kept_work_id, local.kept_work_id) AS kept_work_id,
                    local.removed_work_id,
                    local.rule
                FROM p_duplicates AS local
                LEFT JOIN work_id_map AS mapping
                  ON local.kept_work_id = mapping.old_work_id
            )
            ORDER BY kept_work_id, removed_work_id
        """,
    }


def _exclusion_query(year_from: int, year_to: int) -> str:
    return f"""
        SELECT
            coalesce(mapping.kept_work_id, provenance.work_id) AS work_id,
            provenance.source,
            provenance.source_record_id,
            provenance.source_record_hash,
            winner.year AS publication_year,
            CASE
                WHEN winner.year IS NULL THEN 'missing_publication_year'
                ELSE 'outside_protocol_year_range'
            END AS rule,
            CASE
                WHEN winner.year IS NULL
                    THEN 'Protocol requires a publication year between '
                         || {year_from} || ' and ' || {year_to}
                ELSE 'Canonical publication year ' || winner.year
                     || ' is outside protocol range '
                     || {year_from} || '-' || {year_to}
            END AS detail
        FROM p_provenance AS provenance
        LEFT JOIN work_id_map AS mapping
          ON provenance.work_id = mapping.old_work_id
        JOIN ranked_works AS winner
          ON coalesce(mapping.kept_work_id, provenance.work_id) = winner.work_id
         AND winner._dedup_rank = 1
        WHERE winner.work_id NOT IN (SELECT work_id FROM included_work_ids)
        ORDER BY work_id, source, source_record_id, source_record_hash
    """


def _register_canonical_views(connection: duckdb.DuckDBPyConnection, canonical_root: Path) -> None:
    for table in PART_TABLE_COLUMNS:
        path = canonical_root / f"{table}.parquet"
        connection.execute(
            f"""
            CREATE OR REPLACE TEMP VIEW c_{table} AS
            SELECT * FROM read_parquet({_sql_string(path)})
            """
        )


def _derive_keywords_if_sparse(
    connection: duckdb.DuckDBPyConnection,
    canonical_root: Path,
    *,
    top_per_document: int = 5,
    maximum_vocabulary: int = 5_000,
) -> dict[str, Any]:
    """Derive disclosed full-corpus TF-IDF terms without a corpus-sized Python matrix."""
    document_count, covered = connection.execute(
        """
        SELECT
            (SELECT count(*) FROM c_works),
            (SELECT count(DISTINCT work_id) FROM c_keywords)
        """
    ).fetchone()
    coverage = covered / document_count if document_count else 0.0
    if document_count == 0 or coverage >= 0.5:
        return {
            "applied": False,
            "reason": "source_keyword_coverage_sufficient" if document_count else "empty_corpus",
            "source_coverage": round(coverage, 6),
            "derived_rows": 0,
        }

    stopword_values = ", ".join(f"({_sql_string(word)})" for word in sorted(DERIVED_TERM_STOPWORDS))
    connection.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE derived_stopwords(term VARCHAR);
        INSERT INTO derived_stopwords VALUES {stopword_values}
        """
    )
    minimum_df = 3 if document_count >= 100 else 2
    maximum_df = max(minimum_df, int(document_count * 0.60))
    connection.execute(
        """
        CREATE OR REPLACE TEMP TABLE derived_term_tf AS
        WITH documents AS (
            SELECT
                work_id,
                regexp_split_to_array(
                    lower(coalesce(title, '') || ' ' || coalesce(abstract, '')),
                    '[^a-z0-9]+'
                ) AS tokens
            FROM c_works
        ), unigrams AS (
            SELECT work_id, token AS term
            FROM documents, unnest(tokens) AS values(token)
            WHERE length(token) >= 3
              AND token NOT IN (SELECT term FROM derived_stopwords)
              AND NOT regexp_matches(token, '^[0-9]+$')
        ), bigrams AS (
            SELECT
                work_id,
                tokens[position] || ' ' || tokens[position + 1] AS term
            FROM documents, unnest(range(1, len(tokens))) AS values(position)
            WHERE length(tokens[position]) >= 3
              AND length(tokens[position + 1]) >= 3
              AND tokens[position] NOT IN (SELECT term FROM derived_stopwords)
              AND tokens[position + 1] NOT IN (SELECT term FROM derived_stopwords)
              AND NOT regexp_matches(tokens[position], '^[0-9]+$')
              AND NOT regexp_matches(tokens[position + 1], '^[0-9]+$')
        ), terms AS (
            SELECT * FROM unigrams
            UNION ALL
            SELECT * FROM bigrams
        )
        SELECT work_id, term, count(*)::INTEGER AS term_frequency
        FROM terms
        WHERE length(term) BETWEEN 3 AND 80
        GROUP BY work_id, term
        """
    )
    connection.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE derived_candidates AS
        SELECT term, count(*)::BIGINT AS document_frequency
        FROM derived_term_tf
        GROUP BY term
        HAVING document_frequency BETWEEN {minimum_df} AND {maximum_df}
        ORDER BY document_frequency DESC, term
        LIMIT {maximum_vocabulary}
        """
    )
    connection.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE derived_keywords AS
        SELECT work_id,
               term AS keyword,
               'derived_tfidf_full_corpus' AS keyword_type,
               score
        FROM (
            SELECT
                term_tf.work_id,
                term_tf.term,
                (
                    (1.0 + ln(term_tf.term_frequency)) *
                    (ln(({document_count} + 1.0) / (candidate.document_frequency + 1.0)) + 1.0)
                ) AS score,
                row_number() OVER (
                    PARTITION BY term_tf.work_id
                    ORDER BY score DESC, term_tf.term
                ) AS term_rank
            FROM derived_term_tf AS term_tf
            JOIN derived_candidates AS candidate USING (term)
        )
        WHERE term_rank <= {top_per_document}
        """
    )
    derived_rows, derived_documents, vocabulary = connection.execute(
        """
        SELECT
            (SELECT count(*) FROM derived_keywords),
            (SELECT count(DISTINCT work_id) FROM derived_keywords),
            (SELECT count(*) FROM derived_candidates)
        """
    ).fetchone()
    if derived_rows:
        destination = canonical_root / "keywords.parquet"
        _copy_query(
            connection,
            """
            SELECT work_id, keyword, keyword_type, score FROM c_keywords
            UNION ALL
            SELECT work_id, keyword, keyword_type, score FROM derived_keywords
            ORDER BY work_id, keyword, keyword_type
            """,
            destination,
        )
        connection.execute("DROP VIEW c_keywords")
        connection.execute(
            f"""
            CREATE TEMP VIEW c_keywords AS
            SELECT * FROM read_parquet({_sql_string(destination)})
            """
        )
    return {
        "applied": bool(derived_rows),
        "reason": "source_keyword_coverage_below_0.5",
        "source_coverage": round(coverage, 6),
        "minimum_document_frequency": minimum_df,
        "maximum_document_frequency": maximum_df,
        "maximum_vocabulary": maximum_vocabulary,
        "vocabulary": int(vocabulary),
        "top_per_document": top_per_document,
        "derived_rows": int(derived_rows),
        "derived_documents": int(derived_documents),
        "method": (
            "full-corpus unigram/bigram TF-IDF; no sampling; source and derived "
            "keywords remain distinguishable by keyword_type"
        ),
    }


def _visualization_queries(candidate_pool: int, edge_limit: int) -> dict[str, str]:
    return {
        "annual_output": """
            SELECT
                year,
                count(*) AS documents,
                sum(coalesce(cited_by_count, 0)) AS citations,
                avg(coalesce(cited_by_count, 0)) AS mean_citations
            FROM c_works
            WHERE year IS NOT NULL
            GROUP BY year
            ORDER BY year
        """,
        "document_types": """
            SELECT coalesce(document_type, 'Unknown') AS document_type,
                   count(*) AS documents
            FROM c_works
            GROUP BY document_type
            ORDER BY documents DESC, document_type
        """,
        "languages": """
            SELECT coalesce(language, 'Unknown') AS language, count(*) AS documents
            FROM c_works
            GROUP BY language
            ORDER BY documents DESC, language
        """,
        "source_productivity": """
            SELECT
                works.source_id,
                sources.name AS source_name,
                sources.issn,
                count(*) AS documents,
                sum(coalesce(works.cited_by_count, 0)) AS citations
            FROM c_works AS works
            LEFT JOIN c_sources AS sources USING (source_id)
            GROUP BY works.source_id, sources.name, sources.issn
            ORDER BY documents DESC, citations DESC, source_id
        """,
        "author_productivity": """
            SELECT
                rel.author_id,
                authors.name AS author_name,
                authors.orcid,
                count(DISTINCT rel.work_id) AS documents,
                sum(coalesce(works.cited_by_count, 0)) AS citations
            FROM (SELECT DISTINCT work_id, author_id FROM c_authorships) AS rel
            JOIN c_works AS works USING (work_id)
            LEFT JOIN c_authors AS authors USING (author_id)
            GROUP BY rel.author_id, authors.name, authors.orcid
            ORDER BY documents DESC, citations DESC, rel.author_id
        """,
        "institution_productivity": """
            SELECT
                rel.institution_id,
                institutions.name AS institution_name,
                institutions.ror,
                institutions.country_code,
                count(DISTINCT rel.work_id) AS documents
            FROM (
                SELECT DISTINCT work_id, institution_id
                FROM c_authorships
                WHERE institution_id IS NOT NULL
            ) AS rel
            LEFT JOIN c_institutions AS institutions USING (institution_id)
            GROUP BY rel.institution_id, institutions.name,
                     institutions.ror, institutions.country_code
            ORDER BY documents DESC, rel.institution_id
        """,
        "keyword_occurrences": """
            SELECT keyword, keyword_type, count(DISTINCT work_id) AS occurrences
            FROM c_keywords
            GROUP BY keyword, keyword_type
            ORDER BY occurrences DESC, keyword
        """,
        "topic_occurrences": """
            SELECT topic_id, topic, field, count(DISTINCT work_id) AS occurrences
            FROM c_topics
            GROUP BY topic_id, topic, field
            ORDER BY occurrences DESC, topic
        """,
        "reference_impact": """
            SELECT
                cited_work_id,
                max(cited_doi) AS cited_doi,
                max(cited_title) AS cited_title,
                max(cited_author) AS cited_author,
                max(cited_year) AS cited_year,
                count(DISTINCT citing_work_id) AS local_citations
            FROM c_references
            GROUP BY cited_work_id
            ORDER BY local_citations DESC, cited_work_id
        """,
        "coauthor_edges": f"""
            WITH candidates AS (
                SELECT author_id
                FROM c_authorships
                GROUP BY author_id
                ORDER BY count(DISTINCT work_id) DESC, author_id
                LIMIT {candidate_pool}
            ), membership AS (
                SELECT DISTINCT work_id, author_id
                FROM c_authorships
                WHERE author_id IN (SELECT author_id FROM candidates)
            )
            SELECT left_rel.author_id AS source_id,
                   right_rel.author_id AS target_id,
                   count(*) AS weight
            FROM membership AS left_rel
            JOIN membership AS right_rel
              ON left_rel.work_id = right_rel.work_id
             AND left_rel.author_id < right_rel.author_id
            GROUP BY source_id, target_id
            ORDER BY weight DESC, source_id, target_id
            LIMIT {edge_limit}
        """,
        "institution_collaboration_edges": f"""
            WITH candidates AS (
                SELECT institution_id
                FROM c_authorships
                WHERE institution_id IS NOT NULL
                GROUP BY institution_id
                ORDER BY count(DISTINCT work_id) DESC, institution_id
                LIMIT {candidate_pool}
            ), membership AS (
                SELECT DISTINCT work_id, institution_id
                FROM c_authorships
                WHERE institution_id IN (SELECT institution_id FROM candidates)
            )
            SELECT left_rel.institution_id AS source_id,
                   right_rel.institution_id AS target_id,
                   count(*) AS weight
            FROM membership AS left_rel
            JOIN membership AS right_rel
              ON left_rel.work_id = right_rel.work_id
             AND left_rel.institution_id < right_rel.institution_id
            GROUP BY source_id, target_id
            ORDER BY weight DESC, source_id, target_id
            LIMIT {edge_limit}
        """,
        "keyword_cooccurrence_edges": f"""
            WITH candidates AS (
                SELECT keyword
                FROM c_keywords
                GROUP BY keyword
                ORDER BY count(DISTINCT work_id) DESC, keyword
                LIMIT {candidate_pool}
            ), membership AS (
                SELECT DISTINCT work_id, keyword
                FROM c_keywords
                WHERE keyword IN (SELECT keyword FROM candidates)
            )
            SELECT left_rel.keyword AS source_id,
                   right_rel.keyword AS target_id,
                   count(*) AS weight
            FROM membership AS left_rel
            JOIN membership AS right_rel
              ON left_rel.work_id = right_rel.work_id
             AND left_rel.keyword < right_rel.keyword
            GROUP BY source_id, target_id
            ORDER BY weight DESC, source_id, target_id
            LIMIT {edge_limit}
        """,
        "cocitation_edges": f"""
            WITH candidates AS (
                SELECT cited_work_id
                FROM c_references
                GROUP BY cited_work_id
                ORDER BY count(DISTINCT citing_work_id) DESC, cited_work_id
                LIMIT {candidate_pool}
            ), membership AS (
                SELECT DISTINCT citing_work_id, cited_work_id
                FROM c_references
                WHERE cited_work_id IN (SELECT cited_work_id FROM candidates)
            )
            SELECT left_rel.cited_work_id AS source_id,
                   right_rel.cited_work_id AS target_id,
                   count(*) AS weight
            FROM membership AS left_rel
            JOIN membership AS right_rel
              ON left_rel.citing_work_id = right_rel.citing_work_id
             AND left_rel.cited_work_id < right_rel.cited_work_id
            GROUP BY source_id, target_id
            ORDER BY weight DESC, source_id, target_id
            LIMIT {edge_limit}
        """,
        "direct_citation_edges": f"""
            SELECT refs.citing_work_id AS source_id,
                   refs.cited_work_id AS target_id,
                   count(*) AS weight
            FROM c_references AS refs
            JOIN c_works AS cited ON refs.cited_work_id = cited.work_id
            GROUP BY refs.citing_work_id, refs.cited_work_id
            ORDER BY weight DESC, source_id, target_id
            LIMIT {edge_limit}
        """,
    }


def _quality_report(
    connection: duckdb.DuckDBPyConnection,
    paths: ProjectPaths,
    manifest: dict[str, Any],
    *,
    year_from: int,
    year_to: int,
) -> dict[str, Any]:
    row_counts = {
        table: int(connection.execute(f"SELECT count(*) FROM c_{table}").fetchone()[0])
        for table in PART_TABLE_COLUMNS
    }
    work_count = row_counts["works"]
    coverage: dict[str, dict[str, float | int]] = {}
    for column in (
        "doi",
        "title",
        "abstract",
        "year",
        "publication_date",
        "document_type",
        "language",
        "source_id",
        "publisher",
        "cited_by_count",
        "reference_count",
    ):
        present = int(
            connection.execute(
                f"SELECT count(*) FROM c_works WHERE {column} IS NOT NULL"
            ).fetchone()[0]
        )
        coverage[column] = {
            "present": present,
            "ratio": round(present / work_count, 6) if work_count else 0.0,
        }
    orphan_queries = {
        "authorship_work": """
            SELECT count(*) FROM c_authorships rel
            LEFT JOIN c_works parent USING (work_id)
            WHERE parent.work_id IS NULL
        """,
        "authorship_author": """
            SELECT count(*) FROM c_authorships rel
            LEFT JOIN c_authors parent USING (author_id)
            WHERE parent.author_id IS NULL
        """,
        "authorship_institution": """
            SELECT count(*) FROM c_authorships rel
            LEFT JOIN c_institutions parent USING (institution_id)
            WHERE rel.institution_id IS NOT NULL AND parent.institution_id IS NULL
        """,
        "keyword_work": """
            SELECT count(*) FROM c_keywords rel
            LEFT JOIN c_works parent USING (work_id)
            WHERE parent.work_id IS NULL
        """,
        "topic_work": """
            SELECT count(*) FROM c_topics rel
            LEFT JOIN c_works parent USING (work_id)
            WHERE parent.work_id IS NULL
        """,
        "reference_citing_work": """
            SELECT count(*) FROM c_references rel
            LEFT JOIN c_works parent ON rel.citing_work_id = parent.work_id
            WHERE parent.work_id IS NULL
        """,
        "provenance_work": """
            SELECT count(*) FROM c_provenance rel
            LEFT JOIN c_works parent USING (work_id)
            WHERE parent.work_id IS NULL
        """,
        "work_source": """
            SELECT count(*) FROM c_works rel
            LEFT JOIN c_sources parent USING (source_id)
            WHERE rel.source_id IS NOT NULL AND parent.source_id IS NULL
        """,
    }
    orphans = {
        name: int(connection.execute(query).fetchone()[0]) for name, query in orphan_queries.items()
    }
    uniqueness = {
        table: {
            "rows": row_counts[table],
            "distinct_ids": int(
                connection.execute(
                    f"SELECT count(DISTINCT {identifier}) FROM c_{table}"
                ).fetchone()[0]
            ),
        }
        for table, identifier in {
            "works": "work_id",
            "authors": "author_id",
            "institutions": "institution_id",
            "sources": "source_id",
        }.items()
    }
    year_bounds = connection.execute("SELECT min(year), max(year) FROM c_works").fetchone()
    excluded_records = int(
        connection.execute("SELECT count(*) FROM excluded_records").fetchone()[0]
    )
    out_of_scope = int(
        connection.execute(
            f"""
            SELECT count(*) FROM c_works
            WHERE year IS NULL OR year NOT BETWEEN {year_from} AND {year_to}
            """
        ).fetchone()[0]
    )
    included_source_records = row_counts["provenance"]
    input_accounted = included_source_records + excluded_records
    passed = (
        work_count > 0
        and all(value == 0 for value in orphans.values())
        and all(item["rows"] == item["distinct_ids"] for item in uniqueness.values())
        and row_counts["provenance"] >= work_count
        and out_of_scope == 0
        and input_accounted == int(manifest["records_processed"])
    )
    return {
        "version": 1,
        "created_at": _utc_iso(),
        "passed": passed,
        "input_records": manifest["records_processed"],
        "canonical_records": work_count,
        "duplicates_removed": row_counts["duplicates"],
        "scope_filter": {
            "protocol_year_from": year_from,
            "protocol_year_to": year_to,
            "out_of_scope_in_canonical": out_of_scope,
            "excluded_source_records": excluded_records,
            "included_source_records": included_source_records,
            "input_records_accounted": input_accounted,
            "excluded_records_path": str(paths.quality / "excluded_records.parquet"),
        },
        "table_rows": row_counts,
        "field_coverage": coverage,
        "year_bounds": {"minimum": year_bounds[0], "maximum": year_bounds[1]},
        "foreign_key_orphans": orphans,
        "primary_key_uniqueness": uniqueness,
        "rules": {
            "doi": "lowercase DOI, resolver prefix removed, invalid DOI stored as null",
            "text": "HTML removed, Unicode NFKC normalized, whitespace collapsed",
            "deduplication": (
                "exact normalized DOI; otherwise exact normalized title of at least "
                "20 characters + year; short or missing titles are not auto-merged; "
                "winner selected by metadata completeness with deterministic tie-break"
            ),
            "author_identity": (
                "ORCID first; otherwise source identifier where available; Crossref "
                "fallback uses normalized name + primary normalized affiliation"
            ),
            "relationships": "all loser work identifiers remapped to the retained work",
            "year_scope": (
                "canonical publication year must fall inside the protocol range; "
                "other source records are retained in the exclusion ledger"
            ),
            "statistics": "all rows, no sampling",
            "networks": "full occurrence counts followed by bounded candidate-first expansion",
        },
        "paths": {
            "canonical": str(paths.canonical),
            "visualization": str(paths.canonical / "visualization"),
        },
    }


def _finalize(
    paths: ProjectPaths,
    manifest: dict[str, Any],
    *,
    memory_limit: str,
    candidate_pool: int,
    edge_limit: int,
    year_from: int,
    year_to: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    parts_root = paths.canonical / "_parts"
    temp_directory = paths.canonical / "_duckdb_tmp"
    temp_directory.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(":memory:")
    try:
        connection.execute(f"SET memory_limit={_sql_string(memory_limit)}")
        connection.execute(f"SET temp_directory={_sql_string(temp_directory)}")
        connection.execute("SET preserve_insertion_order=false")
        connection.execute("SET threads=4")
        _create_part_views(connection, parts_root)
        _build_work_mapping(connection, year_from=year_from, year_to=year_to)
        for table, query in _canonical_queries().items():
            _copy_query(connection, query, paths.canonical / f"{table}.parquet")
        exclusion_path = paths.quality / "excluded_records.parquet"
        _copy_query(
            connection,
            _exclusion_query(year_from, year_to),
            exclusion_path,
        )
        _register_canonical_views(connection, paths.canonical)
        connection.execute(
            f"""
            CREATE OR REPLACE TEMP VIEW excluded_records AS
            SELECT * FROM read_parquet({_sql_string(exclusion_path)})
            """
        )
        keyword_derivation = _derive_keywords_if_sparse(connection, paths.canonical)

        visualization_root = paths.canonical / "visualization"
        visualization_files: dict[str, str] = {}
        visualization_rows: dict[str, int] = {}
        for name, query in _visualization_queries(candidate_pool, edge_limit).items():
            destination = visualization_root / f"{name}.parquet"
            _copy_query(connection, query, destination)
            visualization_files[name] = str(destination.relative_to(paths.root))
            visualization_rows[name] = int(
                connection.execute(
                    f"SELECT count(*) FROM read_parquet({_sql_string(destination)})"
                ).fetchone()[0]
            )
        quality = _quality_report(
            connection,
            paths,
            manifest,
            year_from=year_from,
            year_to=year_to,
        )
        quality["keyword_derivation"] = keyword_derivation
    finally:
        connection.close()
        shutil.rmtree(temp_directory, ignore_errors=True)

    canonical_hashes = {
        table: sha256_file(paths.canonical / f"{table}.parquet") for table in PART_TABLE_COLUMNS
    }
    visualization_hashes = {
        name: sha256_file(paths.root / relative) for name, relative in visualization_files.items()
    }
    exclusion_hash = sha256_file(paths.quality / "excluded_records.parquet")
    schema = {
        "version": 1,
        "format": "Parquet",
        "processing_contract": manifest.get("run_contract"),
        "canonical_tables": PART_TABLE_COLUMNS,
        "primary_keys": {
            "works": ["work_id"],
            "authors": ["author_id"],
            "institutions": ["institution_id"],
            "sources": ["source_id"],
        },
        "foreign_keys": {
            "works.source_id": "sources.source_id",
            "authorships.work_id": "works.work_id",
            "authorships.author_id": "authors.author_id",
            "authorships.institution_id": "institutions.institution_id",
            "keywords.work_id": "works.work_id",
            "topics.work_id": "works.work_id",
            "references.citing_work_id": "works.work_id",
            "provenance.work_id": "works.work_id",
        },
        "visualization_tables": {
            name: {
                "path": visualization_files[name],
                "rows": visualization_rows[name],
                "sha256": visualization_hashes[name],
            }
            for name in visualization_files
        },
        "keyword_derivation": keyword_derivation,
        "exclusion_ledger": {
            "path": str((paths.quality / "excluded_records.parquet").relative_to(paths.root)),
            "columns": EXCLUSION_COLUMNS,
            "rows": quality["scope_filter"]["excluded_source_records"],
            "sha256": exclusion_hash,
        },
    }
    write_json(paths.canonical / "schema.json", schema)
    write_json(paths.quality / "processing_report.json", quality)
    outputs = {
        "canonical_sha256": canonical_hashes,
        "visualization_sha256": visualization_hashes,
        "visualization_rows": visualization_rows,
        "exclusion_sha256": exclusion_hash,
        "schema_path": str((paths.canonical / "schema.json").relative_to(paths.root)),
        "quality_path": str((paths.quality / "processing_report.json").relative_to(paths.root)),
    }
    return quality, outputs


def process_large_metadata(
    root: Path,
    config: ProjectConfig,
    *,
    input_path: Path | None = None,
    resume: bool = True,
    chunk_size: int | None = None,
    keep_partitions: bool | None = None,
    refinalize: bool = False,
    batch_budget: int | None = None,
) -> dict[str, Any]:
    """Normalize, deduplicate and materialize a corpus with bounded Python memory."""
    if batch_budget is not None and batch_budget < 1:
        raise ValueError("batch_budget must be positive")
    paths = ProjectPaths(root)
    paths.create()
    input_path = (
        input_path.resolve()
        if input_path is not None
        else (
            paths.staged / "source_records.jsonl.gz"
            if (paths.staged / "source_records.jsonl.gz").exists()
            else paths.staged / "source_records.jsonl"
        )
    )
    if not input_path.exists():
        raise ProcessingError(f"Staged metadata is missing: {input_path}")
    policy = config.processing
    actual_chunk_size = chunk_size or policy.chunk_size
    retain_parts = policy.keep_partitions if keep_partitions is None else keep_partitions
    candidate_pool = policy.candidate_pool_size or max(config.visualization_max_nodes * 8, 400)
    manifest_path = paths.audit / "processing_manifest.json"
    parts_root = paths.canonical / "_parts"
    fingerprint = _input_fingerprint(paths, input_path)
    run_contract = _run_metadata(
        config,
        chunk_size=actual_chunk_size,
        candidate_pool=candidate_pool,
        keep_partitions=retain_parts,
    )
    started = time.perf_counter()

    with processing_lock(paths):
        if resume and manifest_path.exists():
            manifest = read_json(manifest_path)
            if manifest.get("input_sha256") != fingerprint:
                raise ProcessingError(
                    "The staged input changed after the processing checkpoint was created."
                )
            if manifest.get("source") != config.protocol.source.value:
                raise ProcessingError("Checkpoint source does not match project.yml.")
            if int(manifest.get("chunk_size", 0)) != actual_chunk_size:
                raise ProcessingError(
                    "Resume requires the same chunk size; use --no-resume to restart."
                )
            existing_contract = manifest.get("run_contract")
            if existing_contract:
                if (
                    existing_contract.get("cleaning_rules_version")
                    != run_contract["cleaning_rules_version"]
                ):
                    raise ProcessingError(
                        "Cleaning rules changed after partitions were written; "
                        "use --no-resume to rebuild them."
                    )
                old_protocol = existing_contract.get("protocol") or {}
                new_protocol = run_contract["protocol"]
                old_parameters = existing_contract.get("parameters") or {}
                new_parameters = run_contract["parameters"]
                output_contract_changed = (
                    old_protocol.get("year_from") != new_protocol["year_from"]
                    or old_protocol.get("year_to") != new_protocol["year_to"]
                    or old_parameters.get("candidate_pool_size")
                    != new_parameters["candidate_pool_size"]
                    or old_parameters.get("edge_row_limit") != new_parameters["edge_row_limit"]
                )
                if (
                    output_contract_changed
                    and manifest.get("status") == "complete"
                    and not refinalize
                ):
                    raise ProcessingError(
                        "Year scope or visualization limits changed; run with "
                        "--refinalize to rebuild outputs from the saved partitions."
                    )
            manifest["run_contract"] = run_contract
            write_json(manifest_path, manifest)
            if manifest.get("status") == "complete" and not refinalize:
                quality_path = paths.quality / "processing_report.json"
                if quality_path.exists():
                    return {
                        "project_root": str(paths.root),
                        "resumed": True,
                        "already_complete": True,
                        "manifest": str(manifest_path),
                        "quality": read_json(quality_path),
                    }
        else:
            if parts_root.exists():
                resolved = parts_root.resolve()
                if resolved.parent != paths.canonical.resolve():
                    raise ProcessingError("Refusing to remove partitions outside canonical/.")
                shutil.rmtree(resolved)
            manifest = {
                "version": 1,
                "status": "normalizing",
                "source": config.protocol.source.value,
                "input_path": str(input_path),
                "input_sha256": fingerprint,
                "input_bytes": input_path.stat().st_size,
                "chunk_size": actual_chunk_size,
                "records_processed": 0,
                "batches_completed": 0,
                "part_rows": {table: 0 for table in PART_TABLE_COLUMNS},
                "started_at": _utc_iso(),
                "updated_at": _utc_iso(),
                "finished_at": None,
                "outputs": {},
                "run_contract": run_contract,
            }
            write_json(manifest_path, manifest)

        if refinalize:
            expected_batches = int(manifest.get("batches_completed", 0))
            missing_parts = [
                table
                for table in PART_TABLE_COLUMNS
                if len(list((parts_root / table).glob("part-*.parquet"))) != expected_batches
            ]
            if expected_batches < 1 or missing_parts:
                raise ProcessingError(
                    "Cannot re-finalize because normalized partitions are incomplete: "
                    + ", ".join(missing_parts)
                )
            manifest["status"] = "finalizing"
            manifest["updated_at"] = _utc_iso()
            write_json(manifest_path, manifest)
            quality, outputs = _finalize(
                paths,
                manifest,
                memory_limit=policy.duckdb_memory_limit,
                candidate_pool=candidate_pool,
                edge_limit=policy.edge_row_limit,
                year_from=config.protocol.year_from,
                year_to=config.protocol.year_to,
            )
            manifest["status"] = "complete" if quality["passed"] else "quality_failed"
            manifest["outputs"] = outputs
            manifest["candidate_pool_size"] = candidate_pool
            manifest["edge_row_limit"] = policy.edge_row_limit
            manifest["elapsed_seconds"] = round(time.perf_counter() - started, 3)
            manifest["updated_at"] = _utc_iso()
            manifest["finished_at"] = _utc_iso()
            manifest["refinalized"] = True
            write_json(manifest_path, manifest)
            return {
                "project_root": str(paths.root),
                "resumed": True,
                "refinalized": True,
                "manifest": str(manifest_path),
                "records_processed": manifest["records_processed"],
                "elapsed_seconds": manifest["elapsed_seconds"],
                "quality": quality,
                "visualization_rows": outputs["visualization_rows"],
            }

        iterator = iter_staged_records(input_path)
        skip = int(manifest["records_processed"])
        skipped = sum(1 for _ in islice(iterator, skip))
        if skipped != skip:
            raise ProcessingError(
                f"Checkpoint expects {skip} records, but the staged input ended at {skipped}."
            )
        processed_batches_this_run = 0
        try:
            while True:
                batch = list(islice(iterator, actual_chunk_size))
                if not batch:
                    break
                batch_index = int(manifest["batches_completed"]) + 1
                counts = _write_batch(
                    parts_root,
                    config.protocol.source.value,
                    batch,
                    batch_index,
                    int(manifest["records_processed"]),
                )
                manifest["records_processed"] += len(batch)
                manifest["batches_completed"] = batch_index
                for table, count in counts.items():
                    manifest["part_rows"][table] += count
                manifest["updated_at"] = _utc_iso()
                write_json(manifest_path, manifest)
                processed_batches_this_run += 1
                if batch_budget is not None and processed_batches_this_run >= batch_budget:
                    manifest["status"] = "partial"
                    manifest["updated_at"] = _utc_iso()
                    manifest["elapsed_seconds"] = round(time.perf_counter() - started, 3)
                    write_json(manifest_path, manifest)
                    return {
                        "project_root": str(paths.root),
                        "partial": True,
                        "resumable": True,
                        "manifest": str(manifest_path),
                        "records_processed": manifest["records_processed"],
                        "batches_completed": manifest["batches_completed"],
                        "elapsed_seconds": manifest["elapsed_seconds"],
                    }

            manifest["status"] = "finalizing"
            manifest["updated_at"] = _utc_iso()
            write_json(manifest_path, manifest)
            quality, outputs = _finalize(
                paths,
                manifest,
                memory_limit=policy.duckdb_memory_limit,
                candidate_pool=candidate_pool,
                edge_limit=policy.edge_row_limit,
                year_from=config.protocol.year_from,
                year_to=config.protocol.year_to,
            )
            manifest["status"] = "complete" if quality["passed"] else "quality_failed"
            manifest["outputs"] = outputs
            manifest["candidate_pool_size"] = candidate_pool
            manifest["edge_row_limit"] = policy.edge_row_limit
            manifest["elapsed_seconds"] = round(time.perf_counter() - started, 3)
            manifest["updated_at"] = _utc_iso()
            manifest["finished_at"] = _utc_iso()
            write_json(manifest_path, manifest)
            if not retain_parts and quality["passed"]:
                resolved = parts_root.resolve()
                if resolved.parent == paths.canonical.resolve():
                    shutil.rmtree(resolved)
            return {
                "project_root": str(paths.root),
                "resumed": skip > 0,
                "already_complete": False,
                "manifest": str(manifest_path),
                "records_processed": manifest["records_processed"],
                "batches_completed": manifest["batches_completed"],
                "elapsed_seconds": manifest["elapsed_seconds"],
                "quality": quality,
                "visualization_rows": outputs["visualization_rows"],
            }
        except BaseException as exc:
            manifest["status"] = "interrupted"
            manifest["updated_at"] = _utc_iso()
            manifest["last_error"] = {
                "type": type(exc).__name__,
                "message": str(exc)[:2_000],
            }
            write_json(manifest_path, manifest)
            raise
