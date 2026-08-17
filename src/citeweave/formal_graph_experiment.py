from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from .io import load_config, read_json, sha256_file, write_json

Condition = Literal["no_rag", "flat_structured", "graph_rag", "figure_vlm"]


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _display_label(value: Any, *, long_label: bool) -> str:
    text = " ".join(str(value or "").split())
    limit = 25 if long_label else 26
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


@dataclass(frozen=True)
class GraphAtom:
    atom_id: str
    network: str
    subject: str
    predicate: str
    object: Any
    object_type: Literal["entity", "label", "integer", "number", "boolean"]


@dataclass(frozen=True)
class FormalGraphQuestion:
    item_id: str
    dataset_id: str
    network: str
    task_type: str
    question: str
    gold_answer: dict[str, Any] | None
    answerable: bool
    atom_ids: list[str]
    figure_eligible: bool
    figure_path: str | None
    answer_contract: dict[str, Any]


class ProviderProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profile_id: str
    base_url: str
    model: str
    api_key_env: str
    modality: Literal["text", "vision"]
    capability_snapshot: Path
    response_format: Literal["json_schema", "json_object"] = "json_object"
    supports_json_schema: bool = False
    supports_seed: bool = False
    supports_data_uri: bool = False


class CapabilityValidation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profile_id: str
    snapshot_sha256: str
    model_available: bool
    image_input_verified: bool | None
    passed: bool
    reasons: list[str] = Field(default_factory=list)


def semantic_atom_hash(atoms: list[GraphAtom]) -> str:
    records = sorted((asdict(atom) for atom in atoms), key=lambda item: item["atom_id"])
    return _sha(records)


def _atom_index(atoms: list[GraphAtom]) -> dict[str, GraphAtom]:
    indexed = {atom.atom_id: atom for atom in atoms}
    if len(indexed) != len(atoms):
        raise ValueError("Canonical atom ids must be unique")
    return indexed


def render_flat_context(atoms: list[GraphAtom]) -> dict[str, Any]:
    ordered = sorted(atoms, key=lambda atom: atom.atom_id)
    return {
        "representation": "flat_structured_rows",
        "semantic_atom_sha256": semantic_atom_hash(ordered),
        "atom_ids": [atom.atom_id for atom in ordered],
        "rows": [asdict(atom) for atom in ordered],
    }


def render_graph_context(atoms: list[GraphAtom]) -> dict[str, Any]:
    ordered = sorted(atoms, key=lambda atom: atom.atom_id)
    entity_ids = {
        str(value)
        for atom in ordered
        for value in (atom.subject, atom.object)
        if atom.object_type == "entity" or value == atom.subject
    }
    paths = []
    for atom in ordered:
        if atom.predicate in {"connected_to", "strongest_connection_to"}:
            paths.append(
                {
                    "path_id": f"path:{atom.atom_id}",
                    "nodes": [atom.subject, str(atom.object)],
                    "edges": [atom.atom_id],
                }
            )
    return {
        "representation": "graph_subgraph",
        "semantic_atom_sha256": semantic_atom_hash(ordered),
        "atom_ids": [atom.atom_id for atom in ordered],
        "entities": sorted(entity_ids),
        "relationships": [asdict(atom) for atom in ordered],
        "paths": paths,
    }


def assert_semantically_equivalent(
    flat_context: dict[str, Any],
    graph_context: dict[str, Any],
) -> None:
    if flat_context.get("semantic_atom_sha256") != graph_context.get(
        "semantic_atom_sha256"
    ):
        raise ValueError("Flat and Graph contexts do not have the same semantic atom hash")
    if set(flat_context.get("atom_ids", [])) != set(graph_context.get("atom_ids", [])):
        raise ValueError("Flat and Graph contexts do not contain the same atom ids")


def _network_atoms_and_questions(
    *,
    dataset_id: str,
    network: str,
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    figure_path: Path | None,
) -> tuple[list[GraphAtom], list[FormalGraphQuestion]]:
    required_node_columns = {"id", "label", "occurrences", "cluster", "importance"}
    required_edge_columns = {"source", "target", "weight"}
    missing_nodes = required_node_columns - set(nodes)
    missing_edges = required_edge_columns - set(edges)
    if missing_nodes or missing_edges:
        raise ValueError(
            f"{network}: selected-network tables are incomplete; "
            f"missing node={sorted(missing_nodes)}, edge={sorted(missing_edges)}"
        )
    nodes = nodes.copy()
    edges = edges.copy().reset_index(drop=True)
    nodes["id"] = nodes["id"].astype(str)
    edges["source"] = edges["source"].astype(str)
    edges["target"] = edges["target"].astype(str)
    long_label = network in {"cocitation", "citation", "bibliographic_coupling"}
    nodes["display_label"] = nodes["label"].map(
        lambda value: _display_label(value, long_label=long_label)
    )
    label_counts = nodes["display_label"].value_counts()
    nodes["label_unique"] = nodes["display_label"].map(label_counts).eq(1)
    node_by_id = nodes.set_index("id")

    atoms: list[GraphAtom] = [
        GraphAtom(
            atom_id=f"{network}:fact:node_count",
            network=network,
            subject=f"network:{network}",
            predicate="displayed_node_count",
            object=len(nodes),
            object_type="integer",
        ),
        GraphAtom(
            atom_id=f"{network}:fact:edge_count",
            network=network,
            subject=f"network:{network}",
            predicate="displayed_edge_count",
            object=len(edges),
            object_type="integer",
        ),
    ]
    for row in nodes.itertuples(index=False):
        entity = f"{network}:node:{row.id}"
        atoms.extend(
            [
                GraphAtom(
                    atom_id=f"{network}:node:{row.id}:label",
                    network=network,
                    subject=entity,
                    predicate="display_label",
                    object=str(row.display_label),
                    object_type="label",
                ),
                GraphAtom(
                    atom_id=f"{network}:node:{row.id}:occurrences",
                    network=network,
                    subject=entity,
                    predicate="occurrences",
                    object=float(row.occurrences),
                    object_type="number",
                ),
                GraphAtom(
                    atom_id=f"{network}:node:{row.id}:cluster",
                    network=network,
                    subject=entity,
                    predicate="cluster",
                    object=int(row.cluster),
                    object_type="integer",
                ),
            ]
        )
    for index, row in enumerate(edges.itertuples(index=False), 1):
        edge_id = f"{network}:edge:{index:04d}"
        source = f"{network}:node:{row.source}"
        target = f"{network}:node:{row.target}"
        atoms.extend(
            [
                GraphAtom(
                    atom_id=f"{edge_id}:relation",
                    network=network,
                    subject=source,
                    predicate="connected_to",
                    object=target,
                    object_type="entity",
                ),
                GraphAtom(
                    atom_id=f"{edge_id}:weight",
                    network=network,
                    subject=edge_id,
                    predicate="weight",
                    object=float(row.weight),
                    object_type="number",
                ),
            ]
        )

    image_available = bool(figure_path and figure_path.is_file())
    questions: list[FormalGraphQuestion] = []

    def add_question(
        suffix: str,
        task_type: str,
        question: str,
        gold_answer: dict[str, Any] | None,
        atom_ids: list[str],
        *,
        answerable: bool = True,
        figure_eligible: bool = False,
        answer_contract: dict[str, Any],
    ) -> None:
        questions.append(
            FormalGraphQuestion(
                item_id=f"{dataset_id}:{network}:{suffix}",
                dataset_id=dataset_id,
                network=network,
                task_type=task_type,
                question=question,
                gold_answer=gold_answer,
                answerable=answerable,
                atom_ids=atom_ids,
                figure_eligible=figure_eligible and image_available,
                figure_path=str(figure_path.resolve()) if image_available else None,
                answer_contract=answer_contract,
            )
        )

    add_question(
        "network_size",
        "network_size",
        f"How many nodes and links are displayed in the {network} network?",
        {"nodes": len(nodes), "links": len(edges)},
        [f"{network}:fact:node_count", f"{network}:fact:edge_count"],
        figure_eligible=True,
        answer_contract={"nodes": "integer", "links": "integer"},
    )

    unique_nodes = nodes[nodes["label_unique"]]
    overall_top = nodes.sort_values(
        ["occurrences", "display_label"], ascending=[False, True]
    ).iloc[0]
    if bool(overall_top["label_unique"]):
        top = overall_top
        top_entity = f"{network}:node:{top['id']}"
        top_fact = GraphAtom(
            atom_id=f"{network}:fact:highest_occurrence",
            network=network,
            subject=f"network:{network}",
            predicate="highest_occurrence_node",
            object=top_entity,
            object_type="entity",
        )
        atoms.append(top_fact)
        top_atoms = [
            top_fact.atom_id,
            f"{network}:node:{top['id']}:label",
            f"{network}:node:{top['id']}:occurrences",
        ]
        add_question(
            "highest_occurrence",
            "highest_occurrence",
            f"Which labeled node has the highest occurrence in the {network} network?",
            {"label": str(top["display_label"])},
            top_atoms,
            # The formal map winsorizes node sizes, so even a visible label does
            # not make an exact highest-occurrence comparison visually provable.
            figure_eligible=False,
            answer_contract={"label": "visible string"},
        )

    if not edges.empty:
        strongest = edges.assign(
            _left=edges["source"].astype(str), _right=edges["target"].astype(str)
        ).sort_values(["weight", "_left", "_right"], ascending=[False, True, True]).iloc[0]
        source_id = str(strongest["source"])
        target_id = str(strongest["target"])
        if (
            source_id in node_by_id.index
            and target_id in node_by_id.index
            and bool(node_by_id.loc[source_id, "label_unique"])
            and bool(node_by_id.loc[target_id, "label_unique"])
        ):
            source_label = str(node_by_id.loc[source_id, "display_label"])
            target_label = str(node_by_id.loc[target_id, "display_label"])
            if source_label != target_label:
                edge_position = int(strongest.name) + 1
                relation_atom = f"{network}:edge:{edge_position:04d}:relation"
                weight_atom = f"{network}:edge:{edge_position:04d}:weight"
                strongest_fact = GraphAtom(
                    atom_id=f"{network}:fact:strongest_connection",
                    network=network,
                    subject=f"{network}:node:{source_id}",
                    predicate="strongest_connection_to",
                    object=f"{network}:node:{target_id}",
                    object_type="entity",
                )
                atoms.append(strongest_fact)
                add_question(
                    "strongest_connection",
                    "strongest_connection",
                    f"Which labeled pair has the strongest displayed connection in the "
                    f"{network} network, and what is its weight?",
                    {
                        "source_label": source_label,
                        "target_label": target_label,
                        "weight": float(strongest["weight"]),
                    },
                    [
                        strongest_fact.atom_id,
                        relation_atom,
                        weight_atom,
                        f"{network}:node:{source_id}:label",
                        f"{network}:node:{target_id}:label",
                    ],
                    answer_contract={
                        "source_label": "visible string",
                        "target_label": "visible string",
                        "weight": "number",
                    },
                )

    cluster_candidates = unique_nodes.sort_values(
        ["importance", "display_label"], ascending=[False, True]
    )
    if not cluster_candidates.empty:
        member = cluster_candidates.iloc[0]
        add_question(
            "cluster_membership",
            "cluster_membership",
            f"Which detected cluster contains the labeled node '{member['display_label']}' "
            f"in the {network} network?",
            {"label": str(member["display_label"]), "cluster": int(member["cluster"])},
            [
                f"{network}:node:{member['id']}:label",
                f"{network}:node:{member['id']}:cluster",
            ],
            answer_contract={"label": "visible string", "cluster": "integer"},
        )

    absent_label = "CiteWeave absent benchmark node"
    absent_fact = GraphAtom(
        atom_id=f"{network}:fact:absent_target",
        network=network,
        subject=f"network:{network}",
        predicate="contains_visible_label",
        object=False,
        object_type="boolean",
    )
    atoms.append(absent_fact)
    add_question(
        "false_premise",
        "unanswerable_false_premise",
        f"Which cluster contains '{absent_label}' in the {network} network?",
        None,
        [absent_fact.atom_id],
        answerable=False,
        answer_contract={"abstain": "true"},
    )
    return atoms, questions


def build_formal_graph_grounding(workspace: Path) -> dict[str, Any]:
    workspace = workspace.resolve()
    config = load_config(workspace / "project.yml")
    visual_dir = workspace / "analyses" / "visualization"
    figure_manifest_path = workspace / "figures" / "figure_manifest.json"
    if not visual_dir.is_dir():
        raise FileNotFoundError(f"Missing selected-network directory: {visual_dir}")
    if not figure_manifest_path.is_file():
        raise FileNotFoundError(f"Missing figure manifest: {figure_manifest_path}")
    figure_manifest = read_json(figure_manifest_path)
    figure_records = {
        item["name"]: item
        for item in figure_manifest.get("figures", [])
        if isinstance(item, dict) and item.get("name")
    }
    all_atoms: list[GraphAtom] = []
    all_questions: list[FormalGraphQuestion] = []
    for nodes_path in sorted(visual_dir.glob("*_nodes.parquet")):
        network = nodes_path.name.removesuffix("_nodes.parquet")
        edges_path = visual_dir / f"{network}_edges.parquet"
        if not edges_path.is_file():
            continue
        figure_record = figure_records.get(f"network_{network}", {})
        figure_value = figure_record.get("png")
        figure_path = Path(figure_value) if figure_value else None
        if figure_path and not figure_path.is_absolute():
            figure_path = workspace / "figures" / figure_path
        atoms, questions = _network_atoms_and_questions(
            dataset_id=config.project_id,
            network=network,
            nodes=pd.read_parquet(nodes_path),
            edges=pd.read_parquet(edges_path),
            figure_path=figure_path,
        )
        all_atoms.extend(atoms)
        all_questions.extend(questions)
    if not all_questions:
        raise ValueError("No bounded selected-network questions could be generated")
    atom_lookup = _atom_index(all_atoms)
    output = workspace / "evidence" / "formal_graph_experiment"
    context_dir = output / "contexts"
    context_dir.mkdir(parents=True, exist_ok=True)
    context_records = []
    for question in all_questions:
        atoms = [atom_lookup[atom_id] for atom_id in question.atom_ids]
        flat = render_flat_context(atoms)
        graph = render_graph_context(atoms)
        assert_semantically_equivalent(flat, graph)
        safe_id = question.item_id.replace(":", "__")
        flat_path = context_dir / f"{safe_id}.flat.json"
        graph_path = context_dir / f"{safe_id}.graph.json"
        write_json(flat_path, flat)
        write_json(graph_path, graph)
        context_records.append(
            {
                "item_id": question.item_id,
                "semantic_atom_sha256": flat["semantic_atom_sha256"],
                "flat_path": str(flat_path.relative_to(workspace).as_posix()),
                "flat_sha256": sha256_file(flat_path),
                "graph_path": str(graph_path.relative_to(workspace).as_posix()),
                "graph_sha256": sha256_file(graph_path),
                "flat_characters": len(_canonical_json(flat)),
                "graph_characters": len(_canonical_json(graph)),
            }
        )
    atoms_path = output / "canonical_atoms.json"
    questions_path = output / "benchmark.json"
    write_json(atoms_path, [asdict(atom) for atom in all_atoms])
    write_json(questions_path, [asdict(question) for question in all_questions])
    manifest = {
        "version": 1,
        "dataset_id": config.project_id,
        "workspace": str(workspace),
        "source": "bounded selected networks used by formal visualizations",
        "canonical_atoms_sha256": semantic_atom_hash(all_atoms),
        "canonical_atoms_file_sha256": sha256_file(atoms_path),
        "benchmark_sha256": sha256_file(questions_path),
        "figure_manifest_sha256": sha256_file(figure_manifest_path),
        "bounded_graph_qa_sha256": (
            sha256_file(workspace / "evidence" / "graph_qa_benchmark.json")
            if (workspace / "evidence" / "graph_qa_benchmark.json").is_file()
            else None
        ),
        "bounded_graph_facts_sha256": (
            sha256_file(workspace / "evidence" / "graph_facts.json")
            if (workspace / "evidence" / "graph_facts.json").is_file()
            else None
        ),
        "questions": len(all_questions),
        "figure_eligible_questions": sum(item.figure_eligible for item in all_questions),
        "conditions": [
            "no_rag",
            "flat_structured",
            "graph_rag",
            "figure_vlm",
        ],
        "figure_vlm_comparison_role": "cross_model_extension",
        "figure_contract": (
            "Answers use only labels and numbers visibly represented in the image; "
            "canonical node and evidence ids are not required."
        ),
        "contexts": context_records,
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def verify_formal_graph_grounding(workspace: Path) -> dict[str, Any]:
    workspace = workspace.resolve()
    output = workspace / "evidence" / "formal_graph_experiment"
    errors: list[str] = []
    manifest_path = output / "manifest.json"
    benchmark_path = output / "benchmark.json"
    atoms_path = output / "canonical_atoms.json"
    for path in (manifest_path, benchmark_path, atoms_path):
        if not path.is_file():
            errors.append(f"Missing required artifact: {path}")
    if errors:
        return {"passed": False, "errors": errors, "contexts_verified": 0}
    manifest = read_json(manifest_path)
    benchmark = read_json(benchmark_path)
    atom_records = read_json(atoms_path)
    if manifest.get("benchmark_sha256") != sha256_file(benchmark_path):
        errors.append("Benchmark hash does not match the formal graph manifest")
    if manifest.get("canonical_atoms_file_sha256") != sha256_file(atoms_path):
        errors.append("Canonical atom file hash does not match the formal graph manifest")
    try:
        canonical_atoms = [GraphAtom(**record) for record in atom_records]
        _atom_index(canonical_atoms)
        if manifest.get("canonical_atoms_sha256") != semantic_atom_hash(canonical_atoms):
            errors.append("Canonical atom semantic hash does not match the manifest")
    except (TypeError, ValueError) as exc:
        errors.append(f"Canonical atom validation failed: {exc}")
    upstream_hashes = (
        ("figure_manifest_sha256", workspace / "figures" / "figure_manifest.json"),
        ("bounded_graph_qa_sha256", workspace / "evidence" / "graph_qa_benchmark.json"),
        ("bounded_graph_facts_sha256", workspace / "evidence" / "graph_facts.json"),
    )
    for field, path in upstream_hashes:
        expected = manifest.get(field)
        if expected is None:
            continue
        if not path.is_file() or sha256_file(path) != expected:
            errors.append(f"Upstream artifact hash mismatch: {field}")
    item_ids = [str(item.get("item_id")) for item in benchmark]
    if len(item_ids) != len(set(item_ids)):
        errors.append("Benchmark item ids are not unique")
    contexts = manifest.get("contexts") or []
    context_ids = [str(item.get("item_id")) for item in contexts]
    if set(context_ids) != set(item_ids):
        errors.append("Context records do not exactly cover benchmark item ids")
    contexts_verified = 0
    for record in contexts:
        try:
            flat_path = workspace / record["flat_path"]
            graph_path = workspace / record["graph_path"]
            if sha256_file(flat_path) != record["flat_sha256"]:
                raise ValueError("flat context hash mismatch")
            if sha256_file(graph_path) != record["graph_sha256"]:
                raise ValueError("graph context hash mismatch")
            flat = read_json(flat_path)
            graph = read_json(graph_path)
            assert_semantically_equivalent(flat, graph)
            if flat["semantic_atom_sha256"] != record["semantic_atom_sha256"]:
                raise ValueError("context semantic hash mismatch")
            contexts_verified += 1
        except (KeyError, OSError, TypeError, ValueError) as exc:
            errors.append(f"{record.get('item_id', '<unknown>')}: {exc}")
    for item in benchmark:
        if not item.get("figure_eligible"):
            continue
        figure_path = Path(str(item.get("figure_path") or ""))
        if not figure_path.is_file():
            errors.append(f"{item['item_id']}: eligible Figure input is missing")
        contract_text = _canonical_json(item.get("answer_contract", {})).casefold()
        if "canonical" in contract_text or "evidence_id" in contract_text:
            errors.append(
                f"{item['item_id']}: Figure answer contract exposes hidden identifiers"
            )
    if manifest.get("figure_vlm_comparison_role") not in {
        "cross_model_extension",
        "strict_comparison",
    }:
        errors.append("Figure comparison role is not declared")
    return {
        "passed": not errors,
        "errors": errors,
        "questions": len(benchmark),
        "contexts_verified": contexts_verified,
        "manifest_sha256": sha256_file(manifest_path),
    }


def validate_provider_profile(
    profile: ProviderProfile,
    *,
    condition: Condition,
) -> CapabilityValidation:
    reasons: list[str] = []
    path = profile.capability_snapshot.resolve()
    if not path.is_file():
        return CapabilityValidation(
            profile_id=profile.profile_id,
            snapshot_sha256="",
            model_available=False,
            image_input_verified=None,
            passed=False,
            reasons=[f"Capability snapshot does not exist: {path}"],
        )
    snapshot = read_json(path)
    snapshot_base = str(snapshot.get("base_url", "")).rstrip("/")
    if snapshot_base != profile.base_url.rstrip("/"):
        reasons.append("Capability snapshot base_url does not match provider profile")
    models = snapshot.get("models") or []
    features = snapshot.get("features") or {}
    model_available = profile.model in models
    if not model_available:
        reasons.append(f"Model {profile.model!r} is absent from the capability snapshot")
    image_verified: bool | None = None
    if condition == "figure_vlm":
        if profile.modality != "vision":
            reasons.append("figure_vlm requires a vision provider profile")
        probe = snapshot.get("vlm_probe") or {}
        image_verified = bool(
            probe.get("model") == profile.model and probe.get("accepted_image_input") is True
        )
        if not image_verified:
            reasons.append("Image input was not verified for the configured VLM")
        if not profile.supports_data_uri:
            reasons.append("figure_vlm profile must explicitly declare data-URI support")
    if profile.response_format == "json_schema" and (
        not profile.supports_json_schema or features.get("json_schema") is not True
    ):
        reasons.append(
            "json_schema response format was not verified in the capability snapshot"
        )
    if profile.supports_seed and features.get("seed") is not True:
        reasons.append("seed support was not verified in the capability snapshot")
    return CapabilityValidation(
        profile_id=profile.profile_id,
        snapshot_sha256=sha256_file(path),
        model_available=model_available,
        image_input_verified=image_verified,
        passed=not reasons,
        reasons=reasons,
    )


def comparison_design(
    text_profile: ProviderProfile,
    vision_profile: ProviderProfile | None,
) -> dict[str, Any]:
    same_model = bool(
        vision_profile
        and text_profile.base_url.rstrip("/") == vision_profile.base_url.rstrip("/")
        and text_profile.model == vision_profile.model
    )
    return {
        "primary_text_panel": {
            "profile_id": text_profile.profile_id,
            "model": text_profile.model,
            "base_url": text_profile.base_url.rstrip("/"),
            "conditions": ["no_rag", "flat_structured", "graph_rag"],
            "strict_within_model_comparison": True,
        },
        "figure_panel": (
            {
                "profile_id": vision_profile.profile_id,
                "model": vision_profile.model,
                "base_url": vision_profile.base_url.rstrip("/"),
                "conditions": ["figure_vlm"],
                "strict_within_model_comparison": same_model,
                "role": "strict_comparison" if same_model else "cross_model_extension",
            }
            if vision_profile
            else None
        ),
    }


def select_condition_items(
    items: list[dict[str, Any]],
    *,
    condition: Condition,
    design: dict[str, Any],
) -> list[dict[str, Any]]:
    strict_figure_panel = bool(
        (design.get("figure_panel") or {}).get("strict_within_model_comparison")
    )
    require_figure_eligibility = condition == "figure_vlm" or strict_figure_panel
    return [
        item
        for item in items
        if not require_figure_eligibility or bool(item.get("figure_eligible"))
    ]


def provider_for_condition(
    text_profile: ProviderProfile,
    vision_profile: ProviderProfile | None,
    condition: Condition,
) -> ProviderProfile:
    if condition == "figure_vlm":
        if vision_profile is None:
            raise ValueError("figure_vlm requires a vision provider profile")
        return vision_profile
    return text_profile


def formal_run_directory(
    root: Path,
    *,
    dataset_id: str,
    run_id: str,
    condition: Condition,
) -> Path:
    return (
        root.resolve()
        / "experiments"
        / "formal_runs"
        / dataset_id
        / run_id
        / condition
    )


class JsonlCheckpoint:
    def __init__(self, path: Path):
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._records: dict[tuple[str, str], dict[str, Any]] = {}
        self._attempts: dict[tuple[str, str], int] = {}
        if self.path.is_file():
            for line_number, line in enumerate(
                self.path.read_text(encoding="utf-8").splitlines(), 1
            ):
                if not line.strip():
                    continue
                record = json.loads(line)
                key = (str(record["condition"]), str(record["item_id"]))
                if self._records.get(key, {}).get("status") == "complete":
                    raise ValueError(
                        f"Checkpoint has records after completion at line {line_number}: {key}"
                    )
                self._records[key] = record
                self._attempts[key] = self._attempts.get(key, 0) + 1

    def completed(self, condition: Condition, item_id: str) -> bool:
        record = self._records.get((condition, item_id))
        return bool(record and record.get("status") == "complete")

    def append(
        self,
        *,
        run_id: str,
        condition: Condition,
        item_id: str,
        request: dict[str, Any],
        response: dict[str, Any] | None,
        status: Literal["complete", "failed"],
        elapsed_seconds: float,
        error: str | None = None,
    ) -> dict[str, Any]:
        key = (condition, item_id)
        if self._records.get(key, {}).get("status") == "complete":
            raise ValueError(f"Checkpoint already completed {key}")
        attempt = self._attempts.get(key, 0) + 1
        record = {
            "run_id": run_id,
            "condition": condition,
            "item_id": item_id,
            "attempt": attempt,
            "status": status,
            "recorded_at": datetime.now(UTC).isoformat(),
            "request_sha256": _sha(request),
            "request": request,
            "response": response,
            "elapsed_seconds": elapsed_seconds,
            "error": error,
        }
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(_canonical_json(record) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._records[key] = record
        self._attempts[key] = attempt
        return record
