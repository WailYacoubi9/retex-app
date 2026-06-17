from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from clients import Neo4jClient, OllamaClient, QdrantWrapper

logger = logging.getLogger(__name__)

MIN_SCORE_THRESHOLD = 0.52


@dataclass
class RetrievedInfoSecurite:
    info_securite_id: str
    is_number: Optional[str] = None
    annee: Optional[int] = None
    titre: Optional[str] = None
    sujet: Optional[str] = None
    objectif: Optional[str] = None
    contexte: Optional[str] = None
    actions_recommandees: Optional[str] = None
    operateurs_concernes: Optional[str] = None
    llm_resume: Optional[str] = None
    remplace: list[str] = field(default_factory=list)
    remplacee_par: list[str] = field(default_factory=list)
    best_score: float = 0.0
    matched_fields: list[str] = field(default_factory=list)
    is_expanded: bool = False


@dataclass
class RetrievalResultIS:
    items: list[RetrievedInfoSecurite] = field(default_factory=list)
    n_chunks_retrieved: int = 0
    n_direct: int = 0
    n_expanded: int = 0
    below_threshold: bool = False


def retrieve_info_securite(
    question: str,
    ollama: OllamaClient,
    qdrant: QdrantWrapper,
    neo4j: Neo4jClient,
    top_k: int = 5,
    min_score: float = MIN_SCORE_THRESHOLD,
) -> RetrievalResultIS:
    question_vector = ollama.embed(question)

    chunks = qdrant.search(vector=question_vector, top_k=top_k, exclude_test_data=True)
    logger.info("Qdrant returned %d chunks", len(chunks))

    relevant = [c for c in chunks if c.get("score", 0.0) >= min_score]
    logger.info("Chunks above threshold: %d / %d", len(relevant), len(chunks))

    if not relevant:
        return RetrievalResultIS(n_chunks_retrieved=len(chunks), below_threshold=True)

    grouped = _group_chunks_by_is(relevant)
    logger.info("Grouped into %d unique InfoSecurite", len(grouped))

    direct: list[RetrievedInfoSecurite] = []
    for is_id, group_data in grouped.items():
        item = _fetch_is_from_neo4j(neo4j, is_id, group_data)
        if item:
            direct.append(item)

    expanded = _expand_via_remplace(neo4j, direct)

    return RetrievalResultIS(
        items=direct + expanded,
        n_chunks_retrieved=len(chunks),
        n_direct=len(direct),
        n_expanded=len(expanded),
        below_threshold=False,
    )


def _group_chunks_by_is(chunks: list[dict]) -> dict[str, dict]:
    grouped: dict[str, dict] = {}
    for chunk in chunks:
        payload = chunk.get("payload", {})
        is_id = payload.get("info_securite_id")
        if not is_id:
            continue
        score = chunk.get("score", 0.0)
        field_name = payload.get("field_canonical", "unknown")
        if is_id not in grouped:
            grouped[is_id] = {"best_score": score, "matched_fields": [field_name]}
        else:
            if score > grouped[is_id]["best_score"]:
                grouped[is_id]["best_score"] = score
            if field_name not in grouped[is_id]["matched_fields"]:
                grouped[is_id]["matched_fields"].append(field_name)
    return grouped


def _fetch_is_from_neo4j(
    neo4j: Neo4jClient,
    is_id: str,
    group_data: dict,
) -> Optional[RetrievedInfoSecurite]:
    cypher = """
    MATCH (i:InfoSecurite {info_securite_id: $is_id})
    OPTIONAL MATCH (i)-[:REMPLACE]->(ancien:InfoSecurite)
    OPTIONAL MATCH (nouveau:InfoSecurite)-[:REMPLACE]->(i)
    RETURN i AS is_node,
           collect(DISTINCT ancien.is_number) AS remplace,
           collect(DISTINCT nouveau.is_number) AS remplacee_par
    """
    results = neo4j.run(cypher, is_id=is_id)
    if not results:
        logger.warning("InfoSecurite %s introuvable dans Neo4j", is_id)
        return None

    row = results[0]
    node = row["is_node"]

    return RetrievedInfoSecurite(
        info_securite_id=is_id,
        is_number=node.get("is_number"),
        annee=node.get("annee"),
        titre=node.get("titre"),
        sujet=node.get("sujet"),
        objectif=node.get("objectif"),
        contexte=node.get("contexte"),
        actions_recommandees=node.get("actions_recommandees"),
        operateurs_concernes=node.get("operateurs_concernes"),
        llm_resume=node.get("llm_resume"),
        remplace=[n for n in row["remplace"] if n],
        remplacee_par=[n for n in row["remplacee_par"] if n],
        best_score=group_data["best_score"],
        matched_fields=group_data["matched_fields"],
        is_expanded=False,
    )


def _expand_via_remplace(
    neo4j: Neo4jClient,
    seeds: list[RetrievedInfoSecurite],
    limit_per_seed: int = 2,
) -> list[RetrievedInfoSecurite]:
    if not seeds:
        return []

    seed_ids = {s.info_securite_id for s in seeds}
    expanded: dict[str, RetrievedInfoSecurite] = {}

    for seed in seeds:
        cypher = """
        MATCH (i:InfoSecurite {info_securite_id: $is_id})
        MATCH (i)-[:REMPLACE*1..2]-(neighbor:InfoSecurite)
        WHERE neighbor.info_securite_id <> $is_id
          AND NOT coalesce(neighbor.is_stub, false)
        RETURN DISTINCT neighbor.info_securite_id AS neighbor_id
        LIMIT $limit
        """
        results = neo4j.run(cypher, is_id=seed.info_securite_id, limit=limit_per_seed)
        for row in results:
            neighbor_id = row["neighbor_id"]
            if neighbor_id in seed_ids or neighbor_id in expanded:
                continue
            item = _fetch_is_from_neo4j(
                neo4j, neighbor_id, {"best_score": 0.0, "matched_fields": ["graph_expansion"]}
            )
            if item:
                item.is_expanded = True
                expanded[neighbor_id] = item

    logger.info("Graph expansion added %d IS (via REMPLACE)", len(expanded))
    return list(expanded.values())
