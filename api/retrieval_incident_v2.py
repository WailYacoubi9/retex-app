"""
Retrieval pour la voie incidents v2 (:IncidentSecu).

Stratégie :
  1. Embed la question avec bge-m3
  2. Recherche Qdrant scopée sur source_module=incident_securite_v2
  3. Filtre par score >= MIN_SCORE_THRESHOLD
  4. Regroupe par incident_id (best_score + matched_fields)
  5. Enrichit chaque incident depuis Neo4j (sac complet de propriétés + entités)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from clients import Neo4jClient, OllamaClient, QdrantWrapper

logger = logging.getLogger(__name__)

MIN_SCORE_THRESHOLD = 0.45
SOURCE_MODULE = "incident_securite_v2"


@dataclass
class RetrievedIncidentV2:
    incident_id: str
    props: dict
    entites: list[dict]
    best_score: float
    matched_fields: list[str] = field(default_factory=list)


@dataclass
class RetrievalResultV2:
    items: list[RetrievedIncidentV2] = field(default_factory=list)
    n_chunks_retrieved: int = 0
    n_direct: int = 0
    below_threshold: bool = False


def retrieve_incident_v2(
    question: str,
    ollama: OllamaClient,
    qdrant: QdrantWrapper,
    neo4j: Neo4jClient,
    top_k: int = 5,
) -> RetrievalResultV2:
    question_vector = ollama.embed(question)

    chunks = qdrant.search(
        vector=question_vector,
        top_k=top_k,
        exclude_test_data=True,
        source_module=SOURCE_MODULE,
    )
    logger.info("Qdrant returned %d chunks (module=%s)", len(chunks), SOURCE_MODULE)

    relevant = [c for c in chunks if c.get("score", 0.0) >= MIN_SCORE_THRESHOLD]
    logger.info("Chunks above threshold (%.2f): %d / %d", MIN_SCORE_THRESHOLD, len(relevant), len(chunks))

    if not relevant:
        return RetrievalResultV2(n_chunks_retrieved=len(chunks), below_threshold=True)

    grouped = _group_by_incident(relevant)
    items: list[RetrievedIncidentV2] = []
    for incident_id, group in grouped.items():
        enriched = _enrich(neo4j, incident_id, group)
        if enriched:
            items.append(enriched)

    return RetrievalResultV2(
        items=items,
        n_chunks_retrieved=len(chunks),
        n_direct=len(items),
        below_threshold=False,
    )


def _group_by_incident(chunks: list[dict]) -> dict[str, dict]:
    grouped: dict[str, dict] = {}
    for chunk in chunks:
        payload = chunk.get("payload", {})
        incident_id = payload.get("incident_id")
        if not incident_id:
            continue
        score = chunk.get("score", 0.0)
        fc = payload.get("field_canonical", "unknown")
        if incident_id not in grouped:
            grouped[incident_id] = {"best_score": score, "matched_fields": [fc]}
        else:
            if score > grouped[incident_id]["best_score"]:
                grouped[incident_id]["best_score"] = score
            if fc not in grouped[incident_id]["matched_fields"]:
                grouped[incident_id]["matched_fields"].append(fc)
    return grouped


def _enrich(
    neo4j: Neo4jClient,
    incident_id: str,
    group: dict,
) -> Optional[RetrievedIncidentV2]:
    cypher = """
    MATCH (i:IncidentSecu {incident_id: $id})
    OPTIONAL MATCH (i)-[r]->(n)
    RETURN properties(i) AS props,
           collect({rel: type(r), labels: labels(n), props: properties(n)}) AS entites
    """
    results = neo4j.run(cypher, id=incident_id)
    if not results:
        logger.warning("IncidentSecu %s introuvable dans Neo4j", incident_id)
        return None
    row = results[0]
    entites = [e for e in row["entites"] if e.get("props")]
    return RetrievedIncidentV2(
        incident_id=incident_id,
        props=dict(row["props"]),
        entites=entites,
        best_score=group["best_score"],
        matched_fields=group["matched_fields"],
    )
