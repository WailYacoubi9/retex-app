"""
Couche de retrieval du RAG : recherche hybride Qdrant + Neo4j.

Strategie en 5 etapes :
  1. Embedde la question avec bge-m3
  2. Recherche Qdrant : top_k chunks similaires
  3. Filtre par score minimum (evite les reponses sur du non-pertinent)
  4. Regroupe par incident_id -> liste d'incidents uniques
  5. Expansion Neo4j : via referentiels uniquement (pas les personnes)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from clients import Neo4jClient, OllamaClient, QdrantWrapper

logger = logging.getLogger(__name__)

# Seuil minimum de score pour considerer un chunk comme pertinent.
MIN_SCORE_THRESHOLD = 0.45


@dataclass
class RetrievedIncident:
    """Un incident recupere avec son contexte enrichi."""
    incident_id: str
    incident_id_source: str
    numero_fe: Optional[str] = None
    titre: Optional[str] = None
    detail: Optional[str] = None
    resume_llm: Optional[str] = None
    facteur_causal: Optional[str] = None
    severite_percue: Optional[str] = None
    etat_final: Optional[str] = None
    date_evenement: Optional[str] = None
    recolte_faits: Optional[str] = None
    notes_suivi: Optional[str] = None

    personnes: list[dict] = field(default_factory=list)
    societes: list[dict] = field(default_factory=list)
    referentiels: list[dict] = field(default_factory=list)

    best_score: float = 0.0
    matched_fields: list[str] = field(default_factory=list)
    is_expanded: bool = False


@dataclass
class RetrievalResult:
    """Resultat complet d'une operation de retrieval."""
    incidents: list[RetrievedIncident] = field(default_factory=list)
    n_chunks_retrieved: int = 0
    n_incidents_direct: int = 0
    n_incidents_expanded: int = 0
    below_threshold: bool = False


def retrieve(
    question: str,
    ollama: OllamaClient,
    qdrant: QdrantWrapper,
    neo4j: Neo4jClient,
    top_k: int = 5,
    include_test_data: bool = False,
    min_score: float = MIN_SCORE_THRESHOLD,
) -> RetrievalResult:
    """Execute le retrieval complet pour une question donnee."""
    logger.debug("Embedding question: %s", question[:80])
    question_vector = ollama.embed(question)

    chunks = qdrant.search(
        vector=question_vector,
        top_k=top_k,
        exclude_test_data=not include_test_data,
    )
    logger.info("Qdrant returned %d chunks", len(chunks))

    relevant_chunks = [c for c in chunks if c.get("score", 0.0) >= min_score]
    logger.info(
        "Chunks above threshold (%.2f): %d / %d",
        min_score, len(relevant_chunks), len(chunks),
    )

    if not relevant_chunks:
        logger.warning("Aucun chunk pertinent (tous sous le seuil %.2f)", min_score)
        return RetrievalResult(
            n_chunks_retrieved=len(chunks),
            below_threshold=True,
        )

    grouped = _group_chunks_by_incident(relevant_chunks)
    logger.info("Grouped into %d unique incidents", len(grouped))

    direct_incidents: list[RetrievedIncident] = []
    for incident_id, group in grouped.items():
        enriched = _enrich_incident_from_neo4j(neo4j, incident_id, group)
        if enriched:
            direct_incidents.append(enriched)

    expanded_incidents = _expand_via_graph(neo4j, direct_incidents, limit_per_seed=2)

    return RetrievalResult(
        incidents=direct_incidents + expanded_incidents,
        n_chunks_retrieved=len(chunks),
        n_incidents_direct=len(direct_incidents),
        n_incidents_expanded=len(expanded_incidents),
        below_threshold=False,
    )


def _group_chunks_by_incident(chunks: list[dict]) -> dict[str, dict]:
    """Regroupe les chunks par incident_id."""
    grouped: dict[str, dict] = {}

    for chunk in chunks:
        payload = chunk.get("payload", {})
        incident_id = payload.get("incident_id")
        if not incident_id:
            continue

        score = chunk.get("score", 0.0)
        field_canonical = payload.get("field_canonical", "unknown")

        if incident_id not in grouped:
            grouped[incident_id] = {
                "best_score": score,
                "matched_fields": [field_canonical],
            }
        else:
            if score > grouped[incident_id]["best_score"]:
                grouped[incident_id]["best_score"] = score
            if field_canonical not in grouped[incident_id]["matched_fields"]:
                grouped[incident_id]["matched_fields"].append(field_canonical)

    return grouped


def _enrich_incident_from_neo4j(
    neo4j: Neo4jClient,
    incident_id: str,
    group_data: dict,
) -> Optional[RetrievedIncident]:
    """Recupere proprietes et relations d'un incident depuis Neo4j."""
    cypher = """
    MATCH (i:Incident {incident_id: $incident_id})
    OPTIONAL MATCH (i)-[:EMIS_PAR]->(p:Personne)
    OPTIONAL MATCH (i)-[:CONCERNE]->(s:Societe)
    OPTIONAL MATCH (i)-[rel]->(r:Referentiel)
    RETURN
        i AS incident,
        collect(DISTINCT p) AS personnes,
        collect(DISTINCT s) AS societes,
        collect(DISTINCT {ref: r, relation: type(rel)}) AS referentiels
    """

    results = neo4j.run(cypher, incident_id=incident_id)
    if not results:
        logger.warning("Incident %s introuvable dans Neo4j", incident_id)
        return None

    row = results[0]
    incident_node = row["incident"]

    return RetrievedIncident(
        incident_id=incident_id,
        incident_id_source=incident_node.get("incident_id_source", ""),
        numero_fe=incident_node.get("numero_fe"),
        titre=incident_node.get("titre"),
        detail=incident_node.get("detail"),
        resume_llm=incident_node.get("resume_llm"),
        facteur_causal=incident_node.get("facteur_causal"),
        severite_percue=incident_node.get("severite_percue"),
        etat_final=incident_node.get("etat_final"),
        date_evenement=incident_node.get("date_evenement"),
        recolte_faits=incident_node.get("recolte_faits"),
        notes_suivi=incident_node.get("notes_suivi"),
        personnes=[dict(p) for p in row["personnes"] if p is not None],
        societes=[dict(s) for s in row["societes"] if s is not None],
        referentiels=_format_referentiels(row["referentiels"]),
        best_score=group_data["best_score"],
        matched_fields=group_data["matched_fields"],
        is_expanded=False,
    )


def _format_referentiels(raw: list) -> list[dict]:
    """Formate les referentiels avec leur type de relation."""
    formatted = []
    for item in raw:
        if not item or item.get("ref") is None:
            continue
        ref_node = item["ref"]
        formatted.append({
            "family": ref_node.get("family"),
            "code": ref_node.get("code"),
            "label": ref_node.get("label"),
            "relation": item.get("relation"),
        })
    return formatted


def _expand_via_graph(
    neo4j: Neo4jClient,
    seed_incidents: list[RetrievedIncident],
    limit_per_seed: int = 2,
) -> list[RetrievedIncident]:
    """Trouve des incidents voisins partageant des REFERENTIELS.

    Utilisation : l'expansion par personne a ete retiree car le partage
    d'emetteur ne cree pas de similarite metier pertinente. Seul le
    partage de referentiels (service, lieu, type, etc.) est considere.
    """
    if not seed_incidents:
        return []

    seed_ids = {inc.incident_id for inc in seed_incidents}
    expanded: dict[str, RetrievedIncident] = {}

    for seed in seed_incidents:
        neighbors = _find_neighbors_via_refs(neo4j, seed.incident_id, limit_per_seed)
        for neighbor_id in neighbors:
            if neighbor_id in seed_ids or neighbor_id in expanded:
                continue
            enriched = _enrich_incident_from_neo4j(
                neo4j,
                neighbor_id,
                {"best_score": 0.0, "matched_fields": ["graph_expansion"]},
            )
            if enriched:
                enriched.is_expanded = True
                expanded[neighbor_id] = enriched

    logger.info("Graph expansion added %d incidents (via referentiels)", len(expanded))
    return list(expanded.values())


def _find_neighbors_via_refs(
    neo4j: Neo4jClient,
    incident_id: str,
    limit: int = 2,
) -> list[str]:
    """Trouve les incidents partageant des Referentiels.

    Utilisation : ne considere que les Referentiels, pas les Personnes,
    pour eviter le bruit du partage d'emetteur.
    """
    cypher = """
    MATCH (i:Incident {incident_id: $incident_id})-[]->(shared:Referentiel)
    MATCH (shared)<-[]-(other:Incident)
    WHERE other.incident_id <> $incident_id
      AND NOT coalesce(other.is_test_data, false)
    RETURN other.incident_id AS neighbor_id, count(DISTINCT shared) AS connections
    ORDER BY connections DESC
    LIMIT $limit
    """

    results = neo4j.run(cypher, incident_id=incident_id, limit=limit)
    return [r["neighbor_id"] for r in results]
