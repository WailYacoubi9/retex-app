"""
Retrieval pour la voie incidents v2 (:IncidentSecu).

Stratégie (dense — défaut) :
  1. Embed la question avec bge-m3
  2. Recherche Qdrant scopée sur source_module=incident_securite_v2
  3. Filtre par score >= MIN_SCORE
  4. Regroupe par incident_id (best_score + matched_fields)
  5. Enrichit chaque incident depuis Neo4j (sac complet de propriétés + entités)

Mode HYBRIDE (hybrid=True et HYBRID_RETRIEVAL=1, cf. hybrid_retrieval.py) :
  ajoute un signal lexical BM25 (sur le texte des chunks) fusionné au dense par RRF,
  pour le RAPPEL. La précision reste au reranker en aval (reco/synthèse). Le lexical
  n'écarte jamais un candidat ; il n'admet un chunk que s'il partage assez de termes
  (plancher) — ce qui préserve l'abstention sur une requête hors-distribution.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from clients import Neo4jClient, OllamaClient, QdrantWrapper
from domain_synonyms import expand_terms, tokenize
from hybrid_retrieval import (
    BM25_SCORE_FLOOR, EXCLUDED_MATCH_FIELDS, HYBRID_RETRIEVAL, MIN_LEX_TERMS,
    TOP_K_BM25, TOP_K_DENSE, TOP_K_FUSED, get_index, rrf_fuse,
)

logger = logging.getLogger(__name__)

MIN_SCORE = 0.45
SOURCE_MODULE = "incident_securite_v2"


@dataclass
class RetrievedIncidentV2:
    incident_id: str
    props: dict
    entites: list[dict]
    best_score: float
    matched_fields: list[str] = field(default_factory=list)
    fused_score: float = 0.0        # ordre interne (RRF) — hybride
    lexical_only: bool = False      # remonté seulement par BM25 (cosine dense inconnue)


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
    hybrid: bool = False,
) -> RetrievalResultV2:
    use_hybrid = hybrid and HYBRID_RETRIEVAL and get_index() is not None
    question_vector = ollama.embed(question)

    if use_hybrid:
        return _retrieve_hybrid(question, question_vector, qdrant, neo4j)

    # --- Chemin dense (inchangé) ---------------------------------------------
    chunks = qdrant.search(
        vector=question_vector,
        top_k=top_k,
        exclude_test_data=True,
        source_module=SOURCE_MODULE,
    )
    logger.info("Qdrant returned %d chunks (module=%s)", len(chunks), SOURCE_MODULE)

    relevant = [c for c in chunks if c.get("score", 0.0) >= MIN_SCORE]
    logger.info("Chunks above threshold (%.2f): %d / %d", MIN_SCORE, len(relevant), len(chunks))

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


def _retrieve_hybrid(
    question: str,
    question_vector: list,
    qdrant: QdrantWrapper,
    neo4j: Neo4jClient,
) -> RetrievalResultV2:
    """Dense (rappel large) + BM25 lexical → fusion RRF → groupe → enrichit.
    Le reranker aval tranche la précision ; ici on maximise le rappel honnêtement."""
    dense_chunks = qdrant.search(
        vector=question_vector,
        top_k=TOP_K_DENSE,
        exclude_test_data=True,
        source_module=SOURCE_MODULE,
        exclude_fields=list(EXCLUDED_MATCH_FIELDS),  # matching problème↔problème
    )
    index = get_index()
    q_terms = expand_terms(tokenize(question))
    bm25_hits = index.search(q_terms, top_k=TOP_K_BM25)  # [(cid, score, n_overlap, meta)]

    dense_relevant = [c for c in dense_chunks if c.get("score", 0.0) >= MIN_SCORE]
    bm25_admitted = [h for h in bm25_hits
                     if h[2] >= MIN_LEX_TERMS and h[1] >= BM25_SCORE_FLOOR]
    logger.info("Hybride : dense≥%.2f=%d, bm25 admis=%d/%d (min_lex=%d)",
                MIN_SCORE, len(dense_relevant), len(bm25_admitted), len(bm25_hits), MIN_LEX_TERMS)

    # Abstention honnête : ni signal dense ni signal lexical suffisant.
    if not dense_relevant and not bm25_admitted:
        return RetrievalResultV2(n_chunks_retrieved=len(dense_chunks), below_threshold=True)

    # RRF sur les rangs des listes complètes ; pool = dense pertinents ∪ bm25 admis.
    fused = rrf_fuse([c["chunk_id"] for c in dense_chunks], [h[0] for h in bm25_hits])
    dense_by_id = {c["chunk_id"]: c for c in dense_chunks}
    bm25_by_id = {h[0]: h for h in bm25_hits}
    pool_ids = {c["chunk_id"] for c in dense_relevant} | {h[0] for h in bm25_admitted}

    chunks: list[dict] = []
    for cid in pool_ids:
        dc = dense_by_id.get(cid)
        if dc is not None:
            payload = dc.get("payload", {})
            dense_score = dc.get("score", 0.0)
        else:
            meta = bm25_by_id[cid][3]
            payload = {
                "incident_id": meta.get("incident_id"),
                "numero_fe": meta.get("numero_fe"),
                "field_canonical": meta.get("field_canonical", "unknown"),
            }
            dense_score = None  # lexical-only : cosine dense inconnue
        chunks.append({
            "payload": payload,
            "chunk_id": cid,
            "dense_score": dense_score,
            "fused_score": fused.get(cid, 0.0),
        })

    grouped = _group_by_incident(chunks)
    ordered = sorted(grouped.items(), key=lambda kv: -kv[1]["fused_score"])[:TOP_K_FUSED]
    items: list[RetrievedIncidentV2] = []
    for incident_id, group in ordered:
        enriched = _enrich(neo4j, incident_id, group)
        if enriched:
            items.append(enriched)

    return RetrievalResultV2(
        items=items,
        n_chunks_retrieved=len(dense_chunks),
        n_direct=len(items),
        below_threshold=False,
    )


def _group_by_incident(chunks: list[dict]) -> dict[str, dict]:
    """Regroupe des chunks par incident.
    - Chemin dense : chaque chunk a `score` (cosine) → best_score = max cosine.
    - Chemin hybride : chaque chunk a `dense_score` (None si lexical-only) +
      `fused_score` → best_score = meilleure cosine (0.0 si aucune), ordre = fused.
    Rétro-compatible : sans les clés hybrides, comportement d'origine préservé."""
    grouped: dict[str, dict] = {}
    for chunk in chunks:
        payload = chunk.get("payload", {})
        incident_id = payload.get("incident_id")
        if not incident_id:
            continue
        has_dense_key = "dense_score" in chunk
        dense_score = chunk.get("dense_score") if has_dense_key else chunk.get("score", 0.0)
        fused = chunk.get("fused_score", chunk.get("score", 0.0))
        dscore = dense_score if dense_score is not None else 0.0
        fc = payload.get("field_canonical", "unknown")
        g = grouped.get(incident_id)
        if g is None:
            grouped[incident_id] = {
                "best_score": dscore,
                "fused_score": fused,
                "matched_fields": [fc],
                "has_dense": dense_score is not None and dscore > 0.0,
            }
        else:
            if dscore > g["best_score"]:
                g["best_score"] = dscore
            if fused > g["fused_score"]:
                g["fused_score"] = fused
            if fc not in g["matched_fields"]:
                g["matched_fields"].append(fc)
            if dense_score is not None and dscore > 0.0:
                g["has_dense"] = True
    return grouped


def _enrich(
    neo4j: Neo4jClient,
    incident_id: str,
    group: dict,
) -> Optional[RetrievedIncidentV2]:
    # Allow-list de relations : les satellites ANALYTIQUES uniquement. Personne,
    # Notifiant (nominatif) n'entrent JAMAIS dans le contexte (culture juste,
    # cf. confidentialite.py — même précédent que le legacy retrieval.py).
    cypher = """
    MATCH (i:IncidentSecu {incident_id: $id})
    OPTIONAL MATCH (i)-[r]->(n)
    WHERE type(r) IN ['DE_TYPE','LOCALISE_EN','EN_PHASE_DE_VOL','IMPLIQUE_COMPAGNIE',
                      'IMPLIQUE_AERONEF','RESPONSABLE','A_POUR_FACTEUR','A_POUR_ACTION',
                      'CONCERNE','MIS_EN_CAUSE']
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
        fused_score=group.get("fused_score", group["best_score"]),
        lexical_only=not group.get("has_dense", True),
    )
