from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

from clients import Neo4jClient, OllamaClient, QdrantWrapper

logger = logging.getLogger(__name__)

SOURCE_MODULE = "intraknow_tickets"
MIN_SCORE_THRESHOLD = 0.50
# Seuil de pertinence pour CONSERVER un ticket ajoute par expansion graphe.
# Plus bas que MIN_SCORE_THRESHOLD car ce sont des resultats de contexte
# secondaire, mais assez haut pour que l'expansion "se desactive" quand le
# voisin n'a rien a voir avec la question.
EXPANSION_MIN_RELEVANCE = 0.45


def _cosine(a: list[float], b: list[float]) -> float:
    """Similarite cosinus entre deux vecteurs."""
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


@dataclass
class RetrievedTicket:
    ticket_id: str
    numero_fe: Optional[str] = None
    titre: Optional[str] = None
    detail: Optional[str] = None
    type_nc: Optional[str] = None
    importance: Optional[str] = None
    etat: Optional[str] = None
    etape_label: Optional[str] = None
    site_application: Optional[str] = None
    projet_id: Optional[str] = None
    projet_nom: Optional[str] = None
    client: Optional[str] = None
    structure: Optional[str] = None
    urgence: Optional[str] = None
    individu: Optional[str] = None
    date_nc: Optional[str] = None
    llm_resume: Optional[str] = None
    llm_domaine_technique: Optional[str] = None
    best_score: float = 0.0
    matched_fields: list[str] = field(default_factory=list)
    is_expanded: bool = False


@dataclass
class RetrievalResultTickets:
    items: list[RetrievedTicket] = field(default_factory=list)
    n_chunks_retrieved: int = 0
    n_direct: int = 0
    n_expanded: int = 0
    below_threshold: bool = False


def retrieve_tickets(
    question: str,
    ollama: OllamaClient,
    qdrant: QdrantWrapper,
    neo4j: Neo4jClient,
    top_k: int = 5,
    min_score: float = MIN_SCORE_THRESHOLD,
) -> RetrievalResultTickets:
    question_vector = ollama.embed(question)

    chunks = qdrant.search(
        vector=question_vector,
        top_k=top_k,
        exclude_test_data=True,
        source_module=SOURCE_MODULE,
    )
    logger.info("Qdrant returned %d ticket chunks", len(chunks))

    relevant = [c for c in chunks if c.get("score", 0.0) >= min_score]
    logger.info("Chunks above threshold: %d / %d", len(relevant), len(chunks))

    if not relevant:
        return RetrievalResultTickets(n_chunks_retrieved=len(chunks), below_threshold=True)

    grouped = _group_chunks_by_ticket(relevant)
    logger.info("Grouped into %d unique Ticket", len(grouped))

    direct: list[RetrievedTicket] = []
    for ticket_id, group_data in grouped.items():
        item = _fetch_ticket_from_neo4j(neo4j, ticket_id, group_data)
        if item:
            direct.append(item)

    expanded = _expand_via_graph(neo4j, direct, ollama, question_vector)

    return RetrievalResultTickets(
        items=direct + expanded,
        n_chunks_retrieved=len(chunks),
        n_direct=len(direct),
        n_expanded=len(expanded),
        below_threshold=False,
    )


def _group_chunks_by_ticket(chunks: list[dict]) -> dict[str, dict]:
    grouped: dict[str, dict] = {}
    for chunk in chunks:
        payload = chunk.get("payload", {})
        ticket_id = payload.get("ticket_id")
        if not ticket_id:
            continue
        score = chunk.get("score", 0.0)
        field_name = payload.get("field_canonical", "unknown")
        if ticket_id not in grouped:
            grouped[ticket_id] = {"best_score": score, "matched_fields": [field_name]}
        else:
            if score > grouped[ticket_id]["best_score"]:
                grouped[ticket_id]["best_score"] = score
            if field_name not in grouped[ticket_id]["matched_fields"]:
                grouped[ticket_id]["matched_fields"].append(field_name)
    return grouped


def _fetch_ticket_from_neo4j(
    neo4j: Neo4jClient,
    ticket_id: str,
    group_data: dict,
) -> Optional[RetrievedTicket]:
    cypher = """
    MATCH (t:Ticket {ticket_id: $ticket_id})
    OPTIONAL MATCH (t)-[:DANS_PROJET]->(:Projet)-[:POUR_CLIENT]->(c:Client)
    RETURN t AS ticket_node, c.nom AS client
    """
    results = neo4j.run(cypher, ticket_id=ticket_id)
    if not results:
        logger.warning("Ticket %s introuvable dans Neo4j", ticket_id)
        return None

    node = results[0]["ticket_node"]
    client = results[0].get("client")
    return RetrievedTicket(
        ticket_id=ticket_id,
        numero_fe=node.get("numero_fe"),
        titre=node.get("titre"),
        detail=node.get("detail"),
        type_nc=node.get("type_nc"),
        importance=node.get("importance"),
        etat=node.get("etat"),
        etape_label=node.get("etape_label"),
        site_application=node.get("site_application"),
        projet_id=node.get("projet_id"),
        projet_nom=node.get("projet_nom"),
        client=client,
        structure=node.get("structure"),
        urgence=node.get("urgence"),
        individu=node.get("individu"),
        date_nc=node.get("date_nc"),
        llm_resume=node.get("llm_resume"),
        llm_domaine_technique=node.get("llm_domaine_technique"),
        best_score=group_data["best_score"],
        matched_fields=group_data["matched_fields"],
        is_expanded=False,
    )


# Clause commune : voisin valide, distinct, enrichi
_NEIGHBOR_FILTER = (
    "neighbor.ticket_id <> $ticket_id "
    "AND NOT coalesce(neighbor.is_test_data, false) "
    "AND neighbor.llm_resume IS NOT NULL"
)

# Relations d'expansion, en ORDRE DE PRIORITE (la plus pertinente d'abord).
# Chaque voisin est etiquete par sa relation (matched_fields) pour que la
# generation puisse expliquer POURQUOI le ticket est lie.
EXPANSION_RELATIONS: list[tuple[str, str]] = [
    # Hierarchie de tickets : parent / enfant directs
    ("via_enfant", f"""
        MATCH (t:Ticket {{ticket_id: $ticket_id}})-[:ENFANT_DE]-(neighbor:Ticket)
        WHERE {_NEIGHBOR_FILTER}
        RETURN DISTINCT neighbor.ticket_id AS neighbor_id, neighbor.date_nc AS date_nc
        ORDER BY date_nc DESC LIMIT $limit
    """),
    # Meme intervenant (expert qui a traite un cas similaire)
    ("via_expert", f"""
        MATCH (t:Ticket {{ticket_id: $ticket_id}})-[:TRAITE_PAR]->(:Personne)<-[:TRAITE_PAR]-(neighbor:Ticket)
        WHERE {_NEIGHBOR_FILTER}
        RETURN DISTINCT neighbor.ticket_id AS neighbor_id, neighbor.date_nc AS date_nc
        ORDER BY date_nc DESC LIMIT $limit
    """),
    # Meme projet, ou projet parent / sous-projet
    ("via_projet", f"""
        MATCH (t:Ticket {{ticket_id: $ticket_id}})-[:DANS_PROJET]->(p:Projet)
        MATCH (neighbor:Ticket)-[:DANS_PROJET]->(p2:Projet)
        WHERE (p2 = p OR (p)-[:SOUS_PROJET_DE]-(p2)) AND {_NEIGHBOR_FILTER}
        RETURN DISTINCT neighbor.ticket_id AS neighbor_id, neighbor.date_nc AS date_nc
        ORDER BY date_nc DESC LIMIT $limit
    """),
    # Meme client (autres projets du meme client)
    ("via_client", f"""
        MATCH (t:Ticket {{ticket_id: $ticket_id}})-[:DANS_PROJET]->(:Projet)-[:POUR_CLIENT]->(c:Client)
        MATCH (neighbor:Ticket)-[:DANS_PROJET]->(:Projet)-[:POUR_CLIENT]->(c)
        WHERE {_NEIGHBOR_FILTER}
        RETURN DISTINCT neighbor.ticket_id AS neighbor_id, neighbor.date_nc AS date_nc
        ORDER BY date_nc DESC LIMIT $limit
    """),
]


def _expand_via_graph(
    neo4j: Neo4jClient,
    seeds: list[RetrievedTicket],
    ollama: OllamaClient,
    question_vector: list[float],
    per_relation: int = 2,
    total_cap: int = 6,
    min_relevance: float = EXPANSION_MIN_RELEVANCE,
) -> list[RetrievedTicket]:
    """Expansion graphe multi-relations, etiquetee, filtree par pertinence.

    1. Part UNIQUEMENT des seeds (resultats directs au-dessus du seuil) -> pas
       de derive depuis un mauvais match.
    2. Parcourt les relations par priorite ; chaque voisin garde sa relation
       (matched_fields=["via_enfant"|"via_expert"|"via_projet"|"via_client"]).
    3. FILTRE DE PERTINENCE : re-score le resume du voisin contre la question
       et ne garde que ceux >= min_relevance. L'expansion "se desactive" donc
       d'elle-meme quand le voisin n'a rien a voir avec la question.
    Plafonne a total_cap voisins retenus.
    """
    if not seeds:
        return []

    seed_ids = {s.ticket_id for s in seeds}
    expanded: dict[str, RetrievedTicket] = {}
    seen: set[str] = set()  # candidats deja evalues (evite de re-embedder)

    for rel_tag, cypher in EXPANSION_RELATIONS:
        if len(expanded) >= total_cap:
            break
        for seed in seeds:
            if len(expanded) >= total_cap:
                break
            results = neo4j.run(cypher, ticket_id=seed.ticket_id, limit=per_relation)
            for row in results:
                neighbor_id = row["neighbor_id"]
                if neighbor_id in seed_ids or neighbor_id in expanded or neighbor_id in seen:
                    continue
                seen.add(neighbor_id)

                item = _fetch_ticket_from_neo4j(
                    neo4j, neighbor_id, {"best_score": 0.0, "matched_fields": [rel_tag]}
                )
                if not item:
                    continue

                # Filtre de pertinence : le voisin doit etre proche de la question
                texte = item.llm_resume or item.titre or ""
                if not texte:
                    continue
                relevance = _cosine(question_vector, ollama.embed(texte))
                if relevance < min_relevance:
                    continue

                item.is_expanded = True
                item.best_score = relevance  # score reel (utile pour le tri/affichage)
                expanded[neighbor_id] = item
                if len(expanded) >= total_cap:
                    break

    logger.info("Graph expansion : %d tickets retenus (filtre pertinence >= %.2f)",
                len(expanded), min_relevance)
    return list(expanded.values())
