"""
DÉPRÉCIÉ — ce module n'est plus appelé par les routes FastAPI.

La logique de liste est désormais gérée par run_unified_query()
dans query_engine_incident_v2.py (moteur unifié count/repartition/liste).
La route POST /ask/incident-v2/list délègue directement à ce moteur.
"""
from __future__ import annotations

import logging
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from clients import Neo4jClient, OllamaClient

logger = logging.getLogger(__name__)

_ALLOWED_FIELDS = {
    "numero_fe", "titre", "severite", "classification", "etat",
    "date_evenement", "date_creation", "action_corrective", "resume_llm",
}
_DEFAULT_FIELDS = ["numero_fe", "titre", "severite", "date_creation", "action_corrective"]


# ─── Spec contrainte ─────────────────────────────────────────────────────────

class StructuredListSpec(BaseModel):
    question_type: Literal["liste", "count"] = "liste"
    f_severite: Optional[Literal[
        "1 - faible", "2 - tolérable", "3 - important", "4 - élevé", "5 - intolérable"
    ]] = None
    f_classification: Optional[Literal[
        "Incident", "Occurence sans effet sur la sécurité", "Incident sérieux", "Accident"
    ]] = None
    f_condition_lumineuse: Optional[Literal["Jour", "Nuit"]] = None
    f_traitement_termine: Optional[bool] = None
    f_annee: Optional[int] = None
    f_mois: Optional[str] = None
    sort_by: Literal["date_creation", "date_evenement", "severite"] = "date_creation"
    order: Literal["desc", "asc"] = "desc"
    limit: int = Field(default=5, ge=1, le=50)
    fields: list[Literal[
        "numero_fe", "titre", "severite", "classification", "etat",
        "date_evenement", "date_creation", "action_corrective", "resume_llm"
    ]] = Field(default_factory=lambda: list(_DEFAULT_FIELDS))


# ─── Cypher déterministe (conservé pour référence, non appelé) ───────────────

def build_list_cypher(spec: StructuredListSpec) -> tuple[str, dict[str, Any]]:
    where: list[str] = ["coalesce(i.is_test_data, false) = false"]
    params: dict[str, Any] = {}

    if spec.f_severite is not None:
        where.append("i.severite = $sev")
        params["sev"] = spec.f_severite

    if spec.f_classification is not None:
        where.append("i.classification = $cls")
        params["cls"] = spec.f_classification

    if spec.f_condition_lumineuse is not None:
        where.append("i.condition_lumineuse = $cl")
        params["cl"] = spec.f_condition_lumineuse

    if spec.f_traitement_termine is not None:
        where.append("i.traitement_termine = $tt")
        params["tt"] = spec.f_traitement_termine

    if spec.f_annee is not None:
        where.append("i.date_evenement STARTS WITH $annee")
        params["annee"] = str(spec.f_annee)

    if spec.f_mois is not None:
        where.append("i.date_evenement STARTS WITH $mois")
        params["mois"] = spec.f_mois

    where_str = " AND ".join(where)

    safe_fields = [f for f in spec.fields if f in _ALLOWED_FIELDS]
    if not safe_fields:
        safe_fields = list(_DEFAULT_FIELDS)
    returns = ", ".join(f"i.{f} AS {f}" for f in safe_fields)

    params["limit"] = spec.limit
    cypher = (
        f"MATCH (i:IncidentSecu) WHERE {where_str} "
        f"RETURN {returns} "
        f"ORDER BY i.{spec.sort_by} {spec.order.upper()} "
        f"LIMIT $limit"
    )
    return cypher, params


# ─── Point d'entrée ──────────────────────────────────────────────────────────

def run_list_query(
    question: str,
    ollama: OllamaClient,
    neo4j: Neo4jClient,
) -> dict:
    """
    Retourne un dict avec status, spec, records.
    status = "not_list_query" | "ok".
    """
    spec = parse_question_to_list_spec(question, ollama)

    if spec is None:
        return {"status": "not_list_query", "spec": None, "records": []}

    if spec.question_type == "count":
        return {"status": "not_list_query", "spec": spec, "records": []}

    cypher, params = build_list_cypher(spec)
    logger.info("List Cypher: %s | params: %s", cypher, params)

    rows = neo4j.run(cypher, **params)
    records = [dict(r) for r in rows]

    return {"status": "ok", "spec": spec, "records": records}
