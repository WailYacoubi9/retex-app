"""
Voie agrégation pour les incidents v2.

Pipeline :
  NL → AggregationSpec (LLM structured output, vocabulaire fermé)
     → Cypher déterministe paramétré
     → exécution Neo4j
     → rows

Le LLM ne génère pas de Cypher et ne calcule pas de chiffres.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, ValidationError

import prompt_store
from clients import Neo4jClient, OllamaClient

logger = logging.getLogger(__name__)

LLM_MODEL = "qwen2.5:7b"


# ─── Spec contrainte (vocabulaire fermé) ─────────────────────────────────────

class AggregationSpec(BaseModel):
    metric: Literal["count"] = "count"
    group_by: Optional[Literal[
        "severite", "classification", "traitement_termine",
        "condition_lumineuse", "actions_efficaces", "annee", "mois"
    ]] = None
    f_severite: Optional[Literal[
        "1 - faible", "2 - tolérable", "3 - important", "4 - élevé", "5 - intolérable"
    ]] = None
    f_classification: Optional[Literal[
        "Incident", "Occurence sans effet sur la sécurité", "Incident sérieux", "Accident"
    ]] = None
    f_condition_lumineuse: Optional[Literal["Jour", "Nuit"]] = None
    f_traitement_termine: Optional[bool] = None
    f_actions_efficaces: Optional[bool] = None
    f_avec_action_chaud: Optional[bool] = None  # présence d'une action corrective immédiate
    f_annee: Optional[int] = None
    f_mois: Optional[str] = None
    order: Literal["desc", "asc"] = "desc"
    limit: int = Field(default=10, ge=1)





def parse_question_to_spec(question: str, ollama: OllamaClient) -> Optional[AggregationSpec]:
    """Convertit une question NL en AggregationSpec via structured output."""
    schema = AggregationSpec.model_json_schema()
    prompt = prompt_store.rendre("agregation.parseur", question=question)

    for attempt in range(2):
        try:
            raw = ollama.generate_structured(prompt, schema, model=LLM_MODEL)
            spec = AggregationSpec.model_validate_json(raw)
            logger.info("Spec parsed (attempt %d): %s", attempt + 1, spec.model_dump())
            return spec
        except (ValidationError, json.JSONDecodeError, Exception) as e:
            logger.warning("Spec parse failed (attempt %d): %s", attempt + 1, e)

    return None


def _is_degenerate(spec: AggregationSpec) -> bool:
    """Retourne True si la spec ne contient aucun critère (pas une agrégation)."""
    return (
        spec.group_by is None
        and spec.f_severite is None
        and spec.f_classification is None
        and spec.f_condition_lumineuse is None
        and spec.f_traitement_termine is None
        and spec.f_actions_efficaces is None
        and spec.f_avec_action_chaud is None
        and spec.f_annee is None
        and spec.f_mois is None
    )


# ─── Construction Cypher déterministe ────────────────────────────────────────

_DIRECT_FIELDS = {
    "severite", "classification", "traitement_termine", "condition_lumineuse",
    "actions_efficaces",
}


def build_cypher(spec: AggregationSpec) -> tuple[str, dict[str, Any]]:
    """Construit le Cypher paramétré à partir de la spec. Déterministe."""
    where_clauses = ["coalesce(i.is_test_data, false) = false"]
    params: dict[str, Any] = {}

    if spec.f_severite is not None:
        where_clauses.append("i.severite = $sev")
        params["sev"] = spec.f_severite

    if spec.f_classification is not None:
        where_clauses.append("i.classification = $cls")
        params["cls"] = spec.f_classification

    if spec.f_condition_lumineuse is not None:
        where_clauses.append("i.condition_lumineuse = $cl")
        params["cl"] = spec.f_condition_lumineuse

    if spec.f_traitement_termine is not None:
        where_clauses.append("i.traitement_termine = $tt")
        params["tt"] = spec.f_traitement_termine

    if spec.f_actions_efficaces is not None:
        where_clauses.append("i.actions_efficaces = $eff")
        params["eff"] = spec.f_actions_efficaces

    if spec.f_avec_action_chaud is not None:
        where_clauses.append(
            "i.action_corrective IS NOT NULL" if spec.f_avec_action_chaud
            else "i.action_corrective IS NULL"
        )

    if spec.f_annee is not None:
        where_clauses.append("i.date_evenement STARTS WITH $annee")
        params["annee"] = str(spec.f_annee)

    if spec.f_mois is not None:
        where_clauses.append("i.date_evenement STARTS WITH $mois")
        params["mois"] = spec.f_mois

    where_str = " AND ".join(where_clauses)

    if spec.group_by is None:
        cypher = f"MATCH (i:IncidentSecu) WHERE {where_str} RETURN count(*) AS n"
    elif spec.group_by in _DIRECT_FIELDS:
        where_str += f" AND i.{spec.group_by} IS NOT NULL"
        cypher = (
            f"MATCH (i:IncidentSecu) WHERE {where_str} "
            f"RETURN i.{spec.group_by} AS label, count(*) AS n "
            f"ORDER BY n {spec.order.upper()} LIMIT $limit"
        )
        params["limit"] = spec.limit
    elif spec.group_by == "annee":
        cypher = (
            f"MATCH (i:IncidentSecu) WHERE {where_str} AND i.date_evenement IS NOT NULL "
            f"RETURN substring(i.date_evenement, 0, 4) AS label, count(*) AS n "
            f"ORDER BY n {spec.order.upper()} LIMIT $limit"
        )
        params["limit"] = spec.limit
    elif spec.group_by == "mois":
        cypher = (
            f"MATCH (i:IncidentSecu) WHERE {where_str} AND i.date_evenement IS NOT NULL "
            f"RETURN substring(i.date_evenement, 0, 7) AS label, count(*) AS n "
            f"ORDER BY n {spec.order.upper()} LIMIT $limit"
        )
        params["limit"] = spec.limit
    else:
        cypher = f"MATCH (i:IncidentSecu) WHERE {where_str} RETURN count(*) AS n"

    return cypher, params


# ─── Point d'entrée ──────────────────────────────────────────────────────────

def run_aggregation(
    question: str,
    ollama: OllamaClient,
    neo4j: Neo4jClient,
) -> dict:
    """
    Retourne un dict avec status, spec, cypher, rows, total.
    status = "not_aggregation" | "ok".
    """
    spec = parse_question_to_spec(question, ollama)

    if spec is None or _is_degenerate(spec):
        return {"status": "not_aggregation", "spec": None, "cypher": None, "rows": [], "total": None}

    cypher, params = build_cypher(spec)
    logger.info("Cypher: %s | params: %s", cypher, params)

    rows_raw = neo4j.run(cypher, **params)

    if spec.group_by is None:
        total = rows_raw[0]["n"] if rows_raw else 0
        rows = []
    else:
        total = sum(r["n"] for r in rows_raw)
        rows = [{"label": str(r["label"]), "n": r["n"]} for r in rows_raw]

    return {
        "status": "ok",
        "spec": spec,
        "cypher": cypher,
        "rows": rows,
        "total": total,
        "filters_applied": {
            k: v for k, v in spec.model_dump().items()
            if k.startswith("f_") and v is not None
        },
    }
