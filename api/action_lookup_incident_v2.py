"""
Voie actions pour les incidents v2.

Pipeline :
  NL → ActionSpec (LLM structured output)
     → Cypher déterministe paramétré
     → exécution Neo4j
     → résultats (liste d'actions avec incident parent)

Couvre les questions du type :
  - "Quelles actions correctives ont été prises pour les incidents FOD ?"
  - "Quelles actions préventives sont encore en cours ?"
  - "Quelles actions a prises storkhani ?"
  - "Combien d'actions clôturées en 2025 ?"
"""
from __future__ import annotations

import json
import logging
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, ValidationError

import prompt_store
from clients import Neo4jClient, OllamaClient
from field_catalog import module_meta

logger = logging.getLogger(__name__)

LLM_MODEL = "qwen2.5:7b"


# ─── Spec contrainte ─────────────────────────────────────────────────────────

class ActionSpec(BaseModel):
    question_type: Literal["liste", "count"] = "liste"
    type_action: Optional[Literal["corrective", "préventive", "curative"]] = None
    f_statut: Optional[Literal["clôturé", "en cours"]] = None
    f_responsable: Optional[str] = None          # login partiel ou complet
    f_titre_action_keyword: Optional[str] = None  # mot-clé dans le titre de l'action
    f_titre_incident_keyword: Optional[str] = None # mot-clé dans le titre de l'incident
    f_annee_ajout: Optional[int] = None
    f_severite_incident: Optional[Literal[
        "1 - faible", "2 - tolérable", "3 - important", "4 - élevé", "5 - intolérable"
    ]] = None
    f_actions_efficaces: Optional[bool] = None   # jugement d'efficacité (niveau fiche)
    f_avec_action_chaud: Optional[bool] = None   # présence d'une action corrective immédiate (fiche)
    limit: int = Field(default=20, ge=1, le=50)





def parse_question_to_spec(question: str, ollama: OllamaClient,
                           exemples_module: list[str] | None = None) -> Optional[ActionSpec]:
    schema = ActionSpec.model_json_schema()
    exemples = "\n".join(f"- {e}" for e in (exemples_module or [])) or "- (aucun)"
    prompt = prompt_store.rendre("actions.parseur",
                                 question=question, exemples_module=exemples)
    for attempt in range(2):
        try:
            raw = ollama.generate_structured(prompt, schema, model=LLM_MODEL)
            spec = ActionSpec.model_validate_json(raw)
            logger.info("ActionSpec (attempt %d): %s", attempt + 1, spec.model_dump())
            return spec
        except (ValidationError, json.JSONDecodeError, Exception) as e:
            logger.warning("ActionSpec parse failed (attempt %d): %s", attempt + 1, e)
    return None


# ─── Cypher déterministe ─────────────────────────────────────────────────────

# Une action est "clôturée" si statut = '100' OU si une date de clôture existe
# (l'export contient des statuts absents ou aberrants — des dates à la place).
_CLOTUREE = "(a.statut = '100' OR a.date_cloture IS NOT NULL)"


def _has_action_filters(spec: ActionSpec) -> bool:
    """Filtres qui portent sur les nœuds Action (imposent la forme relationnelle)."""
    return any([
        spec.type_action, spec.f_statut, spec.f_responsable,
        spec.f_titre_action_keyword, spec.f_annee_ajout,
    ])


def _has_any_filter(spec: ActionSpec) -> bool:
    return _has_action_filters(spec) or any([
        spec.f_titre_incident_keyword, spec.f_severite_incident,
        spec.f_actions_efficaces is not None, spec.f_avec_action_chaud is not None,
    ])


def build_cypher_chaud(spec: ActionSpec) -> tuple[str, str, dict[str, Any]]:
    """
    Forme incident-centrée pour les questions sur l'action à chaud
    (champ `action_corrective` de la fiche — la majorité des incidents avec
    action à chaud n'ont AUCUN nœud Action structuré).
    """
    where: list[str] = ["coalesce(i.is_test_data, false) = false"]
    params: dict[str, Any] = {}

    where.append("i.action_corrective IS NOT NULL" if spec.f_avec_action_chaud
                 else "i.action_corrective IS NULL")

    if spec.f_titre_incident_keyword:
        where.append("toLower(i.titre) CONTAINS toLower($titre_inc_kw)")
        params["titre_inc_kw"] = spec.f_titre_incident_keyword

    if spec.f_severite_incident:
        where.append("i.severite = $severite")
        params["severite"] = spec.f_severite_incident

    if spec.f_actions_efficaces is not None:
        where.append("i.actions_efficaces = $efficaces")
        params["efficaces"] = spec.f_actions_efficaces

    base = f"MATCH (i:IncidentSecu) WHERE {' AND '.join(where)}"
    cypher_count = f"{base} RETURN count(*) AS n"
    params["limit"] = spec.limit
    cypher_liste = (
        f"{base} "
        f"RETURN i.numero_fe AS fe, i.titre AS titre_incident, i.severite AS severite, "
        f"i.actions_efficaces AS actions_efficaces, "
        f"left(i.action_corrective, 250) AS titre_action, "
        f"'immédiate' AS type_action, "
        f"null AS statut, null AS responsable, null AS date_prevue, null AS date_cloture "
        f"ORDER BY i.date_evenement DESC "
        f"LIMIT $limit"
    )
    return cypher_liste, cypher_count, params


def build_cypher(spec: ActionSpec) -> tuple[str, str, dict[str, Any]]:
    """Retourne (cypher_liste, cypher_count, params). Déterministe."""
    where: list[str] = ["coalesce(i.is_test_data, false) = false"]
    params: dict[str, Any] = {}

    if spec.type_action is not None:
        where.append("r.type_action = $type_action")
        params["type_action"] = spec.type_action

    if spec.f_statut == "clôturé":
        where.append(_CLOTUREE)
    elif spec.f_statut == "en cours":
        where.append(f"NOT {_CLOTUREE}")

    if spec.f_responsable:
        where.append("toLower(coalesce(a.responsable, '')) CONTAINS toLower($responsable)")
        params["responsable"] = spec.f_responsable

    if spec.f_titre_action_keyword:
        where.append("toLower(a.titre_action) CONTAINS toLower($titre_action_kw)")
        params["titre_action_kw"] = spec.f_titre_action_keyword

    if spec.f_titre_incident_keyword:
        where.append("toLower(i.titre) CONTAINS toLower($titre_inc_kw)")
        params["titre_inc_kw"] = spec.f_titre_incident_keyword

    if spec.f_annee_ajout:
        # date_ajout stockée en ISO (AAAA-MM-JJ) par ingest_actions.py
        where.append("a.date_ajout STARTS WITH $annee_ajout")
        params["annee_ajout"] = str(spec.f_annee_ajout)

    if spec.f_severite_incident:
        where.append("i.severite = $severite")
        params["severite"] = spec.f_severite_incident

    if spec.f_actions_efficaces is not None:
        where.append("i.actions_efficaces = $efficaces")
        params["efficaces"] = spec.f_actions_efficaces

    if spec.f_avec_action_chaud is not None:
        where.append("i.action_corrective IS NOT NULL" if spec.f_avec_action_chaud
                     else "i.action_corrective IS NULL")

    where_str = " AND ".join(where)
    base = f"MATCH (i:IncidentSecu)-[r:A_POUR_ACTION]->(a:Action) WHERE {where_str}"

    cypher_count = f"{base} RETURN count(*) AS n"
    params["limit"] = spec.limit
    cypher_liste = (
        f"{base} "
        f"RETURN i.numero_fe AS fe, i.titre AS titre_incident, i.severite AS severite, "
        f"i.actions_efficaces AS actions_efficaces, "
        f"a.titre_action AS titre_action, r.type_action AS type_action, "
        f"a.statut AS statut, a.responsable AS responsable, "
        f"a.date_prevue AS date_prevue, a.date_cloture AS date_cloture "
        f"ORDER BY a.date_ajout DESC "
        f"LIMIT $limit"
    )

    return cypher_liste, cypher_count, params


# ─── Point d'entrée ──────────────────────────────────────────────────────────

def run_action_lookup(
    question: str,
    ollama: OllamaClient,
    neo4j: Neo4jClient,
) -> dict:
    """
    Retourne un dict avec status, spec, rows, total.
    status = "not_action_query" | "ok".
    """
    exemples = module_meta(neo4j).get("exemples") or []
    spec = parse_question_to_spec(question, ollama, exemples_module=exemples)

    if spec is None:
        return {"status": "not_action_query", "spec": None, "rows": [], "total": None}

    # Forme incident-centrée pour l'action à chaud (sauf si des filtres
    # portant sur les nœuds Action imposent la forme relationnelle).
    if spec.f_avec_action_chaud is not None and not _has_action_filters(spec):
        cypher_liste, cypher_count, params = build_cypher_chaud(spec)
    else:
        cypher_liste, cypher_count, params = build_cypher(spec)
    logger.info("Action Cypher: %s | params: %s", cypher_liste, params)

    # Le total est TOUJOURS le vrai count (jamais la taille de page)
    count_params = {k: v for k, v in params.items() if k != "limit"}
    total_raw = neo4j.run(cypher_count, **count_params)
    total = total_raw[0]["n"] if total_raw else 0

    if spec.question_type == "count":
        return {"status": "ok", "spec": spec, "rows": [], "total": total}

    # Garde-fou : une liste sans AUCUN critère reconnu = question mal comprise.
    # On refuse de déverser toute la base en la faisant passer pour une réponse.
    if not _has_any_filter(spec):
        return {"status": "no_filters", "spec": spec, "rows": [], "total": total}

    rows_raw = neo4j.run(cypher_liste, **params)
    rows = [
        {
            "fe":                r.get("fe"),
            "titre_incident":    r.get("titre_incident"),
            "severite":          r.get("severite"),
            "titre_action":      r.get("titre_action"),
            "type_action":       r.get("type_action"),
            # L'action à chaud n'a pas de cycle de vie : statut sans objet
            "statut":            (None if r.get("type_action") == "immédiate"
                                  else ("clôturé"
                                        if r.get("statut") == "100" or r.get("date_cloture")
                                        else "en cours")),
            "responsable":       r.get("responsable"),
            "date_prevue":       r.get("date_prevue"),
            "date_cloture":      r.get("date_cloture"),
            "actions_efficaces": r.get("actions_efficaces"),
        }
        for r in rows_raw
    ]
    return {"status": "ok", "spec": spec, "rows": rows, "total": total}
