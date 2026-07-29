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
  - "Combien d'actions clôturées en 2025 ?"

Culture juste : une question rattachée à une personne (« les actions de X ») est
DÉTECTÉE (f_responsable) puis REFUSÉE explicitement — jamais filtrée ni ignorée.
"""
from __future__ import annotations

import json
import logging
import os
import re
from datetime import date as _date
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, ValidationError

import prompt_store
from clients import Neo4jClient, OllamaClient
from field_catalog import module_meta

logger = logging.getLogger(__name__)

LLM_MODEL = os.environ.get("PLANNER_MODEL", "qwen2.5:14b")  # parseur = planification


# ─── Spec contrainte ─────────────────────────────────────────────────────────

class ActionSpec(BaseModel):
    question_type: Literal["liste", "count"] = "liste"
    type_action: Optional[Literal["corrective", "préventive", "curative"]] = None
    f_statut: Optional[Literal["clôturé", "en cours"]] = None
    f_en_retard: Optional[bool] = None           # échéance dépassée ET non clôturée
    f_responsable: Optional[str] = None          # login partiel ou complet
    f_titre_action_keyword: Optional[str] = None  # mot-clé dans le titre de l'action
    f_titre_incident_keyword: Optional[str] = None # mot-clé dans le titre de l'incident
    f_annee_ajout: Optional[int] = None            # la VALEUR de l'année (nom hérité)
    f_annee_champ: Optional[Literal["ajout", "cloture", "prevue"]] = None  # à quel champ-date l'année se rattache
    f_severite_incident: Optional[Literal[
        "1 - faible", "2 - tolérable", "3 - important", "4 - élevé", "5 - intolérable"
    ]] = None
    f_actions_efficaces: Optional[bool] = None   # jugement d'efficacité (niveau fiche)
    f_avec_action_chaud: Optional[bool] = None   # présence d'une action corrective immédiate (fiche)
    limit: int = Field(default=20, ge=1, le=50)
    # ─── Opérateur de GRAPHE (aligné sur UnifiedQuerySpec) : « actions partagées
    # par ≥N incidents », « actions récurrentes », etc. Compilé par graph_query. ───
    graph_pattern: Optional[Literal["degree", "shared"]] = None
    graph_anchor: Optional[Literal[
        "incident", "action", "compagnie", "type_evenement", "lieu",
        "phase_vol", "aeronef", "service", "societe", "entite"]] = None
    graph_via: Optional[Literal[
        "incident", "action", "compagnie", "type_evenement", "lieu",
        "phase_vol", "aeronef", "service", "societe", "entite"]] = None
    graph_op: Literal[">=", ">", "=", "<=", "<"] = ">="
    graph_n: int = 2
    graph_anchors_fe: list[str] = Field(default_factory=list)





# Traitement déterministe et ROBUSTE du statut (prime sur le 7b dilué).
# Ordre impératif : retard → en cours (incl. négation d'une clôture) → clôturé.
# La négation est testée AVANT le mot de clôture pour ne jamais INVERSER
# ("pas encore terminées" = en cours, pas clôturé).
_CLOS_WORD = r"(?:clôtur|cloturé|terminé|termine\b|réalisé|realise\b|finalisé|" \
             r"\bfinies?\b|\bfini\b|fermé|ferme\b|soldé|achev)"
_RETARD_RE = re.compile(
    r"en\s+retard|échéance[^.,;!?]{0,15}?(?:dépass|pass|expir|échu)|"
    r"délai[^.,;!?]{0,15}?dépass|hors\s+délai|en\s+souffrance|\bpérimé|\béchu", re.IGNORECASE)
_EN_COURS_RE = re.compile(
    r"ouvert|en\s+cours|en\s+suspens|en\s+attente|à\s+faire|à\s+traiter|"
    r"à\s+finaliser|planifié|non\s+traité|en\s+attente", re.IGNORECASE)
# négation (ne…pas / n'… / pas / non / sans / aucune / jamais) suivie, à courte
# distance, d'un mot de clôture → « en cours »
_NEG_CLOS_RE = re.compile(
    r"(?:\bne\b|\bn'|\bpas\b|\bnon\b|\bsans\b|aucune?|jamais)[^.,;!?]{0,25}?" + _CLOS_WORD,
    re.IGNORECASE)
_CLOS_RE = re.compile(_CLOS_WORD, re.IGNORECASE)


_PREVUE_RE = re.compile(r"prévue?s?|échéance|à\s+réaliser|planifié|attendue?s?\s+pour", re.IGNORECASE)


def _postprocess_action_spec(spec: ActionSpec, question: str) -> ActionSpec:
    """Force le statut sur mots-clés — robuste, sans inversion possible ; et
    rattache l'année au bon champ-date selon le verbe (clôturées→cloture, etc.)."""
    neg_clos = bool(_NEG_CLOS_RE.search(question))
    if _RETARD_RE.search(question):
        spec.f_en_retard = True
        spec.f_statut = None  # la contrainte "en retard" (⊂ en cours) suffit
    elif _EN_COURS_RE.search(question) or neg_clos:
        spec.f_statut = "en cours"  # type: ignore[assignment]
    elif _CLOS_RE.search(question):
        spec.f_statut = "clôturé"  # type: ignore[assignment]

    # Année polysémique : « clôturées en 2025 » = date_cloture ; « prévues en 2025 »
    # = date_prevue ; sinon date_ajout (création). Ne s'applique que si une année existe.
    if spec.f_annee_ajout:
        if spec.f_statut == "clôturé" and not neg_clos:
            spec.f_annee_champ = "cloture"  # type: ignore[assignment]
        elif spec.f_en_retard or _PREVUE_RE.search(question):
            spec.f_annee_champ = "prevue"  # type: ignore[assignment]
        else:
            spec.f_annee_champ = "ajout"  # type: ignore[assignment]

    # ─── Opérateur GRAPHE : filet DÉTERMINISTE + validation LLM (aligné sur /query) ───
    import graph_query_incident_v2 as _gq  # import tardif : évite le cycle
    _g = _gq.detect_graph_fields(question)
    if _g is not None:
        spec.graph_pattern = _g.pattern  # type: ignore[assignment]
        spec.graph_anchor = _g.anchor    # type: ignore[assignment]
        spec.graph_via = _g.via          # type: ignore[assignment]
        spec.graph_op = _g.op            # type: ignore[assignment]
        spec.graph_n = _g.n
        spec.graph_anchors_fe = list(_g.anchors_fe)
        spec.question_type = "count" if _g.output == "count" else "liste"  # type: ignore[assignment]
    elif spec.graph_pattern:  # graphe proposé par le LLM (paraphrase) → valider le couple
        _probe = _gq.GraphSpec(pattern=spec.graph_pattern, anchor=spec.graph_anchor or "incident",
                               via=spec.graph_via or "action", op=spec.graph_op, n=spec.graph_n)
        if _gq._satellite_of(_probe) is None:  # couple invalide → on abandonne
            spec.graph_pattern = None  # type: ignore[assignment]
    return spec


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
            spec = _postprocess_action_spec(spec, question)
            d = spec.model_dump()
            if d.get("f_responsable"):
                d["f_responsable"] = "<detecte>"  # jamais de login en log
            logger.info("ActionSpec (attempt %d): %s", attempt + 1, d)
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
        spec.type_action, spec.f_statut,
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
        f"null AS statut, null AS date_prevue, null AS date_cloture "
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

    if spec.f_en_retard:
        # échéance prévue dépassée ET action non clôturée
        where.append(f"a.date_prevue IS NOT NULL AND a.date_prevue < $today AND NOT {_CLOTUREE}")
        params["today"] = _date.today().isoformat()


    if spec.f_titre_action_keyword:
        where.append("toLower(a.titre_action) CONTAINS toLower($titre_action_kw)")
        params["titre_action_kw"] = spec.f_titre_action_keyword

    if spec.f_titre_incident_keyword:
        where.append("toLower(i.titre) CONTAINS toLower($titre_inc_kw)")
        params["titre_inc_kw"] = spec.f_titre_incident_keyword

    if spec.f_annee_ajout:
        # L'année se rattache au champ-date implicite du verbe (dates ISO AAAA-MM-JJ).
        _champ = {"cloture": "date_cloture", "prevue": "date_prevue"}.get(
            spec.f_annee_champ or "ajout", "date_ajout")
        where.append(f"a.{_champ} STARTS WITH $annee")
        params["annee"] = str(spec.f_annee_ajout)

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

    # count(DISTINCT a) et non count(*) : une action liée à plusieurs incidents
    # ne doit être comptée qu'une fois (163 actions sont multi-incidents ; sinon
    # préventives 437→532, clôturées 1089→1249 — sur-comptage par les arêtes).
    cypher_count = f"{base} RETURN count(DISTINCT a) AS n"
    params["limit"] = spec.limit
    cypher_liste = (
        f"{base} "
        f"RETURN i.numero_fe AS fe, i.titre AS titre_incident, i.severite AS severite, "
        f"i.actions_efficaces AS actions_efficaces, "
        f"a.titre_action AS titre_action, r.type_action AS type_action, "
        f"a.statut AS statut, "
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

    # Culture juste : une question rattachée à une PERSONNE (« les actions de X »)
    # se REFUSE explicitement — jamais de retrait silencieux du filtre (sinon on
    # servirait un chiffre global faux en silence : anti-pattern « contrainte
    # évaporée »). Le champ f_responsable ne sert plus qu'à DÉTECTER la mention.
    if spec.f_responsable:
        return {"status": "refus_personne", "spec": spec, "rows": [], "total": None}

    # ── Opérateur GRAPHE (aligné /query) : « actions récurrentes / partagées par ≥N
    # incidents »… délégué au compilateur gouverné (généralise via le parseur LLM). ──
    if getattr(spec, "graph_pattern", None):
        import graph_query_incident_v2 as gq
        gspec = gq.GraphSpec(
            pattern=spec.graph_pattern, anchor=spec.graph_anchor or "incident",
            via=spec.graph_via or "action", op=spec.graph_op, n=spec.graph_n,
            output=("count" if spec.question_type == "count" else "liste"),
            limit=spec.limit, anchors_fe=list(spec.graph_anchors_fe or []),
            f_severite=spec.f_severite_incident,
        )
        gres = gq.execute_graph_spec(gspec, neo4j)
        return {"status": "graph", "spec": spec, "graph": gres,
                "rows": [], "total": gres.get("total")}

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
            "date_prevue":       r.get("date_prevue"),
            "date_cloture":      r.get("date_cloture"),
            "actions_efficaces": r.get("actions_efficaces"),
        }
        for r in rows_raw
    ]
    return {"status": "ok", "spec": spec, "rows": rows, "total": total}
