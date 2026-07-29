"""
Voie GRAPHE pour les incidents v2 — opérateurs de graphe GOUVERNÉS.

Objectif : répondre aux questions qui exploitent le GRAPHE (et non des tables
plates), que ni le moteur unifié (filtre→projette) ni la voie sémantique
(similarité→liste) ne savent exprimer :

  A — DEGRÉ / SEUIL (HAVING)  : « combien d'incidents ont au moins 2 actions »,
                                « quelles actions reviennent sur plusieurs incidents »,
                                « compagnies impliquées dans plus de 10 incidents ».
      motif : ancre → count(DISTINCT voisin) → op N

  C — VOISIN COMMUN           : « incidents qui partagent une même action »,
                                « actions communes à FNE/25/0497 et FNE/24/0112 ».
      motif : WITH voisin, collect(ancre) WHERE size(...) >= N

Principe (identique au reste du moteur) : le LLM MAPPE la question vers une spec
FERMÉE, il n'écrit JAMAIS de Cypher et ne calcule JAMAIS de chiffre. Toute
contrainte non mappable → on le DIT (besoin_precision), jamais le total en silence.

Le résultat est shapé au contrat `/ask/incident-v2/query` (GenericQueryResponse)
→ il s'affiche dans l'onglet 📊 Statistiques & requêtes sans changement frontend.
"""
from __future__ import annotations

import logging
import os
import re
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, ValidationError

import prompt_store
from clients import Neo4jClient, OllamaClient
from query_engine_incident_v2 import (
    _ACCIDENT_RE, _ANNEE_RE, _AUJOURD_RE, _JOUR_RE, _MOTS_NOMBRE, _NUIT_RE,
    _SERIEUX_RE, _SEVERITY_PATTERNS,
)

logger = logging.getLogger(__name__)

LLM_MODEL = os.environ.get("QUERY_LLM_MODEL", "qwen2.5:14b")

# ─── Adjacence du graphe : nœud satellite -> (relation, label, propriété de nom) ──
# Toutes les relations partent de (:IncidentSecu). Un couple (anchor, via) est
# valide SSI exactement l'un des deux est "incident" et l'autre un satellite ci-dessous.
_NODE_REL: dict[str, tuple[str, str, str]] = {
    "action":         ("A_POUR_ACTION",      "Action",        "titre_action"),
    "compagnie":      ("IMPLIQUE_COMPAGNIE", "Compagnie",     "nom"),
    "type_evenement": ("DE_TYPE",            "TypeEvenement", "label"),
    "lieu":           ("LOCALISE_EN",        "Lieu",          "label"),
    "phase_vol":      ("EN_PHASE_DE_VOL",    "PhaseVol",      "label"),
    "aeronef":        ("IMPLIQUE_AERONEF",   "TypeAeronef",   "label"),
    "service":        ("RESPONSABLE",        "Service",       "label"),
    "societe":        ("CONCERNE",           "Societe",       "nom"),
    "entite":         ("MIS_EN_CAUSE",       "Entite",        "nom"),
}
_NODE_KEYS = ("incident",) + tuple(_NODE_REL)

# Libellés lisibles pour la phrase de réponse déterministe
_LISIBLE = {
    "incident": "incident", "action": "action structurée", "compagnie": "compagnie",
    "type_evenement": "type d'événement", "lieu": "lieu", "phase_vol": "phase de vol",
    "aeronef": "type d'aéronef", "service": "service", "societe": "société",
    "entite": "entité mise en cause",
}


# ─── Spec fermée ─────────────────────────────────────────────────────────────

class GraphSpec(BaseModel):
    pattern: Literal["degree", "shared"] = "degree"
    anchor: Literal[
        "incident", "action", "compagnie", "type_evenement", "lieu",
        "phase_vol", "aeronef", "service", "societe", "entite",
    ] = "incident"
    via: Literal[
        "incident", "action", "compagnie", "type_evenement", "lieu",
        "phase_vol", "aeronef", "service", "societe", "entite",
    ] = "action"
    op: Literal[">=", ">", "=", "<=", "<"] = ">="
    n: int = Field(default=2, ge=1, le=999)
    output: Literal["count", "liste"] = "count"
    limit: int = Field(default=20, ge=1, le=200)
    anchors_fe: list[str] = Field(default_factory=list)
    # filtres incident réutilisés (même sémantique que UnifiedQuerySpec)
    f_severite: Optional[str] = None
    f_classification: Optional[str] = None
    f_annee: Optional[int] = None
    f_condition_lumineuse: Optional[str] = None
    f_type_evenement: Optional[str] = None


_GRAPH_SPEC_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "pattern": {"type": "string", "enum": ["degree", "shared"]},
        "anchor":  {"type": "string", "enum": list(_NODE_KEYS)},
        "via":     {"type": "string", "enum": list(_NODE_KEYS)},
        "op":      {"type": "string", "enum": [">=", ">", "=", "<=", "<"]},
        "n":       {"type": "integer", "minimum": 1, "maximum": 999},
        "output":  {"type": "string", "enum": ["count", "liste"]},
        "limit":   {"type": "integer", "minimum": 1, "maximum": 200},
        "anchors_fe": {"type": "array", "items": {"type": "string"}},
        "f_severite":       {"type": ["string", "null"]},
        "f_classification": {"type": ["string", "null"]},
        "f_annee":          {"type": ["integer", "null"]},
        "f_condition_lumineuse": {"type": ["string", "null"]},
        "f_type_evenement": {"type": ["string", "null"]},
    },
    "required": ["pattern", "anchor", "via"],
}


# ─── Routeur déterministe : est-ce une question de GRAPHE ? ───────────────────

_NUM = r"\d+|" + "|".join(_MOTS_NOMBRE)
_REL_NOUN = (r"actions?|mesures?\s+correctives?|incidents?|fiches?|accidents?|occurrences?|"
             r"compagnies?|transporteurs?|soci[ée]t[ée]s?|prestataires?|entit[ée]s?|"
             r"typ\w+|lieux?|zones?|phases?|a[ée]ronefs?|avions?|services?")

# Un quantificateur de seuil suivi (à courte distance) d'un nom de RELATION.
_DEGREE_GATE = re.compile(
    r"(?:au\s+moins|au\s+minimum|minimum|plus\s+de|plus\s+d'|≥|>=|au\s+plus|≤|<=|"
    r"moins\s+de|exactement|au\s+total)\s+(?:" + _NUM + r")?\s*(?:" + _REL_NOUN + r")",
    re.IGNORECASE)
_RECUR_GATE = re.compile(
    r"r[ée]curren\w+|reviennent\s+sur|à\s+r[ée]p[ée]tition|plusieurs\s+(?:" + _REL_NOUN + r")",
    re.IGNORECASE)
# Phase 1 : voisin commun restreint au partage d'ACTION (cas explicite de l'utilisateur).
# On NE matche PAS le partage cross-satellite (compagnies partageant un type, entités sur
# les mêmes lieux…) : ces questions restent aux voies existantes (analyste/unifié).
_SHARED_GATE = re.compile(
    r"m[êe]me[s]?\s+actions?"                                       # incidents … même action
    r"|actions?\s+communes?"                                        # actions communes (à X / à N incidents)
    r"|actions?\s+en\s+commun"
    r"|partag\w+\s+(?:une?\s+|la\s+|les\s+|leur\s+)?(?:m[êe]me\s+)?actions?"  # partagent une action
    r"|actions?\s+(?:qui\s+sont\s+)?partag\w+"                      # actions (qui sont) partagées …
    r"|partag\w+\s+(?:par|entre)\s+\S+\s+incidents?",               # partagées par deux/plusieurs incidents
    re.IGNORECASE)

_FE_RE = re.compile(r"\b[A-Z]{2,4}\s*/\s*\d{2}\s*/\s*\d{2,5}\b", re.IGNORECASE)
_COMBIEN_RE = re.compile(r"\bcombien\b|\bnombre\b|\bquel\s+nombre\b|\bcompte\b", re.IGNORECASE)
_LISTE_RE = re.compile(r"\bliste\b|\bquel(?:le)?s\b|\blesquel|\bdonne[- ]moi\b|\bmontre", re.IGNORECASE)


def is_graph_question(question: str) -> bool:
    """Routeur DÉTERMINISTE. Ne matche QUE des motifs de graphe clairs ; sinon la
    question repart au moteur unifié (défaut inchangé)."""
    q = question or ""
    return bool(_SHARED_GATE.search(q) or _DEGREE_GATE.search(q) or _RECUR_GATE.search(q))


# ─── Détection des nœuds mentionnés (positions) ──────────────────────────────

# Ordre = priorité : les motifs multi-mots / spécifiques d'abord ("type d'événement"
# AVANT "événement" qui, seul, désigne un incident).
_NODE_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("type_evenement", re.compile(r"typ\w*\s+d['e ]\s*[ée]v[ée]nements?|typologies?|typ\w*\s+d'incidents?", re.I)),
    ("phase_vol",      re.compile(r"phases?\s+de\s+vol|phases?", re.I)),
    ("compagnie",      re.compile(r"compagnies?|transporteurs?", re.I)),
    ("societe",        re.compile(r"soci[ée]t[ée]s?|prestataires?|sous-?traitants?", re.I)),
    ("entite",         re.compile(r"entit[ée]s?", re.I)),
    ("aeronef",        re.compile(r"a[ée]ronefs?|avions?|appareils?", re.I)),
    ("service",        re.compile(r"services?", re.I)),
    ("lieu",           re.compile(r"lieux?|zones?|emplacements?", re.I)),
    ("action",         re.compile(r"actions?|mesures?\s+correctives?", re.I)),
    ("incident",       re.compile(r"incidents?|fiches?|[ée]v[ée]nements?|accidents?|occurrences?", re.I)),
]


def _find_nodes(q: str) -> list[tuple[str, int]]:
    """Renvoie [(node_key, start), ...] trié par position, sans chevauchement
    (priorité aux motifs déclarés en premier)."""
    consumed: list[tuple[int, int]] = []
    found: list[tuple[str, int]] = []
    for key, pat in _NODE_PATTERNS:
        for m in pat.finditer(q):
            s, e = m.span()
            if any(s < ce and e > cs for cs, ce in consumed):
                continue
            consumed.append((s, e))
            found.append((key, s))
    return sorted(found, key=lambda x: x[1])


_OP_PHRASES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"au\s+moins|au\s+minimum|minimum|≥|>=|ou\s+plus|et\s+plus", re.I), ">="),
    (re.compile(r"plus\s+de|plus\s+d'|au-?del[àa]\s+de", re.I), ">"),
    (re.compile(r"au\s+plus|≤|<=|maximum", re.I), "<="),
    (re.compile(r"moins\s+de|en\s+de[çc][àa]\s+de", re.I), "<"),
    (re.compile(r"exactement|pr[ée]cis[ée]ment", re.I), "="),
]
_NUM_RE = re.compile(r"(" + _NUM + r")", re.I)


def _extract_op_n(q: str) -> tuple[Optional[str], Optional[int], Optional[int]]:
    """(op, n, position de fin du quantificateur) — déterministe. None si aucun seuil."""
    for pat, op in _OP_PHRASES:
        m = pat.search(q)
        if not m:
            continue
        tail = q[m.end():m.end() + 20]
        mn = _NUM_RE.search(tail)
        if mn:
            tok = mn.group(1).lower()
            n = int(tok) if tok.isdigit() else _MOTS_NOMBRE.get(tok, 2)
            if 1 <= n <= 999:
                return op, n, m.end()
        return op, 2, m.end()  # « plus de » sans nombre → ≥ implicite plus loin
    if re.search(r"plusieurs|r[ée]curren|reviennent|à\s+r[ée]p[ée]tition", q, re.I):
        return ">=", 2, 0
    return None, None, None


def _node_after(nodes: list[tuple[str, int]], pos: int) -> Optional[str]:
    for key, s in nodes:
        if s >= pos:
            return key
    return None


def _node_before(nodes: list[tuple[str, int]], pos: int, exclude: Optional[str]) -> Optional[str]:
    cand = [key for key, s in nodes if s < pos and key != exclude]
    return cand[0] if cand else None


# ─── Filtres incident (sous-ensemble, déterministe) ──────────────────────────

def _detect_filters(spec: GraphSpec, q: str) -> None:
    if spec.f_severite is None:
        for pat, val in _SEVERITY_PATTERNS:
            if re.search(pat, q, re.IGNORECASE):
                if val == "3 - important" and _SERIEUX_RE.search(q):
                    continue
                spec.f_severite = val
                break
    if _ACCIDENT_RE.search(q):
        spec.f_classification = "Accident"
    elif _SERIEUX_RE.search(q):
        spec.f_classification = "Incident sérieux"
    if spec.f_annee is None:
        m = _ANNEE_RE.search(q)
        if m:
            spec.f_annee = int(m.group(1))
    if spec.f_condition_lumineuse is None:
        if _NUIT_RE.search(q):
            spec.f_condition_lumineuse = "Nuit"
        elif _JOUR_RE.search(q) and not _AUJOURD_RE.search(q):
            spec.f_condition_lumineuse = "Jour"


# ─── Parseur : déterministe d'abord, LLM en secours ──────────────────────────

def _parse_deterministic(question: str) -> Optional[GraphSpec]:
    q = question
    nodes = _find_nodes(q)
    fes = [re.sub(r"\s+", "", m.group(0)).upper() for m in _FE_RE.finditer(q)]

    # ─ Voisin commun (C) — Phase 1 : partage d'ACTION uniquement ─
    if _SHARED_GATE.search(q):
        has_incident = any(k == "incident" for k, _ in nodes)
        # exiger un sujet « incidents » OU ≥2 fiches nommées ; sinon (sujet satellite,
        # ex. « compagnies partagent un type… ») on décline → voie normale.
        if not has_incident and len(fes) < 2:
            return None
        # seuil : « partagées par / à / entre / sur N incidents » ; « plusieurs » → 2 ; défaut 2
        n = 2
        mnn = re.search(r"(?:par|entre|[àa]|sur)\s+(" + _NUM + r")\s+incidents?", q, re.IGNORECASE)
        if mnn:
            tok = mnn.group(1).lower()
            n = int(tok) if tok.isdigit() else _MOTS_NOMBRE.get(tok, 2)
        spec = GraphSpec(pattern="shared", anchor="incident", via="action",
                         op=">=", n=max(n, 2), output="liste",
                         anchors_fe=fes if len(fes) >= 2 else [])
        _detect_filters(spec, q)
        return spec

    # ─ Degré / seuil (A) ─
    op, n, qpos = _extract_op_n(q)
    if op is None:
        return None

    via = _node_after(nodes, qpos) if qpos else None
    anchor = _node_before(nodes, qpos, exclude=via) if qpos else None

    # Récurrence : « ces actions reviennent sur plusieurs incidents » — le SUJET est
    # un satellite, le voisin compté est l'incident.
    if _RECUR_GATE.search(q) and (via is None or anchor is None):
        subj = next((k for k, _ in nodes if k in _NODE_REL), None)
        if subj:
            anchor, via = subj, "incident"

    if via is None and anchor is not None:
        via = "incident" if anchor in _NODE_REL else "action"
    if anchor is None and via is not None:
        anchor = "incident" if via in _NODE_REL else "action"
    if anchor is None or via is None:
        return None

    output = "count" if _COMBIEN_RE.search(q) and not _LISTE_RE.search(q) else \
             ("liste" if _LISTE_RE.search(q) else "count")
    spec = GraphSpec(pattern="degree", anchor=anchor, via=via, op=op, n=n or 2,
                     output=output)
    _detect_filters(spec, q)
    return spec


def _parse_llm(question: str, ollama: OllamaClient) -> Optional[GraphSpec]:
    try:
        prompt = prompt_store.rendre("graph_query.parseur", question=question)
    except Exception:  # gabarit absent → pas de secours LLM
        return None
    for attempt in range(2):
        try:
            raw = ollama.generate_structured(prompt, _GRAPH_SPEC_SCHEMA,
                                             model=LLM_MODEL, timeout=120.0)
            spec = GraphSpec.model_validate_json(raw)
            if _satellite_of(spec) is None:
                # couple (anchor, via) cross-satellite / invalide → on ne répond PAS
                # de travers : on laisse la question suivre sa voie normale.
                logger.info("GraphSpec LLM couple (%s,%s) invalide → fall-through",
                            spec.anchor, spec.via)
                return None
            _detect_filters(spec, question)
            logger.info("GraphSpec LLM (essai %d) : %s", attempt + 1, spec.model_dump())
            return spec
        except (ValidationError, Exception) as e:  # noqa: BLE001
            logger.warning("Parse GraphSpec LLM échoué (essai %d) : %s", attempt + 1, e)
    return None


def _satellite_of(spec: GraphSpec) -> Optional[str]:
    """Le nœud satellite du couple (anchor, via). None si le couple est invalide
    (les deux 'incident', les deux satellites, ou un inconnu)."""
    a_sat, v_sat = spec.anchor in _NODE_REL, spec.via in _NODE_REL
    a_inc, v_inc = spec.anchor == "incident", spec.via == "incident"
    if a_inc and v_sat:
        return spec.via
    if v_inc and a_sat:
        return spec.anchor
    return None


# ─── Compilation Cypher déterministe (READ-ONLY, paramétrée) ─────────────────

def _incident_where(spec: GraphSpec) -> tuple[list[str], list[str], dict]:
    """Retourne (matches_relation, clauses_where, params) pour les filtres incident."""
    where = ["coalesce(i.is_test_data, false) = false"]
    matches: list[str] = []
    params: dict[str, Any] = {}
    if spec.f_severite:
        where.append("i.severite = $f_sev"); params["f_sev"] = spec.f_severite
    if spec.f_classification:
        where.append("i.classification = $f_cls"); params["f_cls"] = spec.f_classification
    if spec.f_annee:
        where.append("i.date_evenement STARTS WITH $f_annee"); params["f_annee"] = str(spec.f_annee)
    if spec.f_condition_lumineuse:
        where.append("i.condition_lumineuse = $f_cl"); params["f_cl"] = spec.f_condition_lumineuse
    if spec.f_type_evenement:
        matches.append("MATCH (i)-[:DE_TYPE]->(tf:TypeEvenement)")
        where.append("toLower(tf.label) CONTAINS toLower($f_tev)"); params["f_tev"] = spec.f_type_evenement
    return matches, where, params


def build_graph_cypher(spec: GraphSpec, satellite: str) -> tuple[str, dict]:
    rel, lbl, prop = _NODE_REL[satellite]
    fmatches, fwhere, params = _incident_where(spec)
    base_match = " ".join(["MATCH (i:IncidentSecu)-[:%s]->(x:%s)" % (rel, lbl)] + fmatches)
    where_str = " AND ".join(fwhere)

    if spec.pattern == "shared":
        params["n"] = max(spec.n, 2)
        params["limit"] = min(spec.limit, 200)
        if spec.anchors_fe:
            params["fes"] = spec.anchors_fe
            return (
                f"{base_match} WHERE {where_str} AND i.numero_fe IN $fes "
                f"WITH x, collect(DISTINCT i.numero_fe) AS incidents "
                f"WHERE size(incidents) = size($fes) "
                f"RETURN x.{prop} AS label, size(incidents) AS n, incidents "
                f"ORDER BY n DESC LIMIT $limit"
            ), params
        return (
            f"{base_match} WHERE {where_str} "
            f"WITH x, collect(DISTINCT i.numero_fe) AS incidents "
            f"WHERE size(incidents) >= $n "
            f"RETURN x.{prop} AS label, size(incidents) AS n, incidents[0..12] AS incidents "
            f"ORDER BY n DESC LIMIT $limit"
        ), params

    # ── DEGRÉ ──
    op = spec.op
    params["n"] = spec.n
    if spec.anchor == "incident":                      # count satellites par incident
        deg = "WITH i, count(DISTINCT x) AS deg"
        cond = f"WHERE deg {op} $n"
        if spec.output == "count":
            return f"{base_match} WHERE {where_str} {deg} {cond} RETURN count(i) AS n", params
        params["limit"] = min(spec.limit, 200)
        return (
            f"{base_match} WHERE {where_str} {deg} {cond} "
            f"RETURN i.numero_fe AS numero_fe, i.titre AS titre, i.severite AS severite, "
            f"deg AS n ORDER BY deg DESC LIMIT $limit"
        ), params
    else:                                              # count incidents par satellite
        deg = "WITH x, count(DISTINCT i) AS deg"
        cond = f"WHERE deg {op} $n"
        if spec.output == "count":
            return f"{base_match} WHERE {where_str} {deg} {cond} RETURN count(x) AS n", params
        params["limit"] = min(spec.limit, 200)
        return (
            f"{base_match} WHERE {where_str} {deg} {cond} "
            f"RETURN x.{prop} AS label, deg AS n "
            f"ORDER BY deg DESC LIMIT $limit"
        ), params


def count_graph_cypher(spec: GraphSpec, satellite: str) -> tuple[str, dict]:
    """Cypher du VRAI total (nb d'ancres/voisins qui satisfont le seuil), pour une
    sortie liste — la liste n'en montre qu'une page, le total doit rester exact."""
    rel, lbl, _ = _NODE_REL[satellite]
    fmatches, fwhere, params = _incident_where(spec)
    base = " ".join([f"MATCH (i:IncidentSecu)-[:{rel}]->(x:{lbl})"] + fmatches)
    ws = " AND ".join(fwhere)
    if spec.pattern == "shared":
        params["n"] = max(spec.n, 2)
        if spec.anchors_fe:
            params["fes"] = spec.anchors_fe
            return (f"{base} WHERE {ws} AND i.numero_fe IN $fes "
                    f"WITH x, collect(DISTINCT i.numero_fe) AS f WHERE size(f) = size($fes) "
                    f"RETURN count(x) AS n"), params
        return (f"{base} WHERE {ws} WITH x, collect(DISTINCT i.numero_fe) AS f "
                f"WHERE size(f) >= $n RETURN count(x) AS n"), params
    params["n"] = spec.n
    if spec.anchor == "incident":
        return (f"{base} WHERE {ws} WITH i, count(DISTINCT x) AS deg "
                f"WHERE deg {spec.op} $n RETURN count(i) AS n"), params
    return (f"{base} WHERE {ws} WITH x, count(DISTINCT i) AS deg "
            f"WHERE deg {spec.op} $n RETURN count(x) AS n"), params


# ─── Phrase de réponse déterministe (aucun LLM → zéro fabrication) ────────────

_OP_TXT = {">=": "au moins", ">": "plus de", "<=": "au plus", "<": "moins de", "=": "exactement"}


def _explication(spec: GraphSpec, satellite: str, total: int,
                 rows: list[dict], baseline: Optional[int]) -> str:
    op_txt = _OP_TXT.get(spec.op, spec.op)
    page = f" ; les {len(rows)} premières affichées" if total > len(rows) else ""

    if spec.pattern == "shared":
        via_lbl = _LISIBLE.get(satellite, satellite)
        if spec.anchors_fe:
            return (f"{total} {via_lbl}(s) commune(s) aux incidents {', '.join(spec.anchors_fe)}{page}."
                    if total else f"Aucune {via_lbl} commune à {', '.join(spec.anchors_fe)}.")
        return (f"{total} {via_lbl}(s) sont partagées par au moins {max(spec.n, 2)} "
                f"incidents distincts (triées par nombre d'incidents décroissant){page}.")

    # degré
    if spec.anchor == "incident":
        via_lbl = _LISIBLE.get(satellite, satellite)
        base = f"{total} incident(s) ont {op_txt} {spec.n} {via_lbl}(s) associée(s)"
        if baseline is not None and spec.op in (">=", ">"):
            base += f" (sur {baseline} qui en ont au moins une)"
        return base + ("." if spec.output == "count"
                       else f", triés par nombre décroissant{page}.")
    else:
        anc_lbl = _LISIBLE.get(spec.anchor, spec.anchor)
        base = f"{total} {anc_lbl}(s) sont liées à {op_txt} {spec.n} incident(s) distinct(s)"
        note = ("" if spec.anchor != "action" else
                " (comptage par nœud :Action — la récurrence par THÈME est plus élevée, "
                "les titres d'action étant fragmentés, cf. journal qualité D6)")
        return base + note + ("." if spec.output == "count"
                              else f", triés par nombre décroissant{page}.")


# ─── Point d'entrée ──────────────────────────────────────────────────────────

def detect_graph_fields(question: str) -> Optional[GraphSpec]:
    """Détecteur DÉTERMINISTE (regex). Sert de FILET dans les parseurs LLM des voies
    (unifiée/actions) : il ne fige que les motifs de graphe CLAIRS, sinon None. La
    généralisation (paraphrases) vient du LLM ; ce filet garantit les cas nets."""
    return _parse_deterministic(question)


def run_graph_query(question: str, ollama: OllamaClient, neo4j: Neo4jClient) -> dict:
    """
    Parse (déterministe, LLM en secours) puis exécute. Contrat /query :
      status : "ok" | "besoin_precision" | "parse_failed"
    """
    spec = _parse_deterministic(question)
    if spec is None:
        spec = _parse_llm(question, ollama)
    if spec is None:
        return {"status": "parse_failed", "spec_interpretee": None,
                "cypher_execute": None, "resultat_brut": None,
                "output": None, "total": None, "rows": [], "explication": ""}
    return execute_graph_spec(spec, neo4j)


def execute_graph_spec(spec: GraphSpec, neo4j: Neo4jClient) -> dict:
    """Exécute une GraphSpec DÉJÀ résolue (quelle qu'en soit la source : parseur graphe
    OU champs graphe d'une autre spec de voie) → dict compatible contrat /query."""
    satellite = _satellite_of(spec)
    if satellite is None:
        # Couple (anchor, via) non résolu → on le DIT (anti-« silently wrong »).
        return {"status": "besoin_precision", "spec_interpretee": spec.model_dump(),
                "cypher_execute": None, "resultat_brut": None,
                "output": spec.output, "total": None, "rows": [],
                "explication": ("Précisez la relation à compter : par ex. « incidents ayant "
                                "au moins 2 actions » ou « actions liées à au moins 2 incidents ».")}

    cypher, params = build_graph_cypher(spec, satellite)
    logger.info("GraphQuery Cypher: %s | params: %s", cypher, params)
    rows_raw = neo4j.run_read(cypher, **{k: v for k, v in params.items()})

    is_count = spec.pattern == "degree" and spec.output == "count"
    if is_count:
        total = int(rows_raw[0]["n"]) if rows_raw else 0
        rows: list[dict] = []
    else:
        rows = [dict(r) for r in rows_raw]
        ccy, cpar = count_graph_cypher(spec, satellite)
        cr = neo4j.run_read(ccy, **cpar)
        total = int(cr[0]["n"]) if cr else len(rows)

    # Baseline « ≥1 » pour contextualiser un seuil sur incident→satellite (peu coûteux).
    baseline: Optional[int] = None
    if spec.pattern == "degree" and spec.anchor == "incident" and spec.op in (">=", ">"):
        rel, lbl, _ = _NODE_REL[satellite]
        fmatches, fwhere, bparams = _incident_where(spec)
        bcy = (" ".join(["MATCH (i:IncidentSecu)-[:%s]->(:%s)" % (rel, lbl)] + fmatches)
               + f" WHERE {' AND '.join(fwhere)} RETURN count(DISTINCT i) AS n")
        try:
            br = neo4j.run_read(bcy, **bparams)
            baseline = int(br[0]["n"]) if br else None
        except Exception:  # noqa: BLE001
            baseline = None

    explication = _explication(spec, satellite, total, rows, baseline)
    return {"status": "ok", "spec_interpretee": spec.model_dump(),
            "cypher_execute": cypher, "resultat_brut": (total if is_count else rows),
            "output": ("count" if is_count else "liste"),
            "total": total, "rows": rows, "explication": explication}


def to_action_rows(gres: dict) -> list[dict]:
    """Mappe les lignes d'un résultat graphe vers le schéma de la voie Actions
    (ActionResult : champs tous optionnels). Permet de servir « actions partagées /
    récurrentes » dans l'onglet 🛠️ Actions sans changer son schéma de réponse."""
    spec = gres.get("spec_interpretee") or {}
    rows = gres.get("rows") or []
    out: list[dict] = []
    if spec.get("pattern") == "shared":                    # actions partagées → 1 ligne/action
        for r in rows:
            fes = r.get("incidents") or []
            suffixe = "…" if len(fes) > 6 else ""
            out.append({
                "titre_action": r.get("label"),
                "type_action": "partagée",
                "titre_incident": f"{r.get('n')} incidents : " + ", ".join(fes[:6]) + suffixe,
                "fe": fes[0] if fes else None,
            })
    elif spec.get("anchor") == "action":                   # degré action→incident (récurrentes)
        for r in rows:
            out.append({"titre_action": r.get("label"),
                        "titre_incident": f"{r.get('n')} incidents distincts"})
    else:                                                  # degré incident→action → renvoie des incidents
        for r in rows:
            out.append({"fe": r.get("numero_fe"), "titre_incident": r.get("titre"),
                        "severite": r.get("severite"), "titre_action": f"{r.get('n')} actions"})
    return out
