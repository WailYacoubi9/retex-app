"""Assistant de SYNTHÈSE Safety.

À partir d'un cas déclaré, assemble un BROUILLON d'analyse ancré (contexte +
précédents + tendance + actions typiques/efficaces) que l'analyste corrige.
Réutilise la voie reco (retrieval + re-ranker) pour les précédents, et
l'analytique sur les dimensions FIABLES (type/lieu/date/gravité ~100 %).

Point de convergence : reco (aide à la rédaction) + analyse (patterns) +
signal d'efficacité, avec le jugement humain préservé (c'est un brouillon).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from clients import Neo4jClient, OllamaClient, QdrantWrapper
from recommendation_incident_v2 import run_recommendation

logger = logging.getLogger(__name__)

LLM_MODEL = "qwen2.5:7b"
_B = "coalesce(i.is_test_data,false)=false"
_GRAVE = "['4 - élevé','5 - intolérable']"


@dataclass
class SyntheseResult:
    abstention: bool = False
    type_dominant: str | None = None
    contexte: dict = field(default_factory=dict)
    precedents: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    facteurs: dict = field(default_factory=dict)
    brouillon: str = ""


def _type_dominant(neo4j: Neo4jClient, fes: list[str]) -> str | None:
    rows = neo4j.run(
        "MATCH (i:IncidentSecu)-[:DE_TYPE]->(t:TypeEvenement) WHERE i.numero_fe IN $fes "
        "RETURN t.label AS type, count(*) AS n ORDER BY n DESC LIMIT 1", fes=fes)
    return rows[0]["type"] if rows else None


def _contexte_type(neo4j: Neo4jClient, type_label: str) -> dict:
    match = f"MATCH (i:IncidentSecu)-[:DE_TYPE]->(t:TypeEvenement {{label:$t}}) WHERE {_B}"
    total = (neo4j.run(f"{match} RETURN count(i) AS n", t=type_label)[0]["n"])
    g = (neo4j.run(f"{match} AND i.severite IN {_GRAVE} RETURN count(i) AS n", t=type_label)[0]["n"])
    trend = {}
    for r in neo4j.run(f"{match} RETURN substring(toString(i.date_evenement),0,4) AS a, count(i) AS n", t=type_label):
        if r["a"] and str(r["a"]).isdigit():
            trend[r["a"]] = r["n"]
    trend = dict(sorted(trend.items())[-4:])
    zones = [(r["z"], r["n"]) for r in neo4j.run(
        f"{match} MATCH (i)-[:LOCALISE_EN]->(l:Lieu) RETURN l.label AS z, count(i) AS n "
        f"ORDER BY n DESC LIMIT 3", t=type_label)]
    return {"total": total, "graves": g,
            "pct_graves": round(100 * g / total, 1) if total else 0,
            "tendance": trend, "zones": zones}


def _actions_type(neo4j: Neo4jClient, type_label: str) -> dict:
    match = f"MATCH (i:IncidentSecu)-[:DE_TYPE]->(t:TypeEvenement {{label:$t}}) WHERE {_B}"
    rec = [(r["t"], r["n"]) for r in neo4j.run(
        f"{match} MATCH (i)-[:A_POUR_ACTION]->(a:Action) "
        f"WITH a.titre_action AS t, count(DISTINCT i) AS n WHERE n>=2 "
        f"RETURN t, n ORDER BY n DESC LIMIT 4", t=type_label)]
    eff = neo4j.run(f"{match} AND i.actions_efficaces=true RETURN count(i) AS n", t=type_label)[0]["n"]
    return {"recurrentes": rec, "efficaces": eff}


def _facteurs_type(neo4j: Neo4jClient, type_label: str) -> dict:
    """Profil causal 7M des cas de ce type (analyse Ishikawa saisie sur ~44 % des
    fiches). Donne une ORIENTATION causale (où creuser), pas la cause de CE cas."""
    match = f"MATCH (i:IncidentSecu)-[:DE_TYPE]->(t:TypeEvenement {{label:$t}}) WHERE {_B}"
    n_analyse = neo4j.run(
        f"{match} AND i.facteurs_7m IS NOT NULL RETURN count(i) AS n", t=type_label)[0]["n"]
    axes = [(r["ax"], r["n"]) for r in neo4j.run(
        f"{match} AND i.facteurs_7m IS NOT NULL UNWIND i.facteurs_7m AS ax "
        f"RETURN ax, count(*) AS n ORDER BY n DESC LIMIT 4", t=type_label)]
    return {"n_analyse": n_analyse, "axes": axes}


_PROMPT = """Tu rédiges un BROUILLON d'analyse Safety à partir de DONNÉES FACTUELLES
(un analyste le corrigera). Reste factuel, appuie-toi sur les chiffres, n'invente RIEN.

CAS DÉCLARÉ :
{description}

DONNÉES (type dominant : {type}) :
- {total} cas historiques de ce type, dont {pct_graves}% graves
- tendance par année : {tendance}
- zones concernées : {zones}
- profil causal 7M des cas similaires ({n_analyse} analysés) : {facteurs}
- actions récurrentes : {actions}
- réponses jugées efficaces sur ce type : {efficaces}
- précédents similaires : {precedents}

Rédige un brouillon court et structuré :
1) Contexte (fréquence, gravité de ce type d'événement)
2) Tendance et concentration (évolution, où ça se concentre)
3) Orientation causale probable — d'après le 7M des cas similaires ; dis vers quels
   axes creuser en priorité, en précisant que c'est une PISTE à confirmer sur ce cas
4) Réponses habituelles et ce qui a été jugé efficace
5) Points d'attention
Termine par : « À valider et compléter par l'analyste. »"""


def run_synthese(description: str, ollama: OllamaClient, qdrant: QdrantWrapper,
                 neo4j: Neo4jClient, reranker=None, top_k: int = 20) -> SyntheseResult:
    reco = run_recommendation(description=description, ollama=ollama, qdrant=qdrant,
                              neo4j=neo4j, reranker=reranker, top_k=top_k)
    if reco.below_threshold or not reco.incidents:
        return SyntheseResult(
            abstention=True,
            brouillon=("Aucun cas comparable dans la base : pas de synthèse fiable à "
                       "produire. Décrivez l'événement plus précisément (lieu, matériel, "
                       "type), ou traitez-le comme un cas nouveau."))

    fes = [i["numero_fe"] for i in reco.incidents]
    type_dom = _type_dominant(neo4j, fes)
    ctx = _contexte_type(neo4j, type_dom) if type_dom else {}
    acts = _actions_type(neo4j, type_dom) if type_dom else {"recurrentes": [], "efficaces": 0}
    fact = _facteurs_type(neo4j, type_dom) if type_dom else {"n_analyse": 0, "axes": []}

    prompt = _PROMPT.format(
        description=description[:800], type=type_dom or "non déterminé",
        total=ctx.get("total", "?"), pct_graves=ctx.get("pct_graves", "?"),
        tendance=", ".join(f"{a}:{n}" for a, n in ctx.get("tendance", {}).items()) or "n/d",
        zones=", ".join(f"{z} ({n})" for z, n in ctx.get("zones", [])) or "n/d",
        n_analyse=fact["n_analyse"],
        facteurs="; ".join(f"{ax} ({n})" for ax, n in fact["axes"]) or "non renseigné",
        actions="; ".join(f"{t} ({n}×)" for t, n in acts["recurrentes"]) or "peu structurées",
        efficaces=acts["efficaces"],
        precedents=", ".join(fes[:6]))
    try:
        brouillon = ollama.generate(prompt, model=LLM_MODEL, temperature=0.2, num_ctx=4096).strip()
    except Exception as e:  # noqa: BLE001
        logger.error("Synthèse LLM échouée : %s", e)
        brouillon = "(rédaction indisponible — voir les données factuelles ci-dessus)"

    return SyntheseResult(abstention=False, type_dominant=type_dom, contexte=ctx,
                          precedents=reco.incidents, actions=reco.actions,
                          facteurs=fact, brouillon=brouillon)
