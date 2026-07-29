"""Assistant de SYNTHÈSE Safety — ancré sur le VOISINAGE sémantique réel.

À partir d'un cas déclaré, assemble un BROUILLON de dossier d'analyse à partir des cas
les plus SIMILAIRES (voisinage retrouvé par la reco + re-ranker), PAS d'un bucket de type
unique. Honnête sur la confiance : si le voisinage est hétérogène ou le match faible, elle
le DIT au lieu de verrouiller un type (leçon « fuite de carburant → dossier FOD faux »).

Principe : préparer l'analyse, pas la faire. Le LLM formule/généralise, il ne fabrique pas
(ni chiffres, ni sigles : le glossaire validé est injecté pour bloquer les inventions).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import glossaire
from clients import Neo4jClient, OllamaClient, QdrantWrapper
from recommendation_incident_v2 import run_recommendation

logger = logging.getLogger(__name__)

LLM_MODEL = "qwen2.5:7b"
_B = "coalesce(i.is_test_data,false)=false"
_GRAVE = "['4 - élevé','5 - intolérable']"
_CAUSE_FIELDS = "i.cause_mo, i.cause_machine, i.cause_milieu, i.cause_management, i.cause_methode"

# Palier de confiance : voisinage « ciblé » (type net + bon score) vs « hétérogène ».
_SEUIL_PART_TYPE = 0.60   # part du type majoritaire pour un dossier focalisé
_SEUIL_RERANK = 0.50      # score reranker top pour la confiance

# Notes de PÉRIMÈTRE de type : certains libellés ADL sont plus larges que leur sigle.
# Le brouillon doit l'expliciter pour ne pas induire l'analyste en erreur (ex. une fuite
# de carburant classée « FOD » — défendable mais déroutant sans cette précision).
_TYPE_NOTES = {
    "FOD": ("À ADL, le type d'événement « FOD » regroupe non seulement les débris / objets "
            "étrangers mais AUSSI les épandages et fuites (carburant, hydraulique, huile) sur les aires."),
}


@dataclass
class SyntheseResult:
    abstention: bool = False
    confiance: str = ""                        # "ciblé" | "hétérogène"
    type_dominant: str | None = None
    type_distribution: list = field(default_factory=list)
    note_type: str = ""                        # précision de périmètre du type dominant (si utile)
    contexte: dict = field(default_factory=dict)
    precedents: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    facteurs: dict = field(default_factory=dict)
    causes: list = field(default_factory=list)
    brouillon: str = ""


def _profil_voisinage(neo4j: Neo4jClient, fes: list[str]) -> dict:
    """Répartition des types parmi les cas RETROUVÉS (le voisinage) + type majoritaire."""
    rows = neo4j.run(
        "MATCH (i:IncidentSecu)-[:DE_TYPE]->(t:TypeEvenement) WHERE i.numero_fe IN $fes "
        "RETURN t.label AS type, count(*) AS n ORDER BY n DESC", fes=fes)
    dist = [(r["type"], r["n"]) for r in rows]
    total = sum(n for _, n in dist)
    top_type = dist[0][0] if dist else None
    top_share = round(dist[0][1] / total, 2) if total else 0.0
    return {"distribution": dist, "top_type": top_type, "top_share": top_share, "n": len(fes)}


def _contexte_type(neo4j: Neo4jClient, type_label: str) -> dict:
    """Stats large d'un type (fréquence/tendance/zones) — utilisé SEULEMENT si voisinage ciblé."""
    match = f"MATCH (i:IncidentSecu)-[:DE_TYPE]->(t:TypeEvenement {{label:$t}}) WHERE {_B}"
    total = neo4j.run(f"{match} RETURN count(i) AS n", t=type_label)[0]["n"]
    g = neo4j.run(f"{match} AND i.severite IN {_GRAVE} RETURN count(i) AS n", t=type_label)[0]["n"]
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


def _facteurs_des_cas(neo4j: Neo4jClient, fes: list[str]) -> dict:
    """Orientation causale 7M des cas RETROUVÉS (pas du bucket de type)."""
    n_analyse = neo4j.run(
        "MATCH (i:IncidentSecu) WHERE i.numero_fe IN $fes AND i.facteurs_7m IS NOT NULL "
        "RETURN count(i) AS n", fes=fes)[0]["n"]
    axes = [(r["ax"], r["n"]) for r in neo4j.run(
        "MATCH (i:IncidentSecu) WHERE i.numero_fe IN $fes AND i.facteurs_7m IS NOT NULL "
        "UNWIND i.facteurs_7m AS ax RETURN ax, count(*) AS n ORDER BY n DESC LIMIT 4", fes=fes)]
    return {"n_analyse": n_analyse, "axes": axes}


def _causes_des_cas(neo4j: Neo4jClient, fes: list[str], limit: int = 8) -> list[str]:
    """Causes rédigées des cas RETROUVÉS (exemples à généraliser, pas un bucket de type)."""
    rows = neo4j.run(
        "MATCH (i:IncidentSecu) WHERE i.numero_fe IN $fes "
        f"WITH i, [c IN [{_CAUSE_FIELDS}] WHERE c IS NOT NULL] AS cs WHERE size(cs) > 0 "
        "RETURN cs[0] AS cause LIMIT $lim", fes=fes, lim=limit)
    return [" ".join(str(r["cause"]).split())[:160] for r in rows]


def _efficaces_des_cas(neo4j: Neo4jClient, fes: list[str]) -> int:
    return neo4j.run(
        "MATCH (i:IncidentSecu) WHERE i.numero_fe IN $fes AND i.actions_efficaces=true "
        "RETURN count(i) AS n", fes=fes)[0]["n"]


_PROMPT = """Tu rédiges un BROUILLON d'analyse Safety à partir de DONNÉES FACTUELLES
(un analyste le corrigera). Reste factuel, n'invente RIEN, RESPECTE la confiance ci-dessous.

CAS DÉCLARÉ :
{description}

VOISINAGE — les {n_cas} cas les plus proches trouvés :
- confiance : {confiance}
- répartition des types : {type_distribution}
{note_type}
{bloc_contexte}
- orientation causale 7M des cas proches ({n_analyse} analysés) : {facteurs}
- causes rédigées sur les cas proches (EXEMPLES individuels — À GÉNÉRALISER) : {causes}
- actions récurrentes : {actions}
- réponses jugées efficaces : {efficaces}
- précédents : {precedents}

RÈGLES IMPÉRATIVES :
- Si confiance = « hétérogène » : NE PRÉTENDS PAS un type unique. Dis que la requête recouvre
  plusieurs types ({type_distribution}) et que peu de cas sont vraiment comparables — reste
  prudent, propose des pistes, n'affirme rien.
- Si confiance = « ciblé » : tu peux nommer le type dominant et donner le contexte/la tendance.
- CAUSES : généralise en 2-3 THÈMES récurrents (ex. « oublis de matériel », « non-respect de
  procédure »). NE RECOPIE PAS le verbatim d'un cas précis (pas de « Tango 4 à 05h10 »).
- CHIFFRES : ne cite QUE les nombres explicitement fournis ci-dessus (tendance, comptes).
  N'invente ni pourcentage ni statistique (pas de « 0,5 % des cas »).
- SIGLES : développe un sigle UNIQUEMENT selon le GLOSSAIRE ci-dessous ; n'invente AUCUNE
  signification absente, sinon recopie le sigle tel quel.
- TYPE : si une « NOTE type » est fournie ci-dessus, décris la nature RÉELLE des cas du voisinage
  et NE réduis PAS le libellé du type à son seul sigle.

{glossaire}

Rédige un brouillon court et structuré. Utilise EXACTEMENT ces titres de section, PROPRES
(n'y recopie AUCUNE consigne ni tiret d'instruction) :
1) Nature du voisinage
2) Contexte et tendance
3) Causes probables
4) Réponses habituelles et efficacité
5) Points d'attention
Consignes à appliquer SANS les écrire : la section (2) REPREND le « contexte » fourni
ci-dessus (nb de cas du type, % graves, tendance, zones) ; si ce contexte indique « non
calculé », écris simplement que le contexte de type n'est pas établi (voisinage hétérogène).
La section (3) donne 2-3 thèmes généralisés + l'orientation 7M, en pistes à confirmer.
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
    prof = _profil_voisinage(neo4j, fes)
    rerank_top = max((i.get("rerank_score") or 0.0) for i in reco.incidents)
    cible = prof["top_share"] >= _SEUIL_PART_TYPE and rerank_top >= _SEUIL_RERANK
    confiance = "ciblé" if cible else "hétérogène"

    fact = _facteurs_des_cas(neo4j, fes)
    causes = _causes_des_cas(neo4j, fes)
    efficaces = _efficaces_des_cas(neo4j, fes)
    # actions récurrentes = celles du voisinage (déjà dédupliquées par la reco)
    rec_actions = sorted(reco.actions, key=lambda a: -len(a.fe_sources))[:4]

    # Contexte de type : SEULEMENT si voisinage ciblé (sinon on ne verrouille pas un type).
    ctx = _contexte_type(neo4j, prof["top_type"]) if cible and prof["top_type"] else {}

    # Note de périmètre du type dominant (ex. FOD inclut les épandages carburant).
    note = _TYPE_NOTES.get(prof["top_type"] or "", "")
    note_line = f"- NOTE type : {note}" if note else ""

    dist_str = ", ".join(f"{t} ({n})" for t, n in prof["distribution"][:5]) or "n/d"
    if cible:
        bloc_ctx = (f"- {ctx.get('total', '?')} cas de type « {prof['top_type']} », "
                    f"dont {ctx.get('pct_graves', '?')}% graves\n"
                    f"- tendance : {', '.join(f'{a}:{n}' for a, n in ctx.get('tendance', {}).items()) or 'n/d'}\n"
                    f"- zones : {', '.join(f'{z} ({n})' for z, n in ctx.get('zones', [])) or 'n/d'}")
    else:
        bloc_ctx = "- (contexte de type NON calculé — voisinage hétérogène, pas de type unique)"

    prompt = _PROMPT.format(
        description=description[:800], n_cas=prof["n"], confiance=confiance,
        type_distribution=dist_str, note_type=note_line, bloc_contexte=bloc_ctx,
        n_analyse=fact["n_analyse"],
        facteurs="; ".join(f"{ax} ({n})" for ax, n in fact["axes"]) or "non renseigné",
        causes=" | ".join(causes) or "non renseignées",
        actions="; ".join(f"{a.titre} ({len(a.fe_sources)}×)" for a in rec_actions) or "peu structurées",
        efficaces=efficaces,
        precedents=", ".join(fes[:6]),
        glossaire=glossaire.bloc_glossaire())
    try:
        brouillon = ollama.generate(prompt, model=LLM_MODEL, temperature=0.2, num_ctx=4096).strip()
    except Exception as e:  # noqa: BLE001
        logger.error("Synthèse LLM échouée : %s", e)
        brouillon = "(rédaction indisponible — voir les données factuelles ci-dessus)"

    return SyntheseResult(
        abstention=False, confiance=confiance,
        type_dominant=prof["top_type"], type_distribution=prof["distribution"],
        note_type=note,
        contexte=ctx, precedents=reco.incidents, actions=reco.actions,
        facteurs=fact, causes=causes, brouillon=brouillon)
