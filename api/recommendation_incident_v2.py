"""
Voie RECOMMANDATION pour les incidents v2.

Cas d'usage RETEX : « voilà ce qui vient de se passer — qu'a-t-on fait sur des
incidents similaires ? »

Pipeline :
  1. Description de l'incident → embedding → incidents similaires (Qdrant)
     (réutilise retrieve_incident_v2, qui ramène déjà propriétés + relations,
      y compris les nœuds :Action)
  2. Extraction des actions de ces incidents : structurées (corrective /
     préventive / curative) ET à chaud (champ action_corrective de la fiche)
  3. Déduplication (une action partagée entre plusieurs incidents n'apparaît
     qu'une fois, avec toutes ses fiches sources)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from clients import Neo4jClient, OllamaClient, QdrantWrapper
from retrieval_incident_v2 import retrieve_incident_v2

logger = logging.getLogger(__name__)

_TRIVIAL = {"", "0", "-", "n/a", "na", "ras", "nil"}


@dataclass
class ActionRecommandee:
    type_action: str          # préventive | corrective | curative | à chaud
    titre: str
    fe_sources: list[str] = field(default_factory=list)
    statut: str | None = None
    responsable: str | None = None


@dataclass
class RecommandationResult:
    incidents: list[dict] = field(default_factory=list)
    actions: list[ActionRecommandee] = field(default_factory=list)
    n_chunks: int = 0
    below_threshold: bool = False


def _statut_lisible(props: dict) -> str | None:
    if props.get("statut") == "100" or props.get("date_cloture"):
        return "clôturé"
    if props.get("statut") == "0":
        return "en cours"
    return None


def run_recommendation(
    description: str,
    ollama: OllamaClient,
    qdrant: QdrantWrapper,
    neo4j: Neo4jClient,
    top_k: int = 10,
) -> RecommandationResult:
    retrieval = retrieve_incident_v2(
        question=description,
        ollama=ollama,
        qdrant=qdrant,
        neo4j=neo4j,
        top_k=top_k,
    )

    if retrieval.below_threshold or not retrieval.items:
        return RecommandationResult(n_chunks=retrieval.n_chunks_retrieved,
                                    below_threshold=True)

    incidents: list[dict] = []
    # clé de dédup : (type, titre en minuscules) -> ActionRecommandee
    actions: dict[tuple[str, str], ActionRecommandee] = {}

    for item in retrieval.items:
        props = item.props
        fe = props.get("numero_fe") or "?"
        incidents.append({
            "numero_fe": fe,
            "titre": props.get("titre"),
            "severite": props.get("severite"),
            "date_evenement": props.get("date_evenement"),
            "score": round(item.best_score, 3),
        })

        # Action à chaud (champ de la fiche)
        a_chaud = str(props.get("action_corrective") or "").strip()
        if a_chaud and a_chaud.lower() not in _TRIVIAL:
            cle = ("à chaud", a_chaud[:200].lower())
            if cle not in actions:
                actions[cle] = ActionRecommandee(type_action="à chaud",
                                                 titre=a_chaud[:250])
            if fe not in actions[cle].fe_sources:
                actions[cle].fe_sources.append(fe)

        # Actions structurées (nœuds :Action déjà ramenés par le retrieval)
        for e in item.entites:
            if e.get("rel") != "A_POUR_ACTION":
                continue
            ap = e.get("props") or {}
            titre_a = str(ap.get("titre_action") or "").strip()
            if not titre_a:
                continue
            type_a = ap.get("type_action") or "corrective"
            cle = (type_a, titre_a.lower())
            if cle not in actions:
                actions[cle] = ActionRecommandee(
                    type_action=type_a,
                    titre=titre_a,
                    statut=_statut_lisible(ap),
                    responsable=ap.get("responsable"),
                )
            if fe not in actions[cle].fe_sources:
                actions[cle].fe_sources.append(fe)

    # Préventives d'abord (c'est ce qu'on cherche pour éviter la récidive),
    # puis correctives, puis à chaud.
    ordre = {"préventive": 0, "corrective": 1, "curative": 2, "à chaud": 3}
    tri = sorted(actions.values(),
                 key=lambda a: (ordre.get(a.type_action, 9), -len(a.fe_sources)))

    # nombre d'actions portées par chaque incident (pour l'affichage)
    for inc in incidents:
        inc["n_actions"] = sum(1 for a in tri if inc["numero_fe"] in a.fe_sources)

    return RecommandationResult(
        incidents=incidents,
        actions=tri,
        n_chunks=retrieval.n_chunks_retrieved,
    )
