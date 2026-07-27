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

import json
import logging
import os
from dataclasses import dataclass, field

from clients import Neo4jClient, OllamaClient, QdrantWrapper
from retrieval_incident_v2 import retrieve_incident_v2

logger = logging.getLogger(__name__)

_TRIVIAL = {"", "0", "-", "n/a", "na", "ras", "nil"}

# --- Filtre de pertinence (content-aware) ------------------------------------
# bge-m3 (bi-encodeur) sur-généralise : il rapproche tout « incident piste/aire +
# intervention » sans distinguer le vrai problème (ex. crevaison d'avion vs
# matériel au sol, tous deux ~0.68). Un seuil ne sépare pas ces scores tassés.
# On re-juge donc la pertinence sur le CONTENU avec le LLM, avant de recommander.
_FILTRE_MODEL = os.environ.get("PLANNER_MODEL", "qwen2.5:14b")
_FILTRE_SCHEMA = {
    "type": "object",
    "properties": {"garder": {"type": "array", "items": {"type": "string"}}},
    "required": ["garder"],
}
_FILTRE_PROMPT = """Tu filtres des incidents aéroportuaires par PERTINENCE réelle.

ÉVÉNEMENT DÉCRIT PAR L'UTILISATEUR :
{description}

INCIDENTS CANDIDATS (retrouvés par ressemblance de texte, parfois trompeuse) :
{candidats}

Garde UNIQUEMENT les incidents qui décrivent le MÊME GENRE DE PROBLÈME que
l'événement décrit — pas seulement le même lieu, la même phase ou le même thème
général. Exemple : un matériel gênant au sol n'est PAS le même problème qu'une
crevaison d'avion, même si les deux sont « sur la piste ».
En cas de doute réel sur une vraie ressemblance, garde-le (ne sur-filtre pas).
Si AUCUN candidat n'est vraiment comparable, renvoie une liste vide.

Réponds en JSON strict : {"garder": ["<numero_fe>", ...]}"""


@dataclass
class ActionRecommandee:
    type_action: str          # préventive | corrective | curative | à chaud
    titre: str
    fe_sources: list[str] = field(default_factory=list)
    statut: str | None = None


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


def _type_de(item) -> str:
    """Libellé du type d'événement (relation DE_TYPE), pour aider le filtre."""
    for e in item.entites:
        if e.get("rel") == "DE_TYPE":
            return str((e.get("props") or {}).get("label") or "")
    return ""


def _filtrer_pertinents(description: str, items: list, ollama: OllamaClient) -> list:
    """Re-juge la pertinence des incidents candidats sur leur CONTENU (le LLM lit
    vraiment, contrairement au bi-encodeur). Renvoie le sous-ensemble à garder.
    Robuste : en cas d'échec LLM, on garde tout (aucune régression dure).
    Peut renvoyer une liste vide → l'appelant s'abstient (« aucun cas comparable »)."""
    if len(items) <= 1:
        return items
    lignes = []
    for it in items:
        p = it.props
        fe = p.get("numero_fe") or "?"
        titre = str(p.get("titre") or "").strip()
        extrait = " ".join(str(p.get("detail") or "").split())[:180]
        lignes.append(f"- {fe} | type: {_type_de(it)} | {titre} | {extrait}")
    prompt = (_FILTRE_PROMPT
              .replace("{description}", description[:1200])
              .replace("{candidats}", "\n".join(lignes)))
    try:
        raw = ollama.generate_structured(prompt, _FILTRE_SCHEMA,
                                         model=_FILTRE_MODEL, timeout=60.0)
        garder = {str(x) for x in (json.loads(raw).get("garder") or [])}
    except Exception as e:  # noqa: BLE001
        logger.warning("Filtre pertinence reco échoué (fallback = tout garder) : %s", e)
        return items
    filtres = [it for it in items if (it.props.get("numero_fe") or "?") in garder]
    logger.info("Filtre pertinence reco : %d gardés / %d candidats", len(filtres), len(items))
    return filtres


# --- Re-rank cross-encoder (voie principale) ---------------------------------
# Le bi-encodeur bge-m3 ne sépare pas vraies vs nouvelles (scores tassés, ex.
# vraie R03=0.634 < nouvelle R48=0.696). Le cross-encoder les sépare nettement
# (mesuré : vraies ≥ 0.55, nouvelles ≤ 0.14) → seuil unique fiable.
# Seuils DÉCOUPLÉS : l'abstention se décide sur le MEILLEUR score (top faible = cas
# nouveau) ; l'élagage garde plus large (le cross-encoder ordonne déjà bien, le bruit
# reste en bas). Évite de casser consensus/muet (qui ont besoin de plusieurs fiches).
_SEUIL_ABSTENTION = float(os.environ.get("SEUIL_ABSTENTION", "0.25"))  # top < ça → abstention
_SEUIL_GARDE = float(os.environ.get("SEUIL_GARDE", "0.10"))            # garder les candidats ≥ ça
_MAX_GARDES = int(os.environ.get("MAX_GARDES", "10"))


def _doc_rerank(item) -> str:
    p = item.props
    corps = str(p.get("detail") or p.get("resume_llm") or "")
    return (str(p.get("titre") or "") + ". " + corps)[:600]


def _filtrer_par_rerank(description, items, reranker):
    """Re-note la pertinence via cross-encoder → (items_gardés, ok).
    - garde les candidats de score ≥ seuil (et le rappel est protégé : la vraie
      source remonte en tête) ;
    - si le meilleur score < seuil → liste vide → l'appelant s'abstient ;
    - ok=False → reranker indisponible → l'appelant retombe sur le filtre LLM."""
    if reranker is None:
        return items, False
    if len(items) <= 1:
        return items, True
    try:
        scores = reranker.rerank(description, [_doc_rerank(it) for it in items])
    except Exception as e:  # noqa: BLE001
        logger.warning("Reranker indisponible (fallback filtre LLM) : %s", e)
        return items, False
    paires = sorted(zip(items, scores), key=lambda x: -x[1])
    top = paires[0][1] if paires else 0.0
    if top < _SEUIL_ABSTENTION:
        logger.info("Rerank : top=%.3f < abstention %.2f → aucun cas comparable",
                    top, _SEUIL_ABSTENTION)
        return [], True
    gardes = [it for it, s in paires if s >= _SEUIL_GARDE][:_MAX_GARDES]
    logger.info("Rerank : top=%.3f, %d gardés / %d (abst %.2f / garde %.2f)",
                top, len(gardes), len(items), _SEUIL_ABSTENTION, _SEUIL_GARDE)
    return gardes, True


def run_recommendation(
    description: str,
    ollama: OllamaClient,
    qdrant: QdrantWrapper,
    neo4j: Neo4jClient,
    reranker=None,
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

    # Re-jugement de pertinence sur le CONTENU. Voie principale : cross-encoder
    # (sépare vraies vs nouvelles). Fallback : filtre LLM si le reranker est down.
    items, ok = _filtrer_par_rerank(description, retrieval.items, reranker)
    if not ok:
        items = _filtrer_pertinents(description, retrieval.items, ollama)
    if not items:
        # Aucun incident vraiment comparable → on s'abstient plutôt que d'inventer.
        return RecommandationResult(n_chunks=retrieval.n_chunks_retrieved,
                                    below_threshold=True)

    incidents: list[dict] = []
    # clé de dédup : (type, titre en minuscules) -> ActionRecommandee
    actions: dict[tuple[str, str], ActionRecommandee] = {}

    for item in items:
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
