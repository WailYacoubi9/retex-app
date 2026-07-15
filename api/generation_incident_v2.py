"""
Génération pour la voie incidents v2 (:IncidentSecu).

Le LLM phrase uniquement un contexte déjà calculé — il n'invente rien.

Le contexte est PILOTÉ PAR LE CATALOGUE (field_catalog / :ChampMeta publiés
depuis le schéma YAML) : TOUS les champs non vides de chaque incident sont
présentés au LLM avec leur label métier, les champs qui ont fait matcher
l'incident (matched_fields) en premier et avec un budget plus large.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass

import prompt_store
from clients import Neo4jClient, OllamaClient
from field_catalog import champ_meta, libelle
from retrieval_incident_v2 import RetrievalResultV2, RetrievedIncidentV2

logger = logging.getLogger(__name__)

LLM_MODEL = os.environ.get("LLM_MODEL_INCIDENT_V2", "qwen2.5:7b")
LLM_TEMPERATURE = 0.1
LLM_NUM_CTX = 8192  # contexte enrichi : dépasser les 4096 par défaut d'Ollama

# Propriétés techniques jamais montrées au LLM
_INTERNAL_PROPS = {"source_module", "last_indexed_at", "is_test_data", "incident_id"}

# Bloc d'identité : toujours en tête, dans cet ordre
_IDENTITY_ORDER = [
    "numero_fe", "titre", "date_evenement", "severite", "classification",
    "etat", "aerodrome", "condition_lumineuse",
]

# Champs texte longs (tronqués) — les autres passent en entier
_LONG_TEXT = {
    "detail", "resume_llm", "action_corrective", "analyse_chaud",
    "detail_verification", "desc_cause_1", "desc_cause_3", "desc_cause_5",
}
_MATCHED_MAX = 500   # budget large pour le champ qui a fait matcher l'incident
_LONG_MAX = 280
_TRIVIAL = {"", "0", "-", "n/a", "na", "ras", "nil"}

NO_RESULT_MESSAGE = (
    "Je n'ai pas trouvé d'incidents v2 pertinents dans la base pour répondre "
    "à cette question. Essayez de reformuler avec des termes plus précis."
)


@dataclass
class GenerationResultV2:
    answer: str
    model_used: str
    duration_ms: int


def generate_answer_incident_v2(
    question: str,
    retrieval_result: RetrievalResultV2,
    ollama: OllamaClient,
    neo4j: Neo4jClient | None = None,
) -> GenerationResultV2:
    start = time.time()

    if retrieval_result.below_threshold or not retrieval_result.items:
        return GenerationResultV2(
            answer=NO_RESULT_MESSAGE,
            model_used="none",
            duration_ms=int((time.time() - start) * 1000),
        )

    meta = champ_meta(neo4j) if neo4j is not None else {}
    context = _build_context(retrieval_result.items, meta)
    prompt = _build_prompt(question, context)
    logger.debug("Prompt length: %d chars", len(prompt))

    answer = ollama.generate(prompt=prompt, model=LLM_MODEL,
                             temperature=LLM_TEMPERATURE, num_ctx=LLM_NUM_CTX)

    return GenerationResultV2(
        answer=answer.strip(),
        model_used=LLM_MODEL,
        duration_ms=int((time.time() - start) * 1000),
    )


def _build_context(items: list[RetrievedIncidentV2], meta: dict) -> str:
    blocks: list[str] = []
    for item in items:
        lines = _format_item(item, meta)
        if lines:
            blocks.append("\n".join(lines))
    return "\n\n---\n\n".join(blocks)


def _valeur_affichable(val) -> str | None:
    if val is None:
        return None
    if isinstance(val, bool):
        return "Oui" if val else "Non"
    s = str(val).strip()
    return None if s.lower() in _TRIVIAL else s


def _format_item(item: RetrievedIncidentV2, meta: dict) -> list[str]:
    """Présente TOUS les champs non vides, matched_fields en premier."""
    lines: list[str] = []
    props = item.props
    matched = [f for f in item.matched_fields if f in props]
    vus: set[str] = set(_INTERNAL_PROPS)

    def _ajouter(cle: str, budget: int | None, marqueur: str = "") -> None:
        val = _valeur_affichable(props.get(cle))
        if val is None:
            return
        if budget and len(val) > budget:
            val = val[:budget] + "…"
        if cle == "date_evenement" or cle == "date_creation" or cle == "date_maj":
            val = val[:10]
        lines.append(f"{libelle(meta, cle)}{marqueur} : {val}")
        vus.add(cle)

    # 1. Identité
    for cle in _IDENTITY_ORDER:
        _ajouter(cle, 200)

    # 2. Champs qui ont fait matcher l'incident — budget large, marqués
    for cle in matched:
        if cle not in vus:
            _ajouter(cle, _MATCHED_MAX, " [CORRESPOND À LA QUESTION]")

    # 3. Résumé / détail
    for cle in ("resume_llm", "detail"):
        if cle not in vus:
            _ajouter(cle, _LONG_MAX)

    # 4. Tous les autres champs non vides (causes, flags, contexte…)
    for cle in sorted(props.keys()):
        if cle in vus:
            continue
        _ajouter(cle, _LONG_MAX if cle in _LONG_TEXT else None)

    # 5. Entités liées + actions structurées
    entites: list[str] = []
    actions: list[str] = []
    seen: set[str] = set()
    for e in item.entites or []:
        ep = e.get("props") or {}
        if e.get("rel") == "A_POUR_ACTION":
            titre_a = str(ep.get("titre_action") or "").strip()
            if titre_a:
                type_a = ep.get("type_action") or "corrective"
                statut = ("clôturée" if ep.get("statut") == "100" or ep.get("date_cloture")
                          else "en cours")
                actions.append(f"[{type_a}, {statut}] {titre_a[:150]}")
        else:
            lbls = e.get("labels") or []
            lbl = lbls[0] if lbls else "?"
            name = (ep.get("nom") or ep.get("label") or ep.get("login") or "")
            if name and name not in seen:
                entites.append(f"{lbl}: {name}")
                seen.add(name)
    if entites:
        lines.append("Entités liées : " + " ; ".join(entites[:8]))
    if actions:
        lines.append("Actions engagées : " + " ; ".join(actions[:6]))

    return lines


def _build_prompt(question: str, context: str) -> str:
    return prompt_store.rendre("recherche_semantique.reponse",
                               context=context, question=question)
