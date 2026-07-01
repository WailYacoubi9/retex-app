"""
Génération pour la voie incidents v2 (:IncidentSecu).

Le LLM phrase uniquement un contexte déjà calculé — il n'invente rien.
La projection sur CONTEXT_FIELDS contrôle ce que voit le modèle.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass

from clients import OllamaClient
from retrieval_incident_v2 import RetrievalResultV2, RetrievedIncidentV2

logger = logging.getLogger(__name__)

LLM_MODEL = os.environ.get("LLM_MODEL_INCIDENT_V2", "qwen2.5:7b")
LLM_TEMPERATURE = 0.1

CONTEXT_FIELDS = [
    {"prop": "numero_fe",         "label": "FE",                 "max_len": None},
    {"prop": "titre",             "label": "Titre",              "max_len": 200},
    {"prop": "date_evenement",    "label": "Date",               "max_len": 10},
    {"prop": "severite",          "label": "Sévérité",           "max_len": None},
    {"prop": "classification",    "label": "Classification",     "max_len": None},
    {"prop": "etat",              "label": "État",               "max_len": None},
    {"prop": "action_corrective", "label": "Action corrective",  "max_len": 300},
]

# Les ~9% sans resume_llm restent sur detail. Générer les résumés manquants
# est une tâche d'ingestion séparée (nécessite Ollama/GPU, hors périmètre ici).
_RESUME_PROP = "resume_llm"
_DETAIL_PROP = "detail"
_RESUME_MAX = 400
_DETAIL_MAX = 400

_TRIVIAL = {"", "0", "-", "n/a", "na"}

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
) -> GenerationResultV2:
    start = time.time()

    if retrieval_result.below_threshold or not retrieval_result.items:
        return GenerationResultV2(
            answer=NO_RESULT_MESSAGE,
            model_used="none",
            duration_ms=int((time.time() - start) * 1000),
        )

    context = _build_context(retrieval_result.items)
    prompt = _build_prompt(question, context)
    logger.debug("Prompt length: %d chars", len(prompt))

    answer = ollama.generate(prompt=prompt, model=LLM_MODEL, temperature=LLM_TEMPERATURE)

    return GenerationResultV2(
        answer=answer.strip(),
        model_used=LLM_MODEL,
        duration_ms=int((time.time() - start) * 1000),
    )


def _build_context(items: list[RetrievedIncidentV2]) -> str:
    blocks: list[str] = []
    for item in items:
        lines = _format_item(item)
        if lines:
            blocks.append("\n".join(lines))
    return "\n\n---\n\n".join(blocks)


def _format_item(item: RetrievedIncidentV2) -> list[str]:
    lines: list[str] = []
    props = item.props

    for cfg in CONTEXT_FIELDS:
        val = props.get(cfg["prop"])
        if val is None:
            continue
        val_str = str(val).strip()
        if val_str.lower() in _TRIVIAL:
            continue
        if cfg["max_len"] and len(val_str) > cfg["max_len"]:
            val_str = val_str[: cfg["max_len"]] + "…"
        lines.append(f"{cfg['label']} : {val_str}")

    # Description : résumé LLM en priorité, detail en repli
    desc_raw = props.get(_RESUME_PROP)
    desc_str = str(desc_raw).strip() if desc_raw is not None else ""
    if not desc_str or desc_str.lower() in _TRIVIAL:
        desc_raw = props.get(_DETAIL_PROP)
        desc_str = str(desc_raw).strip() if desc_raw is not None else ""
        desc_label = "Détail"
        desc_max = _DETAIL_MAX
    else:
        desc_label = "Description"
        desc_max = _RESUME_MAX
    if desc_str and desc_str.lower() not in _TRIVIAL:
        if len(desc_str) > desc_max:
            desc_str = desc_str[:desc_max] + "…"
        lines.append(f"{desc_label} : {desc_str}")

    if item.entites:
        parts: list[str] = []
        seen: set[str] = set()
        for e in item.entites:
            lbls = e.get("labels") or []
            lbl = lbls[0] if lbls else "?"
            ep = e.get("props") or {}
            name = ep.get("nom") or ep.get("label") or ep.get("valeur") or ep.get("value") or ""
            if name and name not in seen:
                parts.append(f"{lbl}: {name}")
                seen.add(name)
        if parts:
            lines.append("Entités liées : " + " ; ".join(parts[:6]))

    return lines


def _build_prompt(question: str, context: str) -> str:
    return f"""Tu es un assistant expert en analyse d'incidents de sécurité aéronautique.

RÈGLES IMPÉRATIVES :
1. Réponds UNIQUEMENT à partir des incidents listés dans le CONTEXTE ci-dessous.
2. N'invente AUCUNE information absente du contexte.
3. Si certains incidents du contexte ne sont PAS pertinents pour la question, IGNORE-les silencieusement.
4. Si AUCUN incident du contexte n'est pertinent, réponds simplement : "Aucun incident pertinent trouvé pour cette question."

FORMAT DE LA RÉPONSE :
- Réponds en français naturel et fluide, comme un expert qui synthétise.
- Cite les numéros FE de manière fluide dans tes phrases (ex: "...comme l'illustre l'incident FNE/26/0241").
- PRIVILÉGIE LA SYNTHÈSE par catégories ou thèmes, plutôt qu'une liste brute d'incidents.
- Cite au maximum 5 numéros FE dans ta réponse, même si plus sont fournis.
- 4 à 8 phrases maximum.

CONTEXTE — INCIDENTS DISPONIBLES :

{context}

QUESTION DE L'UTILISATEUR :
{question}

RÉPONSE :"""
