"""
Génération LLM pour les réponses liste structurée d'incidents v2.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import prompt_store
from clients import OllamaClient

logger = logging.getLogger(__name__)

LLM_MODEL = "qwen2.5:7b"

_ORDRE_LABELS = {
    ("date_creation",  "desc"): "du plus récent au plus ancien (date de saisie)",
    ("date_creation",  "asc"):  "du plus ancien au plus récent (date de saisie)",
    ("date_evenement", "desc"): "du plus récent au plus ancien (date d'événement)",
    ("date_evenement", "asc"):  "du plus ancien au plus récent (date d'événement)",
    ("severite",       "desc"): "du plus grave au moins grave",
    ("severite",       "asc"):  "du moins grave au plus grave",
}


@dataclass
class ListResult:
    answer: str
    model_used: str


def _format_context(records: list[dict]) -> str:
    lines = []
    for i, r in enumerate(records, 1):
        parts = [f"**{i}. FE {r.get('numero_fe','?')}** — {r.get('titre','?')}"]
        if r.get("severite"):
            parts.append(f"Sévérité : {r['severite']}")
        if r.get("classification"):
            parts.append(f"Classification : {r['classification']}")
        if r.get("date_evenement"):
            parts.append(f"Date événement : {r['date_evenement'][:10]}")
        if r.get("date_creation"):
            parts.append(f"Date saisie : {r['date_creation'][:10]}")
        if r.get("etat"):
            parts.append(f"État : {r['etat']}")
        if r.get("action_corrective"):
            parts.append(f"Action corrective : {r['action_corrective'][:200]}")
        if r.get("resume_llm"):
            parts.append(f"Résumé : {r['resume_llm'][:200]}")
        lines.append(" | ".join(parts))
    return "\n".join(lines)


def phrase_list_result(
    question: str,
    spec,
    records: list[dict],
    ollama: OllamaClient,
) -> ListResult:
    if not records:
        return ListResult(
            answer="Aucun incident ne correspond à vos critères dans la base de données.",
            model_used=LLM_MODEL,
        )

    ordre_label = _ORDRE_LABELS.get(
        (spec.sort_by, spec.order),
        f"trié par {spec.sort_by} {spec.order}",
    )
    filtre_parts = []
    if spec.f_severite:
        filtre_parts.append(f"sévérité {spec.f_severite}")
    if spec.f_classification:
        filtre_parts.append(spec.f_classification)
    if spec.f_annee:
        filtre_parts.append(f"année {spec.f_annee}")
    filtre_label = ", ".join(filtre_parts) if filtre_parts else "tous types"

    context = _format_context(records)
    prompt = prompt_store.rendre(
        "liste.reponse",
        question=question,
        n=len(records),
        ordre_label=ordre_label,
        filtre_label=filtre_label,
        context=context,
    )

    try:
        answer = ollama.generate(prompt, model=LLM_MODEL, timeout=90.0)
        return ListResult(answer=answer.strip(), model_used=LLM_MODEL)
    except Exception as e:
        logger.error("LLM generation failed: %s", e)
        # Fallback sans LLM
        lines = [f"**{len(records)} incident(s) trouvé(s)** ({filtre_label}, {ordre_label}) :\n"]
        for r in records:
            lines.append(
                f"- **{r.get('numero_fe','?')}** — {r.get('titre','?')} "
                f"[{r.get('severite','')}]"
            )
        return ListResult(answer="\n".join(lines), model_used="fallback")
