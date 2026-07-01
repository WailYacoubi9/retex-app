"""
Génération de la réponse phrasée pour la voie agrégation.

Le LLM reçoit les chiffres DÉJÀ CALCULÉS et les phrase uniquement.
Il ne recalcule rien, n'invente rien.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

from aggregation_incident_v2 import AggregationSpec
from clients import OllamaClient

logger = logging.getLogger(__name__)

LLM_MODEL = os.environ.get("LLM_MODEL_INCIDENT_V2", "qwen2.5:7b")
LLM_TEMPERATURE = 0.1

_PHRASE_PROMPT = """\
Tu es un assistant expert en analyse d'incidents aéronautiques.
On t'a calculé les chiffres suivants. UTILISE EXACTEMENT CES NOMBRES, sans rien recalculer \
ni inventer.

Question : {question}

Résultats calculés :
{results_block}

CONSIGNES :
- Commence par une phrase d'accroche chiffrée (ex: "X incidents correspondent…").
- 2 à 4 phrases au total. Pas de liste à puces. Français naturel.
- Cite les chiffres exacts fournis ci-dessus, sans les modifier.

Réponse :"""


def phrase_result(
    question: str,
    spec: AggregationSpec,
    rows: list[dict],
    total: Optional[int],
    ollama: OllamaClient,
) -> str:
    results_block = _format_results(spec, rows, total)
    prompt = _PHRASE_PROMPT.format(question=question, results_block=results_block)
    return ollama.generate(prompt=prompt, model=LLM_MODEL, temperature=LLM_TEMPERATURE)


def _format_results(spec: AggregationSpec, rows: list[dict], total: Optional[int]) -> str:
    lines: list[str] = []

    filters = {k: v for k, v in spec.model_dump().items() if k.startswith("f_") and v is not None}
    if filters:
        filter_str = ", ".join(f"{k}={v}" for k, v in filters.items())
        lines.append(f"Filtres appliqués : {filter_str}")

    if spec.group_by is None:
        lines.append(f"Total : {total:,} incidents")
    else:
        lines.append(f"Répartition par {spec.group_by} (total={total:,}) :")
        for row in rows:
            lines.append(f"  - {row['label']} : {row['n']:,}")

    return "\n".join(lines)
