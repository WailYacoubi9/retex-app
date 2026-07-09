"""
Génération de la réponse pour la voie entity-lookup incidents v2.

Le LLM ne recherche pas — il phrase uniquement les chiffres et échantillons
fournis par fetch_entity (Neo4j).
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass

from clients import OllamaClient
from entity_lookup_incident_v2 import EntityMatch

logger = logging.getLogger(__name__)

LLM_MODEL = os.environ.get("LLM_MODEL_INCIDENT_V2", "qwen2.5:7b")
LLM_TEMPERATURE = 0.1

_NO_RESULT = (
    "Aucune entité correspondant à cette recherche n'a été trouvée dans la base "
    "d'incidents. Essayez un nom plus court ou vérifiez l'orthographe."
)

_PHRASE_PROMPT = """\
Tu es un assistant expert en analyse d'incidents de sécurité aéronautique.

On t'a fourni la liste des incidents liés à l'entité "{entity_query}" dans notre base.
UTILISE EXACTEMENT LES CHIFFRES ET EXEMPLES FOURNIS, sans rien inventer.

ENTITÉS TROUVÉES :
{entities_block}

CONSIGNES :
- Commence par une phrase d'accroche mentionnant le nombre total d'incidents.
- Identifie les 2-3 thèmes dominants en t'appuyant sur les résumés fournis (champ "resume").
  Si le résumé est absent pour un incident, utilise le titre.
- 3 à 6 phrases au total. Français naturel, ton expert.
- Cite quelques numéros FE pour ancrer la réponse.

Réponse :"""


@dataclass
class EntityLookupResult:
    answer: str
    model_used: str
    duration_ms: int
    matches: list[EntityMatch]


def phrase_entity_result(
    entity_query: str,
    matches: list[EntityMatch],
    ollama: OllamaClient,
) -> EntityLookupResult:
    start = time.time()

    if not matches:
        return EntityLookupResult(
            answer=_NO_RESULT,
            model_used="none",
            duration_ms=0,
            matches=[],
        )

    entities_block = _format_matches(matches)
    prompt = _PHRASE_PROMPT.format(
        entity_query=entity_query,
        entities_block=entities_block,
    )
    answer = ollama.generate(prompt=prompt, model=LLM_MODEL, temperature=LLM_TEMPERATURE)

    return EntityLookupResult(
        answer=answer.strip(),
        model_used=LLM_MODEL,
        duration_ms=int((time.time() - start) * 1000),
        matches=matches,
    )


def _format_matches(matches: list[EntityMatch]) -> str:
    blocks: list[str] = []
    for m in matches:
        lines = [f"• {m.label} « {m.entity_name} » — {m.incident_count} incidents liés"]
        for inc in m.incidents[:10]:
            fe = inc.get("fe") or "?"
            titre = inc.get("titre") or ""
            resume = inc.get("resume") or ""
            desc = (resume[:150] + "…") if len(resume) > 150 else (resume or titre[:100])
            sev = inc.get("severite") or ""
            date = (inc.get("date") or "")[:10]
            lines.append(f"  - FE:{fe} [{date}] sév:{sev} | {desc}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)
