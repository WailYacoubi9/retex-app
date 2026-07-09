"""
Génération LLM pour les réponses sur les actions correctives/préventives.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import prompt_store
from clients import OllamaClient

logger = logging.getLogger(__name__)

LLM_MODEL = "qwen2.5:7b"


@dataclass
class ActionLookupResult:
    answer: str
    model_used: str





def _format_context(rows: list[dict]) -> str:
    if not rows:
        return "Aucune action trouvée."
    lines = []
    for r in rows:
        statut = r.get("statut") or "n/a"
        resp   = r.get("responsable") or "non renseigné"
        eff    = r.get("actions_efficaces")
        eff_txt = "" if eff is None else f" | Jugées efficaces : {'oui' if eff else 'non'}"
        lines.append(
            f"- [{r.get('type_action','?').upper()}] \"{r.get('titre_action','?')}\" "
            f"| Incident : {r.get('fe','?')} — {r.get('titre_incident','?')} "
            f"| Statut : {statut} | Responsable : {resp}{eff_txt}"
        )
    return "\n".join(lines)


def phrase_action_result(
    question: str,
    rows: list[dict],
    total: int | None,
    spec,
    ollama: OllamaClient,
) -> ActionLookupResult:

    if spec is not None and spec.question_type == "count":
        prompt = prompt_store.rendre("actions.reponse_comptage", question=question, total=total or 0)
    elif not rows:
        return ActionLookupResult(
            answer="Aucune action corrective ou préventive ne correspond à votre question dans la base de données.",
            model_used=LLM_MODEL,
        )
    else:
        context = _format_context(rows)
        prompt = prompt_store.rendre(
            "actions.reponse_liste",
            question=question, n=len(rows), total=total or len(rows), context=context,
        )

    try:
        answer = ollama.generate(prompt, model=LLM_MODEL)
        return ActionLookupResult(answer=answer.strip(), model_used=LLM_MODEL)
    except Exception as e:
        logger.error("LLM generation failed: %s", e)
        # Fallback texte brut sans LLM
        if rows:
            lines = [f"**{len(rows)} action(s) trouvée(s) :**\n"]
            for r in rows[:15]:
                lines.append(
                    f"- [{r.get('type_action','?')}] {r.get('titre_action','?')} "
                    f"({r.get('fe','?')} — {r.get('titre_incident','?')}) "
                    f"— {r.get('statut','?')}"
                )
            return ActionLookupResult(answer="\n".join(lines), model_used="fallback")
        return ActionLookupResult(
            answer=f"{total or 0} action(s) trouvée(s).",
            model_used="fallback",
        )
