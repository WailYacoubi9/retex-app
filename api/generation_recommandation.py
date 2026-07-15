"""
Phrasage des recommandations d'actions (recommendation_incident_v2).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import prompt_store
from clients import OllamaClient
from recommendation_incident_v2 import RecommandationResult

logger = logging.getLogger(__name__)

LLM_MODEL = "qwen2.5:7b"

NO_MATCH_MESSAGE = (
    "Je n'ai pas trouvé d'incident suffisamment similaire dans la base pour "
    "proposer des actions. Décrivez l'événement avec plus de détails "
    "(lieu, matériel, type d'événement)."
)


@dataclass
class RecoGenResult:
    answer: str
    model_used: str





def _contexte(result: RecommandationResult) -> str:
    lignes = []
    for a in result.actions[:30]:
        statut = f" | statut : {a.statut}" if a.statut else ""
        resp = f" | responsable : {a.responsable}" if a.responsable else ""
        lignes.append(
            f"- [{a.type_action.upper()}] \"{a.titre}\" "
            f"| fiches : {', '.join(a.fe_sources[:5])}{statut}{resp}"
        )
    return "\n".join(lignes) if lignes else "Aucune action documentée sur ces incidents."


def phrase_recommendation(
    description: str,
    result: RecommandationResult,
    ollama: OllamaClient,
) -> RecoGenResult:
    if result.below_threshold or not result.incidents:
        return RecoGenResult(answer=NO_MATCH_MESSAGE, model_used="deterministe")

    if not result.actions:
        fes = ", ".join(i["numero_fe"] for i in result.incidents[:5])
        return RecoGenResult(
            answer=(f"Des incidents similaires existent ({fes}) mais aucune action "
                    f"n'y est documentée dans la base."),
            model_used="deterministe",
        )

    prompt = prompt_store.rendre(
        "recommandation.reponse",
        description=description,
        n_incidents=len(result.incidents),
        contexte=_contexte(result),
    )
    try:
        answer = ollama.generate(prompt, model=LLM_MODEL)
        return RecoGenResult(answer=answer.strip(), model_used=LLM_MODEL)
    except Exception as e:
        logger.error("Génération recommandation échouée : %s", e)
        lignes = [f"Actions relevées sur {len(result.incidents)} incidents similaires :"]
        for a in result.actions[:15]:
            lignes.append(f"- [{a.type_action}] {a.titre} ({', '.join(a.fe_sources[:3])})")
        return RecoGenResult(answer="\n".join(lignes), model_used="fallback")
