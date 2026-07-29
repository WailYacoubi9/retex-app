"""
Phrasage des recommandations d'actions (recommendation_incident_v2).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import glossaire
import prompt_store
from clients import OllamaClient
from recommendation_incident_v2 import RecommandationResult

logger = logging.getLogger(__name__)

LLM_MODEL = "qwen2.5:7b"

NO_MATCH_MESSAGE = (
    "Aucun cas comparable dans la base : je préfère ne pas recommander d'actions "
    "plutôt que d'en inventer à partir d'un incident sans rapport. Décrivez "
    "l'événement avec plus de détails (lieu, matériel, type d'événement)."
)


@dataclass
class RecoGenResult:
    answer: str
    model_used: str





def _contexte(result: RecommandationResult) -> str:
    lignes = []
    for a in result.actions[:30]:
        n = len(a.fe_sources)
        recur = f" (récurrente : {n} cas)" if n > 1 else ""
        statut = f" [statut : {a.statut}]" if a.statut else ""
        # PAS de codes de fiche ici : la prose ne doit jamais les citer
        # (la liste des fiches sources est affichée séparément, sous la réponse).
        lignes.append(f'- [{a.type_action.upper()}] "{a.titre}"{recur}{statut}')
    return "\n".join(lignes) if lignes else "Aucune action documentée sur ces incidents."


def phrase_recommendation(
    description: str,
    result: RecommandationResult,
    ollama: OllamaClient,
) -> RecoGenResult:
    if result.below_threshold or not result.incidents:
        return RecoGenResult(answer=NO_MATCH_MESSAGE, model_used="deterministe")

    if not result.actions:
        return RecoGenResult(
            answer=("Des cas similaires existent dans la base, mais aucune action n'y est "
                    "documentée. La liste des fiches concernées est affichée ci-dessous."),
            model_used="deterministe",
        )

    prompt = prompt_store.rendre(
        "recommandation.reponse",
        description=description,
        n_incidents=len(result.incidents),
        contexte=_contexte(result),
        glossaire=glossaire.bloc_glossaire(),
    )
    try:
        answer = ollama.generate(prompt, model=LLM_MODEL)
        return RecoGenResult(answer=answer.strip(), model_used=LLM_MODEL)
    except Exception as e:
        logger.error("Génération recommandation échouée : %s", e)
        lignes = [f"Actions relevées sur {len(result.incidents)} cas similaires :"]
        for a in result.actions[:15]:
            n = len(a.fe_sources)
            recur = f" (×{n} cas)" if n > 1 else ""
            lignes.append(f"- [{a.type_action}] {a.titre}{recur}")
        return RecoGenResult(answer="\n".join(lignes), model_used="fallback")
