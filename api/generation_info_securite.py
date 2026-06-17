from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from clients import OllamaClient
from retrieval_info_securite import RetrievalResultIS, RetrievedInfoSecurite

logger = logging.getLogger(__name__)

LLM_MODEL = "qwen2.5:14b"
LLM_TEMPERATURE = 0.1
MAX_TEXT_LENGTH = 500

NO_RESULT_MESSAGE = (
    "Je n'ai pas trouve d'Information Securite pertinente dans la base "
    "pour repondre a cette question. Reformulez avec des termes plus precis."
)


@dataclass
class GenerationResultIS:
    answer: str
    model_used: str
    duration_ms: int


def generate_answer_is(
    question: str,
    retrieval_result: RetrievalResultIS,
    ollama: OllamaClient,
) -> GenerationResultIS:
    start = time.time()

    if retrieval_result.below_threshold or not retrieval_result.items:
        return GenerationResultIS(
            answer=NO_RESULT_MESSAGE,
            model_used="none",
            duration_ms=int((time.time() - start) * 1000),
        )

    prompt = _build_prompt(question, retrieval_result.items)
    answer = ollama.generate(prompt=prompt, model=LLM_MODEL, temperature=LLM_TEMPERATURE)

    return GenerationResultIS(
        answer=answer.strip(),
        model_used=LLM_MODEL,
        duration_ms=int((time.time() - start) * 1000),
    )


def _build_prompt(question: str, items: list[RetrievedInfoSecurite]) -> str:
    context = _format_context(items)
    return f"""Tu es un expert en securite aeronautique, specialiste des Information Securite (IS) de la DGAC.

REGLES IMPERATIVES :
1. Reponds UNIQUEMENT a partir des IS listees dans le CONTEXTE ci-dessous.
2. N'invente AUCUNE information absente du contexte.
3. Les IS du contexte ont été selectionnees par similarite semantique. Utilise-les si elles abordent le meme domaine de securite aeronautique que la question.
4. Si une IS a un lien indirect mais reel avec la question, exploite-la en precisant le lien.
5. Si les IS ne concernent PAS le sujet de la question (domaine completement different, ex: maintenance moteur vs securite operationnelle), reponds uniquement : "Aucune IS directement pertinente trouvee pour cette question."

FORMAT :
- Francais naturel, synthese experte.
- Cite les numeros IS de maniere fluide (ex: "...comme precise dans l'IS 2021/03").
- Privilegier la synthese par theme plutot qu'une liste brute.
- 4 a 8 phrases maximum.

CONTEXTE - INFORMATIONS SECURITE DISPONIBLES :

Note : ces IS ont ete selectionnees par similarite semantique avec la question — utilise-les meme si le lien est indirect.

{context}


QUESTION :
{question}

REPONSE :"""


def _format_context(items: list[RetrievedInfoSecurite]) -> str:
    blocks = []
    for item in items:
        lines = []
        if item.is_number:
            lines.append(f"IS : {item.is_number}")
        if item.annee:
            lines.append(f"Annee : {item.annee}")
        if item.titre:
            lines.append(f"Titre : {item.titre}")
        if item.operateurs_concernes:
            lines.append(f"Operateurs concernes : {item.operateurs_concernes}")
        if item.llm_resume:
            lines.append(f"Resume : {item.llm_resume[:MAX_TEXT_LENGTH]}")
        elif item.sujet:
            lines.append(f"Sujet : {item.sujet[:MAX_TEXT_LENGTH]}")
        if item.actions_recommandees:
            lines.append(f"Actions recommandees : {item.actions_recommandees[:MAX_TEXT_LENGTH]}")
        if item.remplace:
            lines.append(f"Remplace : {', '.join(item.remplace)}")
        blocks.append("\n".join(lines))
    return "\n\n---\n\n".join(blocks)
