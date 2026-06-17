"""
Couche de generation du RAG v2 : construction du prompt + appel LLM.

Ameliorations v2 :
  - Prompt plus directif (synthese forcee, citations fluides)
  - Marqueurs internes ([RESULTAT DIRECT] etc.) retires du contexte
  - Instruction explicite d'ecarter les incidents non pertinents
  - Limite stricte sur le nombre de citations FE dans la reponse
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

from clients import OllamaClient
from retrieval import RetrievalResult, RetrievedIncident

logger = logging.getLogger(__name__)

LLM_MODEL = "llama3.1:8b"
LLM_TEMPERATURE = 0.1

MAX_DETAIL_LENGTH = 500
MAX_RESUME_LENGTH = 300

NO_RESULT_MESSAGE = (
    "Je n'ai pas trouve d'informations pertinentes dans la base "
    "d'incidents pour repondre a cette question. Essayez de "
    "reformuler avec des termes plus precis ou differents."
)


@dataclass
class GenerationResult:
    """Resultat complet d'une operation de generation."""
    answer: str
    model_used: str
    duration_ms: int


def generate_answer(
    question: str,
    retrieval_result: RetrievalResult,
    ollama: OllamaClient,
) -> GenerationResult:
    """Genere une reponse RAG a partir du retrieval."""
    start = time.time()

    if retrieval_result.below_threshold or not retrieval_result.incidents:
        duration_ms = int((time.time() - start) * 1000)
        return GenerationResult(
            answer=NO_RESULT_MESSAGE,
            model_used="none",
            duration_ms=duration_ms,
        )

    prompt = _build_rag_prompt(question, retrieval_result.incidents)
    logger.debug("Prompt length: %d chars", len(prompt))

    answer = ollama.generate(
        prompt=prompt,
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
    )

    duration_ms = int((time.time() - start) * 1000)
    return GenerationResult(
        answer=answer.strip(),
        model_used=LLM_MODEL,
        duration_ms=duration_ms,
    )


def _build_rag_prompt(question: str, incidents: list[RetrievedIncident]) -> str:
    """Construit le prompt complet a envoyer au LLM."""
    context = _format_incidents_as_context(incidents)

    return f"""Tu es un assistant expert en analyse d'incidents de securite aeronautique.

REGLES IMPERATIVES :
1. Reponds UNIQUEMENT a partir des incidents listes dans le CONTEXTE ci-dessous.
2. N'invente AUCUNE information absente du contexte.
3. SI certains incidents du contexte ne sont PAS pertinents pour la question, IGNORE-les silencieusement. Ne les mentionne pas, ne dis pas qu'ils ne sont pas pertinents.
4. Si AUCUN incident du contexte n'est pertinent, reponds simplement : "Aucun incident pertinent trouve pour cette question."

FORMAT DE LA REPONSE :
- Reponds en francais naturel et fluide, comme un expert qui synthetise.
- Cite les numeros FE de maniere fluide dans tes phrases (ex: "...comme l'illustre l'incident FNE/26/0241").
- PRIVILEGIE LA SYNTHESE par categories ou themes, plutot qu'une liste brute d'incidents.
- Cite au maximum 5 numeros FE dans ta reponse, meme si plus sont fournis. Choisis les plus representatifs.
- 4 a 8 phrases maximum.
- N'utilise JAMAIS de formulations comme "Incident 1", "Incident 2", ou "[RESULTAT DIRECT]". Ces mentions sont des marqueurs internes, pas des elements de reponse.

CONTEXTE - INCIDENTS DISPONIBLES :

{context}

QUESTION DE L'UTILISATEUR :
{question}

REPONSE :"""


def _format_incidents_as_context(incidents: list[RetrievedIncident]) -> str:
    """Formate les incidents en bloc de texte pour le contexte du prompt.

    Utilisation : pas de marqueurs visibles [RESULTAT DIRECT] qui seraient
    recopies par le LLM. Format plus discret et naturel.
    """
    blocks: list[str] = []

    for inc in incidents:
        lines = []

        if inc.numero_fe:
            lines.append(f"Numero FE : {inc.numero_fe}")
        if inc.titre:
            lines.append(f"Titre : {inc.titre}")

        if inc.date_evenement:
            date_short = inc.date_evenement[:10]
            lines.append(f"Date : {date_short}")

        classif_parts = []
        if inc.facteur_causal:
            classif_parts.append(f"facteur {inc.facteur_causal}")
        if inc.severite_percue:
            classif_parts.append(f"severite {inc.severite_percue}")
        if inc.etat_final:
            classif_parts.append(f"etat {inc.etat_final}")
        if classif_parts:
            lines.append(f"Classification : {', '.join(classif_parts)}")

        if inc.resume_llm:
            resume = inc.resume_llm[:MAX_RESUME_LENGTH]
            if len(inc.resume_llm) > MAX_RESUME_LENGTH:
                resume += "..."
            lines.append(f"Description : {resume}")
        elif inc.detail:
            detail = inc.detail[:MAX_DETAIL_LENGTH]
            if len(inc.detail) > MAX_DETAIL_LENGTH:
                detail += "..."
            lines.append(f"Description : {detail}")

        if inc.referentiels:
            refs_str = ", ".join(
                f"{r['label']} ({r['relation']})"
                for r in inc.referentiels if r.get('label')
            )
            if refs_str:
                lines.append(f"Entites liees : {refs_str}")

        blocks.append("\n".join(lines))

    # Separateur discret entre incidents (pas de numerotation visible)
    return "\n\n---\n\n".join(blocks)


def build_sources(incidents: list[RetrievedIncident]) -> list[dict]:
    """Transforme les RetrievedIncident en liste de sources pour l'API."""
    sources = []
    for inc in incidents:
        source = {
            "incident_id": inc.incident_id,
            "incident_id_source": inc.incident_id_source,
            "numero_fe": inc.numero_fe,
            "titre": inc.titre,
            "resume_llm": inc.resume_llm,
            "facteur_causal": inc.facteur_causal,
            "severite_percue": inc.severite_percue,
            "date_evenement": inc.date_evenement,
            "best_score": inc.best_score,
            "matched_fields": inc.matched_fields,
        }
        sources.append(source)
    return sources
