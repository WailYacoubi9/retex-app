from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass

from clients import OllamaClient
from retrieval_tickets import RetrievalResultTickets, RetrievedTicket

logger = logging.getLogger(__name__)

# Surchargeable via env TICKETS_LLM_MODEL. Defaut 7b (tient dans 8 Go VRAM).
LLM_MODEL = os.environ.get("TICKETS_LLM_MODEL", "qwen2.5:7b")
LLM_TEMPERATURE = 0.2
MAX_TITRE_LENGTH = 200
MAX_DETAIL_LENGTH = 600
MAX_RESUME_LENGTH = 500

# Libelles lisibles des relations d'expansion graphe
REL_LABELS = {
    "via_enfant": "ticket parent/enfant",
    "via_expert": "meme intervenant",
    "via_projet": "meme projet",
    "via_client": "meme client",
}

NO_RESULT_MESSAGE = (
    "Aucun ticket correspondant trouve dans la base intra'know "
    "pour cette question. Essayez avec des termes differents "
    "(nom d'application, type de probleme, fonctionnalite)."
)


@dataclass
class GenerationResultTickets:
    answer: str
    model_used: str
    duration_ms: int


def generate_answer_tickets(
    question: str,
    retrieval_result: RetrievalResultTickets,
    ollama: OllamaClient,
) -> GenerationResultTickets:
    start = time.time()

    if retrieval_result.below_threshold or not retrieval_result.items:
        return GenerationResultTickets(
            answer=NO_RESULT_MESSAGE,
            model_used="none",
            duration_ms=int((time.time() - start) * 1000),
        )

    prompt = _build_prompt(question, retrieval_result.items)
    answer = ollama.generate(prompt=prompt, model=LLM_MODEL, temperature=LLM_TEMPERATURE)

    return GenerationResultTickets(
        answer=answer.strip(),
        model_used=LLM_MODEL,
        duration_ms=int((time.time() - start) * 1000),
    )


def _build_prompt(question: str, items: list[RetrievedTicket]) -> str:
    direct = [t for t in items if not t.is_expanded]
    expanded = [t for t in items if t.is_expanded]
    context = _format_context(direct, expanded)

    return f"""Tu es un assistant expert qui analyse les tickets de support et de maintenance \
du systeme intra'know pour aider les consultants a identifier les problemes recurrents.

INSTRUCTIONS :
1. Utilise UNIQUEMENT les informations des tickets fournis dans le CONTEXTE.
2. N'invente aucune information absente du contexte.
3. Si aucun ticket ne correspond vraiment a la question, dis-le clairement en une phrase.
4. Sinon, fais une synthese utile et concrete pour un consultant.

FORMAT DE REPONSE :
- Commence par une phrase de synthese globale.
- Cite les numeros de tickets concernes (ex : ticket #32446).
- Mentionne les applications, projets, structures et types de problemes identifies.
- Si des tickets lies ont ete trouves par proximite (meme application ou projet), \
signale-le en fin de reponse.
- Maximum 8 phrases. Francais professionnel.

CONTEXTE - TICKETS INTRA'KNOW :

{context}

QUESTION : {question}

REPONSE :"""


def _format_context(
    direct: list[RetrievedTicket],
    expanded: list[RetrievedTicket],
) -> str:
    blocks = []

    for item in direct:
        lines = [f"[ Ticket #{item.numero_fe} ]"]
        if item.type_nc:
            lines.append(f"  Type        : {item.type_nc}")
        if item.importance:
            lines.append(f"  Importance  : {item.importance}")
        if item.urgence:
            lines.append(f"  Urgence     : {item.urgence}")
        if item.etat or item.etape_label:
            lines.append(f"  Etat        : {item.etat or item.etape_label}")
        if item.site_application:
            lines.append(f"  Application : {item.site_application}")
        if item.projet_nom:
            lines.append(f"  Projet      : {item.projet_nom}")
        if item.client:
            lines.append(f"  Client      : {item.client}")
        if item.structure:
            lines.append(f"  Structure   : {item.structure}")
        if item.llm_domaine_technique:
            lines.append(f"  Domaine     : {item.llm_domaine_technique}")
        if item.date_nc:
            lines.append(f"  Date        : {item.date_nc}")
        if item.llm_resume:
            lines.append(f"  Resume      : {item.llm_resume[:MAX_RESUME_LENGTH]}")
        elif item.titre:
            lines.append(f"  Titre       : {item.titre[:MAX_TITRE_LENGTH]}")
            if item.detail:
                lines.append(f"  Detail      : {item.detail[:MAX_DETAIL_LENGTH]}")
        blocks.append("\n".join(lines))

    if expanded:
        blocks.append("--- Tickets lies par le graphe (contexte) ---")
        for item in expanded:
            rel = item.matched_fields[0] if item.matched_fields else ""
            lien = REL_LABELS.get(rel, "contexte")
            lines = [f"[ Ticket #{item.numero_fe} — lie par {lien} ]"]
            if item.site_application:
                lines.append(f"  Application : {item.site_application}")
            if item.projet_nom:
                lines.append(f"  Projet      : {item.projet_nom}")
            if item.client:
                lines.append(f"  Client      : {item.client}")
            if item.importance:
                lines.append(f"  Importance  : {item.importance}")
            if item.llm_resume:
                lines.append(f"  Resume      : {item.llm_resume[:300]}")
            elif item.titre:
                lines.append(f"  Titre       : {item.titre[:MAX_TITRE_LENGTH]}")
            blocks.append("\n".join(lines))

    return "\n\n".join(blocks)
