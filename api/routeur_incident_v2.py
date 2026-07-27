"""
Routeur d'intention — point d'entrée unique qui CLASSE la question dans une
capacité, puis laisse le code DISPATCHER (doctrine : le LLM classe en vocabulaire
fermé, il ne choisit ni n'exécute la voie ; sortie structurée, jamais de texte
libre à parser ; prompt externalisé prompts/routeur.txt).

Usage :
    from routeur_incident_v2 import router
    capacite = router(question, ollama)["capacite"]  # -> agregation|recherche|actions|recommandation|abstention
    endpoint = DISPATCH[capacite]
"""
from __future__ import annotations

import json
import os

import prompt_store
from clients import OllamaClient

LLM_MODEL = os.environ.get("ROUTEUR_MODEL", "qwen2.5:7b")  # 1er hop léger : 7b (14b sur-tirait recherche sur les reco)

CAPACITES = ["agregation", "recherche", "actions", "recommandation", "abstention"]

# Aiguillage déterministe capacité -> voie (endpoint) — pas de décision LLM ici.
DISPATCH = {
    "agregation":     "/ask/incident-v2/query",     # moteur générique comptage/répartition
    "recherche":      "/ask/incident-v2",           # recherche sémantique
    "actions":        "/ask/incident-v2/actions",   # actions correctives/préventives
    "recommandation": "/ask/incident-v2/recommande",
    "abstention":     None,                          # message d'abstention, aucune voie
}

_SCHEMA = {
    "type": "object",
    "properties": {
        "capacite": {"type": "string", "enum": CAPACITES},
        "justification": {"type": "string"},
    },
    "required": ["capacite", "justification"],
}


def router(question: str, ollama: OllamaClient, modele: str | None = None) -> dict:
    """Classe la question dans une capacité. Retourne {capacite, justification}."""
    prompt = prompt_store.rendre("routeur", question=question)
    for tentative in range(2):
        try:
            raw = ollama.generate_structured(prompt, _SCHEMA, model=modele or LLM_MODEL)
            out = json.loads(raw)
            if out.get("capacite") in CAPACITES:
                return out
        except Exception as e:
            if tentative == 1:
                return {"capacite": "abstention", "justification": f"erreur routeur: {e}"}
    return {"capacite": "abstention", "justification": "capacité non reconnue"}
