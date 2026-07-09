"""
Recherche d'incidents par nom d'entité satellite (:Compagnie, :Societe, :Personne…).

Pipeline :
  nom_entite → Neo4j (recherche insensible à la casse sur les nœuds satellites)
              → incidents liés (avec resume_llm ou detail)
              → rows structurées pour generation_entity_lookup
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from clients import Neo4jClient

logger = logging.getLogger(__name__)

# Labels satellites explorés : (label, propriété de nom, relation vers IncidentSecu)
# Seuls les labels/relations effectivement présents dans le graphe.
_ENTITY_LABELS = [
    ("Compagnie",     "nom",   "IMPLIQUE_COMPAGNIE"),
    ("Societe",       "nom",   "CONCERNE"),
    ("Entite",        "nom",   "MIS_EN_CAUSE"),
    ("TypeAeronef",   "nom",   "IMPLIQUE_AERONEF"),
    ("TypeEvenement", "label", "DE_TYPE"),
    ("Lieu",          "label", "LOCALISE_EN"),
    ("PhaseVol",      "label", "EN_PHASE_DE_VOL"),
    ("Notifiant",     "label", "NOTIFIE_PAR"),
    ("Service",       "label", "RESPONSABLE"),
    ("Personne",      "login", "EMIS_PAR"),
]

# Mots vides à ignorer lors de la tokenisation
_STOPWORDS = {"la", "le", "les", "de", "du", "des", "un", "une", "et", "ou",
              "sur", "en", "par", "pour", "dans", "au", "aux", "avec",
              "compagnie", "societe", "société", "groupe", "agence",
              "the", "and", "or", "of", "for", "in", "company"}

_SAMPLE_LIMIT = 50


@dataclass
class EntityMatch:
    label: str
    rel: str
    entity_name: str
    incident_count: int
    incidents: list[dict] = field(default_factory=list)


def _tokenize(query: str) -> list[str]:
    """Extrait les termes significatifs d'une requête utilisateur."""
    words = re.findall(r"[A-Za-zÀ-ÿ0-9]+", query.lower())
    return [w for w in words if len(w) >= 2 and w not in _STOPWORDS] or [query.lower()]


def fetch_entity(name: str, neo4j: Neo4jClient) -> list[EntityMatch]:
    """
    Recherche tous les nœuds satellites dont la propriété de nom correspond à `name`
    (insensible à la casse, tokenisation des mots clés). Retourne les EntityMatch
    avec un échantillon d'incidents incluant resume_llm pour le phrasage.
    """
    terms = _tokenize(name)
    logger.info("Entity lookup terms: %s", terms)
    results: list[EntityMatch] = []

    for lbl, prop, rel in _ENTITY_LABELS:
        rows = neo4j.run(
            f"MATCH (i:IncidentSecu)-[:{rel}]->(e:`{lbl}`) "
            f"WHERE any(term IN $terms WHERE toLower(e.`{prop}`) CONTAINS term) "
            f"WITH e.`{prop}` AS entity_name, "
            f"     collect(DISTINCT {{fe:i.numero_fe, titre:i.titre, "
            f"                        severite:i.severite, date:i.date_evenement, "
            f"                        resume:i.resume_llm}}) AS all_incidents "
            f"RETURN entity_name, size(all_incidents) AS total, "
            f"       all_incidents[0..{_SAMPLE_LIMIT}] AS incidents",
            terms=terms,
        )
        for row in rows:
            results.append(EntityMatch(
                label=lbl,
                rel=rel,
                entity_name=str(row["entity_name"]),
                incident_count=int(row["total"]),
                incidents=row["incidents"] or [],
            ))

    results.sort(key=lambda m: -m.incident_count)
    return results
