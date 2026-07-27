"""
Politique « données nominatives » de l'assistant (culture juste / RGPD) et ses
outils d'application. UN SEUL endroit porte la règle :

  - INTERDIT partout (réponse, prompt, diagnostic, log) : identifiants de
    personnes — Personne.login, Action.responsable, Notifiant, individus tickets.
  - AUTORISÉ : le niveau organisationnel de pilotage (Service).
  - Societe / Entite (MIS_EN_CAUSE, CONCERNE) : visibles dans le contexte d'une
    fiche individuelle, JAMAIS en classement ni dimension d'analyse
    (classement de responsabilité = contraire à la culture juste).
  - Un filtre par personne (« les actions de X ») se REFUSE explicitement —
    jamais de retrait silencieux (anti-pattern « contrainte évaporée »).

Défense en profondeur : les voies retirent le nominatif À LA SOURCE (registres,
requêtes, prompts) ; ce module fournit en plus le FILET central — une deny-list
énumérée depuis le graphe (les logins existent en base, donc ils s'énumèrent)
compilée en un masque appliqué à toute sortie JSON et au log de capture.
"""
from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger("api.confidentialite")

MASQUE = "⟨agent⟩"

REFUS_PERSONNE = (
    "Je ne fournis pas d'informations rattachées à une personne (culture juste : "
    "le REX analyse le système, pas les individus). Je peux en revanche donner "
    "la répartition par type d'action, par statut, ou par service."
)

# Un login plausible : commence par une lettre, ≥ 4 caractères au total —
# en dessous, le risque de masquer un mot courant dépasse le bénéfice.
_LOGIN_VALIDE_RE = re.compile(r"^[A-Za-zÀ-ÿ][\w.\-]{3,}$")

_masque_re: re.Pattern | None = None
_n_termes: int = 0


def charger(neo4j) -> None:
    """Énumère les identifiants de personnes depuis le graphe et compile le masque.
    Ne lève jamais : sans deny-list, le masquage est simplement inactif (les
    retraits à la source restent en place)."""
    global _masque_re, _n_termes
    try:
        rows = neo4j.run(
            "MATCH (p:Personne) WHERE p.login IS NOT NULL "
            "RETURN DISTINCT p.login AS t "
            "UNION "
            "MATCH (i:IncidentSecu)-[:A_POUR_ACTION]->(a:Action) "
            "WHERE a.responsable IS NOT NULL RETURN DISTINCT a.responsable AS t")
        termes = sorted({str(r["t"]).strip() for r in rows
                         if r["t"] and _LOGIN_VALIDE_RE.match(str(r["t"]).strip())},
                        key=len, reverse=True)
        if not termes:
            logger.warning("Confidentialité : aucune deny-list (0 terme) — masque inactif")
            return
        motif = r"(?<![\w@.\-])(?:" + "|".join(re.escape(t) for t in termes) + r")(?![\w@.\-])"
        _masque_re = re.compile(motif, re.IGNORECASE)
        _n_termes = len(termes)
        logger.info("Confidentialité : deny-list chargée (%d identifiants)", _n_termes)
    except Exception:  # noqa: BLE001
        logger.exception("Confidentialité : chargement de la deny-list échoué — masque inactif")


def actif() -> bool:
    return _masque_re is not None


def masquer_texte(texte: str) -> str:
    if not texte or _masque_re is None:
        return texte
    return _masque_re.sub(MASQUE, texte)


def masquer_nominatif(obj: Any) -> Any:
    """Masque récursivement toute chaîne d'une structure (dict/list/str)."""
    if _masque_re is None or obj is None:
        return obj
    if isinstance(obj, str):
        return masquer_texte(obj)
    if isinstance(obj, dict):
        return {k: masquer_nominatif(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [masquer_nominatif(v) for v in obj]
    return obj
