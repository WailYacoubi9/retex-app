"""
Magasin de prompts — les instructions LLM vivent dans api/prompts/*.txt.

Objectifs :
  - Prompts modifiables par un NON-TECHNICIEN : de simples fichiers texte,
    nommés d'après les onglets de l'interface (voir prompts/README.md).
  - PRISE D'EFFET IMMÉDIATE : chaque fichier est rechargé dès qu'il change
    (vérification de sa date de modification à chaque appel) — aucune
    manipulation technique après édition.
  - Rendu par remplacement littéral des {placeholders} : des accolades
    isolées ajoutées par un éditeur ne cassent rien (contrairement à
    str.format, qui lèverait une exception).
"""
from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_DIR = Path(__file__).resolve().parent / "prompts"
_cache: dict[str, tuple[float, str]] = {}


def charger(nom: str) -> str:
    """Contenu du prompt `nom` (fichier prompts/<nom>.txt), rechargé si modifié."""
    chemin = _DIR / f"{nom}.txt"
    try:
        mtime = chemin.stat().st_mtime
    except FileNotFoundError:
        raise RuntimeError(
            f"Prompt introuvable : {chemin} — voir {_DIR / 'README.md'}"
        ) from None
    entree = _cache.get(nom)
    if entree is None or entree[0] != mtime:
        contenu = chemin.read_text(encoding="utf-8")
        _cache[nom] = (mtime, contenu)
        if entree is not None:
            logger.info("Prompt rechargé après modification : %s", nom)
    return _cache[nom][1]


def rendre(nom: str, **valeurs) -> str:
    """Prompt `nom` avec ses {placeholders} remplacés par les valeurs fournies."""
    texte = charger(nom)
    for cle, val in valeurs.items():
        texte = texte.replace("{" + cle + "}", str(val))
    return texte
