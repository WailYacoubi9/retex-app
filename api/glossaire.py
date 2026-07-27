"""
Glossaire des sigles métier — chargé depuis glossaire_sigles.json (validé métier),
rendu en bloc texte injectable dans les prompts (parseur, résumé, génération).

Même principe que prompt_store : rechargé si le fichier change, aucune
manipulation technique après édition du glossaire.

Le glossaire ne contient QUE des sigles non ambigus (les ambigus sont écartés
en amont, cf. meta du fichier) : ses expansions sont donc sûres à injecter.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_PATH = Path(__file__).resolve().parent / "glossaire_sigles.json"
_cache: tuple[float, list[tuple[str, str]]] | None = None


def _charger() -> list[tuple[str, str]]:
    """Liste (sigle, signification) validés, rechargée si le fichier change."""
    global _cache
    try:
        mtime = _PATH.stat().st_mtime
    except FileNotFoundError:
        return []
    if _cache is None or _cache[0] != mtime:
        d = json.loads(_PATH.read_text(encoding="utf-8"))
        paires: list[tuple[str, str]] = []
        for bloc in ("sigles_deja_definis_a_verifier", "sigles_a_definir"):
            for x in d.get(bloc, []):
                sig = (x.get("sigle") or "").strip()
                sens = (x.get("signification") or "").strip()
                # on n'injecte que les entrées réellement définies
                if sig and sens:
                    paires.append((sig, sens))
        # dédoublonnage en gardant l'ordre, tri alphabétique pour stabilité
        vus, uniques = set(), []
        for s, v in paires:
            if s.upper() not in vus:
                vus.add(s.upper())
                uniques.append((s, v))
        uniques.sort(key=lambda p: p[0].upper())
        _cache = (mtime, uniques)
        logger.info("Glossaire sigles chargé : %d entrées", len(uniques))
    return _cache[1]


def paires() -> list[tuple[str, str]]:
    return _charger()


def bloc_glossaire(entete: str = "GLOSSAIRE DES SIGLES (significations validées — n'en invente aucune autre) :") -> str:
    """Bloc texte prêt à injecter dans un prompt. Vide si glossaire absent."""
    p = _charger()
    if not p:
        return ""
    lignes = [entete] + [f"- {s} = {v}" for s, v in p]
    return "\n".join(lignes)
