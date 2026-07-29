"""Synonymes métier + normalisation lexicale pour le retrieval hybride.

But : combler l'écart de SURFACE que le dense (bge-m3) ne voit pas toujours à
l'échelle du mot (ex. la requête « carburant » doit matcher un texte « kérosène »
ou « avitaillement »). Sert UNIQUEMENT à l'INCLUSION côté BM25 (rappel), jamais à
écarter un candidat. Le reranker cross-encoder en aval reste le juge de précision.

- `normalize` / `tokenize` : même fonction utilisée à la CONSTRUCTION de l'index
  BM25 et à la REQUÊTE (cohérence indispensable).
- `expand_terms` : ajoute les formes de surface des synonymes d'un terme (pour que
  BM25 score un doc contenant n'importe quelle variante du groupe).
- `canonical_of` : réduit une variante à l'identifiant de son groupe, pour COMPTER
  le recouvrement lexical sans double-compter les synonymes (plancher d'abstention).

Aucune dépendance. Les tokens des groupes sont déjà NORMALISÉS (sans accent, minuscule).
"""
from __future__ import annotations

import re
import unicodedata

# Petit set de mots-outils FR : retirés pour ne garder que des termes de contenu.
STOPWORDS: frozenset[str] = frozenset({
    "le", "la", "les", "un", "une", "des", "du", "de", "et", "ou", "a", "au", "aux",
    "en", "dans", "sur", "sous", "par", "pour", "avec", "sans", "que", "qui", "quoi",
    "dont", "ne", "pas", "plus", "se", "sa", "son", "ses", "ce", "cet", "cette", "ces",
    "il", "elle", "on", "nous", "vous", "ils", "elles", "est", "ete", "etre", "aux",
    "leur", "leurs", "mon", "ma", "mes", "notre", "votre", "y", "d", "l", "s", "n", "c",
})

# Groupes de synonymes métier (tokens NORMALISÉS). Ajouts sûrs : termes distinctifs
# du domaine aéroportuaire. On évite volontairement les mots ambigus (ex. « jet » seul).
GROUPS: list[frozenset[str]] = [
    frozenset({"carburant", "kerosene", "avitaillement", "hydrocarbure", "fuel", "carbu", "jeta1"}),
    frozenset({"oiseau", "oiseaux", "aviaire", "volatile", "volatiles", "birdstrike", "animalier", "ornithologique", "bird"}),
    frozenset({"fod", "debris"}),
    frozenset({"foudre", "orage", "eclair", "lightning"}),
    frozenset({"laser", "eblouissement"}),
    frozenset({"drone", "uas", "uav", "telepilote", "aeromodele"}),
]

_TERM_TO_GROUP: dict[str, int] = {}
for _i, _grp in enumerate(GROUPS):
    for _t in _grp:
        _TERM_TO_GROUP[_t] = _i

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text)
                   if unicodedata.category(c) != "Mn")


def normalize(text: str) -> str:
    """Minuscule + accents retirés (NFD). Base commune index/requête."""
    return _strip_accents(text or "").lower()


def tokenize(text: str) -> list[str]:
    """Tokens de contenu : alphanumériques, longueur >= 2, hors stopwords."""
    return [t for t in _TOKEN_RE.findall(normalize(text))
            if len(t) >= 2 and t not in STOPWORDS]


def expand_terms(tokens) -> set[str]:
    """Tokens d'origine ∪ toutes les formes de surface des groupes déclenchés."""
    out: set[str] = set(tokens)
    for t in set(tokens):
        gi = _TERM_TO_GROUP.get(t)
        if gi is not None:
            out |= GROUPS[gi]
    return out


def canonical_of(token: str) -> str:
    """Identifiant de groupe si le token est un synonyme connu, sinon le token.
    Sert au comptage du recouvrement lexical (un groupe = un terme canonique)."""
    gi = _TERM_TO_GROUP.get(token)
    return f"__g{gi}__" if gi is not None else token
