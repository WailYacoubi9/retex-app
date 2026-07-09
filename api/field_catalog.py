"""
Catalogue des champs d'un module — labels et descriptions métier.

Source de vérité : les nœuds (:ModuleMeta)-[:A_POUR_CHAMP]->(:ChampMeta)
publiés dans le graphe par l'ingestion (scripts/publish_schema_meta.py,
qui lit le schéma YAML du module). Ainsi le YAML reste l'unique référence :
YAML → graphe → API, aucune duplication à la main.

Fallback : si les métadonnées ne sont pas (encore) publiées, on retombe sur
le nom technique du champ — l'assistant fonctionne, juste moins bien guidé.
"""
from __future__ import annotations

import logging
from typing import Optional

from clients import Neo4jClient

logger = logging.getLogger(__name__)

_cache: dict[str, dict[str, dict]] = {}


def champ_meta(neo4j: Neo4jClient, module: str = "incident_securite_v2") -> dict[str, dict]:
    """Retourne {cle: {label, description, role, type}} pour un module (mis en cache)."""
    if module in _cache:
        return _cache[module]

    meta: dict[str, dict] = {}
    try:
        rows = neo4j.run(
            "MATCH (m:ModuleMeta {nom_technique: $module})-[:A_POUR_CHAMP]->(c:ChampMeta) "
            "RETURN c.cle AS cle, c.label AS label, c.description AS description, "
            "       c.role AS role, c.type AS type, c.valeurs_possibles AS valeurs_possibles, "
            "       c.noeud AS noeud, c.relation AS relation, c.cle_noeud AS cle_noeud",
            module=module,
        )
        for r in rows:
            meta[r["cle"]] = {
                "label": r.get("label") or r["cle"],
                "description": r.get("description") or "",
                "role": r.get("role") or "propriete",
                "type": r.get("type") or "texte",
                "valeurs_possibles": r.get("valeurs_possibles") or [],
                "noeud": r.get("noeud"),
                "relation": r.get("relation"),
                "cle_noeud": r.get("cle_noeud"),
            }
    except Exception as e:
        logger.warning("Lecture ChampMeta impossible (%s) — fallback noms techniques", e)

    if meta:
        logger.info("Catalogue %s : %d champs chargés depuis le graphe", module, len(meta))
    else:
        logger.warning("Aucun ChampMeta pour %s — lancer scripts/publish_schema_meta.py", module)

    _cache[module] = meta
    return meta


def libelle(meta: dict[str, dict], cle: str) -> str:
    """Label métier d'un champ, ou son nom technique lisible en repli."""
    m = meta.get(cle)
    if m and m.get("label"):
        return m["label"]
    return cle.replace("_", " ")


def description(meta: dict[str, dict], cle: str) -> Optional[str]:
    m = meta.get(cle)
    return m.get("description") if m else None


_PROPS_INTERNES = {"source_module", "last_indexed_at", "is_test_data", "incident_id"}
_ORDRE_IDENTITE = ["numero_fe", "titre", "date_evenement", "severite", "classification",
                   "etat", "aerodrome", "condition_lumineuse"]
_VIDES = {"", "0", "-", "n/a", "na", "ras", "nil"}


def champs_labellises(props: dict, meta: dict[str, dict], max_len: int = 300) -> list[dict]:
    """Tous les champs non vides d'une fiche, étiquetés avec les labels métier
    du schéma : [{champ, label, valeur}] — prêt à afficher en tableau."""
    lignes: list[dict] = []
    vus: set[str] = set(_PROPS_INTERNES)

    def _ajouter(cle: str) -> None:
        if cle in vus:
            return
        val = props.get(cle)
        if val is None:
            return
        if isinstance(val, bool):
            txt = "Oui" if val else "Non"
        else:
            txt = str(val).strip()
            if txt.lower() in _VIDES:
                return
            if len(txt) > max_len:
                txt = txt[:max_len] + "…"
        lignes.append({"champ": cle, "label": libelle(meta, cle), "valeur": txt})
        vus.add(cle)

    for cle in _ORDRE_IDENTITE:
        _ajouter(cle)
    for cle in sorted(props.keys()):
        _ajouter(cle)
    return lignes


_module_cache: dict[str, dict] = {}


def module_meta(neo4j: Neo4jClient, module: str = "incident_securite_v2") -> dict:
    """Métadonnées du module : label du nœud, exemples de questions (few-shot YAML)."""
    if module in _module_cache:
        return _module_cache[module]
    info = {"label_noeud": None, "exemples": [], "synonymes": []}
    try:
        rows = neo4j.run(
            "MATCH (m:ModuleMeta {nom_technique: $module}) "
            "RETURN m.label_noeud AS label_noeud, m.exemples AS exemples, "
            "       m.synonymes AS synonymes",
            module=module,
        )
        if rows:
            info["label_noeud"] = rows[0].get("label_noeud") or None
            info["exemples"] = rows[0].get("exemples") or []
            info["synonymes"] = rows[0].get("synonymes") or []
    except Exception as e:
        logger.warning("Lecture ModuleMeta impossible (%s)", e)
    _module_cache[module] = info
    return info
