"""
Moteur de requête GÉNÉRIQUE pour les incidents v2.

Objectif : répondre à une question sur N'IMPORTE QUEL champ de la fiche incident
(propriété, relation, présence/absence) sans coder un filtre par champ.

Pipeline :
  1. CATALOGUE de champs découvert automatiquement depuis Neo4j au 1er appel
     (types inférés, valeurs possibles des référentiels) — zéro maintenance.
  2. NL → QuerySpec générique (intent + filtres {champ, op, valeur} + group_by)
     via LLM structured output, le catalogue étant fourni dans le prompt.
  3. VALIDATION de chaque filtre contre le catalogue : champ existant (avec
     rattrapage par similarité), opérateur permis pour le type, valeur
     fuzzy-matchée sur les référentiels.
  4. Compilation en Cypher DÉTERMINISTE paramétré (le LLM n'écrit jamais de
     Cypher et ne calcule jamais de chiffres).
"""
from __future__ import annotations

import difflib
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, ValidationError

import prompt_store
from clients import Neo4jClient, OllamaClient
from field_catalog import champ_meta as champ_meta_graphe
from field_catalog import module_meta as module_meta_graphe

logger = logging.getLogger(__name__)

LLM_MODEL = "qwen2.5:7b"

# Propriétés techniques jamais exposées au moteur
_PROPS_EXCLUES = {"source_module", "last_indexed_at", "is_test_data", "incident_id"}

# Seuil : une propriété texte avec peu de valeurs distinctes = référentiel (enum)
_ENUM_MAX_DISTINCT = 30

# Descriptions/alias pour aider le LLM à mapper le vocabulaire utilisateur.
# Optionnel : un champ absent d'ici reste interrogeable (nom + valeurs suffisent souvent).
_DESCRIPTIONS = {
    "numero_fe":            "référence de la fiche (ex FNE/25/0377)",
    "titre":                "titre court de l'incident",
    "detail":               "narratif complet de l'événement",
    "severite":             "niveau de risque ECCAIRS",
    "classification":       "incident / incident sérieux / occurrence sans effet / accident",
    "etat":                 "état de la fiche (Clos, Actif...)",
    "etape":                "étape du workflow",
    "statut_ecc":           "statut ECCAIRS",
    "aerodrome":            "aérodrome concerné (LYS, Lyon Bron)",
    "processus":            "processus interne (PM2, PM4...)",
    "condition_lumineuse":  "il faisait jour ou nuit",
    "presence_blesses":     "y a-t-il eu des blessés",
    "analyse_causes_faite": "une analyse des causes a été faite",
    "traitement_termine":   "le traitement de la fiche est terminé",
    "est_significatif":     "événement significatif (analyse détaillée, cas graves)",
    "est_rex":              "fiche marquée retour d'expérience (REX)",
    "actions_efficaces":    "les actions ont été jugées efficaces",
    "action_corrective":    "action à chaud / action corrective immédiate (texte)",
    "analyse_chaud":        "analyse à chaud du terrain (texte)",
    "detail_verification":  "détail de la vérification (texte)",
    "date_creation":        "date de saisie de la fiche",
    "date_evenement":       "date réelle de l'événement",
    "date_maj":             "date de dernière mise à jour",
    "heure_evenement":      "heure locale de l'événement",
    "organisations_informees": "organisations informées (DSAC...)",
    # relations
    "type_evenement":       "typologie de l'événement (FOD, collision aviaire...)",
    "lieu":                 "lieu de l'événement (piste, aire de trafic...)",
    "phase_vol":            "phase de vol (stationnement, atterrissage...)",
    "compagnie":            "compagnie aérienne impliquée",
    "type_aeronef":         "type d'aéronef (A320...)",
    "notifiant":            "qui a notifié l'événement",
    "service_responsable":  "service responsable du traitement",
    "societe":              "société / prestataire concerné",
    "entite_mise_en_cause": "entité mise en cause",
    "emetteur":             "login de l'agent émetteur",
    "verificateur":         "login de l'agent vérificateur",
    # pseudo-champs
    "a_des_actions":        "l'incident a des actions structurées (corrective/préventive) trackées",
}

# Relations : champ logique -> (relation, label, propriété de nom du nœud)
_RELATIONS = {
    "type_evenement":       ("DE_TYPE",            "TypeEvenement", "label"),
    "lieu":                 ("LOCALISE_EN",        "Lieu",          "label"),
    "phase_vol":            ("EN_PHASE_DE_VOL",    "PhaseVol",      "label"),
    "compagnie":            ("IMPLIQUE_COMPAGNIE", "Compagnie",     "nom"),
    "type_aeronef":         ("IMPLIQUE_AERONEF",   "TypeAeronef",   "label"),
    "notifiant":            ("NOTIFIE_PAR",        "Notifiant",     "label"),
    "service_responsable":  ("RESPONSABLE",        "Service",       "label"),
    "societe":              ("CONCERNE",           "Societe",       "nom"),
    "entite_mise_en_cause": ("MIS_EN_CAUSE",       "Entite",        "nom"),
    "emetteur":             ("EMIS_PAR",           "Personne",      "login"),
    "verificateur":         ("VERIFIE_PAR",        "Personne",      "login"),
}

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}")


# ─── Catalogue ───────────────────────────────────────────────────────────────

@dataclass
class Champ:
    nom: str
    type: str                    # booleen | date | enum | texte | texte_long | relation | pseudo_bool
    valeurs: list[str] = field(default_factory=list)   # pour les enums
    valeurs_sens: list[str] = field(default_factory=list)  # "code — sens" (YAML valeurs_possibles)
    label: str = ""              # label métier (YAML)
    description: str = ""
    # pour les relations : (type_relation, label_noeud, propriete_nom) — dérivé du schéma
    rel_info: Optional[tuple] = None


@dataclass
class Catalogue:
    champs: dict[str, Champ] = field(default_factory=dict)
    exemples_module: list[str] = field(default_factory=list)  # few-shot déclarés dans le YAML

    def get(self, nom: str) -> Optional[Champ]:
        return self.champs.get(nom)

    def resoudre(self, nom: str) -> Optional[Champ]:
        """Résout un nom de champ, avec rattrapage par similarité."""
        if nom in self.champs:
            return self.champs[nom]
        proches = difflib.get_close_matches(nom, self.champs.keys(), n=1, cutoff=0.75)
        if proches:
            logger.info("Champ %r résolu en %r", nom, proches[0])
            return self.champs[proches[0]]
        return None


_catalogue_cache: Optional[Catalogue] = None


def construire_catalogue(neo4j: Neo4jClient) -> Catalogue:
    """Introspecte le graphe : propriétés des IncidentSecu + types + référentiels."""
    global _catalogue_cache
    if _catalogue_cache is not None:
        return _catalogue_cache

    cat = Catalogue()

    keys = neo4j.run(
        "MATCH (i:IncidentSecu) UNWIND keys(i) AS k "
        "RETURN DISTINCT k ORDER BY k"
    )
    for row in keys:
        k = row["k"]
        if k in _PROPS_EXCLUES:
            continue
        sample = neo4j.run(
            f"MATCH (i:IncidentSecu) WHERE i.`{k}` IS NOT NULL "
            f"RETURN DISTINCT i.`{k}` AS v LIMIT {_ENUM_MAX_DISTINCT + 1}"
        )
        vals = [r["v"] for r in sample]
        if not vals:
            continue

        if all(isinstance(v, bool) for v in vals):
            t, enum_vals = "booleen", []
        elif all(isinstance(v, str) and _DATE_RE.match(v) for v in vals):
            t, enum_vals = "date", []
        elif len(vals) <= _ENUM_MAX_DISTINCT:
            t, enum_vals = "enum", sorted(str(v) for v in vals)
        else:
            longueur_moy = sum(len(str(v)) for v in vals) / len(vals)
            t, enum_vals = ("texte_long" if longueur_moy > 80 else "texte"), []

        cat.champs[k] = Champ(nom=k, type=t, valeurs=enum_vals,
                              description=_DESCRIPTIONS.get(k, ""))

    # Relations : dérivées du schéma publié (:ChampMeta role=relation) —
    # fallback sur le dictionnaire statique si les métadonnées manquent.
    meta = champ_meta_graphe(neo4j)
    relations_schema = {
        cle: (m["relation"], m["noeud"], m.get("cle_noeud") or "nom")
        for cle, m in meta.items()
        if m.get("role") == "relation" and m.get("relation") and m.get("noeud")
    }
    for nom, rel_info in (relations_schema or _RELATIONS).items():
        cat.champs[nom] = Champ(nom=nom, type="relation", rel_info=tuple(rel_info),
                                description=_DESCRIPTIONS.get(nom, ""))

    cat.champs["a_des_actions"] = Champ(
        nom="a_des_actions", type="pseudo_bool",
        description=_DESCRIPTIONS["a_des_actions"],
    )

    # Enrichissement par les métadonnées du schéma YAML publiées dans le graphe
    # (:ChampMeta) — labels, descriptions et SENS des valeurs, source de vérité unique.
    for nom, champ in cat.champs.items():
        m = meta.get(nom)
        if not m:
            continue
        label = (m.get("label") or "").strip()
        desc = (m.get("description") or "").strip()
        champ.label = label
        if label and label.lower() != nom.replace("_", " "):
            champ.description = f"« {label} » — {desc}" if desc else f"« {label} »"
        elif desc:
            champ.description = desc
        if m.get("valeurs_possibles"):
            champ.valeurs_sens = list(m["valeurs_possibles"])

    cat.exemples_module = module_meta_graphe(neo4j).get("exemples") or []

    logger.info("Catalogue construit : %d champs (%d enrichis par ChampMeta, "
                "%d relations du schéma, %d exemples module)",
                len(cat.champs), sum(1 for n in cat.champs if n in meta),
                len(relations_schema), len(cat.exemples_module))
    _catalogue_cache = cat
    return cat


# ─── Spec générique ──────────────────────────────────────────────────────────

class FiltreSpec(BaseModel):
    champ: str
    op: Literal["=", "!=", "contient", ">=", "<=", "annee", "mois",
                "est_rempli", "est_vide"] = "="
    valeur: Optional[str] = None


class QuerySpec(BaseModel):
    intent: Literal["count", "liste", "repartition", "non_supporte"] = "liste"
    filtres: list[FiltreSpec] = Field(default_factory=list)
    group_by: Optional[str] = None
    limit: int = Field(default=20, ge=1, le=100)


# Schéma JSON explicite, sans $ref (compat structured output Ollama)
_SPEC_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": ["count", "liste", "repartition", "non_supporte"]},
        "filtres": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "champ": {"type": "string"},
                    "op": {"type": "string",
                           "enum": ["=", "!=", "contient", ">=", "<=",
                                    "annee", "mois", "est_rempli", "est_vide"]},
                    "valeur": {"type": ["string", "null"]},
                },
                "required": ["champ", "op"],
            },
        },
        "group_by": {"type": ["string", "null"]},
        "limit": {"type": "integer"},
    },
    "required": ["intent", "filtres"],
}


def _rendre_catalogue(cat: Catalogue) -> str:
    """Rendu compact du catalogue pour le prompt."""
    lines = []
    for c in cat.champs.values():
        desc = f" — {c.description}" if c.description else ""
        if c.type == "enum":
            # sens des valeurs (YAML) prioritaire sur les valeurs brutes
            source = c.valeurs_sens or c.valeurs
            vals = " | ".join(source[:8]) + (" | ..." if len(source) > 8 else "")
            lines.append(f"- {c.nom} (choix : {vals}){desc}")
        elif c.type in ("booleen", "pseudo_bool"):
            lines.append(f"- {c.nom} (oui/non){desc}")
        elif c.type == "date":
            lines.append(f"- {c.nom} (date){desc}")
        elif c.type == "relation":
            lines.append(f"- {c.nom} (entité liée, filtre par nom){desc}")
        else:
            lines.append(f"- {c.nom} (texte){desc}")
    return "\n".join(lines)


def parser_question(question: str, cat: Catalogue, ollama: OllamaClient) -> Optional[QuerySpec]:
    from datetime import date, timedelta
    aujourdhui = date.today()
    exemples = "\n".join(f"- {e}" for e in cat.exemples_module) or "- (aucun)"
    prompt = prompt_store.rendre(
        "question_libre.parseur",
        catalogue=_rendre_catalogue(cat),
        aujourdhui=aujourdhui.isoformat(),
        hier=(aujourdhui - timedelta(days=1)).isoformat(),
        exemples_module=exemples,
        question=question,
    )
    for attempt in range(2):
        try:
            raw = ollama.generate_structured(prompt, _SPEC_SCHEMA, model=LLM_MODEL)
            spec = QuerySpec.model_validate_json(raw)
            logger.info("QuerySpec (essai %d) : %s", attempt + 1, spec.model_dump())
            return spec
        except (ValidationError, json.JSONDecodeError, Exception) as e:
            logger.warning("Parse QuerySpec échoué (essai %d) : %s", attempt + 1, e)
    return None


# ─── Validation ──────────────────────────────────────────────────────────────

_OPS_PAR_TYPE = {
    "booleen":     {"=", "!=", "est_rempli", "est_vide"},
    "pseudo_bool": {"="},
    "date":        {"=", ">=", "<=", "annee", "mois", "est_rempli", "est_vide"},
    "enum":        {"=", "!=", "contient", "est_rempli", "est_vide"},
    "texte":       {"=", "!=", "contient", "est_rempli", "est_vide"},
    "texte_long":  {"contient", "est_rempli", "est_vide"},
    "relation":    {"=", "contient"},
}


@dataclass
class FiltreValide:
    champ: Champ
    op: str
    valeur: Any = None


def valider_filtre(f: FiltreSpec, cat: Catalogue) -> tuple[Optional[FiltreValide], Optional[str]]:
    """Valide/corrige un filtre. Retourne (filtre, None) ou (None, message d'erreur)."""
    champ = cat.resoudre(f.champ)
    if champ is None:
        proches = difflib.get_close_matches(f.champ, cat.champs.keys(), n=3, cutoff=0.4)
        return None, (f"champ inconnu : '{f.champ}'"
                      + (f" (champs proches : {', '.join(proches)})" if proches else ""))

    op = f.op
    valeur: Any = f.valeur

    # Auto-corrections d'opérateur
    if op not in _OPS_PAR_TYPE[champ.type]:
        if champ.type == "texte_long" and op in ("=", "!="):
            op = "contient"
        elif champ.type in ("booleen", "pseudo_bool"):
            op = "="
        elif champ.type == "relation":
            op = "contient"
        else:
            return None, f"opérateur '{f.op}' non permis pour le champ '{champ.nom}' ({champ.type})"

    if op in ("est_rempli", "est_vide"):
        # sur un booléen, « avec X » = X vrai (le flag est renseigné partout,
        # est_rempli n'y a jamais de sens)
        if champ.type == "booleen":
            return FiltreValide(champ, "=", op == "est_rempli"), None
        return FiltreValide(champ, op), None

    if valeur is None:
        return None, f"valeur manquante pour le filtre sur '{champ.nom}'"

    if champ.type in ("booleen", "pseudo_bool"):
        v = str(valeur).strip().lower()
        if v in ("true", "vrai", "oui", "1"):
            valeur = True
        elif v in ("false", "faux", "non", "0"):
            valeur = False
        else:
            return None, f"valeur booléenne attendue pour '{champ.nom}' (reçu : {valeur!r})"
        return FiltreValide(champ, op, valeur), None

    valeur = str(valeur).strip()

    # Fuzzy-match sur les référentiels : "sérieux" -> "Incident sérieux"
    if champ.type == "enum" and op in ("=", "!="):
        exact = [v for v in champ.valeurs if v.lower() == valeur.lower()]
        if exact:
            valeur = exact[0]
        else:
            partiel = [v for v in champ.valeurs if valeur.lower() in v.lower()]
            if len(partiel) == 1:
                valeur = partiel[0]
            elif not partiel:
                op = "contient"   # dernier recours

    return FiltreValide(champ, op, valeur), None


# ─── Compilation Cypher déterministe ─────────────────────────────────────────

def compiler(spec: QuerySpec, filtres: list[FiltreValide]) -> tuple[str, str, dict[str, Any], Optional[str]]:
    """Retourne (cypher_liste, cypher_count, params, erreur_group_by)."""
    where: list[str] = ["coalesce(i.is_test_data, false) = false"]
    params: dict[str, Any] = {}

    for idx, fv in enumerate(filtres):
        p = f"p{idx}"
        c = fv.champ

        if c.type == "relation":
            rel, lbl, prop = c.rel_info or _RELATIONS[c.nom]
            cmp_ = "=" if fv.op == "=" else "CONTAINS"
            where.append(
                f"EXISTS {{ MATCH (i)-[:{rel}]->(e:{lbl}) "
                f"WHERE toLower(e.{prop}) {cmp_} toLower(${p}) }}"
            )
            params[p] = fv.valeur

        elif c.type == "pseudo_bool":  # a_des_actions
            pattern = "EXISTS { MATCH (i)-[:A_POUR_ACTION]->(:Action) }"
            where.append(pattern if fv.valeur else f"NOT {pattern}")

        elif fv.op == "est_rempli":
            where.append(f"i.`{c.nom}` IS NOT NULL")
        elif fv.op == "est_vide":
            where.append(f"i.`{c.nom}` IS NULL")

        elif c.type == "booleen":
            where.append(f"i.`{c.nom}` {'=' if fv.op == '=' else '<>'} ${p}")
            params[p] = fv.valeur

        elif c.type == "date":
            if fv.op in ("annee", "mois", "="):
                where.append(f"i.`{c.nom}` STARTS WITH ${p}")
            else:  # >= / <=  (ISO : comparaison lexicographique valide)
                where.append(f"i.`{c.nom}` {fv.op} ${p}")
            params[p] = str(fv.valeur)

        else:  # enum / texte / texte_long
            if fv.op == "contient":
                where.append(f"toLower(i.`{c.nom}`) CONTAINS toLower(${p})")
            elif fv.op == "!=":
                where.append(f"toLower(i.`{c.nom}`) <> toLower(${p})")
            else:
                where.append(f"toLower(i.`{c.nom}`) = toLower(${p})")
            params[p] = str(fv.valeur)

    base = f"MATCH (i:IncidentSecu) WHERE {' AND '.join(where)}"
    cypher_count = f"{base} RETURN count(i) AS n"

    # Champs retournés en liste : cœur de fiche + champs filtrés (hors relations)
    retour = ["numero_fe", "titre", "severite", "classification", "etat", "date_evenement"]
    for fv in filtres:
        if fv.champ.type not in ("relation", "pseudo_bool") and fv.champ.nom not in retour:
            retour.append(fv.champ.nom)
    projection = ", ".join(
        f"left(toString(i.`{c}`), 200) AS `{c}`" if c in ("detail", "action_corrective",
                                                          "analyse_chaud", "detail_verification")
        else f"i.`{c}` AS `{c}`"
        for c in retour
    )

    erreur_gb = None
    if spec.intent == "repartition":
        gb = spec.group_by or ""
        if gb in ("annee", "mois"):
            n_char = 4 if gb == "annee" else 7
            cypher_liste = (f"{base} AND i.date_evenement IS NOT NULL "
                            f"RETURN substring(i.date_evenement, 0, {n_char}) AS label, "
                            f"count(*) AS n ORDER BY n DESC LIMIT 30")
        else:
            champ_gb = None if not gb else (
                _catalogue_cache.resoudre(gb) if _catalogue_cache else None)
            if champ_gb is None:
                erreur_gb = f"champ de répartition inconnu : '{gb}'"
                cypher_liste = cypher_count
            elif champ_gb.type == "relation":
                rel, lbl, prop = champ_gb.rel_info or _RELATIONS[champ_gb.nom]
                cypher_liste = (f"{base} MATCH (i)-[:{rel}]->(e:{lbl}) "
                                f"RETURN e.{prop} AS label, count(DISTINCT i) AS n "
                                f"ORDER BY n DESC LIMIT 30")
            else:
                cypher_liste = (f"{base} AND i.`{champ_gb.nom}` IS NOT NULL "
                                f"RETURN toString(i.`{champ_gb.nom}`) AS label, "
                                f"count(*) AS n ORDER BY n DESC LIMIT 30")
    else:
        params["limit"] = spec.limit
        cypher_liste = (f"{base} RETURN {projection} "
                        f"ORDER BY i.date_evenement DESC LIMIT $limit")

    return cypher_liste, cypher_count, params, erreur_gb


# ─── Point d'entrée ──────────────────────────────────────────────────────────

_STOP_TOKENS = {"les", "des", "une", "sans", "avec", "sur", "pour", "dans", "par"}


def _tokens_lex(s: str) -> list[str]:
    return [t for t in re.findall(r"[a-zà-ÿ0-9]+", s.lower())
            if len(t) >= 3 and t not in _STOP_TOKENS]


def _support_lexical(valeur: str, question: str) -> bool:
    """Vrai si un mot de la valeur apparaît (à un préfixe près) dans la question."""
    q_toks = _tokens_lex(question)
    for vt in _tokens_lex(str(valeur)):
        for qt in q_toks:
            if qt.startswith(vt[:4]) or vt.startswith(qt[:4]):
                return True
    return False


def _filtrer_parasites(filtres: list, question: str, cat: Catalogue,
                       synonymes: list[str], erreurs: list[str]) -> list:
    """
    Garde-fou déterministe anti-hallucination : un filtre d'ÉGALITÉ sur un
    référentiel est rejeté si sa valeur n'a aucun appui dans la question.
    - valeur réduite aux synonymes génériques de la fiche (ex. « Incident ») :
      rejetée sauf si la question mentionne le champ (« classé... »).
    - valeur sans aucun mot commun avec la question NI présence dans les
      exemples du module : rejetée.
    """
    synonymes_set = {s.lower() for s in synonymes}
    q_low = question.lower()
    _MARQUEURS_RELATIFS = ("hier", "aujourd", "demain", "cette année", "cette annee",
                           "ce mois", "semaine", "récent", "recent", "dernier")
    _GENERIQUES = {"combien", "répartition", "repartition", "liste", "quels",
                   "quelles", "nombre", "donne", "moi"}

    def _prefixe_commun(toks_a: set, toks_b: set) -> bool:
        return any(a.startswith(b[:4]) or b.startswith(a[:4])
                   for a in toks_a for b in toks_b)

    q_toks_distinctifs = set(_tokens_lex(question)) - synonymes_set - _GENERIQUES

    def _whitelist_exemple(val_low: str) -> bool:
        """Valeur couverte par un exemple du YAML dont la question reprend un mot."""
        for ligne in cat.exemples_module:
            if val_low in ligne.lower():
                l_toks = set(_tokens_lex(ligne)) - synonymes_set - _GENERIQUES
                if _prefixe_commun(q_toks_distinctifs, l_toks):
                    return True
        return False

    gardes = []
    for fv in filtres:
        # nom + label métier UNIQUEMENT (la description contient les valeurs
        # d'enum, elle rendrait le test toujours vrai)
        champ_evoque = _support_lexical(
            fv.champ.nom.replace("_", " ") + " " + (fv.champ.label or ""), question)

        if fv.champ.type == "enum" and fv.op in ("=", "!=", "contient") and fv.valeur is not None:
            val = str(fv.valeur)

            # « contient » sur un référentiel → résolution vers UNE valeur
            # exacte, sinon rejet (bloque les valeurs fantaisistes du LLM)
            if fv.op == "contient":
                v_low = val.lower()
                cands = [v for v in fv.champ.valeurs
                         if v_low in v.lower() or v.lower() in v_low]
                if len(cands) == 1:
                    fv = FiltreValide(fv.champ, "=", cands[0])
                    val = cands[0]
                else:
                    erreurs.append(f"filtre ignoré : {fv.champ.nom} contient {val} "
                                   f"(valeur non résolue dans le référentiel)")
                    continue

            v_toks = _tokens_lex(val)
            distinctifs = [t for t in v_toks if t not in synonymes_set]
            if v_toks and not distinctifs and not champ_evoque:
                erreurs.append(f"filtre ignoré : {fv.champ.nom} = {val} "
                               f"(terme générique de la question, pas un critère)")
                continue
            # le support lexical doit venir des mots DISTINCTIFS de la valeur
            support = _support_lexical(" ".join(distinctifs or v_toks), question)
            if not support and not _whitelist_exemple(val.lower()) and not champ_evoque:
                erreurs.append(f"filtre ignoré : {fv.champ.nom} = {val} "
                               f"(valeur absente de la question)")
                continue

        # filtre de date halluciné : l'année doit apparaître dans la question,
        # sauf formulation relative ("hier", "cette année"...)
        if fv.champ.type == "date" and fv.valeur is not None:
            m_annee = re.search(r"(19|20)\d{2}", str(fv.valeur))
            if (m_annee and m_annee.group(0) not in question
                    and not any(mq in q_low for mq in _MARQUEURS_RELATIFS)):
                erreurs.append(f"filtre ignoré : {fv.champ.nom} {fv.op} {fv.valeur} "
                               f"(date absente de la question)")
                continue

        gardes.append(fv)

    # filtres d'égalité contradictoires sur un même champ → aucun n'est fiable
    par_champ: dict[str, list] = {}
    for fv in gardes:
        if fv.op == "=":
            par_champ.setdefault(fv.champ.nom, []).append(str(fv.valeur))
    contradictoires = {n for n, vals in par_champ.items() if len(set(vals)) > 1}
    if contradictoires:
        erreurs.append(f"filtres contradictoires ignorés sur : {', '.join(contradictoires)}")
        gardes = [fv for fv in gardes
                  if not (fv.op == "=" and fv.champ.nom in contradictoires)]
    return gardes


def _canonical_group_by(gb: Optional[str], cat: Catalogue) -> Optional[str]:
    """Canonise le group_by du LLM : 'mois(date_evenement)' → 'mois',
    champ date brut → 'annee' (jamais de répartition par jour)."""
    if not gb:
        return gb
    g = gb.strip().lower()
    m = re.match(r"^([\w'éèàû]+)\s*\((.*)\)$", g)   # forme fonction(x)
    if m:
        g = m.group(1)
    if "mois" in g or g == "month":
        return "mois"
    if "annee" in g or "année" in g or g == "year":
        return "annee"
    champ = cat.resoudre(g)
    if champ is not None:
        # une répartition par date brute = un comptage par jour, jamais voulu
        return "annee" if champ.type == "date" else champ.nom
    return gb


# Demandes de modification des données : refus explicite (le moteur est
# read-only par construction, mais mieux vaut le dire que répondre à côté).
_ECRITURE_RE = re.compile(
    r"\b(supprim\w*|efface\w*|détrui\w*|delete|drop|modifi\w*|mets? à jour|"
    r"update|insère\w*|ajoute\w* (?:un|une|des) (?:incident|fiche|action))\b",
    re.IGNORECASE,
)


def run_query(question: str, ollama: OllamaClient, neo4j: Neo4jClient) -> dict:
    """
    Retourne un dict : status ("ok" | "not_parsed" | "invalid" | "unsupported"
    | "write_refused"), spec, filtres appliqués (lisibles), erreurs, rows, total.
    """
    if _ECRITURE_RE.search(question):
        return {"status": "write_refused", "spec": None, "filtres": [],
                "erreurs": ["demande de modification des données"], "rows": [], "total": None}

    cat = construire_catalogue(neo4j)
    spec = parser_question(question, cat, ollama)

    if spec is None:
        return {"status": "not_parsed", "spec": None, "filtres": [],
                "erreurs": ["question non comprise"], "rows": [], "total": None}

    if spec.intent == "non_supporte":
        return {"status": "unsupported", "spec": spec, "filtres": [],
                "erreurs": ["calcul ou donnée non disponible"], "rows": [], "total": None}

    spec.group_by = _canonical_group_by(spec.group_by, cat)

    # Un group_by présent = une répartition, quel que soit l'intent déclaré
    if spec.group_by and spec.intent in ("count", "liste"):
        spec.intent = "repartition"

    filtres_valides: list[FiltreValide] = []
    erreurs: list[str] = []
    for f in spec.filtres:
        fv, err = valider_filtre(f, cat)
        if fv:
            filtres_valides.append(fv)
        else:
            erreurs.append(err)

    # Garde-fou anti-filtre halluciné (déterministe, piloté par le YAML)
    synonymes = module_meta_graphe(neo4j).get("synonymes") or []
    filtres_valides = _filtrer_parasites(filtres_valides, question, cat,
                                         synonymes, erreurs)

    # Filtre de relation : la valeur doit correspondre à AU MOINS une entité
    # du graphe (bloque « type_evenement contient Incident » et consorts)
    gardes = []
    for fv in filtres_valides:
        if fv.champ.type == "relation" and fv.valeur:
            rel, lbl, prop = fv.champ.rel_info or _RELATIONS[fv.champ.nom]
            n = neo4j.run(
                f"MATCH (e:{lbl}) WHERE toLower(e.{prop}) CONTAINS toLower($v) "
                f"RETURN count(e) AS n", v=str(fv.valeur),
            )[0]["n"]
            if n == 0:
                erreurs.append(f"filtre ignoré : {fv.champ.nom} contient "
                               f"« {fv.valeur} » (aucune entité correspondante)")
                continue
        gardes.append(fv)
    filtres_valides = gardes

    # Répartition : un filtre d'égalité sur le champ de répartition contredit
    # l'intention même d'une distribution → retiré
    if spec.intent == "repartition" and spec.group_by:
        avant = len(filtres_valides)
        filtres_valides = [fv for fv in filtres_valides
                           if not (fv.op in ("=", "!=") and fv.champ.nom == spec.group_by)]
        if len(filtres_valides) < avant:
            erreurs.append(f"filtre sur le champ de répartition ({spec.group_by}) ignoré")

    if erreurs and not filtres_valides and spec.intent != "repartition":
        return {"status": "invalid", "spec": spec, "filtres": [],
                "erreurs": erreurs, "rows": [], "total": None}

    cypher_liste, cypher_count, params, err_gb = compiler(spec, filtres_valides)
    if err_gb:
        erreurs.append(err_gb)
        return {"status": "invalid", "spec": spec, "filtres": [],
                "erreurs": erreurs, "rows": [], "total": None}

    logger.info("Query Cypher: %s | params: %s", cypher_liste, params)

    count_params = {k: v for k, v in params.items() if k != "limit"}
    total = neo4j.run(cypher_count, **count_params)[0]["n"]

    rows: list[dict] = []
    if spec.intent in ("liste", "repartition"):
        rows = neo4j.run(cypher_liste, **params)

    filtres_lisibles = [
        {"champ": fv.champ.nom, "op": fv.op,
         "valeur": None if fv.valeur is None else str(fv.valeur)}
        for fv in filtres_valides
    ]
    return {"status": "ok", "spec": spec, "filtres": filtres_lisibles,
            "erreurs": erreurs, "rows": rows, "total": total}


# =============================================================================
# MOTEUR UNIFIÉ (spec fermée — remplace /list REGEX + absorbe count/repartition)
# =============================================================================

class UnifiedQuerySpec(BaseModel):
    """Spec fermée pour le moteur unifié count/repartition/liste."""
    output: Literal["count", "repartition", "liste"] = "liste"
    group_by: Optional[Literal[
        "severite", "classification", "condition_lumineuse", "annee", "mois", "etat"
    ]] = None
    f_severite: Optional[Literal[
        "1 - faible", "2 - tolérable", "3 - important", "4 - élevé", "5 - intolérable"
    ]] = None
    f_classification: Optional[Literal[
        "Incident", "Occurence sans effet sur la sécurité", "Incident sérieux", "Accident"
    ]] = None
    f_condition_lumineuse: Optional[Literal["Jour", "Nuit"]] = None
    f_traitement_termine: Optional[bool] = None
    f_presence_blesses: Optional[bool] = None
    f_annee: Optional[int] = None
    f_mois: Optional[str] = None
    sort_by: Literal["date_creation", "date_evenement", "severite"] = "date_creation"
    order: Literal["desc", "asc"] = "desc"
    limit: int = Field(default=5, ge=1, le=200)
    is_hors_domaine: bool = False
    # ─── Lot B : jointure nœuds :Action (additifs) ───────────────────────────
    include_actions: bool = False
    action_type: Optional[Literal["corrective", "préventive", "curative"]] = None
    action_statut: Optional[Literal["en_cours", "cloturee"]] = None
    shape: Literal["par_incident", "par_action"] = "par_incident"


_UNIFIED_SPEC_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "output":               {"type": "string", "enum": ["count", "repartition", "liste"]},
        "group_by":             {"type": ["string", "null"]},
        "f_severite":          {"type": ["string", "null"]},
        "f_classification":    {"type": ["string", "null"]},
        "f_condition_lumineuse": {"type": ["string", "null"]},
        "f_traitement_termine": {"type": ["boolean", "null"]},
        "f_presence_blesses":  {"type": ["boolean", "null"]},
        "f_annee":             {"type": ["integer", "null"]},
        "f_mois":              {"type": ["string", "null"]},
        "sort_by":             {"type": "string", "enum": ["date_creation", "date_evenement", "severite"]},
        "order":               {"type": "string", "enum": ["desc", "asc"]},
        "limit":               {"type": "integer", "minimum": 1, "maximum": 200},
        "is_hors_domaine":     {"type": "boolean"},
        "include_actions":     {"type": "boolean"},
        "action_type":         {"type": ["string", "null"]},
        "action_statut":       {"type": ["string", "null"]},
        "shape":               {"type": "string", "enum": ["par_incident", "par_action"]},
    },
    "required": ["output"],
}

_UNIFIED_FIELDS_ALLOWED = {
    "numero_fe", "titre", "severite", "classification", "etat",
    "date_evenement", "date_creation", "action_corrective", "resume_llm",
}
_UNIFIED_FIELDS_DEFAULT = ["numero_fe", "titre", "severite", "date_creation"]
_UNIFIED_FIELDS_LONG = {"action_corrective", "resume_llm"}  # tronqués à 200 chars


def _compute_unified_fields(question: str, spec: UnifiedQuerySpec) -> list[str]:
    """Calcule la projection de champs de manière déterministe (sans LLM)."""
    qa = question.lower()
    fields: list[str] = list(_UNIFIED_FIELDS_DEFAULT)
    if re.search(r"date.?evenement|date.?événement|par événement|par evenement", qa):
        fields.append("date_evenement")
    if re.search(r"classification|catégorie", qa):
        fields.append("classification")
    if re.search(r"\bétat\b|statut", qa):
        fields.append("etat")
    if re.search(r"résumé|resume", qa):
        fields.append("resume_llm")
    if spec.sort_by == "date_evenement" and "date_evenement" not in fields:
        fields.append("date_evenement")
    # N'inclure action_corrective (texte immédiat) que si pas de jointure :Action
    if not spec.include_actions and "action_corrective" not in fields:
        fields.append("action_corrective")
    return fields


# ─── Patterns action (Lot B) ─────────────────────────────────────────────────
_ACTION_STRUCT_RE = re.compile(
    r"\bactions?\s+(corrective[s]?|pr[ée]ventive[s]?|curative[s]?|de\s+traitement"
    r"|structur[ée]e[s]?|planifi[ée]e[s]?|suivie[s]?|de\s+suivi)\b"
    r"|\bleurs\s+actions?\b"
    r"|\bactions?\s+en\s+cours\b"
    r"|\bactions?\s+cl[oô]tur[ée]e[s]?\b"
    r"|\bactions?\s+termin[ée]e[s]?\b",
    re.IGNORECASE,
)
_ACTION_TYPE_PATS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bcorrective[s]?\b", re.IGNORECASE), "corrective"),
    (re.compile(r"\bpr[ée]ventive[s]?\b", re.IGNORECASE), "préventive"),
    (re.compile(r"\bcurative[s]?\b", re.IGNORECASE), "curative"),
]
_ACTION_EN_COURS_RE = re.compile(r"\ben\s+cours\b|non\s+cl[oô]tur[ée]e[s]?\b", re.IGNORECASE)
_ACTION_CLOTUREE_RE = re.compile(
    r"\bcl[oô]tur[ée]e[s]?\b|termin[ée]e[s]?\b|ferm[ée]e[s]?\b|r[ée]solue[s]?\b",
    re.IGNORECASE,
)
_SHAPE_PAR_INCIDENT_RE = re.compile(
    r"\bleurs\s+actions?\b"
    r"|actions?\s+des?\s+.{0,30}(incidents?|fiches?)\b"
    r"|\bpour\s+les?\s+.{0,30}(incidents?|fiches?)\b",
    re.IGNORECASE,
)

_SEVERITY_PATTERNS: list[tuple[str, str]] = [
    (r"\bintolérable[s]?\b|intolerable[s]?\b|5\s*-\s*intol", "5 - intolérable"),
    (r"\bélevée?[s]?\b|elevee?[s]?\b|grave[s]?\b|sévère[s]?\b|severe[s]?\b|high\b", "4 - élevé"),
    (r"\bimportant[es]?\b|significatif\b", "3 - important"),
    (r"\btolerable[s]?\b|tolérable[s]?\b", "2 - tolérable"),
    (r"\bfaible[s]?\b|mineur[es]?\b|minor\b", "1 - faible"),
]

_ORDER_ASC_RE = re.compile(
    r"\banci(en|enne|ens|ennes)\b|plus\s+vieu[x]?\b|oldest\b|au\s+début\b|premier[s]?\b|les\s+premiers\b",
    re.IGNORECASE,
)
_SERIEUX_RE = re.compile(r"\bsérieus[e]?\b|serieux\b|serious\b", re.IGNORECASE)
_NUIT_RE = re.compile(r"\bnuit\b|nocturne\b", re.IGNORECASE)
_JOUR_RE = re.compile(r"\bjour\b|diurne\b", re.IGNORECASE)
_AUJOURD_RE = re.compile(r"\baujourd", re.IGNORECASE)
_ACCIDENT_RE = re.compile(r"\baccident[s]?\b", re.IGNORECASE)
_BLESSES_RE = re.compile(r"\bblesse[és][s]?\b|blessure[s]?\b|victime[s]?\b", re.IGNORECASE)
_SANS_BLESSES_RE = re.compile(r"sans\s+bless", re.IGNORECASE)
_RECENT_RE = re.compile(r"(plus\s+)?récent[se]?\b|recent[se]?\b", re.IGNORECASE)
_ANNEE_RE = re.compile(r"\b(20\d{2})\b")


def _postprocess_unified(spec: UnifiedQuerySpec, question: str) -> UnifiedQuerySpec:
    """Corrections déterministes post-LLM pour les mappings exacts (non-dépendants du contexte)."""
    q = question

    # Sévérité : synonymes courants non mappés par le LLM
    if spec.f_severite is None:
        for pat, val in _SEVERITY_PATTERNS:
            if re.search(pat, q, re.IGNORECASE):
                # "sérieux" ne doit pas déclencher la sévérité (c'est une classification)
                if val == "3 - important" and _SERIEUX_RE.search(q):
                    continue
                spec.f_severite = val  # type: ignore[assignment]
                break

    # Classification : "sérieux" → "Incident sérieux"
    if spec.f_classification is None and _SERIEUX_RE.search(q):
        spec.f_classification = "Incident sérieux"  # type: ignore[assignment]

    # Classification : "accident" → "Accident"
    if spec.f_classification is None and _ACCIDENT_RE.search(q):
        spec.f_classification = "Accident"  # type: ignore[assignment]

    # Condition lumineuse
    if spec.f_condition_lumineuse is None:
        if _NUIT_RE.search(q):
            spec.f_condition_lumineuse = "Nuit"  # type: ignore[assignment]
        elif _JOUR_RE.search(q) and not _AUJOURD_RE.search(q):
            spec.f_condition_lumineuse = "Jour"  # type: ignore[assignment]

    # Présence blessés
    if spec.f_presence_blesses is None:
        if _SANS_BLESSES_RE.search(q):
            spec.f_presence_blesses = False
        elif _BLESSES_RE.search(q):
            spec.f_presence_blesses = True

    # Ordre : anciens/premiers → asc
    if _ORDER_ASC_RE.search(q):
        spec.order = "asc"

    # sort_by : si filtre sévérité présent ET "récents" → date_evenement
    if spec.f_severite is not None and spec.sort_by == "date_creation" and _RECENT_RE.search(q):
        spec.sort_by = "date_evenement"

    # Année : si le LLM l'a manquée
    if spec.f_annee is None:
        m = _ANNEE_RE.search(q)
        if m:
            spec.f_annee = int(m.group(1))

    # ─── Actions structurées (nœuds :Action) ─────────────────────────────────
    if not spec.include_actions and _ACTION_STRUCT_RE.search(q):
        spec.include_actions = True

    if spec.include_actions:
        # Quand include_actions : "clôturé/terminé" = statut de l'action, pas de l'incident
        spec.f_traitement_termine = None

        if spec.action_type is None:
            for pat, val in _ACTION_TYPE_PATS:
                if pat.search(q):
                    spec.action_type = val  # type: ignore[assignment]
                    break

        if spec.action_statut is None:
            if _ACTION_EN_COURS_RE.search(q):
                spec.action_statut = "en_cours"  # type: ignore[assignment]
            elif _ACTION_CLOTUREE_RE.search(q):
                spec.action_statut = "cloturee"  # type: ignore[assignment]

        # shape : par_incident si la question part clairement des incidents
        if _SHAPE_PAR_INCIDENT_RE.search(q):
            spec.shape = "par_incident"  # type: ignore[assignment]
        else:
            spec.shape = "par_action"  # type: ignore[assignment]

    return spec


def parse_unified_question(question: str, ollama: OllamaClient) -> Optional[UnifiedQuerySpec]:
    """NL → UnifiedQuerySpec via structured output LLM (temp=0, retry 1) + post-processing."""
    from datetime import date as _date
    annee = _date.today().year
    prompt = prompt_store.rendre(
        "unified_query.parseur",
        question=question,
        annee_courante=str(annee),
    )
    for attempt in range(2):
        try:
            raw = ollama.generate_structured(prompt, _UNIFIED_SPEC_SCHEMA, model=LLM_MODEL, timeout=120.0)
            spec = UnifiedQuerySpec.model_validate_json(raw)
            spec = _postprocess_unified(spec, question)
            logger.info("UnifiedQuerySpec (essai %d, post-process) : %s", attempt + 1, spec.model_dump())
            return spec
        except Exception as e:
            logger.warning("Parse UnifiedQuerySpec échoué (essai %d) : %s", attempt + 1, e)
    return None


_CLOTUREE = "(a.statut = '100' OR a.date_cloture IS NOT NULL)"


def _build_action_par_incident(
    spec: UnifiedQuerySpec, base: str, params: dict
) -> tuple[str, dict]:
    """OPTIONAL MATCH : retourne N incidents avec leurs actions imbriquées."""
    aw: list[str] = []
    if spec.action_type is not None:
        aw.append("r.type_action = $atype")
        params["atype"] = spec.action_type
    if spec.action_statut == "cloturee":
        aw.append(_CLOTUREE)
    elif spec.action_statut == "en_cours":
        aw.append("a.statut = '0'")
    opt_where = (" WHERE " + " AND ".join(aw)) if aw else ""
    params["limit"] = min(spec.limit, 200)
    cypher = (
        f"{base} "
        f"WITH i ORDER BY i.{spec.sort_by} {spec.order.upper()} LIMIT $limit "
        f"OPTIONAL MATCH (i)-[r:A_POUR_ACTION]->(a:Action){opt_where} "
        f"WITH i, [x IN collect({{type: r.type_action, titre: a.titre_action, "
        f"statut: a.statut, responsable: a.responsable, avancement: a.etat_avancement, "
        f"date_prevue: a.date_prevue, date_cloture: a.date_cloture}}) "
        f"WHERE x.titre IS NOT NULL] AS actions "
        f"RETURN i.numero_fe AS numero_fe, i.titre AS titre, "
        f"i.severite AS severite, i.date_creation AS date_creation, actions"
    )
    return cypher, params


def _build_action_par_action(
    spec: UnifiedQuerySpec, incident_where: list[str], params: dict
) -> tuple[str, dict]:
    """MATCH direct : une ligne par action (jointure plate incident-action)."""
    all_where: list[str] = list(incident_where)
    if spec.action_type is not None:
        all_where.append("r.type_action = $atype")
        params["atype"] = spec.action_type
    if spec.action_statut == "cloturee":
        all_where.append(_CLOTUREE)
    elif spec.action_statut == "en_cours":
        all_where.append("a.statut = '0'")
    where_str = " AND ".join(all_where)
    params["limit"] = min(spec.limit, 200)
    cypher = (
        f"MATCH (i:IncidentSecu)-[r:A_POUR_ACTION]->(a:Action) WHERE {where_str} "
        f"RETURN i.numero_fe AS numero_fe, i.titre AS titre, "
        f"i.severite AS severite, i.date_creation AS date_creation, "
        f"r.type_action AS type_action, a.titre_action AS titre_action, "
        f"a.statut AS statut, a.responsable AS responsable, "
        f"a.etat_avancement AS avancement, a.date_prevue AS date_prevue, "
        f"a.date_cloture AS date_cloture "
        f"ORDER BY i.{spec.sort_by} {spec.order.upper()} LIMIT $limit"
    )
    return cypher, params


def build_unified_cypher(spec: UnifiedQuerySpec, fields: list[str]) -> tuple[str, dict[str, Any]]:
    """Compile une UnifiedQuerySpec en Cypher déterministe paramétré (READ-ONLY)."""
    where: list[str] = ["coalesce(i.is_test_data, false) = false"]
    params: dict[str, Any] = {}

    if spec.f_severite is not None:
        where.append("i.severite = $sev")
        params["sev"] = spec.f_severite

    if spec.f_classification is not None:
        where.append("i.classification = $cls")
        params["cls"] = spec.f_classification

    if spec.f_condition_lumineuse is not None:
        where.append("i.condition_lumineuse = $cl")
        params["cl"] = spec.f_condition_lumineuse

    if spec.f_traitement_termine is not None:
        where.append("i.traitement_termine = $tt")
        params["tt"] = spec.f_traitement_termine

    if spec.f_presence_blesses is not None:
        where.append("i.presence_blesses = $pb")
        params["pb"] = spec.f_presence_blesses

    if spec.f_annee is not None:
        where.append("i.date_evenement STARTS WITH $annee")
        params["annee"] = str(spec.f_annee)

    if spec.f_mois is not None:
        where.append("i.date_evenement STARTS WITH $mois")
        params["mois"] = spec.f_mois

    where_str = " AND ".join(where)
    base = f"MATCH (i:IncidentSecu) WHERE {where_str}"

    if spec.output == "count":
        if spec.include_actions:
            # Compter les actions (pas les incidents)
            aw: list[str] = list(where)
            if spec.action_type is not None:
                aw.append("r.type_action = $atype")
                params["atype"] = spec.action_type
            if spec.action_statut == "cloturee":
                aw.append(_CLOTUREE)
            elif spec.action_statut == "en_cours":
                aw.append("a.statut = '0'")
            cypher = (
                f"MATCH (i:IncidentSecu)-[r:A_POUR_ACTION]->(a:Action) "
                f"WHERE {' AND '.join(aw)} RETURN count(*) AS n"
            )
        else:
            cypher = f"{base} RETURN count(i) AS n"

    elif spec.output == "repartition":
        gb = spec.group_by or "severite"
        params["rlimit"] = 30
        if gb in ("annee", "mois"):
            n_char = 4 if gb == "annee" else 7
            cypher = (
                f"{base} AND i.date_evenement IS NOT NULL "
                f"RETURN substring(i.date_evenement, 0, {n_char}) AS label, "
                f"count(*) AS n ORDER BY n DESC LIMIT $rlimit"
            )
        else:
            cypher = (
                f"{base} AND i.`{gb}` IS NOT NULL "
                f"RETURN i.`{gb}` AS label, count(*) AS n "
                f"ORDER BY n DESC LIMIT $rlimit"
            )

    else:  # liste
        if spec.include_actions:
            if spec.shape == "par_incident":
                cypher, params = _build_action_par_incident(spec, base, params)
            else:
                cypher, params = _build_action_par_action(spec, where, params)
        else:
            safe = [f for f in fields if f in _UNIFIED_FIELDS_ALLOWED] or list(_UNIFIED_FIELDS_DEFAULT)
            proj = ", ".join(
                f"left(toString(i.{f}), 200) AS {f}" if f in _UNIFIED_FIELDS_LONG
                else f"i.{f} AS {f}"
                for f in safe
            )
            params["limit"] = min(spec.limit, 200)
            cypher = (
                f"{base} RETURN {proj} "
                f"ORDER BY i.{spec.sort_by} {spec.order.upper()} "
                f"LIMIT $limit"
            )

    return cypher, params


def run_unified_query(
    question: str,
    ollama: OllamaClient,
    neo4j: Neo4jClient,
) -> dict:
    """
    Moteur unifié count/repartition/liste sur spec fermée.

    Retourne un dict avec :
      status          : "ok" | "parse_failed" | "besoin_precision"
      spec            : UnifiedQuerySpec (objet) ou None
      spec_interpretee: dict (pour sérialisation JSON) ou None
      cypher_execute  : str ou None
      resultat_brut   : int (count) | list[dict] (liste/repartition) | None
    """
    spec = parse_unified_question(question, ollama)

    if spec is None:
        return {
            "status": "parse_failed",
            "spec": None,
            "spec_interpretee": None,
            "cypher_execute": None,
            "resultat_brut": None,
        }

    if spec.is_hors_domaine:
        return {
            "status": "besoin_precision",
            "spec": spec,
            "spec_interpretee": spec.model_dump(),
            "cypher_execute": None,
            "resultat_brut": None,
        }

    if spec.output == "repartition" and not spec.group_by:
        return {
            "status": "besoin_precision",
            "spec": spec,
            "spec_interpretee": spec.model_dump(),
            "cypher_execute": None,
            "resultat_brut": None,
        }

    fields = _compute_unified_fields(question, spec)
    cypher, params = build_unified_cypher(spec, fields)
    logger.info("UnifiedQuery Cypher: %s | params: %s", cypher, params)

    rows = neo4j.run_read(cypher, **params)

    if spec.output == "count":
        resultat_brut: Any = rows[0]["n"] if rows else 0
    else:
        resultat_brut = [dict(r) for r in rows]

    return {
        "status": "ok",
        "spec": spec,
        "spec_interpretee": spec.model_dump(),
        "cypher_execute": cypher,
        "resultat_brut": resultat_brut,
    }
