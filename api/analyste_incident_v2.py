"""
Couche AGENT-ANALYSTE — version GÉNÉRALE (opérateurs × dimensions × cibles).

Principe fondamental (agentic RAG / TAG, gouverné comme une semantic layer) :
  - un ENSEMBLE FERMÉ d'opérateurs analytiques (le "quoi calculer"),
  - appliqués à N'IMPORTE QUELLE dimension et cible du SCHÉMA (le "sur quoi"),
  - le LLM ne fait que MAPPER la question → (opérateur, dimension, cible, filtres)
    dans des menus ; le CODE exécute des requêtes déterministes et CALCULE ;
    le LLM ne fait que SYNTHÉTISER à partir des faits chiffrés.

Opérateurs :
  - "proportion" : part d'une cible par valeur d'une dimension, classée
        (nuit vs jour graves ; types aux actions efficaces ; types SANS action = part basse).
  - "frequence"  : nombre par valeur d'une dimension, classé
        (types concentrant le plus d'accidents ; compagnie la plus fréquente).
  - "tendance"   : évolution dans le temps (par année : hausse/baisse ; par mois : saisonnalité).
"""
from __future__ import annotations

import json
import logging
import os
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date as _date
from typing import Any, Optional

from clients import Neo4jClient, OllamaClient

logger = logging.getLogger(__name__)
LLM_MODEL = os.environ.get("LLM_MODEL_INCIDENT_V2", "qwen2.5:7b")   # GÉNÉRATION (rédaction fidèle)
# PLANIFICATION (question → spec) : maillon où le 7b flappe → 14b. Right-size par tâche.
PLAN_MODEL = os.environ.get("PLANNER_MODEL", "qwen2.5:14b")

# ── DIMENSIONS groupables (schéma). prop -> propriété nœud ; rel -> (relation,label,prop) ──
_DIM_PROP = {
    "severite": "severite", "classification": "classification",
    "condition_lumineuse": "condition_lumineuse", "etat": "etat",
}
_DIM_REL = {
    "compagnie":      ("IMPLIQUE_COMPAGNIE", "Compagnie",     "nom"),
    "type_evenement": ("DE_TYPE",            "TypeEvenement", "label"),
    "lieu":           ("LOCALISE_EN",        "Lieu",          "label"),
    "phase_vol":      ("EN_PHASE_DE_VOL",    "PhaseVol",      "label"),
    "aeronef":        ("IMPLIQUE_AERONEF",   "TypeAeronef",   "label"),
    "service":        ("RESPONSABLE",        "Service",       "label"),
    "facteur":        ("A_POUR_FACTEUR",     "FacteurContributif", "label"),  # le « pourquoi » extrait du texte
}
_DIMENSIONS = set(_DIM_PROP) | set(_DIM_REL)

# ── CIBLES (condition Cypher, libellé) — le "de quoi mesure-t-on la part" ──
_CIBLES: dict[str, tuple[str, str]] = {
    "gravite":           ("i.severite IN ['4 - élevé', '5 - intolérable']", "graves ou intolérables"),
    "blesses":           ("i.presence_blesses = true", "avec blessés"),
    "avec_action":       ("EXISTS { (i)-[:A_POUR_ACTION]->() }", "avec action corrective"),
    "actions_efficaces": ("i.actions_efficaces = true", "aux actions jugées efficaces"),
}
_MOIS = {"01": "janvier", "02": "février", "03": "mars", "04": "avril", "05": "mai",
         "06": "juin", "07": "juillet", "08": "août", "09": "septembre",
         "10": "octobre", "11": "novembre", "12": "décembre"}
_SEUIL_REL = 30  # ignore les valeurs de relation trop rares (bruit statistique)

# ── ENTITÉ « ACTION » (2ᵉ entité racine) : dimensions et cibles propres ──
# Les opérateurs sont AGNOSTIQUES ; seul change ce registre selon l'entité.
_BASE_ACTION = ("MATCH (i0:IncidentSecu)-[r:A_POUR_ACTION]->(a:Action) "
                "WHERE coalesce(i0.is_test_data, false) = false")
_DIM_ACTION = {                       # dimension -> expression (sur r ou a)
    "type_action": "r.type_action",
    "statut":      "a.etat_avancement",
    # "responsable" retirée volontairement : a.responsable = logins de personnes
    # → classement d'individus interdit (culture juste, cf. confidentialite.py).
}
_CLOT_A = "(a.statut = '100' OR a.date_cloture IS NOT NULL)"
_CIBLES_ACTION: dict[str, tuple[str, str]] = {
    "cloturee":  (_CLOT_A, "clôturées"),
    "en_retard": (f"(a.date_prevue IS NOT NULL AND a.date_prevue < $today AND NOT {_CLOT_A})",
                  "en retard"),
}

_PLAN_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "operateur":  {"type": "string",
                       "enum": ["proportion", "frequence", "tendance", "synthese",
                                "cooccurrence", "non_applicable"]},
        "entite":     {"type": ["string", "null"]},
        "dimension":  {"type": ["string", "null"]},
        "dimensions": {"type": ["array", "null"], "items": {"type": "string"}},
        "cible":      {"type": ["string", "null"]},
        "filtre_classes": {"type": ["array", "null"], "items": {"type": "string"}},
        "filtre_type": {"type": ["string", "null"]},
        "filtre_severite": {"type": ["string", "null"]},
        "conditions": {"type": ["array", "null"], "items": {
            "type": "object",
            "properties": {"dim": {"type": "string"}, "valeur": {"type": "string"}},
            "required": ["dim", "valeur"]}},
        "granularite": {"type": ["string", "null"]},
        "sens":       {"type": ["string", "null"]},
    },
    "required": ["operateur"],
}

_PLAN_PROMPT = """Tu es le PLANIFICATEUR d'un agent analyste d'incidents de sécurité aéroportuaire.
Mappe la question vers UN opérateur analytique et ses paramètres. Retourne UNIQUEMENT le JSON.

operateur :
  "proportion" → la PART d'une caractéristique par catégorie ("plus graves", "proportionnellement",
     "davantage de blessés", "pour quels types les actions sont efficaces", "types SANS action").
  "frequence"  → le NOMBRE par catégorie, un classement ("quels types concentrent le plus de…",
     "le plus fréquent", "répartition par…", "top…"). AUSSI les questions de CHOIX entre
     des valeurs d'une dimension : « plutôt X ou Y ? », « c'est X ou Y ? », « davantage X ou Y ? »
     (ex. « plutôt de jour ou de nuit ? » → frequence sur condition_lumineuse ; l'argmax répond).
  "tendance"   → évolution DANS LE TEMPS ("augmentent-elles", "évolue", "saisonnalité", "par mois").
  "cooccurrence" → CROISER plusieurs dimensions pour trouver des COMBINAISONS / SIGNATURES
     de risque ("quelles situations sont les plus dangereuses", "croiser type, lieu et phase",
     "où ET à quelle phase surviennent les X", "quelle combinaison revient le plus").
     Remplir "dimensions" (liste de 2 à 3), pas "dimension".
  "synthese"   → SYNTHÈSE QUALITATIVE d'une population filtrée, UNIQUEMENT si la question
     demande des MOTIFS / CAUSES RÉCURRENTS ou des ENSEIGNEMENTS :
     « causes récurrentes des Y ? », « que révèlent les X ? », « que retenir de Z ? »,
     « quels enseignements / facteurs récurrents ? ». Exige un filtre (filtre_severite/type/classes).
     ⚠️  NE PAS choisir "synthese" pour un simple RÉCIT ou une DESCRIPTION :
       « raconte les X », « décris les Y », « que s'est-il passé lors de X »,
       « donne des exemples de X », « parle-moi de X » → ce sont de la RECHERCHE → "non_applicable".
       Et « que recommandes-tu / que faire face à X » → recommandation → "non_applicable".
  "non_applicable" → comptage simple, liste, récit/description (recherche), recommandation.

⚠️  RÈGLE DÉCISIVE : proportion/frequence/cooccurrence exigent une CATÉGORIE D'ANALYSE
   nommée dans la question (« par X », « quel type/lieu/statut/facteur… »,
   un superlatif « le plus… », une comparaison « X ou Y »). SANS catégorie explicite,
   un « combien… » ou « nombre de… » est un COMPTAGE SCALAIRE → "non_applicable".
   Ex. « combien d'actions clôturées ? » (aucune catégorie) → non_applicable (PAS proportion).
   Ex. « les actions en cours » (une liste) → non_applicable.

entite (SUR QUOI porte l'analyse) :
  "incident" (défaut) → l'analyse porte sur les incidents.
  "action"   → l'analyse porte sur les ACTIONS de traitement elles-mêmes
     (« répartition des ACTIONS par type », « proportion d'ACTIONS clôturées »,
      « les ACTIONS augmentent-elles »).
     Le SUJET compté/analysé est « action(s) » → entite="action".

dimension (la catégorie d'analyse) :
  si entite="incident" : severite | classification | condition_lumineuse | etat
     | compagnie | type_evenement | lieu | phase_vol | aeronef | service
     | facteur (le FACTEUR CONTRIBUTIF / la CAUSE : « pourquoi », « facteur », « cause »,
       ex. sous-effectif, inexpérience, stress, défaut matériel, non-conformité)
  si entite="action"   : type_action | statut

dimensions (pour "cooccurrence" uniquement) : liste de 2 à 3 dimensions incident à croiser,
  ex. ["type_evenement","lieu","phase_vol"] ou ["compagnie","type_evenement"].

cible (pour "proportion" — ce dont on mesure la part) :
  si entite="incident" : gravite | blesses | avec_action | actions_efficaces
  si entite="action"   : cloturee (part clôturée) | en_retard (part en retard)

filtre_classes (optionnel) : liste de classifications restreignant la population,
  ex. ["Accident","Incident sérieux"] pour "les accidents et incidents sérieux".
filtre_type (pour "tendance"/"synthese") : le type d'incident, recopié ("FOD","collision aviaire","quasi-collision","incursion") ou null.
filtre_severite (pour "synthese") : "5 - intolérable" | "4 - élevé" | "3 - important" ou null.
conditions (optionnel, pour proportion/frequence) : RESTREINT la population À UN SOUS-ENSEMBLE
  avant de compter, quand la question dit « PARMI les X… », « pour les X… », « chez les X… »,
  « dans les X… ». Liste de {dim, valeur} où `dim` est une dimension (condition_lumineuse,
  type_evenement, phase_vol, facteur, compagnie, lieu, aeronef, service) ou "gravite"
  (= graves / intolérables), et `valeur` la catégorie visée, RECOPIÉE de la question.
  La `dimension` reste la CATÉGORIE D'ANALYSE (l'argmax demandé), DISTINCTE des conditions.
  ⚠️  N'utilise PAS `conditions` dans ces cas :
   - « graves / avec blessés » quand c'est CE QU'ON MESURE : « quel type a la plus forte
     PROPORTION d'incidents graves ? » → proportion + cible=gravite, conditions=null.
     Mets `gravite` en condition SEULEMENT si l'argmax porte sur une AUTRE dimension
     (« parmi les incidents GRAVES, quel FACTEUR domine ? » → dimension=facteur + condition gravite).
   - une EXCLUSION (« si on excluait / sauf / hors les X ») → operateur "non_applicable".
   - une simple LISTE (« quelles actions pour les incidents X ») → operateur "non_applicable".
granularite (pour "tendance") : "annee" | "mois".
sens : "haut" (le plus / classement décroissant, défaut) | "bas" (le moins / SANS / classement croissant).

Exemples :
"Quels types d'événements concentrent le plus d'accidents et d'incidents sérieux ?"
→ {"operateur":"frequence","dimension":"type_evenement","cible":null,"filtre_classes":["Accident","Incident sérieux"],"filtre_type":null,"granularite":null,"sens":"haut"}
"Les incidents de nuit sont-ils proportionnellement plus graves que ceux de jour ?"
→ {"operateur":"proportion","dimension":"condition_lumineuse","cible":"gravite","filtre_classes":null,"filtre_type":null,"granularite":null,"sens":"haut"}
"Quel type d'événement a la plus forte proportion d'incidents graves ?"
→ {"operateur":"proportion","dimension":"type_evenement","cible":"gravite","conditions":null,"filtre_classes":null,"filtre_type":null,"granularite":null,"sens":"haut"}
"Si on excluait les FOD et les collisions aviaires, quel type serait le plus fréquent ?"
→ {"operateur":"non_applicable","dimension":null,"cible":null,"conditions":null,"filtre_classes":null,"filtre_type":null,"granularite":null,"sens":null}
"Pour quels types d'incidents les actions correctives sont-elles jugées efficaces ?"
→ {"operateur":"proportion","dimension":"type_evenement","cible":"actions_efficaces","filtre_classes":null,"filtre_type":null,"granularite":null,"sens":"haut"}
"Quels types d'incidents fréquents restent sans action corrective ?"
→ {"operateur":"proportion","dimension":"type_evenement","cible":"avec_action","filtre_classes":null,"filtre_type":null,"granularite":null,"sens":"bas"}
"Quelle compagnie a le plus d'incidents ?"
→ {"operateur":"frequence","dimension":"compagnie","cible":null,"filtre_classes":null,"filtre_type":null,"granularite":null,"sens":"haut"}
"Parmi les incidents de nuit, quel type est le plus fréquent ?"
→ {"operateur":"frequence","dimension":"type_evenement","conditions":[{"dim":"condition_lumineuse","valeur":"Nuit"}],"cible":null,"filtre_classes":null,"filtre_type":null,"granularite":null,"sens":"haut"}
"Quelle compagnie est la plus impliquée dans les collisions aviaires ?"
→ {"operateur":"frequence","dimension":"compagnie","conditions":[{"dim":"type_evenement","valeur":"Collision aviaire"}],"cible":null,"filtre_classes":null,"filtre_type":null,"granularite":null,"sens":"haut"}
"Pour les incidents graves au roulage, quel facteur contributif domine ?"
→ {"operateur":"frequence","dimension":"facteur","conditions":[{"dim":"gravite","valeur":"graves"},{"dim":"phase_vol","valeur":"Roulage"}],"cible":null,"filtre_classes":null,"filtre_type":null,"granularite":null,"sens":"haut"}
"Quels sont les facteurs contributifs les plus fréquents ?"
→ {"operateur":"frequence","entite":"incident","dimension":"facteur","cible":null,"filtre_classes":null,"filtre_type":null,"granularite":null,"sens":"haut"}
"Pour quel type d'événement le sous-effectif revient-il le plus ?"
→ {"operateur":"cooccurrence","dimensions":["type_evenement","facteur"],"cible":null,"dimension":null,"entite":"incident","filtre_classes":null,"filtre_type":null,"filtre_severite":null,"granularite":null,"sens":null}
"Les incidents surviennent-ils plutôt de jour ou de nuit ?"
→ {"operateur":"frequence","entite":"incident","dimension":"condition_lumineuse","cible":null,"filtre_classes":null,"filtre_type":null,"granularite":null,"sens":"haut"}
"Les collisions aviaires suivent-elles une saisonnalité ?"
→ {"operateur":"tendance","dimension":null,"cible":null,"filtre_classes":null,"filtre_type":"collision aviaire","granularite":"mois","sens":null}
"Les incursions augmentent-elles ces dernières années ?"
→ {"operateur":"tendance","dimension":null,"cible":null,"filtre_classes":null,"filtre_type":"incursion","granularite":"annee","sens":null}
"Que révèlent les 22 incidents de gravité intolérable ?"
→ {"operateur":"synthese","dimension":null,"cible":null,"filtre_classes":null,"filtre_type":null,"filtre_severite":"5 - intolérable","granularite":null,"sens":null}
"Quelles sont les causes récurrentes des quasi-collisions avec un véhicule ?"
→ {"operateur":"synthese","dimension":null,"cible":null,"filtre_classes":null,"filtre_type":"quasi-collision","filtre_severite":null,"granularite":null,"sens":null}
"Répartition des actions par type"
→ {"operateur":"frequence","entite":"action","dimension":"type_action","cible":null,"filtre_classes":null,"filtre_type":null,"granularite":null,"sens":"haut"}
"Quel statut d'avancement est le plus courant pour les actions ?"
→ {"operateur":"frequence","entite":"action","dimension":"statut","cible":null,"filtre_classes":null,"filtre_type":null,"granularite":null,"sens":"haut"}
"Quelle proportion des actions correctives sont clôturées ?"
→ {"operateur":"proportion","entite":"action","dimension":"type_action","cible":"cloturee","filtre_classes":null,"filtre_type":null,"granularite":null,"sens":"haut"}
"Les actions correctives augmentent-elles dans le temps ?"
→ {"operateur":"tendance","entite":"action","dimension":null,"cible":null,"filtre_classes":null,"filtre_type":null,"granularite":"annee","sens":null}
"Quelles combinaisons type d'événement, lieu et phase sont les plus dangereuses ?"
→ {"operateur":"cooccurrence","dimensions":["type_evenement","lieu","phase_vol"],"cible":"gravite","dimension":null,"entite":"incident","filtre_classes":null,"filtre_type":null,"filtre_severite":null,"granularite":null,"sens":null}
"Où et à quelle phase surviennent le plus les incidents ?"
→ {"operateur":"cooccurrence","dimensions":["lieu","phase_vol"],"cible":null,"dimension":null,"entite":"incident","filtre_classes":null,"filtre_type":null,"filtre_severite":null,"granularite":null,"sens":null}
"Combien d'accidents en 2024 ?"
→ {"operateur":"non_applicable","dimension":null,"cible":null,"filtre_classes":null,"filtre_type":null,"filtre_severite":null,"granularite":null,"sens":null}
"Quelles actions sont encore en cours ?"
→ {"operateur":"non_applicable","dimension":null,"cible":null,"filtre_classes":null,"filtre_type":null,"filtre_severite":null,"granularite":null,"sens":null}
"Combien d'actions clôturées ?"
→ {"operateur":"non_applicable","dimension":null,"cible":null,"filtre_classes":null,"filtre_type":null,"filtre_severite":null,"granularite":null,"sens":null}

QUESTION : {question}
"""

_SYNTH = """Tu es un analyste sécurité. Réponds à la question UNIQUEMENT à partir des CHIFFRES
CALCULÉS ci-dessous.

⚠️  RÈGLE ABSOLUE DE FIDÉLITÉ : n'écris QUE des nombres présents littéralement ci-dessous.
   N'invente, ne calcule et n'estime AUCUN taux de croissance, moyenne, pourcentage ou
   total qui n'est pas explicitement fourni. Un écart entre deux valeurs fournies est permis,
   rien d'autre. En cas de doute, décris la tendance en MOTS (« augmente », « stable »)
   sans chiffre inventé.

Question : {question}
{bloc}

Consignes : {consigne} 3 à 5 phrases, ton d'expert, factuel.
"""


@dataclass
class AnalysteResult:
    status: str
    answer: str = ""
    plan: Optional[dict] = None
    faits: Any = None


def _plan(question: str, ollama: OllamaClient) -> Optional[dict]:
    try:
        raw = ollama.generate_structured(_PLAN_PROMPT.replace("{question}", question),
                                         _PLAN_SCHEMA, model=PLAN_MODEL, timeout=90.0)
        return json.loads(raw)
    except Exception as e:  # noqa: BLE001
        logger.warning("Plan analyste échoué : %s", e)
        return None


def _ints(obj) -> set[int]:
    """Tous les entiers présents dans une structure (via sa sérialisation)."""
    txt = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False)
    out: set[int] = set()
    for x in re.findall(r"\d[\d\s]*\d|\d", txt):
        n = re.sub(r"[^\d]", "", x)
        if n:
            out.add(int(n))
    return out


def _fantomes(texte: str, autorises: set[int]) -> set[int]:
    """Nombres > 100 du texte absents des faits (candidats hallucination)."""
    return {n for n in _ints(texte) if n > 100 and n not in autorises}


def _retirer_fantomes(texte: str, autorises: set[int]) -> str:
    """Dernier recours : retire les phrases contenant un nombre fantôme."""
    phrases = re.split(r"(?<=[.!?])\s+", texte)
    gardees = [p for p in phrases if not _fantomes(p, autorises)]
    return " ".join(gardees).strip() or texte


def _generer_fidele(prompt: str, faits: Any, ollama: OllamaClient) -> str:
    """Génère puis VÉRIFIE par code que la prose ne contient aucun nombre inventé.
    Instruction seule insuffisante sur un 7b (ex. tendance qui invente un taux) →
    garde-fou déterministe : re-génération avec liste explicite, puis retrait."""
    autorises = _ints(faits)
    ans = ollama.generate(prompt=prompt, model=LLM_MODEL, temperature=0.1).strip()
    if not _fantomes(ans, autorises):
        return ans
    permis = ", ".join(str(x) for x in sorted(n for n in autorises if n > 100)) or "(aucun > 100)"
    rappel = (f"\n\nRAPPEL ABSOLU : les SEULS nombres supérieurs à 100 autorisés sont : {permis}. "
              f"Tout autre est INTERDIT (ne calcule aucun taux, total ou moyenne). Décris le reste en mots.")
    ans2 = ollama.generate(prompt=prompt + rappel, model=LLM_MODEL, temperature=0.0).strip()
    if not _fantomes(ans2, autorises):
        return ans2
    logger.warning("Fidélité : nombres fantômes persistants %s → retrait", _fantomes(ans2, autorises))
    return _retirer_fantomes(ans2 or ans, autorises)


def _mot_cle(v: str) -> str:
    """Mot le plus long, sans accents, minuscule — pour matcher les labels snake_case
    de FacteurContributif (« défaut de communication » → « communication »), dont les
    8 tokens sont tous discriminés par leur mot le plus long."""
    n = "".join(c for c in unicodedata.normalize("NFD", str(v or ""))
                if unicodedata.category(c) != "Mn").lower()
    toks = re.findall(r"[a-z0-9]{3,}", n)
    return max(toks, key=len) if toks else n


def _cond_where(conditions, params) -> tuple[str, bool]:
    """Traduit une liste {dim, valeur} en clauses WHERE qui RESTREIGNENT la population
    (le « parmi les X… » des questions conditionnelles / multi-hop). Match insensible
    à la casse (CONTAINS) ; propriété → sur i, relation → sous-requête EXISTS ;
    "gravite" → sévérités 4-5. Le `facteur` (labels snake_case) matche par mot-clé.
    Retourne (fragment " AND …", conditions_présentes)."""
    frag: list[str] = []
    for k, c in enumerate(conditions or []):
        dim, val = (c or {}).get("dim"), (c or {}).get("valeur")
        if not dim or val in (None, ""):
            continue
        if dim == "gravite":
            frag.append("i.severite IN ['4 - élevé', '5 - intolérable']")
            continue
        pkey = f"c{k}"
        if dim in _DIM_PROP:
            frag.append(f"toLower(i.`{_DIM_PROP[dim]}`) CONTAINS toLower(${pkey})")
            params[pkey] = val
        elif dim in _DIM_REL:
            rel, lbl, prop = _DIM_REL[dim]
            if dim == "facteur":  # labels snake_case → match par mot-clé, « _ » → espace
                frag.append(f"EXISTS {{ MATCH (i)-[:{rel}]->(z{k}:{lbl}) "
                            f"WHERE toLower(replace(z{k}.`{prop}`, '_', ' ')) CONTAINS ${pkey} }}")
                params[pkey] = _mot_cle(val)
            else:
                frag.append(f"EXISTS {{ MATCH (i)-[:{rel}]->(z{k}:{lbl}) "
                            f"WHERE toLower(z{k}.`{prop}`) CONTAINS toLower(${pkey}) }}")
                params[pkey] = val
    return ((" AND " + " AND ".join(frag)) if frag else ""), bool(frag)


def _par_dimension(neo4j, dimension, cible_cond, filtre_classes, conditions=None):
    """UNE requête : par valeur de la dimension, total (+ compte-cible si demandé).
    `conditions` restreint d'abord la population (questions « parmi les X… »)."""
    params: dict[str, Any] = {}
    fclause = ""
    if filtre_classes:
        fclause = " AND i.classification IN $classes"
        params["classes"] = filtre_classes
    cwhere, has_cond = _cond_where(conditions, params)
    fclause += cwhere
    cible_expr = (f"count(DISTINCT CASE WHEN {cible_cond} THEN i END)"
                  if cible_cond else "0")
    if dimension in _DIM_PROP:
        prop = _DIM_PROP[dimension]
        cy = (f"MATCH (i:IncidentSecu) WHERE coalesce(i.is_test_data,false)=false{fclause} "
              f"AND i.`{prop}` IS NOT NULL "
              f"WITH i.`{prop}` AS val, count(DISTINCT i) AS total, {cible_expr} AS cible "
              f"RETURN val, total, cible")
    else:
        rel, lbl, prop = _DIM_REL[dimension]
        # population filtrée → seuil de bruit levé (sinon un argmax rare, ex. n=4, disparaît)
        params["seuil"] = 1 if has_cond else _SEUIL_REL
        cy = (f"MATCH (i:IncidentSecu)-[:{rel}]->(x:{lbl}) "
              f"WHERE coalesce(i.is_test_data,false)=false{fclause} "
              f"WITH x.`{prop}` AS val, count(DISTINCT i) AS total, {cible_expr} AS cible "
              f"WHERE total >= $seuil RETURN val, total, cible")
    rows = neo4j.run(cy, **params)
    return [{"val": r["val"], "total": r["total"], "cible": r["cible"]} for r in rows]


def _par_dimension_action(neo4j, dimension, cible_cond):
    """Idem _par_dimension mais entité racine = ACTION (count DISTINCT a)."""
    expr = _DIM_ACTION[dimension]
    params: dict[str, Any] = {}
    cible_expr = f"count(DISTINCT CASE WHEN {cible_cond} THEN a END)" if cible_cond else "0"
    if cible_cond and "$today" in cible_cond:
        params["today"] = _date.today().isoformat()
    cy = (f"{_BASE_ACTION} AND {expr} IS NOT NULL "
          f"WITH {expr} AS val, count(DISTINCT a) AS total, {cible_expr} AS cible "
          f"RETURN val, total, cible")
    return [{"val": r["val"], "total": r["total"], "cible": r["cible"]} for r in neo4j.run(cy, **params)]


def _scope_txt(plan) -> str:
    """Libellé de la population restreinte (filtre_classes + conditions), pour le bloc."""
    parts: list[str] = list(plan.get("filtre_classes") or [])
    for c in (plan.get("conditions") or []):
        d, v = (c or {}).get("dim"), (c or {}).get("valeur")
        parts.append("graves/intolérables" if d == "gravite" else str(v))
    parts = [p for p in parts if p]
    return (" (parmi les incidents : " + ", ".join(parts) + ")") if parts else ""


def _op_proportion(question, plan, ollama, neo4j):
    ent = plan.get("entite") or "incident"
    dim, cible = plan.get("dimension"), plan.get("cible")
    if plan.get("filtre_type"):
        return AnalysteResult(status="non_analyste", plan=plan)
    if ent == "action":
        if dim not in _DIM_ACTION or cible not in _CIBLES_ACTION:
            return AnalysteResult(status="non_analyste", plan=plan)
        cond, lib = _CIBLES_ACTION[cible]
        rows = _par_dimension_action(neo4j, dim, cond)
        sujet = "actions"
    else:
        if dim not in _DIMENSIONS or cible not in _CIBLES:
            return AnalysteResult(status="non_analyste", plan=plan)
        cond, lib = _CIBLES[cible]
        rows = _par_dimension(neo4j, dim, cond, plan.get("filtre_classes"), plan.get("conditions"))
        sujet = "incidents"
    for r in rows:
        r["pct"] = round(100 * r["cible"] / r["total"], 1) if r["total"] else 0.0
    sens = plan.get("sens") or "haut"
    rows.sort(key=lambda r: r["pct"], reverse=(sens != "bas"))
    rows = rows[:8]
    scope = _scope_txt(plan) if (plan.get("entite") or "incident") != "action" else ""
    bloc = (f"Part de « {lib} » par {dim}{scope} (classé du plus {'faible' if sens=='bas' else 'fort'}) :\n"
            + "\n".join(f"- {r['val']} : {r['cible']}/{r['total']} = {r['pct']} %" for r in rows))
    consigne = ("Dis quelle(s) valeur(s) ressort(ent) et l'écart. Rappelle que c'est PROPORTIONNEL. "
                + ("Les valeurs en tête sont celles qui ont le MOINS cette caractéristique."
                   if sens == "bas" else ""))
    ans = _generer_fidele(_SYNTH.replace("{question}", question)
                          .replace("{bloc}", bloc).replace("{consigne}", consigne), bloc, ollama)
    return AnalysteResult(status="ok", answer=ans, plan=plan, faits=rows)


def _op_frequence(question, plan, ollama, neo4j):
    ent = plan.get("entite") or "incident"
    dim = plan.get("dimension")
    if plan.get("filtre_type"):
        return AnalysteResult(status="non_analyste", plan=plan)
    if ent == "action":
        if dim not in _DIM_ACTION:
            return AnalysteResult(status="non_analyste", plan=plan)
        rows = _par_dimension_action(neo4j, dim, None)
        sujet, scope = "actions", ""
    else:
        if dim not in _DIMENSIONS:
            return AnalysteResult(status="non_analyste", plan=plan)
        rows = _par_dimension(neo4j, dim, None, plan.get("filtre_classes"), plan.get("conditions"))
        sujet = "incidents"
        scope = _scope_txt(plan)
    sens = plan.get("sens") or "haut"
    rows.sort(key=lambda r: r["total"], reverse=(sens != "bas"))
    rows = rows[:8]
    bloc = (f"Nombre d'{sujet} par {dim}{scope} :\n"
            + "\n".join(f"- {r['val']} : {r['total']}" for r in rows))
    consigne = ("Nomme comme dominante EXACTEMENT la valeur de la PREMIÈRE ligne du bloc, "
                "telle qu'écrite — même si son libellé est étrange —, puis donne le "
                "classement de tête. Ne cite aucune valeur absente du bloc.")
    ans = _generer_fidele(_SYNTH.replace("{question}", question)
                          .replace("{bloc}", bloc).replace("{consigne}", consigne), bloc, ollama)
    return AnalysteResult(status="ok", answer=ans, plan=plan, faits=rows)


def _op_tendance(question, plan, ollama, neo4j):
    gran = plan.get("granularite") or "annee"
    if gran not in ("annee", "mois"):
        return AnalysteResult(status="non_analyste", plan=plan)
    ent = plan.get("entite") or "incident"
    params: dict[str, Any] = {}
    if ent == "action":
        # tendance des ACTIONS dans le temps (date de création) ; count DISTINCT a
        datecol, cnt, alias = "a.date_ajout", "count(DISTINCT a)", "a"
        where = [f"coalesce(i0.is_test_data,false)=false", f"{datecol} IS NOT NULL"]
        match = "MATCH (i0:IncidentSecu)-[r:A_POUR_ACTION]->(a:Action)"
        sujet = "actions"
    else:
        datecol, cnt = "i.date_evenement", "count(DISTINCT i)"
        where = ["coalesce(i.is_test_data,false)=false", f"{datecol} IS NOT NULL"]
        match = "MATCH (i:IncidentSecu)"
        sujet = "incidents"
        type_evt = plan.get("filtre_type")
        if type_evt:
            match += " MATCH (i)-[:DE_TYPE]->(t:TypeEvenement)"
            where.append("toLower(t.label) CONTAINS toLower($te)")
            params["te"] = type_evt
    expr = f"substring({datecol},5,2)" if gran == "mois" else f"substring({datecol},0,4)"
    cy = (match + " WHERE " + " AND ".join(where) +
          f" RETURN {expr} AS periode, {cnt} AS n ORDER BY periode")
    serie = [{"periode": r["periode"], "n": r["n"]} for r in neo4j.run(cy, **params)]
    if not serie:
        return AnalysteResult(status="non_analyste", plan=plan)
    if gran == "mois":
        for f in serie:
            f["label"] = _MOIS.get(f["periode"], f["periode"])
        top = sorted(serie, key=lambda f: f["n"], reverse=True)[:3]
        entete = "pics : " + ", ".join(f"{f['label']} ({f['n']})" for f in top)
        bloc = f"{sujet.capitalize()} par mois de l'année :\n" + "\n".join(f"- {f['label']} : {f['n']}" for f in serie)
        consigne = f"Décris la saisonnalité ({entete}), les mois creux, la période à renforcer."
    else:
        ns = [f["n"] for f in serie]
        k = max(2, len(ns) // 3)
        recent, ancien = sum(ns[-k:]) / k, sum(ns[:k]) / k
        direction = ("EN HAUSSE" if recent > ancien * 1.25
                     else "EN BAISSE" if recent < ancien * 0.8 else "STABLE")
        bloc = (f"{sujet.capitalize()} par année :\n" + "\n".join(f"- {f['periode']} : {f['n']}" for f in serie)
                + f"\nTendance calculée : {direction} (récent ~{recent:.0f}/an vs début ~{ancien:.0f}/an)")
        consigne = ("Dis si le phénomène augmente, baisse ou est stable, en citant des années "
                    "de la série. NE CALCULE PAS et n'invente PAS de taux de croissance chiffré : "
                    "cite uniquement des valeurs présentes dans la série et la direction fournie.")
    ans = _generer_fidele(_SYNTH.replace("{question}", question)
                          .replace("{bloc}", bloc).replace("{consigne}", consigne), bloc, ollama)
    return AnalysteResult(status="ok", answer=ans, plan=plan, faits=serie)


def _population(neo4j, plan):
    """Filtre déterministe → (total, échantillon de résumés de CETTE population)."""
    where = ["coalesce(i.is_test_data,false)=false"]
    params: dict[str, Any] = {}
    match = "MATCH (i:IncidentSecu)"
    if plan.get("filtre_type"):
        match += " MATCH (i)-[:DE_TYPE]->(t:TypeEvenement)"
        where.append("toLower(t.label) CONTAINS toLower($te)")
        params["te"] = plan["filtre_type"]
    if plan.get("filtre_severite"):
        where.append("i.severite = $sev")
        params["sev"] = plan["filtre_severite"]
    if plan.get("filtre_classes"):
        where.append("i.classification IN $cls")
        params["cls"] = plan["filtre_classes"]
    base = match + " WHERE " + " AND ".join(where)
    total = neo4j.run(base + " RETURN count(DISTINCT i) AS n", **params)[0]["n"]
    rows = neo4j.run(base + " AND i.resume_llm IS NOT NULL AND i.resume_llm <> '0' "
                     "RETURN DISTINCT i.numero_fe AS fe, left(i.resume_llm, 320) AS r "
                     "ORDER BY i.numero_fe LIMIT 22", **params)
    return total, [{"fe": r["fe"], "resume": r["r"]} for r in rows]


_SYNTH_POP = """Tu es un analyste sécurité (RETEX). Tu dois SYNTHÉTISER ce que révèle une
population d'incidents. Base-toi UNIQUEMENT sur les résumés fournis, n'invente rien.

Question : {question}
Cette population compte {total} incidents. En voici un échantillon de {k} résumés :
{contexte}

Consignes :
- NE dis JAMAIS que la population est vide : elle compte {total} incidents.
- Dégage les MOTIFS ou CAUSES RÉCURRENTS (regroupe par thème), pas une liste fiche par fiche.
- Cite 3 à 5 numéros FE en appui. 4 à 7 phrases, ton d'expert.
"""


def _op_synthese(question, plan, ollama, neo4j):
    if not (plan.get("filtre_severite") or plan.get("filtre_type") or plan.get("filtre_classes")):
        return AnalysteResult(status="non_analyste", plan=plan)
    total, echant = _population(neo4j, plan)
    if total == 0:
        return AnalysteResult(status="ok", plan=plan, faits={"total": 0},
                              answer="Aucun incident ne correspond à ces critères dans la base.")
    contexte = "\n".join(f"[{e['fe']}] {e['resume']}" for e in echant)
    prompt = (_SYNTH_POP.replace("{question}", question).replace("{total}", str(total))
              .replace("{k}", str(len(echant))).replace("{contexte}", contexte))
    ans = ollama.generate(prompt=prompt, model=LLM_MODEL, temperature=0.1, num_ctx=8192)
    return AnalysteResult(status="ok", answer=ans.strip(), plan=plan,
                          faits={"total": total, "echantillon": [e["fe"] for e in echant]})


def _cooccurrence(neo4j, dims, cible_cond, limit=8):
    """TRAVERSÉE : croise N dimensions (satellites) sur le même incident → combinaisons.
    Extensible : chaque dimension ajoute un saut ; demain une dim-chemin = un multi-hop."""
    matches = ["MATCH (i:IncidentSecu)"]
    where = ["coalesce(i.is_test_data,false)=false"]
    params: dict[str, Any] = {}
    if cible_cond:
        where.append(cible_cond)
        if "$today" in cible_cond:
            params["today"] = _date.today().isoformat()
    exprs = []
    for idx, dim in enumerate(dims):
        if dim in _DIM_PROP:
            e = f"i.`{_DIM_PROP[dim]}`"
        else:
            rel, lbl, prop = _DIM_REL[dim]
            matches.append(f"MATCH (i)-[:{rel}]->(d{idx}:{lbl})")
            e = f"d{idx}.`{prop}`"
        where.append(f"{e} IS NOT NULL")
        exprs.append(e)
    ret = ", ".join(f"{e} AS v{idx}" for idx, e in enumerate(exprs))
    cy = (" ".join(matches) + " WHERE " + " AND ".join(where) +
          f" RETURN {ret}, count(DISTINCT i) AS n ORDER BY n DESC LIMIT {limit}")
    return [{"combo": [r[f"v{idx}"] for idx in range(len(dims))], "n": r["n"]}
            for r in neo4j.run(cy, **params)]


def _op_cooccurrence(question, plan, ollama, neo4j):
    dims = [d for d in (plan.get("dimensions") or []) if d in _DIMENSIONS][:3]
    if len(dims) < 2:
        return AnalysteResult(status="non_analyste", plan=plan)
    cible = plan.get("cible")
    cond = _CIBLES[cible][0] if cible in _CIBLES else None
    rows = _cooccurrence(neo4j, dims, cond)
    if not rows:
        return AnalysteResult(status="non_analyste", plan=plan)
    scope = f" ({_CIBLES[cible][1]})" if cible in _CIBLES else ""
    bloc = (f"Combinaisons {' × '.join(dims)}{scope}, classées par nombre d'incidents :\n"
            + "\n".join(f"- {' · '.join(str(v) for v in r['combo'])} : {r['n']}" for r in rows))
    consigne = ("Décris les 2-3 combinaisons (situations) qui ressortent le plus — ce sont des "
                "SIGNATURES de risque : dis quel type se cumule avec quel lieu/quelle phase.")
    ans = _generer_fidele(_SYNTH.replace("{question}", question)
                          .replace("{bloc}", bloc).replace("{consigne}", consigne), bloc, ollama)
    return AnalysteResult(status="ok", answer=ans, plan=plan, faits=rows)


_OPERATEURS = {"proportion": _op_proportion, "frequence": _op_frequence,
               "tendance": _op_tendance, "synthese": _op_synthese,
               "cooccurrence": _op_cooccurrence}


# Garde-fous déterministes AVANT le planner (le 7b sur-grab sous l'effet des exemples
# `conditions`) : hors du périmètre analyste, on décline → la question suit sa bonne voie.
_EXCLUSION_RE = re.compile(r"\b(exclu\w*|sauf|hormis|à\s+part\s+les|hors\s+les|sans\s+les|"
                           r"en\s+excluant|mis\s+à\s+part)\b", re.I)
_LISTE_ACTIONS_RE = re.compile(r"\bquel(?:le|les|s)?\s+actions?\b", re.I)
# comptage SCALAIRE (« combien d'incidents de nuit ? ») : sans catégorie d'analyse nommée,
# c'est l'agrégation qui répond (chiffre sec) — la RÈGLE DÉCISIVE du prompt, rendue autoritaire.
_COMPTAGE_RE = re.compile(r"^\s*(combien|nombre\s+d|quel\s+est\s+le\s+nombre)", re.I)
_CAT_MARQUEURS_RE = re.compile(r"\b(par\s|quel(?:le|les|s)?\s|r[ée]partition|proportion|"
                               r"top\s|le\s+plus|la\s+plus|plut[ôo]t)\b", re.I)


def run_analyste(question: str, ollama: OllamaClient, neo4j: Neo4jClient) -> AnalysteResult:
    # anti-filtre (« si on excluait les FOD ») : le moteur ne fait que de l'INCLUSION → décline.
    # liste d'actions (« quelles actions pour les incidents FOD ») → voie actions, pas analyse.
    # comptage scalaire sans catégorie → voie agregation (le chiffre exact), pas une répartition.
    if (_EXCLUSION_RE.search(question) or _LISTE_ACTIONS_RE.search(question)
            or (_COMPTAGE_RE.search(question) and not _CAT_MARQUEURS_RE.search(question))):
        return AnalysteResult(status="non_analyste")
    plan = _plan(question, ollama)
    if not plan:
        return AnalysteResult(status="non_analyste")
    op = _OPERATEURS.get(plan.get("operateur"))
    if op is None:
        return AnalysteResult(status="non_analyste", plan=plan)
    return op(question, plan, ollama, neo4j)
