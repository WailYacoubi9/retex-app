"""
Phrasage des résultats du moteur de requête générique (query_engine_incident_v2).
Count : phrase déterministe (aucun appel LLM). Liste / répartition : LLM avec fallback.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import prompt_store
from clients import OllamaClient

logger = logging.getLogger(__name__)

LLM_MODEL = "qwen2.5:7b"


@dataclass
class QueryGenResult:
    answer: str
    model_used: str


def _filtres_en_clair(filtres: list[dict]) -> str:
    if not filtres:
        return "aucun filtre"
    morceaux = []
    for f in filtres:
        if f["op"] == "est_rempli":
            morceaux.append(f"{f['champ']} renseigné")
        elif f["op"] == "est_vide":
            morceaux.append(f"{f['champ']} vide")
        else:
            morceaux.append(f"{f['champ']} {f['op']} {f['valeur']}")
    return ", ".join(morceaux)




def _contexte_liste(rows: list[dict]) -> str:
    lignes = []
    for r in rows:
        extra = [f"{k}={v}" for k, v in r.items()
                 if k not in ("numero_fe", "titre", "severite", "date_evenement") and v is not None]
        lignes.append(
            f"- {r.get('numero_fe')} | {r.get('titre')} | {str(r.get('date_evenement'))[:10]} "
            f"| {r.get('severite')}" + (f" | {' | '.join(extra[:4])}" if extra else "")
        )
    return "\n".join(lignes)


def _statut_action(statut_raw: str | None) -> str:
    if statut_raw == "100":
        return "clôturée"
    if statut_raw == "0":
        return "en cours"
    if statut_raw is None:
        return "(inconnu)"
    return f"(anomalie: {statut_raw})"


def _contexte_actions_par_incident(rows: list[dict]) -> str:
    lignes: list[str] = []
    for r in rows:
        actions = r.get("actions") or []
        lignes.append(
            f"- {r.get('numero_fe')} | {r.get('titre')} | {r.get('severite')}"
        )
        if actions:
            for a in actions:
                s = _statut_action(a.get("statut"))
                resp = a.get("responsable") or "?"
                lignes.append(
                    f"    [{a.get('type', '?')}] {a.get('titre', '?')} — {s} (resp: {resp})"
                )
        else:
            lignes.append("    (aucune action de traitement structurée)")
    return "\n".join(lignes)


def _contexte_actions_par_action(rows: list[dict]) -> str:
    lignes: list[str] = []
    for r in rows:
        s = _statut_action(r.get("statut"))
        resp = r.get("responsable") or "?"
        lignes.append(
            f"- {r.get('numero_fe')} | [{r.get('type_action', '?')}] "
            f"{r.get('titre_action', '?')} — {s} (resp: {resp})"
        )
    return "\n".join(lignes)


def _filtres_unified_en_clair(spec) -> str:  # spec: UnifiedQuerySpec
    parts = []
    if spec.f_severite:
        parts.append(f"sévérité={spec.f_severite}")
    if spec.f_classification:
        parts.append(f"classification={spec.f_classification}")
    if spec.f_condition_lumineuse:
        parts.append(f"condition lumineuse={spec.f_condition_lumineuse}")
    if spec.f_presence_blesses is not None:
        parts.append(f"blessés={'oui' if spec.f_presence_blesses else 'non'}")
    if spec.f_traitement_termine is not None:
        parts.append(f"traitement terminé={'oui' if spec.f_traitement_termine else 'non'}")
    if spec.f_annee:
        parts.append(f"année={spec.f_annee}")
    if spec.f_mois:
        parts.append(f"mois={spec.f_mois}")
    return ", ".join(parts) or "aucun filtre"


def phrase_unified_result(question: str, result: dict, ollama: OllamaClient) -> QueryGenResult:
    """Phrase le résultat du moteur unifié (count/repartition/liste)."""
    spec = result["spec"]
    resultat_brut = result["resultat_brut"]
    filtres = _filtres_unified_en_clair(spec)

    if spec.output == "count":
        total = int(resultat_brut) if resultat_brut is not None else 0
        if total == 0:
            return QueryGenResult(
                answer=f"Aucun incident ne correspond aux critères ({filtres}).",
                model_used="deterministe",
            )
        return QueryGenResult(
            answer=f"{total} incident(s) correspondent aux critères : {filtres}.",
            model_used="deterministe",
        )

    rows = resultat_brut or []
    if not rows:
        return QueryGenResult(
            answer=f"Aucun incident ne correspond aux critères ({filtres}).",
            model_used="deterministe",
        )

    if spec.output == "repartition":
        total = sum(r.get("n", 0) for r in rows)
        contexte = "\n".join(
            f"- {r['label']} : {r['n']}"
            + (f" ({r['n'] / total * 100:.1f} %)" if total else "")
            for r in rows
        )
        prompt = prompt_store.rendre(
            "question_libre.reponse_repartition",
            question=question, total=total, filtres=filtres, contexte=contexte,
        )
    elif spec.include_actions and spec.shape == "par_incident":
        contexte = _contexte_actions_par_incident(rows)
        prompt = prompt_store.rendre(
            "question_libre.reponse_liste",
            question=question, total=len(rows), filtres=filtres,
            n=len(rows), contexte=contexte,
        )
    elif spec.include_actions and spec.shape == "par_action":
        contexte = _contexte_actions_par_action(rows)
        prompt = prompt_store.rendre(
            "question_libre.reponse_liste",
            question=question, total=len(rows), filtres=filtres,
            n=len(rows), contexte=contexte,
        )
    else:  # liste incidents standard
        contexte = _contexte_liste(rows)
        prompt = prompt_store.rendre(
            "question_libre.reponse_liste",
            question=question, total=len(rows), filtres=filtres,
            n=len(rows), contexte=contexte,
        )

    try:
        answer = ollama.generate(prompt, model=LLM_MODEL)
        return QueryGenResult(answer=answer.strip(), model_used=LLM_MODEL)
    except Exception as e:
        logger.error("Génération LLM échouée : %s", e)
        if spec.output == "repartition":
            brut = "\n".join(f"- {r['label']} : {r['n']}" for r in rows[:15])
        elif spec.include_actions and spec.shape == "par_incident":
            brut = _contexte_actions_par_incident(rows[:10])
        elif spec.include_actions and spec.shape == "par_action":
            brut = _contexte_actions_par_action(rows[:15])
        else:
            brut = _contexte_liste(rows[:15])
        return QueryGenResult(
            answer=f"{len(rows)} résultat(s) ({filtres}) :\n{brut}",
            model_used="fallback",
        )


def phrase_query_result(question: str, result: dict, ollama: OllamaClient) -> QueryGenResult:
    spec = result["spec"]
    total = result["total"] or 0
    rows = result["rows"]
    filtres = _filtres_en_clair(result["filtres"])

    if spec.intent == "count":
        return QueryGenResult(
            answer=f"{total} incident(s) correspondent aux critères : {filtres}.",
            model_used="deterministe",
        )

    if not rows:
        return QueryGenResult(
            answer=f"Aucun incident ne correspond aux critères : {filtres}.",
            model_used="deterministe",
        )

    if spec.intent == "repartition":
        # Pourcentages précalculés : le LLM cite, il ne calcule jamais.
        contexte = "\n".join(
            f"- {r['label']} : {r['n']}"
            + (f" ({r['n'] / total * 100:.1f} %)" if total else "")
            for r in rows
        )
        prompt = prompt_store.rendre("question_libre.reponse_repartition",
            question=question, total=total, filtres=filtres, contexte=contexte)
    else:
        prompt = prompt_store.rendre("question_libre.reponse_liste",
            question=question, total=total, filtres=filtres,
            n=len(rows), contexte=_contexte_liste(rows))

    try:
        answer = ollama.generate(prompt, model=LLM_MODEL)
        return QueryGenResult(answer=answer.strip(), model_used=LLM_MODEL)
    except Exception as e:
        logger.error("Génération LLM échouée : %s", e)
        if spec.intent == "repartition":
            brut = "\n".join(f"- {r['label']} : {r['n']}" for r in rows[:15])
        else:
            brut = _contexte_liste(rows[:15])
        return QueryGenResult(
            answer=f"{total} incident(s) trouvés ({filtres}) :\n{brut}",
            model_used="fallback",
        )
