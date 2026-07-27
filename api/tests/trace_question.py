"""
Microscope à actes — trace une question à travers les voies à formulaire.

Voies :
  query       moteur UNIFIÉ (run_unified_query) — ce que /ask/incident-v2/query
              exécute RÉELLEMENT en production
  query-open  ancien moteur ouvert à catalogue (run_query) — plus branché sur
              l'endpoint, conservé pour comparaison
  stats       agrégation à vocabulaire fermé (run_aggregation)
  actions     actions correctives/préventives (run_action_lookup)

Pour la voie query, montre la spec LLM BRUTE puis les CORRECTIONS apportées
par le post-processing déterministe (les garde-fous du moteur unifié).

Usage (dans le conteneur API) :
  docker exec -it ia-api python tests/trace_question.py "Les incidents graves"
  docker exec -it ia-api python tests/trace_question.py "..." --voie query-open
  docker exec -it ia-api python tests/trace_question.py "..." --comparer qwen2.5:7b,qwen2.5:14b
"""
from __future__ import annotations

import argparse
import ast
import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clients import Neo4jClient, OllamaClient  # noqa: E402


# ─── Capture du journal des modules ──────────────────────────────────────────

class _Capture(logging.Handler):
    """Garde en mémoire les lignes de log émises pendant un appel."""

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.lignes: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lignes.append(f"[{record.name}] {record.getMessage()}")

    def vider(self) -> list[str]:
        lignes, self.lignes = self.lignes, []
        return lignes


def _extraire_cypher(journal: list[str]) -> str | None:
    for ligne in journal:
        if "Cypher:" in ligne:
            return ligne.split("Cypher:", 1)[1].strip()
    return None


def _extraire_spec_brute(journal: list[str]) -> dict | None:
    """Retrouve la spec LLM AVANT post-processing dans le journal."""
    for ligne in journal:
        if "LLM brute" in ligne and ":" in ligne:
            try:
                return ast.literal_eval(ligne.split(" : ", 1)[1])
            except (ValueError, SyntaxError):
                return None
    return None


# ─── Affichage ───────────────────────────────────────────────────────────────

L = 78


def _titre(txt: str) -> None:
    print("\n" + "═" * L)
    print(f"  {txt}")
    print("═" * L)


def _acte(txt: str) -> None:
    print(f"\n── {txt} " + "─" * max(0, L - len(txt) - 4))


def _rows_extrait(rows: list, n: int = 3) -> list:
    return [
        {k: (str(v)[:60] if v is not None else None) for k, v in r.items()}
        for r in rows[:n]
    ]


def _spec_non_nulle(d: dict) -> dict:
    """Champs significatifs d'une spec (retire les None et défauts silencieux)."""
    return {k: v for k, v in d.items() if v is not None and v is not False}


# ─── VOIE query : moteur UNIFIÉ (production) ─────────────────────────────────

def tracer_query(question: str, ollama, neo4j, capture: _Capture) -> dict:
    from query_engine_incident_v2 import run_unified_query

    _titre("VOIE /query PRODUCTION — moteur unifié (UnifiedQuerySpec)")
    capture.vider()
    t0 = time.time()
    result = run_unified_query(question=question, ollama=ollama, neo4j=neo4j)
    duree = round(time.time() - t0, 1)
    journal = capture.vider()

    spec_brute = _extraire_spec_brute(journal)
    spec_finale = result.get("spec_interpretee")

    _acte("ACTE 1 · Spec BRUTE du LLM (avant tout garde-fou)")
    if spec_brute is None:
        print("  (non capturée — LLM non parsé ?)")
    else:
        for k, v in _spec_non_nulle(spec_brute).items():
            print(f"  {k} = {v!r}")

    _acte("ACTE 2 · Corrections du post-processing déterministe")
    if spec_brute and spec_finale:
        diffs = [(k, spec_brute.get(k), spec_finale.get(k))
                 for k in spec_finale if spec_brute.get(k) != spec_finale.get(k)]
        for k, avant, apres in diffs:
            print(f"  ✎ {k} : {avant!r} → {apres!r}")
        if not diffs:
            print("  (aucune correction — le LLM avait tout bon)")
    else:
        print("  (indisponible)")

    _acte("ACTE 3 · Spec FINALE exécutée")
    if spec_finale:
        for k, v in _spec_non_nulle(spec_finale).items():
            print(f"  {k} = {v!r}")

    _acte("ACTE 4 · Compilation + exécution")
    print(f"  Cypher : {result.get('cypher_execute') or '(non exécuté)'}")
    brut = result.get("resultat_brut")
    if isinstance(brut, list):
        resume = f"{len(brut)} lignes"
    else:
        resume = repr(brut)
    print(f"  status : {result['status']}   resultat_brut : {resume}   durée : {duree}s")
    if isinstance(brut, list):
        for r in _rows_extrait(brut):
            print(f"    · {r}")

    _acte("Journal brut capturé")
    for ligne in journal:
        print(f"  {ligne}")

    return {
        "voie": "query(unifié)", "question": question, "duree_s": duree,
        "status": result["status"],
        "spec_llm_brute": spec_brute,
        "spec_finale": spec_finale,
        "cypher": result.get("cypher_execute"),
        "resultat_brut": brut if not isinstance(brut, list) else _rows_extrait(brut, 20),
        "journal": journal,
    }


# ─── VOIE query-open : ancien moteur ouvert (catalogue) ──────────────────────

def tracer_query_open(question: str, ollama, neo4j, capture: _Capture) -> dict:
    from query_engine_incident_v2 import run_query

    _titre("VOIE query-open — ancien moteur ouvert à catalogue (HORS production)")
    capture.vider()
    t0 = time.time()
    result = run_query(question=question, ollama=ollama, neo4j=neo4j)
    duree = round(time.time() - t0, 1)
    journal = capture.vider()

    spec = result.get("spec")
    _acte("ACTE 1 · Traduction LLM → spec")
    if spec is None:
        print("  spec : AUCUNE (LLM non parsé ou demande d'écriture)")
    else:
        print(f"  intent   : {spec.intent}")
        print(f"  group_by : {spec.group_by}")
        for f in spec.filtres:
            print(f"  filtre LLM : {f.champ} {f.op} {f.valeur!r}")
        if not spec.filtres:
            print("  filtre LLM : (aucun)")

    _acte("ACTE 2 · Filtres RETENUS (après validation + garde-fous)")
    for f in result.get("filtres", []):
        print(f"  ✓ {f['champ']} {f['op']} {f['valeur']!r}")
    if not result.get("filtres"):
        print("  (aucun)")

    _acte("ACTE 3 · Filtres JETÉS / erreurs (avec motif)")
    for e in result.get("erreurs", []):
        print(f"  ✗ {e}")
    if not result.get("erreurs"):
        print("  (aucun rejet)")

    _acte("ACTE 4 · Compilation + exécution")
    cypher = _extraire_cypher(journal)
    print(f"  Cypher : {cypher or '(non exécuté)'}")
    print(f"  status : {result['status']}   total : {result.get('total')}   "
          f"rows : {len(result.get('rows', []))}   durée : {duree}s")
    for r in _rows_extrait(result.get("rows", [])):
        print(f"    · {r}")

    _acte("Journal brut capturé")
    for ligne in journal:
        print(f"  {ligne}")

    return {
        "voie": "query-open", "question": question, "duree_s": duree,
        "status": result["status"],
        "spec_llm": spec.model_dump() if spec else None,
        "filtres_retenus": result.get("filtres", []),
        "rejets": result.get("erreurs", []),
        "cypher": cypher,
        "total": result.get("total"),
        "rows_extrait": _rows_extrait(result.get("rows", [])),
        "journal": journal,
    }


# ─── VOIE stats ──────────────────────────────────────────────────────────────

def tracer_stats(question: str, ollama, neo4j, capture: _Capture) -> dict:
    from aggregation_incident_v2 import run_aggregation

    _titre("VOIE /stats — agrégation à vocabulaire FERMÉ (AggregationSpec)")
    capture.vider()
    t0 = time.time()
    result = run_aggregation(question=question, ollama=ollama, neo4j=neo4j)
    duree = round(time.time() - t0, 1)
    journal = capture.vider()

    spec = result.get("spec")
    _acte("ACTE 1 · Traduction LLM → spec")
    if spec is None:
        print("  spec : AUCUNE (non reconnue comme agrégation, ou dégénérée)")
    else:
        d = spec.model_dump()
        print(f"  group_by : {d.pop('group_by')}")
        for k, v in d.items():
            if k.startswith("f_") and v is not None:
                print(f"  filtre LLM : {k} = {v!r}")

    _acte("ACTE 2/3 · Pas de garde-fou lexical sur cette voie")
    print("  ⚠ vocabulaire fermé : une contrainte hors formulaire est IGNORÉE")
    print("    silencieusement (limitation documentée de la voie /stats).")

    _acte("ACTE 4 · Compilation + exécution")
    print(f"  Cypher : {result.get('cypher') or '(non exécuté)'}")
    print(f"  status : {result['status']}   total : {result.get('total')}   "
          f"durée : {duree}s")
    for r in _rows_extrait(result.get("rows", []), n=5):
        print(f"    · {r}")

    _acte("Journal brut capturé")
    for ligne in journal:
        print(f"  {ligne}")

    return {
        "voie": "stats", "question": question, "duree_s": duree,
        "status": result["status"],
        "spec_llm": spec.model_dump() if spec else None,
        "filtres_appliques": result.get("filters_applied"),
        "cypher": result.get("cypher"),
        "total": result.get("total"),
        "rows_extrait": _rows_extrait(result.get("rows", []), n=5),
        "journal": journal,
    }


# ─── VOIE actions ────────────────────────────────────────────────────────────

def tracer_actions(question: str, ollama, neo4j, capture: _Capture) -> dict:
    from action_lookup_incident_v2 import run_action_lookup

    _titre("VOIE /actions — actions correctives/préventives (ActionSpec)")
    capture.vider()
    t0 = time.time()
    result = run_action_lookup(question=question, ollama=ollama, neo4j=neo4j)
    duree = round(time.time() - t0, 1)
    journal = capture.vider()

    spec = result.get("spec")
    _acte("ACTE 1 · Traduction LLM → spec")
    if spec is None:
        print("  spec : AUCUNE (non reconnue comme question d'actions)")
    else:
        d = spec.model_dump()
        print(f"  question_type : {d.pop('question_type')}")
        print(f"  type_action   : {d.pop('type_action')}")
        for k, v in d.items():
            if k.startswith("f_") and v is not None:
                print(f"  filtre LLM : {k} = {v!r}")

    _acte("ACTE 2/3 · Garde-fou de cette voie")
    print("  ✓ refus de lister sans AUCUN critère (status no_filters)")
    print("  ✓ bascule automatique 'action à chaud' → forme incident-centrée")

    _acte("ACTE 4 · Compilation + exécution")
    cypher = _extraire_cypher(journal)
    print(f"  Cypher : {cypher or '(non exécuté)'}")
    print(f"  status : {result['status']}   total : {result.get('total')}   "
          f"rows : {len(result.get('rows', []))}   durée : {duree}s")
    for r in _rows_extrait(result.get("rows", [])):
        print(f"    · {r}")

    _acte("Journal brut capturé")
    for ligne in journal:
        print(f"  {ligne}")

    return {
        "voie": "actions", "question": question, "duree_s": duree,
        "status": result["status"],
        "spec_llm": spec.model_dump() if spec else None,
        "cypher": cypher,
        "total": result.get("total"),
        "rows_extrait": _rows_extrait(result.get("rows", [])),
        "journal": journal,
    }


# ─── Comparaison de modèles (moteur unifié de production) ────────────────────

def comparer_modeles(question: str, modeles: list[str], ollama, neo4j,
                     capture: _Capture) -> list[dict]:
    """Pose la MÊME question au moteur unifié avec chaque modèle, côte à côte."""
    import query_engine_incident_v2 as qe

    _titre(f"COMPARAISON /query (moteur unifié) — {' vs '.join(modeles)}")
    bilans = []
    for modele in modeles:
        qe.LLM_MODEL = modele          # surcharge à chaud, ce process seulement
        capture.vider()
        t0 = time.time()
        result = qe.run_unified_query(question=question, ollama=ollama, neo4j=neo4j)
        duree = round(time.time() - t0, 1)
        journal = capture.vider()

        spec_brute = _extraire_spec_brute(journal)
        spec_finale = result.get("spec_interpretee")
        brut = result.get("resultat_brut")
        resume = f"{len(brut)} lignes" if isinstance(brut, list) else repr(brut)

        _acte(f"MODÈLE {modele}")
        print("  SPEC LLM BRUTE : " + (json.dumps(_spec_non_nulle(spec_brute),
              ensure_ascii=False) if spec_brute else "(aucune)"))
        if spec_brute and spec_finale:
            diffs = [f"{k}: {spec_brute.get(k)!r}→{spec_finale.get(k)!r}"
                     for k in spec_finale if spec_brute.get(k) != spec_finale.get(k)]
            print("  POST-PROCESS   : " + ("; ".join(diffs) or "(aucune correction)"))
        print(f"  RÉSULTAT       : status={result['status']}  {resume}  ({duree}s)")

        bilans.append({
            "modele": modele, "duree_s": duree, "status": result["status"],
            "spec_llm_brute": spec_brute, "spec_finale": spec_finale,
            "resultat": brut if not isinstance(brut, list) else len(brut),
            "cypher": result.get("cypher_execute"),
        })

    _acte("VERDICT")
    cles = [(b["status"], json.dumps(b["spec_finale"], sort_keys=True, default=str),
             b["resultat"]) for b in bilans]
    if all(c == cles[0] for c in cles):
        print("  ✓ IDENTIQUES : même spec finale, même statut, même résultat.")
    else:
        print("  ✗ DIFFÉRENTS — comparez les blocs ci-dessus :")
        for b in bilans:
            print(f"    {b['modele']:<14} status={b['status']}  "
                  f"resultat={b['resultat']}  ({b['duree_s']}s)")
    return bilans


# ─── Point d'entrée ──────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description="Trace une question, acte par acte")
    ap.add_argument("question")
    ap.add_argument("--voie", choices=["query", "query-open", "stats", "actions", "all"],
                    default="all")
    ap.add_argument("--comparer", default=None, metavar="M1,M2",
                    help="compare des modèles sur le moteur unifié "
                         "(ex : qwen2.5:7b,qwen2.5:14b)")
    ap.add_argument("--json", type=Path, default=None,
                    help="écrit aussi le rapport structuré dans ce fichier")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger().handlers.clear()      # pas de doublon console
    capture = _Capture()
    logging.getLogger().addHandler(capture)

    neo4j = Neo4jClient(
        uri=os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        user=os.environ.get("NEO4J_USER", "neo4j"),
        password=os.environ.get("NEO4J_PASSWORD"),
    )
    ollama = OllamaClient(url=os.environ.get("OLLAMA_URL", "http://localhost:11434"))

    voies = {"query": tracer_query, "query-open": tracer_query_open,
             "stats": tracer_stats, "actions": tracer_actions}
    a_tracer = ["query", "stats", "actions"] if args.voie == "all" else [args.voie]

    print(f"\nQUESTION : « {args.question} »")

    if args.comparer:
        modeles = [m.strip() for m in args.comparer.split(",") if m.strip()]
        rapports = comparer_modeles(args.question, modeles, ollama, neo4j, capture)
    else:
        rapports = []
        for nom in a_tracer:
            try:
                rapports.append(voies[nom](args.question, ollama, neo4j, capture))
            except Exception as e:
                print(f"\n✗ voie {nom} : ÉCHEC — {e}")
                rapports.append({"voie": nom, "question": args.question,
                                 "status": "exception", "erreur": str(e)})

    if args.json:
        args.json.write_text(
            json.dumps(rapports, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        print(f"\nRapport JSON écrit : {args.json}")

    neo4j.close()
    ollama.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
