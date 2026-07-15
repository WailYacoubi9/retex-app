"""
Microscope à actes — trace une question à travers les 3 voies à formulaire.

Pour une question donnée, montre EN DÉTAIL ce que chaque étage a produit :
  ACTE 1 — la spec brute remplie par le LLM (traduction)
  ACTE 2 — filtres retenus après validation + garde-fous
  ACTE 3 — filtres JETÉS, avec le motif exact de chaque rejet
  ACTE 4 — le Cypher compilé + paramètres, et les chiffres exécutés

Le journal (logging) des modules est capturé et affiché : on y voit la
sortie brute du LLM avant toute correction, et la requête Cypher exacte.

Usage (dans le conteneur API, où vivent les modules et les variables d'env) :
  docker exec -it ia-api python tests/trace_question.py "Les incidents graves"
  docker exec -it ia-api python tests/trace_question.py "Combien d'actions en cours ?" --voie actions
  docker exec -it ia-api python tests/trace_question.py "Répartition par sévérité" --voie all --json /app/tests/trace.json
"""
from __future__ import annotations

import argparse
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
    """Retrouve la ligne 'Cypher: ...' dans le journal capturé."""
    for ligne in journal:
        if "Cypher:" in ligne:
            return ligne.split("Cypher:", 1)[1].strip()
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


# ─── Les trois voies ─────────────────────────────────────────────────────────

def tracer_query(question: str, ollama, neo4j, capture: _Capture) -> dict:
    from query_engine_incident_v2 import run_query

    _titre("VOIE /query — moteur générique (QuerySpec)")
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
        "voie": "query", "question": question, "duree_s": duree,
        "status": result["status"],
        "spec_llm": spec.model_dump() if spec else None,
        "filtres_retenus": result.get("filtres", []),
        "rejets": result.get("erreurs", []),
        "cypher": cypher,
        "total": result.get("total"),
        "rows_extrait": _rows_extrait(result.get("rows", [])),
        "journal": journal,
    }


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


# ─── Point d'entrée ──────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description="Trace une question, acte par acte")
    ap.add_argument("question")
    ap.add_argument("--voie", choices=["query", "stats", "actions", "all"],
                    default="all")
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

    voies = {"query": tracer_query, "stats": tracer_stats, "actions": tracer_actions}
    a_tracer = list(voies) if args.voie == "all" else [args.voie]

    print(f"\nQUESTION : « {args.question} »")
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
