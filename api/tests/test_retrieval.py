"""
Test isolé du module de retrieval.
"""
import logging
logging.basicConfig(level=logging.INFO)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clients import Neo4jClient, OllamaClient, QdrantWrapper
from retrieval import retrieve


def test_question(question, neo4j, qdrant, ollama):
    print(f"\n{'='*70}")
    print(f"Question : {question}")
    print('='*70)

    result = retrieve(
        question=question,
        ollama=ollama,
        qdrant=qdrant,
        neo4j=neo4j,
        top_k=5,
    )

    if result.below_threshold:
        print("AUCUN RESULTAT PERTINENT (tous chunks sous seuil 0.45)")
        return

    print(f"Chunks recuperes  : {result.n_chunks_retrieved}")
    print(f"Incidents directs : {result.n_incidents_direct}")
    print(f"Incidents expanded: {result.n_incidents_expanded}")
    print()

    for inc in result.incidents:
        marker = "EXPANDED" if inc.is_expanded else "DIRECT"
        print(f"[{marker}] {inc.numero_fe or '?'} - {inc.titre or '?'}")
        print(f"  Score: {inc.best_score:.3f}")
        print(f"  Champs matches: {inc.matched_fields}")
        if inc.resume_llm:
            print(f"  Resume: {inc.resume_llm[:120]}...")
        print(f"  Relations: "
              f"{len(inc.personnes)} personnes, "
              f"{len(inc.societes)} societes, "
              f"{len(inc.referentiels)} referentiels")
        print()


def main():
    with Neo4jClient(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="retex_dev_pwd",
    ) as neo4j, OllamaClient(url="http://localhost:11434") as ollama:

        qdrant = QdrantWrapper(url="http://localhost:6333")

        # Question 1 : sémantique précise (devrait retrouver un incident très ressemblant)
        test_question(
            "Comment a réagi l'agent suite au mauvais positionnement HELP ?",
            neo4j, qdrant, ollama,
        )

        # Question 2 : sémantique large (devrait ramener plusieurs incidents)
        test_question(
            "Quels incidents impliquent un problème de communication entre agents ?",
            neo4j, qdrant, ollama,
        )

        # Question 3 : volontairement hors sujet (devrait déclencher below_threshold)
        test_question(
            "Comment cuisiner des pâtes carbonara ?",
            neo4j, qdrant, ollama,
        )


if __name__ == "__main__":
    main()