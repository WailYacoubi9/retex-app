"""
Test isolé de la generation RAG.
"""
import logging
logging.basicConfig(level=logging.INFO)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clients import Neo4jClient, OllamaClient, QdrantWrapper
from retrieval import retrieve
from generation import generate_answer


def test_full_rag(question, neo4j, qdrant, ollama):
    print(f"\n{'='*70}")
    print(f"QUESTION : {question}")
    print('='*70)

    # Etape 1 : retrieval
    retrieval = retrieve(
        question=question,
        ollama=ollama,
        qdrant=qdrant,
        neo4j=neo4j,
        top_k=5,
    )

    # Etape 2 : generation
    generation = generate_answer(
        question=question,
        retrieval_result=retrieval,
        ollama=ollama,
    )

    print(f"\n--- REPONSE LLM ---")
    print(generation.answer)
    print(f"\n--- METADONNEES ---")
    print(f"Modele : {generation.model_used}")
    print(f"Duree  : {generation.duration_ms} ms")
    print(f"Sources : {len(retrieval.incidents)} incidents")
    for inc in retrieval.incidents:
        marker = "EXPANDED" if inc.is_expanded else "DIRECT"
        print(f"  [{marker}] {inc.numero_fe} - {inc.titre[:60] if inc.titre else '?'}")


def main():
    with Neo4jClient(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="retex_dev_pwd",
    ) as neo4j, OllamaClient(url="http://localhost:11434") as ollama:

        qdrant = QdrantWrapper(url="http://localhost:6333")

        # Test 1 : question precise
        test_full_rag(
            "Quels incidents impliquent un mauvais positionnement d'avion ou d'engin ?",
            neo4j, qdrant, ollama,
        )

        # Test 2 : question impliquant des passagers difficiles
        test_full_rag(
            "Y a-t-il eu des incidents avec des passagers agressifs ou indisciplines ?",
            neo4j, qdrant, ollama,
        )

        # Test 3 : question hors sujet (devrait refuser)
        test_full_rag(
            "Quelle est la meilleure recette de couscous ?",
            neo4j, qdrant, ollama,
        )


if __name__ == "__main__":
    main()