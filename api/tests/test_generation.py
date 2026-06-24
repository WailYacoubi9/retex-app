"""
Test isolé de la generation RAG.
"""
import logging
logging.basicConfig(level=logging.INFO)

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

from clients import Neo4jClient, OllamaClient, QdrantWrapper

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
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
        uri=NEO4J_URI,
        user=NEO4J_USER,
        password=NEO4J_PASSWORD,
    ) as neo4j, OllamaClient(url=OLLAMA_URL) as ollama:

        qdrant = QdrantWrapper(url=QDRANT_URL)

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