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
from retrieval_info_securite import retrieve_info_securite
from generation_info_securite import generate_answer_is


def test(question, neo4j, qdrant, ollama):
    print(f"\n{'='*70}")
    print(f"QUESTION : {question}")
    print('='*70)

    retrieval = retrieve_info_securite(question, ollama, qdrant, neo4j, top_k=5)

    if retrieval.below_threshold:
        print("AUCUN RESULTAT PERTINENT (sous seuil 0.45)")
        return

    print(f"Direct: {retrieval.n_direct} | Expanded: {retrieval.n_expanded}")
    for item in retrieval.items:
        marker = "EXPANDED" if item.is_expanded else "DIRECT"
        print(f"  [{marker}] {item.is_number} - score={item.best_score:.3f} - champs={item.matched_fields}")

    generation = generate_answer_is(question, retrieval, ollama)
    print(f"\n--- REPONSE ({generation.duration_ms}ms) ---")
    print(generation.answer)


def main():
    with Neo4jClient(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD) as neo4j, \
         OllamaClient(url=OLLAMA_URL) as ollama:

        qdrant = QdrantWrapper(url=QDRANT_URL)

        test("Quelles sont les consignes de securite liees aux drones ?", neo4j, qdrant, ollama)
        test("Quels risques sont lies aux operations par faible visibilite ?", neo4j, qdrant, ollama)
        test("Quelles IS concernent les helicopteres ?", neo4j, qdrant, ollama)
        test("Comment faire une bonne pizza ?", neo4j, qdrant, ollama)


if __name__ == "__main__":
    main()
