import logging
logging.basicConfig(level=logging.INFO)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clients import Neo4jClient, OllamaClient, QdrantWrapper
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
    with Neo4jClient("bolt://localhost:7687", "neo4j", "retex_dev_pwd") as neo4j, \
         OllamaClient(url="http://localhost:11434") as ollama:

        qdrant = QdrantWrapper(url="http://localhost:6333")

        test("Quelles sont les consignes de securite liees aux drones ?", neo4j, qdrant, ollama)
        test("Quels risques sont lies aux operations par faible visibilite ?", neo4j, qdrant, ollama)
        test("Quelles IS concernent les helicopteres ?", neo4j, qdrant, ollama)
        test("Comment faire une bonne pizza ?", neo4j, qdrant, ollama)


if __name__ == "__main__":
    main()
