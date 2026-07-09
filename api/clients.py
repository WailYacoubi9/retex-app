"""
Clients pour Neo4j, Qdrant, Ollama.
Wrappers minimalistes qui exposent juste ce dont on a besoin.

API alignee sur les modules v2 (loader.py, llm_enricher.py, retrieval.py).
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import httpx
from neo4j import GraphDatabase, Driver
from qdrant_client import QdrantClient as _QdrantClient
from qdrant_client.http.models import (
    Distance, VectorParams, PointStruct, FieldCondition, Filter, MatchValue,
)

logger = logging.getLogger(__name__)

INCIDENT_CHUNKS_COLLECTION = "incident_chunks"
EMBEDDING_DIM = 1024  # bge-m3


class Neo4jClient:
    """Wrapper Neo4j. Ouvre une connexion, expose sessions, ferme proprement."""

    def __init__(self, uri: str, user: str, password: str):
        self._driver: Driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self._driver.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def session(self):
        return self._driver.session()

    def run(self, cypher: str, **params) -> list[dict]:
        """Execute une requete Cypher et retourne la liste des records."""
        with self.session() as s:
            result = s.run(cypher, **params)
            return [dict(r) for r in result]

    def execute(self, cypher: str, **params) -> None:
        """Execute une requete Cypher en ecriture, sans retour."""
        with self.session() as s:
            s.run(cypher, **params).consume()

    def count_incidents(self) -> int:
        result = self.run("MATCH (i:Incident) RETURN count(i) AS c")
        return result[0]["c"] if result else 0

    def run_read(self, cypher: str, **params) -> list[dict]:
        """Execute une requête Cypher en session READ-ONLY."""
        with self._driver.session(default_access_mode="READ") as s:
            result = s.run(cypher, **params)
            return [dict(r) for r in result]


class QdrantWrapper:
    """Wrapper Qdrant minimaliste."""

    def __init__(self, url: str):
        self._client = _QdrantClient(url=url, check_compatibility=False)

    def ensure_collection(self):
        """Cree la collection si elle n'existe pas. Idempotent."""
        existing = {c.name for c in self._client.get_collections().collections}
        if INCIDENT_CHUNKS_COLLECTION in existing:
            return
        logger.info("Creating Qdrant collection: %s", INCIDENT_CHUNKS_COLLECTION)
        self._client.create_collection(
            collection_name=INCIDENT_CHUNKS_COLLECTION,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
        for field_name in ("incident_id", "field_canonical", "source_module", "is_test_data"):
            self._client.create_payload_index(
                INCIDENT_CHUNKS_COLLECTION, field_name, field_schema="keyword",
            )

    def upsert(self, point_id: str, vector: list[float], payload: dict) -> None:
        """Upsert d'un seul point."""
        point = PointStruct(id=point_id, vector=vector, payload=payload)
        self.upsert_points([point])

    def upsert_points(self, points: list[PointStruct]) -> None:
        """Upsert d'un batch de points."""
        if not points:
            return
        self._client.upsert(
            collection_name=INCIDENT_CHUNKS_COLLECTION,
            points=points,
            wait=True,
        )

    def search(
        self,
        vector: list[float],
        top_k: int = 5,
        exclude_test_data: bool = True,
        source_module: Optional[str] = None,
    ) -> list[dict]:
        """Recherche les top_k chunks les plus similaires au vecteur fourni.

        Utilisation : appelee par la couche RAG. Retourne une liste de
        dicts avec les payloads des chunks trouves + score de similarite.
        Filtre par defaut les chunks marques is_test_data=true.
        Si source_module est fourni, ne retourne que les chunks de ce module.
        """
        must: list = []
        must_not: list = []

        if exclude_test_data:
            must_not.append(FieldCondition(key="is_test_data", match=MatchValue(value=True)))
        if source_module:
            must.append(FieldCondition(key="source_module", match=MatchValue(value=source_module)))

        query_filter = Filter(must=must, must_not=must_not) if (must or must_not) else None

        response = self._client.query_points(
            collection_name=INCIDENT_CHUNKS_COLLECTION,
            query=vector,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )

        return [
            {
                "score": p.score,
                "payload": p.payload,
                "chunk_id": p.id,
            }
            for p in response.points
        ]

    def count_chunks(self) -> int:
        info = self._client.get_collection(INCIDENT_CHUNKS_COLLECTION)
        return info.points_count or 0


class OllamaClient:
    """Wrapper Ollama pour embeddings et generation."""

    def __init__(self, url: str, embedding_model: str = "bge-m3",
                 llm_model: str = "llama3.1:8b", timeout: float = 120.0):
        self._url = url.rstrip("/")
        self._embedding_model = embedding_model
        self._llm_model = llm_model
        self._client = httpx.Client(timeout=timeout)

    def close(self):
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def embed(self, text: str) -> list[float]:
        """Produit un embedding pour un texte unique."""
        r = self._client.post(
            f"{self._url}/api/embed",
            json={"model": self._embedding_model, "input": text},
        )
        r.raise_for_status()
        data = r.json()
        embeddings = data.get("embeddings", [])
        if not embeddings:
            raise RuntimeError(f"No embedding returned for model {self._embedding_model}")
        return embeddings[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Batch embedding."""
        if not texts:
            return []
        r = self._client.post(
            f"{self._url}/api/embed",
            json={"model": self._embedding_model, "input": texts},
        )
        r.raise_for_status()
        return r.json().get("embeddings", [])

    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        json_format: bool = False,
        temperature: float = 0.2,
        num_ctx: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> str:
        """Generation de texte via le LLM."""
        options: dict[str, Any] = {"temperature": temperature}
        if num_ctx:
            options["num_ctx"] = num_ctx
        payload: dict[str, Any] = {
            "model": model or self._llm_model,
            "prompt": prompt,
            "stream": False,
            "options": options,
        }
        if json_format:
            payload["format"] = "json"

        kwargs: dict[str, Any] = {}
        if timeout is not None:
            kwargs["timeout"] = timeout
        r = self._client.post(f"{self._url}/api/generate", json=payload, **kwargs)
        r.raise_for_status()
        return r.json().get("response", "").strip()

    def generate_structured(
        self,
        prompt: str,
        schema: dict,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> str:
        """Génération avec structured output (JSON schema en format)."""
        payload: dict[str, Any] = {
            "model": model or self._llm_model,
            "prompt": prompt,
            "stream": False,
            "format": schema,
            "options": {"temperature": 0},
        }
        kwargs: dict[str, Any] = {}
        if timeout is not None:
            kwargs["timeout"] = timeout
        r = self._client.post(f"{self._url}/api/generate", json=payload, **kwargs)
        r.raise_for_status()
        return r.json().get("response", "").strip()
