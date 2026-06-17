"""
Application FastAPI principale du projet RETEX.

Expose 3 endpoints :
  - POST /ask    : poser une question RAG, recevoir une reponse + sources
  - GET  /health : verifier que l'API et ses dependances tournent
  - GET  /stats  : obtenir les compteurs des bases (incidents, chunks)

Les connexions aux services backend (Neo4j, Qdrant, Ollama) sont
ouvertes une seule fois au demarrage via le lifespan FastAPI et
fermees proprement a l'arret.

Lancement local :
  uvicorn main:app --reload --port 8000

Lancement Docker (via docker-compose) :
  docker compose up api

Documentation interactive :
  http://localhost:8000/docs
"""
from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware



from clients import (
    INCIDENT_CHUNKS_COLLECTION,
    EMBEDDING_DIM,
    Neo4jClient,
    OllamaClient,
    QdrantWrapper,
)
from schemas import (
    AskRequest,
    AskResponse,
    AskResponseMetadata,
    HealthResponse,
    Neo4jStats,
    QdrantStats,
    ServiceStatus,
    SourceIncident,
    StatsResponse,
    AskResponseIS,
    SourceInfoSecurite,
    AskResponseTickets,
    SourceTicket,
)
from retrieval_info_securite import retrieve_info_securite
from generation_info_securite import generate_answer_is
from retrieval_tickets import retrieve_tickets
from generation_tickets import generate_answer_tickets
from retrieval import retrieve
from generation import build_sources, generate_answer


# =====================================================================
# CONFIGURATION
# =====================================================================

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "retex_dev_pwd")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("api")


# =====================================================================
# LIFESPAN - ouverture/fermeture des connexions
# =====================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gere le cycle de vie de l'application FastAPI.

    Utilisation : ouvre les connexions aux services au demarrage, les
    stocke dans app.state pour les rendre accessibles aux routes, et
    les ferme a l'arret. Pattern standard FastAPI pour les ressources
    couteuses a initialiser.
    """
    logger.info("=== Demarrage de l'API RETEX ===")
    logger.info("Neo4j  : %s", NEO4J_URI)
    logger.info("Qdrant : %s", QDRANT_URL)
    logger.info("Ollama : %s", OLLAMA_URL)

    # Ouverture des connexions persistantes
    app.state.neo4j = Neo4jClient(
        uri=NEO4J_URI,
        user=NEO4J_USER,
        password=NEO4J_PASSWORD,
    )
    app.state.qdrant = QdrantWrapper(url=QDRANT_URL)
    app.state.ollama = OllamaClient(url=OLLAMA_URL)

    logger.info("Connexions ouvertes, l'API est prete")

    yield  # L'application sert les requetes ici

    logger.info("=== Arret de l'API RETEX ===")
    app.state.neo4j.close()
    app.state.ollama.close()
    logger.info("Connexions fermees proprement")


# =====================================================================
# APPLICATION FASTAPI
# =====================================================================

app = FastAPI(
    title="RETEX Assistant API",
    description=(
        "Assistant intelligent base sur LLM et RAG hybride (Neo4j + Qdrant) "
        "pour l'analyse d'incidents de securite aeronautique intra'know."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS ouvert pour la phase POC (a restreindre en production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =====================================================================
# ENDPOINT : POST /ask
# =====================================================================

@app.post(
    "/ask",
    response_model=AskResponse,
    summary="Poser une question RAG",
    description=(
        "Recoit une question en langage naturel, fait une recherche "
        "hybride (Qdrant + Neo4j) et genere une reponse avec citations."
    ),
)
async def ask(request: Request, body: AskRequest) -> AskResponse:
    """Endpoint principal du RAG.

    Utilisation : appele par le frontend (ou curl) pour poser une
    question. Orchestre les couches retrieval et generation, puis
    formate la reponse selon le schema Pydantic.
    """
    start = time.time()

    try:
        # Etape 1 : retrieval (Qdrant + Neo4j)
        retrieval_result = retrieve(
            question=body.question,
            ollama=request.app.state.ollama,
            qdrant=request.app.state.qdrant,
            neo4j=request.app.state.neo4j,
            top_k=body.top_k,
            include_test_data=body.include_test_data,
        )

        # Etape 2 : generation (prompt + LLM)
        generation_result = generate_answer(
            question=body.question,
            retrieval_result=retrieval_result,
            ollama=request.app.state.ollama,
        )

        # Etape 3 : formatage des sources pour la reponse API
        raw_sources = build_sources(retrieval_result.incidents)
        sources = [SourceIncident(**s) for s in raw_sources]

        # Etape 4 : construction de la reponse finale
        total_duration_ms = int((time.time() - start) * 1000)

        return AskResponse(
            answer=generation_result.answer,
            sources=sources,
            metadata=AskResponseMetadata(
                duration_ms=total_duration_ms,
                n_chunks_retrieved=retrieval_result.n_chunks_retrieved,
                n_incidents_unique=retrieval_result.n_incidents_direct,
                n_incidents_expanded=retrieval_result.n_incidents_expanded,
                model_used=generation_result.model_used,
            ),
        )

    except Exception as e:
        logger.exception("Erreur lors du traitement de /ask")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur interne du serveur : {str(e)}",
        )
    
@app.post("/ask/tickets", response_model=AskResponseTickets)
async def ask_tickets(request: Request, body: AskRequest) -> AskResponseTickets:
    start = time.time()
    try:
        retrieval = retrieve_tickets(
            question=body.question,
            ollama=request.app.state.ollama,
            qdrant=request.app.state.qdrant,
            neo4j=request.app.state.neo4j,
            top_k=body.top_k,
        )
        generation = generate_answer_tickets(
            question=body.question,
            retrieval_result=retrieval,
            ollama=request.app.state.ollama,
        )
        sources = [
            SourceTicket(
                ticket_id=item.ticket_id,
                numero_fe=item.numero_fe,
                titre=item.titre,
                type_nc=item.type_nc,
                importance=item.importance,
                etat=item.etat,
                etape_label=item.etape_label,
                site_application=item.site_application,
                projet_nom=item.projet_nom,
                client=item.client,
                structure=item.structure,
                urgence=item.urgence,
                individu=item.individu,
                date_nc=item.date_nc,
                llm_resume=item.llm_resume,
                llm_domaine_technique=item.llm_domaine_technique,
                best_score=item.best_score,
                matched_fields=item.matched_fields,
                is_expanded=item.is_expanded,
            )
            for item in retrieval.items
        ]
        return AskResponseTickets(
            answer=generation.answer,
            sources=sources,
            metadata=AskResponseMetadata(
                duration_ms=int((time.time() - start) * 1000),
                n_chunks_retrieved=retrieval.n_chunks_retrieved,
                n_incidents_unique=retrieval.n_direct,
                n_incidents_expanded=retrieval.n_expanded,
                model_used=generation.model_used,
            ),
        )
    except Exception as e:
        logger.exception("Erreur lors du traitement de /ask/tickets")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask/info-securite", response_model=AskResponseIS)
async def ask_info_securite(request: Request, body: AskRequest) -> AskResponseIS:
    start = time.time()
    try:
        retrieval = retrieve_info_securite(
            question=body.question,
            ollama=request.app.state.ollama,
            qdrant=request.app.state.qdrant,
            neo4j=request.app.state.neo4j,
            top_k=body.top_k,
        )
        generation = generate_answer_is(
            question=body.question,
            retrieval_result=retrieval,
            ollama=request.app.state.ollama,
        )
        sources = [
            SourceInfoSecurite(
                info_securite_id=item.info_securite_id,
                is_number=item.is_number,
                annee=item.annee,
                titre=item.titre,
                llm_resume=item.llm_resume,
                operateurs_concernes=item.operateurs_concernes,
                best_score=item.best_score,
                matched_fields=item.matched_fields,
            )
            for item in retrieval.items
        ]
        return AskResponseIS(
            answer=generation.answer,
            sources=sources,
            metadata=AskResponseMetadata(
                duration_ms=int((time.time() - start) * 1000),
                n_chunks_retrieved=retrieval.n_chunks_retrieved,
                n_incidents_unique=retrieval.n_direct,
                n_incidents_expanded=retrieval.n_expanded,
                model_used=generation.model_used,
            ),
        )
    except Exception as e:
        logger.exception("Erreur lors du traitement de /ask/info-securite")
        raise HTTPException(status_code=500, detail=str(e))



# =====================================================================
# ENDPOINT : GET /health
# =====================================================================

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Verifier l'etat de l'API et de ses services",
)
async def health(request: Request) -> HealthResponse:
    """Endpoint de monitoring.

    Utilisation : appele par des outils de monitoring ou par
    l'utilisateur pour verifier rapidement que tout va bien. Teste
    chaque service backend individuellement.
    """
    services: list[ServiceStatus] = []

    # Test Neo4j
    try:
        request.app.state.neo4j.run("RETURN 1 AS test")
        services.append(ServiceStatus(name="neo4j", status="up"))
    except Exception as e:
        services.append(ServiceStatus(
            name="neo4j", status="down", detail=str(e)[:200],
        ))

    # Test Qdrant
    try:
        request.app.state.qdrant.count_chunks()
        services.append(ServiceStatus(name="qdrant", status="up"))
    except Exception as e:
        services.append(ServiceStatus(
            name="qdrant", status="down", detail=str(e)[:200],
        ))

    # Test Ollama (avec un embedding minimal)
    try:
        request.app.state.ollama.embed("ping")
        services.append(ServiceStatus(name="ollama", status="up"))
    except Exception as e:
        services.append(ServiceStatus(
            name="ollama", status="down", detail=str(e)[:200],
        ))

    # Statut global
    down_count = sum(1 for s in services if s.status == "down")
    if down_count == 0:
        global_status = "healthy"
    elif down_count < len(services):
        global_status = "degraded"
    else:
        global_status = "unhealthy"

    return HealthResponse(
        status=global_status,
        services=services,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


# =====================================================================
# ENDPOINT : GET /stats
# =====================================================================

@app.get(
    "/stats",
    response_model=StatsResponse,
    summary="Obtenir les statistiques des bases de donnees",
)
async def stats(request: Request) -> StatsResponse:
    """Endpoint d'observabilite.

    Utilisation : donne une vue d'ensemble du contenu des bases.
    Utile pour la demo, pour valider une ingestion, et pour le
    suivi du volume de donnees au fil du temps.
    """
    neo4j = request.app.state.neo4j
    qdrant = request.app.state.qdrant

    try:
        # Comptage Neo4j par label
        incidents_count = neo4j.run(
            "MATCH (i:Incident) RETURN count(i) AS c"
        )[0]["c"]

        enriched_count = neo4j.run(
            "MATCH (i:Incident) WHERE i.resume_llm IS NOT NULL RETURN count(i) AS c"
        )[0]["c"]

        tickets_count = neo4j.run(
            "MATCH (t:Ticket) RETURN count(t) AS c"
        )[0]["c"]

        tickets_enriched_count = neo4j.run(
            "MATCH (t:Ticket) WHERE t.llm_resume IS NOT NULL RETURN count(t) AS c"
        )[0]["c"]

        info_securite_count = neo4j.run(
            "MATCH (i:InfoSecurite) RETURN count(i) AS c"
        )[0]["c"]

        societes_count = neo4j.run(
            "MATCH (s:Societe) RETURN count(s) AS c"
        )[0]["c"]

        personnes_count = neo4j.run(
            "MATCH (p:Personne) RETURN count(p) AS c"
        )[0]["c"]

        referentiels_count = neo4j.run(
            "MATCH (r:Referentiel) RETURN count(r) AS c"
        )[0]["c"]

        # Comptage Qdrant
        points_count = qdrant.count_chunks()

        return StatsResponse(
            neo4j=Neo4jStats(
                incidents=incidents_count,
                incidents_enriched=enriched_count,
                tickets=tickets_count,
                tickets_enriched=tickets_enriched_count,
                info_securite=info_securite_count,
                societes=societes_count,
                personnes=personnes_count,
                referentiels=referentiels_count,
            ),
            qdrant=QdrantStats(
                collection=INCIDENT_CHUNKS_COLLECTION,
                points_count=points_count,
                vector_dimension=EMBEDDING_DIM,
            ),
        )

    except Exception as e:
        logger.exception("Erreur lors du calcul des stats")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur recuperation stats : {str(e)}",
        )


# =====================================================================
# ENDPOINT RACINE
# =====================================================================

@app.get("/", summary="Point d'entree de l'API")
async def root():
    """Page d'accueil minimale qui oriente vers la doc."""
    return {
        "name": "RETEX Assistant API",
        "version": "1.0.0",
        "documentation": "/docs",
        "endpoints": ["/ask", "/health", "/stats"],
    }
