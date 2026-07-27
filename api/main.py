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

import json
import logging
import os
import re
import threading
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
    RerankerClient,
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
    AskIncidentV2Response,
    SourceIncidentV2,
    AggregationResponse,
    EntityLookupResponse,
    EntityMatchResponse,
    ActionLookupResponse,
    ActionResult,
    GenericQueryResponse,
    ActionRecommandeeResponse,
    RecommandationResponse,
    StructuredListResponse,
)
from retrieval_info_securite import retrieve_info_securite
from generation_info_securite import generate_answer_is
from retrieval_tickets import retrieve_tickets
from generation_tickets import generate_answer_tickets
from retrieval import retrieve
from generation import build_sources, generate_answer
from retrieval_incident_v2 import retrieve_incident_v2, MIN_SCORE
from generation_incident_v2 import generate_answer_incident_v2
from aggregation_incident_v2 import run_aggregation
from generation_aggregation import phrase_result
from entity_lookup_incident_v2 import fetch_entity
from generation_entity_lookup import phrase_entity_result
from action_lookup_incident_v2 import run_action_lookup
from generation_action_lookup import phrase_action_result
from structured_list_incident_v2 import run_list_query
from generation_list_incident_v2 import phrase_list_result
from query_engine_incident_v2 import run_query, run_unified_query
from generation_query import phrase_query_result, phrase_unified_result
from field_catalog import champ_meta, champs_labellises
from recommendation_incident_v2 import run_recommendation
from synthese_incident_v2 import run_synthese
from generation_recommandation import phrase_recommendation
from routeur_incident_v2 import router
from analyste_incident_v2 import run_analyste
import confidentialite

# Sujet « actions » d'un comptage (garde-fou routage) : « combien … d'actions »,
# « nombre d'actions » — ne matche PAS « combien d'incidents … des actions ».
_SUJET_ACTIONS_RE = re.compile(
    r"\b(?:combien|nombre)\b.{0,15}?\bd['’]\s*actions?\b", re.IGNORECASE | re.DOTALL)

# Classement/palmarès de PERSONNES (« quel responsable/agent a le plus… ») :
# refus déterministe pré-routeur (culture juste). « quel service … » passe —
# le mot après « quel » doit être la personne elle-même.
_PERSONNE_RANK_RE = re.compile(
    r"\bquel(?:le|s|les)?\s+(?:responsable|agent|personne|porteur|opérateur|employé)s?\b"
    r"|\bqui\s+(?:porte|fait|commet|a|ont)\b.{0,30}\ble\s+(?:plus|moins)\b",
    re.IGNORECASE)

# Voies déterministes pré-routeur : sens d'un sigle, et lookup exact par numéro FE.
_SIGLE_TRIGGER_RE = re.compile(
    r"signif|veut\s+dire|c'est\s+quoi|qu'est.?ce\s+qu|\bdéfinition\b|\bdefinition\b|"
    r"abréviat|abreviat|\bsigle\b|acronyme", re.IGNORECASE)
_FE_RE = re.compile(r"FNE(?:\s?[A-Z]{2,})?[\s/\-]?\d{2,4}[/\-]\d{2,4}", re.IGNORECASE)


def _voie_sigle(question: str):
    """Répond au sens d'un sigle depuis le glossaire validé (déjà chargé)."""
    if not _SIGLE_TRIGGER_RE.search(question):
        return None
    up = " " + question.upper() + " "
    for sig, sens in glossaire.paires():
        if sig and re.search(r"[^A-Z0-9]" + re.escape(sig.upper()) + r"[^A-Z0-9]", up):
            return f"**{sig}** = {sens}"
    return None


# ── HOOK DE CAPTURE (Étape 1) : stocke (question, réponse, CONTEXTE) sur toutes
# les voies → corpus de triplets auto-alimenté, jugeable a posteriori. Non bloquant.
# ⚠️ contient des données REX → à traiter comme donnée sensible (accès restreint).
_CAPTURE_PATH = os.environ.get("CAPTURE_PATH", "/app/logs/interactions.jsonl")
_capture_lock = threading.Lock()


def _capturer(question: str, voie: str, answer: str, diag: dict) -> None:
    """Extrait le GROUNDING selon la voie et l'append en JSONL. Ne casse jamais la requête."""
    try:
        ctx: dict = {}
        if diag.get("sources"):
            ctx["sources"] = diag["sources"]                       # recherche : passages retenus
        ana = diag.get("analyste")
        if ana:
            ctx["faits"] = ana.get("faits"); ctx["plan"] = ana.get("plan")  # analyse : chiffres calculés
        if diag.get("incidents_similaires"):
            ctx["incidents_similaires"] = diag["incidents_similaires"]        # reco
        if diag.get("cypher_execute"):
            ctx["cypher"] = diag["cypher_execute"]; ctx["resultat_brut"] = diag.get("resultat_brut")
        if diag.get("interpretation_llm"):
            ctx["spec"] = diag["interpretation_llm"]
        if diag.get("numero_fe"):
            ctx["numero_fe"] = diag["numero_fe"]                    # fiche
        rec = {"ts": datetime.now(timezone.utc).isoformat(), "question": question,
               "voie": voie, "answer": answer, "contexte": ctx,
               "jugeable": voie in ("recherche", "recommandation", "analyse")}
        rec = confidentialite.masquer_nominatif(rec)  # jamais de nominatif dans le log
        os.makedirs(os.path.dirname(_CAPTURE_PATH), exist_ok=True)
        with _capture_lock, open(_CAPTURE_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:  # noqa: BLE001
        logger.exception("Hook de capture échoué (non bloquant)")


def _repondre(question: str, voie: str, answer: str, justification: str,
              diag: dict, t0: float) -> dict:
    """SORTIE UNIQUE de /auto : horodate, capture (masquée par _capturer), et
    construit la réponse. Toute nouvelle voie DOIT sortir par ici — c'est le
    point de passage qui garantit capture + politique de confidentialité."""
    diag["temps_ms"] = int((time.time() - t0) * 1000)
    _capturer(question, voie, answer, diag)
    return {"answer": answer, "voie_choisie": voie,
            "justification": justification, "diagnostics": diag}


def _voie_fiche(question: str, neo4j):
    """Lookup EXACT d'un incident par son numéro FE (recherche ≠ similarité)."""
    m = _FE_RE.search(question)
    if not m:
        return None
    ref = re.sub(r"\s+", "", m.group(0)).upper().rstrip("/-")
    if len(ref) < 6:
        return None
    rows = neo4j.run(
        "MATCH (i:IncidentSecu) WHERE toUpper(replace(i.numero_fe,' ','')) CONTAINS $ref "
        "RETURN i.numero_fe AS fe, i.titre AS titre, i.severite AS sev, "
        "left(i.date_evenement,10) AS date, i.classification AS cls, "
        "left(i.resume_llm,400) AS resume, left(i.action_corrective,300) AS action "
        "ORDER BY i.numero_fe LIMIT 1", ref=ref)
    return rows[0] if rows else None
import prompt_store
import glossaire


# =====================================================================
# CONFIGURATION
# =====================================================================

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_TIMEOUT = float(os.environ.get("OLLAMA_TIMEOUT", "120"))
RERANKER_URL = os.environ.get("RERANKER_URL", "http://ia-reranker-test:80")

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
    app.state.ollama = OllamaClient(url=OLLAMA_URL, timeout=OLLAMA_TIMEOUT)
    app.state.reranker = RerankerClient(url=RERANKER_URL)
    logger.info("Reranker : %s", RERANKER_URL)

    # Deny-list nominative (culture juste) : énumérée depuis le graphe au démarrage.
    confidentialite.charger(app.state.neo4j)

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


@app.middleware("http")
async def _masque_nominatif_middleware(request: Request, call_next):
    """FILET nominatif central (culture juste) : masque les identifiants de la
    deny-list dans TOUTE réponse JSON sortante — les voies retirent déjà le
    nominatif à la source ; ceci attrape le texte libre et le code futur."""
    response = await call_next(request)
    ct = response.headers.get("content-type", "")
    if confidentialite.actif() and ct.startswith("application/json"):
        body = b""
        async for chunk in response.body_iterator:
            body += chunk
        texte = body.decode("utf-8", errors="replace")
        masque = confidentialite.masquer_texte(texte)
        if masque != texte:
            logger.warning("Masquage nominatif appliqué sur %s", request.url.path)
        headers = dict(response.headers)
        headers.pop("content-length", None)
        from starlette.responses import Response as _Resp
        return _Resp(content=masque, status_code=response.status_code,
                     headers=headers, media_type=ct)
    return response


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
        # Comptage Neo4j par label (IncidentSecu = schéma actuel ; :Incident est legacy)
        incidents_count = neo4j.run(
            "MATCH (i:IncidentSecu) WHERE coalesce(i.is_test_data,false)=false "
            "RETURN count(i) AS c"
        )[0]["c"]

        enriched_count = neo4j.run(
            "MATCH (i:IncidentSecu) WHERE coalesce(i.is_test_data,false)=false "
            "AND i.resume_llm IS NOT NULL AND i.resume_llm <> '0' RETURN count(i) AS c"
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
# ENDPOINT : POST /ask/incident-v2
# =====================================================================

@app.post("/ask/incident-v2", response_model=AskIncidentV2Response)
async def ask_incident_v2(request: Request, body: AskRequest) -> AskIncidentV2Response:
    start = time.time()
    try:
        retrieval = retrieve_incident_v2(
            question=body.question,
            ollama=request.app.state.ollama,
            qdrant=request.app.state.qdrant,
            neo4j=request.app.state.neo4j,
            top_k=body.top_k,
        )
        generation = generate_answer_incident_v2(
            question=body.question,
            retrieval_result=retrieval,
            ollama=request.app.state.ollama,
            neo4j=request.app.state.neo4j,
        )
        meta_champs = champ_meta(request.app.state.neo4j)
        sources = [
            SourceIncidentV2(
                numero_fe=item.props.get("numero_fe"),
                titre=item.props.get("titre"),
                severite=item.props.get("severite"),
                classification=item.props.get("classification"),
                etat=item.props.get("etat"),
                date_evenement=item.props.get("date_evenement"),
                resume_llm=item.props.get("resume_llm"),
                action_corrective=item.props.get("action_corrective"),
                score=item.best_score,
                matched_fields=item.matched_fields,
                entites=item.entites,
                champs=champs_labellises(item.props, meta_champs),
            )
            for item in retrieval.items
        ]
        return AskIncidentV2Response(
            answer=generation.answer,
            sources=sources,
            metadata=AskResponseMetadata(
                duration_ms=int((time.time() - start) * 1000),
                n_chunks_retrieved=retrieval.n_chunks_retrieved,
                n_incidents_unique=retrieval.n_direct,
                n_incidents_expanded=0,
                model_used=generation.model_used,
            ),
        )
    except Exception as e:
        logger.exception("Erreur lors du traitement de /ask/incident-v2")
        raise HTTPException(status_code=500, detail=str(e))


# =====================================================================
# ENDPOINT : POST /ask/incident-v2/stats
# =====================================================================

_NOT_AGGREGATION_ANSWER = (
    "Cette question ne semble pas être une demande de comptage ou de répartition. "
    "Posez une question du type : \"Combien d'incidents en 2025 ?\", "
    "\"Répartition par sévérité\", \"Incidents sérieux en 2024\"."
)


@app.post("/ask/incident-v2/stats", response_model=AggregationResponse)
async def ask_incident_v2_stats(request: Request, body: AskRequest) -> AggregationResponse:
    """Voie agrégation à vocabulaire FERMÉ (AggregationSpec) : champs limités
    mais comportement très prévisible. Pour les contraintes hors vocabulaire
    (compagnie, lieu...), utiliser /ask/incident-v2/query (moteur générique).
    ⚠️ Limitation connue : une contrainte hors vocabulaire est ignorée
    silencieusement (ex. « impliquant easyjet »)."""
    try:
        result = run_aggregation(
            question=body.question,
            ollama=request.app.state.ollama,
            neo4j=request.app.state.neo4j,
        )

        if result["status"] == "not_aggregation":
            return AggregationResponse(
                answer=_NOT_AGGREGATION_ANSWER,
                rows=[],
            )

        spec = result["spec"]
        answer = phrase_result(
            question=body.question,
            spec=spec,
            rows=result["rows"],
            total=result["total"],
            ollama=request.app.state.ollama,
        )
        return AggregationResponse(
            answer=answer,
            metric=spec.metric,
            group_by=spec.group_by,
            filters_applied=result["filters_applied"],
            rows=result["rows"],
            total=result["total"],
        )
    except Exception as e:
        logger.exception("Erreur lors du traitement de /ask/incident-v2/stats")
        raise HTTPException(status_code=500, detail=str(e))


# =====================================================================
# ENDPOINT : POST /ask/incident-v2/entity
# =====================================================================

@app.post("/ask/incident-v2/entity", response_model=EntityLookupResponse)
async def ask_incident_v2_entity(request: AskRequest):
    """Recherche d'incidents par nom d'entité satellite (compagnie, société, etc.)."""
    start = time.time()
    try:
        matches = fetch_entity(
            name=request.question,
            neo4j=app.state.neo4j,
        )
        result = phrase_entity_result(
            entity_query=request.question,
            matches=matches,
            ollama=app.state.ollama,
        )
        duration_ms = int((time.time() - start) * 1000)
        return EntityLookupResponse(
            answer=result.answer,
            matches=[
                EntityMatchResponse(
                    label=m.label,
                    rel=m.rel,
                    entity_name=m.entity_name,
                    incident_count=m.incident_count,
                    sample_incidents=m.incidents[:10],
                )
                for m in result.matches
            ],
            metadata=AskResponseMetadata(
                duration_ms=duration_ms,
                n_chunks_retrieved=0,
                n_incidents_unique=sum(m.incident_count for m in result.matches),
                n_incidents_expanded=0,
                model_used=result.model_used,
            ),
        )
    except Exception as e:
        logger.exception("Erreur lors du traitement de /ask/incident-v2/entity")
        raise HTTPException(status_code=500, detail=str(e))


# =====================================================================
# ENDPOINT : POST /ask/incident-v2/actions
# =====================================================================

@app.post("/ask/incident-v2/actions", response_model=ActionLookupResponse)
async def ask_incident_v2_actions(request: Request, body: AskRequest) -> ActionLookupResponse:
    """Questions sur les actions correctives et préventives liées aux incidents."""
    try:
        result = run_action_lookup(
            question=body.question,
            ollama=request.app.state.ollama,
            neo4j=request.app.state.neo4j,
        )

        if result["status"] == "refus_personne":
            return ActionLookupResponse(answer=confidentialite.REFUS_PERSONNE, rows=[])

        if result["status"] == "not_action_query":
            return ActionLookupResponse(
                answer=(
                    "Cette question ne semble pas porter sur les actions correctives "
                    "ou préventives. Exemples : \"Quelles actions correctives pour les "
                    "incidents FOD ?\", \"Actions en cours sur les incidents sérieux\", "
                    "\"Combien d'actions préventives clôturées en 2025 ?\"."
                ),
                rows=[],
            )

        if result["status"] == "no_filters":
            return ActionLookupResponse(
                answer=(
                    f"Votre question ne précise aucun critère reconnu (type d'action, "
                    f"statut, mot-clé, sévérité, action à chaud...). "
                    f"La base contient {result['total']} actions au total — précisez "
                    f"votre question, par exemple : \"Quelles actions correctives pour "
                    f"les incidents FOD ?\", \"Quelles actions sont encore en cours ?\", "
                    f"\"Donne-moi les incidents avec une action à chaud\"."
                ),
                rows=[],
                total=result["total"],
            )

        generation = phrase_action_result(
            question=body.question,
            rows=result["rows"],
            total=result["total"],
            spec=result["spec"],
            ollama=request.app.state.ollama,
        )

        spec = result["spec"]
        filters = {
            k: v for k, v in spec.model_dump().items()
            if k.startswith("f_") and v is not None
        } if spec else {}

        return ActionLookupResponse(
            answer=generation.answer,
            rows=[ActionResult(**r) for r in result["rows"]],
            total=result["total"],
            filters_applied=filters,
        )
    except Exception as e:
        logger.exception("Erreur lors du traitement de /ask/incident-v2/actions")
        raise HTTPException(status_code=500, detail=str(e))


# =====================================================================
# ENDPOINT : POST /ask/incident-v2/query  (moteur générique tous champs)
# =====================================================================

@app.post("/ask/incident-v2/query", response_model=GenericQueryResponse)
async def ask_incident_v2_query(request: Request, body: AskRequest) -> GenericQueryResponse:
    """Moteur unifié count/repartition/liste sur spec fermée — chiffres exacts garantis."""
    try:
        result = run_unified_query(
            question=body.question,
            ollama=request.app.state.ollama,
            neo4j=request.app.state.neo4j,
        )

        if result["status"] == "parse_failed":
            return GenericQueryResponse(
                answer=(
                    "Je n'ai pas réussi à interpréter cette question. Reformulez en "
                    "mentionnant un critère précis : sévérité, année, condition lumineuse, "
                    "présence de blessés, classification, ou demandez un comptage, "
                    "une répartition ou une liste d'incidents."
                ),
            )

        if result["status"] == "besoin_precision":
            return GenericQueryResponse(
                answer=(
                    "Votre question ne porte pas sur les incidents de sécurité, ou manque "
                    "de précision. Exemples : \"Combien d'incidents avec des blessés ?\", "
                    "\"Les 5 derniers incidents de nuit\", \"Répartition par sévérité en 2025\"."
                ),
                spec_interpretee=result.get("spec_interpretee"),
            )

        generation = phrase_unified_result(
            question=body.question,
            result=result,
            ollama=request.app.state.ollama,
        )

        spec = result["spec"]
        resultat_brut = result["resultat_brut"]

        filtres_lisibles = [
            {"champ": k.replace("f_", ""), "op": "=", "valeur": str(v)}
            for k, v in spec.model_dump().items()
            if k.startswith("f_") and v is not None
        ]

        rows: list[dict] = []
        total: int | None = None
        if spec.output == "count":
            total = int(resultat_brut) if resultat_brut is not None else 0
        elif spec.output == "repartition":
            rows = resultat_brut or []
            total = sum(r.get("n", 0) for r in rows)
        else:  # liste
            rows = resultat_brut or []
            total = len(rows)

        return GenericQueryResponse(
            answer=generation.answer,
            intent=spec.output,
            filtres=filtres_lisibles,
            group_by=spec.group_by,
            rows=rows,
            total=total,
            spec_interpretee=result["spec_interpretee"],
            cypher_execute=result["cypher_execute"],
            resultat_brut=resultat_brut,
        )
    except Exception as e:
        logger.exception("Erreur lors du traitement de /ask/incident-v2/query")
        raise HTTPException(status_code=500, detail=str(e))


# =====================================================================
# ENDPOINT : POST /ask/incident-v2/recommande
# =====================================================================

@app.post("/ask/incident-v2/recommande", response_model=RecommandationResponse)
async def ask_incident_v2_recommande(request: Request, body: AskRequest) -> RecommandationResponse:
    """Décrivez un événement : l'assistant retrouve les incidents similaires
    et recommande les actions (préventives, correctives, à chaud) qui avaient
    été prises pour eux."""
    try:
        result = run_recommendation(
            description=body.question,
            ollama=request.app.state.ollama,
            qdrant=request.app.state.qdrant,
            neo4j=request.app.state.neo4j,
            reranker=request.app.state.reranker,
            # large : peu d'incidents portent des actions documentées (~10 %),
            # il faut ratisser assez d'incidents similaires pour en trouver
            top_k=max(body.top_k, 20),
        )
        generation = phrase_recommendation(
            description=body.question,
            result=result,
            ollama=request.app.state.ollama,
        )
        return RecommandationResponse(
            answer=generation.answer,
            incidents_similaires=result.incidents,
            actions=[
                ActionRecommandeeResponse(
                    type_action=a.type_action,
                    titre=a.titre,
                    fe_sources=a.fe_sources,
                    statut=a.statut,
                )
                for a in result.actions
            ],
        )
    except Exception as e:
        logger.exception("Erreur lors du traitement de /ask/incident-v2/recommande")
        raise HTTPException(status_code=500, detail=str(e))


# =====================================================================
# ENDPOINT : POST /ask/incident-v2/synthese  (assistant de synthèse Safety)
# =====================================================================

@app.post("/ask/incident-v2/synthese")
async def ask_incident_v2_synthese(request: Request, body: AskRequest):
    """Brouillon d'analyse Safety à partir d'un cas déclaré : contexte historique,
    précédents similaires, tendance, zones, actions typiques/jugées efficaces.
    À valider par l'analyste (ce n'est pas un verdict)."""
    try:
        res = run_synthese(
            description=body.question,
            ollama=request.app.state.ollama,
            qdrant=request.app.state.qdrant,
            neo4j=request.app.state.neo4j,
            reranker=request.app.state.reranker,
        )
        return {
            "abstention": res.abstention,
            "type_dominant": res.type_dominant,
            "contexte": res.contexte,
            "facteurs": res.facteurs,
            "brouillon": res.brouillon,
            "precedents": res.precedents,
            "actions": [
                {"type_action": a.type_action, "titre": a.titre,
                 "fe_sources": a.fe_sources, "statut": a.statut}
                for a in res.actions
            ],
        }
    except Exception as e:
        logger.exception("Erreur lors du traitement de /ask/incident-v2/synthese")
        raise HTTPException(status_code=500, detail=str(e))


# =====================================================================
# ENDPOINTS ÉVAL — console de mesure du frontend de test
# =====================================================================

_RUNNERS_UI = {  # whitelist : grille -> runner (rien d'autre n'est exécutable)
    "couvrant": "test_auto_couvrant.py",
    "multihop": "test_multihop_grid.py",
    "analyste": "test_analyste_grid.py",
    "metier": "test_analyste_metier.py",
    "culture_juste": "test_culture_juste.py",
    "reco": "test_reco_hitk.py",
}
_RUN_UI_LOG = "/app/tests/run_ui.log"
_RUN_UI_LOCK = "/app/tests/.run_ui.lock"


@app.get("/eval/scores")
async def eval_scores():
    """Derniers scores par grille, lus dans les resultats_*.jsonl des runners."""
    out = []
    for nom in _RUNNERS_UI:
        path = f"/app/tests/resultats_{nom}.jsonl"
        try:
            rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
            oks = []
            for r in rows:  # deux formats historiques : ok booléen, ou verdict OK/FAIL
                if isinstance(r.get("ok"), bool):
                    oks.append(r["ok"])
                elif r.get("verdict") in ("OK", "FAIL"):
                    oks.append(r["verdict"] == "OK")
            out.append({"grille": nom, "n": len(rows),
                        "ok": sum(oks) if oks else None,
                        "pct": round(100 * sum(oks) / len(oks), 1) if oks else None,
                        "date": datetime.fromtimestamp(os.path.getmtime(path),
                                                       tz=timezone.utc).isoformat()})
        except FileNotFoundError:
            out.append({"grille": nom, "n": 0, "ok": None, "pct": None, "date": None})
        except Exception as e:  # noqa: BLE001
            out.append({"grille": nom, "erreur": str(e)})
    return {"scores": out, "run_en_cours": os.path.exists(_RUN_UI_LOCK)}


@app.post("/eval/run")
async def eval_run(body: dict):
    """Lance UNE grille (whitelist) en tâche de fond — un seul run à la fois."""
    grille = (body or {}).get("grille")
    script = _RUNNERS_UI.get(grille)
    if not script:
        raise HTTPException(status_code=400, detail=f"grille inconnue : {grille}")
    if os.path.exists(_RUN_UI_LOCK) and time.time() - os.path.getmtime(_RUN_UI_LOCK) < 1800:
        return {"status": "busy", "detail": "un run est déjà en cours"}
    open(_RUN_UI_LOCK, "w").write(grille)
    import subprocess
    subprocess.Popen(
        ["sh", "-c",
         f"python3 /app/tests/{script} > {_RUN_UI_LOG} 2>&1; rm -f {_RUN_UI_LOCK}"],
        cwd="/app", start_new_session=True)
    return {"status": "lance", "grille": grille}


@app.get("/eval/log")
async def eval_log():
    """Queue du log du run UI + indicateur d'exécution."""
    try:
        lignes = open(_RUN_UI_LOG, encoding="utf-8", errors="replace").read().splitlines()
    except FileNotFoundError:
        lignes = []
    return {"log": lignes[-80:], "run_en_cours": os.path.exists(_RUN_UI_LOCK)}


_FEEDBACK_PATH = os.environ.get("FEEDBACK_PATH", "/app/logs/feedback.jsonl")


@app.post("/feedback")
async def feedback(body: dict):
    """👍/👎 utilisateur sur une réponse — le signal d'adoption du pilote.
    Masqué comme tout le reste (aucun nominatif en log)."""
    rec = {"ts": datetime.now(timezone.utc).isoformat(),
           "question": (body or {}).get("question", "")[:500],
           "voie": (body or {}).get("voie", ""),
           "utile": bool((body or {}).get("utile")),
           "commentaire": ((body or {}).get("commentaire") or "")[:500]}
    rec = confidentialite.masquer_nominatif(rec)
    os.makedirs(os.path.dirname(_FEEDBACK_PATH), exist_ok=True)
    with _capture_lock, open(_FEEDBACK_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return {"status": "merci"}


@app.get("/feedback/stats")
async def feedback_stats():
    """Agrégat du signal 👍/👎 (pilote) : total, part utile, par voie."""
    try:
        rows = [json.loads(l) for l in open(_FEEDBACK_PATH, encoding="utf-8") if l.strip()]
    except FileNotFoundError:
        return {"total": 0, "utiles": 0, "pct": None, "par_voie": {}}
    par_voie: dict = {}
    for r in rows:
        v = par_voie.setdefault(r.get("voie", "?"), {"total": 0, "utiles": 0})
        v["total"] += 1
        v["utiles"] += 1 if r.get("utile") else 0
    utiles = sum(1 for r in rows if r.get("utile"))
    return {"total": len(rows), "utiles": utiles,
            "pct": round(100 * utiles / len(rows), 1) if rows else None,
            "par_voie": par_voie}


@app.get("/capture/recent")
async def capture_recent(n: int = 30, jugeable: bool | None = None):
    """Dernières interactions capturées (déjà masquées par le verrou nominatif)."""
    try:
        lignes = open(_CAPTURE_PATH, encoding="utf-8", errors="replace").read().splitlines()
    except FileNotFoundError:
        return {"total": 0, "interactions": []}
    out = []
    for l in reversed(lignes):
        try:
            r = json.loads(l)
        except Exception:  # noqa: BLE001
            continue
        if jugeable is not None and r.get("jugeable") != jugeable:
            continue
        out.append({"ts": r.get("ts"), "question": r.get("question"),
                    "voie": r.get("voie"), "jugeable": r.get("jugeable"),
                    "answer": (r.get("answer") or "")[:300]})
        if len(out) >= max(1, min(n, 100)):
            break
    return {"total": len(lignes), "interactions": out}


# =====================================================================
# ENDPOINT : POST /ask/incident-v2/auto  (point d'entrée unique — routeur)
# =====================================================================

@app.post("/ask/incident-v2/auto")
async def ask_incident_v2_auto(request: Request, body: AskRequest):
    """Point d'entrée UNIQUE : le routeur classe la question et l'oriente
    automatiquement vers la bonne voie. Renvoie la réponse + la voie choisie
    et la justification du routeur (pour transparence)."""
    ollama = request.app.state.ollama
    neo4j = request.app.state.neo4j
    qdrant = request.app.state.qdrant
    t0 = time.time()
    try:
        # ── VOIES DÉTERMINISTES (pré-routeur) : motif non ambigu, réponse exacte ──
        if _PERSONNE_RANK_RE.search(body.question):
            return _repondre(body.question, "abstention", confidentialite.REFUS_PERSONNE,
                             "culture juste — pas d'analyse rattachée à des personnes",
                             {"pre_routeur": "refus_personne"}, t0)
        sig = _voie_sigle(body.question)
        if sig:
            return _repondre(body.question, "sigle", sig,
                             "définition d'un sigle (glossaire validé)",
                             {"pre_routeur": "sigle"}, t0)
        fiche = _voie_fiche(body.question, neo4j)
        if fiche:
            txt = (f"**{fiche['fe']}** — {fiche['titre']}\n\n"
                   f"Date : {fiche['date'] or '?'} · Sévérité : {fiche['sev']} · "
                   f"{fiche['cls'] or ''}\n\n{fiche['resume'] or '(pas de résumé)'}\n\n"
                   f"Action corrective : {fiche['action'] or '—'}")
            return _repondre(body.question, "fiche", txt,
                             "recherche exacte par numéro FE",
                             {"pre_routeur": "fiche", "numero_fe": fiche["fe"]}, t0)

        decision = router(body.question, ollama)
        cap = decision.get("capacite", "abstention")
        diag: dict = {"routeur": {
            "capacite": cap,
            "justification": decision.get("justification", ""),
            "prompt": prompt_store.rendre("routeur", question=body.question),
        }}
        answer = ""

        # Garde-fou déterministe : un COMPTAGE dont le SUJET est « actions » doit aller
        # à la voie actions, pas à l'agrégation d'incidents (le routeur confond parfois
        # l'opération « comptage » et le sujet « actions »).
        if cap == "agregation" and _SUJET_ACTIONS_RE.search(body.question):
            cap = "actions"
            diag["routeur"]["garde_fou"] = "sujet=actions → forcé de agregation vers actions"

        # ── Couche ANALYSTE (agentic) : pour les questions analytiques (proportion,
        # fréquence/classement, tendance, synthèse d'une population), on tente d'abord
        # l'analyste. S'il n'est pas applicable, on retombe sur la voie normale.
        # "recommandation" est EXCLUE : un récit d'événement n'est jamais analytique,
        # et le planner 7b sur-grabbe cette frontière (vu au test des 51 exemples).
        if cap in ("agregation", "recherche", "actions", "abstention"):
            ana = run_analyste(body.question, request.app.state.ollama, neo4j)
            if ana.status == "ok":
                diag["analyste"] = {"plan": ana.plan, "faits": ana.faits}
                return _repondre(body.question, "analyse", ana.answer,
                                 "question analytique — décomposée, calculée, puis synthétisée",
                                 diag, t0)

        if cap == "abstention":
            answer = ("Cette question sort du périmètre de l'assistant (donnée absente, "
                      "hors domaine, ou commande). Posez une question sur les incidents "
                      "de sécurité — comptage, recherche, actions, ou recommandation.")

        elif cap == "actions":
            result = run_action_lookup(question=body.question, ollama=ollama, neo4j=neo4j)
            diag["statut_voie"] = result.get("status")
            if result["status"] == "refus_personne":
                answer = confidentialite.REFUS_PERSONNE
            elif result["status"] in ("not_action_query", "no_filters"):
                answer = ("Précisez votre question sur les actions correctives ou "
                          "préventives (type, statut, mot-clé...).")
                diag["total_actions_base"] = result.get("total")
            else:
                spec = result["spec"]
                gen = phrase_action_result(question=body.question, rows=result["rows"],
                                           total=result["total"], spec=spec, ollama=ollama)
                answer = gen.answer
                diag["filtres_interpretes"] = ({k: v for k, v in spec.model_dump().items()
                                                if k.startswith("f_") and v is not None} if spec else {})
                diag["total"] = result["total"]
                diag["sources"] = result["rows"][:20]

        elif cap == "recommandation":
            result = run_recommendation(description=body.question, ollama=ollama,
                                        qdrant=qdrant, neo4j=neo4j, top_k=max(body.top_k, 20))
            gen = phrase_recommendation(description=body.question, result=result, ollama=ollama)
            answer = gen.answer
            diag["incidents_similaires"] = result.incidents[:15]
            diag["n_actions_recommandees"] = len(result.actions)

        elif cap == "recherche":
            retrieval = retrieve_incident_v2(question=body.question, ollama=ollama,
                                             qdrant=qdrant, neo4j=neo4j, top_k=body.top_k)
            gen = generate_answer_incident_v2(question=body.question, retrieval_result=retrieval,
                                              ollama=ollama, neo4j=neo4j)
            answer = gen.answer
            diag["prompt_llm"] = gen.prompt
            diag["modele_reponse"] = gen.model_used
            diag["seuil_score"] = MIN_SCORE
            diag["n_chunks_recuperes"] = retrieval.n_chunks_retrieved
            diag["n_incidents_retenus"] = retrieval.n_direct
            diag["aucune_source_pertinente"] = retrieval.below_threshold
            diag["sources"] = [
                {"numero_fe": it.props.get("numero_fe"),
                 "titre": it.props.get("titre"),
                 "score": round(it.best_score, 3),
                 "champs_correspondants": it.matched_fields,
                 "resume": (it.props.get("resume_llm") or "")[:220]}
                for it in retrieval.items
            ]

        else:  # agregation (défaut) -> moteur unifié count/répartition/liste
            result = run_unified_query(question=body.question, ollama=ollama, neo4j=neo4j)
            diag["statut_voie"] = result.get("status")
            diag["interpretation_llm"] = result.get("spec_interpretee")
            diag["prompt_parseur"] = prompt_store.rendre(
                "unified_query.parseur", question=body.question,
                annee_courante=str(datetime.now(timezone.utc).year),
                glossaire=glossaire.bloc_glossaire())
            if result["status"] in ("parse_failed", "besoin_precision"):
                answer = ("Je n'ai pas réussi à interpréter cette question comme un "
                          "comptage ou une répartition. Reformulez avec un critère précis.")
            else:
                gen = phrase_unified_result(question=body.question, result=result, ollama=ollama)
                answer = gen.answer
                diag["cypher_execute"] = result.get("cypher_execute")
                diag["resultat_brut"] = result.get("resultat_brut")

        return _repondre(body.question, cap, answer,
                         decision.get("justification", ""), diag, t0)
    except Exception as e:
        logger.exception("Erreur lors du traitement de /ask/incident-v2/auto")
        raise HTTPException(status_code=500, detail=str(e))


# =====================================================================
# ENDPOINT : POST /ask/incident-v2/list
# =====================================================================

_NOT_LIST_ANSWER = (
    "Cette question ressemble à un comptage plutôt qu'à une demande de liste. "
    "Utilisez le mode **Agrégation / chiffres** ou l'endpoint "
    "/ask/incident-v2/stats pour obtenir des totaux et répartitions."
)


@app.post("/ask/incident-v2/list", response_model=StructuredListResponse)
async def ask_incident_v2_list(request: Request, body: AskRequest) -> StructuredListResponse:
    """Déprécié — délègue à la logique de /query en forçant output='liste'."""
    try:
        result = run_unified_query(
            question=body.question,
            ollama=request.app.state.ollama,
            neo4j=request.app.state.neo4j,
        )

        if result["status"] in ("parse_failed", "besoin_precision"):
            return StructuredListResponse(answer=_NOT_LIST_ANSWER, records=[], count=0)

        spec = result["spec"]

        # Guard : si le moteur interprète comme count ou repartition → redirect
        if spec.output != "liste":
            return StructuredListResponse(answer=_NOT_LIST_ANSWER, records=[], count=0)

        records = result["resultat_brut"] or []

        generation = phrase_unified_result(
            question=body.question,
            result=result,
            ollama=request.app.state.ollama,
        )

        filters = {
            k: v for k, v in spec.model_dump().items()
            if k.startswith("f_") and v is not None
        }

        return StructuredListResponse(
            answer=generation.answer,
            records=records,
            count=len(records),
            sort_by=spec.sort_by,
            order=spec.order,
            limit=spec.limit,
            filters_applied=filters,
        )
    except Exception as e:
        logger.exception("Erreur lors du traitement de /ask/incident-v2/list")
        raise HTTPException(status_code=500, detail=str(e))


# =====================================================================
# ENDPOINT : POST /ask/incident-v2/analyste  (prototype couche agent-analyste)
# =====================================================================

@app.post("/ask/incident-v2/analyste")
async def ask_incident_v2_analyste(request: Request, body: AskRequest):
    """Prototype 'analyste' : décompose une question analytique (proportion comparée),
    appelle plusieurs comptages déterministes, calcule, puis synthétise.
    Renvoie la réponse + le plan + les faits chiffrés (transparence)."""
    try:
        res = run_analyste(
            question=body.question,
            ollama=request.app.state.ollama,
            neo4j=request.app.state.neo4j,
        )
        if res.status != "ok":
            return {
                "answer": ("Cette question n'entre pas (encore) dans le périmètre de "
                           "l'analyste (patron pris en charge : proportion comparée)."),
                "status": res.status,
                "plan": res.plan,
            }
        return {"answer": res.answer, "status": "ok",
                "plan": res.plan, "faits": res.faits}
    except Exception as e:
        logger.exception("Erreur lors du traitement de /ask/incident-v2/analyste")
        raise HTTPException(status_code=500, detail=str(e))


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
