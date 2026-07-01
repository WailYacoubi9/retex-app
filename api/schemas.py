"""
Modeles Pydantic pour l'API FastAPI du projet RETEX.

Definit le contrat formel des requetes et reponses :
  - Ce que le client peut envoyer (validation automatique)
  - Ce que le serveur garantit en retour

Tous les endpoints (POST /ask, GET /health, GET /stats) referencent
ces modeles, qui generent automatiquement la documentation Swagger
accessible sur http://localhost:8000/docs
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


# =====================================================================
# REQUETES (entrantes)
# =====================================================================

class AskRequest(BaseModel):
    """Corps d'une requete POST /ask.

    Utilisation : valide automatiquement les requetes entrantes.
    Une question vide ou trop longue est rejetee avec un 422.
    """

    question: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="Question en langage naturel sur les incidents.",
        examples=["Quels incidents impliquent une erreur humaine ?"],
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Nombre de chunks vectoriels a recuperer dans Qdrant.",
    )

    include_test_data: bool = Field(
        default=False,
        description="Si true, inclut aussi les incidents marques is_test_data.",
    )


# =====================================================================
# REPONSES (sortantes)
# =====================================================================

class SourceIncident(BaseModel):
    """Un incident utilise comme source pour la reponse.

    Utilisation : chaque element de la liste 'sources' dans la reponse
    POST /ask. Permet a l'utilisateur de tracer l'origine de la reponse.
    """

    incident_id: str = Field(
        ...,
        description="UUID interne de l'incident (cle commune Neo4j et Qdrant).",
    )

    incident_id_source: str = Field(
        ...,
        description="ID original de l'incident dans intra'know.",
    )

    numero_fe: Optional[str] = Field(
        default=None,
        description="Numero de Fiche d'Evenement (ex: FNE/26/0245).",
    )

    titre: Optional[str] = Field(
        default=None,
        description="Titre court de l'incident.",
    )

    resume_llm: Optional[str] = Field(
        default=None,
        description="Resume genere par le LLM lors de l'ingestion. Peut etre absent pour les incidents au narratif trop court.",
    )

    facteur_causal: Optional[str] = Field(
        default=None,
        description="Classification LLM du facteur causal (humain, technique, etc.).",
    )

    severite_percue: Optional[str] = Field(
        default=None,
        description="Classification LLM de la severite (mineure, majeure, etc.).",
    )

    date_evenement: Optional[str] = Field(
        default=None,
        description="Date de l'evenement au format ISO.",
    )

    best_score: float = Field(
        ...,
        description="Meilleur score de similarite Qdrant parmi les chunks de cet incident (0 a 1).",
    )

    matched_fields: list[str] = Field(
        default_factory=list,
        description="Liste des champs canoniques qui ont matche la question.",
    )


class AskResponseMetadata(BaseModel):
    """Metadonnees techniques de la reponse RAG.

    Utilisation : informations de debug et de monitoring pour le client.
    """

    duration_ms: int = Field(
        ...,
        description="Duree totale de traitement en millisecondes.",
    )

    n_chunks_retrieved: int = Field(
        ...,
        description="Nombre de chunks Qdrant recuperes.",
    )

    n_incidents_unique: int = Field(
        ...,
        description="Nombre d'incidents uniques apres regroupement.",
    )

    n_incidents_expanded: int = Field(
        ...,
        description="Nombre d'incidents ajoutes par expansion graphe Neo4j.",
    )

    model_used: str = Field(
        ...,
        description="Nom du modele LLM utilise pour la generation.",
    )


class AskResponse(BaseModel):
    """Reponse complete d'un POST /ask.

    Utilisation : objet retourne au client. Contient la reponse en
    langage naturel, les sources, et les metadonnees.
    """

    answer: str = Field(
        ...,
        description="Reponse en francais generee par le LLM a partir du contexte.",
    )

    sources: list[SourceIncident] = Field(
        default_factory=list,
        description="Liste des incidents utilises comme sources de la reponse.",
    )

    metadata: AskResponseMetadata = Field(
        ...,
        description="Metadonnees techniques de la requete.",
    )


# =====================================================================
# REPONSES POUR /health
# =====================================================================

class ServiceStatus(BaseModel):
    """Statut d'un service backend (Neo4j, Qdrant, Ollama).

    Utilisation : un element de la liste services dans HealthResponse.
    """

    name: str = Field(..., description="Nom du service.")
    status: str = Field(..., description="'up', 'down', ou 'unknown'.")
    detail: Optional[str] = Field(
        default=None,
        description="Detail technique (message d'erreur si down).",
    )


class HealthResponse(BaseModel):
    """Reponse de GET /health.

    Utilisation : indique si l'API et ses dependances tournent. Permet
    aux outils de monitoring (et a l'utilisateur) de detecter les
    problemes rapidement.
    """

    status: str = Field(
        ...,
        description="'healthy', 'degraded', ou 'unhealthy'.",
    )

    services: list[ServiceStatus] = Field(
        default_factory=list,
        description="Detail du statut de chaque service backend.",
    )

    timestamp: str = Field(
        ...,
        description="Timestamp ISO du check.",
    )


# =====================================================================
# REPONSES POUR /stats
# =====================================================================

class Neo4jStats(BaseModel):
    """Statistiques sur la base Neo4j."""

    incidents: int = Field(..., description="Nombre total de noeuds Incident.")
    incidents_enriched: int = Field(..., description="Incidents avec resume_llm.")
    tickets: int = Field(default=0, description="Nombre total de noeuds Ticket.")
    tickets_enriched: int = Field(default=0, description="Tickets avec llm_resume.")
    info_securite: int = Field(default=0, description="Nombre total de noeuds InfoSecurite.")
    societes: int = Field(..., description="Nombre de noeuds Societe.")
    personnes: int = Field(..., description="Nombre de noeuds Personne.")
    referentiels: int = Field(..., description="Nombre de noeuds Referentiel.")


class QdrantStats(BaseModel):
    """Statistiques sur la base Qdrant."""

    collection: str = Field(..., description="Nom de la collection.")
    points_count: int = Field(..., description="Nombre de chunks stockes.")
    vector_dimension: int = Field(..., description="Dimension des vecteurs.")


class StatsResponse(BaseModel):
    """Reponse de GET /stats.

    Utilisation : donne une vue d'ensemble du volume de donnees en base.
    Utile pour la demo et pour valider que l'ingestion s'est bien passee.
    """

    neo4j: Neo4jStats = Field(..., description="Statistiques Neo4j.")
    qdrant: QdrantStats = Field(..., description="Statistiques Qdrant.")

class SourceInfoSecurite(BaseModel):
    info_securite_id: str
    is_number: Optional[str] = None
    annee: Optional[int] = None
    titre: Optional[str] = None
    llm_resume: Optional[str] = None
    operateurs_concernes: Optional[list[str]] = None
    best_score: float
    matched_fields: list[str] = Field(default_factory=list)

class AskResponseIS(BaseModel):
    answer: str
    sources: list[SourceInfoSecurite] = Field(default_factory=list)
    metadata: AskResponseMetadata


class SourceTicket(BaseModel):
    ticket_id: str
    numero_fe: Optional[str] = None
    titre: Optional[str] = None
    type_nc: Optional[str] = None
    importance: Optional[str] = None
    etat: Optional[str] = None
    etape_label: Optional[str] = None
    site_application: Optional[str] = None
    projet_nom: Optional[str] = None
    client: Optional[str] = None
    structure: Optional[str] = None
    urgence: Optional[str] = None
    individu: Optional[str] = None
    date_nc: Optional[str] = None
    llm_resume: Optional[str] = None
    llm_domaine_technique: Optional[str] = None
    best_score: float
    matched_fields: list[str] = Field(default_factory=list)
    is_expanded: bool = False


class AskResponseTickets(BaseModel):
    answer: str
    sources: list[SourceTicket] = Field(default_factory=list)
    metadata: AskResponseMetadata


# =====================================================================
# SCHÉMAS POUR /ask/incident-v2
# =====================================================================

class SourceIncidentV2(BaseModel):
    numero_fe: Optional[str] = None
    titre: Optional[str] = None
    severite: Optional[str] = None
    classification: Optional[str] = None
    etat: Optional[str] = None
    date_evenement: Optional[str] = None
    resume_llm: Optional[str] = None
    action_corrective: Optional[str] = None
    score: float
    matched_fields: list[str] = Field(default_factory=list)
    entites: list[dict] = Field(default_factory=list)


class AskIncidentV2Response(BaseModel):
    answer: str
    sources: list[SourceIncidentV2] = Field(default_factory=list)
    metadata: AskResponseMetadata


# =====================================================================
# SCHÉMAS POUR /ask/incident-v2/stats
# =====================================================================

class AggregationResponse(BaseModel):
    answer: str
    metric: str = "count"
    group_by: Optional[str] = None
    filters_applied: dict = Field(default_factory=dict)
    rows: list[dict] = Field(default_factory=list)
    total: Optional[int] = None


# =====================================================================
# SCHÉMAS POUR /ask/incident-v2/entity
# =====================================================================

class EntityMatchResponse(BaseModel):
    label: str
    rel: str
    entity_name: str
    incident_count: int
    sample_incidents: list[dict] = Field(default_factory=list)


class EntityLookupResponse(BaseModel):
    answer: str
    matches: list[EntityMatchResponse] = Field(default_factory=list)
    metadata: AskResponseMetadata
