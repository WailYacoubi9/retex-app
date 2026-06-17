"""
Format canonique du pipeline RETEX (v2).

Toutes les structures de donnees qui circulent entre les couches du
pipeline (extraction -> enrichissement -> chargement) sont definies ici.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time
from typing import Any, Optional


# =====================================================================
# ENTITES DU GRAPHE
# =====================================================================

@dataclass
class SocieteRef:
    """Reference a une societe intra'know (multi-tenancy).

    Utilisation : matérialise un nœud Neo4j (:Societe) lié à l'incident
    par une relation [:CONCERNE]. Capture l'isolation par client.
    """
    id_societe: int

    def to_neo4j_props(self) -> dict[str, Any]:
        """Retourne un dict des proprietes pour MERGE Cypher."""
        return {"id_societe": self.id_societe}


@dataclass
class PersonneRef:
    """Reference a une personne identifiee dans intra'know.

    Utilisation : nœud Neo4j (:Personne) avec login unique, lié à un
    Incident par une relation typée selon le rôle (EMIS_PAR, etc.).
    """
    login: str
    id_emetteur: Optional[int] = None
    role: str = "emetteur"

    def to_neo4j_props(self) -> dict[str, Any]:
        """Properties pour MERGE Cypher (sans le rôle, qui est sur la relation)."""
        props = {"login": self.login}
        if self.id_emetteur is not None:
            props["id_emetteur"] = self.id_emetteur
        return props


@dataclass
class ReferentielRef:
    """Reference a une valeur de liste de reference (ChampSupp\\ListeContenu).

    Utilisation : nœud Neo4j (:Referentiel) identifié par (family, code).
    Lié à l'Incident par une relation typée déclarée dans le YAML.
    """
    family: str
    ref_id: int
    code: str
    label: str
    code_externe: Optional[str] = None
    relation_type: str = "LIE_A"

    @property
    def ref_key(self) -> str:
        """Cle d'unicite Neo4j combinant famille et code."""
        return f"{self.family}:{self.code}"

    def to_neo4j_props(self) -> dict[str, Any]:
        """Properties pour MERGE Cypher."""
        props = {
            "ref_key": self.ref_key,
            "family": self.family,
            "ref_id": self.ref_id,
            "code": self.code,
            "label": self.label,
        }
        if self.code_externe:
            props["code_externe"] = self.code_externe
        return props


# =====================================================================
# ENRICHISSEMENT LLM
# =====================================================================

@dataclass
class LLMEnrichment:
    """Resultat de l'enrichissement LLM sur un incident.

    Utilisation : peuplé par llm_enricher si l'incident a un narratif
    suffisant. Stocké comme propriétés du nœud Incident.
    """
    resume: Optional[str] = None
    facteur_causal: Optional[str] = None
    severite_percue: Optional[str] = None
    etat_final: Optional[str] = None
    model_used: Optional[str] = None


# =====================================================================
# FORMAT PIVOT CANONIQUE
# =====================================================================

@dataclass
class IncidentCanonique:
    """Format pivot d'un incident apres extraction.

    Utilisation : structure intermediaire stable entre l'extracteur
    (qui lit le payload brut) et le loader (qui ecrit dans Neo4j+Qdrant).
    Decouple la source des cibles. Construit a partir du YAML de mapping.
    """

    # Identification
    incident_id: str                          # UUID v5 stable (deterministe)
    incident_id_source: str                   # _id du payload original
    source_module: str                        # ex: "q_incident_securite"
    source_url: Optional[str] = None          # URL HAL si on veut tracer

    # Champs metier core
    numero_fe: Optional[str] = None
    type_nc: Optional[str] = None
    abbr: Optional[str] = None
    titre: Optional[str] = None
    detail: Optional[str] = None

    # Narratifs optionnels
    recolte_faits: Optional[str] = None
    notes_suivi: Optional[str] = None

    # Workflow et etat
    etape: Optional[int] = None
    archive: Optional[int] = None

    # Consequences
    blesses_raw: Optional[int] = None
    presence_blesses: Optional[bool] = None

    # Flags
    pret_envoi_eccairs: bool = False

    # Dates
    date_evenement: Optional[datetime] = None
    date_creation: Optional[datetime] = None
    heure_evenement: Optional[time] = None

    # Relations vers autres entites
    societes: list[SocieteRef] = field(default_factory=list)
    personnes: list[PersonneRef] = field(default_factory=list)
    referentiels: list[ReferentielRef] = field(default_factory=list)

    # Enrichissement LLM (peuple plus tard)
    llm: Optional[LLMEnrichment] = None

    # Champs rares non types
    extra_fields: dict[str, Any] = field(default_factory=dict)

    # Metadonnees techniques
    is_test_data: bool = False
    last_indexed_at: Optional[str] = None

    def has_narrative_content(self, min_length: int = 50) -> bool:
        """Indique si l'incident a assez de contenu pour merite un appel LLM.

        Utilisation : appelee par le pipeline pour decider d'invoquer
        l'enrichissement LLM (qui est couteux). Seuil par defaut : 50 chars
        sur le champ detail (champ narratif principal).
        """
        if not self.detail:
            return False
        return len(self.detail) >= min_length

    def get_text_for_field(self, canonical_name: str) -> Optional[str]:
        """Retourne le texte d'un champ canonique pour embedding.

        Utilisation : appelee par le loader pour generer les chunks
        Qdrant. Permet un acces uniforme par nom canonique.
        """
        return getattr(self, canonical_name, None)


# =====================================================================
# UTILITAIRES
# =====================================================================

def make_incident_uuid(source_module: str, incident_id_source: str) -> str:
    """Genere un UUID v5 stable a partir du module et de l'id source.

    Utilisation : garantit l'idempotence du pipeline. Le meme incident
    ingere plusieurs fois aura le meme UUID, donc MERGE Cypher mettra a
    jour au lieu de dupliquer.
    """
    import uuid
    namespace = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")  # DNS namespace
    name = f"{source_module}:{incident_id_source}"
    return str(uuid.uuid5(namespace, name))
