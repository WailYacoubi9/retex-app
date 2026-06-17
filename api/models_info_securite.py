"""
Modeles canoniques pour le module d'ingestion DGAC Info Securite.

Definit la structure de donnees pivot `InfoSecuriteCanonique` qui sert
d'interface entre le parser (parser_dgac.py) et le loader (loader_info_securite.py).

Convention d'identifiant : info_securite_id est un UUID v5 stable derive
du is_number. Ingerer 10 fois la meme IS produit toujours le meme ID,
ce qui garantit l'idempotence du MERGE Neo4j.

Le dataclass est volontairement plat : peu de structures imbriquees pour
faciliter la serialisation en proprietes Neo4j. Les seules exceptions sont
operateurs_concernes (list[str]), remplace (list[str]), extra_fields (dict)
et llm (objet LLMResumeOperateur).
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Optional


# Namespace UUID pour le module DGAC. Choix arbitraire mais stable.
# Permet de generer des info_securite_id reproductibles depuis is_number.
DGAC_NAMESPACE_UUID = uuid.UUID("a6f4e2c8-9b1d-4e3f-8a7c-1d2e3f4a5b6c")


def make_info_securite_uuid(is_number: str) -> str:
    """Genere un UUID v5 stable a partir du numero d'IS.

    Utilisation : produit toujours le meme UUID pour le meme is_number,
    ce qui rend l'ingestion idempotente. Relancer 10 fois ne cree pas
    de doublons dans Neo4j.

    Args:
        is_number: Le numero canonique au format "AAAA/NN", ex "2024/01".

    Returns:
        Une string UUID v5, ex "3f4a8c9d-1234-5678-90ab-cdef12345678".

    Raises:
        ValueError: Si is_number est vide ou None.
    """
    if not is_number:
        raise ValueError("is_number ne peut pas etre vide")
    return str(uuid.uuid5(DGAC_NAMESPACE_UUID, is_number))


@dataclass
class LLMResumeOperateur:
    """Resume d'une IS DGAC du point de vue d'un operateur terrain.

    Reformulation par LLM (Llama 3.1 8B via Ollama) du contenu de l'IS
    pour faciliter le matching semantique avec des questions du type
    "j'ai rencontre ce probleme, quels conseils ?".

    En V1, c'est le SEUL enrichissement LLM applique aux IS DGAC.
    Pas d'etiquettes de classification (thematique, criticite, etc.)
    en V1 : on attend de voir les vrais cas d'usage avant de figer
    une taxonomie.

    Attributes:
        resume: Le texte du resume, 2-4 phrases, focalise sur le probleme
            terrain que l'IS adresse plutot que sur les recommandations
            administratives.
        model_used: L'identifiant du modele LLM utilise, pour tracabilite
            (ex "llama3.1:8b-instruct-q4_K_M"). Permet de detecter si un
            corpus a ete enrichi avec un ancien modele a re-traiter.
    """
    resume: str
    model_used: str


@dataclass
class InfoSecuriteCanonique:
    """Format pivot pour une Info Securite DGAC.

    Cette classe est l'interface entre :
      - parser_dgac.py qui lit un .md produit par OpenDataLoader
      - llm_enricher_info_securite.py qui attache un resume LLM
      - loader_info_securite.py qui ecrit dans Neo4j + Qdrant

    Tous les champs textuels (contexte, actions_recommandees, annexe, etc.)
    peuvent contenir du Markdown leger : des marqueurs "## Sous-section"
    sont inseres par le parser quand plusieurs sous-categories d'un meme
    champ sont concatenees (cf parser_dgac.py).

    Attributes:
        is_number: Numero canonique au format "AAAA/NN", ex "2024/01".
            Sert de cle naturelle. Genere depuis le nom de fichier.
        annee: Annee de l'IS, derivee de is_number. Pratique pour les
            requetes Cypher de filtrage temporel.
        info_securite_id: UUID v5 stable derive de is_number. Cle de
            stockage Neo4j et Qdrant. Garantit l'idempotence.
        source_module: Constante "dgac_info_securite". Permet de filtrer
            les nœuds Neo4j par module d'origine.

        titre: Titre humain de l'IS, derive du sujet ou du nom de fichier
            si sujet absent. Sert d'affichage rapide dans une UI.
        operateurs_concernes: Liste des operateurs cibles de l'IS,
            ex ["Exploitants d'aeronefs", "Pilotes qualifies IFR"].
            Stocke comme list[str] dans le dataclass et comme propriete
            de liste sur le nœud Neo4j. Pas de noeuds Referentiel separes
            en V1 (decision de simplification).
        sujet: Phrase courte resumant le theme de l'IS, ex "Risques lies
            aux erreurs de calage altimetrique en approche APV baro-VNAV".
        objectif: Phrase decrivant le but du bulletin, ex "Sensibiliser
            les operateurs sur la criticite de l'information altimetrique".

        contexte: Bloc narratif decrivant les faits et la situation qui
            motivent l'IS. Peut contenir des marqueurs "## Sous-section"
            pour des variantes specifiques (Contexte aeronautique,
            Contexte technique, Breve description de l'evenement, etc.).
        actions_recommandees: Bloc decrivant les recommandations DGAC.
            Peut contenir des marqueurs "## Sous-section" pour des
            variantes (Bonnes pratiques identifiees, Suites donnees, etc.).

        annexe: Contenu du champ "Annexe" ou "Annexes" du tableau du PDF
            quand present. Typiquement des references reglementaires
            (OPS 1.345, arretes, SIN EASA) integrees au bulletin.
        references: Contenu du champ "References" ou "Reference" du
            tableau du PDF quand present. Typiquement des liens externes
            (rapports BEA, guides DSAC, articles SKYbrary).

        contenu_hors_tableau: Tout le contenu textuel apres le dernier
            tableau Markdown. Concerne ~4 IS du corpus qui ont des
            annexes ou des references en flux libre apres les tableaux
            (cf 2022/02, 2022/03, 2024/03, 2018/01).

        version_numero: Numero de version de l'IS extrait du pied de page,
            ex 1, 2, 3. None si non detecte.
        date_version: Date de publication ou de revision de l'IS extraite
            du pied de page. None si non detectee. Note : seules ~10 IS
            sur 53 ont leur version detectee a cause des variantes de
            format de pied de page.

        remplace: Liste des numeros d'IS que ce bulletin remplace,
            detectes via "annule et remplace l'IS XXXX/YY" dans le PDF.
            En V1 la regex ne capture qu'1 cas sur 53 ; les autres
            chaines historiques seront a creuser en V2.

        llm: Resume LLM attache apres enrichissement. None tant que
            l'enrichissement n'a pas tourne.

        extra_fields: Dictionnaire pour les champs ad hoc detectes dans
            le tableau qui ne correspondent a aucun label canonique.
            Cles : le label normalise tel que vu dans le PDF.
            Ex pour 2020/05 : {"perspectives": "Le sujet des fume events..."}.
            Ex pour 2018/02 : {"reserves": "Les informations diffusees..."}.
            Ex pour 2021/01 : {"prevention des interferences 5g": "Des mesures..."}.
        is_test_data: Flag pour exclure des donnees de test des requetes
            de production. Par defaut False.
        last_indexed_at: Timestamp ISO 8601 de la derniere indexation,
            mis a jour par le loader. Sert au monitoring.
    """

    # Identification (technique)
    is_number: str
    annee: int
    info_securite_id: str
    titre: str
    source_module: str = "dgac_info_securite"

    # Metadonnees humaines
    operateurs_concernes: list[str] = field(default_factory=list)
    sujet: Optional[str] = None
    objectif: Optional[str] = None

    # Contenu principal du tableau
    contexte: Optional[str] = None
    actions_recommandees: Optional[str] = None

    # Contenu optionnel du tableau
    annexe: Optional[str] = None
    references: Optional[str] = None

    # Contenu hors tableau
    contenu_hors_tableau: Optional[str] = None

    # Versionnement
    version_numero: Optional[int] = None
    date_version: Optional[date] = None

    # Relations IS -> IS
    remplace: list[str] = field(default_factory=list)

    # Enrichissement LLM (attache apres parsing)
    llm: Optional[LLMResumeOperateur] = None

    # Champs ad hoc et techniques
    extra_fields: dict[str, Any] = field(default_factory=dict)
    is_test_data: bool = False
    last_indexed_at: Optional[str] = None