"""
Frontend Streamlit pour l'Assistant RETEX intra'know (v2).

Ameliorations v2 :
  - Avertissement si top_k > 10 (latence elevee)
  - Affichage explicite quand le LLM n'a pas trouve de pertinence
  - Tri des sources par score decroissant pour clarte
"""
from __future__ import annotations

import os
import time
from typing import Any

import httpx
import streamlit as st


API_URL = os.environ.get("API_URL", "http://localhost:8000")
DEFAULT_TOP_K = 5
SLOW_THRESHOLD = 10  # Au-dessus, on previent l'utilisateur
REQUEST_TIMEOUT = 300.0

EXAMPLE_QUESTIONS = [
    "Quels incidents impliquent un mauvais positionnement d'avion ?",
    "Y a-t-il eu des incidents avec des passagers agressifs ?",
    "Quels sont les incidents lies a la securite des aires de trafic ?",
    "Quels incidents impliquent un probleme de communication entre agents ?",
]

EXAMPLE_QUESTIONS_TICKETS = [
    "Quels tickets concernent des problemes de configuration ou de jobs Rundeck ?",
    "Y a-t-il des tickets critiques non resolus sur les modules metier ?",
    "Quels problemes ont ete rencontres sur les jobs d'evenements automatiques ?",
    "Quels tickets signalent des erreurs d'integration avec des API externes ?",
]


st.set_page_config(
    page_title="Assistant RETEX intra'know",
    layout="wide",
    initial_sidebar_state="expanded",
)


def call_health() -> dict[str, Any] | None:
    try:
        r = httpx.get(f"{API_URL}/health", timeout=5.0)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.sidebar.error(f"Erreur /health : {e}")
        return None


def call_stats() -> dict[str, Any] | None:
    try:
        r = httpx.get(f"{API_URL}/stats", timeout=5.0)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.sidebar.error(f"Erreur /stats : {e}")
        return None


def call_ask(question: str, top_k: int) -> dict[str, Any] | None:
    try:
        r = httpx.post(
            f"{API_URL}/ask",
            json={"question": question, "top_k": top_k},
            timeout=REQUEST_TIMEOUT,
        )
        r.raise_for_status()
        return r.json()
    except httpx.TimeoutException:
        st.error("Le serveur a mis trop de temps a repondre. Reessayez.")
        return None
    except Exception as e:
        st.error(f"Erreur lors de l'appel a l'API : {e}")
        return None
    
def call_ask_is(question: str, top_k: int) -> dict[str, Any] | None:
    try:
        r = httpx.post(
            f"{API_URL}/ask/info-securite",
            json={"question": question, "top_k": top_k},
            timeout=REQUEST_TIMEOUT,
        )
        r.raise_for_status()
        return r.json()
    except httpx.TimeoutException:
        st.error("Le serveur a mis trop de temps a repondre. Reessayez.")
        return None
    except Exception as e:
        st.error(f"Erreur lors de l'appel a l'API : {e}")
        return None



def render_sidebar() -> None:
    st.sidebar.title("Etat du systeme")

    st.sidebar.subheader("Services")
    health = call_health()
    if health:
        status = health.get("status", "unknown")
        if status == "healthy":
            st.sidebar.success(f"Statut global : {status}")
        elif status == "degraded":
            st.sidebar.warning(f"Statut global : {status}")
        else:
            st.sidebar.error(f"Statut global : {status}")

        for service in health.get("services", []):
            name = service.get("name", "?")
            svc_status = service.get("status", "?")
            icon = "🟢" if svc_status == "up" else "🔴"
            st.sidebar.write(f"{icon} {name}")
    else:
        st.sidebar.error("API inaccessible")

    st.sidebar.divider()

    st.sidebar.subheader("Bases de donnees")
    stats = call_stats()
    if stats:
        neo4j_stats = stats.get("neo4j", {})
        qdrant_stats = stats.get("qdrant", {})

        st.sidebar.caption("Contenu par base")
        b1, b2, b3 = st.sidebar.columns(3)
        b1.metric("Incidents", neo4j_stats.get("incidents", 0))
        b2.metric("Tickets", neo4j_stats.get("tickets", 0))
        b3.metric("Info Secu.", neo4j_stats.get("info_securite", 0))

        st.sidebar.caption("Enrichis par LLM")
        e1, e2 = st.sidebar.columns(2)
        e1.metric("Incidents", neo4j_stats.get("incidents_enriched", 0))
        e2.metric("Tickets", neo4j_stats.get("tickets_enriched", 0))

        st.sidebar.caption("Graphe")
        g1, g2 = st.sidebar.columns(2)
        g1.metric("Societes", neo4j_stats.get("societes", 0))
        g2.metric("Personnes", neo4j_stats.get("personnes", 0))

        st.sidebar.metric(
            "Chunks Qdrant",
            qdrant_stats.get("points_count", 0),
        )

    st.sidebar.divider()
    st.sidebar.caption(f"API : {API_URL}")


def render_answer(response: dict[str, Any]) -> None:
    answer = response.get("answer", "")
    st.markdown("### Reponse")

    # Detection des reponses "rien trouve" pour affichage adapte
    no_result_markers = [
        "Je n'ai pas trouve",
        "Aucun incident pertinent",
    ]
    is_no_result = any(marker in answer for marker in no_result_markers)

    if is_no_result:
        st.info(answer)
    else:
        st.markdown(answer)


def render_metadata(response: dict[str, Any]) -> None:
    metadata = response.get("metadata", {})

    st.markdown("---")
    cols = st.columns(5)
    cols[0].metric("Duree", f"{metadata.get('duration_ms', 0)} ms")
    cols[1].metric("Chunks", metadata.get("n_chunks_retrieved", 0))
    cols[2].metric("Incidents", metadata.get("n_incidents_unique", 0))
    cols[3].metric("Expanded", metadata.get("n_incidents_expanded", 0))
    cols[4].metric("Modele", metadata.get("model_used", "?"))


def render_sources(response: dict[str, Any]) -> None:
    sources = response.get("sources", [])
    if not sources:
        return

    # Tri par score decroissant (les expanded ont score 0 et passent a la fin)
    sources_sorted = sorted(
        sources,
        key=lambda s: s.get("best_score", 0.0),
        reverse=True,
    )

    st.markdown("### Sources")
    st.caption(f"{len(sources_sorted)} incident(s) recupere(s) par le RAG")

    for source in sources_sorted:
        numero = source.get("numero_fe", "?")
        titre = source.get("titre", "?")
        score = source.get("best_score", 0.0)
        matched = source.get("matched_fields", [])
        is_expanded = "graph_expansion" in matched

        badge = "Voisin graphe" if is_expanded else "Resultat direct"

        with st.expander(f"{badge} | {numero} - {titre}", expanded=False):
            col1, col2 = st.columns([2, 1])

            with col1:
                if source.get("resume_llm"):
                    st.markdown(f"**Resume** : {source['resume_llm']}")

                if source.get("facteur_causal") or source.get("severite_percue"):
                    parts = []
                    if source.get("facteur_causal"):
                        parts.append(f"Facteur : {source['facteur_causal']}")
                    if source.get("severite_percue"):
                        parts.append(f"Severite : {source['severite_percue']}")
                    st.caption(" | ".join(parts))

            with col2:
                st.metric("Score Qdrant", f"{score:.3f}")
                st.caption(f"Champs : {', '.join(matched)}")
                if source.get("date_evenement"):
                    st.caption(f"Date : {source['date_evenement'][:10]}")

def call_ask_tickets(question: str, top_k: int) -> dict[str, Any] | None:
    try:
        r = httpx.post(
            f"{API_URL}/ask/tickets",
            json={"question": question, "top_k": top_k},
            timeout=REQUEST_TIMEOUT,
        )
        r.raise_for_status()
        return r.json()
    except httpx.TimeoutException:
        st.error("Le serveur a mis trop de temps a repondre. Reessayez.")
        return None
    except Exception as e:
        st.error(f"Erreur lors de l'appel a l'API : {e}")
        return None


def render_metadata_tickets(response: dict[str, Any]) -> None:
    metadata = response.get("metadata", {})
    st.markdown("---")
    cols = st.columns(5)
    cols[0].metric("Duree", f"{metadata.get('duration_ms', 0)} ms")
    cols[1].metric("Chunks", metadata.get("n_chunks_retrieved", 0))
    cols[2].metric("Tickets directs", metadata.get("n_incidents_unique", 0))
    cols[3].metric("Tickets lies", metadata.get("n_incidents_expanded", 0))
    cols[4].metric("Modele", metadata.get("model_used", "?"))


def render_sources_tickets(response: dict[str, Any]) -> None:
    sources = response.get("sources", [])
    if not sources:
        return

    sources_sorted = sorted(sources, key=lambda s: s.get("best_score", 0.0), reverse=True)
    st.markdown("### Sources")
    st.caption(f"{len(sources_sorted)} ticket(s) recupere(s) par le RAG")

    REL_LABELS = {
        "via_enfant": "parent/enfant",
        "via_expert": "meme intervenant",
        "via_projet": "meme projet",
        "via_client": "meme client",
    }
    for source in sources_sorted:
        numero = source.get("numero_fe", "?")
        titre = source.get("titre", "Sans titre")
        score = source.get("best_score", 0.0)
        matched = source.get("matched_fields", [])
        is_expanded = source.get("is_expanded", False)
        if is_expanded:
            rel = matched[0] if matched else ""
            badge = f"Voisin graphe ({REL_LABELS.get(rel, 'contexte')})"
        else:
            badge = "Resultat direct"

        with st.expander(f"{badge} | #{numero} - {titre}", expanded=False):
            col1, col2 = st.columns([2, 1])
            with col1:
                if source.get("llm_resume"):
                    st.markdown(f"**Resume** : {source['llm_resume']}")
                pills = []
                if source.get("type_nc"):
                    pills.append(f"Type : **{source['type_nc']}**")
                if source.get("importance"):
                    pills.append(f"Importance : {source['importance']}")
                if source.get("urgence"):
                    pills.append(f"Urgence : {source['urgence']}")
                if source.get("etat"):
                    pills.append(f"Etat : {source['etat']}")
                if source.get("llm_domaine_technique"):
                    pills.append(f"Domaine : {source['llm_domaine_technique']}")
                if pills:
                    st.caption(" | ".join(pills))
            with col2:
                st.metric("Score Qdrant", f"{score:.3f}")
                if source.get("etape_label"):
                    st.caption(f"Etape : {source['etape_label']}")
                if source.get("site_application"):
                    st.caption(f"Application : {source['site_application']}")
                if source.get("projet_nom"):
                    st.caption(f"Projet : {source['projet_nom']}")
                if source.get("client"):
                    st.caption(f"Client : {source['client']}")
                if source.get("structure"):
                    st.caption(f"Structure : {source['structure']}")
                if source.get("individu"):
                    st.caption(f"Individu : {source['individu']}")
                if source.get("date_nc"):
                    st.caption(f"Date : {source['date_nc'][:10]}")
                st.caption(f"Champs : {', '.join(matched)}")


def render_sources_is(response: dict[str, Any]) -> None:
    sources = response.get("sources", [])
    if not sources:
        return

    sources_sorted = sorted(sources, key=lambda s: s.get("best_score", 0.0), reverse=True)
    st.markdown("### Sources")
    st.caption(f"{len(sources_sorted)} IS recuperee(s) par le RAG")

    for source in sources_sorted:
        is_number = source.get("is_number", "?")
        titre = source.get("titre", "?")
        score = source.get("best_score", 0.0)
        matched = source.get("matched_fields", [])

        with st.expander(f"IS {is_number} - {titre}", expanded=False):
            col1, col2 = st.columns([2, 1])
            with col1:
                if source.get("llm_resume"):
                    st.markdown(f"**Resume** : {source['llm_resume']}")
                if source.get("operateurs_concernes"):
                    st.caption(f"Operateurs : {', '.join(source['operateurs_concernes'])}")
            with col2:
                st.metric("Score Qdrant", f"{score:.3f}")
                st.caption(f"Champs : {', '.join(matched)}")
                if source.get("annee"):
                    st.caption(f"Annee : {source['annee']}")


def main() -> None:
    render_sidebar()
    mode = st.radio(
        "Base de donnees",
        ["Incidents intra'know", "Info Securite DGAC", "Tickets intra'know"],
        horizontal=True,
    )

    st.title("Assistant RETEX intra'know")

    if mode == "Tickets intra'know":
        st.caption(
            "Posez une question sur les tickets de support intra'know. "
            "L'assistant recherche par similarite semantique et expansion graphe (application, projet)."
        )
        examples = EXAMPLE_QUESTIONS_TICKETS
        placeholder = "Ex: Quels tickets signalent des erreurs de traitement des donnees ?"
    elif mode == "Info Securite DGAC":
        st.caption(
            "Posez une question sur les Instructions de Securite DGAC. "
            "L'assistant utilise une recherche hybride graphe + vectoriel."
        )
        examples = EXAMPLE_QUESTIONS
        placeholder = "Ex: Quels risques sont lies au transport de charges a l'elingue ?"
    else:
        st.caption(
            "Posez une question en langage naturel sur les incidents de securite "
            "aeronautique. L'assistant utilise une recherche hybride graphe + vectoriel."
        )
        examples = EXAMPLE_QUESTIONS
        placeholder = "Ex: Quels incidents impliquent un mauvais positionnement d'avion ?"

    st.markdown("**Exemples de questions :**")
    cols = st.columns(2)
    for i, example in enumerate(examples):
        with cols[i % 2]:
            if st.button(example, key=f"ex_{i}", use_container_width=True):
                st.session_state["question_input"] = example

    st.markdown("---")

    question = st.text_area(
        "Votre question",
        value=st.session_state.get("question_input", ""),
        height=80,
        placeholder=placeholder,
    )

    col1, col2 = st.columns([3, 1])
    with col1:
        top_k = st.slider(
            "Nombre de chunks a recuperer",
            min_value=1,
            max_value=20,
            value=DEFAULT_TOP_K,
            help=(
                "Plus la valeur est elevee, plus le contexte est large. "
                "Mais la reponse devient plus lente et peut contenir du bruit."
            ),
        )
        if top_k > SLOW_THRESHOLD:
            st.warning(
                f"Avec top_k = {top_k}, la reponse peut prendre 15-25 secondes."
            )
    with col2:
        st.write("")
        st.write("")
        submit = st.button("Poser la question", type="primary", use_container_width=True)

    if submit and question.strip():
        with st.spinner("Recherche dans les bases et generation de la reponse..."):
            start = time.time()
            if mode == "Info Securite DGAC":
                response = call_ask_is(question.strip(), top_k)
            elif mode == "Tickets intra'know":
                response = call_ask_tickets(question.strip(), top_k)
            else:
                response = call_ask(question.strip(), top_k)
            duration = time.time() - start

        if response:
            st.success(f"Reponse generee en {duration:.1f} secondes")
            render_answer(response)
            if mode == "Tickets intra'know":
                render_metadata_tickets(response)
                render_sources_tickets(response)
            elif mode == "Info Securite DGAC":
                render_metadata(response)
                render_sources_is(response)
            else:
                render_metadata(response)
                render_sources(response)

    elif submit and not question.strip():
        st.warning("Veuillez saisir une question avant de soumettre.")


if __name__ == "__main__":
    main()
