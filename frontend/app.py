from __future__ import annotations

import os
import time
from typing import Any

import httpx
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")
DEFAULT_TOP_K = 5
REQUEST_TIMEOUT = 1200.0

# ─── exemples par mode ────────────────────────────────────────────────────────

EX_V2_SEARCH = [
    "FOD sur l'aire de trafic",
    "Intrusion en zone de sécurité par un véhicule",
    "Approche non stabilisée ou remise de gaz",
    "Incident avec fumée à bord",
]
EX_V2_STATS = [
    "Combien d'incidents en 2025 ?",
    "Répartition par sévérité",
    "Combien d'incidents sérieux en 2024 ?",
    "Incidents de nuit vs de jour ?",
]
EX_V2_ENTITY = [
    "compagnie HOP",
    "Air France",
    "Twin Jet",
    "Transavia",
]
EX_LEGACY = [
    "Quels incidents impliquent un mauvais positionnement d'avion ?",
    "Y a-t-il eu des incidents avec des passagers agressifs ?",
    "Quels incidents impliquent un probleme de communication entre agents ?",
    "Quels sont les incidents lies a la securite des aires de trafic ?",
]
EX_TICKETS = [
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


# ─── appels API ───────────────────────────────────────────────────────────────

def _post(path: str, body: dict) -> dict | None:
    try:
        r = httpx.post(f"{API_URL}{path}", json=body, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except httpx.TimeoutException:
        st.error("Le serveur a mis trop de temps à répondre (>20 min). Réessayez.")
        return None
    except Exception as e:
        st.error(f"Erreur API : {e}")
        return None


def call_health() -> dict | None:
    try:
        r = httpx.get(f"{API_URL}/health", timeout=5.0)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def call_stats_api() -> dict | None:
    try:
        r = httpx.get(f"{API_URL}/stats", timeout=5.0)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


# ─── sidebar ──────────────────────────────────────────────────────────────────

def render_sidebar() -> None:
    st.sidebar.title("État du système")

    health = call_health()
    if health:
        status = health.get("status", "unknown")
        fn = st.sidebar.success if status == "healthy" else st.sidebar.warning
        fn(f"Statut : {status}")
        for svc in health.get("services", []):
            icon = "🟢" if svc.get("status") == "up" else "🔴"
            st.sidebar.write(f"{icon} {svc.get('name', '?')}")
    else:
        st.sidebar.error("API inaccessible")

    st.sidebar.divider()

    stats = call_stats_api()
    if stats:
        neo4j = stats.get("neo4j", {})
        qdrant = stats.get("qdrant", {})
        st.sidebar.metric("Chunks Qdrant", f"{qdrant.get('points_count', 0):,}")
        st.sidebar.caption(f"API : {API_URL}")


# ─── render générique ─────────────────────────────────────────────────────────

def _show_examples(examples: list[str], key_prefix: str) -> None:
    st.markdown("**Exemples :**")
    cols = st.columns(2)
    for i, ex in enumerate(examples):
        if cols[i % 2].button(ex, key=f"{key_prefix}_{i}", use_container_width=True):
            st.session_state["q"] = ex


def _question_input(placeholder: str) -> str:
    return st.text_area(
        "Votre question",
        value=st.session_state.get("q", ""),
        height=80,
        placeholder=placeholder,
    )


# ─── mode INCIDENTS v2 — RECHERCHE ───────────────────────────────────────────

def render_v2_search() -> None:
    st.caption("Recherche sémantique sur 9 191 incidents de sécurité (IncidentSecu). "
               "Réponse LLM + sources avec résumé. ⚠️ ~7 min sur CPU.")
    _show_examples(EX_V2_SEARCH, "sv2")
    st.markdown("---")

    top_k = st.slider("Incidents à récupérer", 1, 10, DEFAULT_TOP_K)
    question = _question_input("Ex : FOD sur aire de trafic")

    col1, col2 = st.columns([4, 1])
    with col2:
        submit = st.button("Poser la question", type="primary", use_container_width=True)

    if submit and question.strip():
        with st.spinner("Recherche + génération LLM (peut prendre plusieurs minutes sur CPU)…"):
            t0 = time.time()
            resp = _post("/ask/incident-v2", {"question": question.strip(), "top_k": top_k})
            dur = time.time() - t0

        if resp:
            st.success(f"Réponse en {dur:.0f}s")
            _render_answer(resp)
            _render_meta(resp)
            _render_sources_v2(resp)
    elif submit:
        st.warning("Saisissez une question.")


def _render_answer(resp: dict) -> None:
    answer = resp.get("answer", "")
    if any(m in answer for m in ["n'ai pas trouvé", "Aucun incident pertinent", "non servi"]):
        st.info(answer)
    else:
        st.markdown(f"### Réponse\n{answer}")


def _render_meta(resp: dict) -> None:
    m = resp.get("metadata", {})
    st.markdown("---")
    c = st.columns(4)
    c[0].metric("Durée", f"{m.get('duration_ms', 0)} ms")
    c[1].metric("Incidents", m.get("n_incidents_unique", 0))
    c[2].metric("Chunks", m.get("n_chunks_retrieved", 0))
    c[3].metric("Modèle", m.get("model_used", "?"))


def _render_sources_v2(resp: dict) -> None:
    sources = sorted(resp.get("sources", []), key=lambda s: s.get("score", 0), reverse=True)
    if not sources:
        return
    st.markdown("### Sources")
    for s in sources:
        fe = s.get("numero_fe", "?")
        titre = s.get("titre", "?")
        score = s.get("score", 0.0)
        with st.expander(f"FE {fe} — {titre} (score {score:.3f})", expanded=False):
            c1, c2 = st.columns([2, 1])
            with c1:
                desc = s.get("resume_llm") or s.get("action_corrective") or ""
                if desc:
                    st.markdown(f"**Résumé** : {desc[:300]}")
            with c2:
                if s.get("severite"):
                    st.caption(f"Sévérité : {s['severite']}")
                if s.get("classification"):
                    st.caption(f"Classification : {s['classification']}")
                if s.get("date_evenement"):
                    st.caption(f"Date : {s['date_evenement'][:10]}")
                st.caption(f"Champs : {', '.join(s.get('matched_fields', []))}")


# ─── mode INCIDENTS v2 — AGRÉGATION ──────────────────────────────────────────

def render_v2_stats() -> None:
    st.caption("Posez une question de comptage ou répartition. "
               "Pipeline : LLM → Cypher Neo4j → chiffres exacts. ⚠️ ~7 min sur CPU.")
    _show_examples(EX_V2_STATS, "sv2st")
    st.markdown("---")

    question = _question_input("Ex : Combien d'incidents en 2025 ?")
    _, col2 = st.columns([4, 1])
    with col2:
        submit = st.button("Poser la question", type="primary", use_container_width=True)

    if submit and question.strip():
        with st.spinner("Analyse + requête Neo4j…"):
            t0 = time.time()
            resp = _post("/ask/incident-v2/stats", {"question": question.strip(), "top_k": 5})
            dur = time.time() - t0

        if resp:
            st.success(f"Réponse en {dur:.0f}s")
            answer = resp.get("answer", "")
            if "ne semble pas être une demande" in answer:
                st.warning(answer)
            else:
                st.markdown(f"### Réponse\n{answer}")

            rows = resp.get("rows", [])
            total = resp.get("total")

            if rows:
                st.markdown("### Détail")
                for row in rows:
                    pct = row["n"] / total * 100 if total else 0
                    st.progress(pct / 100, text=f"{row['label']} : **{row['n']:,}** ({pct:.1f}%)")
            elif total is not None:
                st.metric("Total", f"{total:,}")

            if resp.get("filters_applied"):
                st.caption(f"Filtres : {resp['filters_applied']}")
    elif submit:
        st.warning("Saisissez une question.")


# ─── mode INCIDENTS v2 — ENTITÉ ──────────────────────────────────────────────

def render_v2_entity() -> None:
    st.caption("Recherchez tous les incidents liés à une compagnie, société ou entité. "
               "Pas besoin d'orthographe exacte. ⚠️ ~7 min sur CPU pour le phrasage LLM.")
    _show_examples(EX_V2_ENTITY, "sv2ent")
    st.markdown("---")

    question = _question_input("Ex : compagnie HOP  ou  Air France  ou  Twin Jet")
    _, col2 = st.columns([4, 1])
    with col2:
        submit = st.button("Rechercher", type="primary", use_container_width=True)

    if submit and question.strip():
        with st.spinner("Recherche dans le graphe Neo4j…"):
            t0 = time.time()
            resp = _post("/ask/incident-v2/entity", {"question": question.strip(), "top_k": 5})
            dur = time.time() - t0

        if resp:
            st.success(f"Réponse en {dur:.0f}s")
            _render_answer(resp)

            matches = resp.get("matches", [])
            if matches:
                st.markdown("### Entités trouvées")
                for m in matches:
                    with st.expander(
                        f"**{m['label']}** « {m['entity_name']} » — "
                        f"{m['incident_count']} incident(s)",
                        expanded=True,
                    ):
                        for inc in m.get("sample_incidents", [])[:5]:
                            fe = inc.get("fe", "?")
                            titre = inc.get("titre", "?")
                            resume = inc.get("resume") or ""
                            sev = inc.get("severite", "")
                            date = (inc.get("date") or "")[:10]
                            st.markdown(
                                f"- **FE {fe}** [{date}] *{sev}* — "
                                f"{resume[:120] + '…' if len(resume) > 120 else resume or titre}"
                            )
    elif submit:
        st.warning("Saisissez un nom d'entité.")


# ─── modes hérités ────────────────────────────────────────────────────────────

def render_legacy() -> None:
    st.caption("Recherche sur les incidents anciens (100 incidents, format legacy).")
    _show_examples(EX_LEGACY, "sleg")
    st.markdown("---")
    top_k = st.slider("Chunks à récupérer", 1, 20, DEFAULT_TOP_K)
    question = _question_input("Ex : Quels incidents impliquent un mauvais positionnement ?")
    _, col2 = st.columns([4, 1])
    with col2:
        submit = st.button("Poser la question", type="primary", use_container_width=True)
    if submit and question.strip():
        with st.spinner("Recherche…"):
            t0 = time.time()
            resp = _post("/ask", {"question": question.strip(), "top_k": top_k})
            dur = time.time() - t0
        if resp:
            st.success(f"Réponse en {dur:.0f}s")
            _render_answer(resp)
            _render_meta(resp)


def render_tickets() -> None:
    st.caption("Recherche sur les tickets de support intra'know.")
    _show_examples(EX_TICKETS, "stck")
    st.markdown("---")
    top_k = st.slider("Chunks à récupérer", 1, 20, DEFAULT_TOP_K)
    question = _question_input("Ex : Tickets critiques non résolus sur les modules métier ?")
    _, col2 = st.columns([4, 1])
    with col2:
        submit = st.button("Poser la question", type="primary", use_container_width=True)
    if submit and question.strip():
        with st.spinner("Recherche…"):
            t0 = time.time()
            resp = _post("/ask/tickets", {"question": question.strip(), "top_k": top_k})
            dur = time.time() - t0
        if resp:
            st.success(f"Réponse en {dur:.0f}s")
            _render_answer(resp)
            _render_meta(resp)


# ─── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    render_sidebar()

    st.title("Assistant RETEX intra'know")

    mode = st.radio(
        "Base",
        ["Incidents Sécurité v2", "Tickets", "Incidents (legacy)"],
        horizontal=True,
    )

    st.markdown("---")

    if mode == "Incidents Sécurité v2":
        sub = st.radio(
            "Type de question",
            ["🔍 Recherche sémantique", "📊 Agrégation / chiffres", "🏢 Par compagnie / entité"],
            horizontal=True,
        )
        st.markdown("")
        if sub == "🔍 Recherche sémantique":
            render_v2_search()
        elif sub == "📊 Agrégation / chiffres":
            render_v2_stats()
        else:
            render_v2_entity()

    elif mode == "Tickets":
        render_tickets()

    else:
        render_legacy()


if __name__ == "__main__":
    main()
