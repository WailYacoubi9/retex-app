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
EX_V2_ACTIONS = [
    "Quelles actions correctives ont été prises pour les incidents FOD ?",
    "Quelles actions sont encore en cours ?",
    "Combien d'actions préventives clôturées ?",
    "Donne-moi les incidents avec des actions à chaud",
]
EX_V2_QUERY = [
    "Combien d'incidents avec des blessés ?",
    "Les incidents de nuit en 2025",
    "Répartition des incidents par compagnie",
    "Combien d'incidents impliquant easyjet en 2024 ?",
]
EX_V2_LIST = [
    "Les 5 derniers incidents par date de création avec leurs actions correctives",
    "Les 3 incidents les plus récents de sévérité élevée",
    "Les incidents sérieux les plus anciens",
    "Les 10 derniers incidents de nuit en 2024 avec leur résumé",
]
EX_V2_RECO = [
    "Un camion a refusé la priorité à un avion au repoussage sur l'aire de trafic",
    "Découverte d'un morceau de métal sur la piste lors d'une inspection",
    "Collision entre un escabeau et la porte d'un avion au poste de stationnement",
    "Déversement de kérosène pendant l'avitaillement",
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
            # écrire DANS la clé du widget : c'est elle qui porte l'état
            # (le paramètre value= d'un widget à clé est ignoré après création)
            st.session_state[f"ta_{key_prefix}"] = ex


def _question_input(placeholder: str, key_prefix: str) -> str:
    # clé unique par onglet : avec st.tabs, tous les onglets sont rendus
    # simultanément — chaque zone de saisie doit avoir sa propre identité
    return st.text_area(
        "Votre question",
        height=80,
        placeholder=placeholder,
        key=f"ta_{key_prefix}",
    )


def _submit_button(label: str, key_prefix: str) -> bool:
    return st.button(label, type="primary", use_container_width=True,
                     key=f"btn_{key_prefix}")


# ─── mode INCIDENTS v2 — RECHERCHE ───────────────────────────────────────────

def render_v2_search() -> None:
    st.caption("Recherche sémantique sur 9 191 incidents de sécurité (IncidentSecu). "
               "Réponse LLM + sources. Quelques secondes sur GPU.")
    _show_examples(EX_V2_SEARCH, "sv2")
    st.markdown("---")

    top_k = st.slider("Incidents à récupérer", 1, 10, DEFAULT_TOP_K, key="sl_sv2")
    question = _question_input("Ex : FOD sur aire de trafic", "sv2")

    col1, col2 = st.columns([4, 1])
    with col2:
        submit = _submit_button("Poser la question", "sv2")

    if submit and question.strip():
        with st.spinner("Recherche + génération LLM…"):
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

    # Tableau récapitulatif : une ligne par incident
    st.dataframe(
        [
            {
                "FE": s.get("numero_fe"),
                "Titre": s.get("titre"),
                "Date": (s.get("date_evenement") or "")[:10],
                "Sévérité": s.get("severite"),
                "Classification": s.get("classification"),
                "État": s.get("etat"),
                "Score": round(s.get("score", 0.0), 3),
                "Champs matchés": ", ".join(s.get("matched_fields", [])),
            }
            for s in sources
        ],
        use_container_width=True,
        hide_index=True,
    )

    # Détail : tous les champs de chaque fiche (labels métier du schéma)
    for s in sources:
        fe = s.get("numero_fe", "?")
        titre = s.get("titre", "?")
        score = s.get("score", 0.0)
        with st.expander(f"FE {fe} — {titre} (score {score:.3f})", expanded=False):
            champs = s.get("champs") or []
            if champs:
                matched = set(s.get("matched_fields", []))
                st.dataframe(
                    [
                        {
                            "Champ": c.get("label"),
                            "Valeur": c.get("valeur"),
                            "🎯": "✓" if c.get("champ") in matched else "",
                        }
                        for c in champs
                    ],
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                desc = s.get("resume_llm") or s.get("action_corrective") or ""
                if desc:
                    st.markdown(f"**Résumé** : {desc[:300]}")

            entites = s.get("entites") or []
            noms = []
            for e in entites:
                ep = e.get("props") or {}
                nom = ep.get("nom") or ep.get("label") or ep.get("login") or ep.get("titre_action")
                if nom:
                    lbl = (e.get("labels") or ["?"])[0]
                    noms.append(f"{lbl} : {nom}")
            if noms:
                st.caption("Entités liées — " + " ; ".join(noms[:10]))


# ─── mode INCIDENTS v2 — AGRÉGATION ──────────────────────────────────────────

def render_v2_stats() -> None:
    st.caption("Comptages et répartitions sur un vocabulaire FERMÉ "
               "(sévérité, classification, jour/nuit, traitement, efficacité, année, mois) : "
               "très prévisible. Pour filtrer sur d'autres champs (compagnie, lieu, "
               "blessés…), utilisez 🧭 Question libre.")
    _show_examples(EX_V2_STATS, "sv2st")
    st.markdown("---")

    question = _question_input("Ex : Combien d'incidents en 2025 ?", "sv2st")
    _, col2 = st.columns([4, 1])
    with col2:
        submit = _submit_button("Poser la question", "sv2st")

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
               "Pas besoin d'orthographe exacte.")
    _show_examples(EX_V2_ENTITY, "sv2ent")
    st.markdown("---")

    question = _question_input("Ex : compagnie HOP  ou  Air France  ou  Twin Jet", "sv2ent")
    _, col2 = st.columns([4, 1])
    with col2:
        submit = _submit_button("Rechercher", "sv2ent")

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


# ─── mode INCIDENTS v2 — ACTIONS ─────────────────────────────────────────────

def render_v2_actions() -> None:
    st.caption("Questions sur les actions correctives, préventives et à chaud "
               "liées aux incidents (statut, responsable, efficacité…).")
    _show_examples(EX_V2_ACTIONS, "sv2act")
    st.markdown("---")

    question = _question_input("Ex : Quelles actions correctives pour les incidents FOD ?", "sv2act")
    _, col2 = st.columns([4, 1])
    with col2:
        submit = _submit_button("Poser la question", "sv2act")

    if submit and question.strip():
        with st.spinner("Analyse + requête Neo4j + phrasage LLM…"):
            t0 = time.time()
            resp = _post("/ask/incident-v2/actions", {"question": question.strip(), "top_k": 5})
            dur = time.time() - t0

        if resp:
            st.success(f"Réponse en {dur:.0f}s")
            answer = resp.get("answer", "")
            if "ne semble pas porter" in answer or "aucun critère reconnu" in answer:
                st.warning(answer)
            else:
                st.markdown(f"### Réponse\n{answer}")

            if resp.get("total") is not None:
                st.metric("Total correspondant", f"{resp['total']:,}")

            rows = resp.get("rows", [])
            if rows:
                st.markdown("### Actions (détail)")
                st.dataframe(rows, use_container_width=True)

            if resp.get("filters_applied"):
                st.caption(f"Filtres appliqués : {resp['filters_applied']}")
    elif submit:
        st.warning("Saisissez une question.")


# ─── mode INCIDENTS v2 — QUESTION LIBRE (moteur générique) ───────────────────

def render_v2_query() -> None:
    st.caption("Question libre sur N'IMPORTE QUEL champ de la fiche : propriétés, "
               "dates, entités liées, présence/absence, comptages et répartitions. "
               "Chiffres exacts garantis (Cypher déterministe).")
    _show_examples(EX_V2_QUERY, "sv2qry")
    st.markdown("---")

    question = _question_input("Ex : Combien d'incidents avec des blessés ?", "sv2qry")
    _, col2 = st.columns([4, 1])
    with col2:
        submit = _submit_button("Poser la question", "sv2qry")

    if submit and question.strip():
        with st.spinner("Analyse de la question + requête Neo4j…"):
            t0 = time.time()
            resp = _post("/ask/incident-v2/query", {"question": question.strip(), "top_k": 5})
            dur = time.time() - t0

        if resp:
            st.success(f"Réponse en {dur:.0f}s")
            answer = resp.get("answer", "")
            if "pas réussi à interpréter" in answer or "pas pu construire" in answer:
                st.warning(answer)
            else:
                st.markdown(f"### Réponse\n{answer}")

            if resp.get("total") is not None:
                st.metric("Total", f"{resp['total']:,}")

            rows = resp.get("rows", [])
            if rows and resp.get("intent") == "repartition":
                st.markdown("### Répartition")
                total = resp.get("total") or sum(r.get("n", 0) for r in rows)
                for row in rows:
                    pct = row["n"] / total * 100 if total else 0
                    st.progress(pct / 100, text=f"{row['label']} : **{row['n']:,}** ({pct:.1f}%)")
            elif rows:
                st.markdown("### Incidents (détail)")
                st.dataframe(rows, use_container_width=True)

            details = []
            if resp.get("filtres"):
                details.append(f"filtres : {resp['filtres']}")
            if resp.get("group_by"):
                details.append(f"répartition par : {resp['group_by']}")
            if resp.get("erreurs"):
                details.append(f"avertissements : {resp['erreurs']}")
            if details:
                st.caption(" | ".join(details))
    elif submit:
        st.warning("Saisissez une question.")


# ─── mode INCIDENTS v2 — RECOMMANDATION ──────────────────────────────────────

def render_v2_reco() -> None:
    st.caption("Décrivez un événement qui vient de se produire : l'assistant retrouve "
               "les incidents similaires et vous montre les actions (préventives, "
               "correctives, à chaud) qui avaient été prises.")
    _show_examples(EX_V2_RECO, "sv2reco")
    st.markdown("---")

    question = _question_input("Décrivez l'événement : lieu, matériel, ce qui s'est passé…", "sv2reco")
    _, col2 = st.columns([4, 1])
    with col2:
        submit = _submit_button("Recommander", "sv2reco")

    if submit and question.strip():
        with st.spinner("Recherche d'incidents similaires + analyse des actions…"):
            t0 = time.time()
            resp = _post("/ask/incident-v2/recommande", {"question": question.strip(), "top_k": 10})
            dur = time.time() - t0

        if resp:
            st.success(f"Réponse en {dur:.0f}s")
            answer = resp.get("answer", "")
            if "pas trouvé d'incident" in answer:
                st.info(answer)
            else:
                st.markdown(f"### Recommandations\n{answer}")

            actions = resp.get("actions", [])
            st.markdown("### Actions relevées sur les incidents similaires")
            if actions:
                icones = {"préventive": "🛡️", "corrective": "🔧",
                          "curative": "🚑", "à chaud": "⚡"}
                for a in actions:
                    ic = icones.get(a.get("type_action", ""), "•")
                    statut = f" — {a['statut']}" if a.get("statut") else ""
                    resp_a = f" — resp. {a['responsable']}" if a.get("responsable") else ""
                    st.markdown(
                        f"{ic} **[{a.get('type_action', '?')}]** {a.get('titre', '?')}  \n"
                        f"&nbsp;&nbsp;&nbsp;&nbsp;*Fiches : {', '.join(a.get('fe_sources', []))}"
                        f"{statut}{resp_a}*"
                    )
            else:
                st.info("Aucune action documentée sur ces incidents similaires "
                        "(seuls ~10 % des incidents ont des actions structurées, "
                        "~37 % une action à chaud). Reformulez ou précisez la "
                        "description pour élargir la recherche.")

            incidents = resp.get("incidents_similaires", [])
            if incidents:
                with st.expander(f"Incidents similaires utilisés ({len(incidents)})",
                                 expanded=not actions):
                    st.dataframe(
                        [
                            {
                                "FE": i.get("numero_fe"),
                                "Date": str(i.get("date_evenement") or "")[:10],
                                "Sévérité": i.get("severite"),
                                "Titre": i.get("titre"),
                                "Score": i.get("score"),
                                "Actions": i.get("n_actions", 0),
                            }
                            for i in incidents
                        ],
                        use_container_width=True,
                        hide_index=True,
                    )
    elif submit:
        st.warning("Décrivez l'événement.")


# ─── mode INCIDENTS v2 — LISTE STRUCTURÉE ────────────────────────────────────

_LIST_COL_LABELS = {
    "numero_fe":       "FE",
    "titre":           "Titre",
    "severite":        "Sévérité",
    "classification":  "Classification",
    "etat":            "État",
    "date_evenement":  "Date événement",
    "date_creation":   "Date saisie",
    "action_corrective": "Action corrective",
    "resume_llm":      "Résumé",
}


def render_v2_list() -> None:
    st.caption(
        "Liste triée et filtrée d'incidents : précisez un nombre, un critère de tri "
        "(date de saisie, date d'événement, sévérité) et des filtres optionnels "
        "(sévérité, classification, année, condition lumineuse). "
        "Résultat immédiat — pas besoin d'attendre le LLM."
    )
    _show_examples(EX_V2_LIST, "sv2lst")
    st.markdown("---")

    question = _question_input(
        "Ex : Les 5 derniers incidents de sévérité élevée avec leurs actions correctives",
        "sv2lst",
    )
    _, col2 = st.columns([4, 1])
    with col2:
        submit = _submit_button("Lister", "sv2lst")

    if submit and question.strip():
        with st.spinner("Construction de la liste…"):
            t0 = time.time()
            resp = _post("/ask/incident-v2/list", {"question": question.strip(), "top_k": 5})
            dur = time.time() - t0

        if resp:
            count = resp.get("count", 0)
            records = resp.get("records", [])
            answer = resp.get("answer", "")

            if "comptage" in answer or count == 0 and not records:
                st.warning(answer)
            else:
                st.success(f"Réponse en {dur:.0f}s")
                st.markdown(f"### Réponse\n{answer}")

                if records:
                    st.markdown(f"### {count} incident(s)")
                    display_rows = [
                        {
                            _LIST_COL_LABELS.get(k, k): (v[:120] + "…" if isinstance(v, str) and len(v) > 120 else v)
                            for k, v in row.items()
                            if v is not None
                        }
                        for row in records
                    ]
                    st.dataframe(display_rows, use_container_width=True, hide_index=True)

                meta_parts = []
                sort_labels = {
                    "date_creation":   "date de saisie",
                    "date_evenement":  "date d'événement",
                    "severite":        "sévérité",
                }
                meta_parts.append(
                    f"Tri : {sort_labels.get(resp.get('sort_by',''), resp.get('sort_by',''))} "
                    f"({'↓ desc' if resp.get('order') == 'desc' else '↑ asc'}), "
                    f"limite {resp.get('limit', '?')}"
                )
                filters = resp.get("filters_applied", {})
                if filters:
                    meta_parts.append(f"Filtres : {filters}")
                st.caption(" | ".join(meta_parts))
    elif submit:
        st.warning("Saisissez une question.")


# ─── modes hérités ────────────────────────────────────────────────────────────

def render_legacy() -> None:
    st.caption("Recherche sur les incidents anciens (100 incidents, format legacy).")
    _show_examples(EX_LEGACY, "sleg")
    st.markdown("---")
    top_k = st.slider("Chunks à récupérer", 1, 20, DEFAULT_TOP_K, key="sl_leg")
    question = _question_input("Ex : Quels incidents impliquent un mauvais positionnement ?", "sleg")
    _, col2 = st.columns([4, 1])
    with col2:
        submit = _submit_button("Poser la question", "sleg")
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
    top_k = st.slider("Chunks à récupérer", 1, 20, DEFAULT_TOP_K, key="sl_tck")
    question = _question_input("Ex : Tickets critiques non résolus sur les modules métier ?", "stck")
    _, col2 = st.columns([4, 1])
    with col2:
        submit = _submit_button("Poser la question", "stck")
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
        onglets = st.tabs([
            "🧭 Question libre", "💡 Recommandation", "🔍 Recherche sémantique",
            "📊 Agrégation", "📋 Liste structurée", "🛠️ Actions", "🏢 Entités",
        ])
        with onglets[0]:
            render_v2_query()
        with onglets[1]:
            render_v2_reco()
        with onglets[2]:
            render_v2_search()
        with onglets[3]:
            render_v2_stats()
        with onglets[4]:
            render_v2_list()
        with onglets[5]:
            render_v2_actions()
        with onglets[6]:
            render_v2_entity()

    elif mode == "Tickets":
        render_tickets()

    else:
        render_legacy()


if __name__ == "__main__":
    main()
