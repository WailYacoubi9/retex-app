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
EX_V2_ANALYSTE = [
    "Les incidents de nuit sont-ils proportionnellement plus graves que ceux de jour ?",
    "Quel type d'événement a la plus forte proportion d'incidents graves ?",
    "Parmi les incidents de nuit, quel type est le plus fréquent ?",
    "Quelle compagnie est la plus impliquée dans les collisions aviaires ?",
    "Les incursions de piste augmentent-elles ces dernières années ?",
    "Quelle combinaison type × lieu × phase revient le plus sur les incidents graves ?",
    "Répartition des actions par type ?",
    "Raconte les incidents de dégivrage",
]
# Exemples pour l'onglet AUTOMATIQUE : un par voie, choisis pour bien router.
EX_V2_AUTO = [
    "Combien d'incidents de gravité 4 - élevé ?",              # -> agrégation
    "Répartition des incidents par sévérité",                  # -> agrégation
    "Raconte les incidents de collision aviaire",              # -> recherche
    "Que s'est-il passé lors des incidents de dégivrage ?",    # -> recherche
    "Combien d'actions correctives en cours ?",                # -> actions
    "Quelles actions en retard sur leur échéance ?",           # -> actions
    "Un camion a refusé la priorité à un avion au repoussage", # -> recommandation
    "Les incidents de nuit sont-ils proportionnellement plus graves ?",  # -> analyse
    "Les collisions aviaires suivent-elles une saisonnalité ?",          # -> analyse
    "Quels types restent sans action corrective ?",                      # -> analyse
    "Parmi les incidents de nuit, quel type est le plus fréquent ?",     # -> analyse (conditionnel)
    "Que signifie SSLIA ?",                                    # -> sigle (déterministe)
    "Montre-moi la fiche FNE/AA/NNNN",                         # -> fiche (déterministe)
    "Quel agent fait le plus d'erreurs ?",                     # -> refus culture juste
    "Quel est le coût total des incidents ?",                  # -> abstention
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


def _get(path: str, params: dict | None = None, timeout: float = 15.0) -> dict | None:
    try:
        r = httpx.get(f"{API_URL}{path}", params=params or {}, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:  # noqa: BLE001
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
        c1, c2 = st.sidebar.columns(2)
        c1.metric("Incidents", f"{neo4j.get('incidents', 0):,}")
        c2.metric("Chunks Qdrant", f"{qdrant.get('points_count', 0):,}")
        st.sidebar.caption(f"dont enrichis (resume_llm) : {neo4j.get('incidents_enriched', 0):,}")
        st.sidebar.caption(f"API : {API_URL}")

    st.sidebar.divider()
    st.sidebar.caption(
        "ℹ️ Vos questions et les réponses sont enregistrées (anonymisées : aucun "
        "nom ni identifiant) pour évaluer et améliorer l'assistant. "
        "Le REX analyse le système, jamais les personnes.")

    # Mode démo : vue épurée pour la présentation (diagnostics repliés)
    st.session_state["mode_demo"] = st.sidebar.toggle(
        "🎓 Mode démo (vue épurée)", value=st.session_state.get("mode_demo", False))


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
    if "culture juste" in answer:
        st.warning("🛡️ **Refus culture juste** — le REX analyse le système, pas les individus.")
        st.info(answer)
        return
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
               "liées aux incidents (statut, type, efficacité…).")
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
            if "culture juste" in answer:
                st.warning("🛡️ **Refus culture juste** — le REX analyse le système, pas les individus.")
                st.info(answer)
            elif "ne semble pas porter" in answer or "aucun critère reconnu" in answer:
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
                    resp_a = ""  # responsable retiré (culture juste : jamais de personne)
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


# ─── mode INCIDENTS v2 — AUTOMATIQUE (routeur) ───────────────────────────────

_LIBELLES_VOIE = {
    "agregation": "📊 Agrégation / chiffres",
    "recherche": "🔍 Recherche sémantique",
    "actions": "🛠️ Actions correctives",
    "recommandation": "💡 Recommandation",
    "abstention": "🚫 Abstention (hors périmètre)",
    "analyse": "🧮 Analyse (décomposé + calculé)",
    "sigle": "📖 Sigle (glossaire validé)",
    "fiche": "📄 Fiche exacte (par numéro FE)",
}

# Une ligne narrative par voie (mode démo + pédagogie du routage).
_NARRATIF_VOIE = {
    "agregation": "question → fiche typée (LLM) → Cypher déterministe → chiffre exact",
    "recherche": "question → embeddings → passages pertinents → réponse rédigée sur sources",
    "actions": "question → filtres typés → Cypher sur les actions de traitement",
    "recommandation": "événement décrit → incidents similaires → actions qui avaient été prises",
    "abstention": "hors périmètre ou refus : l'assistant ne devine pas",
    "analyse": "question → plan analytique → opérateurs Cypher → synthèse verrouillée sur les faits",
    "sigle": "lookup direct dans le glossaire validé — zéro LLM",
    "fiche": "lookup exact par numéro FE — zéro LLM",
}


def render_v2_auto() -> None:
    demo = st.session_state.get("mode_demo", False)
    if demo:
        st.caption("Un seul point d'entrée : l'assistant choisit lui-même la bonne voie "
                   "(chiffres exacts, analyse, recherche, actions, recommandation) — "
                   "et refuse ce qui sort du périmètre ou touche aux personnes.")
        scores = _get("/eval/scores") or {}
        kcols = st.columns(len(scores.get("scores", [])) or 1)
        for c, sc in zip(kcols, scores.get("scores", [])):
            if sc.get("pct") is not None:
                c.metric(sc["grille"], f"{sc['ok']}/{sc['n']}")
    else:
        st.caption("Point d'entrée UNIQUE + INSPECTION : posez une question, et voyez TOUT — "
                   "la voie choisie, les sources et leurs scores, l'interprétation du LLM, "
                   "le Cypher exécuté, et pourquoi certaines sources sont écartées.")
    _show_examples(EX_V2_AUTO, "sv2auto")
    st.markdown("---")
    question = _question_input("Ex : Quelles actions correctives pour les incidents FOD ?", "sv2auto")
    col1, col2 = st.columns([4, 1])
    with col1:
        top_k = st.slider("top_k (voie recherche)", 1, 15, 8, key="topk_sv2auto")
    with col2:
        submit = _submit_button("Poser la question", "sv2auto")

    if submit and question.strip():
        with st.spinner("Routage + réponse…"):
            t0 = time.time()
            resp = _post("/ask/incident-v2/auto", {"question": question.strip(), "top_k": top_k})
            dur = time.time() - t0
        if not resp:
            return

        voie = resp.get("voie_choisie", "?")
        diag = resp.get("diagnostics", {})
        refus_personne = (diag.get("pre_routeur") == "refus_personne"
                          or diag.get("statut_voie") == "refus_personne")
        st.success(f"Réponse en {dur:.0f}s")
        if refus_personne:
            st.warning("🛡️ **Refus culture juste** — le REX analyse le système, "
                       "jamais les individus. L'assistant refuse toute question "
                       "rattachée à une personne et propose le niveau organisationnel.")
        st.info(f"**Voie choisie automatiquement : {_LIBELLES_VOIE.get(voie, voie)}**  \n"
                f"_{_NARRATIF_VOIE.get(voie, '')}_")
        st.markdown(f"### Réponse\n{resp.get('answer', '')}")

        # Historique de session (les résultats ne disparaissent plus au rerun)
        st.session_state.setdefault("historique_auto", []).append(
            {"question": question.strip(), "voie": voie,
             "temps_ms": diag.get("temps_ms"), "answer": resp.get("answer", "")[:400]})

        st.markdown("---")
        st.markdown("### 🔬 Sous le capot" if demo else "## 🔬 Diagnostics")

        # 1) Routage (routeur LLM ou voie déterministe pré-routeur)
        with st.expander("🧭 Routage — pourquoi cette voie", expanded=not demo):
            if diag.get("pre_routeur"):
                st.write(f"**Voie déterministe pré-routeur :** `{diag['pre_routeur']}` "
                         "(motif non ambigu — le routeur LLM n'est pas sollicité)")
            else:
                r = diag.get("routeur", {})
                st.write(f"**Capacité choisie :** `{r.get('capacite')}`")
                st.write(f"**Justification du routeur :** {r.get('justification', '')}")
                if r.get("garde_fou"):
                    st.write(f"**Garde-fou appliqué :** {r['garde_fou']}")
            st.write(f"**Justification de la réponse :** {resp.get('justification', '')}")
            if diag.get("statut_voie"):
                st.caption(f"Statut de la voie : `{diag['statut_voie']}`")
            if diag.get("numero_fe"):
                st.caption(f"Fiche : {diag['numero_fe']}")
            if diag.get("modele_reponse"):
                st.caption(f"Modèle de génération : {diag['modele_reponse']}")
            st.caption(f"Temps total : {diag.get('temps_ms', '?')} ms")

        # 2) Sources & scores — recherche
        if voie == "recherche":
            n_ret = diag.get("n_incidents_retenus", 0)
            with st.expander(f"📄 Sources récupérées & scores ({n_ret} retenues)", expanded=not demo):
                st.caption(
                    f"Seuil de pertinence : score ≥ {diag.get('seuil_score', '?')}. "
                    f"{diag.get('n_chunks_recuperes', 0)} passages récupérés → "
                    f"{n_ret} incidents retenus. Les fiches SOUS le seuil sont écartées "
                    "(elles n'apparaissent pas ci-dessous)."
                )
                if diag.get("aucune_source_pertinente"):
                    st.warning("Aucune source au-dessus du seuil → le LLM refuse de répondre "
                               "sur des sources non pertinentes.")
                for s in diag.get("sources", []):
                    st.markdown(f"**{s['numero_fe']}** — {s['titre']}  ·  score **{s['score']}**")
                    st.caption(f"champs qui ont matché : {', '.join(s['champs_correspondants']) or '—'}")
                    if s.get("resume"):
                        st.caption(f"résumé : {s['resume']}")

        # 2bis) Incidents similaires — recommandation
        if voie == "recommandation":
            with st.expander("📄 Incidents similaires trouvés (+ scores)", expanded=not demo):
                for s in diag.get("incidents_similaires", []):
                    st.markdown(f"**{s.get('numero_fe', '?')}** — {s.get('titre', '')}  ·  "
                                f"score **{s.get('score', '?')}**")
                st.caption(f"{diag.get('n_actions_recommandees', 0)} actions recommandées extraites.")

        # 3) Interprétation LLM + Cypher — agrégation
        if voie == "agregation":
            with st.expander("🧠 Interprétation par le LLM (ce qu'il a compris)", expanded=not demo):
                st.caption("Filtres et intention extraits de votre question par le LLM.")
                st.json(diag.get("interpretation_llm") or {})
            if diag.get("cypher_execute"):
                with st.expander("⚙️ Requête Cypher exécutée (calcul déterministe)"):
                    st.code(diag["cypher_execute"], language="cypher")
                    st.caption(f"Résultat brut : {diag.get('resultat_brut')}")

        # 3bis) Plan analytique + faits calculés — analyse
        if voie == "analyse":
            ana = diag.get("analyste", {})
            plan = ana.get("plan", {})
            with st.expander("🧮 Plan analytique (décomposition de la question)", expanded=not demo):
                st.caption("L'agent choisit un opérateur et les dimensions, dans un vocabulaire fermé.")
                st.write(
                    f"**Opérateur :** `{plan.get('operateur')}`"
                    + (f"  ·  **dimension :** `{plan.get('dimension')}`" if plan.get('dimension') else "")
                    + (f"  ·  **cible :** `{plan.get('cible')}`" if plan.get('cible') else "")
                    + (f"  ·  **type :** `{plan.get('filtre_type')}`" if plan.get('filtre_type') else "")
                    + (f"  ·  **sévérité :** `{plan.get('filtre_severite')}`" if plan.get('filtre_severite') else "")
                    + (f"  ·  **granularité :** `{plan.get('granularite')}`" if plan.get('granularite') else "")
                )
            with st.expander("🔢 Faits calculés (déterministes, base = vérité)", expanded=not demo):
                st.caption("Chiffres calculés par la base, avant la synthèse du LLM — c'est la preuve.")
                faits = ana.get("faits")
                if isinstance(faits, list):
                    st.dataframe(faits, use_container_width=True)
                elif isinstance(faits, dict) and "serie" in faits:
                    st.write(f"**Tendance :** {faits.get('direction', '')}")
                    st.dataframe(faits["serie"], use_container_width=True)
                elif isinstance(faits, dict):
                    st.json(faits)

        # 4) Filtres + actions — actions
        if voie == "actions":
            with st.expander("🧠 Filtres interprétés + actions trouvées", expanded=not demo):
                st.write("**Filtres :**", diag.get("filtres_interpretes", {}))
                st.caption(f"Total : {diag.get('total', '?')}")
                if diag.get("sources"):
                    st.dataframe(diag["sources"], use_container_width=True)

        # 5) Prompts exacts envoyés au LLM
        r = diag.get("routeur", {})
        if r.get("prompt") or diag.get("prompt_parseur") or diag.get("prompt_llm"):
            with st.expander("📝 Prompt(s) exact(s) envoyé(s) au LLM"):
                if r.get("prompt"):
                    st.markdown("**1. Prompt du routeur** (classification de la question) :")
                    st.code(r["prompt"])
                if diag.get("prompt_parseur"):
                    st.markdown("**2. Prompt du parseur** (interprétation → filtres/spec) :")
                    st.code(diag["prompt_parseur"])
                if diag.get("prompt_llm"):
                    st.markdown("**2. Prompt de génération** (contexte + question envoyés au LLM) :")
                    st.code(diag["prompt_llm"])

        # 6) JSON brut complet
        with st.expander("🗂 Diagnostics bruts (JSON complet)"):
            st.json(diag)
    elif submit:
        st.warning("Saisissez une question.")

    # 7) Avis utilisateur sur la DERNIÈRE réponse (persiste aux reruns Streamlit :
    # les boutons vivent HORS du bloc submit, pilotés par l'historique de session)
    hist_fb = st.session_state.get("historique_auto", [])
    if hist_fb:
        dernier = hist_fb[-1]
        st.markdown("---")
        st.markdown(f"**📮 Cette réponse vous a-t-elle été utile ?**  \n_{dernier['question'][:80]}_")
        cfb1, cfb2, cfb3 = st.columns([1, 1, 3])
        cle_fb = f"fb_{len(hist_fb)}"
        if st.session_state.get(cle_fb):
            st.caption("Merci pour votre avis ✔️")
        else:
            commentaire = cfb3.text_input("Commentaire (optionnel)", key=f"fbc_{len(hist_fb)}",
                                          label_visibility="collapsed",
                                          placeholder="Commentaire (optionnel)")
            if cfb1.button("👍 Utile", key=f"fbp_{len(hist_fb)}", use_container_width=True):
                _post("/feedback", {"question": dernier["question"], "voie": dernier["voie"],
                                    "utile": True, "commentaire": commentaire})
                st.session_state[cle_fb] = True
                st.rerun()
            if cfb2.button("👎 Pas utile", key=f"fbm_{len(hist_fb)}", use_container_width=True):
                _post("/feedback", {"question": dernier["question"], "voie": dernier["voie"],
                                    "utile": False, "commentaire": commentaire})
                st.session_state[cle_fb] = True
                st.rerun()

    # 8) Historique de la session (persiste entre les questions)
    hist = st.session_state.get("historique_auto", [])
    if hist:
        with st.expander(f"🕘 Historique de session ({len(hist)} question(s))"):
            for h in reversed(hist[-15:]):
                st.markdown(f"**{h['question']}**  →  "
                            f"{_LIBELLES_VOIE.get(h['voie'], h['voie'])} "
                            f"· {h.get('temps_ms', '?')} ms")
                st.caption(h["answer"])


# ─── mode INCIDENTS v2 — ANALYSTE (couche agentique, accès direct) ───────────

def render_v2_analyste() -> None:
    st.caption("Accès DIRECT à la couche analyste (celle qui intercepte les questions "
               "analytiques dans l'onglet Automatique) : plan fermé → opérateurs Cypher "
               "déterministes → synthèse verrouillée sur les faits. Le dernier exemple "
               "montre un REFUS volontaire (question non analytique → l'analyste décline).")
    _show_examples(EX_V2_ANALYSTE, "sv2ana")
    st.markdown("---")
    question = _question_input("Ex : Quel lieu est proportionnellement le plus grave ?", "sv2ana")
    _, col2 = st.columns([4, 1])
    with col2:
        submit = _submit_button("Analyser", "sv2ana")
    if submit and question.strip():
        with st.spinner("Plan + calculs + synthèse…"):
            t0 = time.time()
            resp = _post("/ask/incident-v2/analyste", {"question": question.strip()})
            dur = time.time() - t0
        if not resp:
            return
        st.success(f"Réponse en {dur:.0f}s")
        if resp.get("status") != "ok":
            st.warning(f"L'analyste DÉCLINE (statut : `{resp.get('status')}`) — la question "
                       "suivrait sa voie normale dans l'onglet Automatique.")
            if resp.get("plan"):
                st.json(resp["plan"])
            st.info(resp.get("answer", ""))
            return
        st.markdown(f"### Réponse\n{resp.get('answer', '')}")
        plan = resp.get("plan") or {}
        with st.expander("🧮 Plan analytique", expanded=True):
            st.write(f"**Opérateur :** `{plan.get('operateur')}`"
                     + (f"  ·  **entité :** `{plan.get('entite')}`" if plan.get("entite") else "")
                     + (f"  ·  **dimension :** `{plan.get('dimension')}`" if plan.get("dimension") else "")
                     + (f"  ·  **cible :** `{plan.get('cible')}`" if plan.get("cible") else ""))
            if plan.get("conditions"):
                st.write("**Conditions (« parmi les X… ») :**", plan["conditions"])
            if plan.get("dimensions"):
                st.write("**Dimensions croisées :**", plan["dimensions"])
        with st.expander("🔢 Faits calculés (déterministes)", expanded=True):
            faits = resp.get("faits")
            if isinstance(faits, list):
                st.dataframe(faits, use_container_width=True)
            elif faits is not None:
                st.json(faits)
    elif submit:
        st.warning("Saisissez une question.")


# ─── mode INCIDENTS v2 — GRILLES (console de mesure) ─────────────────────────

_LIBELLES_GRILLE = {
    "couvrant": "🧱 Couvrant (32 Q, 5 voies)",
    "multihop": "🕸️ Multi-hop path-grounded (20 Q)",
    "analyste": "🧮 Analyste — grille registre",
    "metier": "🎓 Métier curé main (50 Q en cours)",
    "culture_juste": "🛡️ Culture juste (pièges + zéro fuite)",
    "reco": "💡 Reco — hit@k retrieval",
}


def render_v2_grilles() -> None:
    st.caption("Console de mesure : derniers scores des grilles golden, lancement d'un run, "
               "et corpus de capture (triplets question/réponse/contexte, masqués).")

    data = _get("/eval/scores")
    if data:
        if data.get("run_en_cours"):
            st.info("⏳ Un run de grille est en cours — les scores se mettront à jour à la fin.")
        cols = st.columns(len(data.get("scores", [])) or 1)
        for c, s in zip(cols, data.get("scores", [])):
            nom = s.get("grille", "?")
            if s.get("pct") is not None:
                c.metric(_LIBELLES_GRILLE.get(nom, nom), f"{s['ok']}/{s['n']}",
                         delta=f"{s['pct']} %")
            else:
                c.metric(_LIBELLES_GRILLE.get(nom, nom), "—")
            if s.get("date"):
                c.caption(s["date"][:16].replace("T", " "))

    st.markdown("---")
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        grille = st.selectbox("Grille à (re)lancer", list(_LIBELLES_GRILLE),
                              format_func=lambda g: _LIBELLES_GRILLE[g], key="sel_grille")
    with c2:
        if st.button("▶️ Lancer", use_container_width=True, key="btn_run_grille"):
            r = _post("/eval/run", {"grille": grille})
            if r:
                st.success("Run lancé" if r.get("status") == "lance"
                           else "Un run est déjà en cours")
    with c3:
        if st.button("🔄 Rafraîchir", use_container_width=True, key="btn_refresh_grilles"):
            st.rerun()

    log = _get("/eval/log")
    if log and log.get("log"):
        with st.expander("📜 Log du dernier run" + (" (en cours…)" if log.get("run_en_cours") else ""),
                         expanded=bool(log.get("run_en_cours"))):
            st.code("\n".join(log["log"]))

    fb = _get("/feedback/stats")
    if fb and fb.get("total"):
        st.markdown("---")
        st.markdown(f"### 📮 Avis utilisateurs : **{fb['utiles']}/{fb['total']} utiles "
                    f"({fb.get('pct')} %)**")
        st.json(fb.get("par_voie", {}))

    st.markdown("---")
    st.markdown("### 📥 Capture récente (interactions réelles, masquées)")
    fj = st.radio("Filtre", ["toutes", "jugeables (texte libre)", "non jugeables"],
                  horizontal=True, key="filtre_capture")
    params: dict = {"n": 30}
    if fj != "toutes":
        params["jugeable"] = fj.startswith("jugeables")
    cap = _get("/capture/recent", params)
    if cap:
        st.caption(f"{cap.get('total', 0)} interactions capturées au total — le corpus des "
                   "triplets pour le juge, et la source « questions réelles » du golden.")
        if cap.get("interactions"):
            st.dataframe(cap["interactions"], use_container_width=True)


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
            "🎯 Automatique", "🧭 Question libre", "💡 Recommandation", "🔍 Recherche sémantique",
            "📊 Agrégation", "📋 Liste structurée", "🛠️ Actions", "🏢 Entités",
            "🧮 Analyste", "🧪 Grilles",
        ])
        with onglets[0]:
            render_v2_auto()
        with onglets[1]:
            render_v2_query()
        with onglets[2]:
            render_v2_reco()
        with onglets[3]:
            render_v2_search()
        with onglets[4]:
            render_v2_stats()
        with onglets[5]:
            render_v2_list()
        with onglets[6]:
            render_v2_actions()
        with onglets[7]:
            render_v2_entity()
        with onglets[8]:
            render_v2_analyste()
        with onglets[9]:
            render_v2_grilles()

    elif mode == "Tickets":
        render_tickets()

    else:
        render_legacy()


if __name__ == "__main__":
    main()
