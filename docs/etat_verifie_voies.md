# État vérifié des voies de l'assistant Incident v2

> Audit réalisé le **2026-07-09**.
> Serveur : `172.16.6.10`, API port 8000 · Frontend Streamlit port 8501.
> **GPU OPÉRATIONNEL** : NVIDIA L4, 926 MiB / 23 034 MiB — Ollama actif sur GPU.
> Modèles : `bge-m3` (embed, vram 664 MB), `qwen2.5:7b` (generate, chargé à la demande).

---

## Tableau récapitulatif

| Endpoint | Existe | Dépend d'Ollama | Test 1 | Test 2 | Exactitude vs Cypher | Garde-fou | Statut global |
|---|---|---|---|---|---|---|---|
| `GET /health` | ✓ | Non | healthy, 3 services up | — | N/A | N/A | **VÉRIFIÉ OK** |
| `GET /stats` | ✓ | Non | incidents=0 (bug label) | — | ✗ bug | N/A | **BUG** (label :Incident) |
| `POST /ask/incident-v2` | ✓ | Phrasing seul (embed=bge-m3 + generate) | 18 s — 5 sources aviaire, réponse cohérente Lyon | 10 s — FNE/26/0240 trappe carburant score 0.798 | N/A (sémantique) | ✓ "pas trouvé" si hors corpus | **VÉRIFIÉ OK** |
| `POST /ask/incident-v2/stats` | ✓ | Parsing (generate_structured) + phrasing | 5 s — 1 021 incidents 2025 ✅ | 4 s — répartition sévérité 9 176 total ✅ | ✅ exact vs Cypher | ⚠️ reformule silencieusement une question-liste en count | **VÉRIFIÉ OK** |
| `POST /ask/incident-v2/list` | ✓ | **Parsing : REGEX** (0 ms) · Phrasing : 90 s timeout + fallback | 13 s — 5 records date_creation DESC ✅ | voir détail regex ci-dessous | ✅ exact vs Cypher (3/3) | ✓ count → redirect · ✗ hors-sujet → liste défaut | **VÉRIFIÉ OK** (limites regex) |
| `POST /ask/incident-v2/entity` | ✓ | Phrasing seul (recherche Neo4j sans LLM) | 7 s — EASYJET 587 incidents ✅ | 8 s — "Air France" → résultats incohérents ✗ | EASYJET ✅ / Air France ✗ | — | **BUG PARTIEL** (matching multi-tokens) |
| `POST /ask/incident-v2/actions` | ✓ | Parsing (generate_structured) + phrasing | 29 s — 114 actions correctives FOD ✅ | 4 s — 532 préventives ✅ | ✅ exact vs Cypher | — | **VÉRIFIÉ OK** |
| `POST /ask/incident-v2/recommande` | ✓ | Embed (bge-m3) + phrasing (generate) | 23 s — 20 incidents similaires, 13 actions, réponse structurée | — | N/A | ✓ "aucune action" si vide | **VÉRIFIÉ OK** |
| `POST /ask/incident-v2/query` | ✓ | Parsing (generate_structured) + phrasing (count = déterministe) | 4 s — 21 blessés ✅ | 5 s — 250 incidents nuit 2025 ✅ avec répartition mensuelle | ✅ exact vs Cypher | ✓ "pas réussi à interpréter" si hors-champ | **VÉRIFIÉ OK** |
| `POST /ask` | ✓ | Embed + phrasing | 2 s — 0 sources, "pas trouvé" (corpus vide) | — | N/A | ✓ | **VÉRIFIÉ OK** (corpus legacy vide) |
| `POST /ask/tickets` | ✓ | Embed + phrasing | 1 s — 0 sources, "Aucun ticket trouvé" | — | N/A | ✓ | **VÉRIFIÉ OK** (corpus vide) |
| `POST /ask/info-securite` | ✓ | Embed + phrasing | 1 s — 0 sources, "pas trouvé" | — | N/A | ✓ | **VÉRIFIÉ OK** (corpus vide) |

---

## Données de référence Cypher (vérification en lecture seule)

```
IncidentSecu total        : 9 191   (GET /stats retourne 0 — bug label :Incident)
  dont année 2025         : 1 021   → /stats Q1 = 1 021 ✅
  dont année 2024         : 1 065
  dont sévérité élevée    :   164   → /stats Q2 = 164 ✅ (dans répartition)
  dont présence blessés   :    21   → /query Q1 = 21 ✅
  dont nuit 2025          :   250   → /query Q2 = 250 ✅
Actions (:Action)         : 1 135 nœuds
Relations A_POUR_ACTION   : 1 298
  corrective              :   763
  préventive (avec accent): 532     → /actions Q2 = 532 ✅
  curative                :     3
FOD correctives via titre :   114   → /actions Q1 = 114 ✅
EASYJET via IMPLIQUE_COMPAGNIE : 587 → /entity Q1 = 587 ✅
5 derniers date_creation  : FNE/26/0243, 0244, 0241, 0242, 0240 → /list Q1 = idem ✅
Incidents sérieux ASC     : FNE-2008ADL060, 058, 382, 540, 544 → /list ✅
```

---

## Détail /list — Couverture linguistique du parseur REGEX

Le parsing est **100 % regex déterministe** (aucun appel LLM pour cette étape).
Instantané. Conséquences : robuste sur CPU/GPU, mais couverture lexicale limitée.

| Tournure testée | limit | sort_by | order | f_severite | f_classification | Verdict |
|---|---|---|---|---|---|---|
| `"Les 5 derniers incidents par date de création"` | 5 | date_creation | desc | — | — | ✅ |
| `"les 3 incidents les plus récents de sévérité élevée"` | 3 | date_evenement | desc | 4 - élevé | — | ✅ |
| `"les incidents sérieux les plus anciens"` | 5 | date_creation | asc | — | Incident sérieux | ✅ |
| `"donne-moi les cinq incidents les plus récents"` | **5** (défaut) | date_creation | desc | — | — | ⚠️ "cinq" non reconnu — marche par coïncidence (défaut=5) |
| `"affiche les trois dernières fiches graves"` | **5** (défaut) | date_creation | desc | — | — | ✗ "trois" non reconnu → limit=5 ; "graves" non reconnu → pas de filtre sévérité |
| `"les 10 derniers incidents de nuit en 2024 avec résumé"` | 10 | date_creation | desc | — | — | ✅ f_condition_lumineuse=Nuit, f_annee=2024 |

**Règle du parseur :** nombres en lettres non reconnus (trois, cinq, dix…), adjectifs au pluriel non reconnus (graves, récents, anciens — vs grave, récent, ancien).
**Couverture réelle :** tournures avec chiffres arabes et mots-clés au singulier = 100 %. Tournures familières avec mots en lettres = partielle.

---

## Détail /entity — Bug de matching multi-tokens

- `"EASYJET"` → 587 incidents via `IMPLIQUE_COMPAGNIE` ✅
- `"Air France"` → retourne TypeEvenement "Collision aviaire" (1 683) et Notifiant "Agent aire de trafic" — **pas Air France** ✗

Cause probable : le fuzzy-matching sur plusieurs tokens ("Air" + "France") sélectionne d'abord les entités les plus fréquentes, pas celles dont le nom correspond. Bug documenté. En base : `(:IncidentSecu)-[:IMPLIQUE_COMPAGNIE]->(:Compagnie {nom: "AIR FRANCE"})` existe avec 270 incidents.

---

## Détail /stats — Garde-fou limité

`"Donne-moi les 5 derniers incidents graves"` → /stats interprète silencieusement comme count sur sévérité=élevé (164). Pas de message "ce n'est pas une question d'agrégation". L'utilisateur ne comprend pas pourquoi il reçoit 164 au lieu d'une liste.

---

## GET /stats — Bug label connu

Requête actuelle : `MATCH (i:Incident)` → 0 résultats (ancien label).
Devrait être : `MATCH (i:IncidentSecu)` → 9 191. Qdrant : 27 195 chunks ✅ (non affecté).

---

## Voies démontrables au client (GPU up)

| Voie | Ce qu'on peut montrer | Temps typique |
|---|---|---|
| `GET /health` | Santé des 3 services | < 1 s |
| `/ask/incident-v2` | RAG sémantique libre, sources citées | 10–20 s |
| `/ask/incident-v2/stats` | Comptages et répartitions fiables | 4–8 s |
| `/ask/incident-v2/list` | Liste triée/filtrée, résultat immédiat | 5–15 s |
| `/ask/incident-v2/actions` | Actions correctives/préventives par contexte | 4–30 s |
| `/ask/incident-v2/recommande` | Incidents similaires + actions recommandées | 20–30 s |
| `/ask/incident-v2/query` | Tout champ, count/liste/répartition | 4–8 s |
| `/ask/incident-v2/entity` | Entités nommées exactes (EASYJET, HOP…) | 5–10 s |

## Voies à NE PAS montrer telles quelles / Points de vigilance

| Voie / Point | Risque | Recommandation |
|---|---|---|
| `GET /stats` | Affiche 0 incidents (bug label) | Cacher ou corriger la requête |
| `/entity` avec "Air France" ou noms composés | Résultats incohérents | Tester avec noms courts exacts (EASYJET, HOP…) |
| `/list` avec tournures familières | "les trois dernières fiches graves" → 5 incidents sans filtre | Prévenir l'utilisateur ou améliorer le parseur |
| `/ask/info-securite`, `/ask/tickets`, `/ask (legacy)` | Corpus vide → "aucun résultat" systématique | Ne pas montrer, corpus à re-ingérer |

## À corriger avant point client

| Priorité | Problème | Voie | Fix |
|---|---|---|---|
| 🔴 | GET /stats compte :Incident (0 résultats) | `/stats` | `MATCH (i:IncidentSecu)` |
| 🟠 | Entity: noms composés → mauvais résultats | `/entity` | Revoir le score de similarité / contraindre sur longueur min du token |
| 🟡 | /list: nombres en lettres non reconnus | `/list` | Ajouter dict `{"un":1,"deux":2,"trois":3,"quatre":4,"cinq":5,"dix":10}` |
| 🟡 | /list: pluriels de sévérité non reconnus (graves) | `/list` | `grave` → `graves?` dans regex |
| 🟡 | /list: hors-sujet → liste par défaut sans avertissement | `/list` | Détecter absence de mots-clés métier |
| 🟢 | /stats: pas de message si question non-agrégation | `/stats` | Ajouter détection intent liste → redirect |
