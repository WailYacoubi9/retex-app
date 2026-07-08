# État vérifié des voies de l'assistant Incident v2

> Audit réalisé le **2026-07-08**.
> Serveur : `172.16.6.10`, API port 8000.
> **GPU DOWN** (nvidia-smi : Unknown Error / No devices found).
> Modèles Ollama chargés sur **CPU** : `qwen2.5:7b` (vram=0), `bge-m3` (vram=0).
> Embed bge-m3 : **1,6 s** sur CPU. Generate qwen2.5:7b : **> 60 s** (bloquant).

---

## Tableau récapitulatif

| Endpoint | Existe | Dépend d'Ollama | Test 1 | Test 2 | Exactitude vs Cypher | Garde-fous | Statut global |
|---|---|---|---|---|---|---|---|
| `GET /health` | ✓ | ✗ | `healthy` — neo4j/qdrant/ollama up | — | N/A | N/A | **VÉRIFIÉ OK** |
| `GET /stats` | ✓ | ✗ | `incidents: 0` — bug label | — | N/A | N/A | **BUG** (label :Incident vs :IncidentSecu) |
| `POST /ask/incident-v2` | ✓ | Phrasing LLM (1200 s, pas de fallback) | timeout 20 s | — | — | — | **OK SAUF PHRASAGE** (GPU down) |
| `POST /ask/incident-v2/stats` | ✓ | Parsing LLM (generate_structured) + phrasing | timeout 20 s | — | — | — | **OK SAUF LLM** (GPU down) |
| `POST /ask/incident-v2/list` | ✓ | Phrasing LLM 90 s (fallback texte ✓) | 5 derniers DESC ✓ 5 records | 3 sévérité élevée ✓ filter OK | ✅ exact vs Cypher (3/3) | Count → redirect ✓ ; hors-sujet → liste par défaut ✗ | **VÉRIFIÉ OK** (avec limites regex) |
| `POST /ask/incident-v2/entity` | ✓ | Phrasing LLM (1200 s, pas de fallback) | timeout 20 s | — | — | — | **OK SAUF PHRASAGE** (GPU down) |
| `POST /ask/incident-v2/actions` | ✓ | Parsing LLM (generate_structured) + phrasing | timeout 20 s | — | — | — | **OK SAUF LLM** (GPU down) |
| `POST /ask/incident-v2/recommande` | ✓ | Phrasing LLM (1200 s, pas de fallback) | timeout 20 s | — | — | — | **OK SAUF PHRASAGE** (GPU down) |
| `POST /ask/incident-v2/query` | ✓ | Parsing LLM (generate_structured) — sauf count déterministe | timeout 60 s | — | — | — | **OK SAUF LLM** (GPU down) |
| `POST /ask` | ✓ | Phrasing LLM (1200 s, pas de fallback) | timeout 15 s | — | — | — | **OK SAUF PHRASAGE** (GPU down) |
| `POST /ask/tickets` | ✓ | Phrasing LLM (1200 s, pas de fallback) | timeout 15 s | — | — | — | **OK SAUF PHRASAGE** (GPU down) |
| `POST /ask/info-securite` | ✓ | Phrasing LLM (1200 s, pas de fallback) | timeout 15 s | — | — | — | **OK SAUF PHRASAGE** (GPU down) |

---

## Détail par voie

### /health (GET)
- **Statut** : healthy. Neo4j up, Qdrant up, Ollama up.
- Pas de dépendance LLM.

### GET /stats ← BUG CONNU
- Retourne `incidents: 0` car il compte le label `:Incident` (ancien) et non `:IncidentSecu`.
- Neo4j réel : **9 191** nœuds `:IncidentSecu` (vérifié Cypher).
- Qdrant : 27 195 chunks, dim 1024. ✓

### POST /ask/incident-v2 — Sémantique
- **Parsing** : aucun (embed direct bge-m3, 1,6 s CPU).
- **Phrasing** : `generate` qwen2.5:7b, timeout 1200 s, **pas de fallback**.
- **GPU down** : l'embed se fait en 1,6 s, puis la requête pend indéfiniment côté phrasing LLM.
- Résumés LLM enrichis sur le graphe : **5 158** fiches avec `resume_llm`, 4 033 `resume_skip`.

### POST /ask/incident-v2/stats — Agrégation
- **Parsing** : `generate_structured` (qwen2.5:7b) → AggregationSpec.
- **GPU down** : bloqué dès le parsing (> 60 s sur CPU, OLLAMA_TIMEOUT = 1200 s).
- Architecture correcte, Cypher déterministe, pas de fabrication de chiffres.

### POST /ask/incident-v2/list — Liste structurée ⭐
**Seule voie pleinement fonctionnelle sur CPU.**

- **Parsing** : **regex déterministe** (aucun appel LLM) — instantané.
- **Phrasing** : `generate` qwen2.5:7b avec **timeout 90 s + fallback texte structuré** → renvoie toujours une réponse.

#### Exactitude vs Cypher (contrôle en lecture seule) :
| Question | API /list | Cypher direct | Conforme |
|---|---|---|---|
| 5 derniers par date_creation DESC | FNE/26/0243, 0244, 0241, 0242, 0240 | Identique | ✅ |
| 3 plus récents sévérité "4 - élevé" par date_evenement | FNE SURT/22/0288 (NULL), FNE/26/0241, FNE/26/0096 | Identique | ✅ |
| Incidents sérieux les plus anciens (ASC) | FNE-2008ADL060, 058, 382, 540, 544 | Identique | ✅ |

> Note : FNE SURT/22/0288 apparaît en tête du tri DESC par date_evenement car `date_evenement = NULL` — comportement Cypher attendu (NULL > valeur en DESC). C'est un défaut de donnée, pas un bug code.

#### Limites du parseur regex :
| Tournure | Comportement actuel | Attendu | Verdict |
|---|---|---|---|
| `"les 5 derniers incidents graves"` | sort_by=date_creation, order=desc, limit=5, **f_severite manquant** | f_severite="4 - élevé" | ⚠️ "graves" (pluriel) non reconnu — fix : `graves?` |
| `"donne-moi les cinq incidents les plus récents"` | limit=5 (défaut) | limit=5 | ✓ par chance (défaut = 5) |
| `"affiche les trois dernières fiches graves"` | limit=5, f_severite=None | limit=3, f_severite="4 - élevé" | ✗ nombres en lettres non reconnus |
| `"les incidents de nuit en 2024"` | f_condition_lumineuse="Nuit", f_annee=2024 | idem | ✓ |
| `"les incidents sérieux les plus anciens"` | f_classification="Incident sérieux", order=asc | idem | ✓ |
| `"quel est le nom du directeur de l'aéroport"` | retourne 5 incidents par défaut | message d'erreur ou refus | ✗ **pas de garde-fou sémantique** |

**Couverture linguistique réelle** : les tournures avec des **chiffres arabes** et les **adjectifs au singulier** passent bien. Les **nombres en lettres** (trois, cinq, dix…) et les **adjectifs au pluriel** (graves, récents) ne sont pas reconnus par le regex.

#### Garde-fous /list :
- ✅ Question de comptage ("combien…") → message de redirect vers /stats, 0 records.
- ✗ Question hors-sujet (ex. "nom du directeur") → retourne une liste par défaut sans avertissement.

### POST /ask/incident-v2/entity — Entité
- **Parsing** : recherche fuzzy/keyword directe en Neo4j (pas de LLM).
- **Phrasing** : `generate` qwen2.5:7b, 1200 s, **pas de fallback**.
- **GPU down** : recherche Neo4j fonctionne, phrasing bloqué.
- Bug documenté : CONTAINS/OU sur plusieurs entités parfois incorrect.

### POST /ask/incident-v2/actions — Actions
- **Parsing** : `generate_structured` (ActionSpec, qwen2.5:7b) — bloqué CPU.
- **Données** : 1 298 relations A_POUR_ACTION : 763 correctives, **532 "préventive" (avec accent é)**, 3 curatives.
- ⚠️ Piège : toute requête filtrant par `type_action = "preventive"` (sans accent) retourne 0 résultats. La valeur en base est `"préventive"` avec accent.

### POST /ask/incident-v2/recommande — Recommandation
- **Parsing** : embed bge-m3 (1,6 s CPU ✓).
- **Phrasing** : `generate` qwen2.5:7b, 1200 s, **pas de fallback**.
- **GPU down** : retrieval Qdrant + Neo4j fonctionnel, phrasing bloqué.

### POST /ask/incident-v2/query — Moteur générique
- **Parsing** : `generate_structured` (QuerySpec) — bloqué CPU.
- **Phrasing count** : déterministe, pas de LLM (ex : "X incident(s) correspondent aux critères : …").
- **GPU down** : même les count bloquent car le parsing de la spec passe d'abord par LLM.
- Opérateurs supportés : `=`, `!=`, `contient`, `>=`, `<=`, `annee`, `mois`, `est_rempli`, `est_vide`.

### POST /ask, /ask/tickets, /ask/info-securite — Anciennes voies
- Toutes accèdent à Qdrant (embed bge-m3) + Neo4j puis `generate` LLM.
- **GPU down** : embed OK (1,6 s), puis phrasing bloqué (1200 s, pas de fallback).
- Pas de régression structurelle détectée : les routes sont enregistrées, les dépendances correctes.

---

## Données graphe (contrôle Cypher)

```
IncidentSecu       : 9 191  (0 via GET /stats → bug label)
  avec resume_llm  : 5 158
  avec resume_skip : 4 033
  avec action_corrective (champ) : 3 389
  avec date_evenement            : 9 189
  avec classification            : 9 176
Actions (:Action)  : 1 135 nœuds
Relations A_POUR_ACTION : 1 298
  corrective       : 763
  préventive       : 532  (accent sur é — piège de filtre)
  curative         :   3
Qdrant chunks      : 27 195, dim 1024
```

---

## Voies démontrables aujourd'hui (GPU down)

| Voie | Ce qu'on peut montrer |
|---|---|
| `/ask/incident-v2/list` | Listes triées/filtrées, fallback texte, redirect count. Instantané. |
| `/health` | Santé des services. |
| Données graphe | Cypher read-only sur neo4j (shell direct). |

## Voies à NE PAS montrer au client tant que GPU down

| Voie | Raison |
|---|---|
| `/ask/incident-v2` | Réponse en attente indéfiniment (1200 s timeout, pas de fallback) |
| `/ask/incident-v2/stats` | Parsing bloqué dès la 1re étape |
| `/ask/incident-v2/entity` | Phrasing bloqué |
| `/ask/incident-v2/actions` | Parsing bloqué |
| `/ask/incident-v2/recommande` | Phrasing bloqué |
| `/ask/incident-v2/query` | Parsing bloqué (même les count) |
| `/ask`, `/ask/tickets`, `/ask/info-securite` | Phrasing bloqué |

## Voies à corriger avant point client (indépendamment du GPU)

| Problème | Voie | Fix |
|---|---|---|
| GET /stats compte :Incident au lieu de :IncidentSecu | `/stats` | Modifier la requête Neo4j |
| "graves" pluriel non reconnu | `/list` | Regex `grave` → `graves?` |
| Nombres en lettres (trois, cinq) non reconnus | `/list` | Ajouter dict fr→int ou regex mots |
| Pas de garde-fou hors-sujet | `/list` | Score de pertinence ou liste de mots-clés |
| Phrasing sans fallback sur LLM | `/ask/incident-v2`, `/entity`, `/recommande` | Ajouter fallback texte brut comme sur /list |
| `"preventive"` vs `"préventive"` dans les filtres actions | `/actions` | Normaliser dans le parseur ou en base |
