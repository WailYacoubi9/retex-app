# retex-app

Couche service du projet RETEX : **API** (FastAPI, `api/`) + **Frontend**
(Streamlit, `frontend/`). Se connecte aux services de `retex-backbone` via le
réseau Docker partagé `retex-net`.

## Prérequis
`retex-backbone` doit tourner (il crée le réseau `retex-net` + neo4j/qdrant/ollama).

## Démarrer
```bash
docker compose up -d --build
```
- API : http://localhost:8000 (`/health`, `/stats`, `/ask/tickets`)
- Frontend : http://localhost:8501

## Config (variables d'env dans docker-compose.yml)
- `NEO4J_URI`, `QDRANT_URL`, `OLLAMA_URL` → pointent vers les services du backbone (par nom).
- `TICKETS_LLM_MODEL` → modèle de génération (défaut `qwen2.5:7b`).
