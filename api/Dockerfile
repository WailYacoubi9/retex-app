# Image Docker pour l'API FastAPI du projet RETEX.
# Build :  docker compose build api
# Run   :  docker compose up api

FROM python:3.12-slim

# Variables d'environnement Python pour des conteneurs propres
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Installation des dependances en premier (pour beneficier du cache Docker)
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copie du code applicatif
COPY *.py .

# Port expose par uvicorn
EXPOSE 8000

# Lancement de l'API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
