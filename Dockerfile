FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# TensorBoard (lancé à la demande depuis le conteneur, cf. README)
EXPOSE 6006

# Conteneur "outil" : reste en vie, on exécute les phases via `docker compose exec`
CMD ["sleep", "infinity"]
