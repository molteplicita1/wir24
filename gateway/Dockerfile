# FASE BUILD
FROM python:3.12-slim AS builder

WORKDIR /app

# Copio i file necessari per la build
COPY . /app

# Creazione venv
RUN python -m venv /app/venv

# Attivazione venv e installazione dipendenze
RUN /app/venv/bin/pip install --no-cache-dir -r requirements.txt


# FASE RUNTIME
FROM python:3.12-slim

WORKDIR /app

# Copio solo il codice sorgente e il venv dalla fase builder
COPY --from=builder /app /app

# Configurazione ENV per utilizzare il venv
ENV PATH="/app/venv/bin:$PATH"

EXPOSE 8004

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8004"]