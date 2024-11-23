# FASE BUILD
FROM python:3.12-slim AS builder

WORKDIR /app

COPY . /app

RUN python -m venv /app/venv

RUN /app/venv/bin/pip install --no-cache-dir -r requirements.txt

# FASE RUNTIME
FROM python:3.12-slim

WORKDIR /app

COPY --from=builder /app /app

ENV PATH="/app/venv/bin:$PATH"

EXPOSE 8003

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8003"]