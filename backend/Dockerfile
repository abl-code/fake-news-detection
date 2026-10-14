FROM python:3.13-slim AS dependencies
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt
FROM python:3.13-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY --from=dependencies /wheels /wheels
COPY requirements.txt .
RUN pip install --no-cache-dir --no-index --find-links=/wheels -r requirements.txt && rm -rf /wheels
COPY backend/ backend/
RUN test -s backend/model.pkl && test -s backend/vectorizer.pkl
RUN useradd --create-home appuser
USER appuser
ENV PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
EXPOSE 5000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health', timeout=4)"
CMD ["gunicorn", "--workers", "1", "--threads", "2", "--bind", "0.0.0.0:5000", "--timeout", "120", "--access-logfile", "-", "--error-logfile", "-", "backend.src.app:app"]
