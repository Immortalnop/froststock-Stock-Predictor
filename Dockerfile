FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml README.md ./
COPY stockpred ./stockpred
RUN pip install .

# State lives in volumes: the DuckDB store, trained bundles and run logs.
ENV STOCKPRED_DATA_DIR=/data/store STOCKPRED_ARTIFACTS_DIR=/data/artifacts STOCKPRED_RUNS_DIR=/data/runs
VOLUME /data
EXPOSE 8000
HEALTHCHECK --interval=60s --timeout=10s --start-period=60s CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/health', timeout=8)"
ENTRYPOINT ["stockpred"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8000"]
