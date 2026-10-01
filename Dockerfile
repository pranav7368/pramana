FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements-pilot.txt ./
RUN pip install --no-cache-dir -r requirements-pilot.txt
COPY pyproject.toml README.md ./
COPY run_demo.py ./
COPY src ./src
COPY scripts ./scripts
COPY examples/corpus ./examples/corpus
RUN pip install --no-cache-dir --no-deps . && \
    useradd --uid 10001 --create-home pilot && \
    mkdir -p /app/data/cache && chown -R pilot:pilot /app/data
USER pilot
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/v1/health', timeout=3)"
CMD ["python", "-m", "uvicorn", "pramana.api.service:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-proxy-headers", "--timeout-keep-alive", "5"]
