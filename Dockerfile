# Day-ahead GB carbon intensity forecast API
FROM python:3.12-slim

# LightGBM needs the OpenMP runtime on Linux (a classic Windows-to-Linux gotcha)
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CARBON_FORECAST_ROOT=/app

WORKDIR /app

# Dependencies first: this layer is reused on rebuilds when only code changes
COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt

# Application code, config, and the champion model
COPY pyproject.toml .
COPY src/ src/
COPY configs/ configs/
COPY models/ models/
RUN pip install --no-deps .

# Run as a non-root user
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

# Docker (and later Azure) checks this to know the app is actually responding
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"

CMD ["uvicorn", "carbon_forecast.serving.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]