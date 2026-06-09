# dynaSim — Streamlit container image
#
# Build:  docker build -t dynasim .
# Run:    docker run --rm -p 8501:8501 dynasim
# Open:   http://localhost:8501

FROM python:3.11-slim

# Don't buffer stdout/stderr (so `docker logs` is live), don't write
# .pyc files (smaller image, no scribbles in the source mount).
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies first so this layer is cached when only source
# files change. The root requirements.txt is `-r app/requirements.txt`,
# so both files must be present before pip resolves it.
COPY requirements.txt ./
COPY app/requirements.txt ./app/requirements.txt
RUN pip install --upgrade pip \
 && pip install -r requirements.txt

# Now copy the rest of the source. .dockerignore filters out the bulky
# thesis / build artifacts that would otherwise bloat the image.
COPY . .

# Run as a non-root user — defence in depth.
RUN useradd --create-home --uid 1000 dynasim \
 && chown -R dynasim:dynasim /app
USER dynasim

EXPOSE 8501

# Streamlit ships a built-in health endpoint at /_stcore/health.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; \
sys.exit(0) if urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=3).status == 200 else sys.exit(1)"

CMD ["streamlit", "run", "app/nlds_app.py", "--server.address=0.0.0.0"]
