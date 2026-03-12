FROM python:3.12-slim

WORKDIR /app

# Install curl for health checks
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (cached layer — only rebuilds when requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install vendored lightweight charts fork (selective copy excludes 377MB node_modules)
COPY streamlit_lwc_fork/setup.py ./streamlit_lwc_fork/
COPY streamlit_lwc_fork/streamlit_lightweight_charts/ ./streamlit_lwc_fork/streamlit_lightweight_charts/
RUN pip install --no-cache-dir ./streamlit_lwc_fork

# Copy application code
COPY src/ ./src/
COPY config/ ./config/
COPY user_packs/ ./user_packs/
COPY reference-indicators/ ./reference-indicators/

# Streamlit production config
RUN mkdir -p .streamlit && printf '\
[server]\n\
headless = true\n\
enableCORS = false\n\
enableXsrfProtection = false\n\
maxUploadSize = 10\n\
\n\
[browser]\n\
gatherUsageStats = false\n\
' > .streamlit/config.toml

ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl --fail http://localhost:${PORT:-8501}/_stcore/health || exit 1

CMD streamlit run src/app.py --server.port=${PORT:-8501} --server.address=0.0.0.0 --server.headless=true
