# ------------------------------------------------------------------------------
# Robot Savo LLM server — Dockerfile
#
# Goals:
# - Same behavior on Ubuntu PC (x86_64) and MacBook Air M3 (arm64)
# - Simple, predictable, easy to debug
# - Uses your existing requirements.txt
# ------------------------------------------------------------------------------

# 1) Base image: official slim Python, multi-arch (amd64 + arm64)
FROM python:3.12-slim

# 2) Basic Python hygiene: no .pyc files, unbuffered logs, faster pip
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=on \
    PIP_DISABLE_PIP_VERSION_CHECK=on

# 3) Set working directory inside the container
#    All following commands run under /app
WORKDIR /app

# 4) Install minimal system dependencies
#    - build-essential: needed if any wheel needs compilation (e.g. orjson)
#    - curl: handy for future debugging or health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 5) Install Python dependencies
#    We copy only requirements.txt first so Docker can cache this layer.
COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 6) Copy the rest of your project into the image
#    This includes app/, tools/, tests/, pyproject.toml, etc.
COPY . .

# 7) Expose FastAPI/uvicorn port (for documentation; docker-compose will bind it)
EXPOSE 8000

# 8) Default command:
#    Run your FastAPI app with uvicorn.
#    - host 0.0.0.0 so it’s reachable from outside container
#    - port 8000 matches docker-compose and your current setup
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
