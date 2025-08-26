FROM python:3.12

WORKDIR /app

# Copy dependency files
COPY pyproject.toml /app/

# Copy source code
COPY ./src /app/src
COPY ./models /app/models

# Manage python environment with uv (https://docs.astral.sh/uv/guides/integration/docker/)
# The installer requires curl (and certificates) to download the release archive
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates

# Download the latest installer
ADD https://astral.sh/uv/install.sh /uv-installer.sh

# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"

# Install dependencies using uv sync
RUN uv sync

EXPOSE 80

# Run FastAPI using uv
CMD ["uv", "run", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "80"]