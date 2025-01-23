# Base image
FROM python:3.11-slim AS base

EXPOSE $PORT

WORKDIR /

RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy configuration files
COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY src/ src/
COPY configs/ configs/
COPY data/ data/

# Install dvc with Google Storage support
RUN pip install dvc[gs]

# Initialize dvc and pull data
RUN dvc init --no-scm
COPY .dvc/config .dvc/config
COPY *.dvc .dvc/
RUN dvc config core.no_scm true

ENV GOOGLE_APPLICATION_CREDENTIALS="configs/credentials.json"

# Run dvc pull to fetch data
RUN dvc pull -v
RUN ls -la data/processed/

# Install dependencies
RUN pip install -r requirements.txt
RUN pip install . --no-deps --no-cache-dir --verbose

CMD exec uvicorn api:src/mlops_project --port $PORT --host 0.0.0.0 --workers 1
