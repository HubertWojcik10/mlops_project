# Base image
FROM python:3.11-slim AS base

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

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
WORKDIR /
RUN pip install -r requirements.txt
RUN pip install . --no-deps --no-cache-dir --verbose

# Set entry point for the application
ENTRYPOINT ["python", "-u", "src/mlops_project/train.py"]
