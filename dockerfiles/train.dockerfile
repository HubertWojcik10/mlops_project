# Base image
FROM python:3.11-slim AS base

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY src/ src/
COPY configs/ configs/
COPY data/ data/

WORKDIR /
RUN pip install -r requirements.txt
RUN pip install . --no-deps --no-cache-dir --verbose

RUN dvc init --no-scm
COPY .dvc/config .dvc/config
COPY *.dvc .dvc/
RUN dvc config core.no_scm true
#COPY credentials.json credentials.json
RUN dvc pull

ENTRYPOINT ["python", "-u", "src/mlops_project/train.py"]
