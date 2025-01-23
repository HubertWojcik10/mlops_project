FROM --platform=linux/amd64 python:3.11-slim AS base

WORKDIR /code
COPY ./requirements.txt /code/requirements.txt
COPY ./pyproject.toml /code/pyproject.toml
COPY ./configs/ /code/configs/
COPY ./src/ /code/src/

RUN pip install -r requirements.txt --no-cache-dir --verbose

COPY ./app /code/app

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY src/ src/
COPY configs/ configs/
COPY data/ data/

RUN pip install dvc[gs]
RUN dvc init --no-scm
COPY .dvc/config .dvc/config
COPY *.dvc .dvc/
RUN dvc config core.no_scm true

ENV GOOGLE_APPLICATION_CREDENTIALS="configs/credentials.json"
RUN dvc pull -v

RUN pip install -r requirements.txt
RUN pip install . --no-deps --no-cache-dir --verbose

CMD exec uvicorn src.mlops_project.api:app --port $PORT --host 0.0.0.0 --workers 1
