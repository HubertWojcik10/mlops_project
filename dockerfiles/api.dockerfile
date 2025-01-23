# Change from latest to a specific version if your requirements.txt
FROM python:3.11-slim AS base

WORKDIR /code
COPY ./requirements.txt /code/requirements.txt
COPY ./pyproject.toml /code/pyproject.toml
COPY ./configs/ /code/configs/
COPY ./src/ /code/src/

RUN pip install -r requirements.txt --no-cache-dir --verbose

COPY ./app /code/app

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "80"]