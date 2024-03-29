FROM python:3.11

ENV PYTHONUNBUFFERED True \
    APP_HOME /app \
    POETRY_VIRTUALENVS_CREATE false



RUN apt-get update && apt-get install -y curl poppler-utils git openssh-client libgl1-mesa-glx libglib2.0-0


WORKDIR /app

ENV PATH="/root/.local/bin:$PATH"
RUN curl -sSL https://install.python-poetry.org | python3 -  && poetry config virtualenvs.create false

COPY pyproject.toml ./

#RUN poetry install --without dev
RUN poetry install --no-root

COPY ./ ./

CMD ["uvicorn", "lumineres.main:app", "--host", "0.0.0.0", "--port", "8080"]
