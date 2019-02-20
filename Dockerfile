FROM python:3.7 AS base

ARG DOCKER_DEV
ARG CI_USER_TOKEN
RUN echo "machine github.com\n  login $CI_USER_TOKEN\n" >~/.netrc

ENV \
    PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    PIPENV_HIDE_EMOJIS=true \
    PIPENV_COLORBLIND=true \
    PIPENV_NOSPIN=true \
    PYTHONPATH="/app/src:${PYTHONPATH}"

RUN pip install pipenv

WORKDIR /build
COPY README.rst .
COPY HISTORY.rst .
COPY Pipfile .
COPY Pipfile.lock .
COPY setup.py .
COPY pset_utils/__init__.py src/pset_utils/__init__.py
RUN pipenv install --system --deploy --ignore-pipfile --dev

# --ignore-pipfile --dev

WORKDIR /app
