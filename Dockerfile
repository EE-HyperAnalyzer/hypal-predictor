FROM ghcr.io/astral-sh/uv:debian-slim
RUN apt-get update && apt-get install -y git

WORKDIR /app
ADD pyproject.toml uv.lock /app/
ADD src/hypal_predictor/__init__.py /app/src/hypal_predictor/__init__.py
ADD README.md /app/README.md
RUN uv sync

COPY . /app
CMD ["uv", "run", "python", "main.py"]
