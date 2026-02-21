FROM ghcr.io/astral-sh/uv:debian-slim
RUN apt-get update && apt-get install -y git

WORKDIR /app
ADD pyproject.toml uv.lock /app/
RUN uv sync

COPY . /app
CMD ["uv", "run", "python", "main.py"]
