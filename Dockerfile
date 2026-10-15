FROM python:3.13-slim
WORKDIR /app

COPY requirements.txt pyproject.toml README.md ./
COPY src/ ./src/
COPY web/editor/ ./web/editor/
COPY config/ ./config/

RUN pip install --no-cache-dir -e .

# Corpus, LIFT snapshots, editor sessions and corrections are runtime state.
# They are deliberately not baked into the image.
ENV PYTHONUNBUFFERED=1 \
    MAALULA_STATE_DIR=/state

VOLUME ["/state"]
EXPOSE 8000
CMD ["python", "-m", "maalula_glosser.serve"]
