FROM python:3.11-slim

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/workspace/.cache/huggingface \
    TRANSFORMERS_CACHE=/workspace/.cache/huggingface

WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends tini ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.train.txt /workspace/requirements.train.txt

RUN python -m pip install --upgrade pip \
    && pip install --index-url "${TORCH_INDEX_URL}" torch \
    && pip install -r /workspace/requirements.train.txt

COPY train.py /workspace/train.py

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "train.py", "--help"]
