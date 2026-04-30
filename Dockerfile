FROM python:3.11-slim-bookworm

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=600 \
    PIP_RETRIES=10 \
    HF_HOME=/workspace/.cache/huggingface \
    TRANSFORMERS_CACHE=/workspace/.cache/huggingface

WORKDIR /workspace

COPY requirements.train.txt /workspace/requirements.train.txt

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --retries 10 --timeout 600 --progress-bar off --index-url "${TORCH_INDEX_URL}" torch
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --retries 10 --timeout 600 --progress-bar off -r /workspace/requirements.train.txt

COPY train.py /workspace/train.py
COPY mlops_utils.py /workspace/mlops_utils.py

CMD ["python", "train.py", "--help"]
