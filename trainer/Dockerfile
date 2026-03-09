FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y \
    git curl vim nano build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel

RUN pip install \
    transformers \
    accelerate \
    peft \
    datasets \
    trl \
    bitsandbytes \
    sentencepiece \
    einops \
    scipy \
    scikit-learn \
    pandas \
    numpy \
    jupyterlab \
    ipykernel \
    qdrant-client \
    fastapi \
    uvicorn \
    llama-index \
    llama-index-llms-openai-like \
    llama-index-vector-stores-qdrant \
    langchain \
    langchain-community \
    sentence-transformers \
    FlagEmbedding \
    pypdf \
    unstructured \
    python-dotenv \
    huggingface_hub

WORKDIR /workspace

CMD ["/bin/bash"]
