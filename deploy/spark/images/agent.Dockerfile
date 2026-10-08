# syntax=docker/dockerfile:1.7
# agent-api / mcp 공용 이미지 정의 (linux/arm64, DGX Spark).
# 실행 코드와 고정 의존성만 담는다. 모델·인덱스·임베딩은 번들 볼륨이다(compose.yaml).
# 빌드 컨텍스트는 저장소 루트, 포함 경로는 agent.Dockerfile.dockerignore가 연다.
# 빌드: Spark에서 네이티브 `docker compose build`, 또는 buildx `--platform linux/arm64` (README §2).
# 이 x86_64 개발 서버에서는 빌드하지 않는다.

# python:3.11 = 개발 컨테이너(3.11.9)와 같은 인터프리터 계열. digest 고정은 README §2.
ARG PYTHON_IMAGE=python:3.11.9-slim-bookworm

FROM ${PYTHON_IMAGE} AS base
ARG CODE_COMMIT=unknown
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HOME=/tmp/app-home \
    SHIPCOAT_CODE_COMMIT=${CODE_COMMIT}
LABEL org.opencontainers.image.revision=${CODE_COMMIT} \
      org.opencontainers.image.title="ship-paint-ai spark runtime"
WORKDIR /workspace

# ---------------------------------------------------------------- mcp (CPU)
FROM base AS mcp
COPY deploy/spark/requirements/mcp.lock.txt /tmp/requirements.lock.txt
RUN pip install --no-deps -r /tmp/requirements.lock.txt && pip check
# mcp_servers가 import하는 trainer 모듈: pose_similarity와 그것이 부르는 simulator_csv,
# 그리고 /pose_estimation/sequence가 쓰는 joint_names·skeleton_gate·posture_sequence·
# arm_reference·sequence_dtw·pose_stages와 그것들이 부르는 world_axes·measured_score.
# 전부 표준 라이브러리만 쓴다.
# 이 COPY 목록은 test_spark_deploy_contract.py가 그대로 복사해 새 프로세스에서 import를 확인한다.
COPY trainer/__init__.py trainer/__init__.py
COPY trainer/scripts/pose_similarity.py trainer/scripts/pose_similarity.py
COPY trainer/scripts/simulator_csv.py trainer/scripts/simulator_csv.py
COPY trainer/scripts/world_axes.py trainer/scripts/world_axes.py
COPY trainer/scripts/joint_names.py trainer/scripts/joint_names.py
COPY trainer/scripts/skeleton_gate.py trainer/scripts/skeleton_gate.py
COPY trainer/scripts/rig_profile.py trainer/scripts/rig_profile.py
COPY trainer/scripts/posture_sequence.py trainer/scripts/posture_sequence.py
COPY trainer/scripts/measured_score.py trainer/scripts/measured_score.py
COPY trainer/scripts/arm_reference.py trainer/scripts/arm_reference.py
COPY trainer/scripts/sequence_dtw.py trainer/scripts/sequence_dtw.py
COPY trainer/scripts/pass_segmentation.py trainer/scripts/pass_segmentation.py
COPY trainer/scripts/pose_stages.py trainer/scripts/pose_stages.py
COPY mcp_servers/ mcp_servers/
RUN python3 -c "import mcp_servers.main"
USER 10001:10001
EXPOSE 8100
CMD ["python3", "-m", "uvicorn", "mcp_servers.main:app", "--host", "0.0.0.0", "--port", "8100", "--workers", "1"]

# ---------------------------------------------------------------- agent-api (GPU: bge-m3, reranker)
FROM base AS agent-api
# PyTorch 공식 cu130 인덱스의 aarch64 휠(torch-2.9.1+cu130-cp311-cp311-manylinux_2_28_aarch64.whl, 2026-09-15 인덱스 확인).
# CUDA 런타임은 휠 의존성(nvidia-*-cu13)으로 들어온다. GB10(SM121) 커널 동작은 README §5에서 확인한다.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130
ARG TORCH_VERSION=2.9.1+cu130
RUN pip install --index-url "${TORCH_INDEX_URL}" --extra-index-url https://pypi.org/simple "torch==${TORCH_VERSION}"
COPY deploy/spark/requirements/agent-api.lock.txt /tmp/requirements.lock.txt
RUN pip install --no-deps -r /tmp/requirements.lock.txt && pip check
# /workspace/trainer/configs/{tool_schemas,tool_calling_protocol.json} 등 코드의 /workspace 고정 경로를 WORKDIR로 맞춘다.
COPY trainer/ trainer/
COPY deploy/spark/scripts/ deploy/spark/scripts/
RUN python3 -c "import faiss, peft, sentence_transformers, transformers; import trainer.scripts.agent_pipeline, trainer.scripts.search_index, trainer.demo_api.app" \
 && python3 -c "import torch; assert (torch.version.cuda or '').startswith('13'), torch.version.cuda"
USER 10001:10001
EXPOSE 8120
CMD ["python3", "-m", "trainer.demo_api.app", "--host", "0.0.0.0", "--port", "8120"]
