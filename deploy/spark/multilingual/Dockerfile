ARG BASE_IMAGE=shipcoat/agent-api:swagger-ux
FROM ${BASE_IMAGE}
WORKDIR /workspace
COPY trainer/multilingual/ trainer/multilingual/
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
USER 10001:10001
EXPOSE 8121
CMD ["python3", "-m", "uvicorn", "trainer.multilingual.service:create_app", "--factory", "--host", "0.0.0.0", "--port", "8121", "--workers", "1"]
