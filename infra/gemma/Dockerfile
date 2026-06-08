FROM vllm/vllm-openai:latest

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/models/huggingface
ENV TRANSFORMERS_CACHE=/models/huggingface
ENV HUGGINGFACE_HUB_CACHE=/models/huggingface

COPY check_model.py /app/check_model.py
COPY healthcheck.py /app/healthcheck.py

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=5 \
  CMD python /app/healthcheck.py
