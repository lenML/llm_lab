FROM unsloth/unsloth:latest

RUN pip install --no-cache-dir "transformers>=5.2.0,<5.3" reasoning-gym

ENV PYTHONPATH=/workspace
WORKDIR /workspace