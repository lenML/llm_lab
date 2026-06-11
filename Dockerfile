FROM unsloth/unsloth:latest

# Qwen3.5 needs transformers>=5.2.0
RUN pip install --no-cache-dir "transformers>=5.2.0,<5.3"

# Qwen3.5 linear attention fast path
RUN pip install --no-cache-dir causal-conv1d flash-linear-attention

ENV PYTHONPATH=/workspace
WORKDIR /workspace