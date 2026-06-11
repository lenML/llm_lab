"""Model loading from HuggingFace or ModelScope."""

from pathlib import Path
from typing import Optional

# unsloth must be imported before transformers to apply patches
from unsloth import FastLanguageModel

import torch
from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizer


def load_model_and_tokenizer(
    model_name_or_path: str,
    source: str = "huggingface",
    *,
    max_seq_length: int = 2048,
    dtype: Optional[torch.dtype] = None,
    load_in_4bit: bool = False,
    device_map: str = "auto",
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Load model and tokenizer from HF hub / local path or ModelScope.

    Args:
        model_name_or_path: HF name or local path to model directory.
        source: ``"huggingface"`` or ``"modelscope"``.
        max_seq_length: Maximum sequence length used by Unsloth patching.
        dtype: Torch dtype override (defaults to auto-detection).
        load_in_4bit: Whether to quantize to 4-bit.
        device_map: Device map string.

    Returns:
        Tuple of (model, tokenizer).
    """
    actual_path = _resolve_model_path(model_name_or_path, source)
    is_local = Path(actual_path).exists()
    tokenizer = AutoTokenizer.from_pretrained(
        actual_path,
        trust_remote_code=True,
        local_files_only=is_local,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model, _ = FastLanguageModel.from_pretrained(
        model_name=actual_path,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        device_map=device_map,
        trust_remote_code=True,
        local_files_only=is_local,
    )
    FastLanguageModel.for_inference(model)

    return model, tokenizer


def _resolve_model_path(name_or_path: str, source: str) -> str:
    """Resolve model path. For modelscope source, try local cache first."""
    if source == "modelscope":
        p = Path(name_or_path)
        # if it's already a local path (relative or absolute), use it directly
        if p.is_absolute() or name_or_path.startswith("./") or name_or_path.startswith(".\\"):
            return name_or_path
        local = Path("models") / name_or_path
        if local.exists():
            return str(local)
        try:
            from modelscope import snapshot_download
            return snapshot_download(name_or_path, cache_dir="models")
        except ImportError:
            raise ImportError(
                "modelscope required. pip install modelscope"
            )
    return name_or_path