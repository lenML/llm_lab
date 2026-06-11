"""Model loading from HuggingFace or ModelScope, with LoRA/QLoRA support."""

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
    inference_only: bool = False,
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Load base model and tokenizer.

    Args:
        model_name_or_path: HF name or local path to model directory.
        source: ``"huggingface"`` or ``"modelscope"``.
        max_seq_length: Maximum sequence length used by Unsloth patching.
        dtype: Torch dtype override (defaults to auto-detection).
        load_in_4bit: Whether to quantize to 4-bit (required for QLoRA).
        device_map: Device map string.
        inference_only: If True, call ``for_inference``.  Set to False when
            LoRA adapters will be attached afterward.

    Returns:
        Tuple of (model, tokenizer).  The model is in eval mode if
        ``inference_only=True``, otherwise train mode.
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

    # ── unsloth model ────────────────────────────────────────────
    model, _ = FastLanguageModel.from_pretrained(
        model_name=actual_path,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        device_map=device_map,
        trust_remote_code=True,
        local_files_only=is_local,
    )

    if inference_only:
        FastLanguageModel.for_inference(model)

    return model, tokenizer


def setup_lora(
    model: PreTrainedModel,
    *,
    r: int = 16,
    alpha: int = 16,
    dropout: float = 0.0,
    target_modules: Optional[list[str]] = None,
    use_rslora: bool = False,
    use_gradient_checkpointing: str = "unsloth",
    random_state: int = 42,
) -> PreTrainedModel:
    """Attach LoRA / QLoRA adapters to a base model via unsloth.

    Call this **after** ``load_model_and_tokenizer(..., inference_only=False)``
    and **before** passing the model to a trainer.

    When ``load_in_4bit`` was used during loading, this creates a QLoRA setup.

    Args:
        model: Base model from ``FastLanguageModel.from_pretrained``.
        r: LoRA rank.
        alpha: LoRA scaling factor.
        dropout: LoRA dropout.
        target_modules: List of module names to adapt.  If None, uses
            sensible defaults for common architectures.
        use_rslora: Use ``RsLoRA`` (rank-stabilized scaling).
        use_gradient_checkpointing: ``"unsloth"`` for memory-optimised,
            ``True`` for standard, ``False`` to disable.
        random_state: Seed for adapter initialisation.

    Returns:
        PEFT model ready for training.
    """
    if target_modules is None:
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]

    model = FastLanguageModel.get_peft_model(
        model,
        r=r,
        target_modules=target_modules,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        use_gradient_checkpointing=use_gradient_checkpointing,
        random_state=random_state,
        use_rslora=use_rslora,
        loftq_config=None,
    )
    return model


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