"""Dataset loading and formatting."""

from pathlib import Path
from typing import Optional

from datasets import Dataset, load_dataset as hf_load_dataset


def load_dataset(
    path: str,
    *,
    format: str = "messages",
    split: Optional[str] = None,
    max_samples: Optional[int] = None,
) -> Dataset:
    """Load a dataset from a local file or HuggingFace hub.

    Supported formats for local files:
        - ``.jsonl`` – each line is a JSON object
        - ``.json``  – JSON array of objects
        - ``.parquet``

    Args:
        path: Local file path or HuggingFace dataset name.
        format: ``"messages"`` (chat format) or ``"text"``.
        split: Dataset split (e.g. ``"train"``).
        max_samples: Limit number of samples.

    Returns:
        HuggingFace ``Dataset``.
    """
    p = Path(path)

    # datasets uses "json" for both .json and .jsonl
    if p.suffix in (".json", ".jsonl"):
        dataset = hf_load_dataset("json", data_files=str(p), split=split or "train")
    elif p.suffix == ".parquet":
        dataset = hf_load_dataset("parquet", data_files=str(p), split=split or "train")
    else:
        dataset = hf_load_dataset(path, split=split or "train")

    if max_samples is not None:
        dataset = dataset.select(range(min(len(dataset), max_samples)))

    return dataset


def format_for_sft(example: dict) -> dict:
    """Format dataset example for SFT training.

    Expects ``messages`` field (list of ``{"role": ..., "content": ...}``).
    Returns a dict with ``"text"`` key suitable for ``SFTTrainer``.
    """
    if "messages" in example:
        import json
        return {"text": json.dumps(example["messages"], ensure_ascii=False)}
    return example


def format_for_grpo(example: dict) -> dict:
    """Format dataset example for GRPO training.

    Expects ``prompt`` or ``messages`` field.
    Returns a dict with ``"prompt"`` key.
    """
    if "prompt" in example:
        return example
    if "messages" in example:
        # Use the last user message as prompt
        msgs = example["messages"]
        for m in reversed(msgs):
            if m["role"] == "user":
                return {"prompt": m["content"]}
        # fallback: concatenate all content
        return {"prompt": "\n".join(m["content"] for m in msgs if m.get("content"))}
    return example