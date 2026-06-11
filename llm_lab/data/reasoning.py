"""Reasoning-Gym dataset integration."""

from typing import Callable, Optional

from datasets import Dataset
from reasoning_gym import create_dataset
from reasoning_gym.utils import extract_answer, SYSTEM_PROMPTS


def create_reasoning_dataset(
    dataset_name: str,
    size: int = 1000,
    seed: int = 42,
    system_prompt: str = "DeepSeekZero",
    **kwargs,
) -> tuple:
    """Create a reasoning-gym dataset wrapped for GRPO training.

    Returns ``(dataset, score_fn)`` where ``dataset`` is a HuggingFace
    ``Dataset`` with keys ``prompt`` (raw question) and ``metadata``
    (the original reasoning-gym item), and ``score_fn`` is a callable
    for answer verification.

    The ``prompt`` column must be tokenized later (e.g. by the trainer
    using the tokenizer's ``apply_chat_template``).

    Args:
        dataset_name: Name of reasoning-gym dataset (e.g. ``"chain_sum"``).
        size: Number of samples to generate.
        seed: RNG seed.
        system_prompt: Key into ``SYSTEM_PROMPTS`` dict.
        **kwargs: Passed through to ``create_dataset``.

    Returns:
        Tuple of ``(Dataset, callable)``.
    """
    rg_dataset = create_dataset(dataset_name, seed=seed, size=size, **kwargs)
    sys_prompt = SYSTEM_PROMPTS.get(system_prompt, system_prompt)

    def generator():
        for item in rg_dataset:
            yield {
                "prompt": item["question"],
                "metadata": item,
                "system_prompt": sys_prompt,
            }

    hf_dataset = Dataset.from_generator(generator)
    return hf_dataset, rg_dataset.score_answer


def format_prompt_for_model(
    question: str,
    tokenizer,
    system_prompt: str = "",
) -> str:
    """Format a question into a model-ready prompt using the tokenizer's
    ``apply_chat_template``."""
    chat = []
    if system_prompt:
        chat.append({"role": "system", "content": system_prompt})
    chat.append({"role": "user", "content": question})
    return tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)


def extract_answer_from_completion(completion: str) -> Optional[str]:
    """Extract final answer from model completion using <answer> tags."""
    return extract_answer(completion)