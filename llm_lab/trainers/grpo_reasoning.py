"""GRPO trainer for reasoning-gym datasets."""

import re
from typing import Callable, Optional

import unsloth  # noqa: must be imported before transformers
from datasets import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizer
from trl import GRPOConfig, GRPOTrainer
from unsloth import is_bfloat16_supported

from llm_lab.data.reasoning import format_prompt_for_model


def format_reward(completions, **kwargs):
    """Reward for correct <think> / <answer> structure."""
    regex = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>\n<answer>([\s\S]*?)<\/answer>$"
    matches = [re.match(regex, c, flags=re.DOTALL) for c in completions]
    return [1.0 if m else 0.0 for m in matches]


def _make_accuracy_reward(score_fn: Callable):
    """Create an accuracy reward function bound to a specific score function."""

    def accuracy_reward(completions, metadata, **kwargs):
        from reasoning_gym.utils import extract_answer

        answers = [extract_answer(c) for c in completions]
        return [score_fn(a, entry=m) for a, m in zip(answers, metadata)]

    return accuracy_reward


def tokenize_prompts(dataset: Dataset, tokenizer, system_prompt_key: str = "system_prompt") -> Dataset:
    """Apply chat template to format prompts for the model."""

    def _format(example):
        prompt = format_prompt_for_model(
            example["prompt"],
            tokenizer,
            system_prompt=example.get(system_prompt_key, ""),
        )
        return {"prompt": prompt}

    return dataset.map(_format)


def train_grpo_reasoning(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    train_dataset: Dataset,
    *,
    score_fn: Callable,
    max_seq_length: int = 2048,
    max_prompt_length: int = 512,
    max_completion_length: int = 1024,
    batch_size: int = 2,
    num_generations: int = 8,
    learning_rate: float = 2.0e-6,
    num_epochs: int = 1,
    output_dir: str = "./outputs/grpo_reasoning",
    logging_steps: int = 1,
    save_steps: int = 100,
    extra_reward_funcs: Optional[list] = None,
) -> GRPOTrainer:
    """Run GRPO training on reasoning-gym datasets.

    Uses ``format_reward`` (structure) + ``accuracy_reward`` (answer)
    by default.

    Args:
        model: PreTrainedModel loaded via Unsloth.
        tokenizer: Matching tokenizer.
        train_dataset: Dataset with ``prompt``, ``metadata``, ``system_prompt``.
        score_fn: ``dataset.score_answer(answer, entry)`` callable.
        max_seq_length: Max total sequence length.
        max_prompt_length: Max prompt length.
        max_completion_length: Max model-generated completion length.
        batch_size: Per-device batch size.
        num_generations: Generations per prompt.
        learning_rate: Learning rate.
        num_epochs: Number of epochs.
        output_dir: Save directory.
        logging_steps: Log every N steps.
        save_steps: Save checkpoint every N steps.
        extra_reward_funcs: Additional reward functions.

    Returns:
        Trained ``GRPOTrainer`` instance.
    """
    # Format prompts with chat template
    train_dataset = tokenize_prompts(train_dataset, tokenizer)

    reward_funcs = [format_reward, _make_accuracy_reward(score_fn)]
    if extra_reward_funcs:
        reward_funcs.extend(extra_reward_funcs)

    args = GRPOConfig(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=2,
        remove_unused_columns=False,
        report_to="none",
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        use_vllm=False,
        
        num_generations=num_generations,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    return trainer