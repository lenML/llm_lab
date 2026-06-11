"""GRPO / GRPO variant trainer."""

from typing import Optional, Callable

from datasets import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizer

# unsloth must be imported first
import unsloth
from trl import GRPOConfig, GRPOTrainer

from llm_lab.data.dataset import format_for_grpo


def default_reward_func(prompts: list[str], completions: list[str], **kwargs) -> list[float]:
    """Default reward: length-based heuristic.

    Override this with your own reward function in config.
    """
    return [min(len(c) / 100.0, 1.0) for c in completions]


def train_grpo(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    train_dataset: Dataset,
    *,
    reward_funcs: Optional[list[Callable]] = None,
    output_dir: str = "./outputs/grpo",
    max_seq_length: int = 2048,
    batch_size: int = 2,
    learning_rate: float = 2e-4,
    num_epochs: int = 1,
    logging_steps: int = 10,
    save_steps: int = 100,
    eval_dataset: Optional[Dataset] = None,
) -> GRPOTrainer:
    """Run GRPO reinforcement learning training.

    Args:
        model: Pretrained model loaded via Unsloth.
        tokenizer: Matching tokenizer.
        train_dataset: Training dataset (will be formatted for GRPO).
        reward_funcs: List of reward functions. Defaults to ``[default_reward_func]``.
        output_dir: Save directory.
        max_seq_length: Max token length.
        batch_size: Per-device batch size.
        learning_rate: Adam learning rate.
        num_epochs: Number of training epochs.
        logging_steps: Log every N steps.
        save_steps: Save checkpoint every N steps.
        eval_dataset: Optional evaluation dataset.

    Returns:
        Trained ``GRPOTrainer`` instance.
    """
    if reward_funcs is None:
        reward_funcs = [default_reward_func]

    # Ensure dataset has "prompt" key
    if "prompt" not in train_dataset.column_names:
        train_dataset = train_dataset.map(format_for_grpo)

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
        fp16=not unsloth.is_bfloat16_supported(),
        bf16=unsloth.is_bfloat16_supported(),
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    return trainer