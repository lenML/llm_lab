"""Supervised Fine-Tuning trainer."""

from typing import Optional

from datasets import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizer

# unsloth must be imported first
import unsloth
from trl import SFTConfig, SFTTrainer

from llm_lab.data.dataset import format_for_sft


def train_sft(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    train_dataset: Dataset,
    *,
    output_dir: str = "./outputs/sft",
    max_seq_length: int = 2048,
    batch_size: int = 2,
    learning_rate: float = 2e-4,
    num_epochs: int = 1,
    logging_steps: int = 10,
    save_steps: int = 100,
    eval_dataset: Optional[Dataset] = None,
) -> SFTTrainer:
    """Run SFT training.

    Args:
        model: Pretrained model loaded via Unsloth.
        tokenizer: Matching tokenizer.
        train_dataset: Training dataset (will be formatted for SFT).
        output_dir: Save directory.
        max_seq_length: Max token length.
        batch_size: Per-device batch size.
        learning_rate: Adam learning rate.
        num_epochs: Number of training epochs.
        logging_steps: Log every N steps.
        save_steps: Save checkpoint every N steps.
        eval_dataset: Optional evaluation dataset.

    Returns:
        Trained ``SFTTrainer`` instance.
    """
    # Ensure dataset has "text" key
    if "text" not in train_dataset.column_names:
        train_dataset = train_dataset.map(format_for_sft)

    args = SFTConfig(
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
        max_seq_length=max_seq_length,
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    return trainer