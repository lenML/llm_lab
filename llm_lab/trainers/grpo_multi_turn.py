"""Multi-turn REINFORCE trainer with environment interaction.

This implements a custom training loop (REINFORCE with standardised
advantages + entropy bonus, no KL reference) that supports multiple
turns of model-environment interaction per prompt. Each trajectory
goes through N turns of ``generate → environment step → feedback``
before a final reward is assigned.

Why REINFORCE instead of full GRPO?
  GRPO requires a frozen reference model for the KL penalty, which
  with LoRA/QLoRA means maintaining a separate copy of adapter
  weights.  The simpler REINFORCE + entropy achieves a similar effect
  without the overhead and is sufficient for initial experiments.
"""

import os
from typing import Callable, Optional

import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizer

from llm_lab.environments import BaseEnvironment


# ── helpers ────────────────────────────────────────────────────────────


def _tokenize(
    tokenizer: PreTrainedTokenizer,
    text: str,
    device: torch.device,
    max_length: Optional[int] = None,
) -> torch.Tensor:
    """Tokenize text and return ``[1, seq_len]`` tensor on ``device``."""
    return tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length or 4096,
    )["input_ids"].to(device)


def _compute_trajectory_log_probs(
    model: PreTrainedModel,
    turns: list,
    device: torch.device,
) -> torch.Tensor:
    """Compute summed log probabilities of all generated tokens across turns.

    Each element in *turns* is ``(prompt_ids_1d, gen_ids_1d)``.

    Args:
        model: The policy model.
        turns: List of (prompt_ids, gen_ids) pairs, one per turn.
        device: Torch device.

    Returns:
        Scalar tensor --- sum of per-token log probs for the whole trajectory.
    """
    total = torch.tensor(0.0, device=device)

    for prompt_ids, gen_ids in turns:
        gen_len = len(gen_ids)
        if gen_len == 0 or len(prompt_ids) == 0:
            continue

        prompt_len = len(prompt_ids)
        full = torch.cat([prompt_ids, gen_ids]).unsqueeze(0).to(device)  # [1, seq]

        outputs = model(full, attention_mask=torch.ones_like(full))
        logits = outputs.logits[0]  # [seq, V]

        # logits[i] → token[i+1]; generated tokens span [prompt_len, prompt_len+gen_len)
        gen_logits = logits[prompt_len - 1 : prompt_len + gen_len - 1]  # [gen_len, V]
        gen_lp = torch.log_softmax(gen_logits, dim=-1)

        token_lp = gen_lp[torch.arange(gen_len, device=device), gen_ids]
        total = total + token_lp.sum()

    return total


def _compute_reinforce_loss(
    model: PreTrainedModel,
    trajectories: list,
    entropy_coef: float,
    device: torch.device,
) -> tuple:
    """REINFORCE with standardised advantages + optional entropy bonus.

    No KL penalty against a reference model.  This avoids the need for
    a frozen reference copy and works with LoRA adapters.

    Args:
        model: Current policy model.
        trajectories: List of ``(turns, reward)``.
        entropy_coef: Coefficient for entropy regularisation.
        device: Torch device.

    Returns:
        ``(loss, log_probs, rewards)``.
    """
    curr_log_probs: list[torch.Tensor] = []
    for turns, _ in trajectories:
        lp = _compute_trajectory_log_probs(model, turns, device)
        curr_log_probs.append(lp)

    curr_lp = torch.stack(curr_log_probs)  # [T]
    rewards = torch.tensor(
        [r for _, r in trajectories], device=device, dtype=torch.float
    )

    adv = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

    entropy = -curr_lp.mean()
    loss = -(adv.detach() * curr_lp).mean() - entropy_coef * entropy

    return loss, curr_lp.detach(), rewards.detach()


# ── public trainer ──────────────────────────────────────────────────────


def train_grpo_multi_turn(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    train_dataset: Dataset,
    *,
    env_factory: Callable[[], BaseEnvironment],
    num_generations: int = 4,
    max_turns: int = 3,
    max_prompt_length: int = 512,
    max_completion_length: int = 128,
    max_seq_length: int = 2048,
    entropy_coef: float = 0.01,
    learning_rate: float = 1e-6,
    batch_size: int = 1,
    num_epochs: int = 1,
    output_dir: str = "./outputs/grpo_multi_turn",
    logging_steps: int = 1,
    save_steps: int = 100,
    temperature: float = 0.6,
    gradient_accumulation_steps: int = 1,
    report_to: Optional[str] = None,
) -> None:
    """Run multi-turn REINFORCE training.

    For every prompt in the batch, ``num_generations`` independent
    multi-turn trajectories are rolled out.  The final reward (from the
    environment) is used for the standardised advantage (REINFORCE with
    baseline), and an entropy bonus encourages exploration.

    Args:
        model: Unsloth-loaded model (training mode).
        tokenizer: Matching tokenizer.
        train_dataset: Dataset with ``prompt`` and ``metadata`` keys.
        env_factory: Callable that returns a fresh ``BaseEnvironment``
            per trajectory.
        num_generations: Number of trajectories per prompt.
        max_turns: Maximum environment steps per trajectory.
        max_prompt_length: Truncate accumulated prompt history.
        max_completion_length: Max new tokens per generation call.
        max_seq_length: Max total sequence length (prompt+completion).
        entropy_coef: Entropy regularisation coefficient.
        learning_rate: AdamW learning rate.
        batch_size: Batch size for dataset loader.
        num_epochs: Training epochs.
        output_dir: Model save directory.
        logging_steps: Log every N steps.
        save_steps: Save checkpoint every N steps.
        temperature: Sampling temperature.
        gradient_accumulation_steps: Accumulate gradients over N steps.
        report_to: Optional reporting (wandb etc).
    """
    device = next(model.parameters()).device
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: b,
    )

    global_step = 0
    accumulated_loss = 0.0

    for epoch in range(num_epochs):
        for batch_items in dataloader:
            # ── 1. Roll out trajectories ──────────────────────────
            trajectories: list = []  # [(turns, reward), ...]

            for item in batch_items:
                for _ in range(num_generations):
                    env = env_factory()
                    env.reset(item)
                    turns: list = []  # [(prompt_ids_1d, gen_ids_1d)]

                    while not env.done:
                        prompt_text = env.get_prompt()
                        prompt_ids = _tokenize(
                            tokenizer, prompt_text, device, max_prompt_length
                        )[0]  # 1D

                        with torch.no_grad():
                            output = model.generate(
                                input_ids=prompt_ids.unsqueeze(0),
                                max_new_tokens=max_completion_length,
                                do_sample=True,
                                temperature=temperature,
                                pad_token_id=tokenizer.pad_token_id,
                                eos_token_id=tokenizer.eos_token_id,
                            )

                        gen_ids = output[0, prompt_ids.shape[0]:]  # 1D
                        if len(gen_ids) == 0:
                            env.step("")
                            break

                        response = tokenizer.decode(gen_ids, skip_special_tokens=True)
                        result = env.step(response)

                        turns.append((prompt_ids.detach().cpu(), gen_ids.detach().cpu()))

                    trajectories.append((turns, float(env.final_reward)))

            if not trajectories:
                continue

            # ── 2. REINFORCE loss ────────────────────────────────
            loss, curr_lp_log, rewards_log = _compute_reinforce_loss(
                model, trajectories, entropy_coef, device
            )

            loss_scaled = loss / gradient_accumulation_steps
            loss_scaled.backward()
            accumulated_loss += loss_scaled.item()

            if (global_step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            global_step += 1

            # ── 3. Logging ────────────────────────────────────────
            if global_step % logging_steps == 0:
                traj_rewards = [r for _, r in trajectories]
                mean_r = sum(traj_rewards) / len(traj_rewards)
                max_r = max(traj_rewards)
                print(
                    f"Step {global_step:>5d} | loss {accumulated_loss:.4f} | "
                    f"reward mean {mean_r:.3f} max {max_r:.3f} | "
                    f"mean_lp {curr_lp_log.mean().item():.3f}"
                )
                accumulated_loss = 0.0

            # ── 4. Save checkpoint ────────────────────────────────
            if global_step % save_steps == 0:
                ckpt = os.path.join(output_dir, f"step_{global_step}")
                model.save_pretrained(ckpt)
                tokenizer.save_pretrained(ckpt)
                print(f"  → saved checkpoint to {ckpt}")

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"  ✓ model saved to {output_dir}")