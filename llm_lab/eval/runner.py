"""Evaluation runner — run inference over a dataset, collect rewards & answers."""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import torch
from datasets import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizer

from llm_lab.environments import (
    BaseEnvironment,
    ReasoningGymEnvironment,
    ToolUseEnvironment,
)


@dataclass
class SampleResult:
    """Evaluation result for a single sample."""

    index: int
    prompt: str
    expected: str
    predicted: str
    reward: float
    turns: int
    trajectories: list[dict] = field(default_factory=list)


@dataclass
class EvaluationReport:
    """Aggregated evaluation report."""

    total: int
    correct: int
    accuracy: float
    mean_reward: float
    median_reward: float
    max_reward: float
    min_reward: float
    mean_turns: float
    total_time: float
    samples: list[SampleResult] = field(default_factory=list)

    def summary(self) -> str:
        """One-line summary for console output."""
        return (
            f"Accuracy: {self.accuracy:.2%}  "
            f"({self.correct}/{self.total})  |  "
            f"Mean reward: {self.mean_reward:.3f}  |  "
            f"Avg turns: {self.mean_turns:.1f}  |  "
            f"Time: {self.total_time:.1f}s"
        )

    def print_detail(self) -> None:
        """Print detailed per-sample results."""
        print("=" * 60)
        print(self.summary())
        print("=" * 60)
        for s in self.samples:
            status = "✓" if s.reward >= 1.0 else "✗"
            print(
                f"  [{s.index:>3d}] {status}  reward={s.reward:.2f}  "
                f"turns={s.turns}  |  expected={s.expected[:50]}"
            )


def _make_env_factory(
    env_type: str,
    tokenizer: PreTrainedTokenizer,
    cfg: dict,
) -> Callable[[], BaseEnvironment]:
    """Build an environment factory from config (same structure as train config)."""
    if env_type == "tool_use":
        ec = cfg.get("env", {})
        return lambda: ToolUseEnvironment(
            tokenizer=tokenizer,
            expected_answer="",
            system_prompt=ec.get("system_prompt", ""),
            max_turns=ec.get("max_turns", 5),
        )
    else:
        # reasoning_gym — expects score_fn to be injected later
        from llm_lab.data import create_reasoning_dataset

        rc = cfg.get("reasoning", {})
        _, score_fn = create_reasoning_dataset(
            dataset_name=rc.get("dataset_name", "chain_sum"),
            size=1,  # minimal — will be overridden
            seed=rc.get("seed", 42),
            system_prompt=rc.get("system_prompt", "DeepSeekZero"),
        )

        system_prompt = rc.get("system_prompt", "DeepSeekZero")
        max_turns = rc.get("max_turns", 3)

        def _factory():
            return ReasoningGymEnvironment(
                tokenizer=tokenizer,
                score_fn=score_fn,
                system_prompt=system_prompt,
                max_turns=max_turns,
            )

        return _factory


def _run_single(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    item: dict,
    env_factory: Callable[[], BaseEnvironment],
    max_completion_length: int,
    temperature: float,
    max_prompt_length: int,
    device: torch.device,
) -> SampleResult:
    """Run one evaluation sample (single trajectory, no sampling noise)."""
    env = env_factory()
    env.reset(item)

    turns_data = []
    step = 0

    while not env.done:
        prompt_text = env.get_prompt()
        enc = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=max_prompt_length)
        input_ids = enc["input_ids"].to(device)
        prompt_len = input_ids.shape[1]

        with torch.no_grad():
            output = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_completion_length,
                do_sample=False,  # greedy for eval
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        gen_ids = output[0, prompt_len:]
        if len(gen_ids) == 0:
            env.step("")
            break

        response = tokenizer.decode(gen_ids, skip_special_tokens=True)
        result = env.step(response)
        turns_data.append({"step": step, "action": response, "observation": result.observation})
        step += 1

    # Extract expected answer from item
    expected = str(item.get("answer", item.get("expected_answer", "")))

    predicted = ""
    for msg in reversed(env.messages):
        if msg["role"] == "assistant":
            predicted = msg["content"]
            break

    return SampleResult(
        index=0,
        prompt=item.get("prompt", str(item)),
        expected=expected,
        predicted=predicted,
        reward=float(env.final_reward),
        turns=step,
        trajectories=turns_data,
    )


def evaluate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    dataset: Dataset,
    *,
    env_factory: Optional[Callable[[], BaseEnvironment]] = None,
    env_type: str = "reasoning_gym",
    config: Optional[dict] = None,
    max_completion_length: int = 256,
    max_prompt_length: int = 512,
    temperature: float = 0.0,
    verbose: bool = True,
    max_samples: Optional[int] = None,
) -> EvaluationReport:
    """Evaluate a model on a dataset using the given environment.

    Args:
        model: Model to evaluate.
        tokenizer: Matching tokenizer.
        dataset: Dataset with ``prompt`` (or ``question``) and ``answer`` keys.
        env_factory: Pre-built environment factory.  If None, built from
            ``env_type`` + ``config``.
        env_type: ``"reasoning_gym"`` or ``"tool_use"``.
        config: Training config dict (for env parameters).
        max_completion_length: Max generated tokens.
        max_prompt_length: Max prompt length.
        temperature: Sampling temperature (0 = greedy).
        verbose: Print progress.
        max_samples: Limit number of eval samples.

    Returns:
        ``EvaluationReport``.
    """
    device = next(model.parameters()).device
    model.eval()

    if env_factory is None:
        env_factory = _make_env_factory(env_type, tokenizer, config or {})

    if max_samples is not None and max_samples < len(dataset):
        dataset = dataset.select(range(max_samples))

    samples: list[SampleResult] = []
    start_time = time.time()

    for idx in range(len(dataset)):
        item = dataset[idx]

        result = _run_single(
            model=model,
            tokenizer=tokenizer,
            item=item,
            env_factory=env_factory,
            max_completion_length=max_completion_length,
            temperature=temperature,
            max_prompt_length=max_prompt_length,
            device=device,
        )
        result.index = idx
        samples.append(result)

        if verbose and (idx + 1) % 5 == 0:
            elapsed = time.time() - start_time
            correct_so_far = sum(1 for s in samples if s.reward >= 1.0)
            print(
                f"  [{idx+1}/{len(dataset)}]  "
                f"acc={correct_so_far/(idx+1):.2%}  "
                f"{elapsed:.0f}s"
            )

    elapsed = time.time() - start_time
    rewards = [s.reward for s in samples]
    correct = sum(1 for r in rewards if r >= 1.0)
    rewards_sorted = sorted(rewards)

    report = EvaluationReport(
        total=len(samples),
        correct=correct,
        accuracy=correct / len(samples) if samples else 0.0,
        mean_reward=sum(rewards) / len(rewards) if rewards else 0.0,
        median_reward=rewards_sorted[len(rewards_sorted) // 2] if rewards else 0.0,
        max_reward=max(rewards) if rewards else 0.0,
        min_reward=min(rewards) if rewards else 0.0,
        mean_turns=sum(s.turns for s in samples) / len(samples) if samples else 0.0,
        total_time=elapsed,
        samples=samples,
    )

    return report