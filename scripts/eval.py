"""Evaluation CLI — measure model performance before / after training.

Usage:
    # Evaluate baseline (no checkpoint)
    python scripts/eval.py -c configs/grpo_reasoning.yaml

    # Evaluate a checkpoint after training
    python scripts/eval.py -c configs/grpo_reasoning.yaml \\
        --checkpoint ./outputs/grpo_multi_turn/step_50

    # Compare two runs
    python scripts/eval.py -c configs/grpo_reasoning.yaml \\
        --checkpoint ./outputs/grpo_multi_turn/step_50 \\
        --checkpoint ./outputs/grpo_multi_turn/step_100
"""

import argparse
import sys

import yaml


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="llm_lab evaluation")
    parser.add_argument("--config", "-c", required=True, help="Path to config YAML")
    parser.add_argument(
        "--checkpoint", "-k", action="append", default=[],
        help="Model checkpoint dir(s) to evaluate.  Repeat for comparison.  "
             "Omit to evaluate the base model (before training).",
    )
    parser.add_argument("--max-samples", type=int, default=None, help="Limit samples")
    parser.add_argument("--max-completion-length", type=int, default=256)
    parser.add_argument("--output", "-o", default=None, help="Save report JSON")
    args = parser.parse_args()

    cfg = load_config(args.config)
    mc = cfg["model"]
    tc = cfg["training"]
    algo = tc.get("algorithm", "sft")

    # ---- build dataset & env config ----
    env_type = tc.get("env_type", "reasoning_gym")

    dataset = None
    if env_type == "tool_use":
        from llm_lab.data import load_dataset

        dataset = load_dataset(
            path=cfg["data"]["path"],
            format="messages",
            max_samples=args.max_samples,
        )
    else:
        from llm_lab.data import create_reasoning_dataset

        rc = tc.get("reasoning", {})
        dataset, _ = create_reasoning_dataset(
            dataset_name=rc.get("dataset_name", "chain_sum"),
            size=rc.get("dataset_size", 50),
            seed=rc.get("seed", 42),
            system_prompt=rc.get("system_prompt", "DeepSeekZero"),
        )
        if args.max_samples:
            dataset = dataset.select(range(min(len(dataset), args.max_samples)))

    # ---- load model and evaluate ----
    from llm_lab.models import load_model_and_tokenizer
    from llm_lab.eval import evaluate

    checkpoints = list(args.checkpoint) or [None]

    all_reports = []
    for ckpt in checkpoints:
        label = ckpt or "base (no checkpoint)"
        print(f"\n{'='*60}")
        print(f"  Evaluating: {label}")
        print(f"{'='*60}")

        model, tokenizer = load_model_and_tokenizer(
            model_name_or_path=ckpt or mc["name_or_path"],
            source="local" if ckpt else mc.get("source", "huggingface"),
            max_seq_length=cfg.get("max_seq_length", 2048),
            load_in_4bit=mc.get("load_in_4bit", False),
            inference_only=True,
        )

        # Apply LoRA if config has it (only for base model;
        # checkpoints already have adapters baked in)
        if ckpt is None:
            lc = cfg.get("lora")
            if lc is not None:
                from llm_lab.models import setup_lora

                model = setup_lora(
                    model,
                    r=lc.get("r", 16),
                    alpha=lc.get("alpha", 16),
                    dropout=lc.get("dropout", 0.0),
                    target_modules=lc.get("target_modules"),
                    use_rslora=lc.get("use_rslora", False),
                )

        report = evaluate(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            env_type=env_type,
            config=tc,
            max_completion_length=args.max_completion_length,
            verbose=True,
        )

        print(f"\n  Result: {report.summary()}")
        all_reports.append((label, report))

    # ---- comparison summary ----
    if len(all_reports) > 1:
        print(f"\n{'='*60}")
        print("  COMPARISON")
        print(f"{'='*60}")
        for label, report in all_reports:
            print(f"  {label:40s}  {report.summary()}")
        print()

    # ---- save JSON ----
    if args.output:
        import json

        data = []
        for label, report in all_reports:
            data.append({
                "label": label,
                "total": report.total,
                "correct": report.correct,
                "accuracy": report.accuracy,
                "mean_reward": report.mean_reward,
                "median_reward": report.median_reward,
                "mean_turns": report.mean_turns,
                "total_time": report.total_time,
                "samples": [
                    {"index": s.index, "reward": s.reward, "turns": s.turns,
                     "expected": s.expected, "predicted": s.predicted[:200]}
                    for s in report.samples
                ],
            })
        with open(args.output, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"  Report saved to {args.output}")


if __name__ == "__main__":
    main()