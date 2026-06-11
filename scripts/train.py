"""Training entry point.

Usage:
    python scripts/train.py -c configs/example.yaml
"""

import argparse
import sys

import yaml
from transformers import PreTrainedModel


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _maybe_apply_lora(model: PreTrainedModel, cfg: dict) -> PreTrainedModel:
    """Attach LoRA adapters if ``lora`` section exists in config."""
    lc = cfg.get("lora")
    if lc is None:
        return model

    from llm_lab.models import setup_lora

    print("  attaching LoRA adapters (r=%d, alpha=%d) …" % (lc.get("r", 16), lc.get("alpha", 16)))
    model = setup_lora(
        model,
        r=lc.get("r", 16),
        alpha=lc.get("alpha", 16),
        dropout=lc.get("dropout", 0.0),
        target_modules=lc.get("target_modules"),
        use_rslora=lc.get("use_rslora", False),
    )
    return model


def main():
    parser = argparse.ArgumentParser(description="llm_lab training")
    parser.add_argument("--config", "-c", required=True, help="Path to config YAML")
    args = parser.parse_args()
    cfg = load_config(args.config)

    # ---- model ----
    from llm_lab.models import load_model_and_tokenizer

    mc = cfg["model"]
    model, tokenizer = load_model_and_tokenizer(
        model_name_or_path=mc["name_or_path"],
        source=mc.get("source", "huggingface"),
        max_seq_length=cfg.get("max_seq_length", 2048),
        load_in_4bit=mc.get("load_in_4bit", False),
        inference_only=False,   # we may attach LoRA below
    )

    # ---- optionally apply LoRA / QLoRA ----
    model = _maybe_apply_lora(model, cfg)

    # ---- train config ----
    tc = cfg["training"]
    algo = tc.get("algorithm", "sft")

    # ---- dataset (most algos need this) ----
    if algo not in ("grpo_reasoning", "grpo_multi_turn"):
        from llm_lab.data import load_dataset

        dc = cfg["data"]
        dataset = load_dataset(
            path=dc["path"],
            format=dc.get("format", "messages"),
            max_samples=dc.get("max_samples"),
        )

    # ---- dispatch ----
    if algo == "sft":
        from llm_lab.trainers import train_sft

        train_sft(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            output_dir=tc.get("output_dir", "./outputs/sft"),
            max_seq_length=cfg.get("max_seq_length", 2048),
            batch_size=tc.get("batch_size", 2),
            learning_rate=tc.get("learning_rate", 2.0e-4),
            num_epochs=tc.get("epochs", 1),
            logging_steps=tc.get("logging_steps", 10),
            save_steps=tc.get("save_steps", 100),
        )
    elif algo in ("grpo", "grpo_variant"):
        reward_funcs = None
        if tc.get("reward_func"):
            reward_funcs = [_import_reward_func(tc["reward_func"])]

        from llm_lab.trainers import train_grpo

        train_grpo(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            reward_funcs=reward_funcs,
            output_dir=tc.get("output_dir", "./outputs/grpo"),
            max_seq_length=cfg.get("max_seq_length", 2048),
            batch_size=tc.get("batch_size", 2),
            learning_rate=tc.get("learning_rate", 2.0e-4),
            num_epochs=tc.get("epochs", 1),
            logging_steps=tc.get("logging_steps", 10),
            save_steps=tc.get("save_steps", 100),
        )
    elif algo == "grpo_reasoning":
        from llm_lab.data import create_reasoning_dataset
        from llm_lab.trainers import train_grpo_reasoning

        rc = tc.get("reasoning", {})
        rdataset, score_fn = create_reasoning_dataset(
            dataset_name=rc.get("dataset_name", "chain_sum"),
            size=rc.get("dataset_size", 1000),
            seed=rc.get("seed", 42),
            system_prompt=rc.get("system_prompt", "DeepSeekZero"),
        )
        train_grpo_reasoning(
            model=model,
            tokenizer=tokenizer,
            train_dataset=rdataset,
            score_fn=score_fn,
            max_seq_length=cfg.get("max_seq_length", 2048),
            max_prompt_length=rc.get("max_prompt_length", 512),
            max_completion_length=rc.get("max_completion_length", 1024),
            batch_size=tc.get("batch_size", 2),
            num_generations=rc.get("num_generations", 8),
            learning_rate=tc.get("learning_rate", 2.0e-6),
            num_epochs=tc.get("epochs", 1),
            output_dir=tc.get("output_dir", "./outputs/grpo_reasoning"),
            logging_steps=tc.get("logging_steps", 1),
            save_steps=tc.get("save_steps", 100),
        )
    elif algo == "grpo_multi_turn":
        from llm_lab.data import create_reasoning_dataset
        from llm_lab.environments import ReasoningGymEnvironment
        from llm_lab.trainers import train_grpo_multi_turn

        rc = tc.get("reasoning", {})
        rdataset, score_fn = create_reasoning_dataset(
            dataset_name=rc.get("dataset_name", "chain_sum"),
            size=rc.get("dataset_size", 8),
            seed=rc.get("seed", 42),
            system_prompt=rc.get("system_prompt", "DeepSeekZero"),
        )

        env_factory = lambda: ReasoningGymEnvironment(
            tokenizer=tokenizer,
            score_fn=score_fn,
            system_prompt=rc.get("system_prompt", "DeepSeekZero"),
            max_turns=rc.get("max_turns", 3),
        )

        train_grpo_multi_turn(
            model=model,
            tokenizer=tokenizer,
            train_dataset=rdataset,
            env_factory=env_factory,
            num_generations=tc.get("num_generations", 2),
            max_turns=rc.get("max_turns", 3),
            max_prompt_length=rc.get("max_prompt_length", 256),
            max_completion_length=rc.get("max_completion_length", 128),
            max_seq_length=cfg.get("max_seq_length", 1536),
            beta=tc.get("beta", 0.04),
            learning_rate=tc.get("learning_rate", 1.0e-6),
            batch_size=tc.get("batch_size", 1),
            num_epochs=tc.get("epochs", 1),
            output_dir=tc.get("output_dir", "./outputs/grpo_multi_turn"),
            logging_steps=tc.get("logging_steps", 1),
            save_steps=tc.get("save_steps", 50),
            temperature=tc.get("temperature", 0.6),
        )
    else:
        print(f"Unknown algorithm: {algo}")
        sys.exit(1)

    print("Done.")


def _import_reward_func(path: str):
    mod_path, _, func_name = path.partition(":")
    import importlib

    mod = importlib.import_module(mod_path)
    return getattr(mod, func_name)


if __name__ == "__main__":
    main()