"""Training entry point.

Usage:
    python scripts/train.py -c configs/example.yaml
"""

import argparse
import sys

import yaml


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


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
    )

    # ---- dataset ----
    from llm_lab.data import load_dataset

    dc = cfg["data"]
    dataset = load_dataset(
        path=dc["path"],
        format=dc.get("format", "messages"),
        max_samples=dc.get("max_samples"),
    )

    # ---- train ----
    tc = cfg["training"]
    algo = tc.get("algorithm", "sft")

    if algo == "sft":
        from llm_lab.trainers import train_sft

        train_sft(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            output_dir=tc.get("output_dir", "./outputs/sft"),
            max_seq_length=cfg.get("max_seq_length", 2048),
            batch_size=tc.get("batch_size", 2),
            learning_rate=tc.get("learning_rate", 2e-4),
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
            learning_rate=tc.get("learning_rate", 2e-4),
            num_epochs=tc.get("epochs", 1),
            logging_steps=tc.get("logging_steps", 10),
            save_steps=tc.get("save_steps", 100),
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