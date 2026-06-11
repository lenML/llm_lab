"""
llm_lab - Unsloth training entry point.

Usage:
    python scripts/train.py --config configs/example.yaml
"""

import argparse
import yaml


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="llm_lab training")
    parser.add_argument("--config", "-c", required=True, help="Path to config YAML")
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(f"Config loaded: {cfg['model']['name']}")
    print("Environment ready.")


if __name__ == "__main__":
    main()