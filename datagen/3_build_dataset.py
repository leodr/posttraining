"""Convert conversation JSONL files into a HuggingFace dataset and push to the Hub.

The SFT trainer expects each example to have:
  - prompt: list of messages forming the input (system + user)
  - completion: list of messages forming the expected output (assistant)

Usage:
    uv run python 3_build_dataset.py <hf_repo_id>
    uv run python 3_build_dataset.py myorg/my-sft-dataset
    uv run python 3_build_dataset.py myorg/my-sft-dataset --input-dir ./conversations --private
"""

import argparse
import json
from pathlib import Path

from datasets import Dataset


def convert_messages(messages: list[dict]) -> dict | None:
    """Split a messages list into prompt (everything up to the last user message)
    and completion (everything after)."""
    last_user_idx = -1
    for i, msg in enumerate(messages):
        if msg["role"] == "user":
            last_user_idx = i

    if last_user_idx == -1:
        return None

    prompt = messages[: last_user_idx + 1]
    completion = messages[last_user_idx + 1 :]

    if not completion:
        return None

    return {"prompt": prompt, "completion": completion}


def main():
    parser = argparse.ArgumentParser(description="Build SFT dataset and push to HuggingFace Hub.")
    parser.add_argument("repo_id", help="HuggingFace repo ID (e.g. myorg/my-sft-dataset)")
    parser.add_argument("--input-dir", type=Path, default=Path("conversations"))
    parser.add_argument("--private", action="store_true", help="Make the dataset private")
    args = parser.parse_args()

    jsonl_files = sorted(args.input_dir.glob("*.jsonl"))
    if not jsonl_files:
        print(f"No JSONL files found in {args.input_dir}")
        return

    rows = []
    skipped = 0
    for path in jsonl_files:
        with open(path, encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                converted = convert_messages(record["messages"])
                if converted is None:
                    skipped += 1
                    continue
                rows.append(converted)

    print(f"Loaded {len(rows)} examples from {len(jsonl_files)} files ({skipped} skipped)")

    ds = Dataset.from_list(rows)
    ds = ds.shuffle(seed=42)
    ds.push_to_hub(args.repo_id, private=args.private)
    print(f"Pushed dataset to https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
