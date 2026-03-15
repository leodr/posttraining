"""Convert conversation JSONL files into a HuggingFace dataset in prompt-completion format.

The SFT trainer expects each example to have:
  - prompt: list of messages forming the input (system + user)
  - completion: list of messages forming the expected output (assistant)

Usage:
    uv run python 3_build_dataset.py                              # default: ./conversations -> ./dataset
    uv run python 3_build_dataset.py --input-dir ./conversations --output-dir ./dataset
"""

import argparse
import json
from pathlib import Path

from datasets import Dataset


def convert_messages(messages: list[dict]) -> dict:
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
    parser = argparse.ArgumentParser(description="Build SFT dataset from conversation JSONL files.")
    parser.add_argument("--input-dir", type=Path, default=Path("conversations"))
    parser.add_argument("--output-dir", type=Path, default=Path("dataset"))
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
    ds.save_to_disk(str(args.output_dir))
    print(f"Saved dataset to {args.output_dir}")


if __name__ == "__main__":
    main()
