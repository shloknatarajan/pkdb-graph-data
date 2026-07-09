"""Upload the HF-ready dataset to the Hugging Face Hub.

Loads dataset/hf as an imagefolder (images get embedded into parquet, so the Hub
dataset viewer and `load_dataset(<repo>)` both work with zero path fixups) and
pushes it, then uploads the README card + LICENSE.

Prereqs:
    uv pip install datasets huggingface_hub pillow
    huggingface-cli login          # or set HF_TOKEN

Usage:
    python scripts/push_to_hub.py <namespace>/<dataset-name> [--private]
"""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HF = ROOT / "dataset" / "hf"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("repo_id", help="e.g. shlok/pkdb-figure-timecourse")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    from datasets import load_dataset
    from huggingface_hub import HfApi

    ds = load_dataset("imagefolder", data_dir=str(HF))
    print("loaded:", {k: len(v) for k, v in ds.items()})
    ds.push_to_hub(args.repo_id, private=args.private)
    print(f"pushed splits to https://huggingface.co/datasets/{args.repo_id}")

    api = HfApi()
    for fname in ("README.md", "LICENSE"):
        api.upload_file(
            path_or_fileobj=str(HF / fname),
            path_in_repo=fname,
            repo_id=args.repo_id,
            repo_type="dataset",
        )
    print("uploaded README.md + LICENSE (dataset card)")


if __name__ == "__main__":
    main()
