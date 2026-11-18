"""Download the CelebAMask-HQ parquet shard used by the measured validation.

    python download_data.py --shard 0
    python run_measured.py --n 500

Source: HF dataset v-xchen-v/celebamask_hq (6 shards, 30,000 images; the first
shard holds 5,000 and is enough for the part statistics in this release).
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = ("https://huggingface.co/datasets/v-xchen-v/celebamask_hq/"
        "resolve/main/data/train-{:05d}-of-00006.parquet")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--outdir", type=str, default=str(ROOT / "data" / "celebamaskhq"))
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    dest = outdir / f"shard{args.shard}.parquet"
    url = BASE.format(args.shard)
    if dest.exists() and dest.stat().st_size > 1_000_000:
        print("already present:", dest)
        return
    print("downloading", url)
    urllib.request.urlretrieve(url, dest)
    print("saved", dest, dest.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
