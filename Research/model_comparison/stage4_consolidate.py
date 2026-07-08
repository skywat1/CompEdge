#!/usr/bin/env python3
"""
Stage 4 — consolidate all experiment outputs into parquet files.

Writes three parquet files: classifications (all Stage 1 labels incl. "other"),
the sample manifest, and the full scoring table (model, exact model ID, image,
zpid, room type, replicate, score, reason, room-type judgment, token usage,
latency, computed cost).

Usage:
    python stage4_consolidate.py --classifications-csv outputs/classifications.csv \
        --manifest outputs/sample_manifest.csv --scores-csv outputs/scores.csv \
        --out-dir outputs/parquet
"""

import argparse
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--classifications-csv", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--scores-csv", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    cls = pd.read_csv(args.classifications_csv, dtype={"zpid": str})
    manifest = pd.read_csv(args.manifest, dtype={"zpid": str})
    scores = pd.read_csv(args.scores_csv, dtype={"zpid": str})

    for df, name in [(cls, "classifications"), (manifest, "sample_manifest"), (scores, "scores")]:
        out = args.out_dir / f"{name}.parquet"
        df.to_parquet(out, index=False)
        print(f"{out}: {len(df)} rows, {len(df.columns)} cols")

    errs = scores["error"].notna() & (scores["error"].astype(str).str.len() > 0)
    print(f"\nScoring rows with errors: {errs.sum()} / {len(scores)}")


if __name__ == "__main__":
    main()
