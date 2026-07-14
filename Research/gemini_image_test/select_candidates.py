#!/usr/bin/env python3
"""
set_b step 1 — pick candidate listings spread across price-per-sqft (ppsf).

ppsf = sold_price / area-sqft is our best cheap proxy for room luxury
(Spearman ~0.58-0.61 vs. human scores, vs. 0.34 for raw price). We bin the pool
into log-ppsf quantile bins and select a candidate zpid pool per bin — with the
end bins over-drawn — so the downstream classify + sample steps have enough
spread to reach toward the luxury tails. (True 1s/8s still come from the
user-added custom/ images; ppsf can't isolate the extremes on its own.)

Excludes every set_a listing so set_b is a genuine held-out set. Writes
datasets/set_b/data/candidates.csv (zpid, ppsf, ppsf_bin) — the source of the
ppsf bin for every later step.

Usage:
    python select_candidates.py                    # defaults below
    python select_candidates.py --bins 12 --per-bin 20 --end-mult 2.0
"""
import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
SOLD_CSV = REPO_ROOT / "data" / "cleaned_sold.csv"
IMAGES_DIR = REPO_ROOT / "images"
SET_A_MANIFEST = HERE / "datasets" / "set_a" / "data" / "sample_manifest.csv"
OUT_CSV = HERE / "datasets" / "set_b" / "data" / "candidates.csv"


def _num(s):
    return pd.to_numeric(s.astype(str).str.replace(r"[^0-9.]", "", regex=True),
                         errors="coerce")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bins", type=int, default=12, help="log-ppsf quantile bins (default 12)")
    ap.add_argument("--per-bin", type=int, default=20,
                    help="candidate listings per interior bin (default 20)")
    ap.add_argument("--end-mult", type=float, default=2.0,
                    help="the two bins at each end get this * per-bin (default 2.0)")
    ap.add_argument("--min-sqft", type=int, default=200,
                    help="drop listings below this sqft (fake-extreme ppsf from scrape errors)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(SOLD_CSV)
    df["zpid"] = df["zpid"].astype(str)
    df["price"] = _num(df["sold_price"])
    df["sqft"] = _num(df["area-sqft"])
    df["ppsf"] = df["price"] / df["sqft"]

    n0 = len(df)
    df = df[(df["price"] > 0) & (df["sqft"] >= args.min_sqft) & df["ppsf"].notna()]
    df = df[df["zpid"].map(lambda z: (IMAGES_DIR / z).is_dir())]

    set_a = set(pd.read_csv(SET_A_MANIFEST)["zpid"].astype(str))
    df = df[~df["zpid"].isin(set_a)].drop_duplicates("zpid").reset_index(drop=True)
    print(f"pool: {n0} listings -> {len(df)} candidates "
          f"(valid ppsf + images + not in set_a's {len(set_a)} zpids)")

    # log-ppsf quantile bins, labelled 1..bins (1 = cheapest $/sqft)
    df["ppsf_bin"] = pd.qcut(np.log(df["ppsf"]), args.bins,
                             labels=range(1, args.bins + 1), duplicates="drop").astype(int)
    edges = np.exp(np.quantile(np.log(df["ppsf"]), np.linspace(0, 1, args.bins + 1)))
    print("ppsf bin edges ($/sqft):", [int(x) for x in edges])

    end_bins = {1, 2, args.bins - 1, args.bins}
    rng = random.Random(args.seed)
    picked = []
    for b in range(1, args.bins + 1):
        pool = df[df["ppsf_bin"] == b]["zpid"].tolist()
        rng.shuffle(pool)
        target = int(round(args.per_bin * (args.end_mult if b in end_bins else 1.0)))
        take = pool[:target]
        picked.append(df[df["zpid"].isin(take)])
        print(f"  bin {b:2d}: {len(take):3d} / {len(pool):4d} available")

    out = pd.concat(picked)[["zpid", "ppsf", "ppsf_bin"]].sort_values(
        ["ppsf_bin", "ppsf"]).reset_index(drop=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"\nwrote {len(out)} candidate listings -> {OUT_CSV}")


if __name__ == "__main__":
    main()
