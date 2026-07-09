#!/usr/bin/env python3
"""Before/after per-room comparison for a re-scored gemini-3.5-flash run.

Reads the original scores (5-replicate baseline) and the tuned scores, and for
gemini-3.5-flash prints per-room rho / MAD / signed-bias against both human
references (harvey+robin and all 3), side by side, so you can see whether the
prompt edits pulled the bias toward zero without hurting rank correlation. Also
prints the per-room share of images off by >1 point, before vs after.

Usage (from Research/model_comparison, with the repo venv):
    python compare_tuned.py \
        --old-parquet outputs/parquet/scores.parquet \
        --new-parquet outputs/parquet_tuned/scores.parquet \
        --ratings-parquet outputs/parquet/human_ratings.parquet
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

MODEL = "gemini-3.5-flash"
ROOMS = ["kitchen", "bathroom", "bedroom", "living_room"]
REFS = {"harvey+robin": ["harvey", "robin"], "all (h+r+seb)": ["harvey", "robin", "seb"]}


def load_means(parquet: Path) -> pd.DataFrame:
    s = pd.read_parquet(parquet)
    s = s[s["error"].fillna("").astype(str).str.len() == 0].copy()
    s["score"] = pd.to_numeric(s["score"], errors="coerce")
    s = s[s["model"] == MODEL]
    return (s.groupby(["image_path", "room_type"])["score"]
            .mean().reset_index(name="mean_score"))


def per_room(means: pd.DataFrame, ref: pd.Series) -> pd.DataFrame:
    m = means.set_index("image_path")
    rows = []
    for room in ROOMS:
        mr = m[m["room_type"] == room]["mean_score"]
        j = pd.concat([mr, ref], axis=1, join="inner", keys=["m", "h"]).dropna()
        d = j["m"] - j["h"]
        rows.append({"room": room, "n": len(j),
                     "rho": spearmanr(j["m"], j["h"])[0] if len(j) > 2 else np.nan,
                     "MAD": d.abs().mean(), "bias": d.mean(),
                     "pct>1": 100 * (d.abs() > 1).mean()})
    return pd.DataFrame(rows).set_index("room")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--old-parquet", type=Path, default=Path("outputs/parquet/scores.parquet"))
    ap.add_argument("--new-parquet", type=Path, default=Path("outputs/parquet_tuned/scores.parquet"))
    ap.add_argument("--ratings-parquet", type=Path, default=Path("outputs/parquet/human_ratings.parquet"))
    args = ap.parse_args()

    ratings = pd.read_parquet(args.ratings_parquet)
    old = load_means(args.old_parquet)
    new = load_means(args.new_parquet)

    for ref_label, raters in REFS.items():
        ref = ratings[ratings["rater"].isin(raters)].groupby("image_path")["score"].mean()
        o, n = per_room(old, ref), per_room(new, ref)
        print(f"\n{'='*78}\nHUMAN REFERENCE: {ref_label}\n{'='*78}")
        cmp = pd.DataFrame({
            "n": n["n"],
            "bias_before": o["bias"], "bias_after": n["bias"],
            "|bias|_delta": n["bias"].abs() - o["bias"].abs(),
            "rho_before": o["rho"], "rho_after": n["rho"],
            "MAD_before": o["MAD"], "MAD_after": n["MAD"],
            "pct>1_before": o["pct>1"], "pct>1_after": n["pct>1"],
        })
        print(cmp.round(3).to_string())
        print(f"\n  mean |bias|: {o['bias'].abs().mean():.3f} -> {n['bias'].abs().mean():.3f}"
              f"   |   mean rho: {o['rho'].mean():.3f} -> {n['rho'].mean():.3f}"
              f"   (negative |bias| delta = improvement; rho should hold)")


if __name__ == "__main__":
    main()
