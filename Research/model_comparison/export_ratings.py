#!/usr/bin/env python3
"""
Export Stage 2b human ratings from SQLite to CSV and parquet for Stage 5.

Usage:
    python export_ratings.py --db outputs/ratings/human_ratings.sqlite \
        --out-csv outputs/ratings/human_ratings.csv \
        --out-parquet outputs/parquet/human_ratings.parquet
"""

import argparse
import sqlite3
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument("--out-csv", type=Path, required=True)
    ap.add_argument("--out-parquet", type=Path, required=True)
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    df = pd.read_sql_query(
        "SELECT rater, image_path, room_type, score, timestamp FROM ratings "
        "ORDER BY rater, image_path", conn)
    subset = pd.read_sql_query("SELECT COUNT(*) n FROM subset", conn)["n"][0]
    conn.close()

    for out in (args.out_csv, args.out_parquet):
        out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    df.to_parquet(args.out_parquet, index=False)

    print(f"{len(df)} ratings from {df['rater'].nunique()} raters "
          f"(subset size {subset}) ->\n  {args.out_csv}\n  {args.out_parquet}")
    if len(df):
        done = df.groupby("rater")["image_path"].nunique()
        for rater, n in done.items():
            flag = "" if n == subset else f"  (incomplete: {n}/{subset})"
            print(f"  {rater}: {n} rated{flag}")


if __name__ == "__main__":
    main()
