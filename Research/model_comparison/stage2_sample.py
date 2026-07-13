#!/usr/bin/env python3
"""
Stage 2 — build the stratified scoring sample from Stage 1 labels.

Targets N images per room type (default 38, ~150 total), at most one image per
room type per zpid so every sampled image is an independent observation.
"other" images are excluded (they stay in the Stage 1 CSV). If a room type is
scarce, a minimum of 25 is accepted; below that the script errors out and asks
for more classification.

The manifest this writes is used verbatim for ALL five models — never resample.

Usage:
    python stage2_sample.py --classifications-csv outputs/classify/classifications.csv \
        --out-manifest outputs/sample/sample_manifest.csv
"""

import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

from common import ROOM_TYPES


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--classifications-csv", type=Path, required=True)
    ap.add_argument("--out-manifest", type=Path, required=True)
    ap.add_argument("--per-type", type=int, default=38, help="Target images per room type (default 38)")
    ap.add_argument("--min-per-type", type=int, default=25, help="Acceptable minimum for scarce types (default 25)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # candidates[room_type][zpid] = [image_path, ...]
    candidates = {t: defaultdict(list) for t in ROOM_TYPES}
    with open(args.classifications_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["predicted_label"] in candidates:
                candidates[row["predicted_label"]][row["zpid"]].append(row["image_path"])

    rng = random.Random(args.seed)
    rows, shortfalls = [], []
    for room in ROOM_TYPES:
        zpids = sorted(candidates[room])
        rng.shuffle(zpids)
        picked = []
        for zpid in zpids[: args.per_type]:
            picked.append({"image_path": rng.choice(sorted(candidates[room][zpid])),
                           "zpid": zpid, "room_type": room})
        if len(picked) < args.min_per_type:
            shortfalls.append(f"{room}: only {len(picked)} zpids available (< min {args.min_per_type})")
        rows.extend(picked)
        print(f"{room:12s}: {len(picked)} images from {len(picked)} distinct zpids "
              f"({len(zpids)} zpids available)")

    if shortfalls:
        sys.exit("ERROR — scarce room types, run more Stage 1 classification:\n  "
                 + "\n  ".join(shortfalls))

    args.out_manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_manifest, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image_path", "zpid", "room_type"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nManifest written: {args.out_manifest} ({len(rows)} images)")


if __name__ == "__main__":
    main()
