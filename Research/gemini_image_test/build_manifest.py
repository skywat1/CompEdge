#!/usr/bin/env python3
"""
set_b step 3 — build the ppsf-stratified sample manifest from Stage-1 labels.

Takes classifications.csv (room labels for the set_b candidates) + candidates.csv
(each zpid's ppsf bin) and picks --per-room images/room, spread across the ppsf
bins with the end bins over-allocated (so coverage leans toward the luxury
tails). One image per zpid per room, mirroring stage2_sample.py. Any custom
outlier images the user dropped in datasets/set_b/custom/<room>/ are appended
with source="custom".

Writes datasets/set_b/data/sample_manifest.csv
    columns: image_path, zpid, room_type, source, ppsf_bin
(extra columns beyond the first three are ignored by rescore/gallery/rating app.)

Usage:
    python build_manifest.py                       # 38/room, end-weight 1.7
    python build_manifest.py --per-room 38 --end-weight 1.7
"""
import argparse
import random
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DS = HERE / "datasets" / "set_b"
CLASSIFICATIONS = DS / "data" / "classifications.csv"
CANDIDATES = DS / "data" / "candidates.csv"
CUSTOM_DIR = DS / "custom"
SET_A_MANIFEST = HERE / "datasets" / "set_a" / "data" / "sample_manifest.csv"
OUT_CSV = DS / "data" / "sample_manifest.csv"

ROOM_ORDER = ["kitchen", "bathroom", "bedroom", "living_room"]
VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def allocate(total: int, bins: int, end_weight: float) -> dict:
    """Split `total` picks across bins 1..bins; the two end bins on each side
    get `end_weight`x an interior bin's share. Returns {bin: count} summing to total."""
    end = {1, 2, bins - 1, bins}
    w = {b: (end_weight if b in end else 1.0) for b in range(1, bins + 1)}
    tot_w = sum(w.values())
    raw = {b: total * w[b] / tot_w for b in w}
    base = {b: int(v) for b, v in raw.items()}
    rem = total - sum(base.values())
    for b in sorted(raw, key=lambda b: raw[b] - base[b], reverse=True)[:rem]:
        base[b] += 1
    return base


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-room", type=int, default=38)
    ap.add_argument("--end-weight", type=float, default=1.7,
                    help="end bins get this x an interior bin's allocation (default 1.7)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    cls = pd.read_csv(CLASSIFICATIONS, dtype={"zpid": str})
    cand = pd.read_csv(CANDIDATES, dtype={"zpid": str})
    bin_of = dict(zip(cand["zpid"], cand["ppsf_bin"]))
    bins = int(cand["ppsf_bin"].max())

    set_a_z = set(pd.read_csv(SET_A_MANIFEST)["zpid"].astype(str))
    set_a_img = set(pd.read_csv(SET_A_MANIFEST)["image_path"])

    # room -> bin -> {zpid: [image_path,...]}
    rooms = cls[cls["predicted_label"].isin(ROOM_ORDER)].copy()
    rooms = rooms[~rooms["zpid"].isin(set_a_z) & ~rooms["image_path"].isin(set_a_img)]
    rooms["ppsf_bin"] = rooms["zpid"].map(bin_of)

    rows, shortfalls = [], []
    for room in ROOM_ORDER:
        sub = rooms[rooms["predicted_label"] == room]
        by_bin = {b: {} for b in range(1, bins + 1)}
        for _, r in sub.iterrows():
            b = r["ppsf_bin"]
            if pd.notna(b):
                by_bin[int(b)].setdefault(r["zpid"], []).append(r["image_path"])

        targets = allocate(args.per_room, bins, args.end_weight)
        chosen, used = [], set()

        def take(zpid, b):
            used.add(zpid)
            chosen.append({"image_path": rng.choice(sorted(by_bin[b][zpid])),
                           "zpid": zpid, "room_type": room, "source": "sampled",
                           "ppsf_bin": b})

        # pass 1: honour each bin's target
        for b in range(1, bins + 1):
            avail = [z for z in by_bin[b] if z not in used]
            rng.shuffle(avail)
            for z in avail[: targets[b]]:
                take(z, b)
        # pass 2: top up any deficit from leftover zpids in any bin (random order)
        if len(chosen) < args.per_room:
            leftovers = [(b, z) for b in range(1, bins + 1)
                         for z in by_bin[b] if z not in used]
            rng.shuffle(leftovers)
            for b, z in leftovers:
                if len(chosen) >= args.per_room:
                    break
                take(z, b)

        if len(chosen) < args.per_room:
            shortfalls.append(f"{room}: only {len(chosen)}/{args.per_room} available")
        rows.extend(chosen)
        got = pd.Series([c["ppsf_bin"] for c in chosen]).value_counts().sort_index()
        print(f"{room:12s}: {len(chosen):2d} images | per-bin: "
              + " ".join(f"{b}:{got.get(b,0)}" for b in range(1, bins + 1)))

    # custom outliers (user drop-in)
    n_custom = 0
    for room in ROOM_ORDER:
        d = CUSTOM_DIR / room
        if not d.is_dir():
            continue
        for img in sorted(d.iterdir()):
            if img.suffix.lower() in VALID_EXTS:
                rows.append({"image_path": str(img.relative_to(REPO_ROOT)),
                             "zpid": f"custom_{img.stem}", "room_type": room,
                             "source": "custom", "ppsf_bin": ""})
                n_custom += 1

    out = pd.DataFrame(rows, columns=["image_path", "zpid", "room_type", "source", "ppsf_bin"])
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"\nwrote {len(out)} rows ({len(out)-n_custom} sampled + {n_custom} custom) -> {OUT_CSV}")
    if shortfalls:
        print("SHORTFALLS (classify more candidates):\n  " + "\n  ".join(shortfalls))


if __name__ == "__main__":
    main()
