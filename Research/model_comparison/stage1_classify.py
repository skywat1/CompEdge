#!/usr/bin/env python3
"""
Stage 1 — classify listing images by room type with gpt-4o (detail=low).

Shuffles properties (zpids) with a fixed seed, then processes several properties
concurrently until the Stage 2 quotas are met (enough distinct zpids per room
type) or the call cap is reached.

Per-property EARLY STOP: within a property, images are classified one at a time
until the property has produced a candidate for all four room types (kitchen,
bathroom, bedroom, living_room) or its images are exhausted — extra images of a
property can never enter the Stage 2 sample (which keeps at most one image per
room type per zpid), so classifying them is wasted. "other" images seen before
the stop are still saved.

Concurrency is ACROSS properties (each property runs its own short sequential
classify-until-all-types chain in a worker thread); shared counters and the
append-only checkpoint are guarded by a lock, so writes and the quota/cap logic
stay exact. With W properties in flight the quota can overshoot by up to W
properties — a few extra saved rows, never a correctness issue.

EVERY result — including "other" — is appended to the checkpoint CSV, so the
stage is resumable and idempotent: already-classified image paths are skipped
on re-run. Raw API responses go to a JSONL.

Usage:
    python stage1_classify.py --images-dir ../../images \
        --out-csv outputs/classify/classifications.csv \
        --raw-jsonl outputs/classify/classifications_raw.jsonl --workers 8
"""

import argparse
import csv
import json
import random
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from common import (CLASSIFY_LABELS, ROOM_TYPES, VALID_EXTS, classify_image,
                    compute_cost, set_rate_limit, CLASSIFIER_MODEL)

CSV_FIELDS = ["image_path", "zpid", "predicted_label", "other_label", "response_model",
              "input_tokens", "output_tokens", "cached_input_tokens",
              "latency_s", "cost_usd"]


def load_done(csv_path: Path):
    """From a prior run: set of classified image paths, per-type zpid sets, and
    per-zpid set of room types already recorded (to persist the early-stop
    decision so resume is a true no-op)."""
    done, zpids_by_type, zpid_types = set(), {t: set() for t in ROOM_TYPES}, {}
    if csv_path.exists():
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                done.add(row["image_path"])
                if row["predicted_label"] in zpids_by_type:
                    zpids_by_type[row["predicted_label"]].add(row["zpid"])
                    zpid_types.setdefault(row["zpid"], set()).add(row["predicted_label"])
    return done, zpids_by_type, zpid_types


def quotas_met(zpids_by_type, target):
    return all(len(z) >= target for z in zpids_by_type.values())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images-dir", type=Path, required=True,
                    help="Directory of per-zpid image folders (images/<zpid>/<zpid>_N.jpg)")
    ap.add_argument("--zpids-file", type=Path, default=None,
                    help="Optional CSV/text file with a 'zpid' column (or one zpid per "
                         "line); restrict classification to just these properties. Used to "
                         "drive ppsf-stratified candidate selection.")
    ap.add_argument("--out-csv", type=Path, required=True,
                    help="Append-only classification checkpoint CSV")
    ap.add_argument("--raw-jsonl", type=Path, required=True,
                    help="Append-only raw API response JSONL")
    ap.add_argument("--target-zpids-per-type", type=int, default=38,
                    help="Stop once this many distinct zpids per room type are found (default 38)")
    ap.add_argument("--max-calls", type=int, default=3000,
                    help="Hard cap on total classification calls across all runs (default 3000)")
    ap.add_argument("--seed", type=int, default=42, help="Property shuffle seed (default 42)")
    ap.add_argument("--workers", type=int, default=4,
                    help="Properties classified concurrently (default 4)")
    ap.add_argument("--tpm-limit", type=int, default=28000,
                    help="Client-side OpenAI tokens-per-minute cap to pace calls under your "
                         "account's TPM limit and avoid 429s (default 28000, just under a "
                         "30k Tier-1 gpt-4o limit; raise on higher tiers, 0 disables)")
    args = ap.parse_args()

    set_rate_limit("openai", args.tpm_limit)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    done, zpids_by_type, zpid_types = load_done(args.out_csv)
    state = {"calls": len(done)}
    print(f"Resuming with {state['calls']} images already classified.")
    print("Per-type zpid counts so far:",
          {t: len(z) for t, z in zpids_by_type.items()})

    zpid_dirs = sorted(d for d in args.images_dir.iterdir() if d.is_dir())
    if args.zpids_file is not None:
        text = args.zpids_file.read_text(encoding="utf-8").splitlines()
        header = text[0].split(",") if text else []
        col = header.index("zpid") if "zpid" in header else None
        wanted = set()
        for line in (text[1:] if col is not None else text):
            line = line.strip()
            if line:
                wanted.add(line.split(",")[col] if col is not None else line)
        zpid_dirs = [d for d in zpid_dirs if d.name in wanted]
        print(f"Restricted to {len(zpid_dirs)} of {len(wanted)} listed zpids "
              f"(the rest have no images/<zpid>/ folder).")
    random.Random(args.seed).shuffle(zpid_dirs)

    lock = threading.Lock()
    write_header = not args.out_csv.exists()
    csv_f = open(args.out_csv, "a", newline="", encoding="utf-8")
    raw_f = open(args.raw_jsonl, "a", encoding="utf-8")
    writer = csv.DictWriter(csv_f, fieldnames=CSV_FIELDS)
    if write_header:
        writer.writeheader()

    def record(rel, zpid, result):
        """Write one classification result + update shared state (under lock)."""
        label = result["parsed"]["room_type"]
        if label not in CLASSIFY_LABELS:
            label = "other"
        cost = compute_cost(CLASSIFIER_MODEL, result["input_tokens"], result["output_tokens"],
                            result["cached_input_tokens"], result["cache_write_tokens"])
        other_label = result["parsed"].get("other_label") or ""
        with lock:
            writer.writerow({
                "image_path": rel, "zpid": zpid, "predicted_label": label,
                "other_label": other_label if label == "other" else "",
                "response_model": result["response_model"],
                "input_tokens": result["input_tokens"],
                "output_tokens": result["output_tokens"],
                "cached_input_tokens": result["cached_input_tokens"],
                "latency_s": round(result["latency_s"], 3),
                "cost_usd": f"{cost:.6f}",
            })
            csv_f.flush()
            raw_f.write(json.dumps({"image_path": rel, "zpid": zpid,
                                    "raw": json.loads(result["raw"])}) + "\n")
            raw_f.flush()
            done.add(rel)
            state["calls"] += 1
            if label in zpids_by_type:
                zpids_by_type[label].add(zpid)
        return label

    def process_property(zdir):
        """Classify a property's images until all 4 room types are seen or exhausted.
        Seeds seen-types from any prior run so an already-satisfied property does
        no new work on resume (true idempotency)."""
        zpid = zdir.name
        images = sorted(p for p in zdir.iterdir() if p.suffix.lower() in VALID_EXTS)
        types_here = {t for t in zpid_types.get(zpid, set()) if t in ROOM_TYPES}
        n = 0
        for img in images:
            if types_here.issuperset(ROOM_TYPES):  # early stop: got one of each
                break
            rel = str(img.relative_to(args.images_dir.parent))
            with lock:
                if rel in done:
                    continue
                if state["calls"] >= args.max_calls:
                    break
            label = record(rel, zpid, classify_image(img))
            n += 1
            if label in ROOM_TYPES:
                types_here.add(label)
        with lock:
            counts = {t: len(z) for t, z in zpids_by_type.items()}
            print(f"[{state['calls']}/{args.max_calls}] {zpid}: {n} classified "
                  f"(types {sorted(types_here)}) | zpids per type: {counts}")

    try:
        it = iter(zpid_dirs)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = set()
            for _ in range(args.workers):
                z = next(it, None)
                if z is not None:
                    futures.add(pool.submit(process_property, z))
            stop_reason = "exhausted all properties"
            while futures:
                fin, futures = wait(futures, return_when=FIRST_COMPLETED)
                for f in fin:
                    f.result()  # surface worker exceptions
                with lock:
                    met = quotas_met(zpids_by_type, args.target_zpids_per_type)
                    capped = state["calls"] >= args.max_calls
                if met or capped:
                    stop_reason = "quotas met" if met else f"reached call cap ({args.max_calls})"
                    continue  # don't submit more; drain in-flight
                for _ in range(len(fin)):
                    z = next(it, None)
                    if z is not None:
                        futures.add(pool.submit(process_property, z))
    finally:
        csv_f.close()
        raw_f.close()

    print(f"\nDone ({stop_reason}). Final per-type distinct-zpid counts:",
          {t: len(z) for t, z in zpids_by_type.items()})
    print(f"Total classification calls recorded: {state['calls']}")


if __name__ == "__main__":
    main()
