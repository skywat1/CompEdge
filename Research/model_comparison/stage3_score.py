#!/usr/bin/env python3
"""
Stage 3 — score every manifest image 5 times per model (independent calls).

Each call sends the room rubric prompt + anchor grid + target image exactly as
score.py does, with strict structured output and temperature 0, on each of the
five models. Checkpoints to an append-only CSV keyed by
(model, image_path, replicate); completed calls are skipped on re-run. Full raw
API responses are appended to a per-run JSONL.

Usage:
    python stage3_score.py --manifest outputs/sample/sample_manifest.csv \
        --images-root .. --out-csv outputs/scores/scores.csv \
        --raw-jsonl outputs/scores/scores_raw.jsonl
"""

import argparse
import csv
import json
import signal
import threading
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait as fwait
from pathlib import Path

from common import MODELS, compute_cost, score_image, set_rate_limit

CSV_FIELDS = ["model", "response_model", "image_path", "zpid", "room_type", "replicate",
              "score", "level", "confidence", "valid", "room_type_judgment", "reasoning",
              "input_tokens", "output_tokens", "cached_input_tokens", "cache_write_tokens",
              "thought_tokens", "latency_s", "cost_usd", "error"]


def load_done(csv_path: Path):
    done = set()
    if csv_path.exists():
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if not row.get("error"):
                    done.add((row["model"], row["image_path"], row["replicate"]))
    return done


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True, help="Stage 2 sample manifest CSV")
    ap.add_argument("--images-root", type=Path, required=True,
                    help="Directory that manifest image paths are relative to (the repo root)")
    ap.add_argument("--out-csv", type=Path, required=True, help="Append-only score checkpoint CSV")
    ap.add_argument("--raw-jsonl", type=Path, required=True, help="Append-only raw response JSONL")
    ap.add_argument("--models", nargs="+", default=list(MODELS),
                    help=f"Models to run (default: all of {list(MODELS)})")
    ap.add_argument("--reps", type=int, default=5, help="Independent calls per image (default 5)")
    ap.add_argument("--openai-workers", type=int, default=3,
                    help="Concurrent OpenAI calls (default 3; TPM-limiter is the real gate)")
    ap.add_argument("--gemini-workers", type=int, default=6,
                    help="Concurrent Gemini calls, shared across Gemini models (default 6)")
    ap.add_argument("--anthropic-workers", type=int, default=6,
                    help="Concurrent Anthropic calls, shared across Anthropic models (default 6)")
    ap.add_argument("--tpm-limit", type=int, default=28000,
                    help="Client-side OpenAI tokens-per-minute cap to pace gpt-4o under your "
                         "account limit and avoid 429s (default 28000; raise on higher tiers, 0 disables)")
    args = ap.parse_args()

    for m in args.models:
        if m not in MODELS:
            ap.error(f"unknown model {m!r}; choose from {list(MODELS)}")

    set_rate_limit("openai", args.tpm_limit)

    with open(args.manifest, newline="", encoding="utf-8") as f:
        manifest = list(csv.DictReader(f))

    done = load_done(args.out_csv)
    total = len(manifest) * len(args.models) * args.reps

    # Build the outstanding task list, grouped by provider so each provider runs
    # in its own concurrent pool (wall-clock ~= slowest provider, not the sum).
    tasks_by_provider = defaultdict(list)
    for model in args.models:
        provider = MODELS[model]["provider"]
        for item in manifest:
            for rep in range(1, args.reps + 1):
                if (model, item["image_path"], str(rep)) in done:
                    continue
                tasks_by_provider[provider].append((model, item, rep))
    outstanding = sum(len(t) for t in tasks_by_provider.values())
    print(f"{len(manifest)} images x {len(args.models)} models x {args.reps} reps "
          f"= {total} calls ({len(done)} done, {outstanding} to run)")
    for prov, t in sorted(tasks_by_provider.items()):
        print(f"  {prov}: {len(t)} calls")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not args.out_csv.exists()
    csv_f = open(args.out_csv, "a", newline="", encoding="utf-8")
    raw_f = open(args.raw_jsonl, "a", encoding="utf-8")
    writer = csv.DictWriter(csv_f, fieldnames=CSV_FIELDS)
    if write_header:
        writer.writeheader()

    lock = threading.Lock()
    state = {"n": len(done)}
    stop = threading.Event()  # set on Ctrl-C so pending calls bail out fast

    def _sigint(signum, frame):
        # A busy ThreadPoolExecutor doesn't reliably surface KeyboardInterrupt
        # through future.result(), so we handle SIGINT explicitly: set the flag
        # (the main loop polls it) and restore the default handler so a second
        # Ctrl-C hard-exits.
        if stop.is_set():
            signal.signal(signal.SIGINT, signal.SIG_DFL)
        stop.set()
    signal.signal(signal.SIGINT, _sigint)

    def run_task(model, item, rep):
        if stop.is_set():  # don't start new calls once interrupted
            return
        img = args.images_root / item["image_path"]
        row = {"model": model, "image_path": item["image_path"],
               "zpid": item["zpid"], "room_type": item["room_type"],
               "replicate": rep, "error": ""}
        raw_line = None
        try:
            r = score_image(model, item["room_type"], img)
            p = r["parsed"]
            row.update({
                "response_model": r["response_model"],
                "score": p.get("score"), "level": p.get("level"),
                "confidence": p.get("confidence"), "valid": p.get("valid"),
                "room_type_judgment": p.get("room_type"),
                "reasoning": p.get("reasoning"),
                "input_tokens": r["input_tokens"],
                "output_tokens": r["output_tokens"],
                "cached_input_tokens": r["cached_input_tokens"],
                "cache_write_tokens": r["cache_write_tokens"],
                "thought_tokens": r["thought_tokens"],
                "latency_s": round(r["latency_s"], 3),
                "cost_usd": f"{compute_cost(model, r['input_tokens'], r['output_tokens'], r['cached_input_tokens'], r['cache_write_tokens']):.6f}",
            })
            raw_line = json.dumps({"model": model, "image_path": item["image_path"],
                                   "replicate": rep, "raw": json.loads(r["raw"])})
        except Exception as e:  # exhausted retries — record and move on
            row["error"] = str(e)[:500]
        with lock:
            writer.writerow(row)
            csv_f.flush()
            if raw_line is not None:
                raw_f.write(raw_line + "\n")
                raw_f.flush()
            state["n"] += 1
            print(f"[{state['n']}/{total}] {model} {item['room_type']:12s} "
                  f"{Path(item['image_path']).name} rep {rep}"
                  + (f"  ERROR: {row['error'][:80]}" if row["error"] else ""))

    provider_workers = {"openai": args.openai_workers,
                        "gemini": args.gemini_workers,
                        "anthropic": args.anthropic_workers}
    pools, futures = [], []
    for provider, tasks in tasks_by_provider.items():
        pool = ThreadPoolExecutor(max_workers=provider_workers.get(provider, 4))
        pools.append(pool)
        for (model, item, rep) in tasks:
            futures.append(pool.submit(run_task, model, item, rep))

    pending = set(futures)
    try:
        while pending:
            if stop.is_set():
                print("\nInterrupted — cancelling queued calls, letting in-flight "
                      "ones finish and save (resumable)...")
                for pool in pools:
                    pool.shutdown(wait=False, cancel_futures=True)  # drop not-yet-started
                break
            done_set, pending = fwait(pending, timeout=0.5, return_when=FIRST_COMPLETED)
            for f in done_set:
                f.result()  # surface any unexpected (non-API) worker exception
    finally:
        for pool in pools:
            pool.shutdown(wait=True)  # join in-flight; files stay open until now
        csv_f.close()
        raw_f.close()

    if stop.is_set():
        print(f"\nStopped after {state['n']}/{total} calls. "
              f"Re-run the same command to resume where it left off.")
    else:
        print("\nStage 3 complete.")


if __name__ == "__main__":
    main()
