#!/usr/bin/env python3
"""Rescore selected rooms with gemini-3.5-flash and append a page to the gallery.

Workflow this supports:
  1. Edit a room's prompt in rooms/<room>/prompt.txt (grid.png sits beside it).
  2. Run this and pick which rooms to rescore (multi-select, or pass names/--all).
  3. It scores that room's sample images ONE call each with gemini-3.5-flash,
     saves the run under runs/<timestamp>/, and appends a new tab/page to
     gallery/index.html showing the fresh scores vs the human rankings and the
     run's total cost. Each run is one tab, oldest to newest.

Everything reuses the scoring plumbing in model_comparison/common.py (same
message layout, schema, retry, and per-call cost model).

Examples:
    python rescore.py                 # interactive room picker
    python rescore.py living_room     # rescore just the living room
    python rescore.py kitchen bath    # prefix match: kitchen + bathroom
    python rescore.py --all
    python rescore.py living_room --limit 3   # first 3 images (cheap smoke test)
    python rescore.py living_room --dry-run    # no API calls, fake scores
"""
import argparse
import csv
import datetime as dt
import json
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
IMAGES_ROOT = REPO_ROOT
MODEL = "gemini-3.5-flash"
ROOM_ORDER = ["kitchen", "bathroom", "bedroom", "living_room"]
MAX_LEVEL = 8  # every room scores on the same 1–8 scale (kitchen included)

# scores.csv column order — written incrementally so an interrupted run is resumable
SCORE_FIELDS = ["image_path", "zpid", "room_type", "score", "reasoning", "is_room",
                "other_room", "input_tokens", "output_tokens", "cost_usd"]


def _ts_from_dirname(name: str) -> str:
    """runs/ dir name '2026-07-14T144924' -> timestamp '2026-07-14T14:49:24'."""
    d, _, t = name.partition("T")
    return f"{d}T{t[:2]}:{t[2:4]}:{t[4:6]}" if len(t) >= 6 else name

# reuse the scoring machinery from the model-comparison experiment
sys.path.insert(0, str(REPO_ROOT / "Research" / "model_comparison"))
import common  # noqa: E402
import gallery  # noqa: E402  (sibling module)


def load_assets(room: str):
    """prompt.txt + grid.png from THIS experiment's editable room folder."""
    d = HERE / "rooms" / room
    return (d / "prompt.txt").read_text(encoding="utf-8"), d / "grid.png"


def luxury_schema(room: str) -> dict:
    """Structured-output schema matching each prompt's declared JSON exactly:
    the living-room prompt's four fields, in order —
        is_<room>   bool     (is_kitchen / is_bathroom / is_bedroom / is_living_room)
        other_room  str|null (a concise room type when is_<room> is false)
        reasoning   str|null (luxury reasoning when is_<room> is true)
        score       int|null (1..max_level when is_<room> is true, else null)
    Nothing else — the prompt itself, not a code-appended note, defines the shape.
    Identical for every room: same four fields, same 1–{MAX_LEVEL} score range.
    """
    is_field = f"is_{room}"
    return {
        "type": "object",
        "properties": {
            is_field:     {"type": "boolean"},
            "other_room": {"type": "string", "nullable": True},
            "reasoning":  {"type": "string", "nullable": True},
            "score":      {"type": "integer", "minimum": 1, "maximum": MAX_LEVEL,
                           "nullable": True},
        },
        "required": [is_field, "other_room", "reasoning", "score"],
        "additionalProperties": False,
    }


def score_one(room: str, image_path: str):
    """One gemini-3.5-flash call for a single image; returns a result row dict."""
    prompt_text, grid_path = load_assets(room)
    label = room.replace("_", " ").upper()
    parts = [
        ("text", f"REFERENCE GRID (anchor examples, Level 1 = lowest ... Level {MAX_LEVEL} = highest):"),
        ("image", grid_path),
        ("text", f"{label} TO SCORE:"),
        ("image", (IMAGES_ROOT / image_path).resolve()),
    ]
    r = common.call_with_retry(
        MODEL,
        prompt_text,  # the prompt fully specifies the output; no appended room-judgment note
        parts,
        luxury_schema(room),
        "luxury_score",
        detail=common.IMAGE_DETAIL_SCORING,
        cache_key=f"gemini-image-test-{room}-v1",
    )
    p = r["parsed"]
    cost = common.compute_cost(
        MODEL, r["input_tokens"], r["output_tokens"],
        r.get("cached_input_tokens", 0), r.get("cache_write_tokens", 0))
    return {
        "score": p.get("score"),
        "reasoning": p.get("reasoning") or "",
        "is_room": p.get(f"is_{room}"),
        "other_room": p.get("other_room"),
        "input_tokens": r["input_tokens"], "output_tokens": r["output_tokens"],
        "cost_usd": cost,
    }


def dry_score(room: str):
    s = random.randint(1, MAX_LEVEL)
    return {"score": s, "reasoning": "(dry run — fabricated score)",
            "is_room": True, "other_room": None,
            "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}


def pick_rooms(counts: dict) -> list:
    """Interactive multi-select when no rooms were passed on the command line."""
    print("\nSelect rooms to rescore (space/comma separated numbers, or 'all'):\n")
    for i, r in enumerate(ROOM_ORDER, 1):
        print(f"  {i}) {r:<12} ({counts.get(r, 0)} images)")
    raw = input("\n> ").strip().lower()
    if raw in ("all", "*", ""):
        return ROOM_ORDER[:]
    picked = []
    for tok in raw.replace(",", " ").split():
        if tok.isdigit() and 1 <= int(tok) <= len(ROOM_ORDER):
            picked.append(ROOM_ORDER[int(tok) - 1])
        else:
            picked += [r for r in ROOM_ORDER if r.startswith(tok)]
    # de-dup, keep canonical order
    return [r for r in ROOM_ORDER if r in set(picked)]


def resolve_named(names: list) -> list:
    picked = []
    for n in names:
        n = n.lower()
        picked += [r for r in ROOM_ORDER if r.startswith(n)]
    if not picked:
        sys.exit(f"no rooms matched {names}; valid: {ROOM_ORDER}")
    return [r for r in ROOM_ORDER if r in set(picked)]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rooms", nargs="*", help="room names/prefixes; omit for picker")
    ap.add_argument("--dataset", default="set_a",
                    help="which datasets/<name>/ to score (default set_a)")
    ap.add_argument("--all", action="store_true", help="rescore all four rooms")
    ap.add_argument("--limit", type=int, default=0, help="cap images per room (0 = all)")
    ap.add_argument("--workers", type=int, default=12, help="concurrent API calls")
    ap.add_argument("--dry-run", action="store_true", help="no API calls, fake scores")
    ap.add_argument("--label", default="", help="tab label for this run's page")
    ap.add_argument("--resume", nargs="?", const="latest", default=None, metavar="RUN",
                    help="resume an interrupted run instead of starting fresh: bare "
                         "--resume continues this dataset's most recent run; "
                         "--resume <runs-folder-name> targets a specific one. "
                         "Already-scored images are skipped.")
    args = ap.parse_args()

    gallery.set_dataset(args.dataset)
    ds_root = HERE / "datasets" / args.dataset
    manifest = pd.read_csv(ds_root / "data" / "sample_manifest.csv")
    counts = manifest["room_type"].value_counts().to_dict()

    # locate the run dir: resume an existing one, or start a fresh timestamped run
    runs_root = ds_root / "runs"
    meta = {}
    if args.resume:
        if args.resume == "latest":
            cands = sorted(d for d in runs_root.glob("*") if (d / "scores.csv").exists())
            if not cands:
                sys.exit(f"nothing to resume under {runs_root}")
            run_dir = cands[-1]
        else:
            run_dir = runs_root / args.resume.replace(":", "")
            if not run_dir.exists():
                sys.exit(f"no such run to resume: {run_dir}")
        mp = run_dir / "run_meta.json"
        meta = json.loads(mp.read_text()) if mp.exists() else {}
        ts = meta.get("timestamp") or _ts_from_dirname(run_dir.name)
        print(f"\nResuming run {run_dir.name} — already-scored images are skipped.")
    else:
        ts = dt.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        run_dir = runs_root / ts.replace(":", "")
        run_dir.mkdir(parents=True, exist_ok=True)

    if args.all:
        rooms = ROOM_ORDER[:]
    elif args.rooms:
        rooms = resolve_named(args.rooms)
    elif args.resume and meta.get("rooms"):
        rooms = meta["rooms"]
    else:
        rooms = pick_rooms(counts)
    if not rooms:
        sys.exit("no rooms selected.")

    label = args.label or meta.get("label") or (
        ("dry: " if args.dry_run else "")
        + f"{'+'.join(r.replace('_room','') for r in rooms)} · {ts[5:16]}")
    if not args.resume:  # record what this run is, so it can be resumed later
        (run_dir / "run_meta.json").write_text(
            json.dumps({"timestamp": ts, "rooms": rooms, "label": label}))

    print(f"Rescoring {rooms} with {MODEL} "
          f"({'DRY RUN' if args.dry_run else '1 call/image'})...\n")

    tasks = []
    for room in rooms:
        imgs = manifest[manifest["room_type"] == room]
        if args.limit:
            imgs = imgs.head(args.limit)
        for _, mrow in imgs.iterrows():
            tasks.append((room, mrow["image_path"], str(mrow["zpid"])))

    scores_csv = run_dir / "scores.csv"
    done = set(pd.read_csv(scores_csv)["image_path"]) if scores_csv.exists() else set()
    pending = [t for t in tasks if t[1] not in done]
    print(f"{len(done)} already scored, {len(pending)} to go.\n")

    # Crash-safe: each image is written and flushed the moment it completes, so a
    # Ctrl-C / crash / API error keeps every image done so far. Rerun with
    # --resume to score only what's missing (never re-pays for finished images).
    lock = threading.Lock()
    new_file = not scores_csv.exists() or scores_csv.stat().st_size == 0
    fh = open(scores_csv, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(fh, fieldnames=SCORE_FIELDS)
    if new_file:
        writer.writeheader()
        fh.flush()
    counter = {"n": len(done)}

    def run_and_write(task):
        room, image_path, zpid = task
        res = dry_score(room) if args.dry_run else score_one(room, image_path)
        row = {"image_path": image_path, "zpid": zpid, "room_type": room, **res}
        with lock:
            writer.writerow({k: row.get(k, "") for k in SCORE_FIELDS})
            fh.flush()
            counter["n"] += 1
            print(f"  scored {counter['n']}/{len(tasks)}", end="\r", flush=True)
        return row

    try:
        if args.dry_run:
            for t in pending:
                run_and_write(t)
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                futs = {ex.submit(run_and_write, t): t for t in pending}
                for fut in as_completed(futs):
                    fut.result()
        if pending:
            print()
    finally:
        fh.close()

    run_df = pd.read_csv(scores_csv).sort_values(
        ["room_type", "image_path"]).reset_index(drop=True)
    total_cost = float(run_df["cost_usd"].sum()) if "cost_usd" in run_df else 0.0
    cost_for_page = None if args.dry_run else total_cost

    m = gallery.load_manifest()
    existing = next((r for r in m if r.get("timestamp") == ts), None)
    page_file = gallery.build_page(run_df, rooms, cost_for_page, label, ts,
                                   page_file=existing["file"] if existing else None)
    entry = {"file": page_file, "label": label, "timestamp": ts, "rooms": rooms,
             "model": MODEL, "cost_usd": cost_for_page}
    if existing:
        existing.update(entry)
    else:
        m.append(entry)
    gallery.save_manifest(m)
    gallery.rebuild_index()

    print(f"\nDone. {len(run_df)} images scored ({len(pending)} this run).")
    if not args.dry_run:
        print(f"Run cost: ${total_cost:.4f}")
    print(f"Scores:   {scores_csv}")
    print(f"Page:     gallery/pages/{page_file}")
    print(f"Gallery:  {gallery.INDEX}  (open in a browser; newest tab is selected)")


if __name__ == "__main__":
    main()
