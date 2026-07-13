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
import datetime as dt
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
IMAGES_ROOT = REPO_ROOT
MODEL = "gemini-3.5-flash"
ROOM_ORDER = ["kitchen", "bathroom", "bedroom", "living_room"]
MAX_LEVEL = 8  # every room scores on the same 1–8 scale (kitchen included)

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
    ap.add_argument("--all", action="store_true", help="rescore all four rooms")
    ap.add_argument("--limit", type=int, default=0, help="cap images per room (0 = all)")
    ap.add_argument("--workers", type=int, default=12, help="concurrent API calls")
    ap.add_argument("--dry-run", action="store_true", help="no API calls, fake scores")
    ap.add_argument("--label", default="", help="tab label for this run's page")
    args = ap.parse_args()

    manifest = pd.read_csv(HERE / "data" / "sample_manifest.csv")
    counts = manifest["room_type"].value_counts().to_dict()

    if args.all:
        rooms = ROOM_ORDER[:]
    elif args.rooms:
        rooms = resolve_named(args.rooms)
    else:
        rooms = pick_rooms(counts)
    if not rooms:
        sys.exit("no rooms selected.")

    print(f"\nRescoring {rooms} with {MODEL} "
          f"({'DRY RUN' if args.dry_run else '1 call/image'})...\n")

    tasks = []
    for room in rooms:
        imgs = manifest[manifest["room_type"] == room]
        if args.limit:
            imgs = imgs.head(args.limit)
        for _, m in imgs.iterrows():
            tasks.append((room, m["image_path"], str(m["zpid"])))

    rows = []
    total_cost = 0.0

    def run(task):
        room, image_path, zpid = task
        res = dry_score(room) if args.dry_run else score_one(room, image_path)
        return room, image_path, zpid, res

    done = 0
    if args.dry_run:
        results = [run(t) for t in tasks]
    else:
        results = []
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(run, t): t for t in tasks}
            for fut in as_completed(futs):
                results.append(fut.result())
                done += 1
                print(f"  scored {done}/{len(tasks)}", end="\r", flush=True)
        print()

    for room, image_path, zpid, res in results:
        total_cost += res["cost_usd"]
        rows.append({"image_path": image_path, "zpid": zpid, "room_type": room, **res})

    run_df = pd.DataFrame(rows)
    # keep a stable order: room, then image
    run_df = run_df.sort_values(["room_type", "image_path"]).reset_index(drop=True)

    ts = dt.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    run_dir = HERE / "runs" / ts.replace(":", "")
    run_dir.mkdir(parents=True, exist_ok=True)
    run_df.to_csv(run_dir / "scores.csv", index=False)

    label = args.label or (("dry: " if args.dry_run else "") +
                           f"{'+'.join(r.replace('_room','') for r in rooms)} · {ts[5:16]}")
    cost_for_page = None if args.dry_run else total_cost
    page_file = gallery.build_page(run_df, rooms, cost_for_page, label, ts)

    m = gallery.load_manifest()
    m.append({"file": page_file, "label": label, "timestamp": ts, "rooms": rooms,
              "model": MODEL, "cost_usd": cost_for_page})
    gallery.save_manifest(m)
    gallery.rebuild_index()

    print(f"\nDone. {len(run_df)} images scored.")
    if not args.dry_run:
        print(f"Run cost: ${total_cost:.4f}")
    print(f"Scores:   {run_dir / 'scores.csv'}")
    print(f"Page:     gallery/pages/{page_file}")
    print(f"Gallery:  {gallery.INDEX}  (open in a browser; newest tab is selected)")


if __name__ == "__main__":
    main()
