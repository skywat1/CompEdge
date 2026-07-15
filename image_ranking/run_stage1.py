#!/usr/bin/env python3
"""Stage 1 — classify every listing image by room type (Gemini Batch API).

Modes:
  --live-check N   classify N images LIVE (fast logic proof, ~$0)
  --smoke N        one small batch of N images -> classifications_smoke.csv (~$0.20)
  --full           chunked batch over ALL images -> data/classifications.csv (~$64)

Batch jobs are async (target 24h, usually faster). --smoke/--full submit then
poll to completion. Output CSV is append-only and skips already-done image_paths,
so re-running resumes.
"""
import argparse
import concurrent.futures as cf
import csv
import json
import time
from collections import Counter
from pathlib import Path

from config_pipeline import (CLASSIFICATIONS_CSV, CLASSIFY_FIELDS, CHUNK_MAX_BYTES,
                             CHUNK_MAX_LINES, IMAGES_DIR, OUTPUTS_DIR, REPO_ROOT,
                             STAGE1_MODEL, VALID_EXTS)
import batch_runner as br
import gemini_client as gc


# ---------------------------------------------------------------------------
# Image walking / CSV helpers
# ---------------------------------------------------------------------------
def iter_images():
    """Yield repo-relative image paths ('images/<zpid>/<file>'), sorted stably."""
    for zdir in sorted(p for p in IMAGES_DIR.iterdir() if p.is_dir()):
        for img in sorted(p for p in zdir.iterdir() if p.suffix.lower() in VALID_EXTS):
            yield str(img.relative_to(REPO_ROOT))


def load_done(csv_path: Path) -> set:
    if not csv_path.exists():
        return set()
    with open(csv_path, newline="") as f:
        return {r["image_path"] for r in csv.DictReader(f)}


def append_rows(csv_path: Path, rows: list):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new = not csv_path.exists() or csv_path.stat().st_size == 0
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CLASSIFY_FIELDS)
        if new:
            w.writeheader()
        w.writerows(rows)


def write_jsonl(image_paths: list, out_path: Path) -> int:
    """Write one batch request per image (resize+encode happens here); return byte
    size. Prints encoding progress since this is the slow local step."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(image_paths)
    t0 = time.monotonic()
    with open(out_path, "w") as f:
        for i, p in enumerate(image_paths, 1):
            f.write(json.dumps(gc.build_jsonl_line(p)) + "\n")
            if i % 500 == 0 or i == n:
                rate = i / max(time.monotonic() - t0, 1e-6)
                eta = (n - i) / rate
                print(f"    encoding {i}/{n}  ({rate:.0f}/s, eta {eta:4.0f}s)",
                      end="\r", flush=True)
    if n >= 500:
        print()
    return out_path.stat().st_size


def parse_results_file(results_path: Path, source_chunk: str):
    """Parse a downloaded results JSONL into (rows, errors). cost_usd bills on
    answer + thinking tokens together; is_hero comes from the filename."""
    rows, errors = [], []
    for line in results_path.read_text().splitlines():
        if not line.strip():
            continue
        r = gc.parse_result_obj(json.loads(line))
        if r["room_type"] is None:
            errors.append(r)
            continue
        key = r["key"]
        billed_out = r["output_tokens"] + r["thinking_tokens"]
        rows.append({
            "image_path": key,
            "zpid": Path(key).parent.name,
            "is_hero": int(Path(key).stem.endswith("_hero")),
            "room_type": r["room_type"],
            "other_label": r["other_label"],
            "response_model": r["response_model"],
            "input_tokens": r["input_tokens"],
            "output_tokens": r["output_tokens"],
            "thinking_tokens": r["thinking_tokens"],
            "cost_usd": f"{gc.compute_cost(r['input_tokens'], billed_out):.6f}",
            "source_chunk": source_chunk,
        })
    return rows, errors


def summarize(rows, errors, label):
    dist = Counter(r["room_type"] for r in rows)
    cost = sum(float(r["cost_usd"]) for r in rows)
    in_t = sum(r["input_tokens"] for r in rows)
    out_t = sum(r["output_tokens"] for r in rows)
    think_t = sum(r.get("thinking_tokens", 0) for r in rows)
    n = len(rows)
    print(f"\n{label}: {n} classified, {len(errors)} errors")
    print("  labels:", dict(dist))
    if n:
        print(f"  tokens/img: in {in_t/n:.0f}  out {out_t/n:.0f}  think {think_t/n:.0f}")
        print(f"  cost (batch): ${cost:.4f}  (${cost/n:.6f}/img)")
    for e in errors[:5]:
        print("  ERROR", e["key"], e["error"][:120])


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------
def live_check(n: int):
    imgs = []
    for p in iter_images():
        imgs.append(p)
        if len(imgs) >= n:
            break
    print(f"Live-classifying {len(imgs)} images (gemini, LOW, default thinking)...")
    gc.client()  # force single-threaded client init before the pool
    rows, errors = [], []
    with cf.ThreadPoolExecutor(max_workers=10) as ex:
        futs = {ex.submit(gc.classify_live, REPO_ROOT / p): p for p in imgs}
        for fut in cf.as_completed(futs):
            p = futs[fut]
            try:
                r = fut.result()
                rows.append({"image_path": p, "room_type": r["room_type"],
                             "other_label": r["other_label"] or "",
                             "input_tokens": r["input_tokens"],
                             "output_tokens": r["output_tokens"],
                             "thinking_tokens": r["thinking_tokens"],
                             "cost_usd": f"{gc.compute_cost(r['input_tokens'], r['output_tokens'] + r['thinking_tokens']):.6f}"})
            except Exception as e:  # noqa: BLE001
                errors.append({"key": p, "error": f"{type(e).__name__}: {e}"})
    bad = [r for r in rows if r["room_type"] not in set(gc.LABELS)]
    summarize(rows, errors, "LIVE CHECK")
    print(f"  all labels within taxonomy: {not bad and not errors}")
    if bad:
        print("  OUT-OF-TAXONOMY:", bad)


def smoke(n: int):
    imgs = []
    for p in iter_images():
        imgs.append(p)
        if len(imgs) >= n:
            break
    jsonl = OUTPUTS_DIR / "smoke_requests.jsonl"
    size = write_jsonl(imgs, jsonl)
    print(f"Built {len(imgs)}-image JSONL ({size/1e6:.1f} MB) -> {jsonl}")
    br.submit("smoke", jsonl, STAGE1_MODEL)
    job = br.load_jobs()["smoke"]["job_name"]
    res = br.results_path("smoke")
    br.wait(job, res)
    rows, errors = parse_results_file(res, "smoke")
    out = OUTPUTS_DIR / "classifications_smoke.csv"
    if out.exists():
        out.unlink()
    append_rows(out, rows)
    summarize(rows, errors, "SMOKE")
    print(f"  wrote {out}")


def plan_chunks():
    """Deterministic, STABLE chunking over ALL images (sorted). Same images on
    disk -> same chunk_ids every run, so submit and collect always agree and a
    chunk is never re-submitted/re-paid on resume."""
    chunks, cur, cur_bytes = [], [], 0
    for p in iter_images():
        cur.append(p)
        cur_bytes += 130_000  # measured ~102 KB/image (768px JPEG b64) + headroom
        if len(cur) >= CHUNK_MAX_LINES or cur_bytes >= CHUNK_MAX_BYTES:
            chunks.append(cur); cur, cur_bytes = [], 0
    if cur:
        chunks.append(cur)
    return [(f"full_{i:03d}", c) for i, c in enumerate(chunks)]


def submit_all():
    """Phase 1: submit every chunk not already submitted, then return. Idempotent
    — re-running skips chunks that already have a job. Once this finishes the jobs
    live on Google's servers, so the terminal can be closed."""
    jobs = br.load_jobs()
    plan = plan_chunks()
    print(f"{len(plan)} chunks over {sum(len(c) for _, c in plan)} images.")
    for k, (cid, chunk) in enumerate(plan, 1):
        if jobs.get(cid, {}).get("job_name"):
            print(f"[{k}/{len(plan)}] {cid}: already submitted ({jobs[cid].get('state')}).")
            continue
        print(f"[{k}/{len(plan)}] building {cid} ({len(chunk)} images)...")
        jsonl = OUTPUTS_DIR / f"{cid}_requests.jsonl"
        size = write_jsonl(chunk, jsonl)
        print(f"    JSONL {size/1e6:.0f} MB — uploading + submitting...")
        br.submit(cid, jsonl, STAGE1_MODEL)
        jobs = br.load_jobs()
    print("\nAll chunks submitted. Jobs now run on Google (results kept 6 weeks) — "
          "safe to close the terminal. Run --collect (or --full) later to gather results.")


def collect():
    """Phase 2: download+parse any SUCCEEDED chunk not yet in the CSV. Idempotent,
    never re-submits. Returns (terminal_chunks, total_chunks)."""
    csv_path = CLASSIFICATIONS_CSV
    jobs = br.load_jobs()
    plan_ids = [cid for cid, _ in plan_chunks()]
    n_term = 0
    for cid in plan_ids:
        jn = jobs.get(cid, {}).get("job_name")
        if not jn:
            print(f"  {cid}: not submitted yet.")
            continue
        live = br.state(jn)
        br.record(cid, state=live)
        if live in br.TERMINAL:
            n_term += 1
        if live != "JOB_STATE_SUCCEEDED":
            print(f"  {cid}: {live}")
            continue
        res = br.results_path(cid)
        if not res.exists():
            br.download_results(jn, res)
        rows, errors = parse_results_file(res, cid)
        rows = [r for r in rows if r["image_path"] not in load_done(csv_path)]
        if rows:
            append_rows(csv_path, rows)
        summarize(rows, errors, cid)
    return n_term, len(plan_ids)


def full():
    """Convenience: submit everything, then poll+collect until all chunks reach a
    terminal state. Blocks — but the jobs survive if this process is killed; just
    re-run --full or --collect to resume with zero re-submission."""
    submit_all()
    while True:
        n_term, n_total = collect()
        if n_term >= n_total:
            break
        print(f"  {n_term}/{n_total} chunks terminal; waiting 60s...")
        time.sleep(60)
    csv_path = CLASSIFICATIONS_CSV
    print(f"\nDONE. classifications.csv rows: {len(load_done(csv_path))}")


def status():
    """Query Google for the live state of every recorded batch job, persist it to
    batch_jobs.json, and show overall progress. Safe to run anytime, even while a
    --full run is polling in another terminal."""
    jobs = br.load_jobs()
    if not jobs:
        print("No batch jobs recorded yet (nothing submitted).")
        return
    csv_path = CLASSIFICATIONS_CSV
    print(f"classifications.csv rows: {len(load_done(csv_path))}\n")
    all_done = True
    for cid, meta in jobs.items():
        jn = meta.get("job_name")
        live = br.state(jn) if jn else "?"
        br.record(cid, state=live)  # persist the live state
        downloaded = br.results_path(cid).exists()
        if live not in br.TERMINAL:
            all_done = False
        flag = "✓" if (live == "JOB_STATE_SUCCEEDED" and downloaded) else " "
        print(f" {flag} {cid:12} {live:22} results_downloaded={downloaded}  {jn}")
    print("\n" + ("ALL JOBS TERMINAL." if all_done else "some jobs still running."))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--live-check", type=int, metavar="N")
    g.add_argument("--smoke", type=int, nargs="?", const=250, metavar="N")
    g.add_argument("--submit-all", action="store_true",
                   help="phase 1: submit all chunks, then exit (terminal can close)")
    g.add_argument("--collect", action="store_true",
                   help="phase 2: download+parse any finished chunks, then exit")
    g.add_argument("--full", action="store_true",
                   help="submit-all + poll/collect until done (blocks)")
    g.add_argument("--status", action="store_true",
                   help="show the live state of recorded batch jobs and exit")
    args = ap.parse_args()
    if args.live_check is not None:
        live_check(args.live_check)
    elif args.smoke is not None:
        smoke(args.smoke)
    elif args.submit_all:
        submit_all()
    elif args.collect:
        collect()
    elif args.full:
        full()
    elif args.status:
        status()


if __name__ == "__main__":
    main()
