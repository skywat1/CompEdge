# image_ranking — Stage 1: image classification

Classifies every listing photo in `images/<zpid>/` by room type using the Gemini
**Batch API** (50% cheaper than live). Output feeds Stage 2 (luxury scoring) and,
ultimately, per-listing features for the AVM.

Self-contained — nothing here imports from `Research/`.

- **Model:** `gemini-3.5-flash`, `media_resolution=LOW`, default thinking.
- **Labels:** `kitchen, bathroom, bedroom, living_room, floorplan, other`
  (`other` carries a free-text `other_label`, e.g. `dining room`, `exterior`).
- **Scale / cost:** 76,499 images → **~$65** (batch), measured at ~$0.00086/image.

All commands are run from the **repo root** with the project venv:

```bash
venv/bin/python image_ranking/run_stage1.py <mode>
```

---

## Prerequisites

1. Images downloaded to `images/<zpid>/*` (via `split_data.py` + `download_images.py`).
2. `GEMINI_API_KEY` readable by the repo-root `config.py`.
3. **Funded Gemini billing** — batch draws from the same balance as live and bills
   at job completion. Add **~$80** (≈20% over the $65 estimate) before the full run.

---

## Commands

| mode | what it does | cost | time |
|---|---|---|---|
| `--live-check N` | Classify N images **live** (no batch). Logic/sanity proof. | ~$0 | ~1 min |
| `--smoke [N]` | One small batch (default N=250) → `outputs/classifications_smoke.csv`. | ~$0.20 | minutes |
| `--submit-all` | **Phase 1:** encode + submit all 7 chunks, then exit. | — | ~15–50 min* |
| `--collect` | **Phase 2:** download + parse any finished chunks → `classifications.csv`. | — | ~1–2 min |
| `--full` | `--submit-all` then poll/collect until done (**blocks**). | ~$65 | up to 24h |
| `--status` | Show live state of every submitted chunk. | ~$0 | seconds |

\* `--submit-all` is upload-bound: ~4 min to encode 76k images, the rest is
uploading ~7.5 GB to Google (depends on your upstream bandwidth).

---

## Recommended full run

Two-phase, so the long batch wait is decoupled from your terminal:

```bash
# Phase 1 — submit (keep terminal open until it prints "safe to close")
venv/bin/python image_ranking/run_stage1.py --submit-all

# ...jobs now run on Google (up to 24h; results kept 6 weeks). Close the terminal freely.

# Check progress anytime
venv/bin/python image_ranking/run_stage1.py --status

# Phase 2 — gather results once --status shows chunks SUCCEEDED (re-run as more finish)
# writes the deliverable to data/classifications.csv
venv/bin/python image_ranking/run_stage1.py --collect
```

**Fire-and-forget alternative** — one detached command that survives closing the
terminal and does both phases:

```bash
nohup venv/bin/python image_ranking/run_stage1.py --full > full.log 2>&1 &
tail -f full.log        # or: run_stage1.py --status
```

---

## How you know it's done

`--status` prints each chunk's live state and a footer:

```
 ✓ full_000     JOB_STATE_SUCCEEDED    results_downloaded=True  batches/...
ALL JOBS TERMINAL.
```

**Done** = every chunk `JOB_STATE_SUCCEEDED` with `results_downloaded=True` and
`ALL JOBS TERMINAL`. States progress `PENDING → RUNNING → SUCCEEDED`.

## Resuming / interruptions

Everything is idempotent and safe to re-run:

- A **submitted chunk survives** any process kill — the job runs on Google; re-running
  `--submit-all` skips chunks that already have a job (no re-submit, no re-pay).
- `--collect` skips chunks already downloaded and images already in the CSV.
- **If billing runs out mid-run:** completed chunks are already saved; unfinished
  requests simply aren't written. Top up, then re-run `--collect` (or `--full`) to
  finish only what's missing.

---

## Outputs

The **deliverable** goes to `data/classifications.csv` (alongside the other AVM
inputs; gitignored via `/data`). Intermediates stay in `image_ranking/outputs/`.

- **`data/classifications.csv`** — one row per image (the deliverable):

  | column | meaning |
  |---|---|
  | `image_path` | repo-relative path (the request/result key) |
  | `zpid` | listing id |
  | `is_hero` | 1 if the listing's hero photo (`*_hero`) |
  | `room_type` | one of the 6 labels |
  | `other_label` | free text when `room_type == other`, else empty |
  | `response_model` | exact model snapshot that answered |
  | `input_tokens` / `output_tokens` / `thinking_tokens` | usage (answer vs thinking split) |
  | `cost_usd` | batch cost, billed on input + (output + thinking) |
  | `source_chunk` | which batch chunk produced the row |

- **`raw/<chunk>_results.jsonl`** — full raw responses (kept for debugging).
- **`batch_jobs.json`** — chunk → job name/state bookkeeping (drives resume).
- **`*_requests.jsonl`** — per-chunk input files (~1.1 GB each; deletable after collect).

---

## Files

| file | role |
|---|---|
| `config_pipeline.py` | constants (model, resolution, labels, chunking, pricing, paths) |
| `gemini_client.py` | Gemini client, image encode, schema, cost, request build + result parse |
| `batch_runner.py` | stage-agnostic batch submit / poll / download + `batch_jobs.json` |
| `run_stage1.py` | the driver (all the modes above) |

Stage 2 (luxury scoring) will reuse `batch_runner.py` and the same skeleton; its
config lives under `STAGE2_*` constants (added later) so nothing here changes.
