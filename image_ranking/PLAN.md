# Image ranking — production pipeline plan

Self-contained luxury-scoring pipeline for the AVM. Lives in the repo root under
`image_ranking/` and does **not** reference anything under `Research/` (that was
experimentation). Prompts + grids are copied in, not imported.

**Goal:** assign each relevant listing photo a **luxury score 1–8**, then roll
those up into per-listing features for the AVM regression (`regression.py`).

All model calls go through the **Gemini Batch API** (50% cheaper than live).
Keys come from `config.Config` (never `.env` directly).

---

## Inputs (already in place)

- `data/cleaned_sold.csv` — `all_images` is the only relevant image column.
- `split_data.py` → `data/image_links.csv` (rows with images).
- `download_images.py` → `images/<zpid>/<zpid>_{N|hero}.<ext>`.
- **5,477 listings, 76,499 images** on disk.

## Two stages

1. **Classify** *every* image → room type (kitchen / bathroom / bedroom /
   living_room / other, `other` carries a free-text `other_label` e.g.
   "dining room", "floorplan", "exterior").
2. **Score** only the 4 target room types → luxury 1–8 against that room's
   calibration grid + prompt.

Stage 2 depends on Stage 1's labels, so the pipeline is **two sequential batch
jobs** (each batch targets ~24h, expires at 48h).

---

## Folder layout

```
image_ranking/
  PLAN.md                    # this file
  config_pipeline.py         # models, resolutions, thinking, chunk sizes, paths
  gemini_client.py           # Gemini client, image resize+encode, schemas, cost, result parsing
  rooms/<room>/prompt.txt    # copied verbatim from Research/gemini_image_test/rooms
  rooms/<room>/grid.png      # calibration grid per room
  prep_images.py             # images/ -> images_work/ (resized; transport only)
  stage1_build.py            # build Stage-1 batch JSONL chunks (all images)
  stage2_build.py            # build Stage-2 batch JSONL chunks (room images only)
  batch_runner.py            # submit / poll / download a JSONL batch (shared by both stages)
  stage1_parse.py            # results JSONL -> outputs/classifications.csv
  stage2_parse.py            # results JSONL -> outputs/scores.csv
  consolidate.py             # scores.csv -> outputs/image_features.csv (per-zpid, for AVM)
  outputs/
    classifications.csv  scores.csv  image_features.csv
    raw/*.jsonl          batch_jobs.json   # job ids/state for resume
```

---

## Model config (measured — see `Research/stage1_image_test/TESTS.md`)

| | model | media_resolution | thinking | measured tokens |
|---|---|---|---|---|
| Stage 1 | gemini-3.5-flash | **LOW** (both) | **default**, no cap | 362 in / 126 out |
| Stage 2 | gemini-3.5-flash | **HIGH** grid + **HIGH** target | default | 3,377 in / 722 out |

- `media_resolution` set globally in `GenerateContentConfig` (per-part optional).
- **No context caching** — the Stage-2 reusable prefix (≤2,750 tok) is under
  3.5-flash's 4,096 cache minimum. Confirmed dead; not implemented.
- **No thinking cap** — `thinking_budget` isn't a proportional cap on 3.5-flash
  (any nonzero value just triggers reduced thinking); default's thinking tail is
  already short. Never cap with `max_output_tokens` (truncates the JSON answer).

### Stage 1 request
`system_instruction` = 103-tok classify prompt; `contents` = [target image
(inline, LOW)] ; strict schema:
```json
{ "room_type": "kitchen|bathroom|bedroom|living_room|other",
  "other_label": "string|null" }
```
`other_label` is a free string when `room_type == other`. Promoting a value
(e.g. `floorplan`) to a first-class label later = add it to the enum; Stage 2 is
unaffected.

### Stage 2 request
`system_instruction` = that room's `prompt.txt`; `contents` = [grid intro text,
**grid image (Files-API URI, HIGH)**, "TARGET TO SCORE" text, target image
(inline, HIGH)]; strict schema per room:
```json
{ "is_<room>": true, "other_room": "string|null",
  "reasoning": "string|null", "score": 1 }   // 1..8, null if is_<room> false
```

---

## Image delivery (batch mechanics)

Constraints: batch input JSONL ≤ **2 GB**; Files API = 20 GB/project, 2 GB/file,
**48 h** retention, free. `media_resolution` sets tokens, so resizing is purely to
fit these limits.

**Approach:**
- `prep_images.py` writes a resized working copy (~1024 px longest side) once —
  keeps JSONL small, no quality cost (HIGH caps at 1120 tok ≈ ~900 px anyway).
- **Targets = inline base64** in the JSONL, **chunked** so each file < ~1.5 GB
  (≈ 3–4 Stage-1 chunks, ~2 Stage-2 chunks). Inline avoids the Files 48 h expiry
  race entirely and keeps each chunk independently resumable.
- **Stage-2 grid = Files API**, uploaded once per room (4 files, re-uploaded per
  run — trivially inside 48 h/20 GB), referenced by URI in every request so the
  ~1120-tok grid isn't re-embedded 41k times.

`batch_runner.py`: `client.batches.create(model, src=<uploaded jsonl>)` → poll
`JOB_STATE_*` → `client.files.download` the results JSONL. Job ids + state saved
to `outputs/batch_jobs.json` for resume; already-processed `image_path` keys are
skipped on re-run (idempotent).

---

## Outputs

- **`classifications.csv`** — one row / all 76,499 images: `image_path, zpid,
  room_type, other_label, input_tokens, output_tokens, cost_usd`. + `raw/stage1_*.jsonl`.
- **`scores.csv`** — one row / scored room image: `image_path, zpid, room_type,
  score, reasoning, is_room, other_room, input_tokens, output_tokens, cost_usd`.
  + `raw/stage2_*.jsonl`.
- **`image_features.csv`** — the AVM deliverable, one row per `zpid`:
  `kitchen_score, bathroom_score, bedroom_score, living_room_score` (aggregate =
  **max** per room by default; mean/count also emitted), joined into
  `regression.py`.

---

## Cost (measured, batch pricing)

| stage | images | $/image | total |
|---|---|---|---|
| Stage 1 | 76,499 | $0.00084 | **~$64** |
| Stage 2 | ~41,300 (≈54%) | $0.00578 | **~$239** |
| | | | **≈ $300** |

Range ~$265–$305 depending on the true room fraction (±5%). **Batch requires a
funded Gemini billing account** (same balance as live); fund enough to cover a
whole chunk before submitting it — jobs bill at completion and fail on credit
exhaustion.

---

## Validation before the full run (open items)

1. **Stage-1 accuracy** — fill the `truth` column in
   `Research/stage1_image_test/predictions.csv` (or eyeball `gallery.html`);
   confusion matrix, watch room→other and wrong-room errors. *(Agreement between
   low/default thinking is already 99%, but that's not ground truth.)*
2. **2.5-flash vs 3.5-flash for Stage 1** — 2.5 is ~$6.5 and cacheable; decide on
   accuracy, not cost (small swing).
3. **Smoke test each stage** on ~1 chunk (`--limit`) before the full submit.

## Explicit non-goals

- No S3 / cloud image hosting — local `images/` + inline/Files is sufficient; the
  `all_images` URLs are **not** usable as remote references (Gemini won't fetch them).
- No context caching (see above).
- Not reusing `Research/` code at import time (prompts/grids are copied in).
