# Stage-1 batch-pipeline prep — tests & measurements

Measurements taken while planning the **production** image-ranking pipeline
(`image_ranking/` in the repo root — see its `PLAN.md`). Goal: replace guesses
with real, measured numbers for cost and thinking behaviour before writing prod
code. All calls use **gemini-3.5-flash**, key from `config.Config.GEMINI_API_KEY`.

Run date: 2026-07-14. Pricing basis: batch tier = 50% of standard
(`gemini-3.5-flash`: input $0.75/1M, output $4.50/1M). Thinking tokens bill as
output.

Scripts in this folder (all self-contained, run with the repo-root venv,
e.g. `../../venv/bin/python stage1_cost_test.py`):

| script | what it measures |
|---|---|
| `count_prompt_tokens.py` | Stage-2 prompt+grid prefix token counts (cache-feasibility) |
| `stage1_cost_test.py` | Stage-1 cost, thinking-minimized vs default (100 imgs) |
| `stage1_accuracy_test.py` | Saves per-image predictions + gallery; low-vs-default agreement |
| `thinking_cap_test.py` | Whether `thinking_budget` acts as a thinking cap on 3.5-flash |

Artifacts: `predictions.csv` (100 Stage-1 predictions, both thinking modes, with
an empty `truth` column to score real accuracy) and `gallery.html` (image +
label, disagreements flagged) — for the still-open accuracy check.

---

## Dataset scale

- `data/cleaned_sold.csv` → `all_images` → `images/<zpid>/` (via `split_data.py`
  + `download_images.py`).
- **5,477 listings, 76,499 images** on disk.

## Test 1 — Media-resolution & tokenisation (from docs, confirmed)

- Gemini 3.x charges images by `media_resolution`, a per-image **token budget**
  (a *maximum*, not a flat charge): **LOW 280 / MEDIUM 560 / HIGH 1120**. Default
  ("unspecified") = 1120.
- On Gemini 3.x this budget — **not** the uploaded pixel size — sets the token
  cost. Resizing images only affects **transport** (batch/Files size limits),
  not the bill.
- (Older 2.5 scheme is dimension-based tiling: 258 tok if ≤384px, else 258/tile.)

## Test 2 — Stage-2 prefix token counts (`count_prompt_tokens.py`)

Reusable Stage-2 prefix = room prompt + grid (HIGH = 1120). Measured on the live
tokenizer:

| room | prompt tok | + grid HIGH | ≥ 4096? |
|---|---|---|---|
| kitchen | 1,169 | 2,289 | no |
| bathroom | 1,630 | 2,750 | no |
| bedroom | 1,374 | 2,494 | no |
| living_room | 1,068 | 2,188 | no |

**Conclusion:** context caching is **not usable on 3.5-flash** — its cacheable
prefix minimum is **4,096 tokens** and the biggest prefix is only 2,750. (2.5-flash's
2,048 minimum *would* be cleared, so caching is only an option if Stage 2 runs on
2.5-flash.) Cache read is 10× cheaper ($0.15 vs $1.50/1M) and storage is trivial
($1/1M/hr), but the size floor makes it moot here. **Drop caching from the plan.**

## Test 3 — Stage-1 cost, thinking minimized vs default (`stage1_cost_test.py`)

100 random images, LOW resolution, classify prompt (103 tok) + 1 image.

| condition | input tok | output tok | thinking tok | $/img (batch) | full run (76,499) |
|---|---|---|---|---|---|
| thinking minimized (`thinking_level=low`) | 362 | 65 (med 58) | 48 | $0.00056 | **~$43** |
| thinking **default** | 362 | 126 (med 113) | 110 (max 340) | $0.00084 | **~$64** |

**Conclusion:** classification barely thinks — even at default, output tops out
~340 tok, so the feared "thinking blows up Stage-1 cost" did **not** happen.
Worst case is ~$64. Input measured at 362 tok.

Label distribution on the (unbiased) 100-image sample: 46 other, 20 bedroom,
15 bathroom, 13 living_room, 6 kitchen → **~54% of images are one of the 4
target rooms** (drives the Stage-2 image count).

## Test 4 — Low vs default *agreement* / accuracy artifacts (`stage1_accuracy_test.py`)

Same 100 images classified twice (low vs default thinking).

- **Low-vs-default agreement = 99/100 (99%)**. Only one flip (bathroom↔other).
- ⇒ Dialing thinking down changes the answer ~1% of the time; low thinking loses
  almost nothing **relative to** default.
- **Caveat:** agreement ≠ ground-truth accuracy (both modes can share an error).
  Real accuracy needs the `truth` column in `predictions.csv` filled in (or eyeball
  `gallery.html`); the dangerous errors are room→`other` (data dropped from Stage 2)
  and wrong-room (routed to the wrong rubric). **Still open.**

## Test 5 — Does `thinking_budget` cap thinking? (`thinking_cap_test.py`)

40 images, LOW, gemini-3.5-flash.

| config | mean think | median | max | invalid JSON |
|---|---|---|---|---|
| default | 100 | 91 | 233 | 0/40 |
| `thinking_budget=128` | 43 | 40 | 96 | 0/40 |
| `thinking_budget=512` | 43 | 40 | 96 | 0/40 |
| `thinking_budget=0` | 0 | 0 | 0 | 0/40 |

**Conclusion:** `thinking_budget` is **not a proportional cap** on 3.5-flash — 128
and 512 are identical and both *below* default's natural distribution, i.e. any
nonzero budget just switches the model into a reduced-thinking regime (≈
`thinking_level=low`). There is **no "default thinking but capped at X"** config.
`thinking_budget=0` fully disables thinking (answers stay valid). Default's tail
is already short (max 233/323), so **no cap is needed**. Never cap via
`max_output_tokens` (it bounds thinking+answer together → truncated/empty JSON).

---

## Decisions fed into the plan

- **Stage 1:** gemini-3.5-flash, `media_resolution=LOW`, **default thinking**
  (no `thinking_config`), no cap. ≈ **$64** for all 76,499 images.
- **Stage 2:** gemini-3.5-flash, grid + target both **HIGH**, default thinking.
  Measured 3,377 in / 722 out tok/img → $0.00578/img; ~54% of images (~41k) →
  ≈ **$239**.
- **No context caching** (prefix under the 3.5-flash 4,096 floor).
- **Full pipeline ≈ $300** (batch), ~85% Stage 2.

## Still open (validation TODO before the full run)

1. **Stage-1 absolute accuracy** — fill `truth` in `predictions.csv`; confusion
   matrix, focus on room→other and wrong-room errors.
2. **2.5-flash vs 3.5-flash** for Stage 1 (2.5 ≈ $6.5, and is cacheable) — decide
   on accuracy, not cost (Stage-1 model is a small swing).
3. **Room fraction** (~54% from 100 imgs; treat as ±5%) firms up once Stage 1 runs.
