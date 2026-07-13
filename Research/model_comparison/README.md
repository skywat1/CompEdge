# Luxury-scoring model comparison

Compares five models for the image luxury-scoring stage, using the prompts,
anchor grids, and schema from `Research/llm_image_rank_test/score.py` as the
reference implementation: gpt-4o (incumbent), gemini-3.5-flash,
gemini-2.5-flash, claude-haiku-4-5, claude-sonnet-4-6.

API keys are read from `config.py` at the repo root (`OPENAI_API_KEY`,
`GEMINI_API_KEY`, `CLAUDE_API_KEY`).

Run order (all from this directory, with the repo venv):

```bash
# 0. Dry run: validates schemas/messages/parsing on all models, ~18 calls
python dry_run.py

# 1. Classify images (gpt-4o only, detail=low, capped at 3000 calls, resumable).
#    Runs properties concurrently (--workers); each property early-stops once it
#    has produced one image of each room type, since Stage 2 keeps at most one
#    per room type per zpid anyway. --tpm-limit paces calls under your OpenAI
#    account's tokens-per-minute cap (default 28000, just under a 30k Tier-1
#    gpt-4o limit) so you don't spray 429s; raise it on higher tiers.
python stage1_classify.py --images-dir ../../images \
    --out-csv outputs/classify/classifications.csv --raw-jsonl outputs/classify/classifications_raw.jsonl \
    --workers 4 --tpm-limit 28000

# 2. Build the stratified sample manifest (~38/room type, <=1 per type per zpid)
python stage2_sample.py --classifications-csv outputs/classify/classifications.csv \
    --out-manifest outputs/sample/sample_manifest.csv

# 2b. Human rating app (run any time after the manifest exists; independent of
#     the scoring runs — humans can rate while API calls are in progress)
python stage2b_rating_app.py --manifest outputs/sample/sample_manifest.csv \
    --images-root ../.. --db outputs/ratings/human_ratings.sqlite
# ...then export for the analysis stage:
python export_ratings.py --db outputs/ratings/human_ratings.sqlite \
    --out-csv outputs/ratings/human_ratings.csv \
    --out-parquet outputs/parquet/human_ratings.parquet

# 3. Score: every manifest image x 5 models x 5 replicates (resumable). Each
#    provider runs in its own concurrent pool (wall-clock ~= slowest provider,
#    not the sum); the OpenAI gpt-4o leg is paced under --tpm-limit. Lower
#    --gemini-workers / --anthropic-workers if those providers 429 on your tier.
python stage3_score.py --manifest outputs/sample/sample_manifest.csv --images-root ../.. \
    --out-csv outputs/scores/scores.csv --raw-jsonl outputs/scores/scores_raw.jsonl \
    --openai-workers 3 --gemini-workers 6 --anthropic-workers 6 --tpm-limit 28000

# 4. Consolidate to parquet
python stage4_consolidate.py --classifications-csv outputs/classify/classifications.csv \
    --manifest outputs/sample/sample_manifest.csv --scores-csv outputs/scores/scores.csv \
    --out-dir outputs/parquet

# 5. Analysis + markdown report (--ratings-parquet is optional; it enables the
#    human-agreement sections and the ceiling-based recommendation)
python stage5_report.py --parquet-dir outputs/parquet --images-dir ../../images \
    --out-report outputs/report/report.md --plots-dir outputs/plots \
    --ratings-parquet outputs/parquet/human_ratings.parquet
```

## Outputs layout

```
outputs/
  classify/   Stage 1 — classifications.csv, classifications_raw.jsonl
  sample/     Stage 2 — sample_manifest.csv
  ratings/    Stage 2b/export — human_ratings.sqlite, human_ratings.csv
  scores/     Stage 3 — scores*.csv, scores*_raw.jsonl (base + tuned/tuned2 variants)
  parquet/, parquet_tuned/, parquet_tuned2/   Stage 4 — consolidated parquet per run
  plots/      Stage 5 — chart PNGs
  report/     Stage 5 — report.md, report_explained.{md,html,pdf}, md2html.py
  gallery/    diff_gallery_*.html, make_diff_gallery.py
  dry_run_results.json
```

## Rating app notes (Stage 2b)

The app freezes an ~18-per-room-type subset of the manifest into the SQLite DB
on first startup; restarts never resample, so every rater sees the same images.
Each rater enters a name once, rates every subset image in their own randomized
order, and can close/reopen the tab freely — progress is stored per name.
Scores are entered with number keys (1-7 for kitchens, 1-8 for the rest).

It binds 0.0.0.0 and prints the LAN URL on startup. For raters outside the LAN:

* **Tailscale** (recommended): both machines on your tailnet, share
  `http://<your-tailscale-ip>:5000` — no config changes needed.
* **cloudflared** quick tunnel: `cloudflared tunnel --url http://localhost:5000`
  prints a public `https://*.trycloudflare.com` URL. Anyone with the link can
  rate; stop the tunnel when done.

Checkpointing: stages 1 and 3 append to their CSVs keyed by image path /
(model, image, replicate) and skip completed work on re-run. Raw API responses
(full JSON) are kept in the `*_raw.jsonl` files. Nothing here touches the
production pipeline scripts.
