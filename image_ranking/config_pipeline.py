"""Static configuration for the image-ranking pipeline.

Constants only — no side effects. Paths are resolved relative to the repo root
so scripts work regardless of the current working directory.
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
IMAGES_DIR = REPO_ROOT / "images"
DATA_DIR = REPO_ROOT / "data"
PKG_DIR = REPO_ROOT / "image_ranking"
OUTPUTS_DIR = PKG_DIR / "outputs"
RAW_DIR = OUTPUTS_DIR / "raw"
BATCH_JOBS_FILE = OUTPUTS_DIR / "batch_jobs.json"

# Final Stage-1 deliverable lives in data/ alongside the other AVM inputs
# (cleaned_sold.csv, pluto.csv, ...) and is picked up by the regression.
# Intermediates (raw results, request JSONLs, batch bookkeeping) stay in OUTPUTS_DIR.
CLASSIFICATIONS_CSV = DATA_DIR / "classifications.csv"

# ---------------------------------------------------------------------------
# Stage 1 — classification. Stage-specific values are STAGE1_*-prefixed so Stage 2
# can add its own (e.g. STAGE2_MODEL, STAGE2_GRID_RESOLUTION at HIGH) without
# colliding. Shared infra below (paths, pricing, chunking, batch) stays general.
# ---------------------------------------------------------------------------
STAGE1_MODEL = "gemini-3.5-flash"

# media_resolution sets the per-image token cost on Gemini 3.x (LOW ~= 280 tok).
# Default thinking (no thinking_config): measured ~362 in / ~126 out per image.
# Stage 2 will use HIGH for both the grid and the target image.
STAGE1_MEDIA_RESOLUTION = "MEDIA_RESOLUTION_LOW"

# The 6-label taxonomy. Only the first four go on to Stage-2 scoring; "floorplan"
# is broken out as its own first-class label; everything else is "other" and
# carries a free-text other_label.
SCORED_ROOMS = ["kitchen", "bathroom", "bedroom", "living_room"]
LABELS = SCORED_ROOMS + ["floorplan", "other"]

VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}

# Transport only: shrink images so batch JSONL files stay small. media_resolution
# (not pixel size) determines the token bill on Gemini 3.x, so this is lossless
# for cost. 768 px is comfortably above what LOW resolution consumes. Stage 2 may
# use a larger STAGE2_RESIZE_MAX_DIM to feed HIGH resolution more detail.
STAGE1_RESIZE_MAX_DIM = 768

# ---------------------------------------------------------------------------
# Batch chunking — keep each input JSONL well under the 2 GB batch-file limit.
# ---------------------------------------------------------------------------
CHUNK_MAX_BYTES = 1_500_000_000   # ~1.5 GB per JSONL chunk
CHUNK_MAX_LINES = 20_000          # and no more than this many requests per chunk

# ---------------------------------------------------------------------------
# Pricing ($ / 1M tokens, standard tier). Batch = 50% off both.
# ---------------------------------------------------------------------------
PRICE_INPUT = 1.50
PRICE_OUTPUT = 9.00
BATCH_DISCOUNT = 0.5

# Stage-1 output CSV schema. output_tokens = answer tokens only; thinking_tokens
# is the (default-thinking) reasoning portion — both bill as output, so cost_usd
# is computed on their sum. is_hero/source_chunk/response_model are zero-cost
# provenance (filename, our chunk label, and the response's modelVersion).
CLASSIFY_FIELDS = ["image_path", "zpid", "is_hero", "room_type", "other_label",
                   "response_model", "input_tokens", "output_tokens",
                   "thinking_tokens", "cost_usd", "source_chunk"]
