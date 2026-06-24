#!/usr/bin/env python3
"""
Kitchen luxury-level scorer.

For every image in a folder, sends (prompt + reference grid + kitchen image) to
GPT N times as independent calls to measure reliability, then writes everything
to a single results.txt with per-test outputs, a consistency flag, and averages.

Setup:
    pip install openai python-dotenv
    Create a .env file in this folder containing:
        OPENAI_API_KEY=sk-...

Folder layout expected:
    kitchens/            <- folder of kitchen1.jpg, kitchen2.png, ...
    prompt.txt           <- the system prompt text
    grid.png             <- the reference anchor grid image

Then:
    python score_kitchens.py
"""

import base64
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)  # load OPENAI_API_KEY from a .env file in the current folder

# ---------------------------------------------------------------------------
# CONFIG  -- edit these
# ---------------------------------------------------------------------------
ROOT_FOLDER  = 'test'
IMAGE_FOLDER = f"{ROOT_FOLDER}/images"        # folder containing kitchen1.jpg, kitchen2.jpg, ...
PROMPT_FILE  = f"{ROOT_FOLDER}/prompt.txt"      # your system prompt
GRID_FILE    = f"{ROOT_FOLDER}/grid.png"         # the reference anchor grid
OUTPUT_FILE  = f"{ROOT_FOLDER}/result.txt"      # single output file

MODEL        = "gpt-4o"          # must support vision + structured outputs (gpt-4o / gpt-4.1)
REPS         = 1                # independent calls per image
TEMPERATURE  = 0.2               # lower = more consistent; raise to surface instability
IMAGE_DETAIL = "high"            # "high" reads small anchor tiles better (costs more tokens)
MAX_RETRIES  = 4                 # retries per call on transient API errors

VALID_EXTS   = {".jpg", ".jpeg", ".png", ".webp", ".gif"}

# JSON schema enforced via OpenAI structured outputs (strict mode).
LUXURY_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning":  {"type": "string"},
        "score":      {"type": "number"},
        "level":      {"type": "integer", "enum": [1, 2, 3, 4, 5, 6, 7, 8]},
        "confidence": {"type": "number"},
        "valid":      {"type": "boolean"},
    },
    "required": ["reasoning", "score", "level", "confidence", "valid"],
    "additionalProperties": False,
}

client = OpenAI()  # reads OPENAI_API_KEY from environment

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def mime_for(path: Path) -> str:
    ext = path.suffix.lower()
    return {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png", ".webp": "image/webp", ".gif": "image/gif",
    }.get(ext, "image/jpeg")


def to_data_url(path: Path) -> str:
    """Base64-encode an image as a data URL. Identical bytes -> cache hit."""
    b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime_for(path)};base64,{b64}"


def natural_key(path: Path):
    """Sort kitchen1, kitchen2, ..., kitchen10 in human order (not kitchen10 after kitchen1)."""
    nums = re.findall(r"\d+", path.stem)
    return (int(nums[0]) if nums else float("inf"), path.stem)


def build_messages(prompt_text: str, grid_url: str, kitchen_url: str):
    """
    Static content (system prompt + grid image) comes FIRST so OpenAI's automatic
    prompt caching covers it. The variable kitchen image comes LAST.
    """
    return [
        {"role": "system", "content": prompt_text},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "REFERENCE GRID (anchor examples, Level 1 = lowest ... Level 8 = highest):"},
                {"type": "image_url", "image_url": {"url": grid_url, "detail": IMAGE_DETAIL}},
                {"type": "text", "text": "KITCHEN TO SCORE:"},
                {"type": "image_url", "image_url": {"url": kitchen_url, "detail": IMAGE_DETAIL}},
            ],
        },
    ]


def call_once(messages):
    """One independent API call with structured output + retry. Returns (data_dict, usage)."""
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=TEMPERATURE,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "kitchen_luxury",
                        "strict": True,
                        "schema": LUXURY_SCHEMA,
                    },
                },
                # Optional: stable key helps OpenAI route to the same cache.
                # prompt_cache_key="kitchen-luxury-v1",
            )
            data = json.loads(resp.choices[0].message.content)
            return data, resp.usage
        except Exception as e:  # network / rate-limit / transient parse errors
            last_err = e
            wait = 2 ** attempt
            print(f"    call failed (attempt {attempt}/{MAX_RETRIES}): {e} -- retrying in {wait}s")
            time.sleep(wait)
    print(f"    GIVING UP on this call: {last_err}")
    return None, None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    prompt_path = Path(PROMPT_FILE)
    grid_path   = Path(GRID_FILE)
    folder      = Path(IMAGE_FOLDER)

    for required in (prompt_path, grid_path, folder):
        if not required.exists():
            sys.exit(f"ERROR: '{required}' not found. Check the CONFIG paths.")

    prompt_text = prompt_path.read_text(encoding="utf-8")
    grid_url    = to_data_url(grid_path)  # encoded ONCE, reused for every call (cached prefix)

    images = sorted(
        [p for p in folder.iterdir() if p.suffix.lower() in VALID_EXTS],
        key=natural_key,
    )
    if not images:
        sys.exit(f"ERROR: no images found in '{folder}'.")

    print(f"Found {len(images)} images. Running {REPS} reps each on {MODEL}.\n")

    total_cached = total_prompt = total_completion = 0
    consistent_count = 0

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        out.write(f"Kitchen luxury scoring results\n")
        out.write(f"Model: {MODEL} | reps: {REPS} | temperature: {TEMPERATURE}\n")
        out.write("=" * 70 + "\n\n")

        for img in images:
            print(f"Scoring {img.name} ...")
            kitchen_url = to_data_url(img)
            messages = build_messages(prompt_text, grid_url, kitchen_url)

            results = []  # list of dicts or None
            for rep in range(1, REPS + 1):
                data, usage = call_once(messages)
                results.append(data)
                if usage is not None:
                    total_prompt += usage.prompt_tokens
                    total_completion += usage.completion_tokens
                    cached = getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens", 0) or 0
                    total_cached += cached
                print(f"    rep {rep}/{REPS} done")

            # --- aggregate ---
            ok = [r for r in results if r is not None]
            levels = [r["level"] for r in ok]
            scores = [r["score"] for r in ok]
            confs  = [r["confidence"] for r in ok]

            # "matched every time" = same integer level on all reps AND no failed calls
            consistent = (len(ok) == REPS) and (len(set(levels)) == 1)
            if consistent:
                consistent_count += 1

            avg_score = round(statistics.mean(scores), 3) if scores else None
            avg_conf  = round(statistics.mean(confs), 3) if confs else None

            # --- write block ---
            out.write(f"=== {img.name} ===\n")
            for i, r in enumerate(results, 1):
                if r is None:
                    out.write(f"Test {i:>2}: FAILED (no response after retries)\n")
                else:
                    out.write(
                        f"Test {i:>2}: score={r['score']}, level={r['level']}, "
                        f"confidence={r['confidence']}, valid={r['valid']}, "
                        f"reasoning=\"{r['reasoning']}\"\n"
                    )
            out.write(f"Level consistent across all {REPS} reps: {consistent}\n")
            out.write(f"Average score: {avg_score}\n")
            out.write(f"Average confidence: {avg_conf}\n")
            out.write("\n")
            out.flush()  # write per-image so a crash mid-run doesn't lose progress

    # --- summary ---
    print("\nDone.")
    print(f"Results written to {OUTPUT_FILE}")
    print(f"Consistent (same level all {REPS} reps): {consistent_count}/{len(images)} images")
    print(
        f"Tokens -- prompt: {total_prompt:,} (of which cached: {total_cached:,}), "
        f"completion: {total_completion:,}"
    )
    if total_prompt:
        print(f"Cache hit rate on prompt tokens: {total_cached / total_prompt:.0%}")


if __name__ == "__main__":
    main()