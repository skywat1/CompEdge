"""Stage-1 classification cost test on real listing images.

Two conditions, same 100 images each (gemini-3.5-flash, media_resolution LOW):
  1) thinking minimized  (thinking_level="low")
  2) thinking default     (no thinking_config)

Measures real input / output / thinking tokens per call and extrapolates the
full-run cost (76,499 images) at BATCH pricing. Live calls (identical tokens to
batch, billed 2x) so we get numbers in minutes instead of a 24h batch job.
"""
import concurrent.futures as cf
import random
import statistics as st
import sys
import time
from pathlib import Path

REPO = Path("/Users/skyler/local/CompEdge")
sys.path.insert(0, str(REPO))
from config import Config
from google import genai
from google.genai import types

MODEL = "gemini-3.5-flash"
TOTAL_IMAGES = 76_499
N = 100
WORKERS = 10
VALID = {".jpg", ".jpeg", ".png", ".webp"}
LABELS = ["kitchen", "bathroom", "bedroom", "living_room", "other"]

# batch pricing (50% off standard) $/1M tokens
IN_BATCH, OUT_BATCH = 0.75, 4.50

PROMPT = (
    "Classify the room shown in this real-estate listing photo. "
    "Set room_type to exactly one of: kitchen, bathroom, bedroom, living_room, other. "
    'Use "other" for exteriors, floor plans, hallways, dining rooms, offices, garages, '
    "yards, closets, laundry rooms, or anything not clearly one of the four listed room "
    'types — and in that case also set other_label to a short label; otherwise null.'
)
SCHEMA = {
    "type": "object",
    "properties": {
        "room_type": {"type": "string", "enum": LABELS},
        "other_label": {"type": "string", "nullable": True},
    },
    "required": ["room_type", "other_label"],
}

client = genai.Client(api_key=Config.GEMINI_API_KEY)


def sample_images(n):
    dirs = sorted(d for d in (REPO / "images").iterdir() if d.is_dir())
    random.Random(7).shuffle(dirs)
    picks = []
    for d in dirs:
        imgs = sorted(p for p in d.iterdir() if p.suffix.lower() in VALID)
        picks += imgs[:6]  # up to 6 per listing for variety
        if len(picks) >= n:
            break
    return picks[:n]


def classify(path: Path, minimize_thinking: bool):
    cfg = dict(
        system_instruction=PROMPT,
        temperature=0,
        response_mime_type="application/json",
        response_schema=SCHEMA,
        media_resolution=types.MediaResolution.MEDIA_RESOLUTION_LOW,
    )
    if minimize_thinking:
        cfg["thinking_config"] = types.ThinkingConfig(thinking_level="low")
    t0 = time.monotonic()
    r = client.models.generate_content(
        model=MODEL,
        contents=[types.Part.from_bytes(data=path.read_bytes(),
                                        mime_type="image/jpeg"),
                  "Photo to classify:"],
        config=types.GenerateContentConfig(**cfg),
    )
    dt = time.monotonic() - t0
    um = r.usage_metadata
    thoughts = getattr(um, "thoughts_token_count", 0) or 0
    out = (um.candidates_token_count or 0) + thoughts
    return {"in": um.prompt_token_count or 0, "out": out, "think": thoughts,
            "lat": dt, "text": r.text}


def run_condition(name, imgs, minimize):
    print(f"\n### Condition: {name}  ({len(imgs)} images)")
    rows, errs = [], 0
    t0 = time.monotonic()
    with cf.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(classify, p, minimize): p for p in imgs}
        for i, f in enumerate(cf.as_completed(futs), 1):
            try:
                rows.append(f.result())
            except Exception as e:
                errs += 1
                if errs <= 3:
                    print("  err:", type(e).__name__, str(e)[:140])
            print(f"  {i}/{len(imgs)}", end="\r", flush=True)
    wall = time.monotonic() - t0
    ins = [r["in"] for r in rows]
    outs = [r["out"] for r in rows]
    thinks = [r["think"] for r in rows]
    def cost(i, o):
        return (i * IN_BATCH + o * OUT_BATCH) / 1e6
    per_img = cost(st.mean(ins), st.mean(outs))
    print(f"\n  ok={len(rows)} err={errs}  wall={wall:.1f}s")
    print(f"  input  tok: mean {st.mean(ins):.0f}  median {st.median(ins):.0f}  "
          f"[{min(ins)}-{max(ins)}]")
    print(f"  output tok: mean {st.mean(outs):.0f}  median {st.median(outs):.0f}  "
          f"[{min(outs)}-{max(outs)}]")
    print(f"  think  tok: mean {st.mean(thinks):.0f}  median {st.median(thinks):.0f}  "
          f"[{min(thinks)}-{max(thinks)}]")
    print(f"  $/image (batch): ${per_img:.6f}")
    print(f"  >>> FULL RUN {TOTAL_IMAGES:,} imgs (batch): ${per_img*TOTAL_IMAGES:,.2f}")
    return rows


def main():
    imgs = sample_images(N)
    print(f"Sampled {len(imgs)} images across listings; model={MODEL}, res=LOW")
    r1 = run_condition("1) thinking minimized (thinking_level=low)", imgs, True)
    r2 = run_condition("2) thinking default", imgs, False)
    # quick label sanity check from condition 1
    import json, collections
    labs = collections.Counter()
    for r in r1:
        try:
            labs[json.loads(r["text"])["room_type"]] += 1
        except Exception:
            labs["<unparsed>"] += 1
    print("\nLabel distribution (condition 1):", dict(labs))


if __name__ == "__main__":
    main()
