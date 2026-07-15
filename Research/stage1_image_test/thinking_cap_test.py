"""Does thinking_budget cap thinking on gemini-3.5-flash (Stage-1 classify, LOW)?

Same 40 images through 3 configs:
  default        (no thinking_config)
  budget=128
  budget=512
Reports thinking-token distribution per config + whether every response stayed
valid JSON (a cap that truncates the answer is a bug, not a saving).
"""
import concurrent.futures as cf
import json
import random
import statistics as st
import sys
from pathlib import Path

REPO = Path("/Users/skyler/local/CompEdge")
sys.path.insert(0, str(REPO))
from config import Config
from google import genai
from google.genai import types

MODEL = "gemini-3.5-flash"
N = 40
LABELS = ["kitchen", "bathroom", "bedroom", "living_room", "other"]
PROMPT = (
    "Classify the room shown in this real-estate listing photo. "
    "Set room_type to exactly one of: kitchen, bathroom, bedroom, living_room, other. "
    'Use "other" for exteriors, floor plans, hallways, dining rooms, offices, garages, '
    "yards, closets, laundry rooms, or anything not clearly one of the four listed room "
    'types — and in that case also set other_label to a short label; otherwise null.'
)
SCHEMA = {"type": "object",
          "properties": {"room_type": {"type": "string", "enum": LABELS},
                         "other_label": {"type": "string", "nullable": True}},
          "required": ["room_type", "other_label"]}
client = genai.Client(api_key=Config.GEMINI_API_KEY)


def sample(n):
    dirs = sorted(d for d in (REPO / "images").iterdir() if d.is_dir())
    random.Random(7).shuffle(dirs)
    picks = []
    for d in dirs:
        picks += sorted(p for p in d.iterdir()
                        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})[:6]
        if len(picks) >= n:
            break
    return picks[:n]


def call(path, budget):
    cfg = dict(system_instruction=PROMPT, temperature=0,
               response_mime_type="application/json", response_schema=SCHEMA,
               media_resolution=types.MediaResolution.MEDIA_RESOLUTION_LOW)
    if budget is not None:
        cfg["thinking_config"] = types.ThinkingConfig(thinking_budget=budget)
    r = client.models.generate_content(
        model=MODEL,
        contents=[types.Part.from_bytes(data=path.read_bytes(), mime_type="image/jpeg"),
                  "Photo to classify:"],
        config=types.GenerateContentConfig(**cfg))
    um = r.usage_metadata
    think = getattr(um, "thoughts_token_count", 0) or 0
    valid = True
    try:
        json.loads(r.text)["room_type"]
    except Exception:
        valid = False
    return think, valid


def run(name, imgs, budget):
    thinks, invalid = [], 0
    with cf.ThreadPoolExecutor(max_workers=10) as ex:
        for th, ok in ex.map(lambda p: call(p, budget), imgs):
            thinks.append(th)
            invalid += (not ok)
    print(f"{name:16} think: mean {st.mean(thinks):5.0f}  median {st.median(thinks):4.0f}  "
          f"max {max(thinks):4d}   invalid JSON: {invalid}/{len(imgs)}")


def main():
    imgs = sample(N)
    print(f"{len(imgs)} images, gemini-3.5-flash, LOW\n")
    run("default", imgs, None)
    run("budget=128", imgs, 128)
    run("budget=512", imgs, 512)
    run("budget=0", imgs, 0)


if __name__ == "__main__":
    main()
