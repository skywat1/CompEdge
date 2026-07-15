"""Stage-1 classification: save predictions for accuracy review.

Same 100 images (seed 7), classified twice with gemini-3.5-flash / LOW:
  low     = thinking_level="low"   (the cheap prod candidate)
  default = no thinking_config

Writes, to OUT_DIR:
  predictions.csv  - one row/image: both labels, agreement, tokens.
                     Add your own `truth` column later to score real accuracy.
  gallery.html     - each image + both labels, disagreements flagged, so a
                     human can eyeball accuracy quickly.

Also prints low-vs-default AGREEMENT (does dialing thinking down change answers).
"""
import base64
import concurrent.futures as cf
import csv
import io
import json
import random
import sys
from pathlib import Path

from PIL import Image

REPO = Path("/Users/skyler/local/CompEdge")
OUT_DIR = REPO / "stage1_test"
sys.path.insert(0, str(REPO))
from config import Config
from google import genai
from google.genai import types

MODEL = "gemini-3.5-flash"
N = 100
WORKERS = 10
VALID = {".jpg", ".jpeg", ".png", ".webp"}
LABELS = ["kitchen", "bathroom", "bedroom", "living_room", "other"]

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
        picks += sorted(p for p in d.iterdir() if p.suffix.lower() in VALID)[:6]
        if len(picks) >= n:
            break
    return picks[:n]


def classify(path: Path, minimize: bool):
    cfg = dict(system_instruction=PROMPT, temperature=0,
               response_mime_type="application/json", response_schema=SCHEMA,
               media_resolution=types.MediaResolution.MEDIA_RESOLUTION_LOW)
    if minimize:
        cfg["thinking_config"] = types.ThinkingConfig(thinking_level="low")
    r = client.models.generate_content(
        model=MODEL,
        contents=[types.Part.from_bytes(data=path.read_bytes(), mime_type="image/jpeg"),
                  "Photo to classify:"],
        config=types.GenerateContentConfig(**cfg))
    um = r.usage_metadata
    out = (um.candidates_token_count or 0) + (getattr(um, "thoughts_token_count", 0) or 0)
    p = json.loads(r.text)
    return {"label": p["room_type"], "other": p.get("other_label") or "", "out": out}


def both(path):
    return path, classify(path, True), classify(path, False)


def thumb(path, px=240):
    try:
        im = Image.open(path).convert("RGB")
        im.thumbnail((px, px))
        buf = io.BytesIO(); im.save(buf, "JPEG", quality=70)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return ""


def main():
    OUT_DIR.mkdir(exist_ok=True)
    imgs = sample_images(N)
    print(f"Classifying {len(imgs)} images twice (low vs default thinking)...")
    results = {}
    with cf.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for i, fut in enumerate(cf.as_completed([ex.submit(both, p) for p in imgs]), 1):
            path, lo, de = fut.result()
            results[path] = (lo, de)
            print(f"  {i}/{len(imgs)}", end="\r", flush=True)
    print()

    rows = []
    for path in imgs:  # stable order
        lo, de = results[path]
        rel = str(path.relative_to(REPO))
        rows.append({"image_path": rel, "zpid": path.parent.name,
                     "low_label": lo["label"], "low_other": lo["other"],
                     "default_label": de["label"], "default_other": de["other"],
                     "agree": int(lo["label"] == de["label"]),
                     "low_out_tok": lo["out"], "default_out_tok": de["out"],
                     "truth": ""})  # <- fill in by hand to score real accuracy

    csv_path = OUT_DIR / "predictions.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    agree = sum(r["agree"] for r in rows)
    disagree = [r for r in rows if not r["agree"]]
    print(f"low-vs-default AGREEMENT: {agree}/{len(rows)} = {agree/len(rows):.0%}")
    if disagree:
        print("disagreements (low -> default):")
        for r in disagree:
            print(f"  {r['image_path']}: {r['low_label']} -> {r['default_label']}")

    # gallery for human eyeballing
    cards = []
    for r in rows:
        img = thumb(REPO / r["image_path"])
        flag = "" if r["agree"] else ' style="border:3px solid #d33"'
        lab2 = "" if r["agree"] else f' <b style="color:#d33">/ {r["default_label"]}</b>'
        oth = f' <i>({r["low_other"]})</i>' if r["low_other"] else ""
        cards.append(f'<div class=c{flag}><img src="{img}"><div>{r["low_label"]}{oth}{lab2}</div>'
                     f'<div class=p>{r["image_path"]}</div></div>')
    html = ("<html><head><meta charset=utf-8><style>"
            "body{font:13px system-ui;background:#111;color:#eee;margin:16px}"
            ".g{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:12px}"
            ".c{background:#1c1c1c;border-radius:8px;padding:8px}"
            ".c img{width:100%;border-radius:4px}.p{color:#888;font-size:11px;word-break:break-all}"
            "</style></head><body>"
            f"<h2>Stage-1 predictions — {len(rows)} imgs, gemini-3.5-flash LOW · "
            f"low-vs-default agreement {agree/len(rows):.0%}</h2>"
            "<p>Label shown = low-thinking prediction. Red border = default thinking disagreed "
            "(shows /default). Open predictions.csv and fill the <code>truth</code> column to "
            "score real accuracy.</p>"
            f'<div class=g>{"".join(cards)}</div></body></html>')
    (OUT_DIR / "gallery.html").write_text(html)

    print(f"\nSaved:\n  {csv_path}\n  {OUT_DIR/'gallery.html'}")


if __name__ == "__main__":
    main()
