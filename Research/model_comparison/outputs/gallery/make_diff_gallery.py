#!/usr/bin/env python3
"""Build a self-contained HTML gallery of every image where gemini-3.5-flash's
mean score differs from the human mean by MORE than 1 point.

Each card shows the photo, each rater's score, the human mean, gemini's mean
(and its per-replicate min/max), the room's scale, and the signed gap. Sorted
by room, then by |gap| descending. Images are inlined as base64 so the file is
portable. Human reference defaults to all raters; pass --raters to restrict.

Usage (from this outputs/gallery/ dir, with the repo venv):
    python make_diff_gallery.py
    python make_diff_gallery.py --raters harvey,robin --out diff_gallery_hr.html
"""
import argparse
import base64
import mimetypes
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]  # outputs/gallery/ -> outputs/ -> model_comparison/ -> Research/ -> repo
ROOMS = ["kitchen", "bathroom", "bedroom", "living_room"]
MAX_LEVEL = {"kitchen": 7, "bathroom": 8, "bedroom": 8, "living_room": 8}
MODEL = "gemini-3.5-flash"


def inline_img(path: Path) -> str:
    if not path.exists():
        return ""
    mime = mimetypes.guess_type(str(path))[0] or "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parquet-dir", type=Path, default=HERE.parent / "parquet")
    ap.add_argument("--images-root", type=Path, default=REPO_ROOT,
                    help="Root that image_path values are relative to")
    ap.add_argument("--raters", default="harvey,robin,seb",
                    help="Comma-separated human reference (default all three)")
    ap.add_argument("--sold-csv", type=Path, default=REPO_ROOT / "data" / "cleaned_sold.csv",
                    help="CSV with zpid + listing-link-href, for the Zillow links")
    ap.add_argument("--threshold", type=float, default=1.0)
    ap.add_argument("--out", type=Path, default=HERE / "diff_gallery.html")
    args = ap.parse_args()

    raters = [r.strip() for r in args.raters.split(",") if r.strip()]

    ratings = pd.read_parquet(args.parquet_dir / "human_ratings.parquet")
    scores = pd.read_parquet(args.parquet_dir / "scores.parquet")
    scores = scores[scores["error"].fillna("").astype(str).str.len() == 0].copy()
    scores["score"] = pd.to_numeric(scores["score"], errors="coerce")

    room_of = (scores[["image_path", "room_type"]].drop_duplicates()
               .set_index("image_path")["room_type"])
    zpid_of = (scores[["image_path", "zpid"]].drop_duplicates()
               .set_index("image_path")["zpid"])

    # zpid -> Zillow listing URL from the cleaned sold data
    href_of = {}
    if args.sold_csv.exists():
        sold = pd.read_csv(args.sold_csv, usecols=["zpid", "listing-link-href"])
        href_of = {str(z): h for z, h in
                   zip(sold["zpid"].astype(str), sold["listing-link-href"])}
    else:
        print(f"warning: {args.sold_csv} not found — Zillow links omitted")
    gem_rows = scores[scores["model"] == MODEL]
    g = gem_rows.groupby("image_path")["score"]
    gem = g.mean().rename("gem")
    gem_lo, gem_hi = g.min().rename("gem_lo"), g.max().rename("gem_hi")

    # all 5 replicate reasonings per image, ordered by replicate
    reasonings = {
        path: list(sub.sort_values("replicate")[["replicate", "score", "reasoning"]]
                   .itertuples(index=False, name=None))
        for path, sub in gem_rows.groupby("image_path")
    }

    # each rater's per-image score, wide
    rr = ratings[ratings["rater"].isin(raters)]
    wide = rr.pivot_table(index="image_path", columns="rater", values="score")
    present = [r for r in raters if r in wide.columns]
    hum_mean = wide[present].mean(axis=1).rename("hum")

    df = pd.concat([gem, gem_lo, gem_hi, hum_mean, room_of.rename("room_type")],
                   axis=1, join="inner").dropna(subset=["gem", "hum"])
    df["gap"] = df["gem"] - df["hum"]
    df = df[df["gap"].abs() > args.threshold].copy()
    df = df.join(wide[present])
    df["abs_gap"] = df["gap"].abs()

    # per-room counts for the header
    counts = {room: int((df["room_type"] == room).sum()) for room in ROOMS}

    cards = []
    for room in ROOMS:
        sub = df[df["room_type"] == room].sort_values("abs_gap", ascending=False)
        if sub.empty:
            continue
        cards.append(f'<h2>{room} — {len(sub)} images >{args.threshold:g} off '
                     f'(scale 1–{MAX_LEVEL[room]})</h2>')
        cards.append('<div class="grid">')
        for path, row in sub.iterrows():
            src = inline_img((args.images_root / path).resolve())
            rater_rows = "".join(
                f'<tr><td>{r}</td><td class="v">'
                f'{row[r]:.0f}</td></tr>' if pd.notna(row[r]) else
                f'<tr><td>{r}</td><td class="v">–</td></tr>'
                for r in present)
            direction = "high" if row["gap"] > 0 else "low"

            def esc(t):
                return (str(t).replace("&", "&amp;").replace("<", "&lt;")
                        .replace(">", "&gt;"))
            reas = "".join(
                f'<li><span class="rep">rep {rep} · {sc:.1f}</span> {esc(txt)}</li>'
                for rep, sc, txt in reasonings.get(path, []))
            zpid = str(zpid_of.get(path, ""))
            href = href_of.get(zpid)
            link = (f'<a class="zlink" href="{esc(href)}" target="_blank" '
                    f'rel="noopener">↗ Zillow listing (zpid {zpid})</a>'
                    if isinstance(href, str) and href else
                    f'<span class="zlink none">no listing link (zpid {zpid})</span>')
            cards.append(f'''
<div class="card">
  <img src="{src}" alt="{path}">
  <div class="meta">
    <table class="rt">
      {rater_rows}
      <tr class="sep"><td>human mean</td><td class="v">{row['hum']:.2f}</td></tr>
      <tr class="gem"><td>gemini</td><td class="v">{row['gem']:.2f}</td></tr>
      <tr><td class="sub">gemini range</td><td class="v sub">{row['gem_lo']:.1f}–{row['gem_hi']:.1f}</td></tr>
      <tr class="gap {direction}"><td>gap (gem−human)</td><td class="v">{row['gap']:+.2f} ({direction})</td></tr>
    </table>
    <details class="reason"><summary>gemini reasoning (5 replicates)</summary>
      <ol>{reas}</ol>
    </details>
    {link}
    <div class="path">{path}</div>
  </div>
</div>''')
        cards.append('</div>')

    summary = " · ".join(f"{room}: {counts[room]}" for room in ROOMS)
    css = """
*{box-sizing:border-box}
body{font-family:-apple-system,"Helvetica Neue",Arial,sans-serif;color:#1a1a1a;
     max-width:1200px;margin:0 auto;padding:16px 20px 60px;background:#fff}
h1{font-size:22px;margin:0 0 4px}
.lead{color:#555;margin:0 0 20px;font-size:14px}
h2{font-size:17px;margin:30px 0 10px;padding-top:8px;border-top:2px solid #e2e2e2;
   text-transform:capitalize}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:16px}
.card{border:1px solid #ddd;border-radius:8px;overflow:hidden;background:#fafafa}
.card img{width:100%;height:220px;object-fit:cover;display:block;background:#eee}
.meta{padding:8px 10px 10px}
details.reason{margin-top:8px;font-size:12px}
details.reason>summary{cursor:pointer;color:#1a5fb4;font-weight:600;font-size:12px}
details.reason ol{margin:6px 0 0;padding-left:18px}
details.reason li{margin:0 0 6px;line-height:1.4;color:#333}
details.reason .rep{display:inline-block;color:#8a5a00;font-weight:600;
    font-variant-numeric:tabular-nums;margin-right:4px}
table.rt{width:100%;border-collapse:collapse;font-size:13px}
table.rt td{padding:2px 4px}
table.rt td.v{text-align:right;font-variant-numeric:tabular-nums;font-weight:600}
tr.sep td{border-top:1px solid #ddd;padding-top:4px}
tr.gem td{color:#1a5fb4}
tr.gap td{font-weight:700}
tr.gap.high td{color:#b3261e}
tr.gap.low td{color:#8a5a00}
td.sub,.v.sub{color:#888;font-weight:400;font-size:11px}
.zlink{display:inline-block;margin-top:8px;font-size:12px;font-weight:600;
    color:#1a5fb4;text-decoration:none}
.zlink:hover{text-decoration:underline}
.zlink.none{color:#aaa;font-weight:400}
.path{margin-top:6px;font-size:10px;color:#999;word-break:break-all}
"""
    html = f'''<!doctype html><html><head><meta charset="utf-8">
<title>gemini vs human — images off by >{args.threshold:g}</title>
<style>{css}</style></head><body>
<h1>Images where gemini differs from humans by more than {args.threshold:g} point</h1>
<p class="lead">Human reference: {', '.join(present)}. {len(df)} images total &nbsp;·&nbsp; {summary}.
Red = gemini scored higher than humans, amber = lower. Sorted by gap size within each room.</p>
{''.join(cards)}
</body></html>'''
    args.out.write_text(html, encoding="utf-8")
    print(f"wrote {args.out}  ({len(df)} images: {summary})")


if __name__ == "__main__":
    main()
