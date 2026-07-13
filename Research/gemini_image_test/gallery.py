#!/usr/bin/env python3
"""Gallery builder for the gemini_image_test rescoring harness.

Two jobs:
  1. build_page(...)  -> writes one self-contained page HTML for a single run
                         (cards per room + run cost banner + prompts at bottom).
  2. rebuild_index()  -> regenerates gallery/index.html: a tabbed shell that
                         embeds every page (page 1 = the tuned3 reference) as an
                         isolated iframe srcdoc, so the whole thing works on a
                         plain double-click with no server and no CSS collisions.

Each run appends a page file under gallery/pages/ and a row to
gallery/pages/manifest.json; index.html is regenerated from that manifest.

The card design mirrors model_comparison's diff gallery (per-rater scores,
human mean, model score, signed gap, reasoning, Zillow link) so pages line up
visually with the tuned3 reference on page 1.
"""
import base64
import html
import json
import mimetypes
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
PAGES_DIR = HERE / "gallery" / "pages"
MANIFEST = PAGES_DIR / "manifest.json"
INDEX = HERE / "gallery" / "index.html"
REPO_ROOT = HERE.parents[1]  # Research/gemini_image_test -> Research -> repo
IMAGES_ROOT = REPO_ROOT
SOLD_CSV = REPO_ROOT / "data" / "cleaned_sold.csv"

ROOM_ORDER = ["kitchen", "bathroom", "bedroom", "living_room"]
MAX_LEVEL = 8  # every room scores 1–8


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _inline_img(path: Path) -> str:
    if not path.exists():
        return ""
    mime = mimetypes.guess_type(str(path))[0] or "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()


def _esc(t) -> str:
    return html.escape(str(t))


def load_manifest() -> list:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text())
    return []


def save_manifest(rows: list) -> None:
    MANIFEST.write_text(json.dumps(rows, indent=2) + "\n")


# ---------------------------------------------------------------------------
# per-run page
# ---------------------------------------------------------------------------
PAGE_CSS = """
*{box-sizing:border-box}
body{font-family:-apple-system,"Helvetica Neue",Arial,sans-serif;color:#1a1a1a;
     max-width:1200px;margin:0 auto;padding:16px 20px 60px;background:#fff}
h1{font-size:22px;margin:0 0 4px}
.lead{color:#555;margin:0 0 6px;font-size:14px}
.banner{display:flex;flex-wrap:wrap;gap:8px 18px;margin:12px 0 8px;padding:12px 14px;
   border:1px solid #e2e2e2;border-radius:8px;background:#f7f7f8;font-size:13px}
.banner b{font-variant-numeric:tabular-nums}
.controls{margin:10px 0 4px;font-size:13px}
.controls button{font:inherit;cursor:pointer;border:1px solid #ccc;background:#fff;
   border-radius:6px;padding:4px 10px;margin-right:6px}
.controls button.active{background:#1a5fb4;border-color:#1a5fb4;color:#fff}
h2{font-size:17px;margin:26px 0 10px;padding-top:8px;border-top:2px solid #e2e2e2;
   text-transform:capitalize}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:16px}
.card{border:1px solid #ddd;border-radius:8px;overflow:hidden;background:#fafafa}
.card.hide{display:none}
.card img{width:100%;height:220px;object-fit:cover;display:block;background:#eee}
.meta{padding:8px 10px 10px}
details.reason{margin-top:8px;font-size:12px}
details.reason>summary{cursor:pointer;color:#1a5fb4;font-weight:600;font-size:12px}
details.reason p{margin:6px 0 0;line-height:1.4;color:#333}
table.rt{width:100%;border-collapse:collapse;font-size:13px}
table.rt td{padding:2px 4px}
table.rt td.v{text-align:right;font-variant-numeric:tabular-nums;font-weight:600}
tr.sep td{border-top:1px solid #ddd;padding-top:4px}
tr.gem td{color:#1a5fb4}
tr.gap td{font-weight:700}
tr.gap.high td{color:#b3261e}
tr.gap.low td{color:#8a5a00}
tr.gap.ok td{color:#2a7a2a}
.zlink{display:inline-block;margin-top:8px;font-size:12px;font-weight:600;
    color:#1a5fb4;text-decoration:none}
.zlink:hover{text-decoration:underline}
.zlink.none{color:#aaa;font-weight:400}
.path{margin-top:6px;font-size:10px;color:#999;word-break:break-all}
.prompts{margin-top:40px}
.prompts h2{border-top:2px solid #1a5fb4}
.prompts details{margin:0 0 10px;border:1px solid #e2e2e2;border-radius:8px;background:#fafafa}
.prompts summary{cursor:pointer;font-weight:600;padding:10px 12px;text-transform:capitalize}
.prompts pre{margin:0;padding:0 14px 14px;white-space:pre-wrap;font-size:12px;
   line-height:1.45;color:#333;font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
"""

PAGE_JS = """
function filterCards(mode, btn){
  document.querySelectorAll('.controls button').forEach(function(b){b.classList.remove('active')});
  btn.classList.add('active');
  document.querySelectorAll('.card').forEach(function(c){
    var off = parseFloat(c.dataset.absgap);
    c.classList.toggle('hide', mode==='diff' && !(off>1));
  });
  // re-count each room header to the number of currently visible cards
  document.querySelectorAll('.grid').forEach(function(g){
    var n = g.querySelectorAll('.card:not(.hide)').length;
    var h = g.previousElementSibling;
    var s = h && h.querySelector('.rcount');
    if(s){ s.textContent = n; }
  });
}
"""


def build_page(run_scores: pd.DataFrame, rooms_scored: list, cost_usd: float,
               label: str, timestamp: str, raters=("harvey", "robin", "seb")) -> str:
    """Render one run's page HTML from its gemini scores + the human rankings.

    run_scores columns: image_path, zpid, room_type, score, reasoning.
    Returns the written page filename (relative to gallery/pages/).
    """
    ratings = pd.read_csv(HERE / "data" / "human_ratings.csv")
    raters = [r for r in raters]

    # per-rater wide table + human mean
    rr = ratings[ratings["rater"].isin(raters)]
    wide = rr.pivot_table(index="image_path", columns="rater", values="score")
    present = [r for r in raters if r in wide.columns]
    hum_mean = wide[present].mean(axis=1)

    # zpid -> Zillow listing url
    href_of = {}
    if SOLD_CSV.exists():
        sold = pd.read_csv(SOLD_CSV, usecols=["zpid", "listing-link-href"])
        href_of = {str(z): h for z, h in
                   zip(sold["zpid"].astype(str), sold["listing-link-href"])}

    df = run_scores.copy()
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df["hum"] = df["image_path"].map(hum_mean)
    df["gap"] = df["score"] - df["hum"]
    df["abs_gap"] = df["gap"].abs()

    n_off = int((df["abs_gap"] > 1).sum())
    mean_abs = df["abs_gap"].dropna().mean()

    cards = []
    for room in [r for r in ROOM_ORDER if r in rooms_scored]:
        sub = df[df["room_type"] == room].sort_values("abs_gap", ascending=False,
                                                       na_position="last")
        if sub.empty:
            continue
        cards.append(f'<h2 class="rh">{room.replace("_", " ")} — '
                     f'<span class="rcount">{len(sub)}</span> images '
                     f'(scale 1–{MAX_LEVEL})</h2>')
        cards.append('<div class="grid">')
        for _, row in sub.iterrows():
            path = row["image_path"]
            src = _inline_img((IMAGES_ROOT / path).resolve())
            rater_rows = "".join(
                (f'<tr><td>{r}</td><td class="v">{wide.loc[path, r]:.0f}</td></tr>'
                 if (path in wide.index and pd.notna(wide.loc[path, r]))
                 else f'<tr><td>{r}</td><td class="v">–</td></tr>')
                for r in present)
            gap = row["gap"]
            if pd.isna(gap):
                direction, gap_txt = "ok", "n/a (no human rating)"
                absgap = -1.0
            else:
                direction = "high" if gap > 1 else ("low" if gap < -1 else "ok")
                gap_txt = f"{gap:+.2f} ({direction})"
                absgap = float(row["abs_gap"])
            hum_txt = "–" if pd.isna(row["hum"]) else f"{row['hum']:.2f}"
            zpid = str(row.get("zpid", ""))
            href = href_of.get(zpid)
            link = (f'<a class="zlink" href="{_esc(href)}" target="_blank" '
                    f'rel="noopener">↗ Zillow listing (zpid {zpid})</a>'
                    if isinstance(href, str) and href else
                    f'<span class="zlink none">no listing link (zpid {zpid})</span>')
            reason = _esc(row.get("reasoning", ""))
            if pd.isna(row["score"]):
                other = row.get("other_room")
                gem_txt = f"not {room.replace('_',' ')}" + (
                    f" ({_esc(other)})" if isinstance(other, str) and other else "")
            else:
                gem_txt = f"{row['score']:.0f}"
            cards.append(f'''
<div class="card" data-absgap="{absgap:.3f}">
  <img src="{src}" alt="{_esc(path)}">
  <div class="meta">
    <table class="rt">
      {rater_rows}
      <tr class="sep"><td>human mean</td><td class="v">{hum_txt}</td></tr>
      <tr class="gem"><td>gemini</td><td class="v">{gem_txt}</td></tr>
      <tr class="gap {direction}"><td>gap (gem−human)</td><td class="v">{gap_txt}</td></tr>
    </table>
    <details class="reason"><summary>gemini reasoning</summary>
      <p>{reason}</p>
    </details>
    {link}
    <div class="path">{_esc(path)}</div>
  </div>
</div>''')
        cards.append('</div>')

    # prompts used this run, at the bottom
    prompt_blocks = ['<div class="prompts"><h2>Prompts used this run</h2>']
    for room in [r for r in ROOM_ORDER if r in rooms_scored]:
        ptxt = (HERE / "rooms" / room / "prompt.txt").read_text(encoding="utf-8")
        prompt_blocks.append(
            f'<details><summary>{room.replace("_", " ")} — prompt.txt</summary>'
            f'<pre>{_esc(ptxt)}</pre></details>')
    prompt_blocks.append('</div>')

    cost_txt = "n/a (dry run)" if cost_usd is None else f"${cost_usd:.4f}"
    mean_txt = "n/a" if pd.isna(mean_abs) else f"{mean_abs:.2f}"
    banner = (
        '<div class="banner">'
        f'<span><b>{len(df)}</b> images scored</span>'
        f'<span>rooms: <b>{", ".join(r.replace("_"," ") for r in rooms_scored)}</b></span>'
        f'<span>model: <b>gemini-3.5-flash</b> (1 call/image)</span>'
        f'<span>run cost: <b>{cost_txt}</b></span>'
        f'<span>mean |gap|: <b>{mean_txt}</b></span>'
        f'<span><b>{n_off}</b> off by &gt;1</span>'
        '</div>')

    body = f'''<h1>{_esc(label)}</h1>
<p class="lead">Run {_esc(timestamp)}. Human reference: {", ".join(present)}.
Red = gemini higher than humans, amber = lower, green = within 1. Sorted by gap size within each room.</p>
{banner}
<div class="controls">Show:
  <button class="active" onclick="filterCards('all',this)">All images</button>
  <button onclick="filterCards('diff',this)">Only off by &gt;1</button>
</div>
{''.join(cards)}
{''.join(prompt_blocks)}'''

    doc = (f'<!doctype html><html><head><meta charset="utf-8">'
           f'<title>{_esc(label)}</title><style>{PAGE_CSS}</style></head>'
           f'<body>{body}<script>{PAGE_JS}</script></body></html>')

    # next sequential page number
    existing = sorted(PAGES_DIR.glob("page_*.html"))
    nums = [int(p.stem.split("_")[1]) for p in existing if p.stem.split("_")[1].isdigit()]
    n = (max(nums) + 1) if nums else 1
    fname = f"page_{n:03d}_{timestamp.replace(':', '').replace('-', '')}.html"
    (PAGES_DIR / fname).write_text(doc, encoding="utf-8")
    return fname


# ---------------------------------------------------------------------------
# tabbed index that embeds every page + a Compare view
# ---------------------------------------------------------------------------
INDEX_CSS = """
*{box-sizing:border-box}
html,body{height:100%;margin:0}
body{font-family:-apple-system,"Helvetica Neue",Arial,sans-serif;display:flex;
     flex-direction:column;background:#fff}
header{padding:10px 16px 0;border-bottom:1px solid #e2e2e2;background:#fafafa}
header h1{font-size:15px;margin:0 0 8px;color:#333}
.tabs{display:flex;flex-wrap:wrap;gap:4px}
.tabs button{font:inherit;font-size:13px;cursor:pointer;border:1px solid #ccc;
   border-bottom:none;background:#eee;border-radius:8px 8px 0 0;padding:7px 12px;color:#333}
.tabs button.active{background:#fff;color:#1a5fb4;font-weight:600;
   box-shadow:0 1px 0 #fff;position:relative;top:1px}
.tabs button.cmptab{background:#eef4fc;border-color:#b9d3f2}
.tabs button.cmptab.active{background:#fff}
.tabs .cost{color:#8a5a00;font-weight:600;font-variant-numeric:tabular-nums}
.frames{flex:1;position:relative}
.frames .frame{position:absolute;inset:0;width:100%;height:100%;border:0;display:none;background:#fff}
.frames .frame.active{display:block}
.frames .cmp{overflow:auto}
/* compare view */
.cmpbar{position:sticky;top:0;background:#f7f7f8;border-bottom:1px solid #e2e2e2;
   padding:12px 16px;display:flex;flex-wrap:wrap;gap:8px 14px;align-items:center;font-size:14px;z-index:2}
.cmpbar select{font:inherit;padding:4px 6px;border:1px solid #ccc;border-radius:6px;max-width:280px}
.cmpsum{color:#333}
.cmp h2{font-size:16px;margin:18px 16px 8px;text-transform:capitalize;
   border-top:2px solid #e2e2e2;padding-top:10px}
.cgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:14px;padding:0 16px 8px}
.ccard{border:1px solid #ddd;border-radius:8px;overflow:hidden;background:#fafafa}
.ccard img{width:100%;height:190px;object-fit:cover;display:block;background:#eee}
.cmeta{padding:8px 10px 10px}
table.ct{width:100%;border-collapse:collapse;font-size:13px}
table.ct td{padding:2px 4px}
table.ct td.v{text-align:right;font-weight:600;font-variant-numeric:tabular-nums}
table.ct tr.sep td{border-top:1px solid #ddd}
table.ct tr.delta.up td{color:#1a7f37}
table.ct tr.delta.down td{color:#b3261e}
.cmp details.reason{margin-top:8px;font-size:12px}
.cmp details.reason>summary{cursor:pointer;color:#1a5fb4;font-weight:600}
.cmp details.reason p{margin:6px 0 0;line-height:1.4;color:#333}
.cmp .zlink{display:inline-block;margin-top:8px;font-size:12px;font-weight:600;color:#1a5fb4;text-decoration:none}
.cmp .empty{padding:24px 16px;color:#555}
"""

CMP_JS = """
function cmpEsc(t){var d=document.createElement('div');d.textContent=(t==null?'':t);return d.innerHTML;}
function cmpOptions(){return CMP.runs.map(function(r,i){
  return '<option value="'+i+'">'+cmpEsc(r.label)+'</option>';}).join('');}
function initCmp(){
  var A=document.getElementById('cmpA'), B=document.getElementById('cmpB');
  if(!A) return;
  A.innerHTML=cmpOptions(); B.innerHTML=cmpOptions();
  A.value=Math.max(0,CMP.runs.length-2); B.value=CMP.runs.length-1;
  renderCmp();
}
function renderCmp(){
  var out=document.getElementById('cmpout'), sum=document.getElementById('cmpsum');
  var ai=+document.getElementById('cmpA').value, bi=+document.getElementById('cmpB').value;
  if(CMP.runs.length<2){ sum.textContent=''; out.innerHTML='<p class="empty">Need at least two runs to compare.</p>'; return; }
  if(ai===bi){ sum.textContent='Pick two different runs.'; out.innerHTML=''; return; }
  var A=CMP.runs[ai], B=CMP.runs[bi], shared=0, changed=[];
  for(var p in A.by){ if(B.by.hasOwnProperty(p)){
    var sa=A.by[p].s, sb=B.by[p].s;
    if(sa!=null && sb!=null){ shared++; if(sa!==sb){
      changed.push({p:p, sa:sa, sb:sb, room:(A.by[p].room||B.by[p].room||''),
                    z:A.by[p].z, ra:A.by[p].r, rb:B.by[p].r}); } }
  }}
  changed.sort(function(x,y){ if(x.room<y.room)return -1; if(x.room>y.room)return 1;
    return Math.abs(y.sb-y.sa)-Math.abs(x.sb-x.sa); });
  sum.innerHTML='<b>'+changed.length+'</b> of '+shared+' shared images changed score '
    +'(A = '+cmpEsc(A.label)+', B = '+cmpEsc(B.label)+')';
  if(!changed.length){ out.innerHTML='<p class="empty">No gemini score changed between these two runs.</p>'; return; }
  var html='', room=null;
  changed.forEach(function(c){
    if(c.room!==room){ if(room!==null)html+='</div>'; room=c.room;
      html+='<h2>'+cmpEsc(room.replace('_',' '))+'</h2><div class="cgrid">'; }
    var d=c.sb-c.sa, dir=d>0?'up':'down';
    var hum=CMP.hum[c.p], humtxt=hum?hum.toFixed(2):'–';
    var href=CMP.href[c.z];
    var link=href?'<a class="zlink" href="'+cmpEsc(href)+'" target="_blank" rel="noopener">↗ Zillow (zpid '+cmpEsc(c.z)+')</a>':'';
    html+='<div class="ccard"><img loading="lazy" src="'+(CMP.img[c.p]||'')+'">'
      +'<div class="cmeta"><table class="ct">'
      +'<tr><td>human mean</td><td class="v">'+humtxt+'</td></tr>'
      +'<tr class="sep"><td>A · '+cmpEsc(A.label)+'</td><td class="v">'+c.sa+'</td></tr>'
      +'<tr><td>B · '+cmpEsc(B.label)+'</td><td class="v">'+c.sb+'</td></tr>'
      +'<tr class="delta '+dir+'"><td>change (B−A)</td><td class="v">'+(d>0?'+':'')+d+'</td></tr>'
      +'</table>'
      +'<details class="reason"><summary>reasoning (both runs)</summary>'
      +'<p><b>A · '+cmpEsc(A.label)+':</b> '+cmpEsc(c.ra)+'</p>'
      +'<p><b>B · '+cmpEsc(B.label)+':</b> '+cmpEsc(c.rb)+'</p></details>'
      +link+'</div></div>';
  });
  if(room!==null)html+='</div>';
  out.innerHTML=html;
}
"""


def _run_dir(row: dict) -> Path:
    """runs/<timestamp-without-colons>/ for a manifest row."""
    return HERE / "runs" / str(row.get("timestamp", "")).replace(":", "")


def build_compare_data(rows: list) -> dict:
    """Collect every run's per-image scores plus shared image / human-rating /
    Zillow maps, so the browser can diff any two runs client-side."""
    # human mean per image
    hum = {}
    rpath = HERE / "data" / "human_ratings.csv"
    if rpath.exists():
        ratings = pd.read_csv(rpath)
        wide = ratings.pivot_table(index="image_path", columns="rater", values="score")
        for path, mean in wide.mean(axis=1).items():
            if pd.notna(mean):
                hum[path] = round(float(mean), 4)

    href = {}
    if SOLD_CSV.exists():
        sold = pd.read_csv(SOLD_CSV, usecols=["zpid", "listing-link-href"])
        href = {str(z): h for z, h in zip(sold["zpid"].astype(str), sold["listing-link-href"])
                if isinstance(h, str)}

    runs, images = [], {}
    for row in rows:
        csv = _run_dir(row) / "scores.csv"
        if not csv.exists():
            continue  # a tab with no run data (skip from compare)
        df = pd.read_csv(csv)
        df["score"] = pd.to_numeric(df["score"], errors="coerce")
        by = {}
        for _, r in df.iterrows():
            path = r["image_path"]
            by[path] = {
                "s": None if pd.isna(r["score"]) else int(r["score"]),
                "r": "" if pd.isna(r.get("reasoning")) else str(r.get("reasoning")),
                "room": r.get("room_type", ""),
                "z": str(r.get("zpid", "")),
            }
            if path not in images:
                images[path] = _inline_img((IMAGES_ROOT / path).resolve())
        runs.append({"label": row["label"], "ts": row.get("timestamp", ""), "by": by})

    return {"runs": runs, "img": images, "hum": hum, "href": href}


def rebuild_index() -> None:
    rows = load_manifest()
    tabs, frames = [], []

    # tab 0 = Compare view (a div frame, not an iframe)
    tabs.append('<button class="cmptab" onclick="showTab(0,this)">⇄ Compare runs</button>')
    frames.append(
        '<div class="frame cmp">'
        '<div class="cmpbar">'
        '<label>Run A <select id="cmpA" onchange="renderCmp()"></select></label>'
        '<span>vs</span>'
        '<label>Run B <select id="cmpB" onchange="renderCmp()"></select></label>'
        '<span id="cmpsum" class="cmpsum"></span>'
        '</div><div id="cmpout"></div></div>')

    for i, row in enumerate(rows):
        idx = i + 1  # compare occupies slot 0
        page_path = PAGES_DIR / row["file"]
        content = page_path.read_text(encoding="utf-8") if page_path.exists() else \
            "<p>missing page file</p>"
        srcdoc = content.replace("&", "&amp;").replace('"', "&quot;")
        active = " active" if i == len(rows) - 1 else ""  # newest run open by default
        cost = row.get("cost_usd")
        cost_txt = "" if cost in (None, "",) else f' <span class="cost">${float(cost):.4f}</span>'
        tabs.append(f'<button class="{active.strip()}" onclick="showTab({idx},this)">'
                    f'{_esc(row["label"])}{cost_txt}</button>')
        frames.append(f'<iframe class="frame{active}" srcdoc="{srcdoc}"></iframe>')

    cmp_json = json.dumps(build_compare_data(rows)).replace("</", "<\\/")

    show_js = """
function showTab(i, btn){
  var f=document.querySelectorAll('.frames .frame');
  var b=document.querySelectorAll('.tabs button');
  for(var k=0;k<f.length;k++){f[k].classList.toggle('active',k===i);}
  for(var k=0;k<b.length;k++){b[k].classList.remove('active');}
  btn.classList.add('active');
}
"""
    doc = (f'<!doctype html><html><head><meta charset="utf-8">'
           f'<title>gemini_image_test — gallery</title><style>{INDEX_CSS}</style></head>'
           f'<body><header><h1>gemini-3.5-flash luxury scoring — run gallery '
           f'({len(rows)} runs)</h1><div class="tabs">{"".join(tabs)}</div></header>'
           f'<div class="frames">{"".join(frames)}</div>'
           f'<script>const CMP={cmp_json};{show_js}{CMP_JS}initCmp();</script></body></html>')
    INDEX.write_text(doc, encoding="utf-8")


def prune_missing_runs() -> list:
    """Drop manifest entries (and their orphaned page files) whose runs/<ts>/
    folder was deleted. Returns the labels that were removed."""
    kept, dropped = [], []
    for row in load_manifest():
        if (_run_dir(row) / "scores.csv").exists():
            kept.append(row)
        else:
            dropped.append(row)
            pf = PAGES_DIR / row["file"]
            if pf.exists():
                pf.unlink()
    if dropped:
        save_manifest(kept)
    return [r["label"] for r in dropped]


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(
        description="Rebuild gallery/index.html from the manifest + runs/ data.")
    ap.add_argument("--prune", action="store_true",
                    help="first drop any runs whose runs/<ts>/ folder was deleted "
                         "(removes their manifest entry and page file), then rebuild")
    args = ap.parse_args()
    if args.prune:
        gone = prune_missing_runs()
        print("pruned:", ", ".join(gone) if gone else "(none)")
    rebuild_index()
    print(f"wrote {INDEX}")
