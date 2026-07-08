#!/usr/bin/env python3
"""
Stage 2b — multi-rater human rating web app (single-file Flask).

Serves a keyboard-driven rating UI for a fixed subset of the Stage 2 manifest
(~15-20 images per room type). Every rater rates ALL subset images, each in
their own deterministic randomized order; sessions are resumable per rater and
raters can work concurrently. Ratings land in SQLite (WAL mode, one connection
per request, unique per rater+image) so concurrent writes are safe.

The subset is drawn once on first startup and frozen in the DB — restarts and
flag changes never resample it, so every rater sees the same images.

Runs independently of the scoring stages: humans can rate while API calls run.

Usage:
    python stage2b_rating_app.py --manifest outputs/sample_manifest.csv \
        --images-root ../.. --db outputs/human_ratings.sqlite

Then share http://<lan-ip>:5000 with raters (see README for Tailscale /
cloudflared notes for raters outside the LAN).
"""

import argparse
import csv
import hashlib
import random
import socket
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, abort, jsonify, request, send_file

from common import ROOMS, ROOM_TYPES

app = Flask(__name__)
CFG = {}  # filled in main(): db path, images_root


# ---------------------------------------------------------------------------
# DB
# ---------------------------------------------------------------------------
def db():
    conn = sqlite3.connect(CFG["db"], timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.row_factory = sqlite3.Row
    return conn


def init_db(manifest_path: Path, per_type: int, seed: int):
    conn = db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS subset (
            id INTEGER PRIMARY KEY,
            image_path TEXT UNIQUE NOT NULL,
            zpid TEXT NOT NULL,
            room_type TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS ratings (
            rater TEXT NOT NULL,
            image_path TEXT NOT NULL,
            room_type TEXT NOT NULL,
            score INTEGER NOT NULL,
            timestamp TEXT NOT NULL,
            UNIQUE (rater, image_path)
        );
    """)
    n = conn.execute("SELECT COUNT(*) c FROM subset").fetchone()["c"]
    if n:
        print(f"Reusing frozen image set already in DB ({n} images) — flags ignored.")
    else:
        with open(manifest_path, newline="", encoding="utf-8") as f:
            manifest = list(csv.DictReader(f))
        rng = random.Random(seed)
        if per_type is None:  # no subset — rate every manifest image
            rows = list(manifest)
        else:
            rows = []
            for room in ROOM_TYPES:
                items = [m for m in manifest if m["room_type"] == room]
                rng.shuffle(items)
                rows.extend(items[:per_type])
        conn.executemany(
            "INSERT INTO subset (image_path, zpid, room_type) VALUES (?, ?, ?)",
            [(r["image_path"], r["zpid"], r["room_type"]) for r in rows])
        conn.commit()
        kind = "full manifest" if per_type is None else "rating subset"
        print(f"Froze {kind}: {len(rows)} images")
        for room in ROOM_TYPES:
            print(f"  {room}: {sum(1 for r in rows if r['room_type'] == room)}")
    conn.close()


def rater_order(rater: str):
    """All subset ids in a per-rater deterministic shuffled order."""
    conn = db()
    ids = [r["id"] for r in conn.execute("SELECT id FROM subset ORDER BY id")]
    conn.close()
    seed = int(hashlib.sha256(rater.encode("utf-8")).hexdigest()[:12], 16)
    random.Random(seed).shuffle(ids)
    return ids


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
@app.get("/api/nav")
def api_nav():
    """Position-based navigation over the rater's fixed order so raters can go
    back/forward and re-score. With no `pos`, returns the first unrated image
    (resume point). pos == total means the rater is done."""
    rater = (request.args.get("rater") or "").strip()
    if not rater:
        abort(400, "missing rater")
    conn = db()
    rows_by_id = {r["id"]: r for r in conn.execute("SELECT * FROM subset")}
    scores = {r["image_path"]: r["score"] for r in conn.execute(
        "SELECT image_path, score FROM ratings WHERE rater = ?", (rater,))}
    conn.close()

    order = rater_order(rater)  # deterministic per-rater shuffle of subset ids
    total = len(order)
    done_count = len(scores)

    pos_arg = request.args.get("pos")
    if pos_arg is None:  # resume point = first image without a rating
        pos = next((i for i, sid in enumerate(order)
                    if rows_by_id[sid]["image_path"] not in scores), total)
    else:
        try:
            pos = int(pos_arg)
        except ValueError:
            abort(400, "bad pos")
        pos = max(0, min(pos, total))

    if pos >= total:
        return jsonify({"done": True, "done_count": done_count, "total": total, "pos": total})

    row = rows_by_id[order[pos]]
    return jsonify({
        "done": False, "pos": pos, "total": total, "done_count": done_count,
        "subset_id": row["id"], "image_path": row["image_path"],
        "room_type": row["room_type"], "max_level": ROOMS[row["room_type"]]["max_level"],
        "image_url": f"/image/{row['id']}", "grid_url": f"/grid/{row['room_type']}",
        "existing_score": scores.get(row["image_path"]),  # None if not yet rated
    })


@app.post("/api/rate")
def api_rate():
    data = request.get_json(force=True)
    rater = (data.get("rater") or "").strip()
    sid = data.get("subset_id")
    score = data.get("score")
    if not rater or sid is None or score is None:
        abort(400, "missing rater/subset_id/score")
    conn = db()
    row = conn.execute("SELECT * FROM subset WHERE id = ?", (sid,)).fetchone()
    if row is None:
        conn.close()
        abort(404, "unknown subset id")
    score = int(score)
    if not (1 <= score <= ROOMS[row["room_type"]]["max_level"]):
        conn.close()
        abort(400, "score out of range for this room type")
    conn.execute(
        """INSERT INTO ratings (rater, image_path, room_type, score, timestamp)
           VALUES (?, ?, ?, ?, ?)
           ON CONFLICT (rater, image_path) DO UPDATE
           SET score = excluded.score, timestamp = excluded.timestamp""",
        (rater, row["image_path"], row["room_type"], score,
         datetime.now(timezone.utc).isoformat()))
    conn.commit()
    conn.close()
    return jsonify({"ok": True})


# ---------------------------------------------------------------------------
# Static: subset images + anchor grids only (no arbitrary paths)
# ---------------------------------------------------------------------------
@app.get("/image/<int:sid>")
def image(sid):
    conn = db()
    row = conn.execute("SELECT image_path FROM subset WHERE id = ?", (sid,)).fetchone()
    conn.close()
    if row is None:
        abort(404)
    return send_file(CFG["images_root"] / row["image_path"])


@app.get("/grid/<room_type>")
def grid(room_type):
    if room_type not in ROOMS:
        abort(404)
    return send_file(ROOMS[room_type]["assets_dir"] / "grid.png")


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Luxury rating</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root { --ink:#0b0b0b; --ink2:#52514e; --accent:#2a78d6; --surface:#fcfcfb; --line:#e6e5e2; }
  * { box-sizing:border-box; margin:0; }
  body { font-family:-apple-system,system-ui,sans-serif; background:var(--surface); color:var(--ink); height:100vh; display:flex; flex-direction:column; }
  header { display:flex; align-items:center; gap:16px; padding:10px 16px; border-bottom:1px solid var(--line); }
  header .room { font-weight:700; font-size:18px; text-transform:capitalize; }
  header .hint { color:var(--ink2); font-size:14px; }
  header .prog { margin-left:auto; color:var(--ink2); font-size:14px; font-variant-numeric:tabular-nums; }
  .bar { height:4px; background:var(--line); } .bar>div { height:100%; background:var(--accent); width:0; transition:width .2s; }
  main { flex:1; display:flex; min-height:0; }
  .pane { flex:1; display:flex; flex-direction:column; min-width:0; padding:10px; }
  .pane h2 { font-size:12px; text-transform:uppercase; letter-spacing:.06em; color:var(--ink2); margin-bottom:6px; }
  .pane .imgbox { flex:1; min-height:0; display:flex; align-items:center; justify-content:center; }
  .pane img { max-width:100%; max-height:100%; object-fit:contain; border-radius:4px; }
  #target-pane { border-left:1px solid var(--line); }
  .keys { display:flex; gap:8px; justify-content:center; align-items:center; padding:10px; border-top:1px solid var(--line); flex-wrap:wrap; }
  .keys button { width:44px; height:44px; font-size:18px; font-weight:600; border:1px solid var(--line); border-radius:8px; background:#fff; cursor:pointer; }
  .keys button:hover { border-color:var(--accent); color:var(--accent); }
  .keys button.chosen { background:var(--accent); color:#fff; border-color:var(--accent); }
  .nav { width:auto !important; padding:0 12px; color:var(--ink2); }
  .nav:disabled { opacity:.35; cursor:default; }
  .sep { width:1px; height:28px; background:var(--line); margin:0 4px; }
  .flash { animation:flash .25s; } @keyframes flash { 0%{opacity:.3} 100%{opacity:1} }
  #login,#done { flex:1; display:flex; flex-direction:column; align-items:center; justify-content:center; gap:14px; }
  #login input { font-size:18px; padding:10px 14px; border:1px solid var(--line); border-radius:8px; width:280px; }
  #login button { font-size:16px; padding:10px 20px; border:0; border-radius:8px; background:var(--accent); color:#fff; cursor:pointer; }
  .hidden { display:none !important; }
</style></head>
<body>
<header>
  <span class="room" id="room"></span>
  <span class="hint" id="hint"></span>
  <span class="prog" id="prog"></span>
</header>
<div class="bar"><div id="barfill"></div></div>

<div id="login">
  <h1>Luxury rating</h1>
  <p style="color:var(--ink2)">Enter your name (used to save your progress):</p>
  <input id="name" placeholder="your name" autofocus>
  <button onclick="start()">Start rating</button>
</div>

<main id="rate" class="hidden">
  <div class="pane">
    <h2>Anchor grid (Level 1 = lowest)</h2>
    <div class="imgbox"><img id="gridimg" alt="anchor grid"></div>
  </div>
  <div class="pane" id="target-pane">
    <h2 id="scoreHint">Image to score — press a number key</h2>
    <div class="imgbox"><img id="targetimg" alt="image to score"></div>
    <div class="keys">
      <button class="nav" id="backBtn" title="Previous image (←)">← Back</button>
      <span class="sep"></span>
      <span id="keys" style="display:flex;gap:8px"></span>
      <span class="sep"></span>
      <button class="nav" id="fwdBtn" title="Next image without changing (→)">Skip →</button>
    </div>
  </div>
</main>

<div id="done" class="hidden">
  <h1>All done — thank you!</h1>
  <p id="donemsg" style="color:var(--ink2)"></p>
</div>

<script>
let rater = localStorage.getItem("rater") || "";
let cur = null;
if (rater) { document.getElementById("name").value = rater; }

function start() {
  rater = document.getElementById("name").value.trim();
  if (!rater) return;
  localStorage.setItem("rater", rater);
  document.getElementById("login").classList.add("hidden");
  load();  // no pos -> server resumes at first unrated
}

// Navigate to a position (undefined = resume point). Renders that image plus
// any score already recorded for it, so raters can go back and fix mistakes.
async function load(pos) {
  const q = "/api/nav?rater=" + encodeURIComponent(rater) + (pos === undefined ? "" : "&pos=" + pos);
  cur = await (await fetch(q)).json();
  const prog = document.getElementById("prog");
  if (cur.done) {
    document.getElementById("rate").classList.add("hidden");
    document.getElementById("done").classList.remove("hidden");
    document.getElementById("donemsg").textContent =
      cur.done_count + " / " + cur.total + " images rated as “" + rater + "”.";
    prog.textContent = cur.done_count + " / " + cur.total;
    document.getElementById("barfill").style.width = "100%";
    return;
  }
  document.getElementById("done").classList.add("hidden");
  document.getElementById("rate").classList.remove("hidden");
  document.getElementById("room").textContent = cur.room_type.replace("_", " ");
  const rated = cur.existing_score != null;
  document.getElementById("scoreHint").textContent =
    (rated ? "Already rated " + cur.existing_score + " — press a number to change it"
           : "Image to score — press a number key");
  document.getElementById("hint").textContent =
    "image " + (cur.pos + 1) + " of " + cur.total + " · press 1–" + cur.max_level;
  prog.textContent = cur.done_count + " / " + cur.total + " rated";
  document.getElementById("barfill").style.width = (100 * cur.done_count / cur.total) + "%";
  const g = document.getElementById("gridimg"), t = document.getElementById("targetimg");
  g.src = cur.grid_url; t.src = cur.image_url; t.classList.remove("flash");
  void t.offsetWidth; t.classList.add("flash");
  const keys = document.getElementById("keys");
  keys.innerHTML = "";
  for (let i = 1; i <= cur.max_level; i++) {
    const b = document.createElement("button");
    b.textContent = i;
    if (cur.existing_score === i) b.classList.add("chosen");
    b.onclick = () => rate(i);
    keys.appendChild(b);
  }
  document.getElementById("backBtn").disabled = cur.pos <= 0;
}

let busy = false;
async function rate(score) {
  if (!cur || cur.done || busy) return;
  busy = true;
  await fetch("/api/rate", { method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({rater: rater, subset_id: cur.subset_id, score: score}) });
  busy = false;
  load(cur.pos + 1);  // advance after scoring
}

function goBack() { if (cur && !cur.done && cur.pos > 0) load(cur.pos - 1); }
function goFwd()  { if (cur && !cur.done) load(cur.pos + 1); }  // move on without changing
document.getElementById("backBtn").onclick = goBack;
document.getElementById("fwdBtn").onclick = goFwd;

document.addEventListener("keydown", (e) => {
  if (document.getElementById("login").classList.contains("hidden")) {
    if (e.key === "ArrowLeft") { goBack(); return; }
    if (e.key === "ArrowRight") { goFwd(); return; }
    const n = parseInt(e.key, 10);
    if (cur && !cur.done && n >= 1 && n <= cur.max_level) rate(n);
  } else if (e.key === "Enter") { start(); }
});
</script>
</body></html>"""


@app.get("/")
def index():
    return PAGE


def lan_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True, help="Stage 2 sample manifest CSV")
    ap.add_argument("--images-root", type=Path, required=True,
                    help="Directory manifest image paths are relative to (the repo root)")
    ap.add_argument("--db", type=Path, required=True, help="SQLite DB for subset + ratings")
    ap.add_argument("--per-type", type=int, default=18,
                    help="Images per room type in the rating subset (default 18; frozen on first run)")
    ap.add_argument("--all-images", action="store_true",
                    help="Rate EVERY image in the manifest (no subset); overrides --per-type")
    ap.add_argument("--seed", type=int, default=42, help="Subset draw seed (default 42)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=5000)
    args = ap.parse_args()

    CFG["db"] = args.db
    CFG["images_root"] = args.images_root.resolve()
    args.db.parent.mkdir(parents=True, exist_ok=True)
    init_db(args.manifest, None if args.all_images else args.per_type, args.seed)

    print(f"\nRating app running — share with raters:  http://{lan_ip()}:{args.port}\n")
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
