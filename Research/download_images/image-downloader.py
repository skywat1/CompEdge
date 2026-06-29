#!/usr/bin/env python3
"""
Download Zillow listing photos from the cleaned CompEdge file and link every
image back to its property via zpid.

INPUT
    A one-row-per-listing CSV or Parquet with at least:
        listing-link-href   (contains /<zpid>_zpid/)
        images              (newline / pipe / whitespace-separated photo URLs,
                             or a JSON array of URLs or {"...":url} objects)

OUTPUT
    images/<zpid>/<zpid>_<idx>.<ext>      raw pixels  -> TRANSIENT, delete after scoring
    image_manifest.parquet               the join table -> KEEP THIS

The manifest is the source of truth for image <-> property association and for
resumability. Downstream (CNN room label, LLM luxury score) you append columns
to it and aggregate with groupby("zpid", "room_label").mean().

Usage:
    pip install pandas pyarrow requests
    python download_listing_images.py cleaned_listings.parquet \
        --out-dir images --manifest image_manifest.parquet --workers 8
"""

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ZPID_RE = re.compile(r"/(\d+)_zpid")
# Zillow CDN photos: .../fp/<hash>-cc_ft_<size>.jpg  (also -uncropped_scaled_within_WxH)
HASH_RE = re.compile(r"/([a-f0-9]{16,})[-_]")
SIZE_RE = re.compile(r"(?:cc_ft_|within_\d+_)(\d+)")
EXT_BY_CT = {
    "image/jpeg": "jpg", "image/jpg": "jpg", "image/png": "png",
    "image/gif": "gif", "image/webp": "webp",
}


def parse_zpid(href: str):
    if not isinstance(href, str):
        return None
    m = ZPID_RE.search(href)
    return m.group(1) if m else None


def parse_image_urls(cell):
    """Tolerant: handles newline/pipe/whitespace lists and JSON arrays."""
    if not isinstance(cell, str) or not cell.strip():
        return []
    s = cell.strip()
    urls = []
    if s.startswith("["):                       # JSON array form
        try:
            for item in json.loads(s):
                if isinstance(item, str):
                    urls.append(item)
                elif isinstance(item, dict):
                    urls.extend(v for v in item.values() if isinstance(v, str))
        except json.JSONDecodeError:
            pass
    if not urls:                                # plain text list
        urls = re.split(r"[\s|,]+", s)
    return [u for u in (u.strip() for u in urls) if u.startswith("http")]


def photo_key(url: str):
    """Identity of the underlying photo, independent of resolution."""
    m = HASH_RE.search(url)
    return m.group(1) if m else url.rsplit("/", 1)[-1]


def photo_size(url: str):
    m = SIZE_RE.search(url)
    return int(m.group(1)) if m else 0


def dedup_largest(urls):
    """Collapse multiple resolutions of the same photo, keep the biggest."""
    best = {}
    for u in urls:
        k = photo_key(u)
        if k not in best or photo_size(u) > photo_size(best[k]):
            best[k] = u
    return list(best.values())


def make_session():
    s = requests.Session()
    retry = Retry(total=4, backoff_factor=0.6,
                  status_forcelist=(429, 500, 502, 503, 504),
                  allowed_methods=("GET",))
    s.mount("https://", HTTPAdapter(max_retries=retry, pool_maxsize=32))
    s.headers["User-Agent"] = (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
    return s


def build_tasks(df, out_dir):
    """One task per (deduped) image. Skips files already on disk (resumable)."""
    tasks, skipped = [], 0
    for _, row in df.iterrows():
        zpid = parse_zpid(row.get("listing-link-href"))
        if not zpid:
            continue
        urls = dedup_largest(parse_image_urls(row.get("images")))
        for idx, url in enumerate(urls):
            stem = os.path.join(out_dir, zpid, f"{zpid}_{idx}")
            tasks.append({"zpid": zpid, "img_index": idx,
                          "photo_hash": photo_key(url), "url": url, "stem": stem})
    return tasks, skipped


def download_one(session, task):
    # resumable: if any extension of this stem already exists, skip the fetch
    for ext in EXT_BY_CT.values():
        p = task["stem"] + "." + ext
        if os.path.exists(p) and os.path.getsize(p) > 0:
            return {**task, "path": p, "status": "exists", "bytes": os.path.getsize(p)}
    os.makedirs(os.path.dirname(task["stem"]), exist_ok=True)
    try:
        time.sleep(0.05)  # politeness jitter
        r = session.get(task["url"], timeout=30)
        r.raise_for_status()
        ext = EXT_BY_CT.get(r.headers.get("Content-Type", "").split(";")[0].strip())
        if not ext:
            return {**task, "path": None, "status": "bad_content_type", "bytes": 0}
        path = task["stem"] + "." + ext
        with open(path, "wb") as f:
            f.write(r.content)
        return {**task, "path": path, "status": "ok", "bytes": len(r.content)}
    except Exception as e:                       # noqa: BLE001
        return {**task, "path": None, "status": f"error:{type(e).__name__}", "bytes": 0}


def load_table(path):
    if path.lower().endswith((".parquet", ".pq")):
        return pd.read_parquet(path)
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="cleaned one-row-per-listing CSV or Parquet")
    ap.add_argument("--out-dir", default="images")
    ap.add_argument("--manifest", default="image_manifest.parquet")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv[1:])

    df = load_table(args.input)
    if "listing-link-href" not in df or "images" not in df:
        sys.exit("input must contain 'listing-link-href' and 'images' columns")

    tasks, _ = build_tasks(df, args.out_dir)
    print(f"{len(tasks)} images across {df['listing-link-href'].nunique()} listings")

    session = make_session()
    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(download_one, session, t) for t in tasks]
        for i, fut in enumerate(as_completed(futs), 1):
            rows.append(fut.result())
            if i % 200 == 0:
                print(f"  {i}/{len(tasks)}")

    man = pd.DataFrame(rows, columns=["zpid", "img_index", "photo_hash",
                                      "url", "path", "status", "bytes"])
    man.to_parquet(args.manifest, index=False)
    print(man["status"].value_counts().to_string())
    print(f"manifest -> {args.manifest}")


if __name__ == "__main__":
    main(sys.argv)