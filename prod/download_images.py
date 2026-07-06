#!/usr/bin/env python3
"""
Download every image in photo_links.csv, one folder per listing.

For each row:
    images/<zpid>/
        <zpid>_hero.<ext>   <- the url matching main_image-src
        <zpid>_1.<ext>, <zpid>_2.<ext>, ...  <- every other url in all_images

Only all_images is downloaded; main_image-src already lives inside it and is
just flagged as the hero.

Usage:
    pip install pandas requests
    python download_images.py                       # uses ./photo_links.csv
    python download_images.py --csv photo_links.csv --images-dir images --workers 12
"""

import argparse
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

EXT_BY_CT = {
    "image/jpeg": "jpg", "image/jpg": "jpg", "image/png": "png",
    "image/gif": "gif", "image/webp": "webp",
}


def parse_urls(cell):
    return [u for u in re.split(r"[\s,|]+", cell.strip()) if u.startswith("http")]


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


def build_tasks(df, images_dir):
    tasks = []
    for _, row in df.iterrows():
        zpid = row["zpid"].strip()
        hero = row["main_image-src"].strip()
        folder = os.path.join(images_dir, zpid)
        n = 0
        for url in parse_urls(row["all_images"]):
            if url == hero:
                stem = os.path.join(folder, f"{zpid}_hero")
            else:
                n += 1
                stem = os.path.join(folder, f"{zpid}_{n}")
            tasks.append({"url": url, "stem": stem})
    return tasks


def download_one(session, task):
    for ext in set(EXT_BY_CT.values()):          # resumable: skip if already on disk
        p = task["stem"] + "." + ext
        if os.path.exists(p) and os.path.getsize(p) > 0:
            return "exists"
    os.makedirs(os.path.dirname(task["stem"]), exist_ok=True)
    try:
        time.sleep(0.05)
        r = session.get(task["url"], timeout=30)
        r.raise_for_status()
        if len(r.content) < 1024:                # guard against empty/error bodies
            return "too_small"
        ext = EXT_BY_CT.get(r.headers.get("Content-Type", "").split(";")[0].strip(), "jpg")
        tmp = task["stem"] + ".part"
        with open(tmp, "wb") as f:
            f.write(r.content)
        os.replace(tmp, task["stem"] + "." + ext)   # atomic: no half-written files
        return "ok"
    except Exception as e:  # noqa: BLE001
        return f"error:{type(e).__name__}"


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/photo_links.csv")
    ap.add_argument("--images-dir", default="images")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args(argv[1:])

    df = pd.read_csv(args.csv, dtype=str, keep_default_na=False)
    tasks = build_tasks(df, args.images_dir)
    print(f"{len(tasks)} images across {len(df)} listings")

    session = make_session()
    counts = {}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(download_one, session, t) for t in tasks]
        for i, fut in enumerate(as_completed(futs), 1):
            counts[fut.result()] = counts.get(fut.result(), 0) + 1
            if i % 200 == 0:
                print(f"  {i}/{len(tasks)}")

    for status, c in sorted(counts.items()):
        print(f"  {status}: {c}")


if __name__ == "__main__":
    main(sys.argv)
