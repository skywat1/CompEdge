#!/usr/bin/env python3
"""Open every link from a CSV (like the output of zips_to_urls.py) in the browser.

Usage:
    python open_links.py --in urls.csv
    python open_links.py --in urls.csv --delay 2 --limit 10
    python open_links.py --in urls.csv --start 10          # resume from row 10
    python open_links.py --in urls.csv --dry-run           # just print, open nothing

Requires: pandas
"""
import argparse
import sys
import time
import webbrowser

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="input .csv path")
    ap.add_argument("--url-col", default="url", help="name of the URL column (default: url)")
    ap.add_argument("--delay", type=float, default=1.5, help="seconds to wait between opening tabs")
    ap.add_argument("--limit", type=int, default=None, help="open at most this many links")
    ap.add_argument("--start", type=int, default=0, help="skip this many rows first (for resuming)")
    ap.add_argument("--dry-run", action="store_true", help="print links instead of opening them")
    args = ap.parse_args()

    df = pd.read_csv(args.inp, dtype=str)
    if args.url_col not in df.columns:
        sys.exit(f"Column '{args.url_col}' not found. Available: {list(df.columns)}")

    # drop blanks, apply start offset, then limit
    links = [u for u in df[args.url_col].tolist() if isinstance(u, str) and u.strip()]
    links = links[args.start:]
    if args.limit is not None:
        links = links[:args.limit]

    if not links:
        sys.exit("No links to open.")

    print(f"{'Would open' if args.dry_run else 'Opening'} {len(links)} link(s)"
          f"{'' if args.dry_run else f', {args.delay}s apart'}...")

    for i, url in enumerate(links, 1):
        if args.dry_run:
            print(f"[{i}/{len(links)}] {url}")
            continue
        webbrowser.open_new_tab(url)
        print(f"[{i}/{len(links)}] opened")
        if i < len(links):
            time.sleep(args.delay)


if __name__ == "__main__":
    main()