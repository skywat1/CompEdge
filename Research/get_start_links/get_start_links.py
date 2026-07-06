#!/usr/bin/env python3
"""Read an Excel file with a column of zip codes, emit a one-column CSV of
Zillow 'sold' search URLs (one row per input row, order preserved).

Usage:
    python zips_to_urls.py --in zips.xlsx --out urls.csv
    python zips_to_urls.py --in zips.xlsx --out urls.csv --zip-col ZIP --doz 24m

Requires: pandas, openpyxl
"""
import argparse
import json
import re
import urllib.parse
import sys

import pandas as pd


def zillow_sold_url(zip_code: str, doz: str = "12m", price_min: int = 1001) -> str:
    """Zillow Brooklyn 'sold' URL that is zip-swappable via the path slug.

    Region-locking keys (regionSelection, mapBounds, usersSearchTerm) are
    deliberately omitted so the zip in the path controls the region.
    """
    fs = {
        "price": {"min": price_min},
        "rs":   {"value": True},    # recently sold
        "fsba": {"value": False}, "fsbo": {"value": False},
        "nc":   {"value": False}, "cmsn": {"value": False},
        "auc":  {"value": False}, "fore": {"value": False},
        "mf":   {"value": False}, "land": {"value": False},
        "apa":  {"value": False}, "manu": {"value": False},
        "doz":  {"value": doz},     # sold within last N: 6m/12m/24m/36m
    }
    enc = urllib.parse.quote(json.dumps({"filterState": fs}, separators=(",", ":")), safe="")
    return f"https://www.zillow.com/brooklyn-new-york-ny-{zip_code}/sold/?searchQueryState={enc}"


def normalize_zip(raw) -> str | None:
    """Coerce a cell value to a clean 5-digit zip, or None if it isn't one."""
    if pd.isna(raw):
        return None
    s = str(raw).strip()
    s = re.sub(r"\.0$", "", s)          # kill float artifact e.g. '11201.0'
    m = re.search(r"\d{5}", s)          # first 5-digit run
    return m.group(0) if m else None


def find_zip_col(df: pd.DataFrame, explicit: str | None) -> str:
    if explicit:
        if explicit not in df.columns:
            sys.exit(f"Column '{explicit}' not found. Available: {list(df.columns)}")
        return explicit
    # auto-detect: first column whose name contains 'zip'
    for c in df.columns:
        if "zip" in str(c).lower():
            return c
    sys.exit(f"No --zip-col given and no 'zip'-like column found. Available: {list(df.columns)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="input .xlsx path")
    ap.add_argument("--out", dest="out", required=True, help="output .csv path")
    ap.add_argument("--zip-col", default=None, help="name of the zip column (auto-detected if omitted)")
    ap.add_argument("--sheet", default=0, help="sheet name or index (default: first sheet)")
    ap.add_argument("--doz", default="12m", help="sold window: 1/7/14/30/90 (days) or 6m/12m/24m/36m")
    ap.add_argument("--price-min", type=int, default=1001, help="minimum price filter")
    args = ap.parse_args()

    df = pd.read_excel(args.inp, sheet_name=args.sheet, dtype=str)
    col = find_zip_col(df, args.zip_col)

    urls = []
    skipped = 0
    for raw in df[col]:
        z = normalize_zip(raw)
        if z is None:
            skipped += 1
            urls.append("")        # keep 1:1 alignment with input rows
        else:
            urls.append(zillow_sold_url(z, doz=args.doz, price_min=args.price_min))

    out = pd.DataFrame({"zip": urls})
    out.to_csv(args.out, index=False)
    print(f"Wrote {len(out)} rows to {args.out} (column '{col}', {skipped} rows had no valid zip).")


if __name__ == "__main__":
    main()