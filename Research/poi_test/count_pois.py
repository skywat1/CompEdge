#!/usr/bin/env python3
"""
count_brooklyn_pois.py

Pull every POI in Brooklyn from OpenStreetMap (via the Overpass API),
count them by category (key=value), and write every category sorted
from most to least common to a CSV.

No third-party deps -- uses only the standard library.
"""

import sys
import csv
import json
import urllib.parse
import urllib.request
from collections import Counter

# ---- config ---------------------------------------------------------------
# OSM "primary feature" keys that denote a POI. Trim or extend this list
# to control what counts as a POI for your purposes.
POI_KEYS = [
    "amenity", "shop", "leisure", "tourism", "office",
    "craft", "healthcare", "historic", "emergency", "man_made"
]

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
# ---------------------------------------------------------------------------


def build_query():
    """Resolve Brooklyn's admin boundary -> area, then grab every node/way/
    relation (nwr) carrying any POI key inside it. `out tags;` returns tags
    only (no geometry), which keeps the payload small."""
    blocks = "\n".join(f'  nwr(area.bk)["{key}"];' for key in POI_KEYS)
    return f"""
[out:json][timeout:600];
area["name"="Brooklyn"]["boundary"="administrative"]->.bk;
(
{blocks}
);
out tags;
"""


def fetch():
    payload = urllib.parse.urlencode({"data": build_query()}).encode()
    headers = {
        "User-Agent": "brooklyn-poi-counter/1.0 (research script)",
        "Content-Type": "application/x-www-form-urlencoded",
    }
    req = urllib.request.Request(OVERPASS_URL, data=payload, headers=headers)
    with urllib.request.urlopen(req, timeout=900) as resp:
        return json.loads(resp.read().decode())


def main():
    print("Querying Overpass (this can take 30-90s)...", file=sys.stderr)
    elements = fetch().get("elements", [])
    print(f"Got {len(elements):,} elements.", file=sys.stderr)

    counts = Counter()
    for el in elements:
        tags = el.get("tags", {})
        for key in POI_KEYS:
            if key in tags:
                counts[f"{key}={tags[key]}"] += 1

    # write every category, sorted most -> least
    out_path = "brooklyn_poi_counts.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["category", "count"])
        w.writerows(counts.most_common())

    total = sum(counts.values())
    print(f"Wrote {len(counts):,} categories ({total:,} total POIs) -> {out_path}",
          file=sys.stderr)


if __name__ == "__main__":
    main()