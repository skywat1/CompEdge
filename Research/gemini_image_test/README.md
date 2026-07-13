# gemini_image_test

Iterative prompt-tuning harness for luxury scoring with **gemini-3.5-flash**
(one call per image). Edit a room's prompt, rescore just that room, and see a
new tab appear in a single running gallery — each tab compares gemini's fresh
scores against the human ("user") rankings and shows that run's cost.

## Layout

```
rooms/<room>/prompt.txt   # editable prompt for each room (kitchen/bathroom/bedroom/living_room)
rooms/<room>/grid.png     # the anchor calibration grid (copied from llm_image_rank_test)
data/sample_manifest.csv  # the 152-image sample (38 per room) that gets scored
data/human_ratings.csv    # the user rankings (harvey, robin, seb) used as the reference
runs/<timestamp>/scores.csv   # gemini scores + per-call cost for each run
gallery/index.html        # tabbed gallery — open this in a browser
gallery/pages/            # one HTML page per run (one tab each)
```

Scoring reuses `Research/model_comparison/common.py` (same message layout,
schema, retry, and cost model). Keys come from `config.py` (`GEMINI_API_KEY`).

## Tuning loop

1. Edit e.g. `rooms/living_room/prompt.txt` (the grid sits beside it).
2. Rescore just that room:
   ```bash
   ../../venv/bin/python rescore.py living_room
   ```
   Run with no arguments for an interactive room picker (select many):
   ```bash
   ../../venv/bin/python rescore.py
   #   1) kitchen       (38 images)
   #   2) bathroom      (38 images)
   #   3) bedroom       (38 images)
   #   4) living_room   (38 images)
   #   > 1,4            (or "all", or names: "kitchen living")
   ```
3. Open `gallery/index.html`. A new tab shows the run: each image with the
   per-rater scores, human mean, gemini score, signed gap, gemini's reasoning,
   and the Zillow link — plus a banner with the **run cost** and each room's
   **prompt at the bottom of the page**. Newest tab opens by default.

Each card page has an *All images* / *Only off by >1* toggle; the room header
count updates to the number of visible cards.

## Comparing two runs

The first tab, **⇄ Compare runs**, diffs any two runs entirely in the browser.
Pick *Run A* and *Run B* from the dropdowns (they default to the two most
recent runs) and it shows only the images whose **gemini score changed at all
(even by 1)** between them — each card lists the human mean, the score from
both runs, and the signed change (B−A), with both runs' reasoning behind a
toggle. The comparison data refreshes every time you rescore.

## Flags

```
python rescore.py living_room kitchen   # rooms by name/prefix
python rescore.py --all                 # all four rooms
python rescore.py living_room --limit 3 # first 3 images only (cheap smoke test)
python rescore.py living_room --dry-run # no API calls, fabricated scores
python rescore.py living_room --workers 8 --label "tighter historic rule"
```

A full 152-image run costs roughly **$1.7** (~$0.011/image); one room ~$0.44.
