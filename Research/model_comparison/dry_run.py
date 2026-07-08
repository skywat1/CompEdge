#!/usr/bin/env python3
"""
Dry run — validate schemas, multi-image message structure, and parsing on all
five scoring models (plus the gpt-4o classifier) using a few images from the
reference test sets in Research/llm_image_rank_test, then estimate the total
cost of the full experiment. Makes ~18 API calls total; run BEFORE Stage 1.

Usage:
    python dry_run.py            # 3 images x 5 models + 3 classifier calls
    python dry_run.py --models gpt-4o claude-haiku-4-5
"""

import argparse
import json
from pathlib import Path

from common import (ASSETS_ROOT, MODELS, CLASSIFIER_MODEL, compute_cost,
                    classify_image, score_image)

# (room_type, image) triples from the earlier experiment's labeled test images
DRY_IMAGES = [
    ("kitchen",     ASSETS_ROOT / "kitchen" / "images" / "kitchen1.webp"),
    ("bathroom",    ASSETS_ROOT / "bathroom" / "images" / "bathroom1.webp"),
    ("living_room", ASSETS_ROOT / "living" / "images" / "living1.webp"),
]

# full-experiment volumes for the cost estimate
STAGE1_CALLS = 3000
STAGE3_IMAGES = 150
STAGE3_REPS = 5


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--out-json", type=Path,
                    default=Path(__file__).parent / "outputs" / "dry_run_results.json")
    args = ap.parse_args()

    results = {"classify": [], "score": {}}

    print("=" * 72)
    print(f"CLASSIFIER ({CLASSIFIER_MODEL}, detail=low)")
    print("=" * 72)
    cls_costs = []
    for room, img in DRY_IMAGES:
        r = classify_image(img)
        cost = compute_cost(CLASSIFIER_MODEL, r["input_tokens"], r["output_tokens"],
                            r["cached_input_tokens"], r["cache_write_tokens"])
        cls_costs.append(cost)
        other = r["parsed"].get("other_label")
        print(f"  {img.name:16s} expected={room:12s} -> predicted={r['parsed']['room_type']:12s} "
              + (f"({other}) " if other else "")
              + f"| {r['input_tokens']} in / {r['output_tokens']} out tok "
              + f"| {r['latency_s']:.1f}s | ${cost:.5f} | id={r['response_model']}")
        results["classify"].append({"image": img.name, "expected": room, **{
            k: r[k] for k in ("parsed", "response_model", "input_tokens",
                              "output_tokens", "latency_s")}, "cost_usd": cost})
    cls_per_call = sum(cls_costs) / len(cls_costs)

    score_per_call = {}
    for model in args.models:
        print("\n" + "=" * 72)
        print(f"SCORING MODEL: {model}")
        print("=" * 72)
        costs = []
        results["score"][model] = []
        for room, img in DRY_IMAGES:
            try:
                r = score_image(model, room, img)
            except Exception as e:
                print(f"  {img.name:16s} FAILED: {e}")
                results["score"][model].append({"image": img.name, "error": str(e)})
                continue
            p = r["parsed"]
            cost = compute_cost(model, r["input_tokens"], r["output_tokens"],
                                r["cached_input_tokens"], r["cache_write_tokens"])
            costs.append(cost)
            print(f"  {img.name:16s} score={p['score']:<5} level={p['level']} "
                  f"conf={p['confidence']:<5} valid={p['valid']} room={p['room_type']}")
            print(f"    reasoning: {p['reasoning'][:110]}...")
            print(f"    {r['input_tokens']} in / {r['output_tokens']} out tok "
                  f"(thought {r['thought_tokens']}) | {r['latency_s']:.1f}s "
                  f"| ${cost:.5f} | id={r['response_model']}")
            results["score"][model].append({"image": img.name, **{
                k: r[k] for k in ("parsed", "response_model", "input_tokens",
                                  "output_tokens", "thought_tokens", "latency_s")},
                "cost_usd": cost})
        if costs:
            score_per_call[model] = sum(costs) / len(costs)

    print("\n" + "=" * 72)
    print("ESTIMATED FULL-EXPERIMENT COST (standard pricing)")
    print("=" * 72)
    stage1 = cls_per_call * STAGE1_CALLS
    print(f"Stage 1: {STAGE1_CALLS} classify calls x ${cls_per_call:.5f} = ${stage1:.2f} (upper bound)")
    calls = STAGE3_IMAGES * STAGE3_REPS
    total = stage1
    for model, per in score_per_call.items():
        est = per * calls
        total += est
        print(f"Stage 3: {model:18s} {calls} calls x ${per:.5f} = ${est:.2f}")
    print(f"{'TOTAL (est.)':27s} ${total:.2f}")

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\nDetailed results saved to {args.out_json}")


if __name__ == "__main__":
    main()
