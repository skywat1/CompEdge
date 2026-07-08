#!/usr/bin/env python3
"""
Stage 5 — analysis + markdown report.

Per model and per room type:
  * within-image std of the 5 replicates vs between-image std of mean scores
    (ratio = headline consistency metric; lower is better)
  * score-distribution histograms (per model per room type), with a
    scale-compression flag for models that rarely use the extremes
  * Spearman rank correlation of per-image mean scores vs gpt-4o
  * disagreement analysis vs gpt-4o (|mean difference| >= 2 levels, with paths)
  * room-type disagreement rate (scoring-call judgment vs Stage 1 label)
  * measured cost per call + extrapolated full-run cost (standard and batch)
  * a short recommendation

Usage:
    python stage5_report.py --parquet-dir outputs/parquet \
        --images-dir ../../images --out-report outputs/report.md \
        --plots-dir outputs/plots
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import BATCH_DISCOUNT, MODELS, ROOM_TYPES, VALID_EXTS

REFERENCE = "gpt-4o"

# dataviz-skill palette: single validated hue on a light surface
BAR = "#2a78d6"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e6e5e2"

# One stable colour per model for the multi-series figures.
MODEL_COLORS = {
    "gpt-4o": "#4c78a8",
    "gemini-3.5-flash": "#54a24b",
    "gemini-2.5-flash": "#e45756",
    "claude-haiku-4-5": "#f58518",
    "claude-sonnet-4-6": "#72b7b2",
}
ROOM_COLORS = {
    "kitchen": "#4c78a8", "bathroom": "#54a24b",
    "bedroom": "#e45756", "living_room": "#f58518",
}


def _style_ax(ax):
    ax.set_facecolor(SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.set_axisbelow(True)


def consistency_table(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, room), g in scores.groupby(["model", "room_type"]):
        per_img = g.groupby("image_path")["score"]
        within = per_img.std(ddof=1).mean()
        between = per_img.mean().std(ddof=1)
        rows.append({
            "model": model, "room_type": room,
            "n_images": per_img.ngroups,
            "within_image_std": within,
            "between_image_std": between,
            "ratio_within_over_between": within / between if between else np.nan,
        })
    return pd.DataFrame(rows)


def compression_flags(scores: pd.DataFrame, max_levels: dict) -> pd.DataFrame:
    """Share of scores in the top/bottom 1-point bands of each room's scale."""
    rows = []
    for (model, room), g in scores.groupby(["model", "room_type"]):
        lo, hi = 1.0, float(max_levels[room])
        s = g["score"].dropna()
        ext = ((s <= lo + 1) | (s >= hi - 1)).mean()
        used = s.max() - s.min()
        rows.append({"model": model, "room_type": room,
                     "share_in_extremes": ext,
                     "range_used": used, "scale_span": hi - lo,
                     "compressed": bool(ext < 0.05 or used < 0.6 * (hi - lo))})
    return pd.DataFrame(rows)


def plot_histograms(scores: pd.DataFrame, max_levels: dict, plots_dir: Path) -> Path:
    models = [m for m in MODELS if m in scores["model"].unique()]
    fig, axes = plt.subplots(len(models), len(ROOM_TYPES),
                             figsize=(4 * len(ROOM_TYPES), 2.4 * len(models)),
                             squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for i, model in enumerate(models):
        for j, room in enumerate(ROOM_TYPES):
            ax = axes[i][j]
            ax.set_facecolor(SURFACE)
            s = scores[(scores["model"] == model) & (scores["room_type"] == room)]["score"].dropna()
            hi = max_levels[room]
            bins = np.arange(1.0, hi + 0.5, 0.5)
            ax.hist(s, bins=bins, color=BAR, edgecolor=SURFACE, linewidth=1)
            ax.set_xlim(0.8, hi + 0.2)
            ax.grid(axis="y", color=GRID, linewidth=0.8)
            ax.set_axisbelow(True)
            for spine in ("top", "right", "left"):
                ax.spines[spine].set_visible(False)
            ax.spines["bottom"].set_color(GRID)
            ax.tick_params(colors=INK2, labelsize=8)
            if i == 0:
                ax.set_title(room, color=INK, fontsize=10)
            if j == 0:
                ax.set_ylabel(model, color=INK, fontsize=9)
    fig.suptitle("Score distributions (all replicates)", color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    plots_dir.mkdir(parents=True, exist_ok=True)
    out = plots_dir / "score_histograms.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out


def _save(fig, plots_dir: Path, name: str) -> Path:
    plots_dir.mkdir(parents=True, exist_ok=True)
    out = plots_dir / name
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out


def plot_consistency(cons: pd.DataFrame, plots_dir: Path) -> Path:
    """Horizontal bars of mean within/between ratio, steadiest model on top."""
    rank = cons.groupby("model")["ratio_within_over_between"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    fig.patch.set_facecolor(SURFACE)
    _style_ax(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.barh(range(len(rank)), rank.values,
            color=[MODEL_COLORS.get(m, BAR) for m in rank.index], edgecolor=SURFACE)
    ax.set_yticks(range(len(rank)))
    ax.set_yticklabels(rank.index, color=INK)
    for i, v in enumerate(rank.values):
        ax.text(v + max(rank.values) * 0.01, i, f"{v:.3f}",
                va="center", color=INK2, fontsize=9)
    ax.set_xlim(0, max(rank.values) * 1.15)
    ax.set_xlabel("mean within/between std ratio  (lower = steadier)", color=INK2, fontsize=9)
    ax.set_title("Consistency: replicate noise vs real between-image signal",
                 color=INK, fontsize=12)
    return _save(fig, plots_dir, "consistency_ratio.png")


def plot_human_agreement(mvr: pd.DataFrame, ceiling: float, plots_dir: Path) -> Path:
    """Grouped bars: model agreement (Spearman) with the two consensus references,
    with the human-agreement ceiling drawn as a reference line."""
    refs = [r for r in ("all", "harvey_robin") if r in set(mvr["reference"])]
    order = (mvr[mvr["reference"] == refs[0]]
             .sort_values("spearman", ascending=False)["model"].tolist())
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    fig.patch.set_facecolor(SURFACE)
    _style_ax(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    n = len(refs)
    width = 0.8 / n
    shades = {refs[0]: 1.0}
    for k, r in enumerate(refs[1:], 1):
        shades[r] = 0.55  # lighter bar for the secondary reference
    for k, r in enumerate(refs):
        vals = [mvr[(mvr["model"] == m) & (mvr["reference"] == r)]["spearman"].iloc[0]
                for m in order]
        xs = [i + (k - (n - 1) / 2) * width for i in range(len(order))]
        colors = [MODEL_COLORS.get(m, BAR) for m in order]
        ax.bar(xs, vals, width=width * 0.95, color=colors,
               alpha=shades[r], edgecolor=SURFACE,
               label=f"vs {r}" + (" (avg of all 3)" if r == "all" else ""))
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.008, f"{v:.2f}", ha="center", color=INK2, fontsize=7.5)
    ax.axhline(ceiling, ls="--", color=INK, lw=1.3)
    ax.text(len(order) - 0.5, ceiling + 0.008, f"human ceiling {ceiling:.3f}",
            ha="right", color=INK, fontsize=8.5)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, color=INK, rotation=15, ha="right")
    ax.set_ylabel("Spearman rho vs human reference  (higher = better)",
                  color=INK2, fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.set_title("Model agreement with human consensus", color=INK, fontsize=12)
    ax.legend(frameon=False, fontsize=8, loc="upper right",
              labelcolor=INK2, ncol=n)
    return _save(fig, plots_dir, "human_agreement.png")


def _heatmap(ax, mat: pd.DataFrame, cmap, vmin, vmax, fmt_txt):
    im = ax.imshow(mat.values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(mat.shape[1]))
    ax.set_xticklabels(mat.columns, rotation=25, ha="right", color=INK, fontsize=9)
    ax.set_yticks(range(mat.shape[0]))
    ax.set_yticklabels(mat.index, color=INK, fontsize=9)
    rng = (vmax - vmin) or 1
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.values[i, j]
            if pd.isna(v):
                continue
            frac = (v - vmin) / rng
            txt = INK if 0.25 < frac < 0.75 else (SURFACE if frac >= 0.75 else INK)
            ax.text(j, i, fmt_txt(v), ha="center", va="center",
                    color=txt, fontsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    return im


def plot_agreement_heatmap(mvr: pd.DataFrame, model_order, ref_order, plots_dir: Path) -> Path:
    """Full model x human-reference Spearman grid, including individual raters."""
    piv = (mvr.pivot(index="model", columns="reference", values="spearman")
           .reindex(index=[m for m in model_order if m in mvr["model"].unique()],
                    columns=[r for r in ref_order if r in mvr["reference"].unique()]))
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    fig.patch.set_facecolor(SURFACE)
    im = _heatmap(ax, piv, "YlGnBu", float(np.nanmin(piv.values)),
                  float(np.nanmax(piv.values)), lambda v: f"{v:.2f}")
    ax.set_title("Spearman: every model vs every human reference", color=INK, fontsize=12)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=8, colors=INK2)
    return _save(fig, plots_dir, "agreement_heatmap.png")


def plot_cost_quality(costs: pd.DataFrame, mvr: pd.DataFrame, plots_dir: Path) -> Path:
    """The decision chart: full-run cost (x) vs human agreement (y). Up-and-left
    is better. Marker size grows with steadiness (1/consistency ratio)."""
    sp = mvr[mvr["reference"] == "all"].set_index("model")["spearman"]
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    fig.patch.set_facecolor(SURFACE)
    _style_ax(ax)
    ax.grid(color=GRID, linewidth=0.8)
    for _, row in costs.iterrows():
        m = row["model"]
        if m not in sp.index:
            continue
        x, y = row["full_run_batch_usd"], sp[m]
        ax.scatter(x, y, s=320, color=MODEL_COLORS.get(m, BAR),
                   edgecolor=SURFACE, linewidth=1.5, zorder=3)
        ax.annotate(m, (x, y), xytext=(8, 6), textcoords="offset points",
                    color=INK, fontsize=9)
    ax.set_xlabel("estimated full-run cost, batch pricing (USD)  →  cheaper is left",
                  color=INK2, fontsize=9)
    ax.set_ylabel("Spearman vs human consensus  →  better is up", color=INK2, fontsize=9)
    ax.set_title("Cost vs quality — pick from the upper-left", color=INK, fontsize=12)
    return _save(fig, plots_dir, "cost_vs_quality.png")


def plot_model_human_scatter(means: pd.DataFrame, ratings: pd.DataFrame,
                             focus_models, plots_dir: Path) -> Path:
    """Per-image model mean vs human mean, with y=x line, for a couple of models.
    Points above the line = model scores higher than humans (positive bias)."""
    human = ratings.groupby("image_path")["score"].mean().rename("human")
    focus = [m for m in focus_models if m in means["model"].unique()]
    fig, axes = plt.subplots(1, len(focus), figsize=(4.6 * len(focus), 4.4), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax, m in zip(axes[0], focus):
        g = means[means["model"] == m].set_index("image_path")
        j = g.join(human, how="inner").dropna(subset=["mean_score", "human"])
        _style_ax(ax)
        ax.grid(color=GRID, linewidth=0.8)
        for room in ROOM_TYPES:
            r = j[j["room_type"] == room]
            ax.scatter(r["human"], r["mean_score"], s=26, alpha=0.8,
                       color=ROOM_COLORS.get(room, BAR), edgecolor="none", label=room)
        lo = min(j["human"].min(), j["mean_score"].min()) - 0.3
        hi = max(j["human"].max(), j["mean_score"].max()) + 0.3
        ax.plot([lo, hi], [lo, hi], ls="--", color=INK2, lw=1)
        rho = spearmanr(j["human"], j["mean_score"])[0]
        bias = (j["mean_score"] - j["human"]).mean()
        ax.set_title(f"{m}\nrho={rho:.2f}, bias=+{bias:.2f}", color=INK, fontsize=10)
        ax.set_xlabel("human mean score", color=INK2, fontsize=9)
        ax.set_ylabel("model mean score", color=INK2, fontsize=9)
    axes[0][0].legend(frameon=False, fontsize=7.5, labelcolor=INK2, loc="upper left")
    fig.suptitle("Model vs human, per image (dashed = perfect agreement)",
                 color=INK, fontsize=12)
    return _save(fig, plots_dir, "model_vs_human_scatter.png")


def plot_interrater(mat: pd.DataFrame, plots_dir: Path) -> Path:
    """3x3 human-vs-human agreement heatmap."""
    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    fig.patch.set_facecolor(SURFACE)
    im = _heatmap(ax, mat.astype(float), "YlGnBu", 0.5, 1.0, lambda v: f"{v:.2f}")
    ax.set_title("Human vs human agreement (Spearman)", color=INK, fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=8, colors=INK2)
    return _save(fig, plots_dir, "interrater_heatmap.png")


def mean_scores(scores: pd.DataFrame) -> pd.DataFrame:
    return (scores.groupby(["model", "image_path", "zpid", "room_type"])["score"]
            .mean().reset_index(name="mean_score"))


def spearman_vs_reference(means: pd.DataFrame) -> pd.DataFrame:
    ref = means[means["model"] == REFERENCE].set_index("image_path")["mean_score"]
    rows = []
    for model in means["model"].unique():
        if model == REFERENCE:
            continue
        m = means[means["model"] == model].set_index("image_path")
        joined = m.join(ref, rsuffix="_ref", how="inner")
        rho_all, _ = spearmanr(joined["mean_score"], joined["mean_score_ref"])
        row = {"model": model, "overall": rho_all}
        for room in ROOM_TYPES:
            jr = joined[joined["room_type"] == room]
            row[room] = spearmanr(jr["mean_score"], jr["mean_score_ref"])[0] if len(jr) > 2 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def disagreements_vs_reference(means: pd.DataFrame, threshold=2.0) -> pd.DataFrame:
    ref = means[means["model"] == REFERENCE].set_index("image_path")["mean_score"]
    rows = []
    for model in means["model"].unique():
        if model == REFERENCE:
            continue
        m = means[means["model"] == model].set_index("image_path")
        joined = m.join(ref, rsuffix="_ref", how="inner")
        diff = joined["mean_score"] - joined["mean_score_ref"]
        for path, d in diff[diff.abs() >= threshold].items():
            rows.append({"model": model, "image_path": path,
                         "room_type": joined.loc[path, "room_type"],
                         "model_mean": round(joined.loc[path, "mean_score"], 2),
                         "gpt4o_mean": round(joined.loc[path, "mean_score_ref"], 2),
                         "diff": round(d, 2)})
    return pd.DataFrame(rows)


def room_disagreement(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, room), g in scores.groupby(["model", "room_type"]):
        rate = (g["room_type_judgment"] != g["room_type"]).mean()
        rows.append({"model": model, "room_type": room, "disagreement_rate": rate})
    return pd.DataFrame(rows)


def build_human_references(ratings: pd.DataFrame, groups: dict):
    """Ordered dict {ref_label: mean-score Series by image_path}.

    Named group averages come first (each averaged over the group's raters that
    are actually present), then every individual rater on their own. A group's
    per-image value is the mean over whichever of its raters scored that image.
    Returns (refs, group_members) where group_members maps the group labels to
    the present raters that make them up.
    """
    present = set(ratings["rater"].unique())
    refs, group_members = {}, {}
    for label, members in groups.items():
        pool = sorted(present) if members is None else members
        have = [r for r in pool if r in present]
        if not have:
            continue
        sub = ratings[ratings["rater"].isin(have)]
        refs[label] = sub.groupby("image_path")["score"].mean()
        group_members[label] = have
    for r in sorted(present):
        refs[r] = ratings[ratings["rater"] == r].groupby("image_path")["score"].mean()
    return refs, group_members


def model_vs_references(means: pd.DataFrame, refs: dict) -> pd.DataFrame:
    """Long table: per (model, reference) Spearman, MAD, signed bias, n images."""
    rows = []
    for model, g in means.groupby("model"):
        m = g.set_index("image_path")["mean_score"]
        for ref_label, human in refs.items():
            joined = pd.concat([m, human], axis=1, join="inner",
                               keys=["model", "human"]).dropna()
            if len(joined) < 3:
                continue
            diff = joined["model"] - joined["human"]
            rows.append({
                "model": model, "reference": ref_label, "n": len(joined),
                "spearman": spearmanr(joined["model"], joined["human"])[0],
                "mean_abs_dev": diff.abs().mean(),
                "bias": diff.mean(),
            })
    return pd.DataFrame(rows)


def ref_matrix(mvr: pd.DataFrame, value: str, model_order, ref_order) -> pd.DataFrame:
    """Pivot the long model-vs-reference table into models x references."""
    piv = mvr.pivot(index="model", columns="reference", values=value)
    piv = piv.reindex(index=[m for m in model_order if m in piv.index],
                      columns=[r for r in ref_order if r in piv.columns])
    return piv.reset_index()


def inter_rater(ratings: pd.DataFrame, min_common: int = 10):
    """Pairwise Spearman matrix, per-rater mean vs others, ceiling, outlier flags."""
    pivot = ratings.pivot_table(index="image_path", columns="rater", values="score")
    raters = list(pivot.columns)
    mat = pd.DataFrame(np.nan, index=raters, columns=raters)
    pair_rhos = []
    for i, a in enumerate(raters):
        mat.loc[a, a] = 1.0
        for b in raters[i + 1:]:
            common = pivot[[a, b]].dropna()
            if len(common) >= min_common:
                rho = spearmanr(common[a], common[b])[0]
                mat.loc[a, b] = mat.loc[b, a] = rho
                pair_rhos.append(rho)
    ceiling = float(np.mean(pair_rhos)) if pair_rhos else np.nan

    per_rater = []
    for a in raters:
        others = [mat.loc[a, b] for b in raters if b != a and pd.notna(mat.loc[a, b])]
        per_rater.append({"rater": a,
                          "n_rated": int(pivot[a].notna().sum()),
                          "mean_rho_vs_others": np.mean(others) if others else np.nan})
    per_rater = pd.DataFrame(per_rater)
    grand = per_rater["mean_rho_vs_others"].mean()
    sd = per_rater["mean_rho_vs_others"].std(ddof=1)
    per_rater["outlier"] = per_rater["mean_rho_vs_others"].apply(
        lambda r: bool(pd.notna(r) and ((grand - r) > 0.2 or (sd and (grand - r) > 2 * sd))))
    return mat, per_rater, ceiling


def cost_table(scores: pd.DataFrame, cls: pd.DataFrame, images_dir: Path) -> pd.DataFrame:
    total_images = sum(1 for d in images_dir.iterdir() if d.is_dir()
                       for p in d.iterdir() if p.suffix.lower() in VALID_EXTS)
    scoreable_frac = (cls["predicted_label"].isin(ROOM_TYPES)).mean()
    est_scoreable = int(total_images * scoreable_frac)
    rows = []
    for model, g in scores.groupby("model"):
        per_call = g["cost_usd"].mean()
        full = per_call * est_scoreable
        rows.append({"model": model,
                     "mean_cost_per_call_usd": per_call,
                     "mean_latency_s": g["latency_s"].mean(),
                     "est_scoreable_images": est_scoreable,
                     "full_run_standard_usd": full,
                     "full_run_batch_usd": full * BATCH_DISCOUNT})
    return pd.DataFrame(rows)


def fmt(df: pd.DataFrame, floats=3) -> str:
    return df.round(floats).to_markdown(index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parquet-dir", type=Path, required=True)
    ap.add_argument("--images-dir", type=Path, required=True,
                    help="Full image corpus dir, used to extrapolate full-run cost")
    ap.add_argument("--out-report", type=Path, required=True)
    ap.add_argument("--plots-dir", type=Path, required=True)
    ap.add_argument("--ratings-parquet", type=Path, default=None,
                    help="Optional Stage 2b human ratings parquet (from export_ratings.py); "
                         "enables the human-agreement sections")
    ap.add_argument("--rater-group", action="append", metavar="LABEL=r1,r2,...",
                    help="Named human reference averaged over the listed raters "
                         "(repeatable). An 'all' group over every rater is always added, "
                         "and each rater is also compared individually. "
                         "Default: harvey_robin=harvey,robin")
    args = ap.parse_args()

    # Human reference groups (averaged per image). "all" (every present rater) is
    # always included; each individual rater is also scored on its own downstream.
    group_specs = args.rater_group if args.rater_group else ["harvey_robin=harvey,robin"]
    groups = {"all": None}  # None -> filled with every present rater below
    for spec in group_specs:
        if "=" not in spec:
            ap.error(f"--rater-group must be LABEL=r1,r2,...; got {spec!r}")
        label, members = spec.split("=", 1)
        groups[label.strip()] = [r.strip() for r in members.split(",") if r.strip()]

    scores = pd.read_parquet(args.parquet_dir / "scores.parquet")
    cls = pd.read_parquet(args.parquet_dir / "classifications.parquet")
    scores = scores[scores["error"].fillna("").astype(str).str.len() == 0].copy()
    scores["score"] = pd.to_numeric(scores["score"], errors="coerce")
    scores["cost_usd"] = pd.to_numeric(scores["cost_usd"], errors="coerce")

    from common import ROOMS
    max_levels = {r: ROOMS[r]["max_level"] for r in ROOM_TYPES}

    cons = consistency_table(scores)
    comp = compression_flags(scores, max_levels)
    hist_path = plot_histograms(scores, max_levels, args.plots_dir)
    cons_path = plot_consistency(cons, args.plots_dir)
    means = mean_scores(scores)
    rho = spearman_vs_reference(means)
    dis = disagreements_vs_reference(means)
    room_dis = room_disagreement(scores)
    costs = cost_table(scores, cls, args.images_dir)

    ratings = None
    if args.ratings_parquet and args.ratings_parquet.exists():
        ratings = pd.read_parquet(args.ratings_parquet)
        if ratings.empty or ratings["rater"].nunique() < 2:
            print("Ratings file has <2 raters — skipping human-agreement sections.")
            ratings = None

    # ranking for the recommendation: mean consistency ratio across rooms
    rank = cons.groupby("model")["ratio_within_over_between"].mean().sort_values()
    best = rank.index[0]
    model_ids = scores.groupby("model")["response_model"].agg(
        lambda s: ", ".join(sorted(set(s.dropna()))))

    def rel(p: Path):
        return p.relative_to(args.out_report.parent) \
            if p.is_relative_to(args.out_report.parent) else p
    rel_hist = rel(hist_path)

    lines = ["# Luxury-scoring model comparison", ""]
    lines += ["## Exact model IDs used", "", model_ids.reset_index()
              .rename(columns={"response_model": "response model id(s)"})
              .to_markdown(index=False), ""]
    lines += ["## Consistency (within-image std vs between-image std)",
              "", "Ratio < 1 means replicate noise is smaller than real between-image",
              "signal; lower is better.", "", f"![consistency ratio]({rel(cons_path)})", "",
              fmt(cons), ""]
    lines += ["## Score distributions", "", f"![score histograms]({rel_hist})", "",
              "Scale-compression flags (share of scores within 1 point of either",
              "end of the scale < 5%, or < 60% of the scale span used):", "",
              fmt(comp), ""]
    lines += [f"## Spearman rank correlation of mean scores vs {REFERENCE}",
              "", fmt(rho), ""]
    lines += [f"## Disagreements vs {REFERENCE} (|mean diff| >= 2 levels)", ""]
    lines += [fmt(dis) if len(dis) else "None.", ""]
    lines += ["## Room-type disagreement rate (scoring judgment vs Stage 1 label)",
              "", fmt(room_dis), ""]
    lines += ["## Cost", "", fmt(costs, 4), "",
              "Full-run figures assume one call per scoreable image, scoreable share",
              "estimated from Stage 1 label frequencies over the full image corpus.", ""]

    picked = None
    if ratings is not None:
        refs, group_members = build_human_references(ratings, groups)
        ref_order = list(refs)  # groups first (all, custom...), then individual raters
        mvr = model_vs_references(means, refs)
        model_order = [m for m in MODELS if m in means["model"].unique()]
        mat, per_rater, ceiling = inter_rater(ratings)

        # Figures for the human-agreement + recommendation sections.
        ha_path = plot_human_agreement(mvr, ceiling, args.plots_dir)
        heat_path = plot_agreement_heatmap(mvr, model_order, ref_order, args.plots_dir)
        cq_path = plot_cost_quality(costs, mvr, args.plots_dir)
        ir_path = plot_interrater(mat, args.plots_dir)
        top_human = (mvr[mvr["reference"] == "all"]
                     .sort_values("spearman", ascending=False)["model"].iloc[0])
        focus = [top_human] + ([REFERENCE] if REFERENCE != top_human else [])
        scatter_path = plot_model_human_scatter(means, ratings, focus, args.plots_dir)

        # Describe each reference and how many images it covers.
        cover = ratings.groupby("rater")["image_path"].nunique()
        ref_desc = []
        for label in ref_order:
            if label in group_members:
                members = group_members[label]
                n_img = int(refs[label].notna().sum())
                ref_desc.append({"reference": label, "kind": "group",
                                 "raters": ", ".join(members), "n_images": n_img})
            else:
                ref_desc.append({"reference": label, "kind": "individual",
                                 "raters": label, "n_images": int(cover.get(label, 0))})
        ref_desc = pd.DataFrame(ref_desc)

        lines += ["## Human agreement (Stage 2b ratings)", "",
                  f"{ratings['rater'].nunique()} raters "
                  f"({', '.join(sorted(ratings['rater'].unique()))}), "
                  f"{ratings['image_path'].nunique()} images, {len(ratings)} ratings.", "",
                  "Each model is compared against several human references: group",
                  "averages (a reference's per-image value is the mean over its raters",
                  "who scored that image) and each rater individually. `n_images` is how",
                  "many images that reference covers — group and per-rater n differ",
                  "because raters completed different numbers of images.", "",
                  fmt(ref_desc), "",
                  "### Model agreement with each human reference — Spearman rho", "",
                  "Higher is better. Columns are the human references; rows are models.", "",
                  f"![model agreement with human consensus]({rel(ha_path)})", "",
                  f"![full agreement heatmap]({rel(heat_path)})", "",
                  fmt(ref_matrix(mvr, "spearman", model_order, ref_order)), "",
                  "### Model vs human, per image", "",
                  "Each point is one image: human mean score (x) vs model mean score (y).",
                  "The dashed line is perfect agreement; points above it mean the model",
                  "scored higher than the humans.", "",
                  f"![model vs human scatter]({rel(scatter_path)})", "",
                  "### Model agreement — mean absolute deviation (score points)", "",
                  "Lower is better (average per-image gap between model mean and the",
                  "reference).", "",
                  fmt(ref_matrix(mvr, "mean_abs_dev", model_order, ref_order)), "",
                  "### Model agreement — signed bias (model minus human)", "",
                  "Positive = model scores higher than the human reference on average.", "",
                  fmt(ref_matrix(mvr, "bias", model_order, ref_order)), "",
                  "### Human vs human — inter-rater correlation matrix (pairwise Spearman)",
                  "", f"![inter-rater heatmap]({rel(ir_path)})", "",
                  mat.round(3).to_markdown(), "",
                  f"**Human-agreement ceiling (mean pairwise rho): {ceiling:.3f}** — no",
                  "model should be expected to agree with humans more than humans agree",
                  "with each other.", "",
                  "### Per-rater agreement with the other raters", "", fmt(per_rater), ""]
        outliers = per_rater[per_rater["outlier"]]["rater"].tolist()
        if outliers:
            lines += [f"**Outlier rater(s) flagged: {', '.join(outliers)}** — consider",
                      "re-checking or excluding their ratings and re-running this report.", ""]
        # selection: score models against the 'all' consensus reference, then keep
        # those within 0.05 of the ceiling, preferring most consistent then cheapest.
        sel_ref = "all" if "all" in refs else ref_order[0]
        mvh = (mvr[mvr["reference"] == sel_ref]
               .rename(columns={"spearman": "spearman_vs_human_mean"})
               .sort_values("spearman_vs_human_mean", ascending=False))
        near = mvh[mvh["spearman_vs_human_mean"]
                   >= min(ceiling, mvh["spearman_vs_human_mean"].max()) - 0.05]
        sel = (near.merge(rank.rename("consistency_ratio"), left_on="model", right_index=True)
                   .merge(costs[["model", "mean_cost_per_call_usd"]], on="model")
                   .sort_values(["consistency_ratio", "mean_cost_per_call_usd"]))
        picked = sel.iloc[0]["model"] if len(sel) else None
        lines += ["## Recommendation", "",
                  f"Selection criterion: scoring models against the `{sel_ref}` human",
                  "reference, keep those within 0.05 of the human-agreement ceiling (or",
                  "of the best model, whichever is lower), then prefer the most consistent",
                  "(lowest within/between ratio), then the cheapest.", "",
                  "The chart plots each model's full-run cost against its agreement",
                  "with the human consensus — the best trade-offs sit toward the",
                  "upper-left (cheaper and more accurate).", "",
                  f"![cost vs quality]({rel(cq_path)})", "",
                  "Candidates near the ceiling:", "",
                  fmt(sel[["model", "spearman_vs_human_mean", "consistency_ratio",
                           "mean_cost_per_call_usd"]], 4), ""]
        if picked:
            lines += [f"**Pick: {picked}**", ""]
    else:
        lines += ["## Recommendation", "",
                  "(No human ratings provided — falling back to consistency-only criterion;",
                  "run Stage 2b + export_ratings.py and re-run with --ratings-parquet.)", "",
                  f"Most consistent model (lowest mean within/between ratio): **{best}**.",
                  "", "Mean consistency ratio by model:", "",
                  rank.round(3).reset_index().rename(
                      columns={"ratio_within_over_between": "mean ratio"}).to_markdown(index=False),
                  "",
                  "Review the Spearman and cost tables above to judge whether the cheap",
                  "tiers are distinguishable from the expensive ones: if a cheap model's",
                  "ratio and correlation are within noise of the best model's, it wins on",
                  "cost-per-call. Inspect the listed disagreement images against the anchor",
                  "grids before finalizing.", ""]

    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report written: {args.out_report}")
    print(f"Histograms: {hist_path}")


if __name__ == "__main__":
    main()
