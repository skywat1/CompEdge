# Luxury-scoring model comparison

## Exact model IDs used

| model             | response model id(s)      |
|:------------------|:--------------------------|
| claude-haiku-4-5  | claude-haiku-4-5-20251001 |
| claude-sonnet-4-6 | claude-sonnet-4-6         |
| gemini-2.5-flash  | gemini-2.5-flash          |
| gemini-3.5-flash  | gemini-3.5-flash          |
| gpt-4o            | gpt-4o-2024-08-06         |

## Consistency (within-image std vs between-image std)

Ratio < 1 means replicate noise is smaller than real between-image
signal; lower is better.

![consistency ratio](outputs/plots/consistency_ratio.png)

| model             | room_type   |   n_images |   within_image_std |   between_image_std |   ratio_within_over_between |
|:------------------|:------------|-----------:|-------------------:|--------------------:|----------------------------:|
| claude-haiku-4-5  | bathroom    |         38 |              0.06  |               0.775 |                       0.078 |
| claude-haiku-4-5  | bedroom     |         38 |              0.107 |               1.368 |                       0.078 |
| claude-haiku-4-5  | kitchen     |         38 |              0.09  |               1.119 |                       0.08  |
| claude-haiku-4-5  | living_room |         38 |              0.126 |               0.938 |                       0.135 |
| claude-sonnet-4-6 | bathroom    |         38 |              0.108 |               1.234 |                       0.087 |
| claude-sonnet-4-6 | bedroom     |         38 |              0.105 |               1.367 |                       0.077 |
| claude-sonnet-4-6 | kitchen     |         38 |              0.092 |               1.465 |                       0.063 |
| claude-sonnet-4-6 | living_room |         38 |              0.103 |               1.333 |                       0.077 |
| gemini-2.5-flash  | bathroom    |         38 |              0.096 |               1.034 |                       0.093 |
| gemini-2.5-flash  | bedroom     |         38 |              0.166 |               1.291 |                       0.129 |
| gemini-2.5-flash  | kitchen     |         38 |              0.101 |               1.495 |                       0.067 |
| gemini-2.5-flash  | living_room |         38 |              0.113 |               1.209 |                       0.093 |
| gemini-3.5-flash  | bathroom    |         38 |              0.043 |               1.076 |                       0.04  |
| gemini-3.5-flash  | bedroom     |         38 |              0.058 |               1.133 |                       0.052 |
| gemini-3.5-flash  | kitchen     |         38 |              0.044 |               1.318 |                       0.033 |
| gemini-3.5-flash  | living_room |         38 |              0.049 |               1.286 |                       0.038 |
| gpt-4o            | bathroom    |         38 |              0.102 |               1.086 |                       0.094 |
| gpt-4o            | bedroom     |         38 |              0.079 |               1.416 |                       0.056 |
| gpt-4o            | kitchen     |         38 |              0.12  |               1.425 |                       0.084 |
| gpt-4o            | living_room |         38 |              0.129 |               1.179 |                       0.11  |

## Score distributions

![score histograms](outputs/plots/score_histograms.png)

Scale-compression flags (share of scores within 1 point of either
end of the scale < 5%, or < 60% of the scale span used):

| model             | room_type   |   share_in_extremes |   range_used |   scale_span | compressed   |
|:------------------|:------------|--------------------:|-------------:|-------------:|:-------------|
| claude-haiku-4-5  | bathroom    |               0     |          4   |            7 | True         |
| claude-haiku-4-5  | bedroom     |               0.105 |          4.5 |            7 | False        |
| claude-haiku-4-5  | kitchen     |               0.068 |          3.7 |            6 | False        |
| claude-haiku-4-5  | living_room |               0.026 |          4.5 |            7 | True         |
| claude-sonnet-4-6 | bathroom    |               0     |          4.3 |            7 | True         |
| claude-sonnet-4-6 | bedroom     |               0.016 |          4.8 |            7 | True         |
| claude-sonnet-4-6 | kitchen     |               0.258 |          4.6 |            6 | False        |
| claude-sonnet-4-6 | living_room |               0.026 |          5.5 |            7 | True         |
| gemini-2.5-flash  | bathroom    |               0     |          3.8 |            7 | True         |
| gemini-2.5-flash  | bedroom     |               0     |          4.7 |            7 | True         |
| gemini-2.5-flash  | kitchen     |               0.311 |          4.8 |            6 | False        |
| gemini-2.5-flash  | living_room |               0.026 |          4.9 |            7 | True         |
| gemini-3.5-flash  | bathroom    |               0     |          4   |            7 | True         |
| gemini-3.5-flash  | bedroom     |               0     |          4.5 |            7 | True         |
| gemini-3.5-flash  | kitchen     |               0.2   |          4.5 |            6 | False        |
| gemini-3.5-flash  | living_room |               0.026 |          5.1 |            7 | True         |
| gpt-4o            | bathroom    |               0     |          3.5 |            7 | True         |
| gpt-4o            | bedroom     |               0.137 |          5   |            7 | False        |
| gpt-4o            | kitchen     |               0.411 |          4.8 |            6 | False        |
| gpt-4o            | living_room |               0.026 |          5.3 |            7 | True         |

## Spearman rank correlation of mean scores vs gpt-4o

| model             |   overall |   kitchen |   bathroom |   bedroom |   living_room |
|:------------------|----------:|----------:|-----------:|----------:|--------------:|
| claude-haiku-4-5  |     0.818 |     0.838 |      0.804 |     0.903 |         0.823 |
| claude-sonnet-4-6 |     0.921 |     0.956 |      0.94  |     0.932 |         0.879 |
| gemini-2.5-flash  |     0.926 |     0.946 |      0.929 |     0.884 |         0.861 |
| gemini-3.5-flash  |     0.938 |     0.968 |      0.964 |     0.922 |         0.91  |

## Disagreements vs gpt-4o (|mean diff| >= 2 levels)

None.

## Room-type disagreement rate (scoring judgment vs Stage 1 label)

| model             | room_type   |   disagreement_rate |
|:------------------|:------------|--------------------:|
| claude-haiku-4-5  | bathroom    |                   0 |
| claude-haiku-4-5  | bedroom     |                   0 |
| claude-haiku-4-5  | kitchen     |                   0 |
| claude-haiku-4-5  | living_room |                   0 |
| claude-sonnet-4-6 | bathroom    |                   0 |
| claude-sonnet-4-6 | bedroom     |                   0 |
| claude-sonnet-4-6 | kitchen     |                   0 |
| claude-sonnet-4-6 | living_room |                   0 |
| gemini-2.5-flash  | bathroom    |                   0 |
| gemini-2.5-flash  | bedroom     |                   0 |
| gemini-2.5-flash  | kitchen     |                   0 |
| gemini-2.5-flash  | living_room |                   0 |
| gemini-3.5-flash  | bathroom    |                   0 |
| gemini-3.5-flash  | bedroom     |                   0 |
| gemini-3.5-flash  | kitchen     |                   0 |
| gemini-3.5-flash  | living_room |                   0 |
| gpt-4o            | bathroom    |                   0 |
| gpt-4o            | bedroom     |                   0 |
| gpt-4o            | kitchen     |                   0 |
| gpt-4o            | living_room |                   0 |

## Cost

| model             |   mean_cost_per_call_usd |   mean_latency_s |   est_scoreable_images |   full_run_standard_usd |   full_run_batch_usd |
|:------------------|-------------------------:|-----------------:|-----------------------:|------------------------:|---------------------:|
| claude-haiku-4-5  |                   0.0046 |           4.506  |                  41779 |                 193.891 |              96.9453 |
| claude-sonnet-4-6 |                   0.0058 |           8.9176 |                  41779 |                 243.095 |             121.548  |
| gemini-2.5-flash  |                   0.003  |           8.204  |                  41779 |                 125.385 |              62.6925 |
| gemini-3.5-flash  |                   0.0105 |           7.319  |                  41779 |                 439.455 |             219.727  |
| gpt-4o            |                   0.0049 |           5.8076 |                  41779 |                 202.851 |             101.425  |

Full-run figures assume one call per scoreable image, scoreable share
estimated from Stage 1 label frequencies over the full image corpus.

## Human agreement (Stage 2b ratings)

3 raters (harvey, robin, seb), 152 images, 456 ratings.

Each model is compared against several human references: group
averages (a reference's per-image value is the mean over its raters
who scored that image) and each rater individually. `n_images` is how
many images that reference covers — group and per-rater n differ
because raters completed different numbers of images.

| reference    | kind       | raters             |   n_images |
|:-------------|:-----------|:-------------------|-----------:|
| all          | group      | harvey, robin, seb |        152 |
| harvey_robin | group      | harvey, robin      |        152 |
| harvey       | individual | harvey             |        152 |
| robin        | individual | robin              |        152 |
| seb          | individual | seb                |        152 |

### gemini-3.5-flash prompt tuning vs human raters — score distributions

Same binning as the score-distribution histograms above. Top rows are
each gemini-3.5-flash prompt revision (all replicates); bottom rows are
the human references (group averages, then each rater individually).
Shows whether a tuning pass actually moved the model's distribution
toward where humans land, room type by room type.

![gemini tuning vs human histograms](outputs/plots/tuning_vs_human_histograms.png)

### Model agreement with each human reference — Spearman rho

Higher is better. Columns are the human references; rows are models.

![model agreement with human consensus](outputs/plots/human_agreement.png)

![full agreement heatmap](outputs/plots/agreement_heatmap.png)

| model             |   all |   harvey_robin |   harvey |   robin |   seb |
|:------------------|------:|---------------:|---------:|--------:|------:|
| gpt-4o            | 0.742 |          0.723 |    0.736 |   0.611 | 0.62  |
| gemini-3.5-flash  | 0.795 |          0.776 |    0.763 |   0.673 | 0.66  |
| gemini-2.5-flash  | 0.725 |          0.703 |    0.708 |   0.596 | 0.616 |
| claude-haiku-4-5  | 0.638 |          0.624 |    0.606 |   0.547 | 0.531 |
| claude-sonnet-4-6 | 0.758 |          0.739 |    0.731 |   0.639 | 0.637 |

### Model vs human, per image

Each point is one image: human mean score (x) vs model mean score (y).
The dashed line is perfect agreement; points above it mean the model
scored higher than the humans.

![model vs human scatter](outputs/plots/model_vs_human_scatter.png)

### Model agreement — mean absolute deviation (score points)

Lower is better (average per-image gap between model mean and the
reference).

| model             |   all |   harvey_robin |   harvey |   robin |   seb |
|:------------------|------:|---------------:|---------:|--------:|------:|
| gpt-4o            | 0.896 |          0.851 |    0.881 |   1.067 | 1.223 |
| gemini-3.5-flash  | 0.672 |          0.693 |    0.752 |   0.911 | 1.003 |
| gemini-2.5-flash  | 0.792 |          0.843 |    0.847 |   1.029 | 1.052 |
| claude-haiku-4-5  | 0.813 |          0.827 |    0.877 |   1.053 | 1.082 |
| claude-sonnet-4-6 | 0.786 |          0.783 |    0.827 |   0.976 | 1.092 |

### Model agreement — signed bias (model minus human)

Positive = model scores higher than the human reference on average.

| model             |   all |   harvey_robin |   harvey |   robin |   seb |
|:------------------|------:|---------------:|---------:|--------:|------:|
| gpt-4o            | 0.684 |          0.469 |    0.371 |   0.568 | 1.114 |
| gemini-3.5-flash  | 0.446 |          0.232 |    0.133 |   0.33  | 0.876 |
| gemini-2.5-flash  | 0.42  |          0.205 |    0.106 |   0.304 | 0.85  |
| claude-haiku-4-5  | 0.452 |          0.237 |    0.138 |   0.336 | 0.882 |
| claude-sonnet-4-6 | 0.484 |          0.269 |    0.17  |   0.368 | 0.914 |

### Human vs human — inter-rater correlation matrix (pairwise Spearman)

![inter-rater heatmap](outputs/plots/interrater_heatmap.png)

|        |   harvey |   robin |   seb |
|:-------|---------:|--------:|------:|
| harvey |    1     |   0.68  | 0.672 |
| robin  |    0.68  |   1     | 0.589 |
| seb    |    0.672 |   0.589 | 1     |

**Human-agreement ceiling (mean pairwise rho): 0.647** — no
model should be expected to agree with humans more than humans agree
with each other.

### Per-rater agreement with the other raters

| rater   |   n_rated |   mean_rho_vs_others | outlier   |
|:--------|----------:|---------------------:|:----------|
| harvey  |       152 |                0.676 | False     |
| robin   |       152 |                0.635 | False     |
| seb     |       152 |                0.63  | False     |

## Recommendation

Selection criterion: scoring models against the `all` human
reference, keep those within 0.05 of the human-agreement ceiling (or
of the best model, whichever is lower), then prefer the most consistent
(lowest within/between ratio), then the cheapest.

The chart plots each model's full-run cost against its agreement
with the human consensus — the best trade-offs sit toward the
upper-left (cheaper and more accurate).

![cost vs quality](outputs/plots/cost_vs_quality.png)

Candidates near the ceiling:

| model             |   spearman_vs_human_mean |   consistency_ratio |   mean_cost_per_call_usd |
|:------------------|-------------------------:|--------------------:|-------------------------:|
| gemini-3.5-flash  |                   0.795  |              0.0407 |                   0.0105 |
| claude-sonnet-4-6 |                   0.7585 |              0.0761 |                   0.0058 |
| gpt-4o            |                   0.7424 |              0.0859 |                   0.0049 |
| claude-haiku-4-5  |                   0.638  |              0.0928 |                   0.0046 |
| gemini-2.5-flash  |                   0.7254 |              0.0955 |                   0.003  |

**Pick: gemini-3.5-flash**
