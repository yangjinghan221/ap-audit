# ap-audit

Protocol-ambiguity audit tools for oriented ship detection under missing modalities.

Companion code for the manuscript:

> **Evaluation Protocol, Not Detector Capability: Quantifying Ambiguities in
> Oriented Ship Detection Under Missing Modalities**
> submitted to *IEEE Geoscience and Remote Sensing Letters*, 2026.

## What this does

Given **one frozen set of exported predictions**, these tools recompute average
precision under different protocol choices, so that only the *reading* changes
and never the prediction. That is what makes the protocol effect separable from
detector capability.

| protocol choice | what is varied | effect reported in the letter |
|---|---|---|
| interpolation | 11-point (VOC2007) vs all-point (VOC2010+) | up to 7.6 points, and the sign reverses with recall |
| denominator | full ground truth vs active subset | 50.8 points |
| annotation source | SAR vs RGB annotations | factor of 3.45 |
| aggregation | per-image vs per-availability-group | 7.2 points |

## Requirements

Python 3.7 or newer. **No third-party packages** — the standard library is enough.

```bash
python mos_metrics.py --selftest     # 36 checks, no data required
```

## Files

| file | role |
|---|---|
| `eval_ap.py` | recompute AP from a prediction directory; switches between all-point and 11-point interpolation |
| `scan_ap_interp.py` | sweep recall by global score truncation and print the interpolation bias curve (Table I of the letter) |
| `mos_metrics.py` | rotated-box geometry, IoU, mAOE, directory-level evaluation |
| `mos_core.py` | path resolution and annotation parsing |

## Input format

DOTA-style, one `.txt` per image sharing the image stem, one box per line:

```
x1 y1 x2 y2 x3 y3 x4 y4 [score]
```

Ground-truth files may omit the score. Predictions must carry it; boxes are
ranked by score inside each image.

## Usage

```bash
# AP at a single IoU, all-point (default)
python eval_ap.py <pred_dir> <gt_dir> [score_thr]

# The same prediction under both rules, swept over recall
python scan_ap_interp.py <pred_dir> <gt_dir> [IoU]

# Geometry and angle self-test, no data needed
python mos_metrics.py --selftest
```

`scan_ap_interp.py` prints one row per truncation level:

```
 保留    召回  all-point  11-point   11-all
    1  0.0015     0.0015    0.0909  +0.0894
   10  0.0154     0.0154    0.0909  +0.0755
  ...
  648  0.8981     0.8883    0.8128  -0.0756
 1178  0.9522     0.9317    0.8932  -0.0385
```

The first row is the floor: with a single correct detection the 11-point value
cannot fall below 1/11 = 0.0909. The last rows show the reverse — a deficit at
high recall. Both are reproduced by the same prediction file.

## A note on the two rules

The 11-point rule samples recall at `{0, 0.1, ..., 1.0}` and averages the
maximum precision to the right of each sample point. It is the default of the
`DOTAMetric` implementation that most oriented-box benchmarks use, and it is not
exposed as a configuration option there. It does not inflate AP uniformly; it
quantizes AP onto a 1/11 grid, so the sign of the error depends on where the
recall happens to sit.

## Source language

Comments in the source are in Chinese. The interfaces, argument names and
printed tables are English; no Chinese is required to run the tools.

## License

MIT. See `LICENSE`.
