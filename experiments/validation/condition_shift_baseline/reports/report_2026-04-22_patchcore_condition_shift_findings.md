# Report / 2026-04-22 / PatchCore Condition Shift Findings

## Scope

- baseline: `PatchCore`
- dataset: `MVTec LOCO`
- category: `breakfast_box`
- result source:
  - [breakfast_box_multi.json](/Users/song-inseop/연구/ReGraM/experiments/validation/condition_shift_baseline/reports/patchcore_manifest_shift/breakfast_box_multi.json)
  - per-shift summaries under [patchcore_manifest_shift](/Users/song-inseop/연구/ReGraM/experiments/validation/condition_shift_baseline/reports/patchcore_manifest_shift)
- note:
  - this report is based on the currently saved `multi` summary, which was produced before the later `shift x severity` run split.
  - therefore the interpretation here is valid for the saved numeric results, but the newer wandb logging layout is not required to read this report.

## Question

- Does `PatchCore` mistake `condition-shifted normal` samples as anomaly on `breakfast_box`?
- Which corruption families are mild, and which ones are destructive enough to break the clean-normal threshold?

## Clean Reference

- clean image AUROC: `77.34`
- clean good mean score: `6.138`
- clean good max threshold: `7.245`
- clean anomaly mean score: `6.879`

Interpretation:

- the clean baseline is usable but not especially strong.
- the clean-normal threshold is relatively tight, so strong condition shift can cross it even when the sample is still semantically normal.

## Metric Guide

- `fpr_over_clean_max`
  - fraction of shifted normal images whose score exceeds the `clean-good max` threshold.
  - this is the main robustness metric.
- `mean_score_shift`
  - shifted normal mean score minus clean normal mean score.
  - this shows how far the score distribution moves even before the threshold is crossed.
- `image_auroc_vs_clean_anomaly`
  - AUROC between `shifted normal` and `real anomaly`.
  - if this drops, the shifted normal samples are becoming hard to distinguish from true anomaly.

## Main Findings

### 1. PatchCore is clearly condition-sensitive on `breakfast_box`

- severe `position_shift` and `gaussian_noise` break the baseline decisively.
- moderate and severe `brightness`, `low_light`, and `gaussian_blur` also degrade the detector in a consistent direction.
- `low_resolution` is nearly harmless in the current setting.
- `motion_blur` and `compression` are milder than expected in terms of false positives.

### 2. Worst failure modes

| Shift | Severity | FPR over clean max | Mean score shift | Shifted vs anomaly AUROC |
| --- | --- | ---: | ---: | ---: |
| `position_shift` | `high` | `92.16%` | `+2.732` | `9.94` |
| `gaussian_noise` | `high` | `84.31%` | `+1.342` | `23.13` |
| `position_shift` | `medium` | `58.82%` | `+1.412` | `30.64` |
| `position_shift` | `low` | `21.57%` | `+0.603` | `49.03` |
| `gaussian_blur` | `high` | `5.88%` | `+0.651` | `39.58` |

Interpretation:

- `position_shift` is the strongest failure mode by a large margin.
- `gaussian_noise` is the second major failure mode, especially at `high`.
- both shifts do not merely perturb ranking slightly; they push normal samples deeply into anomaly-like score ranges.

### 3. Severity trend is meaningful

| Shift | Low AUROC | Medium AUROC | High AUROC | High FPR |
| --- | ---: | ---: | ---: | ---: |
| `brightness` | `80.43` | `66.85` | `42.41` | `2.94%` |
| `low_light` | `79.85` | `56.98` | `37.45` | `5.88%` |
| `gaussian_blur` | `75.74` | `56.70` | `39.58` | `5.88%` |
| `motion_blur` | `78.03` | `74.83` | `66.76` | `0.98%` |
| `compression` | `78.39` | `78.87` | `57.97` | `0.98%` |
| `low_resolution` | `77.76` | `77.92` | `77.39` | `0.98%` |
| `gaussian_noise` | `69.16` | `34.66` | `23.13` | `84.31%` |
| `position_shift` | `49.03` | `30.64` | `9.94` | `92.16%` |

Interpretation:

- for most shift families, severity increases in the expected direction.
- `low_resolution` is the main exception: the detector is almost invariant to this corruption family in the current protocol.
- `motion_blur` and `compression` hurt ranking quality somewhat, but do not create large false-positive explosions.

## Per-Shift Notes

### `position_shift`

- the detector is highly sensitive even at `low`.
- this suggests that PatchCore depends strongly on spatial alignment for `breakfast_box`.
- this is a strong signal that relation-preserving but location-shifted normal inputs are still misread as anomaly.

### `gaussian_noise`

- failure is abrupt rather than gradual once severity becomes strong enough.
- `medium` already pushes the AUROC down to `34.66`, and `high` causes a near-collapse.
- noise robustness is therefore a major weakness in the current baseline.

### `brightness`, `low_light`, `gaussian_blur`

- these are not catastrophic at `low`, but become clearly harmful by `medium` and `high`.
- they are useful as mid-strength robustness probes because they reveal score drift before full collapse.

### `motion_blur`, `compression`

- these degrade quality, but FPR stays near the clean threshold level.
- the detector is not robust in a strong sense, but these are not the first-priority failure modes.

### `low_resolution`

- effectively benign in this setup.
- this corruption family is not a useful stressor for PatchCore on `breakfast_box`.

## Decision

- [확정] `PatchCore` already shows meaningful condition sensitivity on `breakfast_box`.
- [확정] the strongest current stressors are `position_shift` and `gaussian_noise`.
- [확정] `brightness`, `low_light`, and `gaussian_blur` are useful secondary robustness probes.
- [확정] `low_resolution` should not be prioritized as a core stress condition in the next comparison round.

## Limitations

- only `breakfast_box` is covered here.
- the clean baseline AUROC itself is moderate, so robustness conclusions should be read as `baseline-specific`, not universal.
- the saved result used here is the older `multi` summary rather than the newer severity-separated run layout.

## Recommended Next Step

1. Re-run the same corruption families with the current `shift x severity` split logging so wandb comparisons use the flattened `shifted_*` metric schema.
2. Use `position_shift`, `gaussian_noise`, `low_light`, and `gaussian_blur` as the primary comparison set for the next baseline.
3. Reproduce the same protocol on `UniVAD` once smoke inference is stable.
