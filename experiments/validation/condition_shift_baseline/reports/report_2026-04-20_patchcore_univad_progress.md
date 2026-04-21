# Progress Report / 2026-04-20 / PatchCore + UniVAD

## Scope

- dataset: `MVTec LOCO`
- primary category checked so far: `breakfast_box`
- primary question: `condition/domain shift가 기존 baseline 성능을 실제로 얼마나 흔드는가?`

## What Was Set Up

- query corruption evaluation was moved from file-backed augmentation to `manifest + on-the-fly` generation.
- identity manifest reproduction was verified before running corruption comparisons.
- `PatchCore` was used as the first external baseline to validate the protocol end-to-end.
- `UniVAD` environment, checkpoints, smoke dataset, and grounding masks were prepared for the next baseline.

## Manifest Reproduction Check

- image identity reproduction:
  - source: [identity_reproduction_check.json](/Users/song-inseop/연구/ReGraM/experiments/validation/condition_shift_baseline/reports/identity_reproduction_check.json)
  - result: `575 / 575` compared, `mismatch_count = 0`
- PatchCore score identity reproduction:
  - source: [breakfast_box.json](/Users/song-inseop/연구/ReGraM/experiments/validation/condition_shift_baseline/reports/patchcore_identity_repro/breakfast_box.json)
  - `mean_abs_diff = 0.0`
  - `max_abs_diff = 0.0`
  - `allclose(atol=1e-6) = true`

Interpretation:

- `manifest + on-the-fly loader` does not alter clean inputs.
- PatchCore score reproducibility is exact on `breakfast_box/test/good`.

## PatchCore Clean Baseline

- source: [patchcore_clean_eval_breakfast_box.json](/Users/song-inseop/연구/ReGraM/experiments/validation/condition_shift_baseline/reports/patchcore_clean_eval_breakfast_box.json)
- category: `breakfast_box`
- train count: `351`
- test count: `275`
- image-level AUROC: `75.11`
- full pixel AUROC: `79.55`
- anomaly-only pixel AUROC: `76.76`
- sampler percentage: `0.001`

Note:

- this is a single-category result, not the 5-category MVTec LOCO mean reported in prior papers.

## PatchCore Condition Shift Results

- source: [breakfast_box_multi.json](/Users/song-inseop/연구/ReGraM/experiments/validation/condition_shift_baseline/reports/patchcore_manifest_shift/breakfast_box_multi.json)
- clean image AUROC: `77.34`
- clean good mean score: `6.138`
- clean max threshold: `7.245`

### Main Findings

| Shift | Low AUROC | Medium AUROC | High AUROC | High FPR |
| --- | ---: | ---: | ---: | ---: |
| `brightness` | 80.43 | 66.85 | 42.41 | 2.94% |
| `low_light` | 79.85 | 56.98 | 37.45 | 5.88% |
| `gaussian_blur` | 75.74 | 56.70 | 39.58 | 5.88% |
| `motion_blur` | 78.03 | 74.83 | 66.76 | 0.98% |
| `compression` | 78.39 | 78.87 | 57.97 | 0.98% |
| `low_resolution` | 77.76 | 77.92 | 77.39 | 0.98% |
| `gaussian_noise` | 69.16 | 34.66 | 23.13 | 84.31% |
| `position_shift` | 49.03 | 30.64 | 9.94 | 92.16% |

### Interpretation

- `low_resolution` was almost harmless in this setting.
- `motion_blur` reduced ranking quality more than false positives.
- `brightness`, `low_light`, and `gaussian_blur` caused clear AUROC drops as severity increased.
- `gaussian_noise` and `position_shift` were the most destructive in the current protocol.
- `position_shift` is currently the strongest failure mode for PatchCore on `breakfast_box`.

## Qualitative Corruption Panels

- overview: [all_augmentations_breakfast_box_000.png](/Users/song-inseop/연구/ReGraM/experiments/validation/condition_shift_baseline/reports/sample_panels/all_augmentations_breakfast_box_000.png)
- per-shift panels:
  - [brightness](/Users/song-inseop/연구/ReGraM/experiments/validation/condition_shift_baseline/reports/sample_panels/brightness_breakfast_box_000.png)
  - [gaussian_blur](/Users/song-inseop/연구/ReGraM/experiments/validation/condition_shift_baseline/reports/sample_panels/gaussian_blur_breakfast_box_000.png)
  - [motion_blur](/Users/song-inseop/연구/ReGraM/experiments/validation/condition_shift_baseline/reports/sample_panels/motion_blur_breakfast_box_000.png)
  - [gaussian_noise](/Users/song-inseop/연구/ReGraM/experiments/validation/condition_shift_baseline/reports/sample_panels/gaussian_noise_breakfast_box_000.png)
  - [compression](/Users/song-inseop/연구/ReGraM/experiments/validation/condition_shift_baseline/reports/sample_panels/compression_breakfast_box_000.png)
  - [low_resolution](/Users/song-inseop/연구/ReGraM/experiments/validation/condition_shift_baseline/reports/sample_panels/low_resolution_breakfast_box_000.png)
  - [position_shift](/Users/song-inseop/연구/ReGraM/experiments/validation/condition_shift_baseline/reports/sample_panels/position_shift_breakfast_box_000.png)
  - [low_light](/Users/song-inseop/연구/ReGraM/experiments/validation/condition_shift_baseline/reports/sample_panels/low_light_breakfast_box_000.png)

## UniVAD Progress

### Completed

- official repo cloned under [external/UniVAD](/Users/song-inseop/연구/ReGraM/external/UniVAD)
- MVTec LOCO formatted root prepared under [data/mvtec_loco_caption](/Users/song-inseop/연구/ReGraM/data/mvtec_loco_caption)
- smoke subset prepared under [data/mvtec_loco_caption_smoke](/Users/song-inseop/연구/ReGraM/data/mvtec_loco_caption_smoke)
- checkpoints downloaded:
  - `sam_hq_vit_h.pth`
  - `groundingdino_swint_ogc.pth`
  - `ViT-L-14-336px.pt`
  - `dino_deitsmall8_300ep_pretrain.pth`
  - `dinov2_vitg14_pretrain.pth`
  - local `bert-base-uncased` files
- smoke grounding masks generated for:
  - [train/good/000](/Users/song-inseop/연구/ReGraM/external/UniVAD/masks/mvtec_loco_caption_smoke/breakfast_box/train/good/000/grounding_mask.png)
  - [test/good/000](/Users/song-inseop/연구/ReGraM/external/UniVAD/masks/mvtec_loco_caption_smoke/breakfast_box/test/good/000/grounding_mask.png)
  - [test/logical_anomalies/000](/Users/song-inseop/연구/ReGraM/external/UniVAD/masks/mvtec_loco_caption_smoke/breakfast_box/test/logical_anomalies/000/grounding_mask.png)

### Local Patches Applied

- [external/UniVAD/test_univad.py](/Users/song-inseop/연구/ReGraM/external/UniVAD/test_univad.py)
  - `--max_samples` support
  - CPU-safe dataloader config
  - `data_path`-based train sample resolution
- [external/UniVAD/UniVAD.py](/Users/song-inseop/연구/ReGraM/external/UniVAD/UniVAD.py)
  - CPU-safe device propagation
  - local CLIP cache path
- [external/UniVAD/modules.py](/Users/song-inseop/연구/ReGraM/external/UniVAD/modules.py)
  - CPU-safe DINO featurizer placement
- [external/UniVAD/models/component_segmentaion.py](/Users/song-inseop/연구/ReGraM/external/UniVAD/models/component_segmentaion.py)
  - `.cuda()` removal for CPU path
  - `MeanShift(n_jobs=1)`
- [external/UniVAD/utils/crf.py](/Users/song-inseop/연구/ReGraM/external/UniVAD/utils/crf.py)
  - `pydensecrf` fallback
- [external/UniVAD/models/GroundingDINO/groundingdino/util/get_tokenlizer.py](/Users/song-inseop/연구/ReGraM/external/UniVAD/models/GroundingDINO/groundingdino/util/get_tokenlizer.py)
  - local BERT path resolution

### Current Status

- smoke run log: [log.txt](/Users/song-inseop/연구/ReGraM/experiments/validation/condition_shift_baseline/reports/univad_smoke/mvtec_loco/log.txt)
- current status: `environment + grounding + dataset path issues resolved`
- remaining issue: `UniVAD setup/forward smoke path still does not complete on CPU`
- latest observed failure mode:
  - full `test_univad.py` loop was too noisy for debugging
  - a direct `setup + forward` smoke runner is now the shortest remaining path
  - OpenCV write warnings appear during component histogram save, but they are not yet confirmed as the terminal error

## Current Takeaway

- the evaluation protocol is stable enough for baseline comparison.
- PatchCore already shows clear condition sensitivity on `breakfast_box`.
- the next milestone is not a new experiment design step but a `UniVAD smoke inference pass` that returns scores for `good` and `logical` on the smoke subset.

## Next Actions

1. finish `UniVAD` direct smoke runner until `pred_score` is returned for `good` and `logical`.
2. once smoke passes, run clean `breakfast_box` evaluation for UniVAD.
3. then reuse the same manifest corruption protocol used for PatchCore.
