# Source Layout

이 디렉터리는 `condition_shift_baseline` 실험에서 실제로 실행되는 Python 소스만 모아둔 곳이다.

## Quick Guide

- `core/`
  - PatchCore 기준 manifest 생성, shift 평가, summary 계약, 보조 검증 스크립트
- `univad/`
  - UniVAD 기준 dataset 준비, smoke test, clean eval

## Primary Entry Points

- [core/build_query_manifest.py](/Users/song-inseop/연구/ReGraM/experiments/validation/condition_shift_baseline/src/core/build_query_manifest.py)
  - 입력: `data/query_normal_clean`
  - 출력: `manifests/query_*.jsonl`
  - 역할: on-the-fly augmentation manifest 생성기

- [core/run_patchcore_manifest_shift.py](/Users/song-inseop/연구/ReGraM/experiments/validation/condition_shift_baseline/src/core/run_patchcore_manifest_shift.py)
  - 입력: `manifest jsonl`, `category`, raw LOCO dataset root
  - 출력: `reports/patchcore_manifest_shift/*.json`, `log.txt`
  - 역할: PatchCore shift evaluation runner

- [univad/run_smoke_colab.py](/Users/song-inseop/연구/ReGraM/experiments/validation/condition_shift_baseline/src/univad/run_smoke_colab.py)
  - 입력: prepared smoke dataset
  - 출력: smoke summary JSON, log
  - 역할: UniVAD smoke runner

- [univad/run_clean_eval.py](/Users/song-inseop/연구/ReGraM/experiments/validation/condition_shift_baseline/src/univad/run_clean_eval.py)
  - 입력: prepared UniVAD dataset
  - 출력: clean eval summary JSON, log
  - 역할: UniVAD clean evaluation runner

## Rule of Thumb

- 실험 입력 정의를 바꾸고 싶다
  - `build_query_manifest.py`
- PatchCore를 실제 실행하고 싶다
  - `run_patchcore_manifest_shift.py`
- notebook에서 어떤 파일이 실행되는지 알고 싶다
  - notebook의 `Runner Config` 셀과 이 문서를 같이 본다
